"""
Device mesh — transport layer.

Two transports, same interface:

  InProcessTransport   — both peers live in the same process (tests, demos).
                         No network; push/pull entries via shared registry.

  HttpTransport        — real devices over TCP.  stdlib http.server +
                         http.client only.  Wire format: JSON.

API seen by MeshNode (the caller):
  transport.push(entries, peer_url)   → int (accepted count); raises on failure
  transport.pull(known_ids, peer_url) → list[SyncEntry];      raises on failure

HttpTransport security model (every endpoint, both directions):
  * Shared key.  All devices hold the same secret.  Every request carries
        X-Mesh-Node       sender node id
        X-Mesh-Timestamp  unix seconds
        X-Mesh-Nonce      random, single-use
        X-Mesh-Signature  HMAC-SHA256(key, method|path|node|ts|nonce|sha256(body))
    The server rejects: unknown node id (peer allowlist), bad signature,
    clock skew > MAX_SKEW seconds, replayed nonce, oversized body.
  * Responses are signed the same way (bound to the request nonce), so a
    man-in-the-middle cannot inject entries into what a client pulls.
  * HMAC gives authenticity + integrity, NOT confidentiality.  Traffic that
    leaves loopback must be TLS: the server refuses a non-loopback bind
    without an ssl_context, and the client refuses plain http:// to a
    non-loopback host.

Server:
  server = HttpTransport.make_server(store, host, port, shared_key=..., allowed_peers={...})
  server.start() / server.stop()

Endpoints (all authenticated):
  POST /sync/push   body: {"entries": [...]}   → {"accepted": N}
  POST /sync/pull   body: {"known_ids": [...]} → {"entries": [...]}
  GET  /sync/health                            → {"ok": true, "node_id": "..."}
"""

from __future__ import annotations

import hashlib
import hmac
import http.client
import ipaddress
import json
import secrets
import ssl
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Protocol
from urllib.parse import urlparse

from synapseos.mesh.sync_store import SyncEntry, SyncStore

MAX_SKEW = 60.0                  # seconds of tolerated clock difference
MAX_BODY = 32 * 1024 * 1024      # bytes accepted per request / response
_NONCE_BYTES = 16
_MIN_KEY_BYTES = 16


# ---------------------------------------------------------------------------
# Transport protocol
# ---------------------------------------------------------------------------

class Transport(Protocol):
    def push(self, entries: list[SyncEntry], peer_url: str) -> int: ...
    def pull(self, known_ids: set[str], peer_url: str) -> list[SyncEntry]: ...


# ---------------------------------------------------------------------------
# In-process transport (tests / single-process multi-node demos)
# ---------------------------------------------------------------------------

class InProcessTransport:
    """
    Registry of named stores.  peer_url is the store's node_id.
    """

    def __init__(self) -> None:
        self._stores: dict[str, SyncStore] = {}

    def register(self, store: SyncStore) -> None:
        self._stores[store.node_id] = store

    def push(self, entries: list[SyncEntry], peer_url: str) -> int:
        store = self._stores.get(peer_url)
        if store is None:
            return -1   # unreachable → caller queues for later
        return store.merge(entries)

    def pull(self, known_ids: set[str], peer_url: str) -> list[SyncEntry]:
        store = self._stores.get(peer_url)
        if store is None:
            return []
        return store.entries_since(known_ids)


# ---------------------------------------------------------------------------
# Signing primitives
# ---------------------------------------------------------------------------

class AuthError(Exception):
    """Request or response failed authentication."""


def _normalize_key(shared_key: str | bytes) -> bytes:
    key = shared_key.encode() if isinstance(shared_key, str) else bytes(shared_key)
    if len(key) < _MIN_KEY_BYTES:
        raise ValueError(f"shared_key must be at least {_MIN_KEY_BYTES} bytes")
    return key


def _sign(key: bytes, kind: str, method: str, path: str, node: str,
          ts: str, nonce: str, body: bytes) -> str:
    # kind separates request and response signatures so one can't be replayed as the other.
    msg = "\n".join((kind, method, path, node, ts, nonce,
                     hashlib.sha256(body).hexdigest())).encode()
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def _is_loopback(host: str | None) -> bool:
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class _NonceCache:
    """Remembers nonces for 2×MAX_SKEW so a captured request can't be replayed."""

    def __init__(self) -> None:
        self._seen: dict[str, float] = {}
        self._lock = threading.Lock()

    def check_and_add(self, node: str, nonce: str, now: float) -> bool:
        key = f"{node}:{nonce}"
        with self._lock:
            horizon = now - 2 * MAX_SKEW
            if len(self._seen) > 1024:
                self._seen = {k: t for k, t in self._seen.items() if t >= horizon}
            if key in self._seen:
                return False
            self._seen[key] = now
            return True


# ---------------------------------------------------------------------------
# HTTP server-side handler
# ---------------------------------------------------------------------------

class _SyncHandler(BaseHTTPRequestHandler):
    """Attached to a _SyncServer; store/key/allowlist injected via self.server."""

    def log_message(self, fmt, *args):
        pass  # quiet

    def _send(self, code: int, obj: dict, nonce: str | None = None) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if nonce is not None:
            node = self.server.store.node_id
            ts = str(time.time())
            self.send_header("X-Mesh-Node", node)
            self.send_header("X-Mesh-Timestamp", ts)
            self.send_header("X-Mesh-Signature",
                             _sign(self.server.key, "response", self.command,
                                   self.path, node, ts, nonce, body))
        self.end_headers()
        self.wfile.write(body)

    def _authenticate(self) -> tuple[bytes, str] | None:
        """Return (body, nonce) if the request is authentic, else send 4xx and None."""
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            self._send(400, {"error": "bad length"})
            return None
        if length < 0 or length > MAX_BODY:
            self._send(413, {"error": "body too large"})
            return None
        body = self.rfile.read(length) if length else b""

        node = self.headers.get("X-Mesh-Node", "")
        ts = self.headers.get("X-Mesh-Timestamp", "")
        nonce = self.headers.get("X-Mesh-Nonce", "")
        sig = self.headers.get("X-Mesh-Signature", "")

        # One opaque 401 for every auth failure — don't tell a probe which check failed.
        expected = _sign(self.server.key, "request", self.command, self.path,
                         node, ts, nonce, body)
        ok = hmac.compare_digest(expected, sig)
        ok = ok and node in self.server.allowed_peers
        try:
            now = time.time()
            ok = ok and abs(now - float(ts)) <= MAX_SKEW
        except ValueError:
            ok = False
        # Only burn a nonce slot for fully valid requests.
        ok = ok and bool(nonce) and self.server.nonces.check_and_add(node, nonce, now)
        if not ok:
            self._send(401, {"error": "unauthorized"})
            return None
        return body, nonce

    def do_GET(self):
        auth = self._authenticate()
        if auth is None:
            return
        _, nonce = auth
        if self.path == "/sync/health":
            self._send(200, {"ok": True, "node_id": self.server.store.node_id}, nonce)
        else:
            self._send(404, {"error": "not found"}, nonce)

    def do_POST(self):
        auth = self._authenticate()
        if auth is None:
            return
        raw_body, nonce = auth
        store: SyncStore = self.server.store
        try:
            body = json.loads(raw_body) if raw_body else {}
            if not isinstance(body, dict):
                raise ValueError("body must be an object")
        except Exception:
            self._send(400, {"error": "bad json"}, nonce)
            return

        if self.path == "/sync/push":
            try:
                entries = [SyncEntry.from_dict(e) for e in body.get("entries", [])]
            except Exception:
                self._send(400, {"error": "bad entries"}, nonce)
                return
            accepted = store.merge(entries)
            store.save()
            self._send(200, {"accepted": accepted}, nonce)

        elif self.path == "/sync/pull":
            known = set(body.get("known_ids", []))
            new_entries = store.entries_since(known)
            self._send(200, {"entries": [e.to_dict() for e in new_entries]}, nonce)

        else:
            self._send(404, {"error": "not found"}, nonce)


class _SyncServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, store: SyncStore, host: str, port: int, key: bytes,
                 allowed_peers: frozenset[str],
                 ssl_context: ssl.SSLContext | None) -> None:
        if not _is_loopback(host) and ssl_context is None:
            raise ValueError(
                f"refusing to listen on {host!r} without TLS: pass ssl_context "
                "(HMAC authenticates but does not encrypt)")
        super().__init__((host, port), _SyncHandler)
        if ssl_context is not None:
            self.socket = ssl_context.wrap_socket(self.socket, server_side=True)
        self.store = store
        self.key = key
        self.allowed_peers = allowed_peers
        self.nonces = _NonceCache()
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self.server_address[1]

    def start(self) -> None:
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.shutdown()
        self.server_close()


# ---------------------------------------------------------------------------
# HTTP client transport
# ---------------------------------------------------------------------------

class HttpTransport:
    """
    Synchronous authenticated HTTP client; one call at a time, no connection pool.

    node_id        this device's id (must be in the peer's allowlist)
    shared_key     secret shared by all mesh devices (>= 16 bytes)
    allowed_peers  node ids whose responses we accept; a response signed by
                   any other node is rejected
    ssl_context    for https:// peers (e.g. pinned CA / self-signed cert)
    """

    timeout: float = 10.0

    def __init__(self, node_id: str, shared_key: str | bytes,
                 allowed_peers: set[str] | frozenset[str] | list[str],
                 ssl_context: ssl.SSLContext | None = None) -> None:
        if not node_id:
            raise ValueError("node_id required")
        if not allowed_peers:
            raise ValueError("allowed_peers must not be empty")
        self.node_id = node_id
        self._key = _normalize_key(shared_key)
        self.allowed_peers = frozenset(allowed_peers)
        self.ssl_context = ssl_context

    def push(self, entries: list[SyncEntry], peer_url: str) -> int:
        body = json.dumps({"entries": [e.to_dict() for e in entries]}).encode()
        resp = self._request("POST", peer_url, "/sync/push", body)
        return int(resp.get("accepted", 0))

    def pull(self, known_ids: set[str], peer_url: str) -> list[SyncEntry]:
        body = json.dumps({"known_ids": sorted(known_ids)}).encode()
        resp = self._request("POST", peer_url, "/sync/pull", body)
        return [SyncEntry.from_dict(e) for e in resp.get("entries", [])]

    def health(self, peer_url: str) -> bool:
        try:
            return bool(self._request("GET", peer_url, "/sync/health", b"").get("ok"))
        except Exception:
            return False

    def _request(self, method: str, base_url: str, path: str, body: bytes) -> dict:
        parsed = urlparse(base_url)
        host = parsed.hostname
        if parsed.scheme not in ("http", "https") or not host:
            raise ValueError(f"bad peer url: {base_url!r}")
        if parsed.scheme == "http" and not _is_loopback(host):
            raise ValueError(f"refusing plain http to non-loopback peer {host!r}; use https")

        ts = str(time.time())
        nonce = secrets.token_hex(_NONCE_BYTES)
        sig = _sign(self._key, "request", method, path, self.node_id, ts, nonce, body)
        headers = {"Content-Type": "application/json",
                   "Content-Length": str(len(body)),
                   "X-Mesh-Node": self.node_id,
                   "X-Mesh-Timestamp": ts,
                   "X-Mesh-Nonce": nonce,
                   "X-Mesh-Signature": sig}

        if parsed.scheme == "https":
            conn = http.client.HTTPSConnection(host, parsed.port or 443,
                                               timeout=self.timeout,
                                               context=self.ssl_context)
        else:
            conn = http.client.HTTPConnection(host, parsed.port or 80,
                                              timeout=self.timeout)
        try:
            conn.request(method, path, body=body, headers=headers)
            r = conn.getresponse()
            raw = r.read(MAX_BODY + 1)
            if len(raw) > MAX_BODY:
                raise AuthError("response too large")
            if r.status == 401:
                raise AuthError(f"peer {base_url} rejected our credentials")
            self._verify_response(r, method, path, nonce, raw)
            if r.status != 200:
                raise RuntimeError(f"peer {base_url} returned {r.status}")
            return json.loads(raw)
        finally:
            conn.close()

    def _verify_response(self, r, method: str, path: str, nonce: str, raw: bytes) -> None:
        node = r.getheader("X-Mesh-Node", "")
        ts = r.getheader("X-Mesh-Timestamp", "")
        sig = r.getheader("X-Mesh-Signature", "")
        expected = _sign(self._key, "response", method, path, node, ts, nonce, raw)
        try:
            fresh = abs(time.time() - float(ts)) <= MAX_SKEW
        except ValueError:
            fresh = False
        if not (hmac.compare_digest(expected, sig) and node in self.allowed_peers and fresh):
            raise AuthError("response failed authentication")

    @staticmethod
    def make_server(store: SyncStore, host: str = "127.0.0.1", port: int = 7700,
                    *, shared_key: str | bytes,
                    allowed_peers: set[str] | frozenset[str] | list[str],
                    ssl_context: ssl.SSLContext | None = None) -> _SyncServer:
        if not allowed_peers:
            raise ValueError("allowed_peers must not be empty")
        return _SyncServer(store, host, port, _normalize_key(shared_key),
                           frozenset(allowed_peers), ssl_context)
