"""
Device mesh — transport layer.

Two transports, same interface:

  InProcessTransport   — both peers live in the same process (tests, demos).
                         No network; push/pull entries via shared queues.

  HttpTransport        — real devices over TCP.  Uses stdlib http.server +
                         http.client only (no third-party deps).  Wire format:
                         JSON with Content-Type application/json.

API seen by MeshNode (the caller):
  transport.push(entries, peer_url)  → int (accepted count)
  transport.pull(known_ids, peer_url) → list[SyncEntry]

HttpTransport also exposes a server side:
  server = HttpTransport.make_server(store, host, port)
  server.start() / server.stop()

The server answers:
  POST /sync/push   body: {"entries": [...]}         → {"accepted": N}
  POST /sync/pull   body: {"known_ids": [...]}        → {"entries": [...]}
  GET  /sync/health                                   → {"ok": true, "node_id": "..."}
"""

from __future__ import annotations

import http.client
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Protocol
from urllib.parse import urlparse

from synapseos.mesh.sync_store import SyncEntry, SyncStore


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
# HTTP server-side handler
# ---------------------------------------------------------------------------

class _SyncHandler(BaseHTTPRequestHandler):
    """Attached to a ThreadingHTTPServer; store injected via server.store."""

    def log_message(self, fmt, *args):
        pass  # quiet

    def _read_json(self) -> dict | None:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except Exception:
            return None

    def _send_json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/sync/health":
            self._send_json(200, {"ok": True,
                                  "node_id": self.server.store.node_id})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        store: SyncStore = self.server.store
        body = self._read_json()
        if body is None:
            self._send_json(400, {"error": "bad json"})
            return

        if self.path == "/sync/push":
            raw = body.get("entries", [])
            entries = [SyncEntry.from_dict(e) for e in raw]
            accepted = store.merge(entries)
            store.save()
            self._send_json(200, {"accepted": accepted})

        elif self.path == "/sync/pull":
            known = set(body.get("known_ids", []))
            new_entries = store.entries_since(known)
            self._send_json(200, {"entries": [e.to_dict() for e in new_entries]})

        else:
            self._send_json(404, {"error": "not found"})


class _SyncServer(ThreadingHTTPServer):
    def __init__(self, store: SyncStore, host: str, port: int) -> None:
        super().__init__((host, port), _SyncHandler)
        self.store = store
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.shutdown()


# ---------------------------------------------------------------------------
# HTTP client transport
# ---------------------------------------------------------------------------

class HttpTransport:
    """Synchronous HTTP client; one call at a time, no connection pool."""

    timeout: float = 10.0

    def push(self, entries: list[SyncEntry], peer_url: str) -> int:
        body = json.dumps({"entries": [e.to_dict() for e in entries]}).encode()
        try:
            resp = self._post(peer_url, "/sync/push", body)
            return resp.get("accepted", 0)
        except Exception:
            return 0

    def pull(self, known_ids: set[str], peer_url: str) -> list[SyncEntry]:
        body = json.dumps({"known_ids": list(known_ids)}).encode()
        try:
            resp = self._post(peer_url, "/sync/pull", body)
            return [SyncEntry.from_dict(e) for e in resp.get("entries", [])]
        except Exception:
            return []

    def health(self, peer_url: str) -> bool:
        try:
            parsed = urlparse(peer_url)
            conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80,
                                              timeout=self.timeout)
            conn.request("GET", "/sync/health")
            r = conn.getresponse()
            return r.status == 200
        except Exception:
            return False

    def _post(self, base_url: str, path: str, body: bytes) -> dict:
        parsed = urlparse(base_url)
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80,
                                          timeout=self.timeout)
        conn.request("POST", path, body=body,
                     headers={"Content-Type": "application/json",
                               "Content-Length": str(len(body))})
        r = conn.getresponse()
        raw = r.read()
        return json.loads(raw)

    @staticmethod
    def make_server(store: SyncStore, host: str = "0.0.0.0",
                    port: int = 7700) -> _SyncServer:
        return _SyncServer(store, host, port)
