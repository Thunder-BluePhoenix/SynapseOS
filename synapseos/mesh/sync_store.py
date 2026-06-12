"""
Device mesh — SyncStore: the shared mutable state that propagates across peers.

Data model: every mutation is a *SyncEntry* (a delta) appended to an immutable
oplog. On merge, conflicting entries for the same key resolve by Last-Write-Wins
on `ts` (Unix float). Concurrent writes with equal timestamps keep the
lexicographically larger `value` to break ties deterministically.

The oplog is the source of truth; the "view" (current key→value dict) is derived
by replaying it. This makes sync trivial: send entries the remote hasn't seen
(by vector clock or simply by entry id), apply them locally, rebuild the view.

Stores tracked:
  - semantic facts (subject, predicate, object triples)  → key "s|p|o", value = confidence
  - episodic summaries                                    → key "ep|<id>", value = summary
  - wonder questions                                      → key "wq|<id>", value = json blob
  - arbitrary named key-values (catch-all)               → key as-is

Each SyncEntry carries:
  node_id    — which device wrote this (opaque string, usually hostname)
  ts         — Unix timestamp (float) — the LWW arbiter
  key        — dotted string namespace
  value      — JSON-serialisable scalar or dict (None = tombstone / deletion)
  entry_id   — sha1[:12] of (node_id, ts, key, repr(value)) — dedup

Merge guarantee: applying the same set of entries in any order to any peer
produces an identical view (commutative, associative, idempotent). The only
state needed to catch up is "which entry_ids do I NOT have yet?"
"""

from __future__ import annotations

import hashlib
import json
import socket
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


def _eid(node_id: str, ts: float, key: str, value: Any) -> str:
    blob = f"{node_id}|{ts}|{key}|{repr(value)}"
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


@dataclass
class SyncEntry:
    entry_id: str
    node_id: str
    ts: float
    key: str
    value: Any          # None → tombstone

    @classmethod
    def make(cls, node_id: str, key: str, value: Any,
             ts: float | None = None) -> "SyncEntry":
        ts = ts if ts is not None else time.time()
        eid = _eid(node_id, ts, key, value)
        return cls(entry_id=eid, node_id=node_id, ts=ts, key=key, value=value)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SyncEntry":
        return cls(**d)


class SyncStore:
    """
    Persistent LWW key-value store backed by an append-only oplog.
    Thread-safe. Can be merged with any remote SyncStore instance.
    """

    def __init__(self, path: str = "data/mesh/sync.json",
                 node_id: str | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.node_id = node_id or socket.gethostname()
        self._lock = threading.Lock()
        self._oplog: list[SyncEntry] = []          # append-only, ordered by arrival
        self._seen: set[str] = set()               # entry_ids already applied
        self._view: dict[str, Any] = {}            # current materialized key→value
        self._load()

    # -- persistence -------------------------------------------------------

    def _load(self) -> None:
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text())
                for d in raw.get("oplog", []):
                    e = SyncEntry.from_dict(d)
                    if e.entry_id not in self._seen:
                        self._oplog.append(e)
                        self._seen.add(e.entry_id)
                self._rebuild_view()
            except Exception:
                pass

    def save(self) -> None:
        with self._lock:
            self.path.write_text(json.dumps(
                {"node_id": self.node_id,
                 "oplog": [e.to_dict() for e in self._oplog]},
                indent=2))

    def _rebuild_view(self) -> None:
        """Replay oplog → deterministic LWW view."""
        tmp: dict[str, SyncEntry] = {}
        for e in self._oplog:
            if e.key not in tmp:
                tmp[e.key] = e
            else:
                prev = tmp[e.key]
                if e.ts > prev.ts or (e.ts == prev.ts and e.value > prev.value):
                    tmp[e.key] = e
        self._view = {k: v.value for k, v in tmp.items() if v.value is not None}

    # -- write / read ------------------------------------------------------

    def put(self, key: str, value: Any, ts: float | None = None) -> SyncEntry:
        """Write a key-value pair. Returns the SyncEntry created."""
        e = SyncEntry.make(self.node_id, key, value, ts)
        with self._lock:
            if e.entry_id not in self._seen:
                self._oplog.append(e)
                self._seen.add(e.entry_id)
                self._rebuild_view()
        return e

    def delete(self, key: str) -> SyncEntry:
        """Tombstone a key."""
        return self.put(key, None)

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._view.get(key, default)

    def view(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._view)

    def keys(self, prefix: str = "") -> list[str]:
        with self._lock:
            return [k for k in self._view if k.startswith(prefix)]

    # -- oplog access for sync ---------------------------------------------

    def entries_since(self, known_ids: set[str]) -> list[SyncEntry]:
        """Return entries the remote doesn't have yet."""
        with self._lock:
            return [e for e in self._oplog if e.entry_id not in known_ids]

    def known_ids(self) -> set[str]:
        with self._lock:
            return set(self._seen)

    # -- merge from remote -------------------------------------------------

    def merge(self, remote_entries: list[SyncEntry]) -> int:
        """
        Apply a list of remote SyncEntries.  Returns the count of new entries
        actually applied (already-seen entries are silently skipped — idempotent).
        """
        applied = 0
        with self._lock:
            for e in remote_entries:
                if e.entry_id not in self._seen:
                    self._oplog.append(e)
                    self._seen.add(e.entry_id)
                    applied += 1
            if applied:
                self._rebuild_view()
        return applied

    # -- helpers for typed namespaces -------------------------------------

    def put_fact(self, subject: str, predicate: str, obj: str,
                 confidence: float) -> SyncEntry:
        key = f"fact|{subject}|{predicate}|{obj}"
        return self.put(key, round(confidence, 4))

    def put_wonder(self, wonder_id: str, blob: dict) -> SyncEntry:
        return self.put(f"wq|{wonder_id}", blob)

    def put_episode(self, ep_id: str, summary: str) -> SyncEntry:
        return self.put(f"ep|{ep_id}", summary)

    def put_kv(self, name: str, value: Any) -> SyncEntry:
        return self.put(f"kv|{name}", value)

    def get_facts(self) -> list[tuple[str, str, str, float]]:
        """All non-tombstoned fact entries as (s, p, o, confidence) tuples."""
        out = []
        with self._lock:
            for k, v in self._view.items():
                if k.startswith("fact|"):
                    parts = k.split("|", 4)
                    if len(parts) == 4:
                        _, s, p, o = parts
                        out.append((s, p, o, float(v)))
        return out

    def stats(self) -> dict:
        with self._lock:
            ns: dict[str, int] = {}
            for k in self._view:
                prefix = k.split("|")[0]
                ns[prefix] = ns.get(prefix, 0) + 1
            return {
                "node_id": self.node_id,
                "oplog_entries": len(self._oplog),
                "live_keys": len(self._view),
                "namespaces": ns,
            }
