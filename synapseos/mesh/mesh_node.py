"""
Device mesh — MeshNode: one device's view of the mesh.

A MeshNode owns a SyncStore and a list of peer URLs it wants to stay in sync
with.  It can be told to sync right now (`sync_once()`), or it can run a
background thread that syncs every N seconds (`start_background()`) — just like
the idle loop / web learning loop pattern.

Offline queue: if a peer is unreachable, unapplied outbound entries are queued.
On the next successful sync with that peer, the queue is flushed first.

The node also integrates with the CognitiveController's data stores so that:
  - semantic facts added during a session propagate to other devices
  - wonder questions sync so the inner life is the same on every device
  - episodic highlights sync so memories are shared

Controller integration is opt-in: pass the relevant stores in wire_controller().
The mesh never reads LLM outputs or agent internals — only the data stores.

Sync protocol (push/pull, one round per peer):
  1. pull: ask peer for entries I haven't seen → merge locally
  2. push: send peer entries it hasn't seen (our oplog \ peer's known_ids)
     -- or, equivalently, push entries queued since last successful push

The two-step ensures both peers converge in one round even when both have new
writes.  Idempotency means retrying is always safe.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from synapseos.mesh.sync_store import SyncEntry, SyncStore
from synapseos.mesh.transport import InProcessTransport


class MeshNode:
    """
    One device's participant in the device mesh.

    Parameters
    ----------
    store       : SyncStore for this device
    peers       : list of peer URL strings (node_id for in-process, http://... for real)
    transport   : InProcessTransport | HttpTransport
    sync_every  : background sync interval in seconds (0 = no background sync)
    """

    def __init__(self, store: SyncStore,
                 peers: list[str] | None = None,
                 transport=None,
                 sync_every: float = 30.0):
        self.store = store
        self.peers: list[str] = list(peers or [])
        self.transport = transport or InProcessTransport()
        self.sync_every = sync_every
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # per-peer offline queue: peer_url → list of SyncEntry
        self._offline_queue: dict[str, list[SyncEntry]] = {}
        # per-peer: set of entry_ids we know the peer has confirmed
        self._peer_known: dict[str, set[str]] = {}

    # -- peer management ---------------------------------------------------

    def add_peer(self, peer_url: str) -> None:
        if peer_url not in self.peers:
            self.peers.append(peer_url)

    def remove_peer(self, peer_url: str) -> None:
        self.peers = [p for p in self.peers if p != peer_url]

    # -- sync --------------------------------------------------------------

    def sync_once(self) -> dict[str, Any]:
        """
        Sync with all known peers.  Returns a summary dict.
        One round = pull-from-peer then push-to-peer.
        """
        results: dict[str, Any] = {}
        for peer in self.peers:
            results[peer] = self._sync_peer(peer)
        return results

    def _sync_peer(self, peer: str) -> dict:
        pulled = pushed = 0
        errors = []

        # --- pull: ask peer what entries we're missing --------------------
        try:
            our_ids = self.store.known_ids()
            new_entries = self.transport.pull(our_ids, peer)
            if new_entries:
                pulled = self.store.merge(new_entries)
                self.store.save()
        except Exception as e:
            errors.append(f"pull: {e}")

        # --- push: send peer what it's missing ----------------------------
        try:
            peer_ids = self._peer_known.get(peer, set())

            # flush offline queue first
            queued = self._offline_queue.pop(peer, [])
            fresh_entries = self.store.entries_since(peer_ids)
            to_send = _dedup(queued + fresh_entries)

            if to_send:
                accepted = self.transport.push(to_send, peer)
                if accepted >= 0:
                    # mark these as known to peer
                    for e in to_send:
                        self._peer_known.setdefault(peer, set()).add(e.entry_id)
                    pushed = len(to_send)
                else:
                    # peer unreachable — queue for later
                    self._offline_queue.setdefault(peer, []).extend(to_send)
                    errors.append("push: peer unreachable, queued")
        except Exception as e:
            self._offline_queue.setdefault(peer, []).extend(
                self.store.entries_since(self._peer_known.get(peer, set())))
            errors.append(f"push: {e}")

        return {"pulled": pulled, "pushed": pushed,
                "queued": len(self._offline_queue.get(peer, [])),
                "errors": errors}

    # -- background sync ---------------------------------------------------

    def start_background(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name=f"mesh-{self.store.node_id}")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.wait(self.sync_every):
            try:
                self.sync_once()
            except Exception:
                pass

    # -- controller integration -------------------------------------------

    def wire_controller(self, controller) -> None:
        """
        Attach this node to a CognitiveController.
        Exports facts from semantic memory, wonder questions, and top episodic
        entries into the SyncStore so they propagate to peers on next sync.
        Call after controller is initialized; call again to re-export.
        """
        # semantic facts (sm.facts is a list of SemanticFact dataclasses)
        try:
            sm = controller.memory.semantic
            for fact in sm.facts:
                self.store.put_fact(fact.subject, fact.predicate, fact.obj,
                                    fact.confidence)
        except Exception:
            pass

        # wonder questions
        try:
            if controller.wonder is not None:
                for w in controller.wonder.open_questions(limit=20):
                    self.store.put_wonder(w.id, w.to_dict())
        except Exception:
            pass

        # episodic highlights (most recent 20)
        try:
            em = controller.memory.episodic
            entries = list(em._store.values())[-20:] if hasattr(em, "_store") else []
            for ep in entries:
                ep_id = ep.get("id") or ep.get("timestamp", "")
                summary = ep.get("summary") or ep.get("content", "")[:200]
                if ep_id and summary:
                    self.store.put_episode(str(ep_id), str(summary))
        except Exception:
            pass

    def apply_to_controller(self, controller) -> dict[str, int]:
        """
        Write synced entries back into the controller's data stores.
        Returns counts of what was applied.
        Applied = new facts/wonders not already in local stores.
        """
        facts_added = wonders_added = 0

        # facts → semantic memory (sm.facts is a list of SemanticFact dataclasses)
        try:
            sm = controller.memory.semantic
            existing = {(f.subject, f.predicate, f.obj) for f in sm.facts}
            for s, p, o, conf in self.store.get_facts():
                if (s, p, o) not in existing:
                    sm.add_fact(s, p, o, confidence=conf, source="mesh")
                    facts_added += 1
        except Exception:
            pass

        # wonder questions → WonderJournal
        try:
            if controller.wonder is not None:
                for key in self.store.keys("wq|"):
                    blob = self.store.get(key)
                    if blob is None:
                        continue
                    try:
                        from synapseos.soul.wonder import Wonder
                        w_remote = Wonder.from_dict(blob)
                        existing_ids = {w.id for w in controller.wonder._wonders.values()}
                        if w_remote.id not in existing_ids:
                            controller.wonder._wonders[w_remote.id] = w_remote
                            wonders_added += 1
                    except Exception:
                        pass
        except Exception:
            pass

        return {"facts_added": facts_added, "wonders_added": wonders_added}

    def stats(self) -> dict:
        queued_total = sum(len(v) for v in self._offline_queue.values())
        return {
            "node_id": self.store.node_id,
            "peers": len(self.peers),
            "background_running": self._thread is not None and self._thread.is_alive(),
            "offline_queued": queued_total,
            "store": self.store.stats(),
        }


# ---------------------------------------------------------------------------

def _dedup(entries: list[SyncEntry]) -> list[SyncEntry]:
    seen: set[str] = set()
    out: list[SyncEntry] = []
    for e in entries:
        if e.entry_id not in seen:
            seen.add(e.entry_id)
            out.append(e)
    return out
