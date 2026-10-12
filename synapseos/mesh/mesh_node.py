"""
Device mesh — MeshNode: one device's view of the mesh.

A MeshNode owns a SyncStore and a list of peer URLs it wants to stay in sync
with.  It can be told to sync right now (`sync_once()`), or it can run a
background thread that syncs every N seconds (`start_background()`) — just like
the idle loop / web learning loop pattern.

Offline queue: if a peer is unreachable (or rejects us), unsent outbound entries
are queued.  On the next successful sync with that peer, the queue is flushed
first.  Entries are only marked as "known to the peer" after the peer accepted
them, so a failed or unauthenticated push never loses data.

The node also integrates with the CognitiveController's data stores so that
the same companion lives on every device:

  keyspace          local store                    import path
  ----------------  -----------------------------  --------------------------
  fact|s|p|o        semantic memory                BeliefReviser (never a
                                                   direct write); deletions
                                                   travel as tombstones
  wq|<id>           WonderJournal (all statuses)   WonderJournal.merge_remote
  affect|state      EmotionTracker state + counts  EmotionTracker.merge_remote
  aesthetics|state  AestheticPreferences           merge_remote
  value|<name>      ValueSystem (modified only)    merge_remote
  reflection|<ts>   SelfReflection snapshots       merge_remote
  ep|<id>           episodic highlights            (export only)

Controller integration is opt-in: pass the controller to wire_controller()
(export) and apply_to_controller() (import).  The mesh never reads LLM outputs
or agent internals — only the data stores.

Fact deletions: SemanticMemory has no tombstones of its own, so the node keeps
device-local bookkeeping in the SyncStore ("facts_known": facts this device
exported or imported).  A known fact that has vanished from semantic memory was
deleted locally and is tombstoned in the mesh; a tombstone received from a peer
removes the matching local fact — unless the local fact was (re)learned after
the tombstone, in which case the newer fact wins and is exported again.

Sync protocol (push/pull, one round per peer):
  1. pull: ask peer for entries I haven't seen → merge locally
  2. push: send peer entries it hasn't confirmed (our oplog \\ peer's known ids)

The two-step ensures both peers converge in one round even when both have new
writes.  Idempotency means retrying is always safe.
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any

from synapseos.mesh.sync_store import SyncEntry, SyncStore
from synapseos.mesh.transport import InProcessTransport

PUSH_BATCH = 2000          # entries per push request (keeps bodies far below MAX_BODY)
_FACT_PREFIX = "fact|"


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
        self._sync_lock = threading.Lock()
        # per-peer offline queue: peer_url → entries not yet accepted by that peer
        self._offline_queue: dict[str, list[SyncEntry]] = {}
        # per-peer: set of entry_ids the peer has confirmed
        self._peer_known: dict[str, set[str]] = {}
        # reviser used when the controller has no web learning loop
        self._own_reviser = None
        # problems met during the most recent export/apply (surfaced in stats)
        self.last_errors: list[str] = []

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
        with self._sync_lock:
            for peer in list(self.peers):
                results[peer] = self._sync_peer(peer)
        return results

    def _sync_peer(self, peer: str) -> dict:
        pulled = pushed = 0
        errors: list[str] = []
        known = self._peer_known.setdefault(peer, set())

        # --- pull: ask peer what entries we're missing --------------------
        try:
            new_entries = self.transport.pull(self.store.known_ids(), peer)
            if new_entries:
                pulled = self.store.merge(new_entries)
                self.store.save()
                # the peer evidently has everything it just sent us
                known.update(e.entry_id for e in new_entries)
        except Exception as e:
            errors.append(f"pull: {type(e).__name__}: {e}")

        # --- push: send peer what it hasn't confirmed ---------------------
        # Queued entries are always a subset of the store's unconfirmed
        # entries, so the union is computed from the store alone; the queue is
        # rewritten (never appended to) so nothing is duplicated or lost.
        to_send = self.store.entries_since(known)
        try:
            for i in range(0, len(to_send), PUSH_BATCH):
                batch = to_send[i:i + PUSH_BATCH]
                accepted = self.transport.push(batch, peer)
                if accepted < 0:
                    raise ConnectionError("peer unreachable")
                known.update(e.entry_id for e in batch)
                pushed += len(batch)
        except Exception as e:
            errors.append(f"push: {type(e).__name__}: {e}"
                          + (" — queued" if to_send else ""))
        unsent = [e for e in to_send if e.entry_id not in known]
        if unsent:
            self._offline_queue[peer] = unsent
        else:
            self._offline_queue.pop(peer, None)

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
            except Exception as e:
                self.last_errors = [f"sync loop: {type(e).__name__}: {e}"]

    # -- controller integration: export -----------------------------------

    def wire_controller(self, controller) -> dict[str, int]:
        """
        Export the controller's inner state into the SyncStore so it propagates
        to peers on the next sync.  Idempotent: unchanged values write nothing.
        Call after the controller is initialised; call again to re-export.

        Soul state (wonders, affect, aesthetics, values, reflection) is first
        reconciled with what the store already holds, so the exported value is
        the join of both sides: an out-of-date device can never overwrite a
        newer remote value with its stale copy (which would ping-pong forever).
        Returns counts of entries written per namespace.
        """
        self.last_errors = []
        out = {"facts": 0, "fact_tombstones": 0, "wonders": 0, "affect": 0,
               "aesthetics": 0, "values": 0, "reflection": 0, "episodes": 0}
        self._merge_soul(controller, {})

        sm = getattr(getattr(controller, "memory", None), "semantic", None)
        if sm is not None:
            self._guard("export facts", lambda: self._export_facts(sm, out))
        wonder = getattr(controller, "wonder", None)
        if wonder is not None:
            def _wonders():
                out["wonders"] = self.store.put_many(
                    {f"wq|{w.id}": w.to_dict() for w in wonder.all_wonders()})
            self._guard("export wonders", _wonders)

        tracker = getattr(controller, "emotion_tracker", None)
        if tracker is not None and tracker.state.observations > 0:
            def _affect():
                out["affect"] = self.store.put_many({"affect|state": {
                    "state": tracker.state.to_dict(),
                    "trigger_counts": tracker._trigger_counts}})
            self._guard("export affect", _affect)

        aesthetics = getattr(controller, "aesthetics", None)
        if aesthetics is not None and aesthetics.exposures > 0:
            def _aesthetics():
                out["aesthetics"] = self.store.put_many({"aesthetics|state": {
                    "weights": aesthetics.weights,
                    "exposures": aesthetics.exposures,
                    "reactions": aesthetics.reactions[-500:]}})
            self._guard("export aesthetics", _aesthetics)

        values = getattr(controller, "values", None)
        if values is not None:
            from synapseos.soul.values import is_default_stance

            def _values():
                out["values"] = self.store.put_many(
                    {f"value|{v.name}": v.to_dict()
                     for v in values.values.values() if not is_default_stance(v)})
            self._guard("export values", _values)

        reflection = getattr(controller, "reflection", None)
        if reflection is not None:
            def _reflection():
                out["reflection"] = self.store.put_many(
                    {f"reflection|{s.timestamp!r}": s.to_dict()
                     for s in reflection.snapshots})
            self._guard("export reflection", _reflection)

        episodic = getattr(getattr(controller, "memory", None), "episodic", None)
        if episodic is not None:
            def _episodes():
                items = {}
                for ep in list(episodic.entries)[-20:]:
                    summary = f"{ep.query[:100]} → {ep.response[:100]}".strip()
                    if ep.id and summary:
                        items[f"ep|{ep.id}"] = summary
                out["episodes"] = self.store.put_many(items)
            self._guard("export episodes", _episodes)

        return out

    def _guard(self, what: str, fn) -> None:
        try:
            fn()
        except Exception as e:
            self.last_errors.append(f"{what}: {type(e).__name__}: {e}")

    # -- facts: export -----------------------------------------------------

    @staticmethod
    def _norm(subject: str, predicate: str, obj: str) -> str:
        """Identity of a fact — same equivalence BeliefReviser uses."""
        return f"{subject.lower()}|{predicate}|{obj.strip().lower()}"

    @staticmethod
    def _fact_epoch(fact) -> float:
        try:
            return datetime.fromisoformat(fact.timestamp).timestamp()
        except Exception:
            return 0.0

    @staticmethod
    def _tombstone_wins(fact, tomb_ts: float) -> bool:
        """A tombstone beats a local fact unless the fact was learned after it
        (KG-import facts are re-created on every device, so they never count)."""
        return (fact.source == "knowledge_graph"
                or MeshNode._fact_epoch(fact) <= tomb_ts)

    def _local_facts(self, sm) -> dict[str, Any]:
        """norm → representative fact (highest confidence); skips unsyncable ones."""
        local: dict[str, Any] = {}
        for f in sm.facts:
            if not f.subject or not f.predicate or not f.obj:
                continue
            if "|" in f.subject or "|" in f.predicate:     # would break the key format
                continue
            n = self._norm(f.subject, f.predicate, f.obj)
            if n not in local or f.confidence > local[n].confidence:
                local[n] = f
        return local

    def _store_facts(self) -> tuple[dict[str, list[str]], dict[str, float]]:
        """(norm → live store keys, norm → latest tombstone ts) for fact| keys."""
        live: dict[str, list[str]] = {}
        for k in self.store.keys(_FACT_PREFIX):
            n = self._norm_of_key(k)
            if n is not None:
                live.setdefault(n, []).append(k)
        tombs: dict[str, float] = {}
        for k in self.store.deleted_keys(_FACT_PREFIX):
            n = self._norm_of_key(k)
            w = self.store.winner(k)
            if n is not None and w is not None:
                tombs[n] = max(tombs.get(n, 0.0), w.ts)
        return live, tombs

    def _norm_of_key(self, key: str) -> str | None:
        parts = key.split("|", 3)
        if len(parts) != 4 or not (parts[1] and parts[2] and parts[3]):
            return None
        return self._norm(parts[1], parts[2], parts[3])

    def _export_facts(self, sm, out: dict[str, int]) -> None:
        local = self._local_facts(sm)
        live, tombs = self._store_facts()
        known = self.store.meta_all("facts_known")

        # Deletions: a fact this device exported/imported before that is no
        # longer in semantic memory was deleted here → tombstone every live key.
        gone = [n for n in known if n not in local]
        doomed = [k for n in gone for k in live.get(n, [])]
        out["fact_tombstones"] = self.store.delete_many(doomed)
        for n in gone:
            known.pop(n, None)

        items: dict[str, Any] = {}
        for n, f in local.items():
            keys = live.get(n)
            key = SyncStore.fact_key(f.subject, f.predicate, f.obj)
            if keys:
                # already in the mesh; refresh confidence only under our own casing
                if key in keys:
                    items[key] = round(f.confidence, 4)
                known[n] = 1
            elif n in tombs and self._tombstone_wins(f, tombs[n]):
                continue        # deleted elsewhere; apply_to_controller removes it
            else:
                items[key] = round(f.confidence, 4)
                known[n] = 1
        out["facts"] = self.store.put_many(items)
        self.store.meta_replace("facts_known", known)

    # -- controller integration: import -----------------------------------

    def apply_to_controller(self, controller) -> dict[str, Any]:
        """
        Fold synced state back into the controller's data stores.

        Facts go through the BeliefReviser (trust-weighted: they may be
        added, reinforced-by-presence, revised or rejected — never written
        blindly); tombstones remove local facts; wonders / affect / aesthetics /
        values / reflection merge with their stores' own deterministic rules.
        Returns counts of what changed (plus an "errors" list).
        """
        self.last_errors = []
        res: dict[str, Any] = {"facts_added": 0, "facts_rejected": 0,
                               "facts_removed": 0}

        sm = getattr(getattr(controller, "memory", None), "semantic", None)
        if sm is not None:
            self._guard("apply facts", lambda: self._apply_facts(controller, sm, res))
        self._merge_soul(controller, res)

        res["errors"] = list(self.last_errors)
        return res

    def _merge_soul(self, controller, res: dict[str, Any]) -> None:
        """Merge synced wonders / affect / aesthetics / values / reflection into
        the controller's stores (each store's merge_remote is a deterministic
        join, so repeated merging converges and never regresses local state)."""
        for k in ("wonders_added", "wonders_updated", "affect_updated",
                  "aesthetics_updated", "values_updated", "reflections_added"):
            res.setdefault(k, 0)

        wonder = getattr(controller, "wonder", None)
        if wonder is not None:
            self._guard("apply wonders", lambda: self._apply_wonders(wonder, res))

        tracker = getattr(controller, "emotion_tracker", None)
        if tracker is not None:
            def _affect():
                blob = self.store.get("affect|state")
                if isinstance(blob, dict) and isinstance(blob.get("state"), dict):
                    if tracker.merge_remote(blob["state"],
                                            blob.get("trigger_counts", {})):
                        res["affect_updated"] += 1
            self._guard("apply affect", _affect)

        aesthetics = getattr(controller, "aesthetics", None)
        if aesthetics is not None:
            def _aesthetics():
                blob = self.store.get("aesthetics|state")
                if isinstance(blob, dict) and aesthetics.merge_remote(blob):
                    res["aesthetics_updated"] += 1
            self._guard("apply aesthetics", _aesthetics)

        values = getattr(controller, "values", None)
        if values is not None:
            def _values():
                for key in self.store.keys("value|"):
                    blob = self.store.get(key)
                    if isinstance(blob, dict) and values.merge_remote(key[6:], blob):
                        res["values_updated"] += 1
            self._guard("apply values", _values)

        reflection = getattr(controller, "reflection", None)
        if reflection is not None:
            def _reflection():
                for key in self.store.keys("reflection|"):
                    blob = self.store.get(key)
                    if isinstance(blob, dict) and reflection.merge_remote(blob):
                        res["reflections_added"] += 1
            self._guard("apply reflection", _reflection)

    def _get_reviser(self, controller, sm):
        loop = getattr(controller, "web_loop", None)
        reviser = getattr(loop, "reviser", None)
        if reviser is not None:
            return reviser
        if self._own_reviser is None or self._own_reviser.memory is not sm:
            from synapseos.web.belief import BeliefReviser
            self._own_reviser = BeliefReviser(
                sm, audit_path=str(self.store.path.parent / "mesh_revisions.json"))
        return self._own_reviser

    def _apply_facts(self, controller, sm, res: dict[str, int]) -> None:
        local = self._local_facts(sm)
        live, tombs = self._store_facts()
        known = self.store.meta_all("facts_known")
        decided = self.store.meta_all("facts_decided")   # key → entry_id already judged

        # 1. tombstones from peers remove the matching local facts
        for n, ts in tombs.items():
            if n in live or n not in local:
                continue            # still alive under another casing / nothing to remove
            if not self._tombstone_wins(local[n], ts):
                continue            # relearned after the deletion; export will revive it
            local.pop(n)
            for f in [f for f in sm.facts
                      if self._norm(f.subject, f.predicate, f.obj) == n]:
                res["facts_removed"] += sm.remove_fact(f.subject, f.predicate, f.obj)
            known.pop(n, None)

        # 2. live facts from peers go through belief revision
        reviser = None
        from synapseos.web.extractor import ExtractedFact
        for n, keys in live.items():
            if n in local:
                known[n] = 1        # both sides have it
                continue
            if n in known:
                continue            # deleted here; the tombstone is pending export
            for key in sorted(keys):
                w = self.store.winner(key)
                if w is None or decided.get(key) == w.entry_id:
                    continue
                try:
                    conf = float(w.value)
                except (TypeError, ValueError):
                    continue
                if conf != conf:    # NaN
                    continue
                conf = min(1.0, max(0.0, conf))
                _, s, p, o = key.split("|", 3)
                if reviser is None:
                    reviser = self._get_reviser(controller, sm)
                rev = reviser.integrate(
                    ExtractedFact(subject=s, predicate=p, obj=o, confidence=conf,
                                  sentence="", pattern="mesh"),
                    source_url=f"mesh://{w.node_id}", source=f"mesh:{w.node_id}")
                decided[key] = w.entry_id
                if rev.action in ("added", "coexists", "revised"):
                    res["facts_added"] += 1
                    known[n] = 1
                elif rev.action == "reinforced":
                    known[n] = 1
                else:
                    res["facts_rejected"] += 1
                break
        if reviser is not None:
            reviser.save()
        self.store.meta_replace("facts_known", known)
        self.store.meta_replace("facts_decided", decided)

    def _apply_wonders(self, wonder, res: dict[str, int]) -> None:
        from synapseos.soul.wonder import Wonder
        changed = False
        for key in self.store.keys("wq|"):
            blob = self.store.get(key)
            if not isinstance(blob, dict):
                continue
            try:
                remote = Wonder.from_dict(blob)
            except Exception:
                continue
            outcome = wonder.merge_remote(remote)
            if outcome == "added":
                res["wonders_added"] += 1
                changed = True
            elif outcome == "updated":
                res["wonders_updated"] += 1
                changed = True
        if changed:
            wonder.save()

    def stats(self) -> dict:
        queued_total = sum(len(v) for v in self._offline_queue.values())
        return {
            "node_id": self.store.node_id,
            "peers": len(self.peers),
            "background_running": self._thread is not None and self._thread.is_alive(),
            "offline_queued": queued_total,
            "last_errors": list(self.last_errors),
            "store": self.store.stats(),
        }
