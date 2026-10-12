"""
Tests for SynapseOS Device Mesh — offline-first peer memory sync.

Covers: LWW determinism, merge idempotency, tombstones, offline queue,
in-process two-peer sync, three-peer convergence, HTTP server/client,
controller integration (facts + wonder export/apply), background sync thread.

Run:
    python tests/test_mesh.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from synapseos.mesh import SyncStore, SyncEntry, MeshNode, InProcessTransport, HttpTransport
from synapseos.mesh.transport import HttpTransport, AuthError, _sign


results = []


def test(name: str):
    def decorator(fn):
        def wrapper():
            t0 = time.time()
            try:
                fn()
                r = {"name": name, "passed": True, "ms": round((time.time()-t0)*1000, 1), "detail": ""}
            except AssertionError as e:
                r = {"name": name, "passed": False, "ms": round((time.time()-t0)*1000, 1), "detail": str(e)}
            except Exception as e:
                r = {"name": name, "passed": False, "ms": round((time.time()-t0)*1000, 1), "detail": f"{type(e).__name__}: {e}"}
            results.append(r)
        wrapper.__name__ = fn.__name__
        return wrapper
    return decorator


def _store(node_id: str = "test") -> SyncStore:
    d = tempfile.mkdtemp(prefix="mesh_test_")
    return SyncStore(path=os.path.join(d, "sync.json"), node_id=node_id)


# ---------------------------------------------------------------------------
# SyncStore — LWW semantics
# ---------------------------------------------------------------------------

@test("put and get round-trips")
def test_put_get():
    s = _store()
    s.put("greeting", "hello")
    assert s.get("greeting") == "hello"


@test("later timestamp wins (LWW)")
def test_lww_wins():
    s = _store()
    t_old = time.time() - 10
    t_new = time.time()
    s.put("key", "old", ts=t_old)
    s.put("key", "new", ts=t_new)
    assert s.get("key") == "new"
    # order of application doesn't matter
    s2 = _store()
    s2.put("key", "new", ts=t_new)
    s2.put("key", "old", ts=t_old)
    assert s2.get("key") == "new"


@test("equal-timestamp tie broken by lexicographic value")
def test_tie_break():
    s = _store()
    ts = time.time()
    s.put("k", "alpha", ts=ts)
    s.put("k", "zebra", ts=ts)
    assert s.get("k") == "zebra"


@test("tombstone deletes key from view")
def test_tombstone():
    s = _store()
    s.put("k", "alive")
    s.delete("k")
    assert s.get("k") is None
    assert "k" not in s.view()


@test("merge is idempotent — applying same entries twice is safe")
def test_idempotent():
    a = _store("a")
    a.put("x", 1)
    entries = list(a._oplog)
    a.merge(entries)   # re-apply same entries
    assert a.stats()["oplog_entries"] == 1   # no duplicates


@test("entries_since returns only new entries")
def test_entries_since():
    s = _store()
    e1 = s.put("k1", "v1")
    e2 = s.put("k2", "v2")
    newer = s.entries_since({e1.entry_id})
    assert len(newer) == 1 and newer[0].entry_id == e2.entry_id


@test("oplog persists and reloads correctly")
def test_persist():
    d = tempfile.mkdtemp(prefix="mesh_test_")
    path = os.path.join(d, "sync.json")
    s = SyncStore(path=path, node_id="n1")
    s.put("colour", "blue")
    s.save()
    s2 = SyncStore(path=path, node_id="n1")
    assert s2.get("colour") == "blue"
    assert s2.stats()["oplog_entries"] == 1


@test("typed helpers: put_fact, get_facts, put_wonder, put_episode")
def test_typed_helpers():
    s = _store()
    s.put_fact("water", "is_a", "liquid", 0.99)
    s.put_fact("ice", "is_a", "solid", 0.95)
    facts = s.get_facts()
    assert len(facts) == 2
    subjects = {f[0] for f in facts}
    assert "water" in subjects and "ice" in subjects
    s.put_wonder("wid1", {"question": "why?", "id": "wid1"})
    assert s.get("wq|wid1")["question"] == "why?"
    s.put_episode("ep1", "user asked about water")
    assert s.get("ep|ep1") == "user asked about water"


# ---------------------------------------------------------------------------
# Two-peer sync via InProcessTransport
# ---------------------------------------------------------------------------

@test("two peers converge after one sync round")
def test_two_peer_converge():
    transport = InProcessTransport()
    a = _store("alice")
    b = _store("bob")
    transport.register(a)
    transport.register(b)

    a.put("from_alice", "hi")
    b.put("from_bob", "hey")

    node_a = MeshNode(a, peers=["bob"], transport=transport)
    node_b = MeshNode(b, peers=["alice"], transport=transport)

    node_a.sync_once()
    node_b.sync_once()

    # after two rounds both have each other's writes
    assert a.get("from_bob") == "hey"
    assert b.get("from_alice") == "hi"


@test("conflict resolution: LWW across peers")
def test_cross_peer_lww():
    transport = InProcessTransport()
    a = _store("alice")
    b = _store("bob")
    transport.register(a)
    transport.register(b)

    t_early = time.time() - 5
    t_late  = time.time()
    a.put("shared", "old", ts=t_early)
    b.put("shared", "NEW", ts=t_late)   # bob has the newer value

    node_a = MeshNode(a, peers=["bob"], transport=transport)
    node_a.sync_once()

    assert a.get("shared") == "NEW"     # alice got the newer value


@test("three-peer ring converges in two rounds")
def test_three_peer_ring():
    transport = InProcessTransport()
    stores = {n: _store(n) for n in ("A", "B", "C")}
    for s in stores.values():
        transport.register(s)

    # each peer writes one unique key
    stores["A"].put("from_a", 1)
    stores["B"].put("from_b", 2)
    stores["C"].put("from_c", 3)

    nodes = {
        "A": MeshNode(stores["A"], peers=["B"], transport=transport),
        "B": MeshNode(stores["B"], peers=["A", "C"], transport=transport),
        "C": MeshNode(stores["C"], peers=["B"], transport=transport),
    }

    # two rounds
    for _ in range(2):
        for node in nodes.values():
            node.sync_once()

    # all three stores should have all three keys
    for name, s in stores.items():
        assert s.get("from_a") == 1, f"{name} missing from_a"
        assert s.get("from_b") == 2, f"{name} missing from_b"
        assert s.get("from_c") == 3, f"{name} missing from_c"


@test("offline queue: writes delivered on reconnect")
def test_offline_queue():
    transport = InProcessTransport()
    a = _store("alice")
    b = _store("bob")
    transport.register(a)
    # NOTE: b is NOT registered → b is "offline" from alice's perspective

    a.put("offline_write", "will arrive later")
    node_a = MeshNode(a, peers=["bob"], transport=transport)
    node_a.sync_once()
    # bob not reachable → entries in offline queue
    assert len(node_a._offline_queue.get("bob", [])) > 0

    # bring bob online
    transport.register(b)
    node_a.sync_once()
    # queued entries flushed → bob now has alice's data
    assert b.get("offline_write") == "will arrive later"


# ---------------------------------------------------------------------------
# HTTP transport (real TCP, loopback)
# ---------------------------------------------------------------------------

KEY = "k" * 32          # >= 16 bytes
BADKEY = "x" * 32


def _server(store, allowed=("alice",), key=KEY):
    server = HttpTransport.make_server(store, host="127.0.0.1", port=0,
                                       shared_key=key, allowed_peers=set(allowed))
    server.start()
    return server


def _raw(port, path, body: bytes, *, key=KEY, node="alice", method="POST",
         ts=None, nonce=None, sign_body=None):
    """Send a hand-built signed request with http.client. Returns (status, payload)."""
    import http.client
    ts = ts if ts is not None else str(time.time())
    nonce = nonce if nonce is not None else os.urandom(8).hex()
    sig = _sign(key.encode(), "request", method, path, node, ts, nonce,
                body if sign_body is None else sign_body)
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request(method, path, body=body, headers={
            "Content-Type": "application/json", "Content-Length": str(len(body)),
            "X-Mesh-Node": node, "X-Mesh-Timestamp": ts,
            "X-Mesh-Nonce": nonce, "X-Mesh-Signature": sig})
        r = conn.getresponse()
        return r.status, r.read()
    finally:
        conn.close()


def _push_body(entry) -> bytes:
    import json
    return json.dumps({"entries": [entry.to_dict()]}).encode()


@test("HTTP push/pull between authenticated server and client (loopback)")
def test_http_transport():
    server_store = _store("bob")
    server_store.put("server_key", "server_val")
    server = _server(server_store)
    try:
        client = HttpTransport("alice", KEY, {"bob"})
        peer = f"http://127.0.0.1:{server.port}"

        assert client.health(peer)

        pulled = client.pull(set(), peer)
        assert any(e.key == "server_key" for e in pulled)

        client_store = _store("alice")
        entry = client_store.put("client_key", "client_val")
        assert client.push([entry], peer) == 1
        assert server_store.get("client_key") == "client_val"
    finally:
        server.stop()


@test("HTTP: two MeshNodes converge over authenticated transport")
def test_http_meshnode_roundtrip():
    bob_store = _store("bob")
    bob_store.put("from_bob", 2)
    server = _server(bob_store)
    try:
        alice_store = _store("alice")
        alice_store.put("from_alice", 1)
        node = MeshNode(alice_store, peers=[f"http://127.0.0.1:{server.port}"],
                        transport=HttpTransport("alice", KEY, {"bob"}))
        res = node.sync_once()
        r = next(iter(res.values()))
        assert not r["errors"], r
        assert alice_store.get("from_bob") == 2
        assert bob_store.get("from_alice") == 1
    finally:
        server.stop()


@test("HTTP: wrong key → AuthError, nothing merged, nothing marked delivered")
def test_http_wrong_key():
    server_store = _store("bob")
    server = _server(server_store)
    try:
        peer = f"http://127.0.0.1:{server.port}"
        bad = HttpTransport("alice", BADKEY, {"bob"})
        e = _store("alice").put("evil", 1)
        for call in (lambda: bad.push([e], peer), lambda: bad.pull(set(), peer)):
            try:
                call()
            except AuthError:
                pass
            else:
                raise AssertionError("expected AuthError")
        assert not bad.health(peer)
        assert server_store.get("evil") is None

        # MeshNode: error surfaced, entries stay queued, never marked known
        alice_store = _store("alice")
        alice_store.put("mine", "v")
        node = MeshNode(alice_store, peers=[peer], transport=bad)
        r = node.sync_once()[peer]
        assert any("AuthError" in err for err in r["errors"]), r
        assert r["pushed"] == 0 and r["queued"] == 1
        assert node._peer_known[peer] == set()
        assert server_store.get("mine") is None
    finally:
        server.stop()


@test("HTTP: node_id outside the allowlist is rejected")
def test_http_not_allowlisted():
    server_store = _store("bob")
    server = _server(server_store, allowed=("alice",))
    try:
        peer = f"http://127.0.0.1:{server.port}"
        mallory = HttpTransport("mallory", KEY, {"bob"})   # right key, wrong identity
        e = _store("mallory").put("evil", 1)
        try:
            mallory.push([e], peer)
        except AuthError:
            pass
        else:
            raise AssertionError("expected AuthError")
        assert server_store.get("evil") is None
        status, _ = _raw(server.port, "/sync/health", b"", node="mallory",
                         method="GET")
        assert status == 401
    finally:
        server.stop()


@test("HTTP: replayed request (same nonce) is rejected")
def test_http_replay():
    server_store = _store("bob")
    server = _server(server_store)
    try:
        e = _store("alice").put("k", "v")
        body = _push_body(e)
        nonce = "fixednonce0001"
        s1, _ = _raw(server.port, "/sync/push", body, nonce=nonce)
        assert s1 == 200 and server_store.get("k") == "v"
        server_store.delete("k")                       # observe a second merge, if any
        s2, _ = _raw(server.port, "/sync/push", body, nonce=nonce)
        assert s2 == 401
        assert server_store.get("k") is None           # replay merged nothing
    finally:
        server.stop()


@test("HTTP: tampered body is rejected and nothing is merged")
def test_http_tamper():
    server_store = _store("bob")
    server = _server(server_store)
    try:
        good = _store("alice").put("k", "good")
        evil = _store("alice").put("k", "evil", ts=time.time() + 100)
        status, _ = _raw(server.port, "/sync/push", _push_body(evil),
                         sign_body=_push_body(good))
        assert status == 401
        assert server_store.get("k") is None
        assert server_store.stats()["oplog_entries"] == 0
    finally:
        server.stop()


@test("HTTP: stale timestamp (outside skew window) is rejected")
def test_http_stale_timestamp():
    server_store = _store("bob")
    server = _server(server_store)
    try:
        e = _store("alice").put("k", "v")
        status, _ = _raw(server.port, "/sync/push", _push_body(e),
                         ts=str(time.time() - 3600))
        assert status == 401 and server_store.get("k") is None
    finally:
        server.stop()


@test("HTTP: unauthenticated request to every endpoint gets 401")
def test_http_unsigned():
    import http.client
    server_store = _store("bob")
    server = _server(server_store)
    try:
        for method, path in (("GET", "/sync/health"), ("POST", "/sync/push"),
                             ("POST", "/sync/pull")):
            conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
            conn.request(method, path, body=b"{}" if method == "POST" else None)
            assert conn.getresponse().status == 401, path
            conn.close()
    finally:
        server.stop()


@test("HTTP: forged response (server with wrong key) is rejected by client")
def test_http_forged_response():
    evil_store = _store("bob")
    evil_store.put("poison", "x")
    # attacker's server happens to accept alice but signs with a different key
    server = HttpTransport.make_server(evil_store, host="127.0.0.1", port=0,
                                       shared_key=BADKEY, allowed_peers={"alice"})
    server.start()
    try:
        client = HttpTransport("alice", KEY, {"bob"})
        peer = f"http://127.0.0.1:{server.port}"
        try:
            client.pull(set(), peer)
        except AuthError:
            pass
        else:
            raise AssertionError("expected AuthError")
    finally:
        server.stop()


@test("HTTP: no unauthenticated / plaintext network defaults")
def test_http_secure_defaults():
    for bad in (lambda: HttpTransport(),                        # key/allowlist required
                lambda: HttpTransport("a", "short", {"b"}),     # key too short
                lambda: HttpTransport("a", KEY, set()),         # empty allowlist
                lambda: HttpTransport.make_server(_store("s"), shared_key=KEY,
                                                  allowed_peers=set()),
                lambda: HttpTransport.make_server(_store("s"), host="0.0.0.0", port=0,
                                                  shared_key=KEY,
                                                  allowed_peers={"a"})):  # no TLS
        try:
            bad()
        except (TypeError, ValueError):
            continue
        raise AssertionError("insecure configuration was accepted")
    client = HttpTransport("a", KEY, {"b"})
    try:
        client.pull(set(), "http://192.0.2.1:7700")             # plain http, non-loopback
    except ValueError:
        pass
    else:
        raise AssertionError("plain http to non-loopback peer was attempted")


@test("HTTP health check returns False for unreachable host")
def test_http_health_unreachable():
    client = HttpTransport("alice", KEY, {"bob"})
    assert not client.health("http://127.0.0.1:1")


# ---------------------------------------------------------------------------
# Offline queue / failure handling
# ---------------------------------------------------------------------------

class _FlakyTransport:
    """Wraps a transport; push/pull raise while `fail` is set."""

    def __init__(self, inner, exc):
        self.inner, self.exc, self.fail = inner, exc, True
        self.pushes = []

    def pull(self, known_ids, peer):
        if self.fail:
            raise self.exc
        return self.inner.pull(known_ids, peer)

    def push(self, entries, peer):
        if self.fail:
            raise self.exc
        self.pushes.append([e.entry_id for e in entries])
        return self.inner.push(entries, peer)


@test("failed push queues entries once (no loss, no duplication) and flushes later")
def test_push_failure_queue():
    inner = InProcessTransport()
    a, b = _store("alice"), _store("bob")
    inner.register(a)
    inner.register(b)
    flaky = _FlakyTransport(inner, AuthError("rejected"))
    node = MeshNode(a, peers=["bob"], transport=flaky)
    a.put("k1", 1)
    a.put("k2", 2)
    for _ in range(3):                                   # repeated failures
        r = node.sync_once()["bob"]
        assert r["queued"] == 2 and r["pushed"] == 0, r
        assert any("AuthError" in e for e in r["errors"]), r
    assert len(node._offline_queue["bob"]) == 2
    assert node._peer_known["bob"] == set()
    a.put("k3", 3)
    flaky.fail = False
    r = node.sync_once()["bob"]
    assert r["pushed"] == 3 and r["queued"] == 0 and not r["errors"], r
    assert sorted(flaky.pushes[0]) == sorted(e.entry_id for e in a.entries_since(set()))
    assert b.get("k1") == 1 and b.get("k3") == 3
    assert node.stats()["offline_queued"] == 0


# ---------------------------------------------------------------------------
# Expanded state sync (fake controller built from real stores)
# ---------------------------------------------------------------------------

def _device(node_id: str, transport, peer: str, reviser=None):
    """A MeshNode plus a controller-shaped namespace backed by real stores."""
    from types import SimpleNamespace
    from synapseos.phase5.memory import SemanticMemory, EpisodicMemory
    from synapseos.affect import EmotionTracker
    from synapseos.soul import (AestheticPreferences, SelfReflection, ValueSystem,
                                WonderJournal)
    d = tempfile.mkdtemp(prefix="mesh_dev_")
    p = lambda n: os.path.join(d, n)
    sem = SemanticMemory(store_path=p("sem.json"), kg_path=p("none.json"))
    ctl = SimpleNamespace(
        memory=SimpleNamespace(semantic=sem,
                               episodic=EpisodicMemory(store_path=p("ep.json"))),
        emotion_tracker=EmotionTracker(semantic_memory=sem, state_path=p("aff.json")),
        aesthetics=AestheticPreferences(store_path=p("aes.json")),
        values=ValueSystem(store_path=p("val.json")),
        reflection=SelfReflection(store_path=p("ref.json")),
        wonder=WonderJournal(path=p("won.json")),
        web_loop=SimpleNamespace(reviser=reviser) if reviser else None,
    )
    store = SyncStore(path=p("sync.json"), node_id=node_id)
    transport.register(store)
    node = MeshNode(store, peers=[peer], transport=transport)
    return node, ctl


def _round(*devs):
    """export → sync → apply on each device, twice, so everything settles."""
    for _ in range(2):
        for node, ctl in devs:
            node.wire_controller(ctl)
            node.sync_once()
            node.apply_to_controller(ctl)


@test("affect, aesthetics, values and reflection sync across devices")
def test_soul_state_sync():
    t = InProcessTransport()
    a = _device("alice", t, "bob")
    b = _device("bob", t, "alice")
    na, ca = a
    nb, cb = b

    ca.emotion_tracker.observe("I am so frustrated with this broken scraper, it fails again")
    assert ca.emotion_tracker.state.observations > 0
    ca.aesthetics.encounter("proof", "a short symmetric proof: x = x", "proof")
    ca.aesthetics.encounter("tune", "a b a b a b", "melody")
    ca.values.challenge("honesty", "no")                  # pressure hardens conviction
    ca.reflection.capture(ca.emotion_tracker, ca.aesthetics, ca.values,
                          ca.memory, note="alice snapshot")

    _round(a, b)

    assert cb.emotion_tracker.state.observations == ca.emotion_tracker.state.observations
    assert cb.emotion_tracker.state.valence_ema == ca.emotion_tracker.state.valence_ema
    assert cb.aesthetics.exposures == ca.aesthetics.exposures == 2
    assert cb.aesthetics.weights == ca.aesthetics.weights
    assert len(cb.aesthetics.reactions) == 2
    assert cb.values.values["honesty"].pressure_count == 1
    assert cb.values.values["honesty"].conviction == ca.values.values["honesty"].conviction
    assert [s.note for s in cb.reflection.snapshots] == ["alice snapshot"]
    # untouched built-in values do not travel (they would clobber real convictions)
    assert nb.store.get("value|privacy") is None


@test("repeated export is idempotent (no oplog growth) and re-apply is a no-op")
def test_export_idempotent():
    t = InProcessTransport()
    a = _device("alice", t, "bob")
    na, ca = a
    ca.memory.semantic.add_fact("tea", "is_a", "drink", confidence=0.9, source="user_stated")
    ca.wonder.wonder_about("Why is tea bitter?", topic="tea")
    na.wire_controller(ca)
    n1 = na.store.stats()["oplog_entries"]
    out = na.wire_controller(ca)
    assert na.store.stats()["oplog_entries"] == n1
    assert not any(out.values()), out
    applied = na.apply_to_controller(ca)
    assert applied["facts_added"] == applied["facts_removed"] == 0
    assert applied["wonders_added"] == applied["wonders_updated"] == 0


@test("wonder updates (resolve / intensity / notes) propagate, not just new wonders")
def test_wonder_updates_sync():
    t = InProcessTransport()
    a = _device("alice", t, "bob")
    b = _device("bob", t, "alice")
    na, ca = a
    nb, cb = b
    w = ca.wonder.wonder_about("Why is the sky blue?", topic="sky")
    _round(a, b)
    assert cb.wonder._wonders[w.id].status == "open"

    ca.wonder.note(w.id, "Rayleigh scattering")
    ca.wonder.resolve(w.id, "scattering favours short wavelengths")
    ca.wonder._wonders[w.id].last_visited = time.time() + 5
    _round(a, b)
    got = cb.wonder._wonders[w.id]
    assert got.status == "resolved"
    assert got.answer == "scattering favours short wavelengths"
    assert "Rayleigh scattering" in got.notes

    # a dormant/ignored wonder on bob never un-resolves it
    cb.wonder._wonders[w.id].status = "dormant"
    _round(a, b)
    assert ca.wonder._wonders[w.id].status == "resolved"
    assert cb.wonder._wonders[w.id].status == "resolved"


@test("incoming facts go through the belief reviser, not a direct write")
def test_facts_use_reviser():
    from types import SimpleNamespace

    class SpyReviser:
        def __init__(self):
            self.calls = []
            self.saved = 0

        def integrate(self, fact, source_url, source=None):
            self.calls.append((fact.subject, fact.predicate, fact.obj,
                               fact.confidence, source_url, source))
            return SimpleNamespace(action="rejected")      # reviser says no

        def save(self):
            self.saved += 1

    t = InProcessTransport()
    spy = SpyReviser()
    a = _device("alice", t, "bob")
    b = _device("bob", t, "alice", reviser=spy)
    na, ca = a
    nb, cb = b
    ca.memory.semantic.add_fact("mars", "has_property", "red", confidence=0.8,
                                source="user_stated")
    _round(a, b)
    assert len(spy.calls) == 1, spy.calls
    s, p, o, conf, url, src = spy.calls[0]
    assert (s, p, o) == ("mars", "has_property", "red") and conf == 0.8
    assert src == "mesh:alice" and url.startswith("mesh://alice")
    assert not cb.memory.semantic.facts          # rejected → NOT written directly
    assert spy.saved >= 1
    # a rejected fact is judged once per remote version, not re-litigated each round
    _round(a, b)
    assert len(spy.calls) == 1


@test("belief reviser resolves a functional conflict against a remote fact")
def test_facts_conflict_resolution():
    t = InProcessTransport()
    a = _device("alice", t, "bob")
    b = _device("bob", t, "alice")
    na, ca = a
    nb, cb = b
    # bob firmly believes cat; alice's weak, unknown-provenance dog loses
    cb.memory.semantic.add_fact("rex", "is_a", "cat", confidence=0.95, source="user_stated")
    ca.memory.semantic.add_fact("rex", "is_a", "dog", confidence=0.2, source="inferred")
    ca.memory.semantic.add_fact("rex", "has_property", "loud", confidence=0.7,
                                source="inferred")
    _round(a, b)
    objs = {(f.predicate, f.obj) for f in cb.memory.semantic.facts if f.subject == "rex"}
    assert ("is_a", "cat") in objs and ("is_a", "dog") not in objs
    assert ("has_property", "loud") in objs              # no conflict → added
    mesh_fact = next(f for f in cb.memory.semantic.facts if f.obj == "loud")
    assert mesh_fact.source == "mesh:alice"


@test("fact deletion propagates as a tombstone and the fact is removed remotely")
def test_fact_tombstone():
    t = InProcessTransport()
    a = _device("alice", t, "bob")
    b = _device("bob", t, "alice")
    na, ca = a
    nb, cb = b
    ca.memory.semantic.add_fact("pluto", "is_a", "planet", confidence=0.9,
                                source="user_stated")
    _round(a, b)
    assert any(f.obj == "planet" for f in cb.memory.semantic.facts)

    time.sleep(0.01)
    assert ca.memory.semantic.remove_fact("pluto", "is_a", "planet") == 1
    _round(a, b)
    key = SyncStore.fact_key("pluto", "is_a", "planet")
    assert na.store.is_deleted(key) and nb.store.is_deleted(key)
    assert not any(f.subject == "pluto" for f in cb.memory.semantic.facts)
    assert not any(f.subject == "pluto" for f in ca.memory.semantic.facts)
    assert not nb.store.get_facts()


@test("a tombstone does not delete a fact the device learned afterwards")
def test_tombstone_loses_to_newer_fact():
    from datetime import datetime, timedelta
    t = InProcessTransport()
    a = _device("alice", t, "bob")
    b = _device("bob", t, "alice")
    na, ca = a
    nb, cb = b
    ca.memory.semantic.add_fact("pluto", "is_a", "planet", confidence=0.9,
                                source="user_stated")
    _round(a, b)
    time.sleep(0.01)
    ca.memory.semantic.remove_fact("pluto", "is_a", "planet")
    na.wire_controller(ca)
    na.sync_once()
    # bob re-learned the fact after alice's deletion
    fact = next(f for f in cb.memory.semantic.facts if f.subject == "pluto")
    fact.timestamp = (datetime.now() + timedelta(seconds=30)).isoformat()
    _round(a, b)
    assert any(f.subject == "pluto" for f in cb.memory.semantic.facts)
    assert nb.store.get(SyncStore.fact_key("pluto", "is_a", "planet")) is not None


@test("store: later local edit supersedes a remote value even with a slow clock")
def test_causal_local_write():
    s = _store("alice")
    future = time.time() + 1000
    s.merge([SyncEntry.make("bob", "k", "remote", ts=future)])
    s.put("k", "local")
    assert s.get("k") == "local"
    s.delete("k")
    assert s.get("k") is None and s.is_deleted("k")


@test("store: device-local bookkeeping persists but is not synced")
def test_store_meta_local():
    d = tempfile.mkdtemp(prefix="mesh_test_")
    path = os.path.join(d, "sync.json")
    s = SyncStore(path=path, node_id="n1")
    s.meta_replace("facts_known", {"a": 1})
    s.save()
    assert SyncStore(path=path, node_id="n1").meta_all("facts_known") == {"a": 1}
    assert all(e.key != "facts_known" for e in s.entries_since(set()))


# ---------------------------------------------------------------------------
# Controller integration
# ---------------------------------------------------------------------------

@test("wire_controller exports facts and wonder; apply imports them")
def test_controller_integration():
    from synapseos.phase5.cognitive_controller import CognitiveController
    d = tempfile.mkdtemp(prefix="mesh_test_")

    # build a minimal controller with known semantic facts + a wonder
    c = CognitiveController(
        memory_episodic_path=os.path.join(d, "e.json"),
        memory_semantic_path=os.path.join(d, "s.json"),
        kg_path=os.path.join(d, "k.json"),
        rules_path=os.path.join(d, "r.json"),
        affect_state_path=os.path.join(d, "a.json"),
        world_model_dir=os.path.join(d, "wm"),
        enable_mesh=False,      # we'll wire manually
    )
    c.memory.semantic.add_fact("synapse", "is_a", "junction", confidence=0.9,
                               source="test")
    from synapseos.soul.wonder import WonderJournal
    c.wonder = WonderJournal(path=os.path.join(d, "wonder.json"))
    c.wonder.wonder_about("What is a synapse?", topic="synapse")

    # create a second store and node to receive exports
    transport = InProcessTransport()
    store_a = SyncStore(path=os.path.join(d, "store_a.json"), node_id="device_a")
    store_b = SyncStore(path=os.path.join(d, "store_b.json"), node_id="device_b")
    transport.register(store_a)
    transport.register(store_b)

    node_a = MeshNode(store_a, peers=["device_b"], transport=transport)
    node_b = MeshNode(store_b, peers=["device_a"], transport=transport)

    # export controller state into node_a's store
    node_a.wire_controller(c)
    assert len(store_a.get_facts()) >= 1
    # wonder keys are hashes; check the value contains the question text
    wq_keys = store_a.keys("wq|")
    assert len(wq_keys) >= 1
    assert any("synapse" in str(store_a.get(k)) for k in wq_keys)

    # sync to node_b
    node_a.sync_once()

    # node_b should now have the facts in its store
    assert len(store_b.get_facts()) >= 1

    c.shutdown()


@test("controller enable_mesh flag wires and stats reports")
def test_controller_mesh_flag():
    from synapseos.phase5.cognitive_controller import CognitiveController
    d = tempfile.mkdtemp(prefix="mesh_test_")
    transport = InProcessTransport()

    c = CognitiveController(
        memory_episodic_path=os.path.join(d, "e.json"),
        memory_semantic_path=os.path.join(d, "s.json"),
        kg_path=os.path.join(d, "k.json"),
        rules_path=os.path.join(d, "r.json"),
        affect_state_path=os.path.join(d, "a.json"),
        world_model_dir=os.path.join(d, "wm"),
        enable_mesh=True,
        mesh_store_path=os.path.join(d, "mesh.json"),
        mesh_transport=transport,
        mesh_peers=[],
    )
    assert c.mesh is not None
    s = c.stats()
    assert s["mesh"] is not None
    assert s["mesh"]["node_id"] == c.mesh.store.node_id
    c.shutdown()
    # store should be persisted on shutdown
    assert Path(os.path.join(d, "mesh.json")).exists()


@test("mesh_sync() exports + sync round + apply returns counts")
def test_mesh_sync_method():
    from synapseos.phase5.cognitive_controller import CognitiveController
    d = tempfile.mkdtemp(prefix="mesh_test_")
    transport = InProcessTransport()

    c = CognitiveController(
        memory_episodic_path=os.path.join(d, "e.json"),
        memory_semantic_path=os.path.join(d, "s.json"),
        kg_path=os.path.join(d, "k.json"),
        rules_path=os.path.join(d, "r.json"),
        affect_state_path=os.path.join(d, "a.json"),
        world_model_dir=os.path.join(d, "wm"),
        enable_mesh=True,
        mesh_store_path=os.path.join(d, "mesh.json"),
        mesh_transport=transport,
        mesh_peers=[],
    )
    c.memory.semantic.add_fact("photon", "is_a", "particle", confidence=0.8,
                               source="test")
    result = c.mesh_sync()
    assert "synced" in result and "applied" in result
    assert c.mesh.store.get_facts()   # facts were exported
    c.shutdown()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_put_get, test_lww_wins, test_tie_break, test_tombstone,
        test_idempotent, test_entries_since, test_persist, test_typed_helpers,
        test_two_peer_converge, test_cross_peer_lww, test_three_peer_ring,
        test_offline_queue,
        test_http_transport, test_http_meshnode_roundtrip, test_http_wrong_key,
        test_http_not_allowlisted, test_http_replay, test_http_tamper,
        test_http_stale_timestamp, test_http_unsigned, test_http_forged_response,
        test_http_secure_defaults, test_http_health_unreachable,
        test_push_failure_queue, test_soul_state_sync, test_export_idempotent,
        test_wonder_updates_sync, test_facts_use_reviser,
        test_facts_conflict_resolution, test_fact_tombstone,
        test_tombstone_loses_to_newer_fact, test_causal_local_write,
        test_store_meta_local,
        test_controller_integration, test_controller_mesh_flag,
        test_mesh_sync_method,
    ]
    for t in tests:
        t()

    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed
    print(f"\n{'='*60}")
    print(f"Mesh Tests: {passed}/{len(results)} passed\n")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        detail = f"  → {r['detail']}" if not r["passed"] else ""
        print(f"  [{status}] {r['name']}  ({r['ms']}ms){detail}")
    sys.exit(0 if failed == 0 else 1)
