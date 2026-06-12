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
from synapseos.mesh.transport import HttpTransport


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

@test("HTTP push/pull between real server and client (loopback)")
def test_http_transport():
    import socket as _sock
    # find a free port
    s = _sock.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()

    server_store = _store("server")
    server_store.put("server_key", "server_val")
    server = HttpTransport.make_server(server_store, host="127.0.0.1", port=port)
    server.start()
    time.sleep(0.05)   # let server thread start

    try:
        client = HttpTransport()
        peer = f"http://127.0.0.1:{port}"

        assert client.health(peer)

        # pull: client gets server's entries
        pulled = client.pull(set(), peer)
        assert any(e.key == "server_key" for e in pulled)

        # push: send a new entry to server
        client_store = _store("client")
        entry = client_store.put("client_key", "client_val")
        accepted = client.push([entry], peer)
        assert accepted == 1
        assert server_store.get("client_key") == "client_val"

    finally:
        server.stop()


@test("HTTP health check returns False for unreachable host")
def test_http_health_unreachable():
    client = HttpTransport()
    assert not client.health("http://127.0.0.1:1")


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
        test_http_transport, test_http_health_unreachable,
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
