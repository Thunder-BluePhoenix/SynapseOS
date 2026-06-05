"""
Tests for the network transport — real multi-node distribution over HTTP/TCP.

Verifies NodeServer + RemoteNodeProxy + NetworkBus actually carry LayerRequests
across real sockets (localhost), including health checks and fault tolerance.

Run:
    python tests/test_transport.py
    python -m pytest tests/test_transport.py -v
"""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from synapseos.phase5.distributed import LayerRequest, LayerType
from synapseos.phase5.transport import (
    NodeServer, RemoteNodeProxy, NetworkBus, free_port,
)

KG    = "data/graph/knowledge_graph.json"
RULES = "data/rules/rules.json"

results = []


def test(name: str):
    def decorator(fn):
        def wrapper():
            t0 = time.time()
            try:
                fn()
                r = {"name": name, "passed": True,  "ms": round((time.time()-t0)*1000, 1), "detail": ""}
            except AssertionError as e:
                r = {"name": name, "passed": False, "ms": round((time.time()-t0)*1000, 1), "detail": str(e)}
            except Exception as e:
                r = {"name": name, "passed": False, "ms": round((time.time()-t0)*1000, 1), "detail": f"{type(e).__name__}: {e}"}
            results.append(r)
        wrapper.__name__ = fn.__name__
        return wrapper
    return decorator


def _req(layer: LayerType, payload: dict) -> LayerRequest:
    return LayerRequest(request_id=str(uuid.uuid4()), layer_type=layer,
                        payload=payload, source_node="test")


# ---------------------------------------------------------------------------

@test("transport: free_port returns a usable port")
def test_free_port():
    p = free_port()
    assert isinstance(p, int)
    assert 1024 < p < 65536


@test("transport: NodeServer starts and exposes an endpoint")
def test_server_start():
    srv = NodeServer(LayerType.TEXT_ENCODER, "t-enc", kg_path=KG, rules_path=RULES)
    ep = srv.start()
    try:
        assert ep.startswith("http://127.0.0.1:")
    finally:
        srv.stop()


@test("transport: health check over HTTP returns True for live node")
def test_health_live():
    srv = NodeServer(LayerType.TEXT_ENCODER, "t-enc2", kg_path=KG, rules_path=RULES)
    srv.start()
    time.sleep(0.2)
    try:
        proxy = RemoteNodeProxy(srv.endpoint)
        assert proxy.health_check() is True
    finally:
        srv.stop()


@test("transport: health check returns False for dead endpoint")
def test_health_dead():
    proxy = RemoteNodeProxy(f"http://127.0.0.1:{free_port()}")  # nothing listening
    assert proxy.health_check() is False


@test("transport: TextEncoder request over TCP returns latent")
def test_encoder_over_tcp():
    srv = NodeServer(LayerType.TEXT_ENCODER, "t-enc3", kg_path=KG, rules_path=RULES)
    srv.start()
    time.sleep(0.2)
    try:
        proxy = RemoteNodeProxy(srv.endpoint)
        resp = proxy.request(_req(LayerType.TEXT_ENCODER, {"text": "Sales Invoice"}))
        assert resp.success
        assert "latent" in resp.payload
        assert len(resp.payload["latent"]) == 64
    finally:
        srv.stop()


@test("transport: OutputRenderer request over TCP returns output")
def test_renderer_over_tcp():
    srv = NodeServer(LayerType.OUTPUT_RENDERER, "t-rend", kg_path=KG, rules_path=RULES)
    srv.start()
    time.sleep(0.2)
    try:
        proxy = RemoteNodeProxy(srv.endpoint)
        resp = proxy.request(_req(LayerType.OUTPUT_RENDERER,
                                  {"query": "test", "facts": [{"f": "x"}]}))
        assert resp.success
        assert "output" in resp.payload
    finally:
        srv.stop()


@test("transport: request to down node returns graceful error (no crash)")
def test_down_node_graceful():
    proxy = RemoteNodeProxy(f"http://127.0.0.1:{free_port()}")
    resp = proxy.request(_req(LayerType.TEXT_ENCODER, {"text": "x"}))
    assert not resp.success
    assert "transport" in resp.error


@test("transport: NetworkBus routes to remote node")
def test_networkbus_remote():
    srv = NodeServer(LayerType.TEXT_ENCODER, "t-enc4", kg_path=KG, rules_path=RULES)
    srv.start()
    time.sleep(0.2)
    try:
        bus = NetworkBus()
        bus.register_remote("t-enc4", srv.endpoint)
        assert "t-enc4" in bus.remote_nodes
        resp = bus.request("t-enc4", _req(LayerType.TEXT_ENCODER, {"text": "hi"}))
        assert resp.success
        assert bus.health("t-enc4") is True
    finally:
        srv.stop()


@test("transport: 3-node pipeline routes end-to-end over TCP")
def test_three_node_pipeline():
    servers = [
        NodeServer(LayerType.TEXT_ENCODER,    "p-enc",  kg_path=KG, rules_path=RULES),
        NodeServer(LayerType.NEUROSYMBOLIC,   "p-ns",   kg_path=KG, rules_path=RULES),
        NodeServer(LayerType.OUTPUT_RENDERER, "p-rend", kg_path=KG, rules_path=RULES),
    ]
    for s in servers:
        s.start()
    time.sleep(0.3)
    try:
        bus = NetworkBus()
        for s in servers:
            bus.register_remote(s.node_id, s.endpoint)

        payload = {"text": "What fields for Sales Invoice?", "query": "What fields for Sales Invoice?"}
        stages = [("p-enc", LayerType.TEXT_ENCODER),
                  ("p-ns", LayerType.NEUROSYMBOLIC),
                  ("p-rend", LayerType.OUTPUT_RENDERER)]
        ok = 0
        for node_id, layer in stages:
            resp = bus.request(node_id, _req(layer, payload))
            if resp.success:
                payload = {**payload, **resp.payload}
                ok += 1
        assert ok == 3, f"Expected all 3 stages ok, got {ok}"
        assert "output" in payload
    finally:
        for s in servers:
            s.stop()


@test("transport: pipeline survives a node going down (fault tolerance)")
def test_fault_tolerance():
    servers = [
        NodeServer(LayerType.TEXT_ENCODER,    "f-enc",  kg_path=KG, rules_path=RULES),
        NodeServer(LayerType.NEUROSYMBOLIC,   "f-ns",   kg_path=KG, rules_path=RULES),
        NodeServer(LayerType.OUTPUT_RENDERER, "f-rend", kg_path=KG, rules_path=RULES),
    ]
    for s in servers:
        s.start()
    time.sleep(0.3)
    try:
        bus = NetworkBus()
        for s in servers:
            bus.register_remote(s.node_id, s.endpoint)

        servers[1].stop()   # kill the middle node
        time.sleep(0.2)

        payload = {"text": "q", "query": "q"}
        degraded, succeeded = [], []
        for node_id, layer in [("f-enc", LayerType.TEXT_ENCODER),
                               ("f-ns", LayerType.NEUROSYMBOLIC),
                               ("f-rend", LayerType.OUTPUT_RENDERER)]:
            resp = bus.request(node_id, _req(layer, payload))
            if resp.success:
                payload = {**payload, **resp.payload}
                succeeded.append(node_id)
            else:
                degraded.append(node_id)
        # the two live nodes still worked; the dead one degraded gracefully
        assert "f-ns" in degraded
        assert "f-enc" in succeeded and "f-rend" in succeeded
        assert "output" in payload   # pipeline still produced output
    finally:
        for s in servers:
            s.stop()


@test("transport: LayerResponse survives JSON round-trip over the wire")
def test_response_serde_over_wire():
    srv = NodeServer(LayerType.TEXT_ENCODER, "s-enc", kg_path=KG, rules_path=RULES)
    srv.start()
    time.sleep(0.2)
    try:
        proxy = RemoteNodeProxy(srv.endpoint)
        resp = proxy.request(_req(LayerType.TEXT_ENCODER, {"text": "roundtrip"}))
        # layer_type must come back as the enum, not a string
        assert isinstance(resp.layer_type, LayerType)
        assert resp.request_id  # preserved
    finally:
        srv.stop()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_free_port,
        test_server_start,
        test_health_live,
        test_health_dead,
        test_encoder_over_tcp,
        test_renderer_over_tcp,
        test_down_node_graceful,
        test_networkbus_remote,
        test_three_node_pipeline,
        test_fault_tolerance,
        test_response_serde_over_wire,
    ]

    for t in tests:
        t()

    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed

    print(f"\n{'='*60}")
    print(f"Transport (multi-node) Tests: {passed}/{len(results)} passed\n")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        detail = f"  → {r['detail']}" if not r["passed"] else ""
        print(f"  [{status}] {r['name']}  ({r['ms']}ms){detail}")

    sys.exit(0 if failed == 0 else 1)
