"""
Network transport for the distributed pipeline — real cross-machine nodes.

`distributed.py` proved the architecture in-process (MessageBus). This makes
the same `endpoint` strings actually carry traffic over the network, turning
"process" into "machine" without changing the pipeline code. It uses only the
Python standard library (http.server + http.client), so there is no `grpcio`
or other install — it runs here today and on separate hosts unchanged.

Pieces:
  NodeServer     — wraps a LayerNode, serves it over HTTP (POST /process, GET /health)
  RemoteNodeProxy— client with the same request()/health_check() shape as a LayerNode,
                   but sends LayerRequest as JSON over TCP and parses LayerResponse
  NetworkBus     — a MessageBus that routes by endpoint: "local" → in-process,
                   "http://host:port" → RemoteNodeProxy

Fault tolerance: a down node returns an error LayerResponse (connection refused →
the pipeline marks that stage degraded and continues), never crashing the caller.

Demo (spins up 3 real HTTP node servers on localhost and routes a pipeline
through them over TCP, then kills one to show graceful degradation):
    python -m synapseos.phase5.transport
"""

from __future__ import annotations

import http.client
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from synapseos.phase5.distributed import (
    LayerNode, LayerRequest, LayerResponse, LayerType, NodeConfig,
    MessageBus, TextEncoderNode, NeurosymbolicNode, OutputRendererNode,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _make_node(layer_type: LayerType, node_id: str, bus: MessageBus,
               kg_path: str, rules_path: str) -> LayerNode:
    cfg = NodeConfig(node_id=node_id, layer_type=layer_type, endpoint="local")
    if layer_type == LayerType.TEXT_ENCODER:
        return TextEncoderNode(cfg, bus)
    if layer_type == LayerType.NEUROSYMBOLIC:
        return NeurosymbolicNode(cfg, bus, kg_path, rules_path)
    if layer_type == LayerType.OUTPUT_RENDERER:
        return OutputRendererNode(cfg, bus)
    return LayerNode(cfg, bus)


# ---------------------------------------------------------------------------
# Node server (runs on a worker machine)
# ---------------------------------------------------------------------------

class NodeServer:
    """
    Serves a single LayerNode over HTTP. On a real deployment this process runs
    on a separate machine; here we run several on localhost ports.
    """

    def __init__(
        self,
        layer_type: LayerType,
        node_id: str,
        host: str = "127.0.0.1",
        port: int | None = None,
        kg_path: str = "data/graph/knowledge_graph.json",
        rules_path: str = "data/rules/rules.json",
    ):
        self.host = host
        self.port = port or free_port()
        self.layer_type = layer_type
        self.node_id = node_id

        self._bus  = MessageBus()
        self._node = _make_node(layer_type, node_id, self._bus, kg_path, rules_path)
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> str:
        node = self._node
        bus  = self._bus
        node_id = self.node_id

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass  # silence

            def _send(self, code: int, obj: dict):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/health":
                    self._send(200, {"healthy": node.health_check(), "node": node_id,
                                     "layer": node.layer_type.value})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                if self.path != "/process":
                    self._send(404, {"error": "not found"})
                    return
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                try:
                    req = LayerRequest.from_dict(json.loads(raw))
                    resp = bus.request(node_id, req)
                    self._send(200, resp.to_dict())
                except Exception as e:
                    self._send(500, {"error": str(e)})

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self.endpoint

    def stop(self) -> None:
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._thread:
            self._thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Remote proxy (client side — same shape as a LayerNode)
# ---------------------------------------------------------------------------

class RemoteNodeProxy:
    """
    Client stub for a remote NodeServer. Implements request()/health_check()
    so it is interchangeable with an in-process LayerNode.
    """

    def __init__(self, endpoint: str, timeout: float = 10.0):
        self.endpoint = endpoint
        self.timeout  = timeout
        parsed = urlparse(endpoint)
        self.host = parsed.hostname
        self.port = parsed.port

    def _conn(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)

    def health_check(self) -> bool:
        try:
            c = self._conn()
            c.request("GET", "/health")
            r = c.getresponse()
            ok = r.status == 200 and json.loads(r.read()).get("healthy", False)
            c.close()
            return bool(ok)
        except Exception:
            return False

    def request(self, req: LayerRequest) -> LayerResponse:
        try:
            c = self._conn()
            body = json.dumps(req.to_dict()).encode()
            c.request("POST", "/process", body, {"Content-Type": "application/json"})
            r = c.getresponse()
            data = json.loads(r.read())
            c.close()
            data["layer_type"] = LayerType(data["layer_type"])
            return LayerResponse(**data)
        except Exception as e:
            # Down node → graceful error response (pipeline marks stage degraded)
            return LayerResponse(
                request_id=req.request_id,
                layer_type=req.layer_type,
                payload={},
                source_node=self.endpoint,
                success=False,
                error=f"transport: {e}",
                latency_ms=0.0,
            )


# ---------------------------------------------------------------------------
# Network-aware bus
# ---------------------------------------------------------------------------

class NetworkBus(MessageBus):
    """
    MessageBus that routes by endpoint:
      - registered local node_id  → in-process (super().request)
      - "http://host:port"        → RemoteNodeProxy over TCP

    Lets a DistributedPipeline mix local and remote stages transparently.
    """

    def __init__(self):
        super().__init__()
        self._remotes: dict[str, RemoteNodeProxy] = {}

    def register_remote(self, node_id: str, endpoint: str) -> None:
        self._remotes[node_id] = RemoteNodeProxy(endpoint)

    def request(self, node_id: str, req: LayerRequest, timeout: float = 30.0) -> LayerResponse:
        if node_id in self._remotes:
            return self._remotes[node_id].request(req)
        return super().request(node_id, req, timeout)

    def health(self, node_id: str) -> bool:
        if node_id in self._remotes:
            return self._remotes[node_id].health_check()
        return self.is_registered(node_id)

    @property
    def remote_nodes(self) -> list[str]:
        return list(self._remotes)


# ---------------------------------------------------------------------------
# Demo — 3 real HTTP node servers + fault tolerance
# ---------------------------------------------------------------------------

def _demo() -> None:
    import uuid
    kg, rules = "data/graph/knowledge_graph.json", "data/rules/rules.json"

    print("Starting 3 node servers on localhost (real HTTP/TCP) ...")
    servers = [
        NodeServer(LayerType.TEXT_ENCODER,    "node-encoder",  kg_path=kg, rules_path=rules),
        NodeServer(LayerType.NEUROSYMBOLIC,   "node-ns",       kg_path=kg, rules_path=rules),
        NodeServer(LayerType.OUTPUT_RENDERER, "node-renderer", kg_path=kg, rules_path=rules),
    ]
    for s in servers:
        ep = s.start()
        print(f"  {s.node_id:14s} → {ep}")
    time.sleep(0.3)  # let servers bind

    # Client bus talks to all three over the network
    bus = NetworkBus()
    for s in servers:
        bus.register_remote(s.node_id, s.endpoint)

    def run_stage(node_id, layer, payload):
        req = LayerRequest(request_id=str(uuid.uuid4()), layer_type=layer,
                           payload=payload, source_node="client")
        return bus.request(node_id, req)

    print("\nHealth checks (over HTTP):")
    for s in servers:
        print(f"  {s.node_id:14s} healthy={bus.health(s.node_id)}")

    print("\nRouting a query through the network pipeline:")
    query = "What fields are required for a Sales Invoice?"
    payload = {"text": query, "query": query}
    for node_id, layer in [("node-encoder", LayerType.TEXT_ENCODER),
                           ("node-ns", LayerType.NEUROSYMBOLIC),
                           ("node-renderer", LayerType.OUTPUT_RENDERER)]:
        resp = run_stage(node_id, layer, payload)
        if resp.success:
            payload = {**payload, **resp.payload}
        print(f"  {node_id:14s} ok={resp.success} {resp.latency_ms:.0f}ms (over TCP)")
    print(f"  output: {payload.get('output', '(none)')}")

    # Fault tolerance: kill the neurosymbolic node, re-run
    print("\nFault tolerance — killing node-ns, re-running:")
    servers[1].stop()
    time.sleep(0.2)
    payload = {"text": query, "query": query}
    degraded = []
    for node_id, layer in [("node-encoder", LayerType.TEXT_ENCODER),
                           ("node-ns", LayerType.NEUROSYMBOLIC),
                           ("node-renderer", LayerType.OUTPUT_RENDERER)]:
        resp = run_stage(node_id, layer, payload)
        if resp.success:
            payload = {**payload, **resp.payload}
        else:
            degraded.append(node_id)
        status = "ok" if resp.success else f"DOWN ({resp.error[:30]})"
        print(f"  {node_id:14s} {status}")
    print(f"  → pipeline survived; degraded stages: {degraded}")
    print(f"  → output still produced: {payload.get('output', '(none)')}")

    for s in servers:
        s.stop()
    print("\nServers stopped. Same architecture as in-process — only the transport changed.")


if __name__ == "__main__":
    _demo()
