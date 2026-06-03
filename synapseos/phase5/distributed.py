"""
Phase 5 — Distributed Multi-Node Architecture.

Provides the interface layer for running SynapseOS pipeline stages across
multiple nodes. Currently implemented as an in-process local bus — the
interface is identical to the distributed version so swapping to gRPC,
ZMQ, or Redis pub/sub requires only a new MessageBus backend.

Architecture:

    ┌──────────────┐    MessageBus    ┌──────────────┐
    │  TextEncoder │ ────────────►   │  JEPAFusion  │
    │  (Node A)    │                 │  (Node B)     │
    └──────────────┘                 └──────────────┘
                                            │
                              ┌─────────────┴─────────────┐
                              ▼                           ▼
                      ┌──────────────┐         ┌──────────────┐
                      │ Neurosymbolic│         │   KAN Layer  │
                      │ (Node C)     │         │   (Node D)   │
                      └──────────────┘         └──────────────┘
                                            │
                                     ┌──────┴──────┐
                                     │  Output     │
                                     │  Renderer   │
                                     └─────────────┘

Each LayerNode:
  - Declares its layer_type and endpoint
  - Has a health_check() → bool
  - Receives a LayerRequest and returns a LayerResponse
  - Falls back to local_fallback() when the bus is unavailable

The DistributedPipeline:
  - Maintains a PipelineTopology (which layer on which node)
  - Routes requests through the MessageBus
  - Handles node failures: if a node is unhealthy, falls back to local execution
  - Collects throughput metrics

Run:
    python -m synapseos.phase5.distributed
    python -m synapseos.phase5.distributed --query "list fields for Sales Invoice"
"""

from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Layer types
# ---------------------------------------------------------------------------

class LayerType(str, Enum):
    TEXT_ENCODER      = "text_encoder"
    JEPA_FUSION       = "jepa_fusion"
    N2S_BRIDGE        = "n2s_bridge"
    NEUROSYMBOLIC     = "neurosymbolic"
    S2N_BRIDGE        = "s2n_bridge"
    KAN_REASONING     = "kan_reasoning"
    OUTPUT_RENDERER   = "output_renderer"
    CREATIVITY        = "creativity"
    CODE_SYNTHESIS    = "code_synthesis"


# ---------------------------------------------------------------------------
# Request / Response
# ---------------------------------------------------------------------------

@dataclass
class LayerRequest:
    request_id: str
    layer_type: LayerType
    payload: dict[str, Any]
    source_node: str
    timestamp: float = field(default_factory=time.time)
    timeout: float = 30.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["layer_type"] = self.layer_type.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LayerRequest":
        d = dict(d)
        d["layer_type"] = LayerType(d["layer_type"])
        return cls(**d)


@dataclass
class LayerResponse:
    request_id: str
    layer_type: LayerType
    payload: dict[str, Any]
    source_node: str
    success: bool
    error: str = ""
    latency_ms: float = 0.0
    fallback_used: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["layer_type"] = self.layer_type.value
        return d


# ---------------------------------------------------------------------------
# Message Bus (in-process local implementation)
# ---------------------------------------------------------------------------

class MessageBus:
    """
    In-process message bus backed by per-node queues.

    Interface contract (identical for gRPC/ZMQ/Redis backends):
      - publish(node_id, request) → None
      - subscribe(node_id, handler) → None
      - request(node_id, request, timeout) → LayerResponse
      - broadcast(request) → list[LayerResponse]
    """

    def __init__(self):
        self._queues: dict[str, queue.Queue] = {}
        self._handlers: dict[str, Callable[[LayerRequest], LayerResponse]] = {}
        self._lock = threading.Lock()
        self._metrics: dict[str, list[float]] = {}   # node_id → list of latency_ms

    def register_node(self, node_id: str, handler: Callable) -> None:
        with self._lock:
            self._queues[node_id]  = queue.Queue()
            self._handlers[node_id] = handler
            self._metrics[node_id]  = []

    def deregister_node(self, node_id: str) -> None:
        with self._lock:
            self._queues.pop(node_id, None)
            self._handlers.pop(node_id, None)
            self._metrics.pop(node_id, None)

    def is_registered(self, node_id: str) -> bool:
        return node_id in self._handlers

    def request(
        self,
        node_id: str,
        req: LayerRequest,
        timeout: float = 30.0,
    ) -> LayerResponse:
        """Synchronous request → response via in-process call."""
        if node_id not in self._handlers:
            return LayerResponse(
                request_id=req.request_id,
                layer_type=req.layer_type,
                payload={},
                source_node="bus",
                success=False,
                error=f"Node '{node_id}' not registered",
            )

        t0 = time.time()
        try:
            handler = self._handlers[node_id]
            response = handler(req)
            latency = (time.time() - t0) * 1000
            response.latency_ms = latency
            self._metrics[node_id].append(latency)
            return response
        except Exception as e:
            return LayerResponse(
                request_id=req.request_id,
                layer_type=req.layer_type,
                payload={},
                source_node=node_id,
                success=False,
                error=str(e),
                latency_ms=(time.time() - t0) * 1000,
            )

    def broadcast(self, req: LayerRequest) -> list[LayerResponse]:
        """Send request to all registered nodes; collect all responses."""
        return [self.request(node_id, req) for node_id in list(self._handlers)]

    def node_latency_stats(self, node_id: str) -> dict:
        lats = self._metrics.get(node_id, [])
        if not lats:
            return {"node": node_id, "calls": 0}
        return {
            "node":   node_id,
            "calls":  len(lats),
            "p50_ms": round(sorted(lats)[len(lats) // 2], 2),
            "p99_ms": round(sorted(lats)[int(len(lats) * 0.99)], 2),
            "mean_ms": round(sum(lats) / len(lats), 2),
        }

    @property
    def registered_nodes(self) -> list[str]:
        return list(self._handlers)


# ---------------------------------------------------------------------------
# Layer Node
# ---------------------------------------------------------------------------

@dataclass
class NodeConfig:
    node_id: str
    layer_type: LayerType
    endpoint: str            # "local" | "grpc://host:port" | "redis://host/channel"
    max_concurrent: int = 4
    health_check_interval: float = 30.0


class LayerNode:
    """
    A single pipeline layer deployed as an independent processing unit.

    Local mode: handler runs in-process.
    Remote mode (future): handler sends/receives via gRPC/ZMQ.
    """

    def __init__(self, config: NodeConfig, bus: MessageBus):
        self.config = config
        self.bus    = bus
        self._healthy = True
        self._last_check = time.time()
        self._request_count = 0
        self._error_count = 0
        bus.register_node(config.node_id, self._process)

    @property
    def node_id(self) -> str:
        return self.config.node_id

    @property
    def layer_type(self) -> LayerType:
        return self.config.layer_type

    def health_check(self) -> bool:
        """Ping the node — override in remote subclass to do a network call."""
        self._last_check = time.time()
        return self._healthy

    def _process(self, req: LayerRequest) -> LayerResponse:
        """Dispatch request to the appropriate local layer handler."""
        self._request_count += 1
        t0 = time.time()
        try:
            result = self._local_execute(req)
            return LayerResponse(
                request_id=req.request_id,
                layer_type=req.layer_type,
                payload=result,
                source_node=self.node_id,
                success=True,
                latency_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            self._error_count += 1
            # Try fallback
            try:
                result = self._local_fallback(req)
                return LayerResponse(
                    request_id=req.request_id,
                    layer_type=req.layer_type,
                    payload=result,
                    source_node=self.node_id,
                    success=True,
                    latency_ms=(time.time() - t0) * 1000,
                    fallback_used=True,
                )
            except Exception as e2:
                return LayerResponse(
                    request_id=req.request_id,
                    layer_type=req.layer_type,
                    payload={},
                    source_node=self.node_id,
                    success=False,
                    error=f"{e} | fallback: {e2}",
                    latency_ms=(time.time() - t0) * 1000,
                )

    def _local_execute(self, req: LayerRequest) -> dict[str, Any]:
        """
        Execute this layer locally. Override per layer type.
        Default: pass payload through unchanged (identity transform for stubs).
        """
        return req.payload

    def _local_fallback(self, req: LayerRequest) -> dict[str, Any]:
        """Minimal fallback when primary execution fails."""
        return {"fallback": True, "input": req.payload.get("text", "")}

    def stats(self) -> dict:
        return {
            "node_id":       self.node_id,
            "layer_type":    self.layer_type.value,
            "endpoint":      self.config.endpoint,
            "healthy":       self._healthy,
            "requests":      self._request_count,
            "errors":        self._error_count,
            "error_rate":    round(self._error_count / max(1, self._request_count), 3),
        }


# ---------------------------------------------------------------------------
# Specialized stub nodes
# ---------------------------------------------------------------------------

class TextEncoderNode(LayerNode):
    """Stub for the text encoder layer."""

    def _local_execute(self, req: LayerRequest) -> dict[str, Any]:
        text = req.payload.get("text", "")
        # Stub: hash-based latent vector (CPU, no mamba-ssm needed)
        import hashlib
        digest = hashlib.md5(text.encode()).digest()
        latent = [b / 255.0 for b in digest] * 4   # 64-dim stub
        return {"latent": latent, "text_len": len(text), "layer": "text_encoder"}


class NeurosymbolicNode(LayerNode):
    """Wraps the real NeurosymbolicCore."""

    def __init__(self, config: NodeConfig, bus: MessageBus,
                 kg_path: str = "data/graph/knowledge_graph.json",
                 rules_path: str = "data/rules/rules.json"):
        super().__init__(config, bus)
        self._core = None
        self._kg_path    = kg_path
        self._rules_path = rules_path

    def _get_core(self):
        if self._core is None:
            from synapseos.phase3.neurosymbolic_core import NeurosymbolicCore
            self._core = NeurosymbolicCore(self._kg_path, self._rules_path)
        return self._core

    def _local_execute(self, req: LayerRequest) -> dict[str, Any]:
        query   = req.payload.get("text", "")
        qt      = req.payload.get("query_type", "retrieve")
        doctype = req.payload.get("doctype", "")
        core = self._get_core()
        from synapseos.phase3.neurosymbolic_core import QueryType
        result = core.query(query, QueryType(qt), doctype)
        return {
            "facts":      result.facts[:5],
            "rules":      result.rules_applied[:5],
            "confidence": result.confidence,
            "verified":   result.verified,
            "layer":      "neurosymbolic",
        }

    def _local_fallback(self, req: LayerRequest) -> dict[str, Any]:
        return {"facts": [], "rules": [], "confidence": 0.0, "verified": False, "fallback": True}


class OutputRendererNode(LayerNode):
    """Produces the final natural language response."""

    def _local_execute(self, req: LayerRequest) -> dict[str, Any]:
        facts = req.payload.get("facts", [])
        query = req.payload.get("query", "")
        if facts:
            summary = f"Found {len(facts)} relevant facts for: {query[:60]}"
        else:
            summary = f"No specific facts found for: {query[:60]}"
        return {"output": summary, "layer": "output_renderer"}


# ---------------------------------------------------------------------------
# Pipeline Topology
# ---------------------------------------------------------------------------

@dataclass
class PipelineTopology:
    """
    Maps each layer type to a node_id and endpoint.
    Serializable to JSON for config-driven deployment.
    """
    stages: list[dict[str, str]]   # ordered list of {layer_type, node_id, endpoint}

    @classmethod
    def local_default(cls, kg_path: str = "data/graph/knowledge_graph.json",
                      rules_path: str = "data/rules/rules.json") -> "PipelineTopology":
        """Single-machine topology — all layers on local node."""
        return cls(stages=[
            {"layer_type": LayerType.TEXT_ENCODER.value,    "node_id": "node-encoder",  "endpoint": "local"},
            {"layer_type": LayerType.NEUROSYMBOLIC.value,   "node_id": "node-ns",       "endpoint": "local"},
            {"layer_type": LayerType.OUTPUT_RENDERER.value, "node_id": "node-renderer", "endpoint": "local"},
        ])

    @classmethod
    def distributed_default(cls) -> "PipelineTopology":
        """Multi-node topology — each layer on a separate host (stubs for now)."""
        return cls(stages=[
            {"layer_type": LayerType.TEXT_ENCODER.value,    "node_id": "node-encoder",  "endpoint": "grpc://encoder:50051"},
            {"layer_type": LayerType.JEPA_FUSION.value,     "node_id": "node-jepa",     "endpoint": "grpc://jepa:50052"},
            {"layer_type": LayerType.NEUROSYMBOLIC.value,   "node_id": "node-ns",       "endpoint": "grpc://ns:50053"},
            {"layer_type": LayerType.KAN_REASONING.value,   "node_id": "node-kan",      "endpoint": "grpc://kan:50054"},
            {"layer_type": LayerType.OUTPUT_RENDERER.value, "node_id": "node-renderer", "endpoint": "grpc://renderer:50055"},
        ])

    def to_json(self) -> str:
        return json.dumps({"stages": self.stages}, indent=2)

    def save(self, path: str) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str) -> "PipelineTopology":
        data = json.loads(Path(path).read_text())
        return cls(stages=data["stages"])


# ---------------------------------------------------------------------------
# Distributed Pipeline
# ---------------------------------------------------------------------------

class DistributedPipeline:
    """
    Orchestrates LayerNodes via the MessageBus.

    - Builds a topology from PipelineTopology config
    - Routes requests through each stage in order
    - On node failure: skips that stage and uses fallback output
    - Collects per-stage latency metrics

    This class is the drop-in replacement for synapseos/phase3/pipeline.py
    in a distributed deployment.
    """

    def __init__(
        self,
        topology: PipelineTopology | None = None,
        kg_path: str = "data/graph/knowledge_graph.json",
        rules_path: str = "data/rules/rules.json",
    ):
        self.bus      = MessageBus()
        self.topology = topology or PipelineTopology.local_default(kg_path, rules_path)
        self._nodes: dict[str, LayerNode] = {}
        self._stage_latencies: dict[str, list[float]] = {}
        self._kg_path    = kg_path
        self._rules_path = rules_path
        self._build_nodes()

    def _build_nodes(self) -> None:
        for stage in self.topology.stages:
            layer_type = LayerType(stage["layer_type"])
            node_id    = stage["node_id"]
            endpoint   = stage["endpoint"]
            config     = NodeConfig(node_id=node_id, layer_type=layer_type, endpoint=endpoint)

            # Instantiate appropriate node class
            if layer_type == LayerType.TEXT_ENCODER:
                node = TextEncoderNode(config, self.bus)
            elif layer_type == LayerType.NEUROSYMBOLIC:
                node = NeurosymbolicNode(config, self.bus, self._kg_path, self._rules_path)
            elif layer_type == LayerType.OUTPUT_RENDERER:
                node = OutputRendererNode(config, self.bus)
            else:
                node = LayerNode(config, self.bus)

            self._nodes[node_id] = node
            self._stage_latencies[node_id] = []

    def health_check_all(self) -> dict[str, bool]:
        return {nid: node.health_check() for nid, node in self._nodes.items()}

    def run(self, query: str) -> dict[str, Any]:
        """
        Execute the pipeline for a query.
        Returns the final output dict from the last stage.
        """
        payload: dict[str, Any] = {"text": query, "query": query}
        stage_outputs: list[dict] = []

        for stage in self.topology.stages:
            node_id    = stage["node_id"]
            layer_type = LayerType(stage["layer_type"])

            req = LayerRequest(
                request_id=str(uuid.uuid4()),
                layer_type=layer_type,
                payload=payload,
                source_node="pipeline",
            )

            response = self.bus.request(node_id, req)
            self._stage_latencies[node_id].append(response.latency_ms)

            if response.success:
                payload = {**payload, **response.payload}
            else:
                # Stage failed — continue with current payload, mark degraded
                payload["_degraded"] = payload.get("_degraded", []) + [node_id]

            stage_outputs.append({
                "node":        node_id,
                "layer":       layer_type.value,
                "success":     response.success,
                "latency_ms":  round(response.latency_ms, 2),
                "fallback":    response.fallback_used,
            })

        payload["_stage_trace"] = stage_outputs
        return payload

    def latency_report(self) -> dict:
        report = {}
        for node_id, lats in self._stage_latencies.items():
            if not lats:
                report[node_id] = {"calls": 0}
                continue
            report[node_id] = {
                "calls":   len(lats),
                "p50_ms":  round(sorted(lats)[len(lats) // 2], 2),
                "mean_ms": round(sum(lats) / len(lats), 2),
                "total_ms": round(sum(lats), 2),
            }
        return report

    def node_stats(self) -> list[dict]:
        return [node.stats() for node in self._nodes.values()]


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Distributed Pipeline Demo")
    parser.add_argument("--query", default="What fields are required for a Sales Invoice?")
    parser.add_argument("--topology", default="local", choices=["local", "distributed"])
    args = parser.parse_args()

    print(f"\nDistributed Pipeline [{args.topology} mode]")
    print(f"Query: {args.query}\n{'='*60}\n")

    if args.topology == "distributed":
        topo = PipelineTopology.distributed_default()
        print("[Note] Distributed topology — remote nodes will fallback to local stubs")
    else:
        topo = PipelineTopology.local_default()

    pipeline = DistributedPipeline(topology=topo)

    # Health check
    health = pipeline.health_check_all()
    print("Node health:")
    for node_id, ok in health.items():
        print(f"  {node_id}: {'OK' if ok else 'DOWN'}")

    # Run
    t0 = time.time()
    result = pipeline.run(args.query)
    total_ms = (time.time() - t0) * 1000

    print(f"\nOutput: {result.get('output', '[no output renderer output]')}")
    print(f"\nStage trace:")
    for stage in result.get("_stage_trace", []):
        fallback = " (fallback)" if stage.get("fallback") else ""
        status = "OK" if stage["success"] else "FAILED"
        print(f"  [{status}] {stage['node']} ({stage['layer']}) — {stage['latency_ms']}ms{fallback}")

    print(f"\nTotal: {total_ms:.1f}ms")
    print(f"\nLatency report:")
    for node_id, stats in pipeline.latency_report().items():
        print(f"  {node_id}: {stats}")


if __name__ == "__main__":
    main()
