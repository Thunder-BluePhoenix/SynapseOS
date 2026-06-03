"""
Phase 5 — ATLAS: Infrastructure Specialist.

ATLAS reasons about the SynapseOS deployment topology, scaling requirements,
resource allocation, and fault tolerance posture.

It consumes data from:
  - distributed.py   → current node topology and health
  - nova.py          → KG bottlenecks and rule density (identifies hot layers)
  - drift_monitor.py → Liquid NN adaptation status

Capabilities:
  1. Topology analysis     — evaluate pipeline topology for single points of failure
  2. Scaling recommendations — which layers need horizontal scaling and why
  3. Deployment config     — generate docker-compose or systemd unit stubs per node
  4. Health report         — aggregate health across all registered nodes
  5. Resource estimation   — CPU/memory per layer type based on complexity
  6. Full infrastructure audit — comprehensive one-shot report

ATLAS never modifies infrastructure — it only analyses and recommends.
All recommendations are traceable to specific data points.

Run:
    python -m synapseos.phase5.agents.atlas
    python -m synapseos.phase5.agents.atlas --query "identify single points of failure in the pipeline"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any

from synapseos.phase5.agents.base import SubAgent, AgentResult, AgentCapability
from synapseos.phase5.distributed import (
    DistributedPipeline, PipelineTopology, LayerType,
)


# ---------------------------------------------------------------------------
# Capability declaration
# ---------------------------------------------------------------------------

ATLAS_CAPABILITY = AgentCapability(
    name="ATLAS",
    keywords=[
        "infrastructure", "deploy", "deployment", "topology", "node",
        "scale", "scaling", "replica", "horizontal", "vertical",
        "fault", "resilience", "failover", "single point of failure",
        "health", "monitor", "status", "uptime",
        "resource", "cpu", "memory", "load", "capacity",
        "docker", "container", "service", "cluster", "distributed",
        "latency", "throughput", "concurrent", "requests",
    ],
    anti_keywords=["security audit", "business rule", "optimize rule"],
    base_confidence=0.58,
    description="Infrastructure — deployment topology, scaling, fault tolerance, resource estimation",
)


# ---------------------------------------------------------------------------
# ATLAS data structures
# ---------------------------------------------------------------------------

# Resource estimates per layer type (CPU cores, memory MB, notes)
LAYER_RESOURCES: dict[str, dict] = {
    LayerType.TEXT_ENCODER.value: {
        "cpu_cores": 2, "memory_mb": 2048, "gpu": False,
        "note": "CPU hash-fallback is lightweight; Mamba needs GPU + 14GB VRAM",
        "scale_trigger": "input_qps > 50",
    },
    LayerType.JEPA_FUSION.value: {
        "cpu_cores": 1, "memory_mb": 512, "gpu": False,
        "note": "Lightweight projection layer; scales well horizontally",
        "scale_trigger": "input_qps > 200",
    },
    LayerType.N2S_BRIDGE.value: {
        "cpu_cores": 1, "memory_mb": 256, "gpu": False,
        "note": "Simple classifier; very fast",
        "scale_trigger": "input_qps > 500",
    },
    LayerType.NEUROSYMBOLIC.value: {
        "cpu_cores": 4, "memory_mb": 1024, "gpu": False,
        "note": "KG traversal + rule eval is CPU-bound; cache hot DocTypes",
        "scale_trigger": "input_qps > 20",
    },
    LayerType.S2N_BRIDGE.value: {
        "cpu_cores": 1, "memory_mb": 256, "gpu": False,
        "note": "Embedding projection; lightweight",
        "scale_trigger": "input_qps > 200",
    },
    LayerType.KAN_REASONING.value: {
        "cpu_cores": 2, "memory_mb": 512, "gpu": False,
        "note": "Symbolic formula evaluation; deterministic and fast",
        "scale_trigger": "input_qps > 100",
    },
    LayerType.OUTPUT_RENDERER.value: {
        "cpu_cores": 4, "memory_mb": 4096, "gpu": True,
        "note": "Mistral/Llama generation is the pipeline bottleneck; needs 2–4 replicas",
        "scale_trigger": "input_qps > 5",
    },
    LayerType.CREATIVITY.value: {
        "cpu_cores": 2, "memory_mb": 1024, "gpu": False,
        "note": "Graph walk is CPU-bound; can run async",
        "scale_trigger": "input_qps > 30",
    },
    LayerType.CODE_SYNTHESIS.value: {
        "cpu_cores": 2, "memory_mb": 512, "gpu": False,
        "note": "Symbolic synthesis is fast; AST validation is CPU-only",
        "scale_trigger": "input_qps > 50",
    },
}


@dataclass
class NodeHealthEntry:
    node_id: str
    layer_type: str
    endpoint: str
    healthy: bool
    request_count: int
    error_count: int
    error_rate: float


@dataclass
class ScalingRecommendation:
    layer_type: str
    current_replicas: int
    recommended_replicas: int
    reason: str
    priority: str           # HIGH | MEDIUM | LOW
    estimated_cost_usd: str


@dataclass
class FaultRisk:
    node_id: str
    layer_type: str
    risk: str               # CRITICAL | HIGH | MEDIUM
    description: str
    mitigation: str


@dataclass
class ResourceEstimate:
    layer_type: str
    cpu_cores: int
    memory_mb: int
    gpu_required: bool
    scale_trigger: str
    note: str


@dataclass
class DeploymentConfig:
    """Docker Compose or systemd unit stub for a layer node."""
    node_id: str
    layer_type: str
    service_name: str
    image: str
    cpu_limit: str
    memory_limit: str
    replicas: int
    env_vars: dict[str, str]
    ports: list[str]

    def to_compose_fragment(self) -> str:
        lines = [
            f"  {self.service_name}:",
            f"    image: {self.image}",
            f"    deploy:",
            f"      replicas: {self.replicas}",
            f"      resources:",
            f"        limits:",
            f"          cpus: '{self.cpu_limit}'",
            f"          memory: {self.memory_limit}",
            f"    environment:",
        ]
        for k, v in self.env_vars.items():
            lines.append(f"      {k}: \"{v}\"")
        if self.ports:
            lines.append("    ports:")
            for p in self.ports:
                lines.append(f"      - \"{p}\"")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ATLAS Agent
# ---------------------------------------------------------------------------

class ATLASAgent(SubAgent):
    """
    ATLAS — infrastructure specialist.

    Analyses the distributed pipeline topology and produces actionable
    deployment, scaling, and fault tolerance recommendations.
    Everything is data-driven and traceable.
    """

    capability = ATLAS_CAPABILITY

    def __init__(
        self,
        kg_path: str = "data/graph/knowledge_graph.json",
        rules_path: str = "data/rules/rules.json",
    ):
        self._kg_path    = kg_path
        self._rules_path = rules_path
        self._topology   = PipelineTopology.local_default(kg_path, rules_path)
        self._pipeline   = DistributedPipeline(
            topology=self._topology,
            kg_path=kg_path,
            rules_path=rules_path,
        )

    # ------------------------------------------------------------------
    # Health report
    # ------------------------------------------------------------------

    def health_report(self) -> list[NodeHealthEntry]:
        health_map = self._pipeline.health_check_all()
        node_stats = {s["node_id"]: s for s in self._pipeline.node_stats()}
        entries = []
        for stage in self._topology.stages:
            node_id    = stage["node_id"]
            layer_type = stage["layer_type"]
            endpoint   = stage["endpoint"]
            stats      = node_stats.get(node_id, {})
            entries.append(NodeHealthEntry(
                node_id=node_id,
                layer_type=layer_type,
                endpoint=endpoint,
                healthy=health_map.get(node_id, False),
                request_count=stats.get("requests", 0),
                error_count=stats.get("errors", 0),
                error_rate=stats.get("error_rate", 0.0),
            ))
        return entries

    # ------------------------------------------------------------------
    # Fault tolerance analysis
    # ------------------------------------------------------------------

    def fault_risks(self) -> list[FaultRisk]:
        """
        Identify nodes with no redundancy (replicas == 1 and no failover).
        A single-node deployment of any required layer is a SPOF.
        """
        risks = []
        # In the current local topology every node is a SPOF
        for stage in self._topology.stages:
            layer_type = stage["layer_type"]
            node_id    = stage["node_id"]
            res        = LAYER_RESOURCES.get(layer_type, {})
            gpu_req    = res.get("gpu", False)

            risk_level = "CRITICAL" if layer_type in (
                LayerType.OUTPUT_RENDERER.value,
                LayerType.NEUROSYMBOLIC.value,
            ) else "HIGH"

            risks.append(FaultRisk(
                node_id=node_id,
                layer_type=layer_type,
                risk=risk_level,
                description=(
                    f"{node_id} has no replica — if it fails, "
                    f"the {layer_type} stage is unavailable"
                ),
                mitigation=(
                    "Add ≥2 replicas behind a load balancer. "
                    f"{'GPU required for production instance.' if gpu_req else 'CPU-only — easy to replicate.'}"
                ),
            ))

        return risks

    # ------------------------------------------------------------------
    # Scaling recommendations
    # ------------------------------------------------------------------

    def scaling_recommendations(
        self, target_qps: int = 50
    ) -> list[ScalingRecommendation]:
        """
        Recommend replica counts for each layer to handle target_qps.
        Uses LAYER_RESOURCES scale_trigger thresholds.
        """
        recs = []
        for stage in self._topology.stages:
            layer_type = stage["layer_type"]
            res = LAYER_RESOURCES.get(layer_type, {})
            trigger_str = res.get("scale_trigger", "input_qps > 100")

            # Parse trigger threshold
            threshold = 100
            try:
                threshold = int(trigger_str.split(">")[1].strip())
            except (IndexError, ValueError):
                pass

            if target_qps <= threshold:
                replicas = 1
                priority = "LOW"
            elif target_qps <= threshold * 3:
                replicas = 2
                priority = "MEDIUM"
            else:
                replicas = max(2, target_qps // threshold)
                priority = "HIGH"

            if layer_type == LayerType.OUTPUT_RENDERER.value:
                # Generation is always the bottleneck
                replicas = max(replicas, 2)
                priority = "HIGH"

            gpu = res.get("gpu", False)
            cost = f"~${replicas * (800 if gpu else 200)}/mo" if replicas > 1 else "$0 extra"

            recs.append(ScalingRecommendation(
                layer_type=layer_type,
                current_replicas=1,
                recommended_replicas=replicas,
                reason=f"Scale trigger: {trigger_str} (target={target_qps} QPS)",
                priority=priority,
                estimated_cost_usd=cost,
            ))

        recs.sort(key=lambda r: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(r.priority, 3))
        return recs

    # ------------------------------------------------------------------
    # Resource estimation
    # ------------------------------------------------------------------

    def resource_estimates(self) -> list[ResourceEstimate]:
        estimates = []
        seen: set[str] = set()
        for stage in self._topology.stages:
            lt = stage["layer_type"]
            if lt in seen:
                continue
            seen.add(lt)
            res = LAYER_RESOURCES.get(lt, {
                "cpu_cores": 1, "memory_mb": 512, "gpu": False,
                "note": "Unknown layer", "scale_trigger": "n/a",
            })
            estimates.append(ResourceEstimate(
                layer_type=lt,
                cpu_cores=res["cpu_cores"],
                memory_mb=res["memory_mb"],
                gpu_required=res["gpu"],
                scale_trigger=res["scale_trigger"],
                note=res["note"],
            ))
        return estimates

    # ------------------------------------------------------------------
    # Deployment config generation
    # ------------------------------------------------------------------

    def generate_deployment_configs(
        self, replicas_override: dict[str, int] | None = None
    ) -> list[DeploymentConfig]:
        configs = []
        replicas_override = replicas_override or {}

        for stage in self._topology.stages:
            layer_type = stage["layer_type"]
            node_id    = stage["node_id"]
            res        = LAYER_RESOURCES.get(layer_type, {})

            replicas  = replicas_override.get(layer_type, 1)
            cpu_cores = res.get("cpu_cores", 1)
            memory_mb = res.get("memory_mb", 512)
            gpu       = res.get("gpu", False)

            service_name = layer_type.replace("_", "-")
            image = f"synapseos/{service_name}:latest"
            port  = 50050 + list(LayerType).index(LayerType(layer_type))

            configs.append(DeploymentConfig(
                node_id=node_id,
                layer_type=layer_type,
                service_name=service_name,
                image=image,
                cpu_limit=str(cpu_cores),
                memory_limit=f"{memory_mb}M",
                replicas=replicas,
                env_vars={
                    "KG_PATH":    self._kg_path,
                    "RULES_PATH": self._rules_path,
                    "LAYER_TYPE": layer_type,
                    "LOG_LEVEL":  "INFO",
                    **({"GPU_ENABLED": "true"} if gpu else {}),
                },
                ports=[f"{port}:{port}"],
            ))
        return configs

    def generate_compose_yaml(self, replicas_override: dict[str, int] | None = None) -> str:
        configs = self.generate_deployment_configs(replicas_override)
        lines = [
            "version: '3.9'",
            "services:",
        ]
        for cfg in configs:
            lines.append(cfg.to_compose_fragment())
        lines += [
            "",
            "networks:",
            "  default:",
            "    name: synapseos-net",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Latency report
    # ------------------------------------------------------------------

    def latency_report(self) -> dict[str, dict]:
        return self._pipeline.latency_report()

    # ------------------------------------------------------------------
    # SubAgent interface
    # ------------------------------------------------------------------

    def _handle(self, query: str, context: Any) -> AgentResult:
        trace: list[str] = [f"ATLAS received: {query[:80]}"]
        q_lower = query.lower()

        if any(kw in q_lower for kw in ("health", "status", "uptime", "up")):
            report = self.health_report()
            healthy = sum(1 for e in report if e.healthy)
            lines = [f"Node Health Report — {healthy}/{len(report)} healthy"]
            for e in report:
                icon = "✓" if e.healthy else "✗"
                lines.append(
                    f"  [{icon}] {e.node_id} ({e.layer_type}) | "
                    f"reqs={e.request_count} errors={e.error_count} "
                    f"err_rate={e.error_rate:.1%}"
                )
            output = "\n".join(lines)
            structured = {
                "total_nodes": len(report),
                "healthy": healthy,
                "nodes": [{"node": e.node_id, "healthy": e.healthy} for e in report],
            }
            trace.append(f"Health report: {healthy}/{len(report)} healthy")
            confidence = 0.90

        elif any(kw in q_lower for kw in ("fault", "resilience", "spof", "single point", "failure")):
            risks = self.fault_risks()
            critical = [r for r in risks if r.risk == "CRITICAL"]
            high     = [r for r in risks if r.risk == "HIGH"]
            lines    = [f"Fault Tolerance Analysis — {len(risks)} risks identified"]
            for r in sorted(risks, key=lambda x: {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}.get(x.risk, 3)):
                lines.append(f"\n  [{r.risk}] {r.node_id} ({r.layer_type})")
                lines.append(f"    Risk: {r.description}")
                lines.append(f"    Fix : {r.mitigation}")
            output = "\n".join(lines)
            structured = {
                "total_risks": len(risks),
                "critical":    len(critical),
                "high":        len(high),
                "risks": [{"node": r.node_id, "risk": r.risk, "layer": r.layer_type} for r in risks],
            }
            trace.append(f"Fault analysis: {len(critical)} CRITICAL, {len(high)} HIGH")
            confidence = 0.88

        elif any(kw in q_lower for kw in ("scale", "scaling", "replica", "concurrent", "throughput", "qps")):
            # Extract target QPS from query if present
            import re
            qps_match = re.search(r"(\d+)\s*(qps|requests|concurrent)", q_lower)
            target_qps = int(qps_match.group(1)) if qps_match else 50
            recs = self.scaling_recommendations(target_qps=target_qps)
            lines = [f"Scaling Recommendations (target: {target_qps} QPS)"]
            for r in recs:
                if r.recommended_replicas > 1 or r.priority == "HIGH":
                    lines.append(
                        f"  [{r.priority}] {r.layer_type}: "
                        f"{r.current_replicas} → {r.recommended_replicas} replicas | {r.estimated_cost_usd}"
                    )
                    lines.append(f"    {r.reason}")
            output = "\n".join(lines)
            structured = {
                "target_qps": target_qps,
                "recommendations": [
                    {"layer": r.layer_type, "replicas": r.recommended_replicas, "priority": r.priority}
                    for r in recs
                ],
            }
            trace.append(f"Scaling for {target_qps} QPS: {sum(r.recommended_replicas for r in recs)} total replicas")
            confidence = 0.85

        elif any(kw in q_lower for kw in ("deploy", "docker", "compose", "container", "config")):
            yaml_out = self.generate_compose_yaml()
            output   = f"Docker Compose configuration:\n\n```yaml\n{yaml_out}\n```"
            configs  = self.generate_deployment_configs()
            structured = {
                "services": [{"service": c.service_name, "image": c.image, "replicas": c.replicas}
                             for c in configs],
                "compose_yaml": yaml_out,
            }
            trace.append(f"Generated compose config: {len(configs)} services")
            confidence = 0.87

        elif any(kw in q_lower for kw in ("resource", "cpu", "memory", "ram", "estimate")):
            estimates = self.resource_estimates()
            lines = ["Resource Estimates per Layer:"]
            total_cpu = 0
            total_mem = 0
            for e in estimates:
                gpu_tag = " [GPU]" if e.gpu_required else ""
                lines.append(
                    f"  {e.layer_type}: {e.cpu_cores} CPU cores, "
                    f"{e.memory_mb}MB RAM{gpu_tag}"
                )
                lines.append(f"    {e.note}")
                total_cpu += e.cpu_cores
                total_mem += e.memory_mb
            lines.append(f"\nTotal (1 replica each): {total_cpu} cores, {total_mem}MB RAM")
            output = "\n".join(lines)
            structured = {
                "estimates": [
                    {"layer": e.layer_type, "cpu": e.cpu_cores, "mem_mb": e.memory_mb, "gpu": e.gpu_required}
                    for e in estimates
                ],
                "total_cpu_cores": total_cpu,
                "total_memory_mb": total_mem,
            }
            trace.append(f"Resource estimate: {total_cpu} cores, {total_mem}MB total")
            confidence = 0.88

        else:
            # Full infrastructure audit
            health   = self.health_report()
            risks    = self.fault_risks()
            recs     = self.scaling_recommendations(target_qps=50)
            estimates = self.resource_estimates()

            healthy_n = sum(1 for e in health if e.healthy)
            critical_n = sum(1 for r in risks if r.risk == "CRITICAL")
            scale_needed = sum(1 for r in recs if r.recommended_replicas > 1)
            total_cpu = sum(e.cpu_cores for e in estimates)
            total_mem = sum(e.memory_mb for e in estimates)

            lines = [
                "=== ATLAS Infrastructure Audit ===",
                "",
                f"Nodes           : {healthy_n}/{len(health)} healthy",
                f"Fault risks     : {critical_n} CRITICAL, {len(risks)-critical_n} HIGH",
                f"Scale needed    : {scale_needed} layers require additional replicas at 50 QPS",
                f"Resources (1×)  : {total_cpu} CPU cores, {total_mem}MB RAM",
                f"GPU layers      : {sum(1 for e in estimates if e.gpu_required)} (Output Renderer)",
                "",
                "Top fault risks:",
            ]
            for r in risks[:3]:
                lines.append(f"  [{r.risk}] {r.layer_type} — {r.description[:60]}")

            lines += ["", "Scale priorities (50 QPS):"]
            for r in [x for x in recs if x.recommended_replicas > 1][:3]:
                lines.append(f"  [{r.priority}] {r.layer_type}: ×{r.recommended_replicas} ({r.estimated_cost_usd})")

            output = "\n".join(lines)
            structured = {
                "healthy_nodes":  healthy_n,
                "total_nodes":    len(health),
                "critical_risks": critical_n,
                "scale_needed":   scale_needed,
                "total_cpu":      total_cpu,
                "total_mem_mb":   total_mem,
            }
            trace.append("Full infrastructure audit complete")
            confidence = 0.88

        trace.append(f"ATLAS complete | confidence={confidence:.2f}")
        return AgentResult(
            agent_name="ATLAS",
            query=query,
            output=output,
            structured=structured,
            confidence=confidence,
            latency_ms=0.0,
            traceable=True,
            trace=trace,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ATLAS — Infrastructure Agent")
    parser.add_argument("--query", default="run a full infrastructure audit")
    args = parser.parse_args()

    agent = ATLASAgent()
    print(f"\nATLAS — Infrastructure Specialist")
    print(f"Query: {args.query}\n{'='*60}\n")

    result = agent.handle(args.query)
    print(result.output)
    print(f"\nConfidence : {result.confidence:.3f}")
    print(f"Latency    : {result.latency_ms:.1f}ms")
    print(f"\nTrace:")
    for step in result.trace:
        print(f"  {step}")


if __name__ == "__main__":
    main()
