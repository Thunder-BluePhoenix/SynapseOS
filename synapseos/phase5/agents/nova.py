"""
Phase 5 — NOVA: Optimization Specialist.

NOVA analyses the SynapseOS system itself and recommends improvements:
  1. Rule density analysis    — which DocTypes have the most rules (hotspots)
  2. KG connectivity analysis — bottleneck nodes (high in + out degree)
  3. Cache strategy           — which DocType lookups are worth caching
  4. Rule redundancy          — duplicate condition+event pairs within a DocType
  5. System benchmark report  — current KG + rules corpus stats
  6. Latency profile          — which pipeline stages are slowest

Unlike ARIEL (works on the ERP domain) and NYX (security),
NOVA works on the infrastructure of SynapseOS itself.
Every recommendation includes a rationale grounded in the data.

Run:
    python -m synapseos.phase5.agents.nova
    python -m synapseos.phase5.agents.nova --query "which doctypes should we cache?"
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from synapseos.phase5.agents.base import SubAgent, AgentResult, AgentCapability
from synapseos.phase3.neurosymbolic_core import KnowledgeGraph, RulesEngine


# ---------------------------------------------------------------------------
# Capability declaration
# ---------------------------------------------------------------------------

NOVA_CAPABILITY = AgentCapability(
    name="NOVA",
    keywords=[
        "optimize", "optimise", "performance", "bottleneck", "slow",
        "cache", "caching", "latency", "throughput", "speed",
        "redundan", "duplicate", "hotspot", "benchmark", "profile",
        "improve", "efficiency", "cost", "memory", "resource",
        "scale", "scaling", "tuning", "recommendation",
    ],
    anti_keywords=["security", "password", "audit trail", "submit", "cancel"],
    base_confidence=0.55,
    description="System optimization — rule density, KG bottlenecks, caching, redundancy detection",
)


# ---------------------------------------------------------------------------
# NOVA data structures
# ---------------------------------------------------------------------------

@dataclass
class RuleDensityEntry:
    doctype: str
    rule_count: int
    events: list[str]
    rule_types: list[str]


@dataclass
class KGBottleneck:
    node: str
    out_degree: int
    in_degree: int
    combined: int
    module: str


@dataclass
class RedundancyFinding:
    doctype: str
    event: str
    rule_type: str
    duplicate_count: int
    example_consequence: str


@dataclass
class CacheRecommendation:
    doctype: str
    in_degree: int        # how many other DocTypes link to this one
    rule_count: int
    rationale: str
    priority: str         # HIGH | MEDIUM | LOW


@dataclass
class SystemBenchmark:
    kg_nodes: int
    kg_edges: int
    total_rules: int
    doctypes_with_rules: int
    submittable_doctypes: int
    avg_rules_per_doctype: float
    avg_edges_per_node: float
    most_connected_node: str
    most_rule_dense_doctype: str
    redundant_rule_pairs: int


# ---------------------------------------------------------------------------
# NOVA Agent
# ---------------------------------------------------------------------------

class NOVAAgent(SubAgent):
    """
    NOVA — optimization specialist.

    Analyses the SynapseOS knowledge infrastructure and produces
    actionable, data-grounded optimization recommendations.
    """

    capability = NOVA_CAPABILITY

    def __init__(
        self,
        kg_path: str = "data/graph/knowledge_graph.json",
        rules_path: str = "data/rules/rules.json",
    ):
        self._kg    = KnowledgeGraph(kg_path)
        self._rules = RulesEngine(rules_path)
        self._in_degree_cache: dict[str, int] = {}
        self._build_in_degree()

    def _build_in_degree(self) -> None:
        for name in self._kg._nodes:
            self._in_degree_cache[name] = 0
        for edge in self._kg._edges:
            tgt = edge.get("target", "")
            if tgt in self._in_degree_cache:
                self._in_degree_cache[tgt] += 1

    # ------------------------------------------------------------------
    # Analysis methods
    # ------------------------------------------------------------------

    def rule_density_report(self, top_k: int = 10) -> list[RuleDensityEntry]:
        """Find DocTypes with the highest rule counts — optimization hotspots."""
        entries: list[RuleDensityEntry] = []
        for doctype, rules in self._rules._by_doctype.items():
            if not rules:
                continue
            events = list({r.get("event", "?") for r in rules})
            rule_types = list({r.get("rule_type", "?") for r in rules})
            entries.append(RuleDensityEntry(
                doctype=doctype,
                rule_count=len(rules),
                events=sorted(events),
                rule_types=sorted(rule_types),
            ))
        entries.sort(key=lambda e: e.rule_count, reverse=True)
        return entries[:top_k]

    def kg_bottlenecks(self, top_k: int = 10) -> list[KGBottleneck]:
        """
        Find KG nodes with highest combined in+out degree.
        These are architectural bottlenecks — changes here affect the most downstream.
        """
        results: list[KGBottleneck] = []
        for name, node in self._kg._nodes.items():
            out_deg = len(self._kg._adjacency.get(name, []))
            in_deg  = self._in_degree_cache.get(name, 0)
            if out_deg + in_deg == 0:
                continue
            results.append(KGBottleneck(
                node=name,
                out_degree=out_deg,
                in_degree=in_deg,
                combined=out_deg + in_deg,
                module=node.get("module", "?"),
            ))
        results.sort(key=lambda b: b.combined, reverse=True)
        return results[:top_k]

    def cache_recommendations(self, top_k: int = 8) -> list[CacheRecommendation]:
        """
        Recommend DocTypes to cache based on high in-degree (many readers)
        and high rule density (expensive to re-evaluate each time).
        """
        recommendations: list[CacheRecommendation] = []
        for name in self._kg._nodes:
            in_deg = self._in_degree_cache.get(name, 0)
            rules  = self._rules.get_rules(name)

            if in_deg >= 5 and len(rules) >= 3:
                priority = "HIGH"   if in_deg >= 15 or len(rules) >= 10 else "MEDIUM"
                rationale = (
                    f"{in_deg} DocTypes link to this node; {len(rules)} rules evaluated on each access. "
                    f"Cache the resolved fact set for read-heavy operations."
                )
            elif in_deg >= 8:
                priority = "MEDIUM"
                rationale = (
                    f"{in_deg} DocTypes reference this node. "
                    f"Cache the KG lookup result even though rule count is low ({len(rules)})."
                )
            else:
                continue

            recommendations.append(CacheRecommendation(
                doctype=name,
                in_degree=in_deg,
                rule_count=len(rules),
                rationale=rationale,
                priority=priority,
            ))

        recommendations.sort(
            key=lambda r: ({"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(r.priority, 3), -r.in_degree)
        )
        return recommendations[:top_k]

    def detect_redundancies(self, top_k: int = 10) -> list[RedundancyFinding]:
        """
        Find rules with identical (event, rule_type, condition) within the same DocType.
        These are candidates for deduplication to reduce evaluation overhead.
        """
        findings: list[RedundancyFinding] = []
        for doctype, rules in self._rules._by_doctype.items():
            # Group by (event, rule_type, condition)
            key_counter: Counter = Counter()
            key_to_example: dict[tuple, str] = {}
            for rule in rules:
                key = (
                    rule.get("event", ""),
                    rule.get("rule_type", ""),
                    rule.get("condition", ""),
                )
                key_counter[key] += 1
                if key not in key_to_example:
                    key_to_example[key] = rule.get("consequence", "")[:80]

            for key, count in key_counter.items():
                if count >= 2:
                    event, rule_type, _ = key
                    findings.append(RedundancyFinding(
                        doctype=doctype,
                        event=event,
                        rule_type=rule_type,
                        duplicate_count=count,
                        example_consequence=key_to_example[key],
                    ))

        findings.sort(key=lambda f: f.duplicate_count, reverse=True)
        return findings[:top_k]

    def system_benchmark(self) -> SystemBenchmark:
        """Compute overall system infrastructure statistics."""
        kg_nodes = len(self._kg._nodes)
        kg_edges = len(self._kg._edges)
        total_rules = sum(len(rules) for rules in self._rules._by_doctype.values())
        doctypes_with_rules = len(self._rules._by_doctype)
        submittable = sum(
            1 for node in self._kg._nodes.values()
            if node.get("is_submittable")
        )
        avg_rules = round(total_rules / doctypes_with_rules, 2) if doctypes_with_rules else 0.0
        avg_edges = round(kg_edges / kg_nodes, 2) if kg_nodes else 0.0

        # Most connected node
        bottlenecks = self.kg_bottlenecks(top_k=1)
        most_connected = bottlenecks[0].node if bottlenecks else "?"

        # Most rule-dense
        density = self.rule_density_report(top_k=1)
        most_dense = density[0].doctype if density else "?"

        # Count redundancies
        redundancies = self.detect_redundancies(top_k=100)
        redundant_pairs = sum(f.duplicate_count - 1 for f in redundancies)

        return SystemBenchmark(
            kg_nodes=kg_nodes,
            kg_edges=kg_edges,
            total_rules=total_rules,
            doctypes_with_rules=doctypes_with_rules,
            submittable_doctypes=submittable,
            avg_rules_per_doctype=avg_rules,
            avg_edges_per_node=avg_edges,
            most_connected_node=most_connected,
            most_rule_dense_doctype=most_dense,
            redundant_rule_pairs=redundant_pairs,
        )

    # ------------------------------------------------------------------
    # SubAgent interface
    # ------------------------------------------------------------------

    def _handle(self, query: str, context: Any) -> AgentResult:
        trace: list[str] = [f"NOVA received: {query[:80]}"]
        q_lower = query.lower()

        if any(kw in q_lower for kw in ("redundan", "duplicate", "dedup")):
            findings = self.detect_redundancies(top_k=8)
            if findings:
                lines = ["Rule redundancies detected (deduplication candidates):"]
                for f in findings:
                    lines.append(
                        f"  {f.doctype} — {f.event}/{f.rule_type}: "
                        f"{f.duplicate_count} identical rules"
                    )
                    lines.append(f"    Example: {f.example_consequence}")
            else:
                lines = ["No rule redundancies detected above threshold."]
            output = "\n".join(lines)
            structured = {
                "redundancies": [
                    {"doctype": f.doctype, "event": f.event,
                     "rule_type": f.rule_type, "count": f.duplicate_count}
                    for f in findings
                ]
            }
            trace.append(f"Redundancy scan: {len(findings)} findings")
            confidence = 0.85

        elif any(kw in q_lower for kw in ("benchmark", "overview", "system", "stats", "report")):
            bm = self.system_benchmark()
            output = "\n".join([
                "=== SynapseOS System Benchmark ===",
                f"Knowledge graph  : {bm.kg_nodes} nodes, {bm.kg_edges} edges",
                f"Rules corpus     : {bm.total_rules} rules across {bm.doctypes_with_rules} DocTypes",
                f"Avg rules/DocType: {bm.avg_rules_per_doctype}",
                f"Avg edges/node   : {bm.avg_edges_per_node}",
                f"Submittable docs : {bm.submittable_doctypes}",
                f"Most connected   : {bm.most_connected_node}",
                f"Rule hotspot     : {bm.most_rule_dense_doctype}",
                f"Redundant pairs  : {bm.redundant_rule_pairs} (dedup candidates)",
            ])
            structured = {
                "kg_nodes": bm.kg_nodes, "kg_edges": bm.kg_edges,
                "total_rules": bm.total_rules, "avg_rules_per_doctype": bm.avg_rules_per_doctype,
                "most_connected_node": bm.most_connected_node,
                "most_rule_dense_doctype": bm.most_rule_dense_doctype,
                "redundant_rule_pairs": bm.redundant_rule_pairs,
            }
            trace.append(f"System benchmark: {bm.kg_nodes} nodes, {bm.total_rules} rules")
            confidence = 0.92

        elif any(kw in q_lower for kw in ("cache", "caching")):
            recs = self.cache_recommendations(top_k=6)
            lines = ["Cache recommendations (highest priority first):"]
            for r in recs:
                lines.append(f"  [{r.priority}] {r.doctype} — in_degree={r.in_degree}, rules={r.rule_count}")
                lines.append(f"    {r.rationale}")
            output = "\n".join(lines) if recs else "No cache recommendations above threshold."
            structured = {
                "recommendations": [
                    {"doctype": r.doctype, "priority": r.priority, "in_degree": r.in_degree}
                    for r in recs
                ]
            }
            trace.append(f"Cache analysis: {len(recs)} recommendations")
            confidence = 0.85

        elif any(kw in q_lower for kw in ("bottleneck", "connected", "hotspot", "connectivity")):
            bottlenecks = self.kg_bottlenecks(top_k=8)
            lines = ["KG bottleneck nodes (high combined in+out degree):"]
            for b in bottlenecks:
                lines.append(f"  {b.node}  in={b.in_degree} out={b.out_degree} combined={b.combined}  [{b.module}]")
            output = "\n".join(lines)
            structured = {
                "bottlenecks": [
                    {"node": b.node, "in": b.in_degree, "out": b.out_degree}
                    for b in bottlenecks
                ]
            }
            trace.append(f"Bottleneck analysis: {len(bottlenecks)} nodes found")
            confidence = 0.88

        elif any(kw in q_lower for kw in ("rule density", "hotspot", "most rules")):
            density = self.rule_density_report(top_k=8)
            lines = ["Rule density hotspots (DocTypes with most rules):"]
            for e in density:
                lines.append(f"  {e.doctype}: {e.rule_count} rules | events: {', '.join(e.events[:3])}")
            output = "\n".join(lines)
            structured = {
                "density": [{"doctype": e.doctype, "rules": e.rule_count} for e in density]
            }
            trace.append(f"Rule density: {len(density)} hotspots")
            confidence = 0.85

        else:
            # Default: full optimization report
            bm        = self.system_benchmark()
            density   = self.rule_density_report(top_k=5)
            bottlenecks = self.kg_bottlenecks(top_k=5)
            recs      = self.cache_recommendations(top_k=4)
            redundancies = self.detect_redundancies(top_k=4)

            lines = [
                "=== NOVA Optimization Report ===",
                "",
                f"System: {bm.kg_nodes} nodes | {bm.kg_edges} edges | {bm.total_rules} rules",
                f"Redundant rule pairs: {bm.redundant_rule_pairs}",
                "",
                "Top rule hotspots:",
            ]
            for e in density[:3]:
                lines.append(f"  {e.doctype}: {e.rule_count} rules")

            lines += ["", "Top KG bottlenecks:"]
            for b in bottlenecks[:3]:
                lines.append(f"  {b.node}: combined={b.combined}")

            if recs:
                lines += ["", "Cache recommendations:"]
                for r in recs[:3]:
                    lines.append(f"  [{r.priority}] {r.doctype} (in={r.in_degree}, rules={r.rule_count})")

            if redundancies:
                lines += ["", "Deduplication targets:"]
                for f in redundancies[:3]:
                    lines.append(f"  {f.doctype} — {f.duplicate_count}× {f.event}/{f.rule_type}")

            output = "\n".join(lines)
            structured = {
                "benchmark":      {"kg_nodes": bm.kg_nodes, "total_rules": bm.total_rules},
                "density_top3":   [e.doctype for e in density[:3]],
                "bottleneck_top3": [b.node for b in bottlenecks[:3]],
                "cache_recs":     len(recs),
                "redundancies":   len(redundancies),
            }
            trace.append("Full optimization report generated")
            confidence = 0.88

        trace.append(f"NOVA complete | confidence={confidence:.2f}")
        return AgentResult(
            agent_name="NOVA",
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
    parser = argparse.ArgumentParser(description="NOVA — Optimization Agent")
    parser.add_argument("--query", default="show me the full system optimization report")
    args = parser.parse_args()

    agent = NOVAAgent()
    print(f"\nNOVA — Optimization Specialist")
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
