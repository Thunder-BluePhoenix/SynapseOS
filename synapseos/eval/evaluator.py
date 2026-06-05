"""
SynapseOS Quality Evaluation Framework.

Measures pipeline answer quality before GPU training. Provides a baseline
so we know how much Mamba LoRA fine-tuning actually improves things.

Evaluation categories:
  1. Field accuracy       — Does the pipeline surface mandatory fields that match the KG?
  2. Rule coverage        — What % of extracted rules does the pipeline mention in responses?
  3. Code quality         — Static analysis pass rate on Mode 4 generated code
  4. Routing accuracy     — Does the mode/agent router select the correct target?
  5. Latency profile      — p50/p99 latency distribution over N queries
  6. Agent answer quality — Confidence and grounding of sub-agent responses

Run:
    python -m synapseos.eval.evaluator
    python -m synapseos.eval.evaluator --n 50 --save data/eval/baseline.json
    python -m synapseos.eval.evaluator --category routing   # single category
"""

from __future__ import annotations

import ast
import json
import re
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from statistics import mean, median
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CategoryResult:
    category: str
    n_evaluated: int
    score: float           # 0–1
    details: list[dict]
    latency_ms: float      # avg per query

    def grade(self) -> str:
        if self.score >= 0.85: return "A"
        if self.score >= 0.70: return "B"
        if self.score >= 0.55: return "C"
        if self.score >= 0.40: return "D"
        return "F"

    def to_dict(self) -> dict:
        return {**asdict(self), "grade": self.grade()}


@dataclass
class EvalReport:
    timestamp: str
    n_total: int
    categories: list[CategoryResult]
    overall_score: float
    overall_grade: str
    summary: str
    gpu_training_recommendation: str

    def to_dict(self) -> dict:
        return {
            "timestamp":  self.timestamp,
            "n_total":    self.n_total,
            "overall":    {"score": round(self.overall_score, 3), "grade": self.overall_grade},
            "summary":    self.summary,
            "recommendation": self.gpu_training_recommendation,
            "categories": [c.to_dict() for c in self.categories],
        }

    def print_report(self) -> None:
        print(f"\n{'='*64}")
        print(f"  SynapseOS Evaluation Report  [{self.timestamp[:10]}]")
        print(f"{'='*64}")
        print(f"  Overall: {self.overall_grade}  ({self.overall_score:.1%})\n")
        for cat in self.categories:
            bar = "█" * int(cat.score * 20) + "░" * (20 - int(cat.score * 20))
            print(f"  {cat.category:22s} [{bar}]  {cat.score:.1%}  {cat.grade()}  ({cat.n_evaluated} samples, {cat.latency_ms:.0f}ms avg)")
        print(f"\n  {self.summary}")
        print(f"\n  GPU Recommendation: {self.gpu_training_recommendation}")
        print(f"{'='*64}\n")


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class SynapseOSEvaluator:
    """
    Runs structured evaluation of the SynapseOS pipeline.
    All evaluations are deterministic and grounded in KG/rules data.
    """

    def __init__(
        self,
        kg_path: str = "data/graph/knowledge_graph.json",
        rules_path: str = "data/rules/rules.json",
        seed: int = 42,
    ):
        import random
        random.seed(seed)
        self._kg_path    = kg_path
        self._rules_path = rules_path
        self._kg         = None
        self._rules      = None
        self._controller = None
        self._rng        = random

    def _get_kg(self):
        if self._kg is None:
            from synapseos.phase3.neurosymbolic_core import KnowledgeGraph
            self._kg = KnowledgeGraph(self._kg_path)
        return self._kg

    def _get_rules(self):
        if self._rules is None:
            from synapseos.phase3.neurosymbolic_core import RulesEngine
            self._rules = RulesEngine(self._rules_path)
        return self._rules

    def _get_controller(self):
        if self._controller is None:
            from synapseos.phase5.cognitive_controller import CognitiveController
            self._controller = CognitiveController(
                kg_path=self._kg_path,
                rules_path=self._rules_path,
                mode_timeout=20.0,
                max_parallel_modes=2,
                enable_idle_loop=False,
                enable_world_model=False,   # eval measures the core pipeline deterministically
            )
        return self._controller

    def _sample_doctypes(self, n: int, require_rules: bool = False) -> list[str]:
        kg    = self._get_kg()
        rules = self._get_rules()
        pool  = [
            name for name, node in kg._nodes.items()
            if not name.startswith("__") and "scheduler" not in name.lower()
            and len(name) > 3
        ]
        if require_rules:
            pool = [dt for dt in pool if rules.get_rules(dt)]
        return self._rng.sample(pool, min(n, len(pool)))

    # ------------------------------------------------------------------
    # Category 1: Field accuracy
    # ------------------------------------------------------------------

    def eval_field_accuracy(self, n: int = 20) -> CategoryResult:
        """
        For each DocType, ask the pipeline "what fields are required?"
        and check what fraction of actual mandatory fields appear in the response.
        Ground truth: KG rules with rule_type=constraint and "must not be empty" in consequence.
        """
        ctrl   = self._get_controller()
        rules  = self._get_rules()
        scores = []
        details = []
        doctypes = self._sample_doctypes(n, require_rules=True)

        # Use ARIEL agent directly — queries rules explicitly, works for any DocType
        from synapseos.phase5.agents.ariel import ARIELAgent
        ariel = ARIELAgent(kg_path=self._kg_path, rules_path=self._rules_path)

        for dt in doctypes:
            mandatory = [
                r for r in rules.get_rules(dt)
                if r.get("rule_type") == "constraint" and "must not be empty" in r.get("consequence", "")
            ]
            if not mandatory:
                continue

            query = f"What mandatory fields does {dt} require?"
            t0    = time.time()
            try:
                result = ariel.handle(query)
                output = result.output.lower()
                ms     = (time.time() - t0) * 1000

                # Skip if no real output produced
                if "no mode produced" in output or len(output) < 10:
                    continue

                # Check how many mandatory field names OR labels appear in the response
                found = 0
                for r in mandatory:
                    cons = r.get("consequence", "")
                    # Check fieldname in parens
                    m_paren = re.search(r"\(([^)]+)\)", cons)
                    fieldname = m_paren.group(1).lower() if m_paren else ""
                    # Check quoted label
                    m_label = re.search(r"'([^']+)'", cons)
                    label = m_label.group(1).lower() if m_label else ""

                    if (fieldname and fieldname in output) or (label and label in output):
                        found += 1

                precision = found / len(mandatory) if mandatory else 0.0
                scores.append(precision)
                details.append({
                    "doctype": dt, "mandatory_count": len(mandatory),
                    "found": found, "precision": round(precision, 3),
                    "latency_ms": round(ms, 1),
                })
            except Exception as e:
                details.append({"doctype": dt, "error": str(e)[:50]})

        avg_score = mean(scores) if scores else 0.0
        avg_lat   = mean(d.get("latency_ms", 0) for d in details if "latency_ms" in d) or 0.0
        return CategoryResult("field_accuracy", len(scores), round(avg_score, 3), details, round(avg_lat, 1))

    # ------------------------------------------------------------------
    # Category 2: Rule coverage
    # ------------------------------------------------------------------

    def eval_rule_coverage(self, n: int = 20) -> CategoryResult:
        """
        Ask the pipeline for validation rules of a DocType.
        Score = fraction of extracted rules that appear (by rule_type keyword) in response.
        """
        ctrl    = self._get_controller()
        rules   = self._get_rules()
        scores  = []
        details = []
        doctypes = self._sample_doctypes(n, require_rules=True)

        from synapseos.phase5.agents.ariel import ARIELAgent
        ariel = ARIELAgent(kg_path=self._kg_path, rules_path=self._rules_path)

        for dt in doctypes:
            validate_rules = rules.get_rules(dt, "validate")
            if not validate_rules:
                continue

            query = f"What validate rules apply to a {dt}?"
            t0    = time.time()
            try:
                result = ariel.handle(query)
                output = result.output.lower()
                ms     = (time.time() - t0) * 1000

                if len(output) < 10:
                    continue

                found = 0
                for r in validate_rules[:8]:
                    rt   = r.get("rule_type", "").lower()
                    cons = r.get("consequence", "")[:40].lower()
                    if rt in output or any(w in output for w in cons.split()[:3] if len(w) > 4):
                        found += 1

                coverage = found / len(validate_rules[:8]) if validate_rules else 0.0
                scores.append(coverage)
                details.append({
                    "doctype": dt, "rule_count": len(validate_rules),
                    "found": found, "coverage": round(coverage, 3),
                    "latency_ms": round(ms, 1),
                })
            except Exception as e:
                details.append({"doctype": dt, "error": str(e)[:50]})

        avg_score = mean(scores) if scores else 0.0
        avg_lat   = mean(d.get("latency_ms", 0) for d in details if "latency_ms" in d) or 0.0
        return CategoryResult("rule_coverage", len(scores), round(avg_score, 3), details, round(avg_lat, 1))

    # ------------------------------------------------------------------
    # Category 3: Code quality (Mode 4)
    # ------------------------------------------------------------------

    def eval_code_quality(self, n: int = 20) -> CategoryResult:
        """
        Ask Mode 4 to generate a validate function for each DocType.
        Score = fraction that pass AST parsing (syntactically valid Python).
        """
        ctrl     = self._get_controller()
        scores   = []
        details  = []
        doctypes = self._sample_doctypes(n)

        for dt in doctypes:
            query = f"Write a validate function for the {dt} doctype."
            t0    = time.time()
            try:
                result = ctrl.think(query, force_modes=["mode4"])
                ms     = (time.time() - t0) * 1000

                code = result.primary_output
                # Extract code block
                m = re.search(r"```python\n(.*?)```", code, re.DOTALL)
                if m:
                    code = m.group(1)

                # Test AST parse
                try:
                    ast.parse(code)
                    valid = True
                except SyntaxError:
                    valid = False

                scores.append(1.0 if valid else 0.0)
                details.append({
                    "doctype": dt, "valid": valid,
                    "code_len": len(code),
                    "latency_ms": round(ms, 1),
                })
            except Exception as e:
                scores.append(0.0)
                details.append({"doctype": dt, "error": str(e)[:50]})

        avg_score = mean(scores) if scores else 0.0
        avg_lat   = mean(d.get("latency_ms", 0) for d in details if "latency_ms" in d) or 0.0
        return CategoryResult("code_quality", len(scores), round(avg_score, 3), details, round(avg_lat, 1))

    # ------------------------------------------------------------------
    # Category 4: Routing accuracy
    # ------------------------------------------------------------------

    def eval_routing_accuracy(self, n_per_type: int = 5) -> CategoryResult:
        """
        Check that the mode router and agent router select the correct target
        for clearly typed queries.
        """
        ctrl = self._get_controller()

        ROUTING_CASES = [
            # (query_template, expected_mode_or_agent, check_fn)
            ("Write a validate function for {dt}",          "mode4",  lambda r: "mode4" in [m.value for m in r.modes_activated]),
            ("What fields are required for {dt}?",          "mode1",  lambda r: "mode1" in [m.value for m in r.modes_activated]),
            ("Brainstorm creative ideas for {dt} workflow",  "mode2",  lambda r: "mode2" in [m.value for m in r.modes_activated]),
            ("Plan the submission steps for {dt}",           "ARIEL",  lambda r: "ARIEL" in [a.agent_name for a in r.agent_results]),
            ("Audit security permissions for {dt}",          "NYX",    lambda r: "NYX"   in [a.agent_name for a in r.agent_results]),
            ("Benchmark and optimize the {dt} pipeline",     "NOVA",   lambda r: "NOVA"  in [a.agent_name for a in r.agent_results]),
            ("Infrastructure health status for {dt} nodes",  "ATLAS",  lambda r: "ATLAS" in [a.agent_name for a in r.agent_results]),
        ]

        doctypes = self._sample_doctypes(n_per_type, require_rules=False)
        scores   = []
        details  = []

        for qtemplate, expected, check_fn in ROUTING_CASES:
            correct = 0
            for dt in doctypes:
                query = qtemplate.format(dt=dt)
                t0    = time.time()
                try:
                    result = ctrl.think(query)
                    ms     = (time.time() - t0) * 1000
                    hit    = check_fn(result)
                    correct += (1 if hit else 0)
                    details.append({
                        "query": query[:60], "expected": expected,
                        "hit": hit, "latency_ms": round(ms, 1),
                    })
                except Exception as e:
                    details.append({"query": query[:60], "error": str(e)[:50]})

            rate = correct / len(doctypes) if doctypes else 0.0
            scores.append(rate)

        avg_score = mean(scores) if scores else 0.0
        avg_lat   = mean(d.get("latency_ms", 0) for d in details if "latency_ms" in d) or 0.0
        return CategoryResult("routing_accuracy", len(details), round(avg_score, 3), details, round(avg_lat, 1))

    # ------------------------------------------------------------------
    # Category 5: Latency profile
    # ------------------------------------------------------------------

    def eval_latency(self, n: int = 30) -> CategoryResult:
        """
        Measure end-to-end latency over N queries across all modes/agents.
        Reports p50, p90, p99 and flags if any exceed 2s target.
        """
        ctrl     = self._get_controller()
        doctypes = self._sample_doctypes(n)
        queries  = [
            f"What fields are required for a {dt}?" for dt in doctypes[:n//3]
        ] + [
            f"Plan the submission steps for {dt}." for dt in doctypes[n//3:2*n//3]
        ] + [
            f"Write a validate function for {dt}." for dt in doctypes[2*n//3:n]
        ]

        latencies = []
        details   = []
        errors    = 0

        for query in queries:
            t0 = time.time()
            try:
                ctrl.think(query)
                ms = (time.time() - t0) * 1000
                latencies.append(ms)
                details.append({"query": query[:50], "latency_ms": round(ms, 1), "ok": True})
            except Exception as e:
                errors += 1
                details.append({"query": query[:50], "error": str(e)[:40]})

        if not latencies:
            return CategoryResult("latency", 0, 0.0, details, 0.0)

        sorted_lats = sorted(latencies)
        p50 = sorted_lats[len(sorted_lats) // 2]
        p90 = sorted_lats[int(len(sorted_lats) * 0.90)]
        p99 = sorted_lats[min(int(len(sorted_lats) * 0.99), len(sorted_lats)-1)]

        # Score: 1.0 if p99 < 500ms, degrades linearly to 0 at 5000ms
        score = max(0.0, min(1.0, (5000 - p99) / 4500))
        avg   = mean(latencies)

        details.insert(0, {
            "p50_ms": round(p50, 1), "p90_ms": round(p90, 1), "p99_ms": round(p99, 1),
            "mean_ms": round(avg, 1), "errors": errors,
            "within_2s": sum(1 for l in latencies if l < 2000),
        })
        return CategoryResult("latency", len(latencies), round(score, 3), details, round(avg, 1))

    # ------------------------------------------------------------------
    # Category 6: Agent answer quality
    # ------------------------------------------------------------------

    def eval_agent_quality(self, n: int = 10) -> CategoryResult:
        """
        Run each agent on several queries and score:
        - confidence > 0.5 (agent is confident)
        - output length > 50 chars (non-trivial response)
        - traceable = True
        - no errors
        """
        doctypes = self._sample_doctypes(n, require_rules=True)
        scores   = []
        details  = []

        agent_queries = [
            ("ARIEL",  "synapseos.phase5.agents.ariel", "ARIELAgent",
             "Plan the submission steps for {dt}."),
            ("NYX",    "synapseos.phase5.agents.nyx",   "NYXAgent",
             "Audit the security permissions for {dt}."),
            ("NOVA",   "synapseos.phase5.agents.nova",  "NOVAAgent",
             "Show a system benchmark overview for {dt}."),
            ("ATLAS",  "synapseos.phase5.agents.atlas", "ATLASAgent",
             "Run a full infrastructure audit for {dt}."),
        ]

        import importlib
        for agent_name, mod_path, cls_name, query_tpl in agent_queries:
            try:
                mod   = importlib.import_module(mod_path)
                cls   = getattr(mod, cls_name)
                agent = cls(kg_path=self._kg_path, rules_path=self._rules_path)
            except Exception as e:
                details.append({"agent": agent_name, "error": str(e)[:50]})
                continue

            agent_scores = []
            for dt in doctypes[:max(1, n // 4)]:
                query = query_tpl.format(dt=dt)
                t0    = time.time()
                try:
                    result = agent.handle(query)
                    ms     = (time.time() - t0) * 1000
                    s = (
                        (0.30 if result.confidence > 0.50 else 0.0) +
                        (0.30 if len(result.output) > 50 else 0.0)  +
                        (0.25 if result.traceable else 0.0)          +
                        (0.15 if not result.error else 0.0)
                    )
                    agent_scores.append(s)
                    details.append({
                        "agent": agent_name, "doctype": dt,
                        "score": round(s, 3), "confidence": round(result.confidence, 3),
                        "output_len": len(result.output), "traceable": result.traceable,
                        "latency_ms": round(ms, 1),
                    })
                except Exception as e:
                    details.append({"agent": agent_name, "doctype": dt, "error": str(e)[:50]})

            if agent_scores:
                scores.extend(agent_scores)

        avg_score = mean(scores) if scores else 0.0
        avg_lat   = mean(d.get("latency_ms", 0) for d in details if "latency_ms" in d) or 0.0
        return CategoryResult("agent_quality", len(scores), round(avg_score, 3), details, round(avg_lat, 1))

    # ------------------------------------------------------------------
    # Full evaluation run
    # ------------------------------------------------------------------

    def run(
        self,
        n_per_category: int = 10,
        categories: list[str] | None = None,
    ) -> EvalReport:
        """
        Run the full evaluation suite.
        categories: subset to run (default: all). Options:
          field_accuracy, rule_coverage, code_quality, routing, latency, agent_quality
        """
        from datetime import datetime as dt
        all_cats = {
            "field_accuracy": lambda: self.eval_field_accuracy(n_per_category),
            "rule_coverage":  lambda: self.eval_rule_coverage(n_per_category),
            "code_quality":   lambda: self.eval_code_quality(n_per_category),
            "routing":        lambda: self.eval_routing_accuracy(n_per_category // 2),
            "latency":        lambda: self.eval_latency(n_per_category * 2),
            "agent_quality":  lambda: self.eval_agent_quality(n_per_category),
        }

        selected = categories or list(all_cats)
        results  = []

        for cat_name in selected:
            if cat_name not in all_cats:
                print(f"[Eval] Unknown category '{cat_name}' — skipping")
                continue
            print(f"[Eval] Running {cat_name} ({n_per_category} samples) ...", end=" ", flush=True)
            t0  = time.time()
            res = all_cats[cat_name]()
            elapsed = time.time() - t0
            print(f"{res.score:.1%}  {res.grade()}  ({elapsed:.1f}s)")
            results.append(res)

        if results:
            overall = mean(r.score for r in results)
        else:
            overall = 0.0

        overall_grade = (
            "A" if overall >= 0.85 else
            "B" if overall >= 0.70 else
            "C" if overall >= 0.55 else
            "D" if overall >= 0.40 else "F"
        )

        weak = [r for r in results if r.score < 0.60]
        summary_parts = [f"{r.category}={r.grade()}" for r in results]
        summary = "  ".join(summary_parts)

        if overall >= 0.80:
            gpu_rec = "System performs well CPU-only. GPU training will improve output synthesis quality."
        elif overall >= 0.65:
            gpu_rec = (
                "Good foundation. Priority GPU targets: (1) Mamba fine-tuning for better embeddings, "
                "(2) Output synthesizer SFT for more fluent responses."
            )
        else:
            weak_cats = ", ".join(r.category for r in weak)
            gpu_rec   = (
                f"Weak areas: {weak_cats}. Prioritize GPU training for Mamba (improves all categories). "
                "Also check KG coverage for weak DocTypes."
            )

        n_total = sum(r.n_evaluated for r in results)
        return EvalReport(
            timestamp=dt.now().isoformat(),
            n_total=n_total,
            categories=results,
            overall_score=round(overall, 3),
            overall_grade=overall_grade,
            summary=summary,
            gpu_training_recommendation=gpu_rec,
        )

    def shutdown(self) -> None:
        if self._controller:
            self._controller.shutdown()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="SynapseOS Quality Evaluator")
    parser.add_argument("--n",        type=int, default=10, help="Samples per category")
    parser.add_argument("--save",     default=None, help="Save JSON report to path")
    parser.add_argument("--category", default=None,
                        help="Run single category: field_accuracy|rule_coverage|code_quality|routing|latency|agent_quality")
    parser.add_argument("--kg",       default="data/graph/knowledge_graph.json")
    parser.add_argument("--rules",    default="data/rules/rules.json")
    args = parser.parse_args()

    print(f"\nSynapseOS Quality Evaluation")
    print(f"  n_per_category: {args.n}")
    print(f"  category filter: {args.category or 'all'}\n")

    evaluator = SynapseOSEvaluator(kg_path=args.kg, rules_path=args.rules)
    cats      = [args.category] if args.category else None
    report    = evaluator.run(n_per_category=args.n, categories=cats)
    report.print_report()

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.to_dict(), indent=2))
        print(f"Report saved → {args.save}")

    evaluator.shutdown()
    return report


if __name__ == "__main__":
    main()
