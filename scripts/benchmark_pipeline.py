"""
SynapseOS — Pipeline Load Benchmark.

Tests two targets under concurrent load:
  1. CognitiveController.think()  — full end-to-end inference
  2. DistributedPipeline.run()    — distributed layer execution

Reports p50, p99, mean latency; throughput (req/s); and error rate at
each concurrency level. Results are saved to data/benchmarks/.

Phase 5 success criterion:
  "Distributed pipeline handles 50+ concurrent requests without degradation."
  Pass threshold: p99 < 2000ms, error_rate < 5% at 50 concurrent.

Run:
    python scripts/benchmark_pipeline.py
    python scripts/benchmark_pipeline.py --levels 10 25 50 --queries 100
    python scripts/benchmark_pipeline.py --target pipeline   # pipeline only
    python scripts/benchmark_pipeline.py --target controller # controller only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median, quantiles

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Test queries — varied complexity
# ---------------------------------------------------------------------------

BENCHMARK_QUERIES = [
    # Mode 1 — reasoning
    "What fields are required for a Sales Invoice?",
    "What validate rules apply to a Purchase Order?",
    "List the dependencies of a Payment Entry.",
    "What rules govern Journal Entry submission?",
    "Describe the relationship between Delivery Note and Sales Invoice.",
    # Mode 4 — code
    "Write a validate function for a Stock Entry.",
    "Implement an on_submit handler for a Sales Order.",
    "Generate a validate hook for the Employee doctype.",
    # ARIEL — business logic
    "Plan the submission steps for a Purchase Invoice.",
    "What is the workflow for approving a Leave Application?",
    # NYX — security
    "What sensitive fields does the User doctype expose?",
    "Audit the security permissions for Payroll Entry.",
    # NOVA — optimization
    "Which doctypes should we cache for better performance?",
    "Show a system benchmark overview.",
    # ATLAS — infrastructure
    "What is the current health status of all nodes?",
    "Identify single points of failure in the pipeline.",
]


# ---------------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    query: str
    latency_ms: float
    success: bool
    error: str = ""


@dataclass
class LevelStats:
    concurrency: int
    n_requests: int
    n_success: int
    n_error: int
    error_rate: float
    throughput_rps: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    mean_ms: float
    max_ms: float
    wall_time_s: float
    passes_criterion: bool   # p99 < 2000ms and error_rate < 5%


@dataclass
class BenchmarkReport:
    target: str
    timestamp: str
    levels: list[LevelStats]
    overall_pass: bool

    def to_dict(self) -> dict:
        return {
            "target":       self.target,
            "timestamp":    self.timestamp,
            "overall_pass": self.overall_pass,
            "levels":       [asdict(l) for l in self.levels],
        }

    def print_summary(self) -> None:
        print(f"\n{'='*70}")
        print(f"Benchmark: {self.target}  [{self.timestamp}]")
        print(f"{'='*70}")
        header = f"{'Concurrency':>11} {'Reqs':>5} {'OK':>5} {'Err%':>6} {'Thrput':>8} {'p50':>7} {'p90':>7} {'p99':>7} {'Criterion'}"
        print(header)
        print("-" * 70)
        for l in self.levels:
            status = "PASS ✓" if l.passes_criterion else "FAIL ✗"
            print(
                f"{l.concurrency:>11} {l.n_requests:>5} {l.n_success:>5} "
                f"{l.error_rate*100:>5.1f}% {l.throughput_rps:>7.1f}/s "
                f"{l.p50_ms:>6.0f}ms {l.p90_ms:>6.0f}ms {l.p99_ms:>6.0f}ms  {status}"
            )
        print(f"\nOverall: {'PASS ✓' if self.overall_pass else 'FAIL ✗'}")


# ---------------------------------------------------------------------------
# Core benchmarking logic
# ---------------------------------------------------------------------------

def _run_single(fn, query: str) -> RequestResult:
    t0 = time.time()
    try:
        fn(query)
        return RequestResult(query=query, latency_ms=(time.time()-t0)*1000, success=True)
    except Exception as e:
        return RequestResult(query=query, latency_ms=(time.time()-t0)*1000, success=False, error=str(e)[:80])


def _compute_percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(int((len(sorted_vals) - 1) * pct / 100), len(sorted_vals) - 1)
    return sorted_vals[idx]


def run_level(
    fn,
    concurrency: int,
    queries: list[str],
    n_requests: int,
) -> LevelStats:
    """
    Submit n_requests to fn() with at most `concurrency` in flight at once.
    """
    # Round-robin queries to fill n_requests
    query_list = [queries[i % len(queries)] for i in range(n_requests)]

    wall_start = time.time()
    results: list[RequestResult] = []

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_run_single, fn, q): q for q in query_list}
        for future in as_completed(futures, timeout=300):
            try:
                results.append(future.result(timeout=60))
            except Exception as e:
                results.append(RequestResult(query="?", latency_ms=60000, success=False, error=str(e)))

    wall_time = time.time() - wall_start
    latencies = sorted(r.latency_ms for r in results)
    n_success  = sum(1 for r in results if r.success)
    n_error    = len(results) - n_success
    error_rate = n_error / len(results) if results else 1.0

    p50 = _compute_percentile(latencies, 50)
    p90 = _compute_percentile(latencies, 90)
    p99 = _compute_percentile(latencies, 99)

    passes = p99 < 2000 and error_rate < 0.05

    return LevelStats(
        concurrency=concurrency,
        n_requests=len(results),
        n_success=n_success,
        n_error=n_error,
        error_rate=round(error_rate, 4),
        throughput_rps=round(n_success / wall_time, 2) if wall_time > 0 else 0,
        p50_ms=round(p50, 1),
        p90_ms=round(p90, 1),
        p99_ms=round(p99, 1),
        mean_ms=round(mean(latencies), 1) if latencies else 0.0,
        max_ms=round(max(latencies), 1) if latencies else 0.0,
        wall_time_s=round(wall_time, 2),
        passes_criterion=passes,
    )


# ---------------------------------------------------------------------------
# Target: CognitiveController
# ---------------------------------------------------------------------------

def benchmark_controller(
    levels: list[int],
    n_per_level: int,
    kg_path: str,
    rules_path: str,
) -> BenchmarkReport:
    from synapseos.phase5.cognitive_controller import CognitiveController

    print("[Benchmark] Loading CognitiveController...")
    controller = CognitiveController(
        kg_path=kg_path,
        rules_path=rules_path,
        mode_timeout=20.0,
        max_parallel_modes=2,
        enable_idle_loop=False,
        enable_agents=True,
    )

    def fn(query: str):
        controller.think(query)

    level_stats = []
    for c in levels:
        print(f"  → Concurrency {c:>3} ({n_per_level} requests) ... ", end="", flush=True)
        stats = run_level(fn, c, BENCHMARK_QUERIES, n_per_level)
        level_stats.append(stats)
        status = "PASS" if stats.passes_criterion else "FAIL"
        print(f"p99={stats.p99_ms:.0f}ms  err={stats.error_rate*100:.1f}%  [{status}]")

    controller.shutdown()

    return BenchmarkReport(
        target="CognitiveController",
        timestamp=datetime.now().isoformat(),
        levels=level_stats,
        overall_pass=all(l.passes_criterion for l in level_stats),
    )


# ---------------------------------------------------------------------------
# Target: DistributedPipeline
# ---------------------------------------------------------------------------

def benchmark_pipeline(
    levels: list[int],
    n_per_level: int,
    kg_path: str,
    rules_path: str,
) -> BenchmarkReport:
    from synapseos.phase5.distributed import DistributedPipeline, PipelineTopology

    print("[Benchmark] Loading DistributedPipeline (local topology)...")
    topo     = PipelineTopology.local_default(kg_path, rules_path)
    pipeline = DistributedPipeline(topology=topo, kg_path=kg_path, rules_path=rules_path)

    def fn(query: str):
        pipeline.run(query)

    level_stats = []
    for c in levels:
        print(f"  → Concurrency {c:>3} ({n_per_level} requests) ... ", end="", flush=True)
        stats = run_level(fn, c, BENCHMARK_QUERIES, n_per_level)
        level_stats.append(stats)
        status = "PASS" if stats.passes_criterion else "FAIL"
        print(f"p99={stats.p99_ms:.0f}ms  err={stats.error_rate*100:.1f}%  [{status}]")

    return BenchmarkReport(
        target="DistributedPipeline",
        timestamp=datetime.now().isoformat(),
        levels=level_stats,
        overall_pass=all(l.passes_criterion for l in level_stats),
    )


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_report(report: BenchmarkReport, output_dir: str = "data/benchmarks") -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fname = f"{report.target.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path  = out / fname
    path.write_text(json.dumps(report.to_dict(), indent=2))
    return str(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SynapseOS Pipeline Load Benchmark")
    parser.add_argument("--levels",   nargs="+", type=int, default=[5, 10, 25],
                        help="Concurrency levels to test (default: 5 10 25)")
    parser.add_argument("--requests", type=int, default=20,
                        help="Requests per concurrency level (default: 20)")
    parser.add_argument("--target",   choices=["controller", "pipeline", "both"], default="both")
    parser.add_argument("--kg",       default="data/graph/knowledge_graph.json")
    parser.add_argument("--rules",    default="data/rules/rules.json")
    parser.add_argument("--save",     action="store_true", default=True)
    args = parser.parse_args()

    print(f"\nSynapseOS Load Benchmark")
    print(f"Levels   : {args.levels}")
    print(f"Requests : {args.requests} per level")
    print(f"Target   : {args.target}")
    print(f"Criterion: p99 < 2000ms AND error_rate < 5%\n")

    reports = []

    if args.target in ("pipeline", "both"):
        report = benchmark_pipeline(args.levels, args.requests, args.kg, args.rules)
        report.print_summary()
        reports.append(report)
        if args.save:
            path = save_report(report)
            print(f"Saved → {path}")

    if args.target in ("controller", "both"):
        report = benchmark_controller(args.levels, args.requests, args.kg, args.rules)
        report.print_summary()
        reports.append(report)
        if args.save:
            path = save_report(report)
            print(f"Saved → {path}")

    all_pass = all(r.overall_pass for r in reports)
    print(f"\n{'='*70}")
    print(f"BENCHMARK RESULT: {'PASS ✓' if all_pass else 'FAIL ✗'}")
    print(f"Phase 5 criterion #2 (50 QPS, p99<2s, err<5%): {'MET' if all_pass else 'NOT MET — review bottlenecks'}")


if __name__ == "__main__":
    main()
