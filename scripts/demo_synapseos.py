"""
SynapseOS — Integrated System Demo.

Demonstrates the full SynapseOS stack running end-to-end.
Each section can be run independently with --section.

Sections:
  1. modes      — All 5 thinking modes via CognitiveController
  2. agents     — ARIEL / NYX / NOVA / ATLAS sub-agents
  3. creativity — Cross-domain analogy with expanded KG
  4. voice      — Voice pipeline round-trip (stub mode)
  5. drift      — Liquid NN drift monitor
  6. pipeline   — Distributed pipeline topology
  7. memory     — Episodic + semantic memory retrieval
  all           — Run all sections

Run:
    python scripts/demo_synapseos.py
    python scripts/demo_synapseos.py --section agents
    python scripts/demo_synapseos.py --section creativity --expanded-kg
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

SEPARATOR = "=" * 64
SUBSEP    = "-" * 48


def _header(title: str, icon: str = "◆") -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {icon}  {title}")
    print(SEPARATOR)


def _sub(label: str) -> None:
    print(f"\n{SUBSEP}")
    print(f"  {label}")
    print(SUBSEP)


def _result(label: str, value: str, width: int = 18) -> None:
    print(f"  {label:<{width}}: {value}")


# ---------------------------------------------------------------------------
# Section 1 — Thinking Modes
# ---------------------------------------------------------------------------

def demo_modes(kg: str, rules: str) -> None:
    _header("Thinking Modes (1–5)", "🧠")
    from synapseos.phase5.cognitive_controller import CognitiveController, ThinkingMode

    controller = CognitiveController(kg_path=kg, rules_path=rules,
                                     enable_idle_loop=False, enable_agents=False)

    demos = [
        ("mode1", ThinkingMode.MODE1_REASONING,
         "What validate rules apply to a Sales Invoice?"),
        ("mode2", ThinkingMode.MODE2_CREATIVE,
         "Generate creative ideas for automating HR onboarding"),
        ("mode3", ThinkingMode.MODE3_SYNTHESIS,
         "Find an analogy between inventory management and network routing"),
        ("mode4", ThinkingMode.MODE4_CODE,
         "Write a validate function for the Purchase Order doctype"),
    ]

    for label, mode, query in demos:
        _sub(f"Mode {label[-1]} — {mode.value}")
        print(f"  Query: {query}")
        t0     = time.time()
        result = controller.think(query, force_modes=[mode.value])
        ms     = (time.time() - t0) * 1000
        _result("Latency",    f"{ms:.0f}ms")
        _result("Output type", result.output_type)
        _result("Confidence", f"{result.confidence:.3f}")
        print(f"\n  Output (first 300 chars):")
        print(f"  {result.primary_output[:300].replace(chr(10), chr(10) + '  ')}")

    _sub("Mode 5 — Idle Exploration (30-second background run)")
    print("  Starting idle loop daemon for 30s ...")
    from synapseos.phase4.idle_loop import IdleLoop
    loop = IdleLoop(graph_path=kg, idle_threshold=0.0, tick_interval=1.0)
    loop.start_background()
    time.sleep(30)
    loop.stop()
    report = loop.get_session_report()
    _result("Walks completed",    str(report.walks_completed))
    _result("Hypotheses new",     str(report.hypotheses_generated))
    _result("Hypotheses reinforced", str(report.hypotheses_reinforced))
    _result("Domains explored",   str(len(report.domains_explored)))
    if report.top_hypotheses:
        print(f"  Top hypothesis: [{report.top_hypotheses[0]['confidence']:.2f}] "
              f"{report.top_hypotheses[0]['description'][:80]}")

    controller.shutdown()
    print(f"\n  Mode demo complete.")


# ---------------------------------------------------------------------------
# Section 2 — Sub-agents
# ---------------------------------------------------------------------------

def demo_agents(kg: str, rules: str) -> None:
    _header("Sub-Agents: ARIEL / NYX / NOVA / ATLAS", "🤖")

    agent_demos = [
        ("ARIEL",  "synapseos.phase5.agents.ariel", "ARIELAgent",
         "plan the submission steps for a Sales Invoice"),
        ("NYX",    "synapseos.phase5.agents.nyx",   "NYXAgent",
         "generate an audit trail specification for Sales Invoice submit"),
        ("NOVA",   "synapseos.phase5.agents.nova",  "NOVAAgent",
         "which doctypes should we cache and why?"),
        ("ATLAS",  "synapseos.phase5.agents.atlas", "ATLASAgent",
         "run a full infrastructure audit and identify fault risks"),
    ]

    for name, module_path, cls_name, query in agent_demos:
        _sub(f"{name} — {query[:50]}")
        import importlib
        mod   = importlib.import_module(module_path)
        cls   = getattr(mod, cls_name)
        agent = cls(kg_path=kg, rules_path=rules)

        score = agent.can_handle(query)
        _result("Routing confidence", f"{score:.3f}")

        t0     = time.time()
        result = agent.handle(query)
        ms     = (time.time() - t0) * 1000

        _result("Latency",    f"{ms:.0f}ms")
        _result("Confidence", f"{result.confidence:.3f}")
        _result("Traceable",  str(result.traceable))
        _result("Trace steps", str(len(result.trace)))
        print(f"\n  Output (first 400 chars):")
        print(f"  {result.output[:400].replace(chr(10), chr(10) + '  ')}")

    print(f"\n  Sub-agent demo complete.")


# ---------------------------------------------------------------------------
# Section 3 — Cross-domain creativity with expanded KG
# ---------------------------------------------------------------------------

def demo_creativity(kg: str, expanded_kg: str | None = None) -> None:
    _header("Creativity Engine — Cross-Domain Analogy", "💡")

    graph_path = expanded_kg if (expanded_kg and Path(expanded_kg).exists()) else kg
    using_expanded = graph_path != kg

    print(f"  Using {'expanded KG (ERP + CS + Biology + Physics + Economics)' if using_expanded else 'base ERP KG'}")
    print(f"  Graph: {graph_path}")

    from synapseos.phase4.creativity_engine import CreativityEngine

    engine = CreativityEngine(graph_path=graph_path)
    seeds  = [
        ("Sales Invoice",        "ERP → ?"),
        ("Workflow",             "ERP process → ?"),
        ("Item",                 "inventory → ?"),
        ("Transaction" if using_expanded else "Payment Entry", "financial atomicity → ?"),
    ]

    for seed, label in seeds:
        _sub(f"Seed: \"{seed}\"  [{label}]")
        ideas = engine.generate(seed=seed, apply_debate=True, top_k=2)
        for i, idea in enumerate(ideas, 1):
            _result(f"Idea {i} type",   idea.idea_type)
            _result(f"  Concepts",      ", ".join(idea.source_concepts[:3]))
            _result(f"  Score",         f"novelty={idea.novelty_score:.2f} conf={idea.confidence_score:.2f} combined={idea.combined_score:.2f}")
            print(f"    Title: {idea.title}")
            print(f"    Desc : {idea.description[:180].replace(chr(10), chr(10) + '    ')}")

    if using_expanded:
        _sub("Structural bridges (cross-domain analogies)")
        bridge_pairs = [
            ("Transaction",     "Journal Entry"),
            ("Homeostasis",     "Budget"),
            ("NaturalSelection","Supplier Scorecard"),
            ("StateMachine",    "Workflow"),
            ("Market",          "Pricelist"),
        ]
        for cross, erp in bridge_pairs:
            cross_node = engine.concepts.get(cross)
            erp_node   = engine.concepts.get(erp)
            if cross_node and erp_node:
                analogies = engine.find_analogies(cross, top_k=3)
                erp_match = next((a for a in analogies if a.target_concept == erp), None)
                if erp_match:
                    print(f"  {cross:20} ↔ {erp:22} score={erp_match.confidence:.2f}: {erp_match.shared_structure[:50]}")
                else:
                    print(f"  {cross:20} ↔ {erp:22} [bridge edge present, structural score below threshold]")

    print(f"\n  Creativity demo complete.")


# ---------------------------------------------------------------------------
# Section 4 — Voice pipeline
# ---------------------------------------------------------------------------

def demo_voice(kg: str, rules: str) -> None:
    _header("Voice Pipeline (Stub Mode)", "🎙")
    from synapseos.phase5.voice import VoicePipeline

    pipeline = VoicePipeline(kg_path=kg, rules_path=rules)
    _result("Whisper mode", "real" if pipeline.transcriber.is_real else "stub")
    _result("TTS mode",     "pyttsx3" if pipeline.synthesizer.is_real else "stub")
    _result("VAD mode",     "real (webrtcvad)" if pipeline.vad.is_real else "stub")

    test_queries = [
        "What validate rules apply to a Sales Invoice?",
        "Plan the submission steps for a Purchase Order.",
    ]

    for query in test_queries:
        _sub(f"Query: \"{query[:60]}\"")
        result = pipeline.process_text(query)
        _result("Total latency",    f"{result.total_latency_ms:.0f}ms")
        _result("Within 2s target", str(result.within_target))
        _result("Transcription",    result.transcription.text[:50])
        _result("Synthesis path",   result.synthesis.audio_path)
        print(f"  Cognition: {result.cognition_output[:200].replace(chr(10), chr(10) + '  ')}")

    stats = pipeline.session_stats()
    _sub("Session statistics")
    _result("Sessions",       str(stats["sessions"]))
    _result("Within target",  f"{stats['within_target']}/{stats['sessions']}")
    _result("Mean latency",   f"{stats['mean_latency_ms']:.0f}ms")
    _result("Target rate",    f"{stats['target_rate']*100:.0f}%")

    pipeline.shutdown()
    print(f"\n  Voice demo complete.")


# ---------------------------------------------------------------------------
# Section 5 — Drift monitor
# ---------------------------------------------------------------------------

def demo_drift() -> None:
    _header("Liquid Layer Drift Monitor", "📊")
    import torch
    import torch.nn as nn
    from synapseos.phase5.drift_monitor import DriftMonitor

    monitor = DriftMonitor(
        checkpoint_dir="data/checkpoints/demo_drift",
        log_path="data/drift/demo_log.json",
        drift_threshold=0.10,
        rollback_threshold=0.25,
    )

    model = nn.Sequential(nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 8))
    _sub("Capturing baseline snapshot")
    baseline_id = monitor.capture(model, label="demo_baseline")
    snap = monitor.get_snapshot(baseline_id)
    _result("Snapshot ID",    snap.snapshot_id[-16:])
    _result("Total params",   str(snap.total_params))
    _result("Param groups",   str(len(snap.param_norms)))

    _sub("Simulating minor adaptation (±5% noise)")
    with torch.no_grad():
        for p in model.parameters():
            p.data += torch.randn_like(p.data) * 0.05
    metric_minor = monitor.measure(model, baseline_id)
    _result("Overall drift",  f"{metric_minor.overall_drift:.4f}")
    _result("Alert",          str(metric_minor.alert))
    _result("Status",         metric_minor.summary())

    _sub("Simulating large drift (±40% noise)")
    with torch.no_grad():
        for p in model.parameters():
            p.data += torch.randn_like(p.data) * 0.40
    metric_large = monitor.measure(model, baseline_id)
    _result("Overall drift",  f"{metric_large.overall_drift:.4f}")
    _result("Alert",          str(metric_large.alert))
    _result("Force rollback", str(metric_large.force_rollback))

    if metric_large.force_rollback:
        _sub("Rolling back to baseline ...")
        monitor.rollback(model, baseline_id)
        metric_post = monitor.measure(model, baseline_id)
        _result("Post-rollback drift", f"{metric_post.overall_drift:.6f}")

    _sub("Trend analysis")
    trend = monitor.log.trend()
    _result("Measurements",   str(trend.get("count", 0)))
    _result("Alert rate",     f"{trend.get('alert_rate', 0)*100:.0f}%")
    _result("Drift range",    f"{trend.get('min', 0):.4f} – {trend.get('max', 0):.4f}")

    print(f"\n  Drift monitor demo complete.")


# ---------------------------------------------------------------------------
# Section 6 — Distributed pipeline
# ---------------------------------------------------------------------------

def demo_pipeline(kg: str, rules: str) -> None:
    _header("Distributed Pipeline", "🔗")
    from synapseos.phase5.distributed import DistributedPipeline, PipelineTopology

    _sub("Local topology (single-machine)")
    topo     = PipelineTopology.local_default(kg, rules)
    pipeline = DistributedPipeline(topology=topo, kg_path=kg, rules_path=rules)

    health = pipeline.health_check_all()
    _result("Nodes", f"{sum(health.values())}/{len(health)} healthy")
    for node_id, ok in health.items():
        print(f"    {'✓' if ok else '✗'} {node_id}")

    queries = [
        "What fields are required for a Sales Invoice?",
        "Describe the Delivery Note document.",
        "List validate rules for Journal Entry.",
    ]
    _sub("Running 3 queries through pipeline")
    for q in queries:
        t0     = time.time()
        result = pipeline.run(q)
        ms     = (time.time() - t0) * 1000
        stages  = result.get("_stage_trace", [])
        ok_cnt  = sum(1 for s in stages if s["success"])
        print(f"  [{ms:5.0f}ms] {ok_cnt}/{len(stages)} stages OK — {q[:55]}")

    _sub("Latency breakdown")
    for node_id, stats in pipeline.latency_report().items():
        if stats.get("calls", 0) > 0:
            _result(node_id, f"p50={stats.get('p50_ms',0):.1f}ms  mean={stats.get('mean_ms',0):.1f}ms  calls={stats['calls']}")

    _sub("Distributed topology config (preview)")
    dist_topo = PipelineTopology.distributed_default()
    for stage in dist_topo.stages:
        print(f"    {stage['layer_type']:22} → {stage['node_id']:15} @ {stage['endpoint']}")

    print(f"\n  Pipeline demo complete.")


# ---------------------------------------------------------------------------
# Section 7 — Memory
# ---------------------------------------------------------------------------

def demo_memory(kg: str) -> None:
    _header("Episodic + Semantic Memory", "🧩")
    from synapseos.phase5.memory import MemoryController

    mc = MemoryController(
        episodic_path="data/memory/demo_episodic.json",
        semantic_path="data/memory/demo_semantic.json",
        kg_path=kg,
    )
    mc.episodic.entries.clear()

    _sub("Episodic memory — storing interactions")
    interactions = [
        ("What fields does Sales Invoice require?",  "Customer, Date, Items are mandatory.", "mode1"),
        ("Submit validation for Purchase Order",     "Checks supplier, items, and amounts.", "ariel"),
        ("Security audit for Payroll Entry",         "Found sensitive field: bank_account.",  "nyx"),
        ("Cache recommendations for the system",     "Sales Invoice has highest in-degree.",  "nova"),
        ("Identify fault risks in the pipeline",     "All 3 nodes are SPOFs currently.",     "atlas"),
    ]
    for q, r, mode in interactions:
        mc.record_interaction(q, r, mode)
        print(f"    Stored [{mode}]: {q[:50]}")

    _result("Total episodic entries", str(mc.stats()["episodic_entries"]))

    _sub("Episodic retrieval — similar past interactions")
    test_queries = [
        "Sales Invoice required fields validation",
        "security sensitive data audit",
    ]
    for test_q in test_queries:
        print(f"  Query: \"{test_q}\"")
        matches = mc.episodic.retrieve(test_q, top_k=2)
        for score, entry in matches:
            print(f"    [{score:.3f}] \"{entry.query[:55]}\"")

    _sub("Semantic memory — KG facts")
    _result("Total facts", str(mc.stats()["semantic_facts"]))
    facts = mc.semantic.retrieve("Sales Invoice", top_k=3)
    print(f"  Facts about 'Sales Invoice':")
    for f in facts[:3]:
        print(f"    {f.subject} {f.predicate} {f.obj}  (conf={f.confidence:.1f})")

    mc.assert_fact("SynapseOS", "achieves", "cross-domain reasoning", source="demo")
    _sub("Working memory")
    mc.set_working("demo_session", "running", ttl=300)
    mc.set_working("last_query",   "Sales Invoice validation")
    print(f"  Slots: {mc.working.keys()}")
    print(f"  demo_session = {mc.get_working('demo_session')}")

    print(f"\n  Memory demo complete.")


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

SECTIONS = {
    "modes":      demo_modes,
    "agents":     demo_agents,
    "creativity": demo_creativity,
    "voice":      demo_voice,
    "drift":      demo_drift,
    "pipeline":   demo_pipeline,
    "memory":     demo_memory,
}

def main():
    parser = argparse.ArgumentParser(description="SynapseOS Integrated Demo")
    parser.add_argument("--section",      default="all",
                        choices=list(SECTIONS) + ["all"],
                        help="Which demo section to run (default: all)")
    parser.add_argument("--kg",           default="data/graph/knowledge_graph.json")
    parser.add_argument("--expanded-kg",  default="data/graph/knowledge_graph_expanded.json")
    parser.add_argument("--rules",        default="data/rules/rules.json")
    parser.add_argument("--skip-modes",   action="store_true",
                        help="Skip the modes section (Mode 5 idle loop takes 30s)")
    args = parser.parse_args()

    expanded_kg = args.expanded_kg if Path(args.expanded_kg).exists() else None

    print(f"\n{'='*64}")
    print(f"  SynapseOS — Integrated System Demo")
    print(f"  KG: {args.kg}  ({Path(args.kg).stat().st_size // 1024}KB)")
    if expanded_kg:
        print(f"  Expanded KG: {expanded_kg}")
    print(f"  Rules: {args.rules}")
    print(f"{'='*64}")

    sections_to_run = list(SECTIONS) if args.section == "all" else [args.section]

    for section in sections_to_run:
        if section == "modes" and args.skip_modes:
            print(f"\n  [Skipping modes section (--skip-modes)]")
            continue

        t0 = time.time()
        try:
            if section == "creativity":
                demo_creativity(args.kg, expanded_kg)
            elif section in ("modes", "agents", "voice", "pipeline"):
                SECTIONS[section](args.kg, args.rules)
            elif section == "memory":
                demo_memory(args.kg)
            elif section == "drift":
                demo_drift()
        except Exception as e:
            print(f"\n  [Section '{section}' error: {e}]")
            import traceback
            traceback.print_exc()
        finally:
            elapsed = (time.time() - t0)
            print(f"\n  [{section}] completed in {elapsed:.1f}s")

    print(f"\n{'='*64}")
    print(f"  Demo complete.")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
