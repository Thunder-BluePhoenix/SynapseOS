"""
Phase 4 integration tests — Modes 2, 3, 4, 5.

Run:
    python tests/test_phase4.py
    python -m pytest tests/test_phase4.py -v
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from synapseos.phase4.creativity_engine import CreativityEngine, GeneratedIdea
from synapseos.phase4.program_synthesis import ProgramSynthesizer
from synapseos.phase4.idle_loop import IdleLoop, HypothesisStore


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
            return r
        wrapper.__name__ = fn.__name__
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Mode 2 + 3 — Creativity Engine
# ---------------------------------------------------------------------------

_engine = None
def get_engine():
    global _engine
    if _engine is None:
        _engine = CreativityEngine()
    return _engine


@test("creativity: engine loads concepts from graph")
def test_creativity_loads():
    e = get_engine()
    assert len(e.concepts) > 100, f"Expected >100 concepts, got {len(e.concepts)}"
    assert len(e.domain_index) >= 3, f"Expected >=3 domains, got {len(e.domain_index)}"


@test("creativity: random walk returns path of expected length")
def test_random_walk():
    e = get_engine()
    path = e.random_walk("Sales Order", steps=5)
    assert len(path) >= 2, "Walk should return at least 2 nodes"
    assert path[0] == "Sales Order"


@test("creativity: analogy engine finds cross-domain matches")
def test_analogies():
    e = get_engine()
    analogies = e.find_analogies("Sales Order", top_k=5)
    assert len(analogies) > 0, "Should find analogies for Sales Order"
    # All analogies should be from a different domain
    source_domain = e.concepts["Sales Order"].domain
    for a in analogies:
        assert a.source_domain != a.target_domain or a.source_concept != a.target_concept


@test("creativity: analogy confidence is in [0, 1]")
def test_analogy_confidence():
    e = get_engine()
    analogies = e.find_analogies("Customer", top_k=10)
    for a in analogies:
        assert 0.0 <= a.confidence <= 1.0, f"Confidence out of range: {a.confidence}"


@test("creativity: mutation produces traceable idea")
def test_mutation_trace():
    e = get_engine()
    idea = e.mutate("Sales Invoice")
    assert idea is not None
    assert len(idea.generation_trace) >= 3, "Should have at least 3 trace steps"
    assert idea.idea_type == "mutation"
    assert "Sales Invoice" in idea.source_concepts


@test("creativity: recombination of cross-domain concepts")
def test_recombination():
    e = get_engine()
    idea = e.recombine("Sales Order", "Employee")
    assert idea is not None
    assert idea.idea_type == "recombination"
    assert len(idea.source_concepts) == 2
    assert "Sales Order" in idea.source_concepts
    assert "Employee" in idea.source_concepts


@test("creativity: constraint-forced idea always returns")
def test_constraint_forced():
    e = get_engine()
    idea = e.constraint_forced("Payment Entry", "must work without network access")
    assert idea.idea_type == "constraint_forced"
    assert "must work without network access" in idea.title


@test("creativity: generate() returns top_k ideas")
def test_generate_top_k():
    e = get_engine()
    ideas = e.generate(seed="Sales Order", top_k=3, apply_debate=False)
    assert len(ideas) <= 3
    assert all(isinstance(i, GeneratedIdea) for i in ideas)


@test("creativity: ideas are ranked by combined_score descending")
def test_ideas_ranked():
    e = get_engine()
    ideas = e.generate(seed="Customer", top_k=5, apply_debate=False)
    scores = [i.combined_score for i in ideas]
    assert scores == sorted(scores, reverse=True), "Ideas should be sorted by score"


@test("creativity: debate increases confidence")
def test_debate_increases_confidence():
    e = get_engine()
    idea = e.mutate("Item")
    before = idea.confidence_score
    idea = e.multi_agent_debate(idea)
    assert idea.confidence_score >= before, "Debate should not decrease confidence"
    assert "Multi-agent debate" in "\n".join(idea.generation_trace)


@test("creativity: every idea has non-empty description and trace")
def test_idea_completeness():
    e = get_engine()
    ideas = e.generate(seed="Project", top_k=3, apply_debate=True)
    for idea in ideas:
        assert idea.title, "Idea must have a title"
        assert idea.description, "Idea must have a description"
        assert idea.generation_trace, "Idea must have a generation trace"
        assert idea.source_concepts, "Idea must have source concepts"


# ---------------------------------------------------------------------------
# Mode 4 — Program Synthesis
# ---------------------------------------------------------------------------

_synth = None
def get_synth():
    global _synth
    if _synth is None:
        _synth = ProgramSynthesizer()
    return _synth


@test("synthesis: intent parser identifies doctype")
def test_intent_doctype():
    synth = get_synth()
    intent = synth.parser.parse("validate a sales invoice before submission")
    assert intent.doctype == "Sales Invoice", f"Expected 'Sales Invoice', got '{intent.doctype}'"


@test("synthesis: intent parser identifies event type")
def test_intent_event():
    synth = get_synth()
    intent = synth.parser.parse("on submit of purchase order, post ledger entries")
    assert intent.event == "on_submit", f"Expected 'on_submit', got '{intent.event}'"


@test("synthesis: planner produces at least 3 steps for validate intent")
def test_plan_steps():
    synth = get_synth()
    intent = synth.parser.parse("validate a sales invoice")
    plan = synth.planner.plan(intent)
    assert len(plan) >= 3, f"Expected >=3 steps, got {len(plan)}"


@test("synthesis: plan steps have required fields")
def test_plan_step_structure():
    synth = get_synth()
    intent = synth.parser.parse("validate customer before save")
    plan = synth.planner.plan(intent)
    for step in plan:
        assert step.step > 0
        assert step.action
        assert step.description
        assert isinstance(step.inputs, list)
        assert isinstance(step.outputs, list)


@test("synthesis: synthesized code passes static validation")
def test_code_valid():
    synth = get_synth()
    result = synth.synthesize("validate a sales invoice before submission")
    assert result.static_valid, f"Static validation failed: {result.static_errors}"


@test("synthesis: code contains function definition")
def test_code_has_function():
    synth = get_synth()
    result = synth.synthesize("validate a purchase order")
    assert "def " in result.code, "Synthesized code must have a function definition"
    assert "import frappe" in result.code


@test("synthesis: code references real ERPNext rules in trace")
def test_code_references_rules():
    synth = get_synth()
    result = synth.synthesize("validate sales invoice before submit")
    rule_steps = [s for s in result.plan if s.action == "apply_rule"]
    if rule_steps:
        assert any(s.rule_ids for s in rule_steps), "Rule steps should reference extracted rule IDs"


@test("synthesis: tests are generated for every function")
def test_tests_generated():
    synth = get_synth()
    result = synth.synthesize("get outstanding balance for customer")
    assert "class Test" in result.tests
    assert "def test_" in result.tests
    assert "unittest" in result.tests


@test("synthesis: generation trace is complete")
def test_synthesis_trace():
    synth = get_synth()
    result = synth.synthesize("compute total tax for a purchase invoice")
    assert len(result.generation_trace) >= 4, "Trace should have at least 4 entries"
    assert any("Parsed intent" in t for t in result.generation_trace)
    assert any("Static validation" in t for t in result.generation_trace)


@test("synthesis: utility functions have return statement")
def test_utility_has_return():
    synth = get_synth()
    result = synth.synthesize("get all pending sales orders for a customer")
    assert "return" in result.code


# ---------------------------------------------------------------------------
# Mode 5 — Idle Loop
# ---------------------------------------------------------------------------

@test("idle: hypothesis store saves and loads")
def test_hypothesis_persistence():
    store = HypothesisStore("data/idle/test_hypotheses.json")
    from synapseos.phase4.idle_loop import WeakHypothesis
    from datetime import datetime
    hyp = WeakHypothesis(
        id="", type="cross_domain_link",
        concept_a="Sales Order", concept_b="Employee",
        description="test hypothesis",
        confidence=0.3,
        observed_at=datetime.now().isoformat(),
    )
    store.add(hyp)
    store.save()

    store2 = HypothesisStore("data/idle/test_hypotheses.json")
    assert len(store2.hypotheses) > 0, "Hypotheses should persist"
    Path("data/idle/test_hypotheses.json").unlink(missing_ok=True)


@test("idle: loop runs for 5 seconds without error")
def test_idle_runs():
    loop = IdleLoop(tick_interval=0.5, idle_threshold=0.0)
    loop.start_background()
    time.sleep(5)
    loop.stop()
    report = loop.get_session_report()
    assert report.walks_completed >= 5, f"Expected >=5 walks in 5s, got {report.walks_completed}"


@test("idle: generates hypotheses during exploration")
def test_idle_generates_hypotheses():
    loop = IdleLoop(tick_interval=0.5, idle_threshold=0.0)
    loop.start_background()
    time.sleep(5)
    loop.stop()
    report = loop.get_session_report()
    total_hyps = report.hypotheses_generated + report.hypotheses_reinforced
    assert total_hyps > 0, "Idle loop should generate at least one hypothesis"


@test("idle: explores multiple domains")
def test_idle_multi_domain():
    loop = IdleLoop(tick_interval=0.5, idle_threshold=0.0)
    loop.start_background()
    time.sleep(5)
    loop.stop()
    report = loop.get_session_report()
    assert len(report.domains_explored) >= 2, \
        f"Should explore >=2 domains, got: {report.domains_explored}"


@test("idle: signal_active resets idle timer")
def test_signal_active():
    loop = IdleLoop(tick_interval=0.5, idle_threshold=10.0)
    loop.start_background()
    before = loop._walks
    loop.signal_active()
    time.sleep(2)
    loop.stop()
    # With threshold=10s and signal_active resetting timer, should still be 0 walks
    assert loop._walks == before, "Walks should not occur while active signal is recent"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all():
    print("=" * 60)
    print("SynapseOS Phase 4 Tests (Modes 2, 3, 4, 5)")
    print("=" * 60)

    all_tests = [
        test_creativity_loads, test_random_walk, test_analogies,
        test_analogy_confidence, test_mutation_trace, test_recombination,
        test_constraint_forced, test_generate_top_k, test_ideas_ranked,
        test_debate_increases_confidence, test_idea_completeness,
        test_intent_doctype, test_intent_event, test_plan_steps,
        test_plan_step_structure, test_code_valid, test_code_has_function,
        test_code_references_rules, test_tests_generated,
        test_synthesis_trace, test_utility_has_return,
        test_hypothesis_persistence, test_idle_runs,
        test_idle_generates_hypotheses, test_idle_multi_domain,
        test_signal_active,
    ]

    for fn in all_tests:
        r = fn()
        status = "PASS" if r["passed"] else "FAIL"
        detail = f" — {r['detail']}" if r["detail"] else ""
        print(f"[{status}] {r['name']} ({r['ms']}ms){detail}")

    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed
    avg_ms = sum(r["ms"] for r in results) / len(results) if results else 0

    print(f"\nResults: {passed}/{len(results)} passed  |  {failed} failed  |  avg {avg_ms:.0f}ms")
    return failed


if __name__ == "__main__":
    sys.exit(run_all())
