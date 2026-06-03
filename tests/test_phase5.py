"""
Phase 5 integration tests — Memory architecture + Cognitive Controller.

Run:
    python tests/test_phase5.py
    python -m pytest tests/test_phase5.py -v
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from synapseos.phase5.memory import (
    EpisodicMemory,
    SemanticMemory,
    WorkingMemory,
    MemoryController,
    _text_to_vector,
    _cosine,
)
from synapseos.phase5.cognitive_controller import (
    ModeRouter,
    ThinkingMode,
    CognitiveController,
    AgentRouter,
    ParallelExecutor,
    ResultSynthesizer,
)
from synapseos.phase5.agents.base import SubAgent, AgentResult, AgentCapability
from synapseos.phase5.agents.ariel import ARIELAgent
from synapseos.phase5.agents.nyx import NYXAgent


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
# Vector utilities
# ---------------------------------------------------------------------------

@test("memory: text_to_vector produces correct-dimension output")
def test_vector_dim():
    vec = _text_to_vector("Sales Invoice", dim=128)
    assert len(vec) == 128, f"Expected 128, got {len(vec)}"


@test("memory: text_to_vector is L2-normalised")
def test_vector_normalised():
    import math
    vec = _text_to_vector("Purchase Order validate", dim=64)
    norm = math.sqrt(sum(x*x for x in vec))
    assert abs(norm - 1.0) < 1e-6, f"Not normalised: norm={norm}"


@test("memory: cosine similarity — identical vectors score 1.0")
def test_cosine_identical():
    vec = _text_to_vector("Sales Invoice", dim=64)
    score = _cosine(vec, vec)
    assert abs(score - 1.0) < 1e-6, f"Expected 1.0, got {score}"


@test("memory: cosine similarity — similar texts score higher than unrelated")
def test_cosine_ordering():
    va = _text_to_vector("Sales Invoice validate fields", dim=128)
    vb = _text_to_vector("Sales Invoice submit rules",   dim=128)
    vc = _text_to_vector("Payroll tax bracket computation", dim=128)
    score_similar  = _cosine(va, vb)
    score_unrelated = _cosine(va, vc)
    assert score_similar > score_unrelated, (
        f"Similar={score_similar:.3f} should > unrelated={score_unrelated:.3f}"
    )


# ---------------------------------------------------------------------------
# Episodic Memory
# ---------------------------------------------------------------------------

@test("episodic: add and retrieve entries")
def test_episodic_add_retrieve():
    mem = EpisodicMemory(store_path="/tmp/test_episodic.json", max_entries=100)
    mem.entries.clear()

    mem.add("What fields does Sales Invoice need?",      "Customer, Date, Items.", "mode1")
    mem.add("Write a validate function for Purchase Order.", "def validate...",    "mode4")
    mem.add("Generate a creative HR onboarding workflow.",  "Hybrid idea...",      "mode2")

    assert len(mem) == 3, f"Expected 3, got {len(mem)}"


@test("episodic: retrieve returns top-k most similar")
def test_episodic_retrieve_topk():
    mem = EpisodicMemory(store_path="/tmp/test_episodic.json", max_entries=100)
    mem.entries.clear()

    mem.add("Sales Invoice required fields",       "Customer, Date.", "mode1")
    mem.add("Sales Invoice validation rules",      "Mandatory item.",  "mode1")
    mem.add("HR payroll calculation steps",        "Salary formula.",  "mode1")
    mem.add("Manufacturing BOM structure",         "BOM explode.",     "mode1")

    results_out = mem.retrieve("Sales Invoice fields check", top_k=2)
    assert len(results_out) <= 2, f"Expected ≤2, got {len(results_out)}"

    if len(results_out) >= 2:
        scores = [s for s, _ in results_out]
        # Sales-related entries should rank above HR/Manufacturing
        assert scores[0] >= scores[1], "Results not sorted by descending score"


@test("episodic: get_recent returns last N entries in reverse order")
def test_episodic_recent():
    mem = EpisodicMemory(store_path="/tmp/test_episodic.json", max_entries=100)
    mem.entries.clear()
    for i in range(5):
        mem.add(f"query {i}", f"response {i}", "mode1")
    recent = mem.get_recent(3)
    assert len(recent) == 3
    assert recent[0].query == "query 4", f"Expected 'query 4', got '{recent[0].query}'"


@test("episodic: max_entries eviction works")
def test_episodic_eviction():
    mem = EpisodicMemory(store_path="/tmp/test_episodic_evict.json", max_entries=5)
    mem.entries.clear()
    for i in range(8):
        mem.add(f"query {i}", f"response {i}", "mode1")
    assert len(mem) == 5, f"Expected 5 after eviction, got {len(mem)}"


# ---------------------------------------------------------------------------
# Semantic Memory
# ---------------------------------------------------------------------------

@test("semantic: initialises without error (no KG path)")
def test_semantic_init():
    mem = SemanticMemory(store_path="/tmp/test_semantic.json", kg_path="/tmp/nonexistent_kg.json")
    assert mem is not None


@test("semantic: add_fact and retrieve by subject")
def test_semantic_add_retrieve():
    mem = SemanticMemory(store_path="/tmp/test_semantic.json", kg_path="/tmp/nonexistent_kg.json")
    mem.facts.clear()
    mem._fact_index.clear()

    mem.add_fact("Sales Invoice", "requires", "Customer", confidence=1.0, source="test")
    mem.add_fact("Sales Invoice", "requires", "Posting Date", confidence=1.0, source="test")
    mem.add_fact("Purchase Order", "requires", "Supplier", confidence=1.0, source="test")

    facts = mem.retrieve("Sales Invoice")
    assert len(facts) >= 2, f"Expected ≥2 facts, got {len(facts)}"
    subjects = {f.subject for f in facts}
    assert "Sales Invoice" in subjects


@test("semantic: retrieve_text matches on query tokens")
def test_semantic_retrieve_text():
    mem = SemanticMemory(store_path="/tmp/test_semantic.json", kg_path="/tmp/nonexistent_kg.json")
    mem.facts.clear()
    mem._fact_index.clear()

    mem.add_fact("Sales Invoice", "validates", "Customer", source="test")
    mem.add_fact("Purchase Order", "validates", "Supplier", source="test")
    mem.add_fact("Employee", "belongs_to", "Department", source="test")

    facts = mem.retrieve_text("Sales Invoice validation", top_k=3)
    subjects = {f.subject for f in facts}
    assert "Sales Invoice" in subjects, f"Expected Sales Invoice in {subjects}"


# ---------------------------------------------------------------------------
# Working Memory
# ---------------------------------------------------------------------------

@test("working: write and read")
def test_working_write_read():
    wm = WorkingMemory()
    wm.write("current_query", "What fields are required?")
    val = wm.read("current_query")
    assert val == "What fields are required?"


@test("working: expired entries return default")
def test_working_expiry():
    wm = WorkingMemory(default_ttl=0.01)
    wm.write("key", "value", ttl=0.01)
    time.sleep(0.05)
    val = wm.read("key", default="gone")
    assert val == "gone", f"Expected 'gone', got '{val}'"


@test("working: clear empties all slots")
def test_working_clear():
    wm = WorkingMemory()
    wm.write("a", 1)
    wm.write("b", 2)
    wm.clear()
    assert len(wm) == 0


@test("working: snapshot returns active slots only")
def test_working_snapshot():
    wm = WorkingMemory()
    wm.write("active", "yes", ttl=300)
    wm.write("expired", "no", ttl=0.01)
    time.sleep(0.05)
    snap = wm.snapshot()
    assert "active" in snap
    assert "expired" not in snap


# ---------------------------------------------------------------------------
# Memory Controller
# ---------------------------------------------------------------------------

@test("memory controller: initialises and returns stats")
def test_memory_controller_init():
    mc = MemoryController(
        episodic_path="/tmp/test_mc_episodic.json",
        semantic_path="/tmp/test_mc_semantic.json",
        kg_path="/tmp/nonexistent_kg.json",
    )
    stats = mc.stats()
    assert "episodic_entries" in stats
    assert "semantic_facts" in stats
    assert "working_slots" in stats


@test("memory controller: record_interaction and get_context")
def test_memory_controller_record_context():
    mc = MemoryController(
        episodic_path="/tmp/test_mc_episodic2.json",
        semantic_path="/tmp/test_mc_semantic2.json",
        kg_path="/tmp/nonexistent_kg.json",
    )
    mc.episodic.entries.clear()

    mc.record_interaction("Sales Invoice fields", "Customer required.", "mode1")
    mc.record_interaction("Purchase Order fields", "Supplier required.", "mode1")

    ctx = mc.get_context("Sales Invoice required fields")
    assert ctx.summary, "Expected non-empty summary"
    assert len(ctx.recent_interactions) >= 1


@test("memory controller: assert_fact persists to semantic memory")
def test_memory_controller_assert_fact():
    mc = MemoryController(
        episodic_path="/tmp/test_mc_episodic3.json",
        semantic_path="/tmp/test_mc_semantic3.json",
        kg_path="/tmp/nonexistent_kg.json",
    )
    mc.semantic.facts.clear()
    mc.semantic._fact_index.clear()

    mc.assert_fact("Delivery Note", "links_to", "Sales Invoice", source="test")
    facts = mc.semantic.retrieve("Delivery Note")
    assert len(facts) >= 1


@test("memory controller: working memory read/write via controller")
def test_memory_controller_working():
    mc = MemoryController(
        episodic_path="/tmp/test_mc_episodic4.json",
        semantic_path="/tmp/test_mc_semantic4.json",
        kg_path="/tmp/nonexistent_kg.json",
    )
    mc.set_working("task", "validate Sales Invoice")
    val = mc.get_working("task")
    assert val == "validate Sales Invoice"


# ---------------------------------------------------------------------------
# Mode Router
# ---------------------------------------------------------------------------

@test("router: code query routes to mode4")
def test_router_code():
    router = ModeRouter()
    modes = router.select_modes("write a validate function for Sales Order", max_modes=2)
    assert ThinkingMode.MODE4_CODE in modes, f"Expected mode4 in {modes}"


@test("router: creative query routes to mode2")
def test_router_creative():
    router = ModeRouter()
    modes = router.select_modes("brainstorm creative ideas for HR automation", max_modes=2)
    assert ThinkingMode.MODE2_CREATIVE in modes, f"Expected mode2 in {modes}"


@test("router: reasoning query routes to mode1")
def test_router_reasoning():
    router = ModeRouter()
    modes = router.select_modes("what fields are required for a Sales Invoice?", max_modes=2)
    assert ThinkingMode.MODE1_REASONING in modes, f"Expected mode1 in {modes}"


@test("router: synthesis query routes to mode3")
def test_router_synthesis():
    router = ModeRouter()
    modes = router.select_modes("find an analogy between inventory and HR workflows", max_modes=3)
    assert ThinkingMode.MODE3_SYNTHESIS in modes, f"Expected mode3 in {modes}"


@test("router: unknown query defaults to mode1")
def test_router_unknown():
    router = ModeRouter()
    modes = router.select_modes("xyzzy frobnicator blorp", max_modes=2)
    assert len(modes) >= 1
    assert ThinkingMode.MODE1_REASONING in modes


@test("router: classify returns confidence scores")
def test_router_classify_scores():
    router = ModeRouter()
    ranked = router.classify("write and implement a Sales Invoice validate function")
    assert len(ranked) >= 1
    for mode, conf in ranked:
        assert 0.0 <= conf <= 1.0, f"Confidence out of range: {conf}"


# ---------------------------------------------------------------------------
# Cognitive Controller — end-to-end
# ---------------------------------------------------------------------------

_controller = None

def get_controller():
    global _controller
    if _controller is None:
        _controller = CognitiveController(
            memory_episodic_path="/tmp/test_cc_episodic.json",
            memory_semantic_path="/tmp/test_cc_semantic.json",
            kg_path="data/graph/knowledge_graph.json",
            mode_timeout=20.0,
            max_parallel_modes=2,
            enable_idle_loop=False,
        )
    return _controller


@test("controller: initialises without error")
def test_controller_init():
    c = get_controller()
    assert c is not None
    stats = c.stats()
    assert "episodic_entries" in stats


@test("controller: think() returns a CognitionResult")
def test_controller_think_basic():
    c = get_controller()
    result = c.think("What fields are required for a Sales Invoice?")
    assert result is not None
    assert result.query == "What fields are required for a Sales Invoice?"
    assert result.primary_output, "Expected non-empty output"
    assert len(result.modes_activated) >= 1
    assert result.total_latency_ms > 0


@test("controller: think() with forced mode4 activates code mode")
def test_controller_think_forced_mode4():
    c = get_controller()
    result = c.think("validate a Purchase Order", force_modes=["mode4"])
    assert ThinkingMode.MODE4_CODE in result.modes_activated
    assert result.output_type in ("code", "language", "both")


@test("controller: think() with forced mode2 activates creative mode")
def test_controller_think_forced_mode2():
    c = get_controller()
    result = c.think("brainstorm", force_modes=["mode2"])
    assert ThinkingMode.MODE2_CREATIVE in result.modes_activated


@test("controller: think() routes code query to mode4 automatically")
def test_controller_auto_route_code():
    c = get_controller()
    result = c.think("write a validate function for the Sales Invoice doctype")
    assert ThinkingMode.MODE4_CODE in result.modes_activated


@test("controller: think() routes reasoning query to mode1 automatically")
def test_controller_auto_route_reasoning():
    c = get_controller()
    result = c.think("what required fields does a Sales Order have?")
    assert ThinkingMode.MODE1_REASONING in result.modes_activated


@test("controller: parallel mode execution — mode1+mode4")
def test_controller_parallel_modes():
    c = get_controller()
    result = c.think(
        "explain the required fields and write a validate function for Purchase Order",
        force_modes=["mode1", "mode4"],
    )
    activated = result.modes_activated
    assert ThinkingMode.MODE1_REASONING in activated
    assert ThinkingMode.MODE4_CODE in activated
    assert len(result.mode_results) == 2


@test("controller: episodic memory records each interaction")
def test_controller_memory_records():
    c = get_controller()
    before = c.memory.stats()["episodic_entries"]
    c.think("What validate rules apply to Stock Entry?")
    after = c.memory.stats()["episodic_entries"]
    assert after > before, f"Expected episodic count to grow: {before} → {after}"


@test("controller: to_dict() produces serialisable output")
def test_controller_to_dict():
    import json
    c = get_controller()
    result = c.think("list fields for Payment Entry", force_modes=["mode1"])
    d = result.to_dict()
    serialised = json.dumps(d)
    assert serialised, "Expected non-empty JSON"
    parsed = json.loads(serialised)
    assert "query" in parsed
    assert "modes_activated" in parsed
    assert "primary_output" in parsed


@test("controller: confidence is in [0, 1]")
def test_controller_confidence_range():
    c = get_controller()
    result = c.think("What is a Sales Invoice?")
    assert 0.0 <= result.confidence <= 1.0, f"Confidence out of range: {result.confidence}"


@test("controller: latency is measured and positive")
def test_controller_latency():
    c = get_controller()
    result = c.think("list the required fields for a Journal Entry", force_modes=["mode1"])
    assert result.total_latency_ms > 0, "Latency must be positive"


@test("controller: stats() returns correct keys")
def test_controller_stats():
    c = get_controller()
    stats = c.stats()
    for key in ("episodic_entries", "semantic_facts", "working_slots", "idle_loop_active", "agents_loaded"):
        assert key in stats, f"Missing key: {key}"


@test("controller: agents_loaded includes ARIEL and NYX")
def test_controller_agents_loaded():
    c = get_controller()
    agents = c.stats()["agents_loaded"]
    assert "ARIEL" in agents, f"ARIEL missing from {agents}"
    assert "NYX" in agents, f"NYX missing from {agents}"


# ---------------------------------------------------------------------------
# ARIEL sub-agent tests
# ---------------------------------------------------------------------------

_ariel = None
def get_ariel():
    global _ariel
    if _ariel is None:
        _ariel = ARIELAgent(
            kg_path="data/graph/knowledge_graph.json",
            rules_path="data/rules/rules.json",
        )
    return _ariel


@test("ARIEL: can_handle scores business logic queries high")
def test_ariel_can_handle_high():
    agent = get_ariel()
    score = agent.can_handle("plan the submission steps for a Sales Invoice")
    assert score >= 0.55, f"Expected ≥0.55, got {score}"


@test("ARIEL: can_handle scores security queries low")
def test_ariel_can_handle_low():
    agent = get_ariel()
    score = agent.can_handle("audit the security permissions for Sales Invoice")
    assert score < 0.7, f"Expected <0.7 for security query, got {score}"


@test("ARIEL: find_doctype extracts known DocType from query")
def test_ariel_find_doctype():
    agent = get_ariel()
    dt = agent.find_doctype("plan submission steps for a Sales Invoice")
    assert dt != "Unknown", f"Expected a real DocType, got {dt}"


@test("ARIEL: find_operation extracts submit")
def test_ariel_find_operation():
    agent = get_ariel()
    op = agent.find_operation("how do I submit a Sales Invoice?")
    assert op == "submit", f"Expected 'submit', got '{op}'"


@test("ARIEL: analyze_workflow returns WorkflowAnalysis with fields populated")
def test_ariel_analyze_workflow():
    agent = get_ariel()
    analysis = agent.analyze_workflow("Sales Invoice")
    assert analysis.doctype == "Sales Invoice"
    assert isinstance(analysis.rule_count, int)
    assert isinstance(analysis.linked_doctypes, list)
    assert isinstance(analysis.reverse_links, list)


@test("ARIEL: plan_operation returns OperationPlan with steps")
def test_ariel_plan_operation():
    agent = get_ariel()
    plan = agent.plan_operation("Sales Invoice", "submit")
    assert plan.operation == "submit"
    assert len(plan.steps) >= 2, f"Expected ≥2 steps, got {len(plan.steps)}"
    assert isinstance(plan.warnings, list)


@test("ARIEL: get_mandatory_fields returns list")
def test_ariel_mandatory_fields():
    agent = get_ariel()
    fields = agent.get_mandatory_fields("Sales Invoice")
    assert isinstance(fields, list)


@test("ARIEL: handle() returns AgentResult with output")
def test_ariel_handle_basic():
    agent = get_ariel()
    result = agent.handle("plan the submission steps for Sales Invoice")
    assert result.agent_name == "ARIEL"
    assert result.output, "Expected non-empty output"
    assert result.traceable
    assert len(result.trace) >= 2
    assert result.latency_ms > 0


@test("ARIEL: handle() routes dependency query to workflow analysis")
def test_ariel_handle_dependency():
    agent = get_ariel()
    result = agent.handle("what does Sales Invoice depend on and link to?")
    assert result.agent_name == "ARIEL"
    assert not result.error, f"Error: {result.error}"
    assert "linked" in result.structured or "linked_doctypes" in result.structured


@test("ARIEL: handle() routes mandatory fields query")
def test_ariel_handle_mandatory():
    agent = get_ariel()
    result = agent.handle("what mandatory fields does Sales Invoice require?")
    assert result.agent_name == "ARIEL"
    assert "mandatory_fields" in result.structured


@test("ARIEL: confidence is in [0, 1]")
def test_ariel_confidence_range():
    agent = get_ariel()
    result = agent.handle("plan the submission for a Purchase Order")
    assert 0.0 <= result.confidence <= 1.0, f"Confidence out of range: {result.confidence}"


# ---------------------------------------------------------------------------
# NYX sub-agent tests
# ---------------------------------------------------------------------------

_nyx = None
def get_nyx():
    global _nyx
    if _nyx is None:
        _nyx = NYXAgent(
            kg_path="data/graph/knowledge_graph.json",
            rules_path="data/rules/rules.json",
        )
    return _nyx


@test("NYX: can_handle scores security queries high")
def test_nyx_can_handle_high():
    agent = get_nyx()
    score = agent.can_handle("audit the security permissions for Sales Invoice")
    assert score >= 0.60, f"Expected ≥0.60, got {score}"


@test("NYX: can_handle scores business logic queries lower")
def test_nyx_can_handle_lower():
    agent = get_nyx()
    score = agent.can_handle("plan the submission steps for a Sales Invoice")
    assert score < 0.65, f"Expected <0.65 for business logic query, got {score}"


@test("NYX: detect_sensitive_fields returns list")
def test_nyx_detect_sensitive():
    agent = get_nyx()
    fields = agent.detect_sensitive_fields("Sales Invoice")
    assert isinstance(fields, list)


@test("NYX: sensitive field patterns compile and match correctly")
def test_nyx_pattern_matching():
    agent = get_nyx()
    # Test internal pattern matching directly
    hit = agent._check_sensitive("api_key", "TestDocType")
    assert hit is not None, "Expected a hit for 'api_key'"
    assert hit.risk_level == "CRITICAL"
    assert hit.category == "credential"

    hit2 = agent._check_sensitive("customer_email", "TestDocType")
    assert hit2 is not None, "Expected a hit for 'customer_email'"
    assert hit2.risk_level == "MEDIUM"

    no_hit = agent._check_sensitive("item_name", "TestDocType")
    assert no_hit is None, "Expected no hit for 'item_name'"


@test("NYX: generate_audit_spec returns AuditSpec with must_log")
def test_nyx_audit_spec():
    agent = get_nyx()
    spec = agent.generate_audit_spec("Sales Invoice", "submit")
    assert spec.doctype == "Sales Invoice"
    assert spec.operation == "submit"
    assert len(spec.must_log) >= 3, f"Expected ≥3 must_log items, got {len(spec.must_log)}"
    assert any("user" in item.lower() for item in spec.must_log), "Expected 'user' in must_log"
    assert any("timestamp" in item.lower() for item in spec.must_log), "Expected 'timestamp' in must_log"


@test("NYX: security_report returns SecurityReport with risk_level")
def test_nyx_security_report():
    agent = get_nyx()
    report = agent.security_report("Sales Invoice")
    assert report.doctype == "Sales Invoice"
    assert report.risk_level in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "OK")
    assert isinstance(report.findings, list)
    assert isinstance(report.recommendations, list)


@test("NYX: scan_high_risk_doctypes returns list")
def test_nyx_surface_scan():
    agent = get_nyx()
    results_out = agent.scan_high_risk_doctypes(top_k=5)
    assert isinstance(results_out, list)
    assert len(results_out) <= 5
    for dt, level in results_out:
        assert level in ("CRITICAL", "HIGH"), f"Surface scan should only return CRITICAL/HIGH, got {level}"


@test("NYX: handle() returns AgentResult with output")
def test_nyx_handle_basic():
    agent = get_nyx()
    result = agent.handle("audit the permission model for Sales Invoice")
    assert result.agent_name == "NYX"
    assert result.output, "Expected non-empty output"
    assert result.traceable
    assert result.latency_ms > 0


@test("NYX: handle() routes audit query to audit spec")
def test_nyx_handle_audit():
    agent = get_nyx()
    result = agent.handle("generate an audit trail spec for Sales Invoice submit")
    assert result.agent_name == "NYX"
    assert not result.error, f"Error: {result.error}"
    assert "must_log" in result.structured


@test("NYX: handle() routes sensitive field query")
def test_nyx_handle_sensitive():
    agent = get_nyx()
    result = agent.handle("what sensitive fields does the User doctype expose?")
    assert result.agent_name == "NYX"
    assert "sensitive_fields" in result.structured


@test("NYX: handle() routes surface scan query")
def test_nyx_handle_surface_scan():
    agent = get_nyx()
    result = agent.handle("scan for high risk doctypes in the system")
    assert result.agent_name == "NYX"
    assert "high_risk_doctypes" in result.structured


@test("NYX: confidence is in [0, 1]")
def test_nyx_confidence_range():
    agent = get_nyx()
    result = agent.handle("check security for Purchase Order")
    assert 0.0 <= result.confidence <= 1.0, f"Confidence out of range: {result.confidence}"


# ---------------------------------------------------------------------------
# AgentRouter tests
# ---------------------------------------------------------------------------

_agent_router = None
def get_agent_router():
    global _agent_router
    if _agent_router is None:
        _agent_router = AgentRouter(
            kg_path="data/graph/knowledge_graph.json",
            rules_path="data/rules/rules.json",
        )
    return _agent_router


@test("agent router: loads ARIEL and NYX")
def test_agent_router_loads():
    router = get_agent_router()
    assert "ARIEL" in router.agent_names
    assert "NYX" in router.agent_names


@test("agent router: score() returns agents above threshold for business query")
def test_agent_router_score_business():
    router = get_agent_router()
    scored = router.score("plan the submission workflow for Sales Invoice")
    assert len(scored) >= 1, "Expected at least one agent above threshold"
    names = [a.name for _, a in scored]
    assert "ARIEL" in names, f"Expected ARIEL in scored agents: {names}"


@test("agent router: score() returns NYX for security query")
def test_agent_router_score_security():
    router = get_agent_router()
    scored = router.score("audit security permissions for Sales Invoice")
    names = [a.name for _, a in scored]
    assert "NYX" in names, f"Expected NYX in scored agents: {names}"


@test("agent router: dispatch() returns AgentResult list")
def test_agent_router_dispatch():
    router = get_agent_router()
    agent_results = router.dispatch("plan submission steps for Sales Invoice", context=None, timeout=20.0)
    assert isinstance(agent_results, list)
    assert len(agent_results) >= 1
    for r in agent_results:
        assert hasattr(r, "agent_name")
        assert hasattr(r, "output")
        assert hasattr(r, "confidence")


# ---------------------------------------------------------------------------
# CognitiveController + agents end-to-end
# ---------------------------------------------------------------------------

@test("controller: think() with business query routes to ARIEL")
def test_controller_agent_ariel_routing():
    c = get_controller()
    result = c.think("plan the submission workflow steps for a Sales Invoice")
    # ARIEL should have been dispatched (either primary or in agent_results)
    agent_names = [a.agent_name for a in result.agent_results]
    assert "ARIEL" in agent_names, f"Expected ARIEL in agent_results: {agent_names}"


@test("controller: think() with security query routes to NYX")
def test_controller_agent_nyx_routing():
    c = get_controller()
    result = c.think("audit the security and permissions for Sales Invoice")
    agent_names = [a.agent_name for a in result.agent_results]
    assert "NYX" in agent_names, f"Expected NYX in agent_results: {agent_names}"


@test("controller: routed_to_agent is set when agent confidence wins")
def test_controller_routed_to_agent_field():
    c = get_controller()
    result = c.think("plan the submission steps for Sales Invoice")
    # routed_to_agent should be a string (empty or agent name)
    assert isinstance(result.routed_to_agent, str)


@test("controller: agent_results included in to_dict()")
def test_controller_agent_results_in_dict():
    import json
    c = get_controller()
    result = c.think("audit security for Sales Invoice")
    d = result.to_dict()
    assert "agent_results" in d
    assert isinstance(d["agent_results"], list)
    json.dumps(d)   # must be JSON-serialisable


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_vector_dim,
        test_vector_normalised,
        test_cosine_identical,
        test_cosine_ordering,
        test_episodic_add_retrieve,
        test_episodic_retrieve_topk,
        test_episodic_recent,
        test_episodic_eviction,
        test_semantic_init,
        test_semantic_add_retrieve,
        test_semantic_retrieve_text,
        test_working_write_read,
        test_working_expiry,
        test_working_clear,
        test_working_snapshot,
        test_memory_controller_init,
        test_memory_controller_record_context,
        test_memory_controller_assert_fact,
        test_memory_controller_working,
        test_router_code,
        test_router_creative,
        test_router_reasoning,
        test_router_synthesis,
        test_router_unknown,
        test_router_classify_scores,
        test_controller_init,
        test_controller_think_basic,
        test_controller_think_forced_mode4,
        test_controller_think_forced_mode2,
        test_controller_auto_route_code,
        test_controller_auto_route_reasoning,
        test_controller_parallel_modes,
        test_controller_memory_records,
        test_controller_to_dict,
        test_controller_confidence_range,
        test_controller_latency,
        test_controller_stats,
        test_controller_agents_loaded,
        # ARIEL
        test_ariel_can_handle_high,
        test_ariel_can_handle_low,
        test_ariel_find_doctype,
        test_ariel_find_operation,
        test_ariel_analyze_workflow,
        test_ariel_plan_operation,
        test_ariel_mandatory_fields,
        test_ariel_handle_basic,
        test_ariel_handle_dependency,
        test_ariel_handle_mandatory,
        test_ariel_confidence_range,
        # NYX
        test_nyx_can_handle_high,
        test_nyx_can_handle_lower,
        test_nyx_detect_sensitive,
        test_nyx_pattern_matching,
        test_nyx_audit_spec,
        test_nyx_security_report,
        test_nyx_surface_scan,
        test_nyx_handle_basic,
        test_nyx_handle_audit,
        test_nyx_handle_sensitive,
        test_nyx_handle_surface_scan,
        test_nyx_confidence_range,
        # AgentRouter
        test_agent_router_loads,
        test_agent_router_score_business,
        test_agent_router_score_security,
        test_agent_router_dispatch,
        # Controller + agents
        test_controller_agent_ariel_routing,
        test_controller_agent_nyx_routing,
        test_controller_routed_to_agent_field,
        test_controller_agent_results_in_dict,
    ]

    for t in tests:
        t()

    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed

    print(f"\n{'='*60}")
    print(f"Phase 5 Tests: {passed}/{len(results)} passed\n")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        detail = f"  → {r['detail']}" if not r["passed"] else ""
        print(f"  [{status}] {r['name']}  ({r['ms']}ms){detail}")

    if _controller:
        _controller.shutdown()

    sys.exit(0 if failed == 0 else 1)
