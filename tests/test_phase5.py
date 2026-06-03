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
from synapseos.phase5.agents.nova import NOVAAgent
from synapseos.phase5.agents.atlas import ATLASAgent
from synapseos.phase5.drift_monitor import DriftMonitor, AdaptationLog, WeightSnapshot, DriftMetric
from synapseos.phase5.distributed import (
    MessageBus, LayerNode, LayerRequest, LayerResponse,
    LayerType, NodeConfig, PipelineTopology, DistributedPipeline,
    TextEncoderNode, NeurosymbolicNode, OutputRendererNode,
)
from synapseos.phase5.voice import (
    WhisperTranscriber, TTSSynthesizer, VoicePipeline,
    TranscriptionResult, SynthesisResult, VoiceActivityDetector,
)


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
# NOVA sub-agent tests
# ---------------------------------------------------------------------------

_nova = None
def get_nova():
    global _nova
    if _nova is None:
        _nova = NOVAAgent(
            kg_path="data/graph/knowledge_graph.json",
            rules_path="data/rules/rules.json",
        )
    return _nova


@test("NOVA: can_handle scores optimization queries high")
def test_nova_can_handle_high():
    agent = get_nova()
    score = agent.can_handle("which doctypes should we cache for better performance?")
    assert score >= 0.55, f"Expected ≥0.55, got {score}"


@test("NOVA: can_handle scores security queries low")
def test_nova_can_handle_low():
    agent = get_nova()
    score = agent.can_handle("audit the security permissions")
    assert score < 0.60, f"Expected <0.60 for security query, got {score}"


@test("NOVA: rule_density_report returns sorted list")
def test_nova_rule_density():
    agent = get_nova()
    entries = agent.rule_density_report(top_k=5)
    assert len(entries) <= 5
    assert len(entries) >= 1
    for i in range(len(entries) - 1):
        assert entries[i].rule_count >= entries[i+1].rule_count, "Expected sorted descending"


@test("NOVA: kg_bottlenecks returns nodes with degree info")
def test_nova_kg_bottlenecks():
    agent = get_nova()
    bottlenecks = agent.kg_bottlenecks(top_k=5)
    assert len(bottlenecks) >= 1
    for b in bottlenecks:
        assert b.combined == b.in_degree + b.out_degree
        assert b.combined >= 1


@test("NOVA: cache_recommendations returns valid entries")
def test_nova_cache_recommendations():
    agent = get_nova()
    recs = agent.cache_recommendations(top_k=8)
    assert isinstance(recs, list)
    for r in recs:
        assert r.priority in ("HIGH", "MEDIUM", "LOW")
        assert r.in_degree >= 1


@test("NOVA: detect_redundancies returns findings with duplicate_count >= 2")
def test_nova_redundancies():
    agent = get_nova()
    findings = agent.detect_redundancies(top_k=10)
    assert isinstance(findings, list)
    for f in findings:
        assert f.duplicate_count >= 2, f"Expected ≥2 duplicates, got {f.duplicate_count}"


@test("NOVA: system_benchmark returns populated SystemBenchmark")
def test_nova_benchmark():
    agent = get_nova()
    bm = agent.system_benchmark()
    assert bm.kg_nodes > 0
    assert bm.kg_edges > 0
    assert bm.total_rules > 0
    assert bm.avg_rules_per_doctype > 0
    assert bm.most_connected_node != ""
    assert bm.most_rule_dense_doctype != ""


@test("NOVA: handle() returns AgentResult with output")
def test_nova_handle_basic():
    agent = get_nova()
    result = agent.handle("show the full system optimization report")
    assert result.agent_name == "NOVA"
    assert result.output, "Expected non-empty output"
    assert result.traceable
    assert result.latency_ms > 0


@test("NOVA: handle() routes cache query")
def test_nova_handle_cache():
    agent = get_nova()
    result = agent.handle("which doctypes should we cache?")
    assert result.agent_name == "NOVA"
    assert "recommendations" in result.structured


@test("NOVA: handle() routes bottleneck query")
def test_nova_handle_bottleneck():
    agent = get_nova()
    result = agent.handle("identify the KG connectivity bottlenecks")
    assert result.agent_name == "NOVA"
    assert "bottlenecks" in result.structured


@test("NOVA: handle() routes redundancy query")
def test_nova_handle_redundancy():
    agent = get_nova()
    result = agent.handle("find redundant duplicate rules in the system")
    assert result.agent_name == "NOVA"
    assert "redundancies" in result.structured


@test("NOVA: handle() routes benchmark query")
def test_nova_handle_benchmark():
    agent = get_nova()
    result = agent.handle("show system benchmark stats report")
    assert result.agent_name == "NOVA"
    assert "kg_nodes" in result.structured


@test("NOVA: confidence is in [0, 1]")
def test_nova_confidence_range():
    agent = get_nova()
    result = agent.handle("optimize caching strategy for the pipeline")
    assert 0.0 <= result.confidence <= 1.0, f"Confidence out of range: {result.confidence}"


@test("controller: NOVA loaded in agent router")
def test_controller_nova_loaded():
    c = get_controller()
    agents = c.stats()["agents_loaded"]
    assert "NOVA" in agents, f"NOVA missing from {agents}"


@test("controller: optimization query routes to NOVA")
def test_controller_nova_routing():
    c = get_controller()
    result = c.think("optimize and benchmark the system performance")
    agent_names = [a.agent_name for a in result.agent_results]
    assert "NOVA" in agent_names, f"Expected NOVA in agent_results: {agent_names}"


# ---------------------------------------------------------------------------
# Drift Monitor tests
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn

def _small_model() -> nn.Module:
    return nn.Sequential(nn.Linear(8, 4), nn.Tanh(), nn.Linear(4, 2))


@test("drift monitor: initialises and reports stats")
def test_drift_init():
    monitor = DriftMonitor(
        checkpoint_dir="/tmp/test_drift_ckpt",
        log_path="/tmp/test_drift_log.json",
    )
    stats = monitor.stats()
    assert "snapshots" in stats
    assert "drift_threshold" in stats
    assert "rollback_threshold" in stats


@test("drift monitor: capture() saves snapshot and returns id")
def test_drift_capture():
    monitor = DriftMonitor(checkpoint_dir="/tmp/test_drift_ckpt", log_path="/tmp/test_drift_log.json")
    model = _small_model()
    snap_id = monitor.capture(model, label="test_capture")
    assert snap_id, "Expected non-empty snapshot ID"
    snap = monitor.get_snapshot(snap_id)
    assert snap is not None
    assert snap.label == "test_capture"
    assert len(snap.param_norms) > 0
    assert snap.total_params > 0


@test("drift monitor: capture_from_file() works with saved .pt")
def test_drift_capture_from_file():
    import tempfile
    model = _small_model()
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(model.state_dict(), f.name)
        tmp_path = f.name

    monitor = DriftMonitor(checkpoint_dir="/tmp/test_drift_ckpt", log_path="/tmp/test_drift_log.json")
    snap_id = monitor.capture_from_file(tmp_path, label="file_test")
    snap = monitor.get_snapshot(snap_id)
    assert snap is not None
    assert len(snap.param_norms) > 0


@test("drift monitor: measure() detects zero drift on same model")
def test_drift_measure_zero():
    monitor = DriftMonitor(checkpoint_dir="/tmp/test_drift_ckpt", log_path="/tmp/test_drift_log.json")
    model = _small_model()
    baseline_id = monitor.capture(model, label="baseline")
    metric = monitor.measure(model, baseline_id)
    # Same model, same weights → zero drift
    assert metric.overall_drift < 1e-4, f"Expected near-zero drift, got {metric.overall_drift}"
    assert metric.within_threshold
    assert not metric.alert
    assert not metric.force_rollback


@test("drift monitor: measure() detects high drift after perturbation")
def test_drift_measure_perturbed():
    monitor = DriftMonitor(
        checkpoint_dir="/tmp/test_drift_ckpt",
        log_path="/tmp/test_drift_log.json",
        drift_threshold=0.10,
        rollback_threshold=0.25,
    )
    model = _small_model()
    baseline_id = monitor.capture(model, label="pre_perturb")

    # Apply large perturbation
    with torch.no_grad():
        for p in model.parameters():
            p.data += torch.randn_like(p.data) * 2.0   # large noise

    metric = monitor.measure(model, baseline_id)
    assert metric.overall_drift > 0.0, "Expected non-zero drift after perturbation"


@test("drift monitor: measure_between() compares two snapshots")
def test_drift_measure_between():
    monitor = DriftMonitor(checkpoint_dir="/tmp/test_drift_ckpt", log_path="/tmp/test_drift_log.json")
    model_a = _small_model()
    model_b = _small_model()  # different init
    id_a = monitor.capture(model_a, label="snap_a")
    id_b = monitor.capture(model_b, label="snap_b")
    metric = monitor.measure_between(id_a, id_b)
    assert isinstance(metric.overall_drift, float)
    assert metric.baseline_id == id_a
    assert metric.current_id == id_b


@test("drift monitor: rollback() restores model weights")
def test_drift_rollback():
    monitor = DriftMonitor(checkpoint_dir="/tmp/test_drift_ckpt", log_path="/tmp/test_drift_log.json")
    model = _small_model()
    baseline_id = monitor.capture(model, label="rollback_baseline")

    # Get original first param value
    orig_val = next(model.parameters()).data[0, 0].item()

    # Perturb
    with torch.no_grad():
        for p in model.parameters():
            p.data.fill_(99.0)

    # Rollback
    monitor.rollback(model, baseline_id)
    restored_val = next(model.parameters()).data[0, 0].item()
    assert abs(restored_val - orig_val) < 1e-5, f"Expected {orig_val}, got {restored_val}"


@test("drift monitor: adaptation log records measurements")
def test_drift_log_records():
    monitor = DriftMonitor(checkpoint_dir="/tmp/test_drift_ckpt", log_path="/tmp/test_drift_log2.json")
    monitor.log.measurements.clear()
    model_a = _small_model()
    model_b = _small_model()
    id_a = monitor.capture(model_a, label="log_test_a")
    id_b = monitor.capture(model_b, label="log_test_b")
    monitor.measure_between(id_a, id_b)
    assert len(monitor.log.measurements) >= 1


@test("drift monitor: consolidate() blends adapted weights")
def test_drift_consolidate():
    monitor = DriftMonitor(checkpoint_dir="/tmp/test_drift_ckpt", log_path="/tmp/test_drift_log3.json")
    base    = _small_model()
    adapted = _small_model()

    # Capture baseline first
    baseline_id = monitor.capture(base, label="consol_base")

    # Get base param before consolidation
    p_before = next(base.parameters()).data.clone()

    new_id = monitor.consolidate(adapted, base, alpha=0.1, label="consolidated")
    assert new_id, "Expected non-empty consolidated snapshot ID"

    # Base weights should have changed (blended)
    p_after = next(base.parameters()).data
    assert not torch.allclose(p_before, p_after), "Expected weights to change after consolidation"


@test("drift monitor: list_snapshots returns snapshots sorted by timestamp")
def test_drift_list_snapshots():
    monitor = DriftMonitor(checkpoint_dir="/tmp/test_drift_ckpt", log_path="/tmp/test_drift_log.json")
    snaps = monitor.list_snapshots()
    assert isinstance(snaps, list)
    if len(snaps) >= 2:
        assert snaps[0].timestamp >= snaps[1].timestamp


# ---------------------------------------------------------------------------
# Distributed Pipeline tests
# ---------------------------------------------------------------------------

@test("distributed: MessageBus registers and routes requests")
def test_bus_register_route():
    bus = MessageBus()
    responses = []

    def handler(req: LayerRequest) -> LayerResponse:
        return LayerResponse(
            request_id=req.request_id,
            layer_type=req.layer_type,
            payload={"echo": req.payload.get("text", "")},
            source_node="test_node",
            success=True,
        )

    bus.register_node("test_node", handler)
    assert bus.is_registered("test_node")

    req = LayerRequest(
        request_id="req-001",
        layer_type=LayerType.TEXT_ENCODER,
        payload={"text": "hello"},
        source_node="client",
    )
    resp = bus.request("test_node", req)
    assert resp.success
    assert resp.payload["echo"] == "hello"


@test("distributed: MessageBus returns error for unknown node")
def test_bus_unknown_node():
    bus = MessageBus()
    req = LayerRequest(
        request_id="req-002",
        layer_type=LayerType.TEXT_ENCODER,
        payload={"text": "hello"},
        source_node="client",
    )
    resp = bus.request("nonexistent_node", req)
    assert not resp.success
    assert "not registered" in resp.error


@test("distributed: TextEncoderNode produces latent vector")
def test_text_encoder_node():
    bus = MessageBus()
    config = NodeConfig(node_id="enc-1", layer_type=LayerType.TEXT_ENCODER, endpoint="local")
    node = TextEncoderNode(config, bus)

    req = LayerRequest(
        request_id="req-003",
        layer_type=LayerType.TEXT_ENCODER,
        payload={"text": "Sales Invoice validation"},
        source_node="client",
    )
    resp = bus.request("enc-1", req)
    assert resp.success
    assert "latent" in resp.payload
    assert len(resp.payload["latent"]) == 64


@test("distributed: OutputRendererNode produces output string")
def test_output_renderer_node():
    bus = MessageBus()
    config = NodeConfig(node_id="rend-1", layer_type=LayerType.OUTPUT_RENDERER, endpoint="local")
    node = OutputRendererNode(config, bus)

    req = LayerRequest(
        request_id="req-004",
        layer_type=LayerType.OUTPUT_RENDERER,
        payload={"query": "What fields are required?", "facts": [{"f": "Customer"}]},
        source_node="client",
    )
    resp = bus.request("rend-1", req)
    assert resp.success
    assert "output" in resp.payload
    assert resp.payload["output"]


@test("distributed: PipelineTopology local_default has 3 stages")
def test_topology_local_default():
    topo = PipelineTopology.local_default()
    assert len(topo.stages) == 3
    layer_types = {s["layer_type"] for s in topo.stages}
    assert LayerType.TEXT_ENCODER.value in layer_types
    assert LayerType.NEUROSYMBOLIC.value in layer_types
    assert LayerType.OUTPUT_RENDERER.value in layer_types


@test("distributed: PipelineTopology serialises to JSON and back")
def test_topology_serialise():
    import json
    topo = PipelineTopology.local_default()
    serialised = topo.to_json()
    parsed = json.loads(serialised)
    assert "stages" in parsed
    assert len(parsed["stages"]) == 3


@test("distributed: DistributedPipeline initialises with local topology")
def test_distributed_pipeline_init():
    pipeline = DistributedPipeline(
        topology=PipelineTopology.local_default(),
        kg_path="data/graph/knowledge_graph.json",
        rules_path="data/rules/rules.json",
    )
    health = pipeline.health_check_all()
    assert len(health) == 3
    assert all(ok for ok in health.values())


@test("distributed: DistributedPipeline.run() returns payload with stage trace")
def test_distributed_pipeline_run():
    pipeline = DistributedPipeline(
        topology=PipelineTopology.local_default(),
        kg_path="data/graph/knowledge_graph.json",
        rules_path="data/rules/rules.json",
    )
    result = pipeline.run("What fields are required for Sales Invoice?")
    assert "_stage_trace" in result
    assert len(result["_stage_trace"]) == 3
    for stage in result["_stage_trace"]:
        assert "node" in stage
        assert "success" in stage
        assert "latency_ms" in stage


@test("distributed: DistributedPipeline collects latency metrics")
def test_distributed_pipeline_latency():
    pipeline = DistributedPipeline(
        topology=PipelineTopology.local_default(),
        kg_path="data/graph/knowledge_graph.json",
        rules_path="data/rules/rules.json",
    )
    pipeline.run("test latency query")
    report = pipeline.latency_report()
    assert len(report) == 3
    for node_id, stats in report.items():
        assert stats["calls"] >= 1


@test("distributed: DistributedPipeline.node_stats() returns stats list")
def test_distributed_pipeline_node_stats():
    pipeline = DistributedPipeline(
        topology=PipelineTopology.local_default(),
        kg_path="data/graph/knowledge_graph.json",
        rules_path="data/rules/rules.json",
    )
    stats = pipeline.node_stats()
    assert len(stats) == 3
    for s in stats:
        assert "node_id" in s
        assert "layer_type" in s
        assert "healthy" in s


@test("distributed: LayerRequest serialises and deserialises")
def test_layer_request_serde():
    req = LayerRequest(
        request_id="r-001",
        layer_type=LayerType.NEUROSYMBOLIC,
        payload={"text": "hello", "query_type": "retrieve"},
        source_node="pipeline",
    )
    d   = req.to_dict()
    req2 = LayerRequest.from_dict(d)
    assert req2.request_id == req.request_id
    assert req2.layer_type == req.layer_type
    assert req2.payload == req.payload


# ---------------------------------------------------------------------------
# ATLAS sub-agent tests
# ---------------------------------------------------------------------------

_atlas = None
def get_atlas():
    global _atlas
    if _atlas is None:
        _atlas = ATLASAgent(
            kg_path="data/graph/knowledge_graph.json",
            rules_path="data/rules/rules.json",
        )
    return _atlas


@test("ATLAS: can_handle scores infrastructure queries high")
def test_atlas_can_handle_high():
    agent = get_atlas()
    score = agent.can_handle("identify single points of failure in the pipeline topology")
    assert score >= 0.58, f"Expected ≥0.58, got {score}"


@test("ATLAS: can_handle scores business logic queries low")
def test_atlas_can_handle_low():
    agent = get_atlas()
    score = agent.can_handle("plan the submission steps for a Sales Invoice")
    assert score < 0.65, f"Expected <0.65, got {score}"


@test("ATLAS: health_report returns node entries")
def test_atlas_health_report():
    agent = get_atlas()
    report = agent.health_report()
    assert isinstance(report, list)
    assert len(report) >= 1
    for e in report:
        assert hasattr(e, "node_id")
        assert hasattr(e, "healthy")
        assert isinstance(e.healthy, bool)


@test("ATLAS: fault_risks identifies SPOFs")
def test_atlas_fault_risks():
    agent = get_atlas()
    risks = agent.fault_risks()
    assert isinstance(risks, list)
    assert len(risks) >= 1
    for r in risks:
        assert r.risk in ("CRITICAL", "HIGH", "MEDIUM")
        assert r.mitigation, "Expected non-empty mitigation"


@test("ATLAS: scaling_recommendations returns list sorted by priority")
def test_atlas_scaling():
    agent = get_atlas()
    recs = agent.scaling_recommendations(target_qps=50)
    assert isinstance(recs, list)
    assert len(recs) >= 1
    for r in recs:
        assert r.priority in ("HIGH", "MEDIUM", "LOW")
        assert r.recommended_replicas >= 1


@test("ATLAS: scaling_recommendations adjusts replicas for high QPS")
def test_atlas_scaling_high_qps():
    agent = get_atlas()
    recs_low  = agent.scaling_recommendations(target_qps=5)
    recs_high = agent.scaling_recommendations(target_qps=500)
    total_low  = sum(r.recommended_replicas for r in recs_low)
    total_high = sum(r.recommended_replicas for r in recs_high)
    assert total_high >= total_low, "Higher QPS should recommend more replicas"


@test("ATLAS: resource_estimates returns list with CPU + memory info")
def test_atlas_resource_estimates():
    agent = get_atlas()
    estimates = agent.resource_estimates()
    assert isinstance(estimates, list)
    assert len(estimates) >= 1
    for e in estimates:
        assert e.cpu_cores >= 1
        assert e.memory_mb >= 64
        assert isinstance(e.gpu_required, bool)


@test("ATLAS: generate_compose_yaml produces valid YAML string")
def test_atlas_compose_yaml():
    agent = get_atlas()
    yaml_out = agent.generate_compose_yaml()
    assert "version:" in yaml_out
    assert "services:" in yaml_out
    assert "synapseos/" in yaml_out


@test("ATLAS: generate_deployment_configs returns one config per stage")
def test_atlas_deployment_configs():
    agent = get_atlas()
    configs = agent.generate_deployment_configs()
    assert len(configs) == len(agent._topology.stages)
    for c in configs:
        assert c.image.startswith("synapseos/")
        assert c.replicas >= 1
        assert "KG_PATH" in c.env_vars


@test("ATLAS: handle() returns AgentResult with output")
def test_atlas_handle_basic():
    agent = get_atlas()
    result = agent.handle("run a full infrastructure audit")
    assert result.agent_name == "ATLAS"
    assert result.output, "Expected non-empty output"
    assert result.traceable
    assert result.latency_ms > 0


@test("ATLAS: handle() routes health query")
def test_atlas_handle_health():
    agent = get_atlas()
    result = agent.handle("what is the current health status of all nodes?")
    assert result.agent_name == "ATLAS"
    assert "healthy" in result.structured


@test("ATLAS: handle() routes fault tolerance query")
def test_atlas_handle_fault():
    agent = get_atlas()
    result = agent.handle("identify single points of failure and fault risks")
    assert result.agent_name == "ATLAS"
    assert "risks" in result.structured


@test("ATLAS: handle() routes scaling query")
def test_atlas_handle_scaling():
    agent = get_atlas()
    result = agent.handle("how should we scale for 100 concurrent requests?")
    assert result.agent_name == "ATLAS"
    assert "recommendations" in result.structured


@test("ATLAS: handle() routes deployment config query")
def test_atlas_handle_deploy():
    agent = get_atlas()
    result = agent.handle("generate a docker compose deployment configuration")
    assert result.agent_name == "ATLAS"
    assert "services" in result.structured
    assert "compose_yaml" in result.structured


@test("ATLAS: handle() routes resource estimation query")
def test_atlas_handle_resources():
    agent = get_atlas()
    result = agent.handle("estimate cpu and memory resource requirements")
    assert result.agent_name == "ATLAS"
    assert "estimates" in result.structured
    assert "total_cpu_cores" in result.structured


@test("ATLAS: confidence is in [0, 1]")
def test_atlas_confidence_range():
    agent = get_atlas()
    result = agent.handle("topology and scaling analysis")
    assert 0.0 <= result.confidence <= 1.0, f"Confidence out of range: {result.confidence}"


@test("controller: ATLAS loaded in agent router")
def test_controller_atlas_loaded():
    c = get_controller()
    agents = c.stats()["agents_loaded"]
    assert "ATLAS" in agents, f"ATLAS missing from {agents}"


@test("controller: infrastructure query routes to ATLAS")
def test_controller_atlas_routing():
    c = get_controller()
    result = c.think("identify single points of failure in the infrastructure topology")
    agent_names = [a.agent_name for a in result.agent_results]
    assert "ATLAS" in agent_names, f"Expected ATLAS in agent_results: {agent_names}"


# ---------------------------------------------------------------------------
# Voice interface tests
# ---------------------------------------------------------------------------

@test("voice: WhisperTranscriber initialises in stub mode")
def test_whisper_init_stub():
    t = WhisperTranscriber(model_size="tiny")
    # In CI/no-GPU environment, will be stub
    assert t is not None
    assert isinstance(t.is_real, bool)


@test("voice: WhisperTranscriber.transcribe_text() returns TranscriptionResult")
def test_whisper_transcribe_text():
    t = WhisperTranscriber()
    result = t.transcribe_text("What fields are required for a Sales Invoice?")
    assert isinstance(result, TranscriptionResult)
    assert result.text == "What fields are required for a Sales Invoice?"
    assert result.confidence == 1.0
    assert result.model_used == "text_passthrough"
    assert result.latency_ms == 0.0


@test("voice: WhisperTranscriber.transcribe() handles missing file gracefully")
def test_whisper_transcribe_missing():
    t = WhisperTranscriber()
    result = t.transcribe("/tmp/nonexistent_audio_file.wav")
    assert isinstance(result, TranscriptionResult)
    assert result.model_used == "stub"
    # Stub returns placeholder text, not an exception
    assert result.text, "Expected non-empty stub transcription"


@test("voice: WhisperTranscriber.transcribe() reads .txt sidecar if present")
def test_whisper_sidecar():
    import tempfile
    t = WhisperTranscriber()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio_path = f.name
    txt_path = audio_path.replace(".wav", ".txt")
    Path(txt_path).write_text("Sales Invoice required fields include customer and date.")
    try:
        result = t.transcribe(audio_path)
        assert "Sales Invoice" in result.text, f"Expected sidecar text, got: {result.text}"
        assert result.confidence == 1.0
    finally:
        Path(txt_path).unlink(missing_ok=True)


@test("voice: TTSSynthesizer initialises in stub mode")
def test_tts_init_stub():
    tts = TTSSynthesizer(output_dir="/tmp/test_tts_output")
    assert tts is not None
    assert isinstance(tts.is_real, bool)


@test("voice: TTSSynthesizer.synthesize() produces a result")
def test_tts_synthesize():
    tts = TTSSynthesizer(output_dir="/tmp/test_tts_output")
    result = tts.synthesize("The customer field is required.", filename="test_synth.wav")
    assert isinstance(result, SynthesisResult)
    assert result.text == "The customer field is required."
    assert result.chars == len("The customer field is required.")
    assert result.audio_path, "Expected non-empty audio path"
    assert result.latency_ms >= 0
    assert result.duration_seconds > 0


@test("voice: TTSSynthesizer stub writes .txt sidecar")
def test_tts_stub_writes_file():
    tts = TTSSynthesizer(output_dir="/tmp/test_tts_output")
    if not tts.is_real:
        result = tts.synthesize("Hello world.", filename="test_stub_out.wav")
        # Stub writes .txt
        assert Path(result.audio_path).exists(), f"Stub should write file at {result.audio_path}"


@test("voice: VoiceActivityDetector initialises")
def test_vad_init():
    vad = VoiceActivityDetector()
    assert vad is not None
    assert isinstance(vad.is_real, bool)


@test("voice: VoiceActivityDetector stub always returns True")
def test_vad_stub():
    vad = VoiceActivityDetector()
    if not vad.is_real:
        assert vad.is_speech(b"\x00" * 320) == True


@test("voice: VoicePipeline initialises without controller (lazy)")
def test_voice_pipeline_init():
    vp = VoicePipeline(
        kg_path="data/graph/knowledge_graph.json",
        rules_path="data/rules/rules.json",
    )
    assert vp is not None
    assert vp._controller is None, "Controller should be lazy-loaded"


@test("voice: VoicePipeline.process_text() returns VoiceRoundTrip")
def test_voice_pipeline_process_text():
    vp = VoicePipeline(
        kg_path="data/graph/knowledge_graph.json",
        rules_path="data/rules/rules.json",
    )
    result = vp.process_text("What fields are required for a Sales Invoice?")
    assert result is not None
    assert isinstance(result.total_latency_ms, float)
    assert result.total_latency_ms > 0
    assert result.transcription.text == "What fields are required for a Sales Invoice?"
    assert result.cognition_output, "Expected non-empty cognition output"
    assert result.synthesis.audio_path, "Expected synthesis output path"
    vp.shutdown()


@test("voice: VoiceRoundTrip.summary() returns non-empty string")
def test_voice_roundtrip_summary():
    vp = VoicePipeline(
        kg_path="data/graph/knowledge_graph.json",
        rules_path="data/rules/rules.json",
    )
    result = vp.process_text("list fields for Purchase Order")
    summary = result.summary()
    assert summary, "Expected non-empty summary"
    assert "Transcription" in summary
    assert "Synthesis" in summary
    assert "TOTAL" in summary
    vp.shutdown()


@test("voice: VoicePipeline.session_stats() tracks sessions")
def test_voice_pipeline_session_stats():
    vp = VoicePipeline(
        kg_path="data/graph/knowledge_graph.json",
        rules_path="data/rules/rules.json",
    )
    vp.process_text("Sales Invoice validation")
    vp.process_text("Purchase Order rules")
    stats = vp.session_stats()
    assert stats["sessions"] == 2
    assert "mean_latency_ms" in stats
    assert "within_target" in stats
    vp.shutdown()


@test("voice: VoicePipeline._trim_for_speech removes code blocks and annotations")
def test_voice_trim_for_speech():
    text = "Here is the answer.\n```python\ncode here\n```\n[Memory context: blah blah]"
    trimmed = VoicePipeline._trim_for_speech(text)
    assert "```" not in trimmed
    assert "Memory context" not in trimmed
    assert "Here is the answer" in trimmed


@test("voice: VoiceRoundTrip.to_dict() is JSON-serialisable")
def test_voice_roundtrip_to_dict():
    import json
    vp = VoicePipeline(
        kg_path="data/graph/knowledge_graph.json",
        rules_path="data/rules/rules.json",
    )
    result = vp.process_text("What is a Sales Invoice?")
    d = result.to_dict()
    serialised = json.dumps(d)
    assert serialised, "Expected non-empty JSON"
    parsed = json.loads(serialised)
    assert "transcription" in parsed
    assert "synthesis" in parsed
    assert "total_ms" in parsed
    vp.shutdown()


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
        # NOVA
        test_nova_can_handle_high,
        test_nova_can_handle_low,
        test_nova_rule_density,
        test_nova_kg_bottlenecks,
        test_nova_cache_recommendations,
        test_nova_redundancies,
        test_nova_benchmark,
        test_nova_handle_basic,
        test_nova_handle_cache,
        test_nova_handle_bottleneck,
        test_nova_handle_redundancy,
        test_nova_handle_benchmark,
        test_nova_confidence_range,
        test_controller_nova_loaded,
        test_controller_nova_routing,
        # ATLAS
        test_atlas_can_handle_high,
        test_atlas_can_handle_low,
        test_atlas_health_report,
        test_atlas_fault_risks,
        test_atlas_scaling,
        test_atlas_scaling_high_qps,
        test_atlas_resource_estimates,
        test_atlas_compose_yaml,
        test_atlas_deployment_configs,
        test_atlas_handle_basic,
        test_atlas_handle_health,
        test_atlas_handle_fault,
        test_atlas_handle_scaling,
        test_atlas_handle_deploy,
        test_atlas_handle_resources,
        test_atlas_confidence_range,
        test_controller_atlas_loaded,
        test_controller_atlas_routing,
        # Voice
        test_whisper_init_stub,
        test_whisper_transcribe_text,
        test_whisper_transcribe_missing,
        test_whisper_sidecar,
        test_tts_init_stub,
        test_tts_synthesize,
        test_tts_stub_writes_file,
        test_vad_init,
        test_vad_stub,
        test_voice_pipeline_init,
        test_voice_pipeline_process_text,
        test_voice_roundtrip_summary,
        test_voice_pipeline_session_stats,
        test_voice_trim_for_speech,
        test_voice_roundtrip_to_dict,
        # Drift Monitor
        test_drift_init,
        test_drift_capture,
        test_drift_capture_from_file,
        test_drift_measure_zero,
        test_drift_measure_perturbed,
        test_drift_measure_between,
        test_drift_rollback,
        test_drift_log_records,
        test_drift_consolidate,
        test_drift_list_snapshots,
        # Distributed Pipeline
        test_bus_register_route,
        test_bus_unknown_node,
        test_text_encoder_node,
        test_output_renderer_node,
        test_topology_local_default,
        test_topology_serialise,
        test_distributed_pipeline_init,
        test_distributed_pipeline_run,
        test_distributed_pipeline_latency,
        test_distributed_pipeline_node_stats,
        test_layer_request_serde,
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
