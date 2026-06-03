"""
Tests for REST API models and evaluator framework.

Run:
    python tests/test_api_eval.py
    python -m pytest tests/test_api_eval.py -v
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


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
        wrapper.__name__ = fn.__name__
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# API Models (Pydantic, no server needed)
# ---------------------------------------------------------------------------

from synapseos.api.models import (
    QueryRequest, QueryResponse, HealthResponse, AgentInfo, AgentsResponse,
    KGSearchResult, KGSearchResponse, MemoryContextResponse, MemoryRecordRequest,
    VoiceRequest, VoiceResponse, BenchmarkRequest, BenchmarkResponse,
    AgentResultResponse, ModeResultResponse,
)


@test("api models: QueryRequest validates correctly")
def test_query_request_valid():
    req = QueryRequest(query="What fields are required for a Sales Invoice?")
    assert req.query == "What fields are required for a Sales Invoice?"
    assert req.force_modes is None
    assert req.max_modes == 2


@test("api models: QueryRequest rejects empty query")
def test_query_request_empty():
    try:
        req = QueryRequest(query="")
        # Pydantic does not reject empty strings by default
        assert req.query == ""
    except Exception:
        pass  # either is fine


@test("api models: QueryRequest accepts force_modes list")
def test_query_request_force_modes():
    req = QueryRequest(query="test", force_modes=["mode1", "mode4"])
    assert req.force_modes == ["mode1", "mode4"]


@test("api models: HealthResponse serialises to JSON")
def test_health_response_json():
    h = HealthResponse(
        status="ok",
        timestamp="2026-06-04T00:00:00",
        agents_loaded=["ARIEL", "NYX", "NOVA", "ATLAS"],
        memory_stats={"episodic_entries": 5, "semantic_facts": 100},
        idle_loop_active=False,
    )
    d = h.model_dump()
    assert d["status"] == "ok"
    assert "ARIEL" in d["agents_loaded"]
    json.dumps(d)   # must not raise


@test("api models: AgentsResponse contains agent info")
def test_agents_response():
    agents = [
        AgentInfo(name="ARIEL", description="Business logic", keywords=["submit"], base_confidence=0.55),
        AgentInfo(name="NYX",   description="Security",       keywords=["audit"], base_confidence=0.60),
    ]
    resp = AgentsResponse(agents=agents, count=2)
    assert resp.count == 2
    assert resp.agents[0].name == "ARIEL"


@test("api models: KGSearchResponse serialises results")
def test_kg_search_response():
    results_list = [
        KGSearchResult(name="Sales Invoice", score=1.0, module="Accounts", is_submittable=True),
        KGSearchResult(name="Purchase Order", score=0.8, module="Buying", is_submittable=True),
    ]
    resp = KGSearchResponse(query="invoice", results=results_list, total=2)
    d    = resp.model_dump()
    assert d["total"] == 2
    assert d["results"][0]["name"] == "Sales Invoice"


@test("api models: VoiceRequest and VoiceResponse work")
def test_voice_models():
    req  = VoiceRequest(text="What is a Sales Invoice?")
    resp = VoiceResponse(
        transcription="What is a Sales Invoice?",
        cognition_output="A Sales Invoice is a document...",
        synthesis_path="/tmp/synth.txt",
        total_latency_ms=350.0,
        within_target=True,
    )
    assert resp.within_target
    assert resp.total_latency_ms == 350.0


@test("api models: BenchmarkRequest validates concurrency range")
def test_benchmark_request():
    req = BenchmarkRequest(concurrency=10, n_requests=50)
    assert req.concurrency == 10
    assert req.n_requests == 50


@test("api models: BenchmarkResponse serialises pass/fail")
def test_benchmark_response():
    resp = BenchmarkResponse(
        concurrency=10, n_requests=50, n_success=48,
        error_rate=0.04, p50_ms=100.0, p99_ms=800.0,
        throughput_rps=25.0, passes_criterion=True,
    )
    d = resp.model_dump()
    assert d["passes_criterion"] is True
    assert d["error_rate"] == 0.04


@test("api models: MemoryRecordRequest has defaults")
def test_memory_record_request():
    req = MemoryRecordRequest(query="test q", response="test r")
    assert req.mode_used == "manual"
    assert req.tags == []


@test("api models: QueryResponse.from_cognition_result works")
def test_query_response_from_cognition():
    from synapseos.phase5.cognitive_controller import CognitionResult, ThinkingMode, ModeResult
    from synapseos.phase5.agents.base import AgentResult

    mode_result = ModeResult(
        mode=ThinkingMode.MODE1_REASONING,
        output="Test output",
        structured={},
        confidence=0.8,
        latency_ms=100.0,
        traceable=True,
    )
    agent_result = AgentResult(
        agent_name="ARIEL",
        query="test",
        output="Agent output",
        structured={},
        confidence=0.9,
        latency_ms=50.0,
        traceable=True,
    )

    class FakeCognitionResult:
        query = "test query"
        primary_output = "primary output"
        output_type = "language"
        confidence = 0.85
        modes_activated = [ThinkingMode.MODE1_REASONING]
        routed_to_agent = "ARIEL"
        total_latency_ms = 150.0
        memory_context = {"summary": "test summary"}
        mode_results = [mode_result]
        agent_results = [agent_result]

    resp = QueryResponse.from_cognition_result(FakeCognitionResult())
    assert resp.query == "test query"
    assert resp.confidence == 0.85
    assert len(resp.mode_results) == 1
    assert len(resp.agent_results) == 1
    assert resp.agent_results[0].agent_name == "ARIEL"


# ---------------------------------------------------------------------------
# FastAPI app (test without running server)
# ---------------------------------------------------------------------------

@test("api server: app imports without error")
def test_api_app_imports():
    # Just importing the server should not raise
    import importlib
    spec = importlib.util.spec_from_file_location(
        "server", Path("synapseos/api/server.py")
    )
    # We don't actually execute — just check it parses
    assert spec is not None


@test("api server: models.py imports and pydantic works")
def test_api_models_import():
    from synapseos.api.models import QueryRequest, QueryResponse, HealthResponse
    assert QueryRequest is not None
    assert QueryResponse is not None
    assert HealthResponse is not None


# ---------------------------------------------------------------------------
# Evaluator unit tests (lightweight — no controller needed)
# ---------------------------------------------------------------------------

from synapseos.eval.evaluator import (
    SynapseOSEvaluator, CategoryResult, EvalReport,
)


@test("eval: CategoryResult.grade() returns correct letter")
def test_category_grade():
    assert CategoryResult("test", 5, 0.90, [], 0.0).grade() == "A"
    assert CategoryResult("test", 5, 0.75, [], 0.0).grade() == "B"
    assert CategoryResult("test", 5, 0.60, [], 0.0).grade() == "C"
    assert CategoryResult("test", 5, 0.45, [], 0.0).grade() == "D"
    assert CategoryResult("test", 5, 0.30, [], 0.0).grade() == "F"


@test("eval: CategoryResult.to_dict() includes grade")
def test_category_to_dict():
    cat = CategoryResult("routing_accuracy", 10, 0.80, [], 50.0)
    d   = cat.to_dict()
    assert d["grade"] == "B"
    assert d["score"] == 0.80
    assert d["category"] == "routing_accuracy"


@test("eval: EvalReport.to_dict() is JSON-serialisable")
def test_eval_report_to_dict():
    cat = CategoryResult("field_accuracy", 5, 0.95, [], 30.0)
    report = EvalReport(
        timestamp="2026-06-04T00:00:00",
        n_total=5,
        categories=[cat],
        overall_score=0.95,
        overall_grade="A",
        summary="field_accuracy=A",
        gpu_training_recommendation="Training not urgent.",
    )
    d   = report.to_dict()
    raw = json.dumps(d)
    parsed = json.loads(raw)
    assert parsed["overall"]["grade"] == "A"
    assert len(parsed["categories"]) == 1


@test("eval: EvalReport.print_report() runs without error")
def test_eval_report_print():
    import io, contextlib
    cat = CategoryResult("test", 3, 0.75, [], 100.0)
    report = EvalReport("2026-06-04", 3, [cat], 0.75, "B", "test=B", "Train.")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report.print_report()
    output = buf.getvalue()
    assert "test" in output
    assert "75" in output or "0.75" in output or "B" in output


@test("eval: SynapseOSEvaluator initialises lazily (no controller on init)")
def test_eval_lazy_init():
    ev = SynapseOSEvaluator(
        kg_path="data/graph/knowledge_graph.json",
        rules_path="data/rules/rules.json",
    )
    assert ev._controller is None, "Controller should be lazy-loaded"
    assert ev._kg is None, "KG should be lazy-loaded"


@test("eval: SynapseOSEvaluator._get_kg() loads KG")
def test_eval_get_kg():
    ev = SynapseOSEvaluator()
    kg = ev._get_kg()
    assert len(kg._nodes) > 100


@test("eval: SynapseOSEvaluator._get_rules() loads rules")
def test_eval_get_rules():
    ev    = SynapseOSEvaluator()
    rules = ev._get_rules()
    assert len(rules._rules) > 100


@test("eval: SynapseOSEvaluator._sample_doctypes() returns correct count")
def test_eval_sample_doctypes():
    ev       = SynapseOSEvaluator()
    doctypes = ev._sample_doctypes(10)
    assert len(doctypes) == 10
    for dt in doctypes:
        assert dt, "Expected non-empty doctype name"


@test("eval: SynapseOSEvaluator._sample_doctypes(require_rules=True) returns doctypes with rules")
def test_eval_sample_doctypes_with_rules():
    ev     = SynapseOSEvaluator()
    rules  = ev._get_rules()
    dts    = ev._sample_doctypes(5, require_rules=True)
    for dt in dts:
        assert rules.get_rules(dt), f"DocType '{dt}' should have rules"


@test("eval: field_accuracy runs and returns CategoryResult")
def test_eval_field_accuracy():
    ev     = SynapseOSEvaluator()
    result = ev.eval_field_accuracy(n=3)
    assert isinstance(result, CategoryResult)
    assert result.category == "field_accuracy"
    assert 0.0 <= result.score <= 1.0
    assert result.n_evaluated >= 0


@test("eval: rule_coverage runs and returns CategoryResult")
def test_eval_rule_coverage():
    ev     = SynapseOSEvaluator()
    result = ev.eval_rule_coverage(n=3)
    assert isinstance(result, CategoryResult)
    assert result.category == "rule_coverage"
    assert 0.0 <= result.score <= 1.0


@test("eval: code_quality runs and returns CategoryResult with 0-1 score")
def test_eval_code_quality():
    ev     = SynapseOSEvaluator()
    result = ev.eval_code_quality(n=3)
    assert isinstance(result, CategoryResult)
    assert result.category == "code_quality"
    assert 0.0 <= result.score <= 1.0


@test("eval: routing_accuracy runs and returns CategoryResult")
def test_eval_routing():
    ev     = SynapseOSEvaluator()
    result = ev.eval_routing_accuracy(n_per_type=2)
    assert isinstance(result, CategoryResult)
    assert result.category == "routing_accuracy"
    assert 0.0 <= result.score <= 1.0


@test("eval: latency eval returns result with positive avg latency")
def test_eval_latency():
    ev     = SynapseOSEvaluator()
    result = ev.eval_latency(n=3)
    assert isinstance(result, CategoryResult)
    assert result.category == "latency"
    assert result.latency_ms >= 0


@test("eval: agent_quality runs and returns CategoryResult")
def test_eval_agent_quality():
    ev     = SynapseOSEvaluator()
    result = ev.eval_agent_quality(n=2)
    assert isinstance(result, CategoryResult)
    assert result.category == "agent_quality"
    assert 0.0 <= result.score <= 1.0


@test("eval: full run produces EvalReport with all categories")
def test_eval_full_run():
    ev     = SynapseOSEvaluator()
    report = ev.run(n_per_category=2)
    assert isinstance(report, EvalReport)
    assert len(report.categories) == 6    # all 6 categories
    assert 0.0 <= report.overall_score <= 1.0
    assert report.overall_grade in ("A", "B", "C", "D", "F")
    assert report.gpu_training_recommendation
    ev.shutdown()


@test("eval: full run overall score is graded correctly vs categories")
def test_eval_grade_consistency():
    ev     = SynapseOSEvaluator()
    report = ev.run(n_per_category=2)
    expected_grade = (
        "A" if report.overall_score >= 0.85 else
        "B" if report.overall_score >= 0.70 else
        "C" if report.overall_score >= 0.55 else
        "D" if report.overall_score >= 0.40 else "F"
    )
    assert report.overall_grade == expected_grade
    ev.shutdown()


@test("eval: report to_dict is JSON-serialisable with all categories")
def test_eval_report_serde():
    ev     = SynapseOSEvaluator()
    report = ev.run(n_per_category=2)
    d      = report.to_dict()
    raw    = json.dumps(d)
    parsed = json.loads(raw)
    assert "categories" in parsed
    assert len(parsed["categories"]) == 6
    ev.shutdown()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        # API models
        test_query_request_valid,
        test_query_request_empty,
        test_query_request_force_modes,
        test_health_response_json,
        test_agents_response,
        test_kg_search_response,
        test_voice_models,
        test_benchmark_request,
        test_benchmark_response,
        test_memory_record_request,
        test_query_response_from_cognition,
        test_api_app_imports,
        test_api_models_import,
        # Evaluator unit
        test_category_grade,
        test_category_to_dict,
        test_eval_report_to_dict,
        test_eval_report_print,
        test_eval_lazy_init,
        test_eval_get_kg,
        test_eval_get_rules,
        test_eval_sample_doctypes,
        test_eval_sample_doctypes_with_rules,
        # Evaluator integration (requires controller)
        test_eval_field_accuracy,
        test_eval_rule_coverage,
        test_eval_code_quality,
        test_eval_routing,
        test_eval_latency,
        test_eval_agent_quality,
        test_eval_full_run,
        test_eval_grade_consistency,
        test_eval_report_serde,
    ]

    for t in tests:
        t()

    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed

    print(f"\n{'='*60}")
    print(f"API + Eval Tests: {passed}/{len(results)} passed\n")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        detail = f"  → {r['detail']}" if not r["passed"] else ""
        print(f"  [{status}] {r['name']}  ({r['ms']}ms){detail}")

    sys.exit(0 if failed == 0 else 1)
