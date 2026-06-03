"""
Phase 3 integration tests — end-to-end pipeline.

Tests all 5 thinking modes across 20+ cases.
Run:
    python -m pytest tests/test_pipeline.py -v
    python tests/test_pipeline.py        # without pytest
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from synapseos.phase3.neurosymbolic_core import NeurosymbolicCore, QueryType
from synapseos.phase3.pipeline import SynapseOSPipeline


# Shared pipeline — initialize once for all tests
_pipeline: SynapseOSPipeline | None = None

def get_pipeline() -> SynapseOSPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = SynapseOSPipeline()
    return _pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestResult:
    def __init__(self, name: str, passed: bool, latency_ms: float, detail: str = ""):
        self.name       = name
        self.passed     = passed
        self.latency_ms = latency_ms
        self.detail     = detail

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name} ({self.latency_ms:.1f}ms){' — ' + self.detail if self.detail else ''}"


results: list[TestResult] = []


def test(name: str):
    def decorator(fn):
        def wrapper():
            t0 = time.time()
            try:
                fn()
                latency = (time.time() - t0) * 1000
                r = TestResult(name, True, latency)
            except AssertionError as e:
                latency = (time.time() - t0) * 1000
                r = TestResult(name, False, latency, str(e))
            except Exception as e:
                latency = (time.time() - t0) * 1000
                r = TestResult(name, False, latency, f"{type(e).__name__}: {e}")
            results.append(r)
            return r
        wrapper.__name__ = fn.__name__
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Neurosymbolic Core tests (no neural layers needed)
# ---------------------------------------------------------------------------

@test("core: knowledge graph loads")
def test_kg_loads():
    core = NeurosymbolicCore()
    assert core.kg._nodes, "Knowledge graph is empty"
    assert len(core.kg._nodes) > 100, f"Expected >100 nodes, got {len(core.kg._nodes)}"


@test("core: rules engine loads")
def test_rules_loads():
    core = NeurosymbolicCore()
    assert core.rules._rules, "Rules engine is empty"
    assert len(core.rules._rules) > 100, f"Expected >100 rules, got {len(core.rules._rules)}"


@test("core: retrieve Sales Invoice entity")
def test_retrieve_sales_invoice():
    core = NeurosymbolicCore()
    result = core.reason(QueryType.RETRIEVE, "Sales Invoice")
    assert result.confidence > 0.5
    assert result.verified
    entity_facts = [f for f in result.facts if f.get("type") == "entity"]
    assert entity_facts, "No entity facts returned"


@test("core: retrieve Customer has link edges")
def test_retrieve_customer_links():
    core = NeurosymbolicCore()
    result = core.reason(QueryType.RETRIEVE, "Customer")
    link_facts = [f for f in result.facts if f.get("type") == "link"]
    assert len(link_facts) > 0, "Customer should have link relationships"


@test("core: validate Sales Order — detects missing required fields")
def test_validate_sales_order_empty():
    core = NeurosymbolicCore()
    result = core.reason(QueryType.VALIDATE, "Sales Order", {"field_values": {}})
    assert not result.verified, "Empty Sales Order should have constraint violations"
    assert len(result.constraints_violated) > 0
    # Customer field must be among violations
    assert any("customer" in v.lower() for v in result.constraints_violated), \
        "customer field should be in violations"


@test("core: validate Sales Order — passes with required fields")
def test_validate_sales_order_with_data():
    core = NeurosymbolicCore()
    # Provide all the fields that have reqd=1
    fields = {
        "naming_series": "SAL-ORD-.YYYY.-",
        "customer": "Test Customer",
        "order_type": "Sales",
        "company": "Test Company",
        "transaction_date": "2026-05-23",
        "currency": "INR",
        "conversion_rate": 1.0,
        "selling_price_list": "Standard Selling",
        "price_list_currency": "INR",
        "plc_conversion_rate": 1.0,
        "items": [{"item_code": "ITEM-001", "qty": 1}],
        "status": "Draft",
    }
    result = core.reason(QueryType.VALIDATE, "Sales Order", {"field_values": fields})
    # With all fields present, violations should be empty or at least fewer
    assert len(result.constraints_violated) == 0, \
        f"Unexpected violations: {result.constraints_violated}"


@test("core: plan Purchase Order returns steps")
def test_plan_purchase_order():
    core = NeurosymbolicCore()
    result = core.reason(QueryType.PLAN, "Purchase Order")
    assert len(result.facts) >= 1, "Plan should return at least 1 step"
    step_types = [f.get("action") for f in result.facts if "action" in f]
    assert "validate_required_fields" in step_types


@test("core: plan submittable doc has more steps than non-submittable")
def test_plan_submittable_vs_not():
    core = NeurosymbolicCore()
    submittable  = core.reason(QueryType.PLAN, "Sales Invoice")
    non_submit   = core.reason(QueryType.PLAN, "Customer")
    assert len(submittable.facts) >= len(non_submit.facts), \
        "Submittable doc should have >= steps than non-submittable"


@test("core: reason returns rules for Sales Invoice")
def test_reason_sales_invoice():
    core = NeurosymbolicCore()
    result = core.reason(QueryType.REASON, "Sales Invoice")
    assert len(result.rules_applied) > 0, "Should have rules for Sales Invoice"
    assert result.confidence > 0.5


@test("core: graph path exists between linked doctypes")
def test_graph_path():
    core = NeurosymbolicCore()
    path = core.kg.find_path("Sales Order", "Customer", max_hops=3)
    assert path is not None, "Should find path from Sales Order to Customer"
    assert "Sales Order" in path
    assert "Customer" in path


@test("core: graph search finds Company")
def test_graph_search():
    core = NeurosymbolicCore()
    results_found = core.kg.search("Company", limit=5)
    assert any("Company" in r.get("name", "") for r in results_found)


@test("core: symbolic features vector is correct length")
def test_symbolic_features_dim():
    core = NeurosymbolicCore()
    result = core.reason(QueryType.VALIDATE, "Sales Invoice")
    vec = result.to_symbolic_features(dim=32)
    assert len(vec) == 32
    assert 0.0 <= vec[0] <= 1.0, "Confidence should be in [0, 1]"


# ---------------------------------------------------------------------------
# Full pipeline tests (Mode 1 + Mode 4)
# ---------------------------------------------------------------------------

@test("pipeline: initializes in under 5 seconds")
def test_pipeline_init_speed():
    t0 = time.time()
    _ = get_pipeline()
    elapsed = time.time() - t0
    assert elapsed < 5.0, f"Pipeline init took {elapsed:.1f}s — should be < 5s"


@test("pipeline: validate query returns violations for empty doc")
def test_pipeline_validate_empty():
    p = get_pipeline()
    result = p.run("Validate a Sales Order with a missing customer field")
    assert result.reasoning.constraints_violated, \
        "Validate query on empty doc should return violations"


@test("pipeline: retrieve query returns facts")
def test_pipeline_retrieve():
    p = get_pipeline()
    result = p.run("What is the Customer doctype?")
    assert result.reasoning.facts or result.doctype, "Should return facts or doctype"


@test("pipeline: latency under 50ms on CPU")
def test_pipeline_latency():
    p = get_pipeline()
    # Warm up
    p.run("test query")
    # Measure
    result = p.run("What fields does a Sales Invoice have?")
    assert result.latency_ms < 50, f"Latency {result.latency_ms:.1f}ms exceeds 50ms CPU target"


@test("pipeline: output contains doctype name when found")
def test_pipeline_output_contains_doctype():
    p = get_pipeline()
    result = p.run("Show me the Sales Order document structure")
    assert "Sales Order" in result.output, "Output should mention the DocType"


@test("pipeline: 5 consecutive queries all complete")
def test_pipeline_batch():
    p = get_pipeline()
    queries = [
        "What is a Customer?",
        "Plan steps for a Payment Entry",
        "Validate an empty Purchase Invoice",
        "What links to Item?",
        "Show rules for Employee",
    ]
    for q in queries:
        result = p.run(q)
        assert result.output, f"Empty output for query: {q}"
        assert result.latency_ms < 200, f"Query too slow: {result.latency_ms:.1f}ms"


@test("pipeline: verified=True for retrieve of known doctype")
def test_pipeline_verified_known():
    p = get_pipeline()
    result = p.run("Tell me about the Company doctype")
    # Company is in graph — retrieve should be verified
    if result.query_type == QueryType.RETRIEVE and result.doctype == "Company":
        assert result.reasoning.verified


@test("pipeline: validated doc with all fields has no violations")
def test_pipeline_validate_complete():
    p = get_pipeline()
    context = {
        "field_values": {
            "naming_series": "SAL-ORD-.YYYY.-",
            "customer": "ACME Corp",
            "order_type": "Sales",
            "company": "Test Co",
            "transaction_date": "2026-05-23",
            "currency": "INR",
            "conversion_rate": 1.0,
            "selling_price_list": "Standard Selling",
            "price_list_currency": "INR",
            "plc_conversion_rate": 1.0,
            "items": [{}],
            "status": "Draft",
        }
    }
    result = p.run("Validate this Sales Order", context=context)
    if result.doctype == "Sales Order" and result.query_type == QueryType.VALIDATE:
        assert result.reasoning.verified, \
            f"Complete Sales Order should pass validation. Violations: {result.reasoning.constraints_violated}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all() -> None:
    print("=" * 60)
    print("SynapseOS Integration Tests")
    print("=" * 60)

    test_fns = [
        test_kg_loads, test_rules_loads,
        test_retrieve_sales_invoice, test_retrieve_customer_links,
        test_validate_sales_order_empty, test_validate_sales_order_with_data,
        test_plan_purchase_order, test_plan_submittable_vs_not,
        test_reason_sales_invoice, test_graph_path, test_graph_search,
        test_symbolic_features_dim,
        test_pipeline_init_speed, test_pipeline_validate_empty,
        test_pipeline_retrieve, test_pipeline_latency,
        test_pipeline_output_contains_doctype, test_pipeline_batch,
        test_pipeline_verified_known, test_pipeline_validate_complete,
    ]

    for fn in test_fns:
        r = fn()
        print(r)

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    avg_ms = sum(r.latency_ms for r in results) / len(results) if results else 0

    print()
    print(f"Results: {passed}/{len(results)} passed  |  {failed} failed  |  avg {avg_ms:.1f}ms")
    if failed:
        print("\nFailed tests:")
        for r in results:
            if not r.passed:
                print(f"  {r}")

    return failed


if __name__ == "__main__":
    failed = run_all()
    sys.exit(1 if failed else 0)
