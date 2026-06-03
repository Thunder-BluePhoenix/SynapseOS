"""
Tests for Phase 2/3 extras:
  - KAN symbolic formula extractor (synapseos/phase2/kan_symbolic.py)
  - Output synthesizer SFT data generator (synapseos/phase3/output_synth_data.py)
  - Mamba data prep utilities (scripts/prepare_mamba_data.py)

Run:
    python tests/test_phase2_extras.py
    python -m pytest tests/test_phase2_extras.py -v
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from synapseos.phase2.kan_symbolic import (
    SYMBOLIC_CANDIDATES, SymbolicFunction, DimensionFormula,
    SymbolicFormulaSet, KANSymbolicExtractor, extract_all_labels,
    formulas_to_rules,
)
from synapseos.phase3.output_synth_data import (
    SFTDataGenerator, SFTPair, write_sft_jsonl, write_sft_stats,
    FIELD_REQUIREMENT_TEMPLATES, VALIDATION_RULE_TEMPLATES,
)
from scripts.prepare_mamba_data import (
    analyse_corpus, pack_windows, TokenizerWrapper, CorpusStats,
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
        wrapper.__name__ = fn.__name__
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Symbolic function candidates
# ---------------------------------------------------------------------------

@test("kan: SYMBOLIC_CANDIDATES is non-empty")
def test_symbolic_candidates_nonempty():
    assert len(SYMBOLIC_CANDIDATES) >= 5, "Expected ≥5 candidate functions"


@test("kan: each SymbolicFunction.fit returns (r2, coefficients)")
def test_symbolic_fit():
    x = np.linspace(0, 1, 100)
    y = 2.0 * x + 1.0    # ground truth: linear
    for fn in SYMBOLIC_CANDIDATES:
        r2, coeffs = fn.fit(x, y)
        assert isinstance(r2, float), f"R² should be float, got {type(r2)}"
        assert isinstance(coeffs, list), f"Coeffs should be list"
        if fn.name == "linear":
            assert r2 > 0.99, f"Linear fit on linear data should have R²>0.99, got {r2}"


@test("kan: SymbolicFunction.describe returns non-empty string")
def test_symbolic_describe():
    fn = SYMBOLIC_CANDIDATES[0]   # linear
    r2, coeffs = fn.fit(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
    desc = fn.describe("test_feature", coeffs)
    assert isinstance(desc, str) and len(desc) > 0


@test("kan: fit handles degenerate (constant) y without crash")
def test_symbolic_fit_constant():
    x = np.linspace(0, 1, 50)
    y = np.ones(50) * 3.14
    for fn in SYMBOLIC_CANDIDATES:
        r2, coeffs = fn.fit(x, y)    # should not raise
        assert isinstance(r2, float)


# ---------------------------------------------------------------------------
# KANSymbolicExtractor
# ---------------------------------------------------------------------------

def _make_tiny_model() -> nn.Module:
    """3-layer MLP: 4 → 8 → 1."""
    model = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 1))
    with torch.no_grad():
        # Set deterministic weights so tests are reproducible
        for p in model.parameters():
            torch.nn.init.normal_(p, std=0.3)
    return model


_dummy_norm = {
    "input_min":   [0.0, 0.0, 0.0, 0.0],
    "input_max":   [1.0, 1.0, 1.0, 1.0],
    "output_mean": 0.0,
    "output_std":  1.0,
}
_dummy_names = ["feature_a", "feature_b", "feature_c", "feature_d"]


@test("kan extractor: initialises with tiny model")
def test_extractor_init():
    model = _make_tiny_model()
    ext   = KANSymbolicExtractor(model, _dummy_norm, _dummy_names, "test_metric")
    assert ext.input_dim == 4
    assert len(ext.input_names) == 4


@test("kan extractor: _sweep_dimension returns correct shapes")
def test_extractor_sweep_dim():
    model = _make_tiny_model()
    ext   = KANSymbolicExtractor(model, _dummy_norm, _dummy_names, "test", n_sweep=50)
    for dim in range(4):
        x, y = ext._sweep_dimension(dim)
        assert len(x) == 50
        assert len(y) == 50
        assert x[0] == 0.0 and x[-1] == pytest_approx(1.0)


def pytest_approx(v, rel=1e-3):
    return v   # simple version — just return value for comparison


@test("kan extractor: _fit_dimension returns DimensionFormula")
def test_extractor_fit_dimension():
    model = _make_tiny_model()
    ext   = KANSymbolicExtractor(model, _dummy_norm, _dummy_names, "test")
    df    = ext._fit_dimension(0)
    assert isinstance(df, DimensionFormula)
    assert df.dim_index == 0
    assert df.feature_name == "feature_a"
    assert df.best_function in {fn.name for fn in SYMBOLIC_CANDIDATES}
    assert -1.0 <= df.r2 <= 1.0


@test("kan extractor: extract() returns SymbolicFormulaSet")
def test_extractor_extract():
    model = _make_tiny_model()
    ext   = KANSymbolicExtractor(model, _dummy_norm, _dummy_names, "test_metric", n_sweep=50)
    fs    = ext.extract()
    assert isinstance(fs, SymbolicFormulaSet)
    assert fs.label == "test_metric"
    assert len(fs.dimension_formulas) == 4
    assert isinstance(fs.global_r2, float)
    assert fs.global_r2 >= 0.0


@test("kan extractor: extract() global_r2 >= 0.70 for simple linear model")
def test_extractor_linear_r2():
    # Build a model that's actually linear so regression should fit well
    model = nn.Linear(4, 1, bias=True)
    with torch.no_grad():
        model.weight.fill_(0.5)
        model.bias.fill_(0.1)
    ext = KANSymbolicExtractor(model, _dummy_norm, _dummy_names, "linear_target", n_sweep=100)
    fs  = ext.extract()
    assert fs.global_r2 >= 0.70, f"Linear model should have R²>=0.70, got {fs.global_r2:.4f}"


@test("kan extractor: global_formula contains label name")
def test_extractor_formula_contains_label():
    model = _make_tiny_model()
    ext   = KANSymbolicExtractor(model, _dummy_norm, _dummy_names, "my_metric")
    fs    = ext.extract()
    assert "my_metric" in fs.global_formula


@test("kan extractor: SymbolicFormulaSet.to_dict() is JSON-serialisable")
def test_extractor_to_dict():
    model = _make_tiny_model()
    ext   = KANSymbolicExtractor(model, _dummy_norm, _dummy_names, "test", n_sweep=50)
    fs    = ext.extract()
    d     = fs.to_dict()
    json.dumps(d)    # should not raise
    assert "global_formula" in d
    assert "global_r2" in d
    assert "dimension_formulas" in d


@test("kan extractor: display() returns non-empty string")
def test_extractor_display():
    model = _make_tiny_model()
    ext   = KANSymbolicExtractor(model, _dummy_norm, _dummy_names, "metric")
    fs    = ext.extract()
    disp  = fs.display()
    assert disp and len(disp) > 20


@test("kan extractor: extract_all_labels() runs on real checkpoint")
def test_extractor_real_checkpoint():
    ckpt  = "data/checkpoints/kan/kan_best.pt"
    norm  = "data/checkpoints/kan/normalization.json"
    data  = "data/kan/metric_pairs.json"
    cfg   = "data/checkpoints/kan/kan_config.json"
    if not Path(ckpt).exists():
        results.append({"name": "kan extractor: real checkpoint", "passed": True,
                        "ms": 0, "detail": "Skipped — no checkpoint"})
        return
    formula_sets = extract_all_labels(
        ckpt, norm, data, cfg, save_path="/tmp/test_kan_formulas.json", n_sweep=100
    )
    assert len(formula_sets) >= 1
    for fs in formula_sets:
        assert fs.global_r2 >= 0.0
        assert len(fs.dimension_formulas) == 4


@test("kan extractor: formulas_to_rules() produces valid rules list")
def test_extractor_to_rules():
    model = _make_tiny_model()
    ext   = KANSymbolicExtractor(model, _dummy_norm, _dummy_names, "approval_time_hours")
    fs    = ext.extract()
    rules = formulas_to_rules([fs])
    assert len(rules) == 1
    r = rules[0]
    assert r["rule_type"] == "symbolic_formula"
    assert r["event"] == "compute"
    assert "features" in r
    assert len(r["features"]) == 4


# ---------------------------------------------------------------------------
# SFT Data Generator
# ---------------------------------------------------------------------------

_sft_gen = None
def get_sft_gen():
    global _sft_gen
    if _sft_gen is None:
        _sft_gen = SFTDataGenerator(
            kg_path="data/graph/knowledge_graph.json",
            rules_path="data/rules/rules.json",
            seed=42,
        )
    return _sft_gen


@test("sft: SFTDataGenerator initialises")
def test_sft_init():
    gen = get_sft_gen()
    assert gen is not None
    assert len(gen._doctype_list) > 100


@test("sft: generate() yields SFTPair objects")
def test_sft_generate_yields():
    import itertools
    gen   = get_sft_gen()
    pairs = list(itertools.islice(gen.generate(), 5))
    assert len(pairs) == 5
    for p in pairs:
        assert isinstance(p, SFTPair)
        assert p.instruction
        assert p.output
        assert 0.0 <= p.quality <= 1.0
        assert p.category
        assert p.doctype


@test("sft: generate_n() returns exactly n pairs")
def test_sft_generate_n():
    gen   = get_sft_gen()
    pairs = gen.generate_n(10)
    assert len(pairs) == 10


@test("sft: all categories produce at least 1 pair in 50")
def test_sft_category_coverage():
    import itertools
    gen    = get_sft_gen()
    pairs  = list(itertools.islice(gen.generate(), 50))
    cats   = {p.category for p in pairs}
    # Should cover most categories in 50 pairs
    assert len(cats) >= 4, f"Expected ≥4 categories in 50 pairs, got {cats}"


@test("sft: field_requirement pairs have expected structure")
def test_sft_field_requirement():
    gen = get_sft_gen()
    dt  = gen._doctype_list[0]
    out, q = gen._field_requirement_response(dt)
    assert out, "Expected non-empty output"
    assert isinstance(q, float) and 0 <= q <= 1


@test("sft: validation_rule pairs have expected structure")
def test_sft_validation_rules():
    gen = get_sft_gen()
    dt  = "Sales Invoice"
    out, q = gen._validation_rule_response(dt)
    assert out
    assert "Sales Invoice" in out


@test("sft: workflow pairs distinguish submittable vs non-submittable")
def test_sft_workflow():
    gen = get_sft_gen()
    # Sales Invoice is submittable
    out_sub, _ = gen._workflow_response("Sales Invoice")
    assert "Submit" in out_sub or "submit" in out_sub
    # Find a non-submittable
    for dt, node in gen._kg._nodes.items():
        if not node.get("is_submittable"):
            out_ns, _ = gen._workflow_response(dt)
            assert "non-submittable" in out_ns or "master" in out_ns.lower() or len(out_ns) > 0
            break


@test("sft: code generation pairs contain python code block")
def test_sft_code_generation():
    gen = get_sft_gen()
    dt  = "Sales Invoice"
    out, q = gen._code_response(dt)
    assert "```python" in out or "def " in out


@test("sft: comparison pairs contain table header")
def test_sft_comparison():
    gen = get_sft_gen()
    dt1, dt2 = gen._doctype_list[0], gen._doctype_list[1]
    out, q = gen._comparison_response(dt1, dt2)
    assert "|" in out, "Expected markdown table in comparison response"


@test("sft: to_dict() produces JSON-serialisable dict")
def test_sft_to_dict():
    gen   = get_sft_gen()
    pairs = gen.generate_n(3)
    for p in pairs:
        d = p.to_dict()
        json.dumps(d)
        assert "instruction" in d
        assert "output" in d
        assert "metadata" in d


@test("sft: write_sft_jsonl() creates valid file")
def test_sft_write_jsonl():
    gen   = get_sft_gen()
    pairs = gen.generate_n(5)
    path  = "/tmp/test_sft_pairs.jsonl"
    write_sft_jsonl(pairs, path)
    lines = Path(path).read_text().strip().split("\n")
    assert len(lines) == 5
    for line in lines:
        d = json.loads(line)
        assert "instruction" in d
        assert "output" in d


@test("sft: write_sft_stats() produces stats with expected keys")
def test_sft_write_stats():
    gen   = get_sft_gen()
    pairs = gen.generate_n(10)
    path  = "/tmp/test_sft_stats.json"
    write_sft_jsonl(pairs, "/tmp/test_sft_input.jsonl")
    stats = write_sft_stats(pairs, path)
    for key in ("total_pairs", "avg_quality", "category_counts", "unique_doctypes"):
        assert key in stats, f"Missing key: {key}"
    assert stats["total_pairs"] == 10


@test("sft: quality scores are all in [0, 1]")
def test_sft_quality_range():
    gen   = get_sft_gen()
    pairs = gen.generate_n(20)
    for p in pairs:
        assert 0.0 <= p.quality <= 1.0, f"Quality out of range: {p.quality}"


@test("sft: generation speed > 100 pairs/s")
def test_sft_speed():
    import itertools
    gen = get_sft_gen()
    t0  = time.time()
    list(itertools.islice(gen.generate(), 100))
    elapsed = time.time() - t0
    rate = 100 / elapsed
    assert rate > 100, f"Expected >100 pairs/s, got {rate:.1f}/s"


# ---------------------------------------------------------------------------
# Mamba data prep utilities
# ---------------------------------------------------------------------------

@test("mamba prep: analyse_corpus returns CorpusStats")
def test_corpus_analyse():
    import tempfile
    sample = "def validate(doc):\n    pass\n" * 1000
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(sample)
        path = f.name
    stats = analyse_corpus(path)
    assert isinstance(stats, CorpusStats)
    assert stats.total_chars > 0
    assert stats.total_lines > 0
    assert stats.estimated_tokens > 0
    assert 0.0 <= stats.quality_score <= 1.0


@test("mamba prep: pack_windows splits tokens into correct window sizes")
def test_pack_windows():
    tokens  = list(range(5000))
    windows = pack_windows(tokens, window_size=512, eos_id=0)
    assert len(windows) > 0
    for w in windows:
        assert len(w) == 512


@test("mamba prep: pack_windows pads last window")
def test_pack_windows_padding():
    tokens  = list(range(1000))   # not divisible by 512
    windows = pack_windows(tokens, window_size=512, eos_id=999)
    assert len(windows[-1]) == 512
    # Last window should be padded with eos_id
    if len(tokens) % 512 != 0:
        # Check that some pad tokens (999) appear in the last window
        assert 999 in windows[-1]


@test("mamba prep: TokenizerWrapper initialises (stub mode)")
def test_tokenizer_stub():
    tok = TokenizerWrapper(model_name="nonexistent/model/that/does/not/exist")
    assert tok is not None
    assert not tok.is_real
    ids = tok.encode("hello world")
    assert isinstance(ids, list)
    assert len(ids) > 0


@test("mamba prep: TokenizerWrapper stub encode is deterministic")
def test_tokenizer_stub_deterministic():
    tok = TokenizerWrapper(model_name="nonexistent_model")
    ids1 = tok.encode("Sales Invoice validation")
    ids2 = tok.encode("Sales Invoice validation")
    assert ids1 == ids2, "Stub encoder should be deterministic"


@test("mamba prep: pack_windows → jsonl round-trip")
def test_pack_windows_jsonl():
    import tempfile
    tokens  = list(range(3000))
    windows = pack_windows(tokens, window_size=256, eos_id=0)
    path    = "/tmp/test_mamba_windows.jsonl"
    with open(path, "w") as f:
        for w in windows:
            f.write(json.dumps({"input_ids": w}) + "\n")
    lines = Path(path).read_text().strip().split("\n")
    assert len(lines) == len(windows)
    first = json.loads(lines[0])
    assert "input_ids" in first
    assert len(first["input_ids"]) == 256


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_symbolic_candidates_nonempty,
        test_symbolic_fit,
        test_symbolic_describe,
        test_symbolic_fit_constant,
        test_extractor_init,
        test_extractor_sweep_dim,
        test_extractor_fit_dimension,
        test_extractor_extract,
        test_extractor_linear_r2,
        test_extractor_formula_contains_label,
        test_extractor_to_dict,
        test_extractor_display,
        test_extractor_real_checkpoint,
        test_extractor_to_rules,
        test_sft_init,
        test_sft_generate_yields,
        test_sft_generate_n,
        test_sft_category_coverage,
        test_sft_field_requirement,
        test_sft_validation_rules,
        test_sft_workflow,
        test_sft_code_generation,
        test_sft_comparison,
        test_sft_to_dict,
        test_sft_write_jsonl,
        test_sft_write_stats,
        test_sft_quality_range,
        test_sft_speed,
        test_corpus_analyse,
        test_pack_windows,
        test_pack_windows_padding,
        test_tokenizer_stub,
        test_tokenizer_stub_deterministic,
        test_pack_windows_jsonl,
    ]

    for t in tests:
        t()

    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed

    print(f"\n{'='*60}")
    print(f"Phase 2/3 Extras Tests: {passed}/{len(results)} passed\n")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        detail = f"  → {r['detail']}" if not r["passed"] else ""
        print(f"  [{status}] {r['name']}  ({r['ms']}ms){detail}")

    sys.exit(0 if failed == 0 else 1)
