"""
Tests for the SynapseOS World Model — the persistent cognitive substrate.

Covers:
  - WorldState persistence + versioning
  - ObservationEncoder determinism + similarity structure
  - WorldModelNet state evolution + prediction
  - The perceive() loop: surprise gating, novelty→familiar→expected
  - Predictive memory: temporal/causal links, salience eviction
  - State-conditioned retrieval
  - Consolidation: pattern distillation → semantic memory + predictor improvement

Run:
    python tests/test_world_model.py
    python -m pytest tests/test_world_model.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from synapseos.world_model.state import WorldState
from synapseos.world_model.encoder import ObservationEncoder, cosine_distance
from synapseos.world_model.world_model import WorldModel, WorldModelNet, PerceptionResult
from synapseos.world_model.predictive_memory import PredictiveMemory, Episode
from synapseos.world_model.retrieval import StateConditionedRetrieval
from synapseos.world_model.consolidation import Consolidator, ConsolidationReport


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


def _tmp_wm(**kw) -> WorldModel:
    d = tempfile.mkdtemp(prefix="wm_test_")
    return WorldModel(
        state_path=os.path.join(d, "state.json"),
        episodes_path=os.path.join(d, "episodes.json"),
        snapshot_dir=os.path.join(d, "snaps"),
        **kw,
    )


# ---------------------------------------------------------------------------
# WorldState
# ---------------------------------------------------------------------------

@test("state: fresh creates correct dims")
def test_state_fresh():
    s = WorldState.fresh(state_dim=64, obs_dim=128)
    assert len(s.z) == 64
    assert len(s.prediction) == 128
    assert s.step == 0


@test("state: set_z and z_tensor round-trip")
def test_state_set_z():
    s = WorldState.fresh(8, 16)
    z = torch.randn(8)
    s.set_z(z)
    assert torch.allclose(s.z_tensor(), z, atol=1e-5)


@test("state: surprise EMA updates")
def test_state_surprise_ema():
    s = WorldState.fresh(4, 4)
    s.update_surprise(1.0, alpha=0.5)
    s.update_surprise(0.0, alpha=0.5)
    assert 0.0 < s.surprise_ema < 1.0


@test("state: save and load round-trip")
def test_state_save_load():
    d = tempfile.mkdtemp()
    path = os.path.join(d, "ws.json")
    s = WorldState.fresh(8, 16)
    s.step = 42
    s.set_z(torch.ones(8))
    s.save(path)
    s2 = WorldState.load(path, 8, 16)
    assert s2.step == 42
    assert len(s2.z) == 8


@test("state: load returns fresh when file absent")
def test_state_load_fresh():
    s = WorldState.load("/tmp/nonexistent_ws_xyz.json", 32, 64)
    assert s.step == 0
    assert len(s.z) == 32


@test("state: snapshot and restore for rollback")
def test_state_snapshot_restore():
    d = tempfile.mkdtemp()
    s = WorldState.fresh(8, 8)
    s.step = 10
    s.set_z(torch.ones(8) * 5)
    snap_id = s.snapshot(d)
    # mutate
    s.step = 999
    s.set_z(torch.zeros(8))
    restored = WorldState.restore(d, snap_id)
    assert restored.step == 10
    assert abs(restored.z[0] - 5.0) < 1e-5


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

@test("encoder: produces correct dim, L2-normalised")
def test_encoder_dim():
    enc = ObservationEncoder(obs_dim=128)
    v = enc.encode("Sales Invoice")
    assert v.shape == (128,)
    assert abs(v.norm(2).item() - 1.0) < 1e-5


@test("encoder: deterministic — same text same vector")
def test_encoder_deterministic():
    enc = ObservationEncoder(obs_dim=64)
    a = enc.encode("Purchase Order validation")
    b = enc.encode("Purchase Order validation")
    assert torch.allclose(a, b)


@test("encoder: similar texts closer than unrelated")
def test_encoder_similarity():
    enc = ObservationEncoder(obs_dim=128)
    base    = enc.encode("Sales Invoice customer required")
    similar = enc.encode("Sales Invoice customer mandatory")
    distant = enc.encode("Payroll tax bracket computation")
    d_sim = cosine_distance(base, similar)
    d_far = cosine_distance(base, distant)
    assert d_sim < d_far, f"similar={d_sim:.3f} should be < distant={d_far:.3f}"


@test("encoder: cosine_distance bounds")
def test_cosine_distance_bounds():
    v = torch.randn(32)
    assert abs(cosine_distance(v, v)) < 1e-5         # identical → 0
    assert 0.0 <= cosine_distance(torch.randn(32), torch.randn(32)) <= 2.0


# ---------------------------------------------------------------------------
# WorldModelNet
# ---------------------------------------------------------------------------

@test("net: evolve produces state-dim vector")
def test_net_evolve():
    net = WorldModelNet(obs_dim=64, state_dim=32)
    o = torch.randn(1, 64)
    z = torch.zeros(1, 32)
    z2 = net.evolve(o, z, dt=1.0)
    assert z2.shape == (1, 32)


@test("net: predict produces normalised obs-dim vector")
def test_net_predict():
    net = WorldModelNet(obs_dim=64, state_dim=32)
    z = torch.randn(1, 32)
    p = net.predict(z)
    assert p.shape == (1, 64)
    assert abs(p.norm(dim=-1).item() - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# perceive() loop
# ---------------------------------------------------------------------------

@test("perceive: returns PerceptionResult with required fields")
def test_perceive_result():
    wm = _tmp_wm()
    r = wm.perceive("User opened a Sales Invoice")
    assert isinstance(r, PerceptionResult)
    assert r.step == 1
    assert 0.0 <= r.surprise <= 2.0
    assert r.novelty in ("novel", "familiar", "expected")


@test("perceive: first observation is maximally surprising and stored")
def test_perceive_first_novel():
    wm = _tmp_wm()
    r = wm.perceive("Completely fresh observation")
    assert r.surprise >= 0.9, f"First obs should be very surprising, got {r.surprise}"
    assert r.stored, "First observation should be stored"


@test("perceive: repetition reduces surprise (model learns its world)")
def test_perceive_learns():
    wm = _tmp_wm(surprise_threshold=0.35)
    ev = "User submitted the Purchase Order for approval"
    first = wm.perceive(ev).surprise
    last = first
    for _ in range(6):
        last = wm.perceive(ev).surprise
    assert last < first, f"Surprise should drop with repetition: {first:.3f} → {last:.3f}"
    assert last < 0.35, f"Repeated event should become expected, got {last:.3f}"


@test("perceive: repeated event eventually stops being stored")
def test_perceive_gating():
    wm = _tmp_wm(surprise_threshold=0.35)
    ev = "Daily routine sync of inventory levels"
    stored_flags = [wm.perceive(ev).stored for _ in range(8)]
    assert stored_flags[0], "First occurrence should be stored"
    assert not stored_flags[-1], "Repeated routine should stop being stored"


@test("perceive: state.step increments and persists")
def test_perceive_step():
    wm = _tmp_wm()
    for i in range(3):
        wm.perceive(f"event {i}")
    assert wm.state.step == 3


@test("perceive: surprise_ema decreases over a learned routine")
def test_perceive_ema_drops():
    wm = _tmp_wm()
    for _ in range(8):
        wm.perceive("recurring maintenance event")
    assert wm.state.surprise_ema < 0.5


# ---------------------------------------------------------------------------
# Predictive memory
# ---------------------------------------------------------------------------

@test("memory: should_write respects threshold")
def test_memory_threshold():
    m = PredictiveMemory(store_path="/tmp/wm_mem_test.json", surprise_threshold=0.4)
    assert m.should_write(0.5)
    assert not m.should_write(0.3)


@test("memory: write creates episode with temporal link")
def test_memory_temporal_link():
    m = PredictiveMemory(store_path="/tmp/wm_mem_test2.json", surprise_threshold=0.1)
    m.episodes.clear(); m._last_id = ""
    o = torch.randn(128); o = o / o.norm()
    z = torch.randn(64)
    e1 = m.write("event one", o, z, surprise=0.9, step=0)
    e2 = m.write("event two", o, z, surprise=0.9, step=1)
    assert e1 is not None and e2 is not None
    assert e2.prev_id == e1.id, "Second episode should link to first"


@test("memory: causal_parent links similar episodes")
def test_memory_causal_link():
    m = PredictiveMemory(store_path="/tmp/wm_mem_test3.json", surprise_threshold=0.1)
    m.episodes.clear(); m._last_id = ""
    o1 = torch.zeros(128); o1[0] = 1.0
    o2 = torch.zeros(128); o2[0] = 0.95; o2[1] = 0.05
    o2 = o2 / o2.norm()
    z = torch.randn(64)
    e1 = m.write("similar A", o1, z, surprise=0.9, step=0)
    e2 = m.write("similar B", o2, z, surprise=0.9, step=1)
    assert e2.causal_parent == e1.id, "Similar episode should get causal parent"


@test("memory: timeline walks prev_id chain")
def test_memory_timeline():
    m = PredictiveMemory(store_path="/tmp/wm_mem_test4.json", surprise_threshold=0.1)
    m.episodes.clear(); m._last_id = ""
    o = torch.randn(128); o = o / o.norm()
    z = torch.randn(64)
    ids = [m.write(f"step {i}", o, z, 0.9, i).id for i in range(4)]
    chain = m.timeline(ids[-1], back=3)
    assert len(chain) == 3
    assert chain[0].id == ids[-1]


@test("memory: high_salience returns surprising unconsolidated episodes")
def test_memory_high_salience():
    m = PredictiveMemory(store_path="/tmp/wm_mem_test5.json", surprise_threshold=0.1)
    m.episodes.clear(); m._last_id = ""
    o = torch.randn(128); o = o / o.norm(); z = torch.randn(64)
    m.write("low", o, z, surprise=0.2, step=0)
    m.write("high", o, z, surprise=0.8, step=1)
    sal = m.high_salience(threshold=0.5)
    assert len(sal) == 1
    assert sal[0].text == "high"


@test("memory: stats reports temporal/causal link counts")
def test_memory_stats():
    wm = _tmp_wm(surprise_threshold=0.1)
    for i in range(4):
        wm.perceive(f"distinct event number {i} alpha beta")
    s = wm.memory.stats()
    assert s["episodes"] >= 2
    assert s["has_temporal_links"] >= 1


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

@test("retrieval: returns episodes ranked by score")
def test_retrieval_ranked():
    wm = _tmp_wm(surprise_threshold=0.1)
    wm.perceive("Security alert unauthorized access detected")
    wm.perceive("User created a normal Sales Order")
    wm.perceive("Payroll salary computation completed")
    hits = wm.reason("security access breach", top_k=3)
    assert len(hits) >= 1
    # scores should be descending
    for i in range(len(hits) - 1):
        assert hits[i].score >= hits[i+1].score


@test("retrieval: state-conditioned — security query surfaces security episode")
def test_retrieval_relevant():
    wm = _tmp_wm(surprise_threshold=0.1)
    wm.perceive("Security breach: unauthorized Payroll access")
    wm.perceive("Routine inventory count for warehouse")
    wm.perceive("Customer added to Sales Order")
    hits = wm.reason("unauthorized security access", top_k=1)
    assert len(hits) == 1
    assert "security" in hits[0].episode.text.lower() or "unauthorized" in hits[0].episode.text.lower()


@test("retrieval: recall_story includes timeline")
def test_retrieval_story():
    wm = _tmp_wm(surprise_threshold=0.1)
    for i in range(4):
        wm.perceive(f"workflow phase {i} with distinct vocabulary term{i}")
    stories = wm.recall_story("workflow phase", top_k=2)
    assert len(stories) >= 1
    assert "match" in stories[0]
    assert "timeline" in stories[0]


@test("retrieval: empty memory returns empty list")
def test_retrieval_empty():
    wm = _tmp_wm()
    hits = wm.reason("anything", top_k=5)
    assert hits == []


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------

@test("consolidation: returns ConsolidationReport")
def test_consolidation_report():
    wm = _tmp_wm(surprise_threshold=0.3)
    for _ in range(3):
        wm.perceive("Sales Order created by user")
        wm.perceive("Delivery Note generated automatically")
    c = Consolidator(wm, replay_epochs=3)
    report = c.consolidate()
    assert isinstance(report, ConsolidationReport)


@test("consolidation: distills temporal patterns to semantic memory")
def test_consolidation_distill():
    from synapseos.phase5.memory import SemanticMemory
    wm = _tmp_wm(surprise_threshold=0.3)
    sem = SemanticMemory(store_path="/tmp/wm_consol_sem.json", kg_path="/tmp/nope.json")
    sem.facts.clear(); sem._fact_index.clear()
    # A → B → C repeated
    for _ in range(4):
        wm.perceive("User created a Sales Order")
        wm.perceive("System generated a Delivery Note")
        wm.perceive("Accounts posted a Sales Invoice")
    c = Consolidator(wm, semantic_memory=sem, replay_epochs=3, min_pattern_count=2)
    report = c.consolidate(salience_threshold=0.3)
    assert report.facts_distilled >= 1, f"Expected distilled facts, got {report.facts_distilled}"
    # semantic memory should contain precedence facts
    preceds = [f for f in sem.facts if f.predicate == "frequently_precedes"]
    assert len(preceds) >= 1


@test("consolidation: marks replayed episodes consolidated")
def test_consolidation_marks():
    wm = _tmp_wm(surprise_threshold=0.2)
    for i in range(5):
        wm.perceive(f"highly novel distinct event {i} xyzzy{i}")
    c = Consolidator(wm, replay_epochs=2)
    before = wm.memory.stats().get("consolidated", 0)
    c.consolidate(salience_threshold=0.2)
    after = wm.memory.stats().get("consolidated", 0)
    assert after > before


@test("consolidation: transition counter tracks all perceptions")
def test_transition_counter():
    wm = _tmp_wm(surprise_threshold=0.9)  # high threshold → few stored
    for _ in range(3):
        wm.perceive("Alpha document opened")
        wm.perceive("Beta document processed")
    counts = wm.transition_counts()
    # alpha→beta should be counted even though events become unsurprising
    assert any(n >= 2 for n in counts.values()), f"Expected repeated transition, got {counts}"


@test("consolidation: predictor loss does not increase after replay")
def test_consolidation_predictor():
    wm = _tmp_wm(surprise_threshold=0.2)
    for i in range(6):
        wm.perceive(f"event alpha beta {i}")
    c = Consolidator(wm, replay_epochs=5)
    report = c.consolidate(salience_threshold=0.2)
    if report.episodes_replayed > 0:
        assert report.predictor_loss_after <= report.predictor_loss_before + 1e-3


# ---------------------------------------------------------------------------
# Persistence of the whole model
# ---------------------------------------------------------------------------

@test("world model: save and reload preserves state + episodes")
def test_wm_persistence():
    d = tempfile.mkdtemp(prefix="wm_persist_")
    paths = dict(
        state_path=os.path.join(d, "state.json"),
        episodes_path=os.path.join(d, "episodes.json"),
        snapshot_dir=os.path.join(d, "snaps"),
    )
    wm = WorldModel(**paths, surprise_threshold=0.1)
    wm.perceive("Memorable surprising event one")
    wm.perceive("Memorable surprising event two")
    step_before = wm.state.step
    ep_before   = len(wm.memory)
    wm.save()

    wm2 = WorldModel(**paths, surprise_threshold=0.1)
    assert wm2.state.step == step_before
    assert len(wm2.memory) == ep_before


@test("world model: snapshot and rollback restores state")
def test_wm_rollback():
    wm = _tmp_wm()
    wm.perceive("event before snapshot")
    snap = wm.snapshot()
    step_at_snap = wm.state.step
    wm.perceive("event after snapshot")
    wm.perceive("another event after")
    assert wm.state.step > step_at_snap
    wm.rollback(snap)
    assert wm.state.step == step_at_snap


@test("world model: stats returns expected keys")
def test_wm_stats():
    wm = _tmp_wm()
    wm.perceive("test event")
    s = wm.stats()
    for k in ("state", "memory", "transitions", "net_params", "device"):
        assert k in s


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        # State
        test_state_fresh, test_state_set_z, test_state_surprise_ema,
        test_state_save_load, test_state_load_fresh, test_state_snapshot_restore,
        # Encoder
        test_encoder_dim, test_encoder_deterministic, test_encoder_similarity,
        test_cosine_distance_bounds,
        # Net
        test_net_evolve, test_net_predict,
        # perceive loop
        test_perceive_result, test_perceive_first_novel, test_perceive_learns,
        test_perceive_gating, test_perceive_step, test_perceive_ema_drops,
        # Memory
        test_memory_threshold, test_memory_temporal_link, test_memory_causal_link,
        test_memory_timeline, test_memory_high_salience, test_memory_stats,
        # Retrieval
        test_retrieval_ranked, test_retrieval_relevant, test_retrieval_story,
        test_retrieval_empty,
        # Consolidation
        test_consolidation_report, test_consolidation_distill, test_consolidation_marks,
        test_transition_counter, test_consolidation_predictor,
        # Persistence
        test_wm_persistence, test_wm_rollback, test_wm_stats,
    ]

    for t in tests:
        t()

    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed

    print(f"\n{'='*60}")
    print(f"World Model Tests: {passed}/{len(results)} passed\n")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        detail = f"  → {r['detail']}" if not r["passed"] else ""
        print(f"  [{status}] {r['name']}  ({r['ms']}ms){detail}")

    sys.exit(0 if failed == 0 else 1)
