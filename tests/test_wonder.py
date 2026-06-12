"""
Tests for SynapseOS inner life — the WonderJournal.

Covers: forming questions from surprise/contradiction/frontier/user,
the attention economy (intensity decay that never hits zero, neglect
boosting liveliness), deepening into sub-questions, resolution, persistence,
"I've been wondering for N days", curiosity feed, and controller integration.

Run:
    python tests/test_wonder.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from synapseos.soul.wonder import WonderJournal, DAY


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


def _tmp_journal(**kw) -> WonderJournal:
    d = tempfile.mkdtemp(prefix="wonder_test_")
    return WonderJournal(path=os.path.join(d, "wonder.json"), **kw)


@test("forms questions from the four sources")
def test_sources():
    j = _tmp_journal()
    assert j.from_surprise("a falling cat always lands on its feet", 0.8) is not None
    assert j.from_surprise("boring routine thing", 0.2) is None     # not surprising
    c = j.from_contradiction("Pluto", "is_a", "planet", "dwarf planet")
    assert c.source == "contradiction" and c.intensity >= 0.7
    f = j.from_frontier("quasicrystal", degree=1)
    assert f.source == "frontier"
    u = j.wonder_about("What makes music move us?", source="user")
    assert u.source == "user"
    assert j.stats()["total"] == 4


@test("reinforcing a question stokes it, doesn't duplicate")
def test_reinforce():
    j = _tmp_journal()
    a = j.wonder_about("Why is the sky blue?", intensity=0.5)
    b = j.wonder_about("why is the sky blue?", intensity=0.5)   # same, diff case
    assert a.id == b.id
    assert b.intensity > 0.5            # stoked
    assert j.stats()["total"] == 1


@test("intensity decays when ignored but never to zero")
def test_decay_floor():
    j = _tmp_journal(decay_per_day=0.2)
    w = j.wonder_about("a lingering question", intensity=0.5)
    w.last_visited = time.time() - 100 * DAY     # ignored for 100 days
    j.ponder()                                    # triggers a decay pass
    refreshed = j._wonders[w.id]
    assert refreshed.intensity >= 0.08            # floor holds — it still nags
    assert refreshed.intensity < 0.5


@test("ponder is prioritized, not random — neglect + intensity win")
def test_ponder_priority():
    j = _tmp_journal()
    weak = j.wonder_about("a mild passing thought", intensity=0.2)
    strong = j.wonder_about("a question that really bothers me", intensity=0.95)
    # run several times: the strong one should dominate
    picks = [j.ponder().id for _ in range(5)]
    assert picks.count(strong.id) > picks.count(weak.id)


@test("a long-neglected intense question resurfaces")
def test_neglect_resurfaces():
    j = _tmp_journal(decay_per_day=0.01)
    fresh = j.wonder_about("just thought of this", intensity=0.6)
    old = j.wonder_about("the deep one I keep circling back to", intensity=0.7)
    old.last_visited = time.time() - 10 * DAY     # neglected → high liveliness
    old.created_at = time.time() - 20 * DAY
    chosen = j.ponder()
    assert chosen.id == old.id


@test("deepening spawns a sub-question linked to its parent")
def test_deepen():
    j = _tmp_journal()
    parent = j.wonder_about("How does memory work?")
    sub = j.deepen(parent.id, "Does forgetting serve a purpose?")
    assert sub.parent_id == parent.id and sub.source == "deepened"


@test("questions resolve and leave the open set")
def test_resolve():
    j = _tmp_journal()
    w = j.wonder_about("What is entropy?")
    j.resolve(w.id, "a measure of disorder / unavailable energy")
    assert j._wonders[w.id].status == "resolved"
    assert w.id not in {x.id for x in j.open_questions()}


@test("surface narrates the longest-held question with its age")
def test_surface():
    j = _tmp_journal()
    recent = j.wonder_about("a question from just now")
    old = j.wonder_about("the question I've carried a while")
    old.created_at = time.time() - 21 * DAY
    line = j.surface()
    assert "21 days" in line and "carried a while" in line


@test("persists across instances")
def test_persist():
    d = tempfile.mkdtemp(prefix="wonder_test_")
    path = os.path.join(d, "w.json")
    j = WonderJournal(path=path)
    j.wonder_about("does this survive a restart?", intensity=0.8)
    j.save()
    j2 = WonderJournal(path=path)
    assert j2.stats()["total"] == 1
    assert j2.open_questions()[0].intensity == 0.8


@test("feeds its liveliest questions to the curiosity engine")
def test_feed_curiosity():
    j = _tmp_journal()
    j.wonder_about("What is dark matter made of?", topic="dark matter", intensity=0.9)

    class _FakeGoal:
        def __init__(self): self.source = ""; self.reason = ""

    class _FakeCuriosity:
        def __init__(self): self.topics = []
        def add_user_topic(self, topic, priority=1.0):
            self.topics.append(topic)
            return _FakeGoal()

    cur = _FakeCuriosity()
    fed = j.feed_curiosity(cur)
    assert fed == 1 and "dark matter" in cur.topics


@test("controller exposes ponder() and wonder_about(), surprise plants a question")
def test_controller():
    from synapseos.phase5.cognitive_controller import CognitiveController
    d = tempfile.mkdtemp(prefix="wonder_test_")
    c = CognitiveController(
        memory_episodic_path=os.path.join(d, "e.json"),
        memory_semantic_path=os.path.join(d, "s.json"),
        kg_path=os.path.join(d, "k.json"), rules_path=os.path.join(d, "r.json"),
        affect_state_path=os.path.join(d, "a.json"),
        world_model_dir=os.path.join(d, "wm"))
    # ensure wonder uses a temp store
    from synapseos.soul.wonder import WonderJournal
    c.wonder = WonderJournal(path=os.path.join(d, "wonder.json"))

    planted = c.wonder_about("Will it remember what it wonders?", topic="memory")
    assert planted["status"] == "open"
    out = c.ponder()
    assert out["pondering"] is not None
    assert "on_my_mind" in out
    # a couple of genuinely novel inputs should create surprise-born wonders
    for q in ("xylophone quantum entanglement banana", "flarn glorptastic zibber"):
        c.think(q)
    assert c.wonder.stats()["total"] >= 1
    c.shutdown()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_sources, test_reinforce, test_decay_floor, test_ponder_priority,
        test_neglect_resurfaces, test_deepen, test_resolve, test_surface,
        test_persist, test_feed_curiosity, test_controller,
    ]
    for t in tests:
        t()

    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed
    print(f"\n{'='*60}")
    print(f"Wonder Tests: {passed}/{len(results)} passed\n")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        detail = f"  → {r['detail']}" if not r["passed"] else ""
        print(f"  [{status}] {r['name']}  ({r['ms']}ms){detail}")
    sys.exit(0 if failed == 0 else 1)
