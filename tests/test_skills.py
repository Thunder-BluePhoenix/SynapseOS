"""
Tests for SynapseOS Skills — math solver, language detection, curriculum,
and the MENTOR teacher agent.

All deterministic, zero network.

Run:
    python tests/test_skills.py
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from synapseos.skills.math_solver import MathSolver
from synapseos.skills.language import detect_language, phrase
from synapseos.phase5.memory import SemanticMemory


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


def _tmp_dir() -> str:
    return tempfile.mkdtemp(prefix="skills_test_")


def _tmp_semantic() -> SemanticMemory:
    d = _tmp_dir()
    return SemanticMemory(store_path=os.path.join(d, "sem.json"),
                          kg_path=os.path.join(d, "missing.json"))


solver = MathSolver()


# ---------------------------------------------------------------------------
# Math: arithmetic & algebra
# ---------------------------------------------------------------------------

@test("arithmetic expression evaluation")
def test_arithmetic():
    a = solver.solve("calculate (3 + 4) * 2 - 5")
    assert a is not None and a.value == 9

@test("arithmetic with functions and power")
def test_arithmetic_funcs():
    a = solver.solve("what is sqrt(144) + 2^3")
    assert a is not None and a.value == 20

@test("linear equation")
def test_linear():
    a = solver.solve("solve 2x + 5 = 15")
    assert a is not None and abs(a.value - 5) < 1e-9
    assert a.topic == "algebra"

@test("quadratic equation real roots")
def test_quadratic():
    a = solver.solve("solve x^2 - 5x + 6 = 0")
    assert a is not None and sorted(a.value) == [2.0, 3.0]
    assert any("discriminant" in s for s in a.steps)

@test("quadratic complex roots reported honestly")
def test_quadratic_complex():
    a = solver.solve("solve x^2 + 1 = 0")
    assert a is not None and "complex" in a.answer

# ---------------------------------------------------------------------------
# Math: calculus
# ---------------------------------------------------------------------------

@test("symbolic polynomial derivative")
def test_derivative():
    a = solver.solve("derivative of 3x^2 + 2x - 5")
    assert a is not None and a.answer == "f'(x) = 6x + 2", a.answer

@test("derivative evaluated at a point")
def test_derivative_at():
    a = solver.solve("derivative of x^2 at x = 4")
    assert a is not None and abs(a.value - 8) < 1e-9

@test("numeric definite integral")
def test_integral():
    a = solver.solve("integrate x^2 from 0 to 3")
    assert a is not None and abs(a.value - 9.0) < 1e-3
    assert not a.exact   # honestly labelled as approximation

@test("integral of trig function")
def test_integral_trig():
    a = solver.solve("integrate sin(x) from 0 to 3.14159265")
    assert a is not None and abs(a.value - 2.0) < 1e-3

# ---------------------------------------------------------------------------
# Math: stats, probability, geometry, trig, constants
# ---------------------------------------------------------------------------

@test("statistics mean and std")
def test_stats():
    a = solver.solve("mean of 2, 4, 6, 8")
    assert a is not None and a.value == 5
    b = solver.solve("standard deviation of 2, 4, 4, 4, 5, 5, 7, 9")
    assert b is not None and abs(b.value - 2.138) < 0.01

@test("combinations and permutations")
def test_comb():
    a = solver.solve("5 c 2")
    assert a is not None and a.value == 10
    b = solver.solve("permutations of 5 taken 2")
    assert b is not None and b.value == 20

@test("geometry: circle area with value")
def test_geometry():
    a = solver.solve("area of a circle with radius 5")
    assert a is not None and abs(a.value - math.pi * 25) < 1e-9
    assert "π·r²" in a.answer

@test("mensuration: cylinder volume")
def test_mensuration():
    a = solver.solve("volume of a cylinder with radius 2 and height 10")
    assert a is not None and abs(a.value - math.pi * 4 * 10) < 1e-9

@test("geometry formula lookup without numbers")
def test_geometry_formula_only():
    a = solver.solve("what is the volume of a sphere")
    assert a is not None and "4/3·π·r³" in a.answer

@test("trigonometry in degrees")
def test_trig():
    a = solver.solve("sin(30 deg)")
    assert a is not None and abs(a.value - 0.5) < 1e-9
    b = solver.solve("cos(60 degrees)")
    assert b is not None and abs(b.value - 0.5) < 1e-9

@test("physics and quantum constants")
def test_constants():
    a = solver.solve("what is the speed of light")
    assert a is not None and a.value == 299_792_458.0
    b = solver.solve("value of the planck constant")
    assert b is not None and abs(b.value - 6.62607015e-34) < 1e-40

@test("non-math text returns None")
def test_not_math():
    assert solver.solve("tell me about the French Revolution") is None
    assert solver.solve("how are you today?") is None

@test("unsafe expressions rejected")
def test_safety():
    assert solver.solve("__import__('os').system('ls')") is None
    assert solver.solve("calculate open('/etc/passwd')") is None

# ---------------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------------

@test("script detection: bengali, hindi, japanese")
def test_scripts():
    assert detect_language("আমি বাংলায় কথা বলি").language == "bengali"
    assert detect_language("मैं हिंदी बोलता हूँ").language == "hindi"
    assert detect_language("こんにちは元気ですか").language == "japanese"

@test("stopword detection: english vs spanish")
def test_latin_langs():
    assert detect_language("the cat is on the table and it is happy").language == "english"
    assert detect_language("el gato está en la mesa y es feliz").language == "spanish"

@test("phrasebook lookup")
def test_phrasebook():
    assert "dhonnobad" in phrase("bengali", "thank you")
    assert phrase("spanish", "hello") == "hola"
    assert phrase("klingon", "hello") is None

# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------

@test("curriculum seeds subject goals with priority order")
def test_curriculum_seed():
    from synapseos.web.curiosity import CuriosityEngine, GoalQueue
    from synapseos.web.curriculum import seed_curriculum, curriculum_progress
    engine = CuriosityEngine(queue=GoalQueue(os.path.join(_tmp_dir(), "g.json")))
    added = seed_curriculum(engine, "physics")
    assert added["physics"] >= 8
    pending = engine.queue.pending()
    assert any("Quantum mechanics" in g.topic for g in pending)
    # fundamentals first
    topics = [g.topic for g in pending]
    assert topics.index("Classical mechanics") < topics.index("Quantum mechanics")
    progress = curriculum_progress(engine)
    assert progress["physics"]["pending"] >= 8

@test("curriculum all subjects + idempotent")
def test_curriculum_all():
    from synapseos.web.curiosity import CuriosityEngine, GoalQueue
    from synapseos.web.curriculum import seed_curriculum, SUBJECTS
    engine = CuriosityEngine(queue=GoalQueue(os.path.join(_tmp_dir(), "g.json")))
    added = seed_curriculum(engine, "all")
    assert set(added) == set(SUBJECTS)
    n = len(engine.queue)
    seed_curriculum(engine, "all")          # re-seed
    assert len(engine.queue) == n           # no duplicates

# ---------------------------------------------------------------------------
# MENTOR agent
# ---------------------------------------------------------------------------

def _mentor(mem=None):
    from synapseos.phase5.agents.mentor import MENTORAgent
    agent = MENTORAgent()
    agent.wire(semantic_memory=mem)
    return agent

@test("MENTOR routes teaching queries")
def test_mentor_routing():
    agent = _mentor()
    assert agent.can_handle("teach me about photosynthesis") > 0.5
    assert agent.can_handle("solve x^2 - 4 = 0") > 0.5
    assert agent.can_handle("show invoice doctype fields") == 0.0

@test("MENTOR claims bare math expressions without keywords")
def test_mentor_bare_math():
    agent = _mentor()
    assert agent.can_handle("sin(30 deg)") >= 0.85
    assert agent.can_handle("(3 + 4) * 2 - 5") >= 0.85
    result = agent.handle("sin(30 deg)")
    assert "0.5" in result.output

@test("MENTOR solves math with worked steps")
def test_mentor_math():
    result = _mentor().handle("solve 2x + 6 = 0")
    assert "x = -3" in result.output
    assert "Working:" in result.output
    assert result.confidence == 0.9

@test("MENTOR teaches from learned facts with structure")
def test_mentor_lesson():
    mem = _tmp_semantic()
    mem.add_fact("Photosynthesis", "is_a", "biological process", 0.7, "web:en.wikipedia.org")
    mem.add_fact("Photosynthesis", "used_for", "converting light into energy", 0.6, "web:en.wikipedia.org")
    result = _mentor(mem).handle("teach me about photosynthesis")
    assert "Definition: Photosynthesis is a biological process" in result.output
    assert "Key points:" in result.output
    # teaching is recorded
    taught = [f for f in mem.retrieve("user") if f.predicate == "was_taught"]
    assert taught

@test("MENTOR quizzes from known facts withholding answers")
def test_mentor_quiz():
    mem = _tmp_semantic()
    mem.add_fact("Paris", "located_in", "France", 0.8, "web:en.wikipedia.org")
    mem.add_fact("Python", "created_by", "Guido van Rossum", 0.8, "web:en.wikipedia.org")
    result = _mentor(mem).handle("quiz me on Paris and Python")
    assert "Q1." in result.output
    assert "France" not in result.output.split("Answers")[0].replace(
        "France", "France") or "France" not in result.output  # answers withheld
    assert result.structured["answers"]

@test("MENTOR phrasebook and honest translation limit")
def test_mentor_language():
    result = _mentor().handle("how do you say thank you in bengali")
    assert "ধন্যবাদ" in result.output
    assert "neural" in result.output      # honest about the limit

@test("MENTOR detects language of text")
def test_mentor_detect():
    result = _mentor().handle('what language is "こんにちは元気ですか"')
    assert "japanese" in result.output.lower()

@test("MENTOR unknown topic queues learning")
def test_mentor_unknown():
    class _FakeCuriosity:
        def __init__(self): self.topics = []
        def add_user_topic(self, t, priority=1.0): self.topics.append(t)
    class _FakeLoop:
        def __init__(self): self.curiosity = _FakeCuriosity()
    mem = _tmp_semantic()
    agent = _mentor(mem)
    agent.web_loop = _FakeLoop()
    result = agent.handle("teach me about the Treaty of Westphalia")
    assert "haven't studied" in result.output
    assert agent.web_loop.curiosity.topics

# ---------------------------------------------------------------------------
# Controller integration
# ---------------------------------------------------------------------------

@test("controller: math question routed to MENTOR end to end")
def test_controller_mentor():
    from synapseos.phase5.cognitive_controller import CognitiveController
    d = _tmp_dir()
    c = CognitiveController(
        memory_episodic_path=os.path.join(d, "epi.json"),
        memory_semantic_path=os.path.join(d, "sem.json"),
        kg_path=os.path.join(d, "missing_kg.json"),
        rules_path=os.path.join(d, "missing_rules.json"),
        enable_world_model=False,
        affect_state_path=os.path.join(d, "affect.json"),
    )
    result = c.think("solve x^2 - 9 = 0")
    assert result.routed_to_agent == "MENTOR", result.routed_to_agent
    assert "x = 3" in result.primary_output and "x = -3" in result.primary_output
    c.shutdown()

@test("controller: study() seeds curriculum")
def test_controller_study():
    from synapseos.phase5.cognitive_controller import CognitiveController
    from synapseos.web import FakeFetcher
    d = _tmp_dir()
    c = CognitiveController(
        memory_episodic_path=os.path.join(d, "epi.json"),
        memory_semantic_path=os.path.join(d, "sem.json"),
        kg_path=os.path.join(d, "missing_kg.json"),
        rules_path=os.path.join(d, "missing_rules.json"),
        enable_agents=False, enable_world_model=False,
        enable_web_learning=True, web_data_dir=os.path.join(d, "web"),
        web_fetcher=FakeFetcher(),
        affect_state_path=os.path.join(d, "affect.json"),
    )
    out = c.study("mathematics")
    assert out["seeded"]["mathematics"] >= 10
    assert out["progress"]["mathematics"]["pending"] >= 10
    c.shutdown()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        # Math
        test_arithmetic, test_arithmetic_funcs, test_linear, test_quadratic,
        test_quadratic_complex, test_derivative, test_derivative_at,
        test_integral, test_integral_trig, test_stats, test_comb,
        test_geometry, test_mensuration, test_geometry_formula_only,
        test_trig, test_constants, test_not_math, test_safety,
        # Languages
        test_scripts, test_latin_langs, test_phrasebook,
        # Curriculum
        test_curriculum_seed, test_curriculum_all,
        # MENTOR
        test_mentor_routing, test_mentor_bare_math, test_mentor_math, test_mentor_lesson,
        test_mentor_quiz, test_mentor_language, test_mentor_detect,
        test_mentor_unknown,
        # Controller
        test_controller_mentor, test_controller_study,
    ]

    for t in tests:
        t()

    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed

    print(f"\n{'='*60}")
    print(f"Skills Tests: {passed}/{len(results)} passed\n")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        detail = f"  → {r['detail']}" if not r["passed"] else ""
        print(f"  [{status}] {r['name']}  ({r['ms']}ms){detail}")

    sys.exit(0 if failed == 0 else 1)
