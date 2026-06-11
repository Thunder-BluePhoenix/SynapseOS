"""
Tests for SynapseOS Affect — non-LLM emotion understanding.

Covers:
  - TextAffectDetector: lexicon hits, negation, intensifiers, caps/punctuation,
    emoticons, evidence preservation, neutral text
  - Prosody: synthesized WAVs (calm sine vs agitated bursts) → arousal ordering
  - EmotionTracker: EMA state, mood-shift detection, trigger pattern learning
    into semantic memory, persistence
  - CognitiveController integration: affect in CognitionResult, emotion tags on
    world-model episodes, learned "user frustrated_by X" facts in answers

No network, no whisper binary required.

Run:
    python tests/test_affect.py
"""

from __future__ import annotations

import json
import math
import os
import struct
import sys
import tempfile
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from synapseos.affect.text_affect import TextAffectDetector
from synapseos.affect.prosody import analyze_wav
from synapseos.affect.tracker import EmotionTracker
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
    return tempfile.mkdtemp(prefix="affect_test_")


def _tmp_semantic() -> SemanticMemory:
    d = _tmp_dir()
    return SemanticMemory(store_path=os.path.join(d, "sem.json"),
                          kg_path=os.path.join(d, "missing.json"))


def _write_wav(path: str, samples: list[float], rate: int = 16000) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = b"".join(struct.pack("<h", int(max(-1, min(1, s)) * 32000))
                          for s in samples)
        w.writeframes(frames)


def _calm_wav(path: str) -> None:
    """Quiet steady tone with long pauses — low arousal."""
    rate = 16000
    samples = []
    for i in range(rate * 2):
        t = i / rate
        in_speech = (t % 1.0) < 0.3            # 30% duty cycle — lots of pauses
        amp = 0.15 if in_speech else 0.0
        samples.append(amp * math.sin(2 * math.pi * 120 * t))
    _write_wav(path, samples, rate)


def _agitated_wav(path: str) -> None:
    """Loud rapid bursts with varying pitch — high arousal."""
    rate = 16000
    samples = []
    for i in range(rate * 2):
        t = i / rate
        in_speech = (t % 0.25) < 0.18          # rapid bursts, few pauses
        pitch = 180 + 120 * math.sin(2 * math.pi * 1.5 * t)   # swooping pitch
        amp = 0.9 if in_speech else 0.0
        samples.append(amp * math.sin(2 * math.pi * pitch * t))
    _write_wav(path, samples, rate)


# ---------------------------------------------------------------------------
# Text affect
# ---------------------------------------------------------------------------

@test("detects frustration with evidence")
def test_frustration():
    r = TextAffectDetector().detect("This is so frustrating, the build failed again!")
    assert r.dominant_emotion == "frustration", r.dominant_emotion
    assert r.valence < 0
    assert "frustrating" in r.evidence
    assert r.confidence > 0.3


@test("detects joy and excitement")
def test_joy():
    r = TextAffectDetector().detect("This is amazing, I love it! Finally works!")
    assert r.valence > 0.4
    assert r.dominant_emotion in ("joy", "excitement")
    assert r.arousal > 0.5     # exclamation marks raise arousal


@test("negation flips valence")
def test_negation():
    pos = TextAffectDetector().detect("I am happy with this.")
    neg = TextAffectDetector().detect("I am not happy with this.")
    assert pos.valence > 0
    assert neg.valence < 0
    assert neg.dominant_emotion == "sadness"   # negated joy → sadness bucket


@test("intensifiers and diminishers scale")
def test_intensity():
    base = TextAffectDetector().detect("I am angry.")
    strong = TextAffectDetector().detect("I am extremely angry.")
    mild = TextAffectDetector().detect("I am slightly angry.")
    assert strong.valence < base.valence < mild.valence < 0


@test("caps and punctuation raise arousal")
def test_caps():
    quiet = TextAffectDetector().detect("this is wrong")
    loud = TextAffectDetector().detect("THIS IS WRONG!!!")
    assert loud.arousal > quiet.arousal


@test("emoticons carry affect")
def test_emoticons():
    r = TextAffectDetector().detect("ok then :(")
    assert r.valence < 0 and r.dominant_emotion == "sadness"


@test("neutral text stays neutral")
def test_neutral():
    r = TextAffectDetector().detect("The invoice has twelve line items.")
    assert r.is_neutral and r.valence == 0.0


@test("hesitation from ellipsis and hedging")
def test_hesitation():
    r = TextAffectDetector().detect("Hmm... maybe, I'm unsure about this...")
    assert r.dominant_emotion == "hesitation"


# ---------------------------------------------------------------------------
# Prosody
# ---------------------------------------------------------------------------

@test("prosody: agitated speech scores higher arousal than calm")
def test_prosody_ordering():
    d = _tmp_dir()
    calm_path = os.path.join(d, "calm.wav")
    agit_path = os.path.join(d, "agitated.wav")
    _calm_wav(calm_path)
    _agitated_wav(agit_path)
    calm = analyze_wav(calm_path)
    agit = analyze_wav(agit_path)
    assert agit.arousal > calm.arousal, f"{agit.arousal} <= {calm.arousal}"
    assert agit.energy_rms > calm.energy_rms or agit.pause_ratio < calm.pause_ratio


@test("prosody: detects pauses and duration")
def test_prosody_pauses():
    d = _tmp_dir()
    path = os.path.join(d, "calm.wav")
    _calm_wav(path)
    r = analyze_wav(path)
    assert 1.9 < r.duration_s < 2.1
    assert r.pause_ratio > 0.4     # 30% duty cycle → mostly silence


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

@test("tracker EMA moves with observations")
def test_tracker_ema():
    t = EmotionTracker(state_path=os.path.join(_tmp_dir(), "state.json"))
    for _ in range(4):
        t.observe("This is frustrating, everything is broken again!")
    assert t.state.valence_ema < -0.2
    assert t.state.dominant_emotion == "frustration"
    assert t.state.observations == 4


@test("tracker detects mood shift")
def test_tracker_shift():
    t = EmotionTracker(state_path=os.path.join(_tmp_dir(), "state.json"))
    for _ in range(3):
        t.observe("I am sad and disappointed about everything.")
    t.observe("This is amazing, absolutely wonderful, I love it!")
    assert "improved" in t.state.last_shift


@test("tracker learns repeated triggers into semantic memory")
def test_tracker_patterns():
    mem = _tmp_semantic()
    t = EmotionTracker(semantic_memory=mem,
                       state_path=os.path.join(_tmp_dir(), "state.json"))
    t.observe("The deployment pipeline is so frustrating!")
    t.observe("The deployment pipeline is so frustrating!")
    facts = mem.retrieve("user")
    assert any(f.predicate == "frustrated_by" and "deployment" in f.obj
               for f in facts), [f.to_dict() for f in facts]
    assert all(f.source == "affect_observation" for f in facts)


@test("tracker prosody arousal overrides flat words")
def test_tracker_prosody_fusion():
    t = EmotionTracker(state_path=os.path.join(_tmp_dir(), "state.json"))
    flat = t.observe("I'm fine.")                          # no prosody
    tense = t.observe("I'm fine.", prosody_arousal=0.9)    # said tensely
    assert tense.arousal > flat.arousal + 0.2


@test("tracker state persists across instances")
def test_tracker_persist():
    path = os.path.join(_tmp_dir(), "state.json")
    t = EmotionTracker(state_path=path)
    t.observe("I am very happy today!")
    t.save()
    t2 = EmotionTracker(state_path=path)
    assert t2.state.observations == 1
    assert t2.state.valence_ema > 0


# ---------------------------------------------------------------------------
# Controller integration
# ---------------------------------------------------------------------------

def _tmp_controller(**kw):
    from synapseos.phase5.cognitive_controller import CognitiveController
    d = _tmp_dir()
    kw.setdefault("enable_agents", False)
    return CognitiveController(
        memory_episodic_path=os.path.join(d, "epi.json"),
        memory_semantic_path=os.path.join(d, "sem.json"),
        kg_path=os.path.join(d, "missing_kg.json"),
        rules_path=os.path.join(d, "missing_rules.json"),
        affect_state_path=os.path.join(d, "affect.json"),
        world_model_dir=os.path.join(d, "wm"),
        **kw,
    )


@test("controller: affect appears in CognitionResult")
def test_controller_affect():
    c = _tmp_controller()
    result = c.think("I'm so frustrated, this report is broken again!")
    assert result.affect, "affect dict empty"
    assert result.affect["detected"]["dominant_emotion"] == "frustration"
    assert "frustration" in result.affect["state"]   # one observation only nudges the EMA
    assert c.memory.get_working("user_affect", "").startswith("frustration")
    c.shutdown()


@test("controller: emotion tags ride on world-model episodes")
def test_controller_emotion_tags():
    c = _tmp_controller()
    assert c.world_model is not None
    c.think("I am absolutely furious about this terrible terrible failure!")
    tagged = [ep for ep in c.world_model.memory.episodes
              if any(t.startswith("emotion:") for t in ep.tags)]
    assert tagged, "no emotion-tagged episodes stored"
    c.shutdown()


@test("controller: learned emotional patterns surface in answers")
def test_controller_pattern_in_answer():
    c = _tmp_controller(enable_world_model=False)
    c.think("The deployment process is frustrating!")
    c.think("The deployment process is frustrating!")
    result = c.think("Tell me about the deployment process")
    assert "frustrated by" in result.primary_output.lower(), \
        result.primary_output[:300]
    c.shutdown()


@test("controller: affect disabled cleanly")
def test_controller_affect_disabled():
    c = _tmp_controller(enable_affect=False, enable_world_model=False)
    result = c.think("I am very happy!")
    assert result.affect == {}
    assert c.stats()["affect"] is None
    c.shutdown()


# ---------------------------------------------------------------------------
# Level 3: response modulation
# ---------------------------------------------------------------------------

@test("modulator adds empathic ack on frustration")
def test_mod_empathy():
    from synapseos.affect import ResponseModulator, EmotionalState
    from synapseos.affect.text_affect import TextAffectDetector
    affect = TextAffectDetector().detect("This is so frustrating and broken!")
    mod = ResponseModulator().modulate("Here is the fix.", affect,
                                       EmotionalState(), "fix the build")
    assert "frustrating" in mod.output.split("\n")[0].lower()
    assert "empathic_ack:frustration" in mod.adjustments


@test("modulator defers risky decisions under sustained stress")
def test_mod_risk_stress():
    from synapseos.affect import ResponseModulator, EmotionalState
    state = EmotionalState(valence_ema=-0.5, arousal_ema=0.7, observations=5)
    mod = ResponseModulator().modulate(
        "Analysis of the investment.", None, state,
        "Should I invest all my savings in this?")
    assert "risk_deferral:stress" in mod.adjustments
    assert "sleeping on it" in mod.output


@test("modulator nudges validation on excited risky decisions")
def test_mod_excited():
    from synapseos.affect import ResponseModulator, EmotionalState
    from synapseos.affect.text_affect import TextAffectDetector
    affect = TextAffectDetector().detect("This is amazing!! I'm so excited!!")
    mod = ResponseModulator().modulate("Plan.", affect, EmotionalState(),
                                       "I want to quit my job and launch this!")
    assert "validate_first:excitement" in mod.adjustments


@test("modulator leaves calm neutral queries untouched")
def test_mod_neutral():
    from synapseos.affect import ResponseModulator, EmotionalState
    mod = ResponseModulator().modulate("Answer.", None, EmotionalState(),
                                       "What fields does an invoice have?")
    assert not mod.modified


# ---------------------------------------------------------------------------
# Advisor agents
# ---------------------------------------------------------------------------

@test("advisors route by domain keywords")
def test_advisor_routing():
    from synapseos.phase5.agents.advisors import (
        VITAAgent, LEXAgent, AURUMAgent, SAGEAgent, ORACLEAgent)
    assert VITAAgent().can_handle("My sleep and diet have been terrible") > 0.5
    assert LEXAgent().can_handle("Is this software license legal to use?") > 0.5
    assert AURUMAgent().can_handle("How should I plan my tax and budget?") > 0.5
    assert SAGEAgent().can_handle("I'm feeling overwhelmed and anxious") > 0.5
    assert ORACLEAgent().can_handle("What happens if we launch early, predict the outcome") > 0.5
    assert VITAAgent().can_handle("Show invoice validation rules") == 0.0


@test("regulated advisors always include disclaimer")
def test_advisor_disclaimers():
    from synapseos.phase5.agents.advisors import VITAAgent, LEXAgent, AURUMAgent
    for agent_cls, word in ((VITAAgent, "doctor"), (LEXAgent, "lawyer"),
                            (AURUMAgent, "accountant")):
        agent = agent_cls()
        result = agent.handle("test query about " + agent.capability.keywords[0])
        assert word in result.output.lower(), f"{agent_cls.__name__}: {result.output}"


@test("advisor answers from learned facts with sources")
def test_advisor_uses_knowledge():
    from synapseos.phase5.agents.advisors import VITAAgent
    mem = _tmp_semantic()
    mem.add_fact("Sleep deprivation", "causes", "impaired concentration",
                 confidence=0.7, source="web:en.wikipedia.org")
    agent = VITAAgent()
    agent.wire(semantic_memory=mem)
    result = agent.handle("How does sleep affect health?")
    assert "Sleep deprivation" in result.output
    assert "web:en.wikipedia.org" in result.output
    assert result.confidence > 0.5


@test("ignorant advisor queues curiosity goals")
def test_advisor_queues_learning():
    from synapseos.phase5.agents.advisors import LEXAgent

    class _FakeCuriosity:
        def __init__(self): self.topics = []
        def add_user_topic(self, t, priority=1.0): self.topics.append(t)

    class _FakeLoop:
        def __init__(self): self.curiosity = _FakeCuriosity()

    mem = _tmp_semantic()
    loop = _FakeLoop()
    agent = LEXAgent()
    agent.wire(semantic_memory=mem, web_loop=loop)
    result = agent.handle("What does the new data law require?")
    assert "queued" in result.output.lower()
    assert loop.curiosity.topics, "no curiosity goals queued"


@test("SAGE grounds counselling in tracked emotional state")
def test_sage_uses_state():
    from synapseos.phase5.agents.advisors import SAGEAgent
    tracker = EmotionTracker(state_path=os.path.join(_tmp_dir(), "s.json"))
    tracker.observe("The deployment pipeline is so frustrating!")
    tracker.observe("The deployment pipeline is so frustrating!")
    agent = SAGEAgent()
    agent.wire(emotion_tracker=tracker)
    result = agent.handle("I'm feeling overwhelmed lately")
    assert "Where you seem to be" in result.output
    assert "deployment" in result.output      # recurring trigger surfaced
    assert "professional" in result.output    # disclaimer


@test("ORACLE projects from evidence, never prophecy")
def test_oracle_honest():
    from synapseos.phase5.agents.advisors import ORACLEAgent
    agent = ORACLEAgent()                     # nothing wired = no evidence
    result = agent.handle("Predict what happens if I launch next week")
    assert "I don't predict the future" in result.output
    assert "Not enough observed history" in result.output
    assert result.confidence <= 0.3
    # With learned causal facts → cites them
    mem = _tmp_semantic()
    mem.add_fact("Launching early", "causes", "more bug reports",
                 confidence=0.6, source="web:example.org")
    agent.wire(semantic_memory=mem)
    result2 = agent.handle("Predict what happens if I launch the product early")
    assert "more bug reports" in result2.output
    assert result2.confidence > 0.3


@test("controller routes counselling query to SAGE end to end")
def test_controller_advisor_e2e():
    c = _tmp_controller(enable_agents=True, enable_world_model=False)
    # _tmp_controller passes enable_agents=False by default — override needs care
    result = c.think("I'm feeling overwhelmed and anxious, how do I cope?")
    assert result.routed_to_agent == "SAGE", result.routed_to_agent
    assert "counsellor" in result.primary_output
    c.shutdown()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        # Text
        test_frustration, test_joy, test_negation, test_intensity,
        test_caps, test_emoticons, test_neutral, test_hesitation,
        # Prosody
        test_prosody_ordering, test_prosody_pauses,
        # Tracker
        test_tracker_ema, test_tracker_shift, test_tracker_patterns,
        test_tracker_prosody_fusion, test_tracker_persist,
        # Controller
        test_controller_affect, test_controller_emotion_tags,
        test_controller_pattern_in_answer, test_controller_affect_disabled,
        # Level 3 modulation
        test_mod_empathy, test_mod_risk_stress, test_mod_excited, test_mod_neutral,
        # Advisors
        test_advisor_routing, test_advisor_disclaimers, test_advisor_uses_knowledge,
        test_advisor_queues_learning, test_sage_uses_state, test_oracle_honest,
        test_controller_advisor_e2e,
    ]

    for t in tests:
        t()

    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed

    print(f"\n{'='*60}")
    print(f"Affect Tests: {passed}/{len(results)} passed\n")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        detail = f"  → {r['detail']}" if not r["passed"] else ""
        print(f"  [{status}] {r['name']}  ({r['ms']}ms){detail}")

    sys.exit(0 if failed == 0 else 1)
