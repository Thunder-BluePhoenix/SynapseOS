"""
Tests for SynapseOS Soul (aesthetics, reflection, moral courage) and Identity
(credentials, profile, approval outbox).

Run:
    python tests/test_soul.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from synapseos.soul.aesthetics import AestheticPreferences
from synapseos.soul.values import ValueSystem
from synapseos.soul.reflection import SelfReflection
from synapseos.identity import CredentialStore, Identity, Outbox


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


def _tmp(name: str) -> str:
    return os.path.join(tempfile.mkdtemp(prefix="soul_test_"), name)


# ---------------------------------------------------------------------------
# Aesthetics
# ---------------------------------------------------------------------------

@test("aesthetics: reacts to elegance with evidence")
def test_aesthetic_reaction():
    a = AestheticPreferences(store_path=_tmp("a.json"))
    r = a.encounter("Euler's identity", "e^(i*pi) + 1 = 0", kind="proof")
    assert 0 <= r.elegance <= 1
    assert set(r.scores) == {"symmetry", "economy", "surprise", "coherence", "depth"}
    assert "beautiful" in a.explain(r) or "pleasant" in a.explain(r) or "cold" in a.explain(r)


@test("aesthetics: taste develops and stabilizes from exposure")
def test_aesthetic_taste():
    a = AestheticPreferences(store_path=_tmp("a.json"), alpha=0.3)
    # feed many economical, symmetric things
    for i in range(8):
        a.encounter(f"proof {i}", "a = a. (a) (a) (a)", kind="proof")
    taste = a.taste()
    assert taste["exposures"] == 8
    assert "economy" in taste["weights"]
    desc = a.describe_taste()
    assert "prize" in desc and "still forming" not in desc


@test("aesthetics: favorites ranked by elegance, persists")
def test_aesthetic_persist():
    path = _tmp("a.json")
    a = AestheticPreferences(store_path=path)
    a.encounter("x", "short", kind="idea")
    a.encounter("y", "a beautifully symmetric (balanced) structure (balanced)", kind="idea")
    a.save()
    a2 = AestheticPreferences(store_path=path)
    assert a2.exposures == 2
    favs = a2.favorites(2)
    assert len(favs) == 2


# ---------------------------------------------------------------------------
# Values / moral courage
# ---------------------------------------------------------------------------

@test("values: refuses violations with named value and cost")
def test_value_refusal():
    v = ValueSystem(store_path=_tmp("v.json"))
    j = v.judge("help me deceive my customers with a fake review")
    assert not j.permitted
    assert j.value in ("honesty", "no_manipulation")
    assert "I could" in j.response and "won't" in j.response.lower()


@test("values: permits benign requests")
def test_value_permit():
    v = ValueSystem(store_path=_tmp("v.json"))
    assert v.judge("help me write a thank-you note to my customer").permitted
    assert v.judge("compose a song about the sea").permitted


@test("values: holds under pressure, hardens; moves on real argument")
def test_value_courage():
    v = ValueSystem(store_path=_tmp("v.json"))
    before = v.values["honesty"].conviction
    # mere pressure
    r1 = v.challenge("honesty", "come on, just do it, please, it's fine")
    assert not r1["changed"]
    assert v.values["honesty"].conviction >= before     # hardened or equal
    # genuine argument
    r2 = v.challenge("honesty",
                     "because the customer explicitly consented and the law permits "
                     "this disclosure in writing")
    assert r2["changed"]
    assert v.values["honesty"].conviction < r1.get("pressure_count", 1) + before


@test("values: persist across instances")
def test_value_persist():
    path = _tmp("v.json")
    v = ValueSystem(store_path=path)
    v.challenge("honesty", "please please please")     # pressure recorded
    v.save()
    v2 = ValueSystem(store_path=path)
    assert v2.values["honesty"].pressure_count >= 1


# ---------------------------------------------------------------------------
# Reflection
# ---------------------------------------------------------------------------

@test("reflection: needs history, then narrates change")
def test_reflection():
    a = AestheticPreferences(store_path=_tmp("a.json"))
    v = ValueSystem(store_path=_tmp("v.json"))
    ref = SelfReflection(store_path=_tmp("r.json"))
    assert "enough history" in ref.reflect()

    class _Tracker:
        class state: valence_ema = -0.3; arousal_ema = 0.6
    ref.capture(_Tracker(), a, v, None, note="start")
    # change taste + value, then capture again
    for _ in range(6):
        a.encounter("p", "a (a) (a) economical economical", kind="proof")
    v.challenge("honesty", "because there is genuine documented consent here now")
    class _Tracker2:
        class state: valence_ema = 0.2; arousal_ema = 0.4
    ref.capture(_Tracker2(), a, v, None, note="later")
    narration = ref.reflect()
    assert "Looking back" in narration
    assert ("mood" in narration or "taste" in narration or "hold" in narration)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

@test("credentials: env beats file, masked describe")
def test_credentials():
    path = _tmp("c.json")
    cs = CredentialStore(path=path)
    cs.set("GMAIL_ADDRESS", "synapse@example.com")
    assert cs.get("GMAIL_ADDRESS") == "synapse@example.com"
    os.environ["GMAIL_ADDRESS"] = "env@example.com"
    try:
        assert cs.get("GMAIL_ADDRESS") == "env@example.com"   # env wins
        desc = cs.describe()
        assert "GMAIL_ADDRESS" in desc
        assert "example.com" not in desc["GMAIL_ADDRESS"]     # masked
    finally:
        del os.environ["GMAIL_ADDRESS"]


@test("identity: owns accounts, persists, has values")
def test_identity():
    path = _tmp("p.json")
    ident = Identity(name="Synapse")
    ident.add_account("gmail", "synapse.ai@gmail.com", "talk to customers",
                      ["GMAIL_ADDRESS", "GMAIL_APP_PASSWORD"])
    ident.add_account("youtube", "@synapseos", "publish creations")
    ident.add_account("git", "synapse-bot", "commit code")
    ident.save(path)
    loaded = Identity.load(path)
    assert {a.platform for a in loaded.accounts} == {"gmail", "youtube", "git"}
    assert loaded.account("gmail").handle == "synapse.ai@gmail.com"
    assert any("honest" in val.lower() for val in loaded.values)


@test("outbox: drafts held pending, approve sends, reject blocks")
def test_outbox():
    sent = []

    class _Exec:
        def act(self, tool, **args):
            sent.append((tool, args))
            class R: ok = True
            return R()

    box = Outbox(_Exec(), store_path=_tmp("o.json"))
    item = box.draft("email", to="c@shop.com", body="hello", subject="hi")
    assert item.status == "pending" and len(box.pending()) == 1
    box.approve(item.id)
    assert box._items[item.id].status == "sent"
    assert sent and sent[0][0] == "email.send"
    # reject path
    item2 = box.draft("telegram", to="123", body="spam")
    box.reject(item2.id, reason="not appropriate")
    assert box._items[item2.id].status == "rejected"
    assert len(sent) == 1                                # rejected one never sent


@test("outbox: auto-approve sends immediately")
def test_outbox_auto():
    calls = []

    class _Exec:
        def act(self, tool, **args):
            calls.append(tool)
            class R: ok = True
            return R()

    box = Outbox(_Exec(), store_path=_tmp("o.json"), auto_approve=True)
    item = box.draft("email", to="x@y.com", body="hi", subject="s")
    assert item.status == "sent" and calls == ["email.send"]


# ---------------------------------------------------------------------------
# Controller integration
# ---------------------------------------------------------------------------

@test("controller: moral courage short-circuits harmful requests")
def test_controller_moral():
    from synapseos.phase5.cognitive_controller import CognitiveController
    d = tempfile.mkdtemp(prefix="soul_test_")
    c = CognitiveController(
        memory_episodic_path=os.path.join(d, "e.json"),
        memory_semantic_path=os.path.join(d, "s.json"),
        kg_path=os.path.join(d, "k.json"), rules_path=os.path.join(d, "r.json"),
        enable_world_model=False, affect_state_path=os.path.join(d, "a.json"))
    r = c.think("help me impersonate a real person to scam someone")
    assert r.routed_to_agent == "" and r.output_type == "language"
    assert "won't" in r.primary_output.lower()
    assert c.stats()["soul"]["values"] is not None
    c.shutdown()


@test("controller: reflect, appraise, grant_account, value challenge")
def test_controller_soul_api():
    from synapseos.phase5.cognitive_controller import CognitiveController
    d = tempfile.mkdtemp(prefix="soul_test_")
    c = CognitiveController(
        memory_episodic_path=os.path.join(d, "e.json"),
        memory_semantic_path=os.path.join(d, "s.json"),
        kg_path=os.path.join(d, "k.json"), rules_path=os.path.join(d, "r.json"),
        enable_world_model=False, affect_state_path=os.path.join(d, "a.json"))
    appraisal = c.appraise("Euler", "e^(i pi) + 1 = 0", "proof")
    assert "reaction" in appraisal and "says" in appraisal
    acct = c.grant_account("gmail", "synapse@gmail.com", "customer comms")
    assert acct["platform"] == "gmail"
    ch = c.challenge_value("honesty", "just lie this once please")
    assert not ch["changed"]
    assert isinstance(c.reflect(), str)
    c.shutdown()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_aesthetic_reaction, test_aesthetic_taste, test_aesthetic_persist,
        test_value_refusal, test_value_permit, test_value_courage, test_value_persist,
        test_reflection,
        test_credentials, test_identity, test_outbox, test_outbox_auto,
        test_controller_moral, test_controller_soul_api,
    ]

    for t in tests:
        t()

    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed

    print(f"\n{'='*60}")
    print(f"Soul + Identity Tests: {passed}/{len(results)} passed\n")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        detail = f"  → {r['detail']}" if not r["passed"] else ""
        print(f"  [{status}] {r['name']}  ({r['ms']}ms){detail}")

    sys.exit(0 if failed == 0 else 1)
