"""
Tests for SynapseOS Actions — the policy-gated, audit-logged hands.

Covers: workspace sandbox enforcement, command allowlist, audit trail
persistence, dry-run mode, scheduler tick/one-shot/recurring, KARMA agent
routing and refusal behavior. Zero network.

Run:
    python tests/test_actions.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from synapseos.actions import ActionExecutor, PermissionPolicy, TaskScheduler
from synapseos.actions.executor import OBSERVE, STANDARD, TRUSTED, FULL


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


def _tmp_executor(trust=STANDARD, **policy_kw) -> ActionExecutor:
    d = tempfile.mkdtemp(prefix="actions_test_")
    policy = PermissionPolicy(trust_level=trust, workspace=os.path.join(d, "ws"),
                              read_roots=[d], **policy_kw)
    return ActionExecutor(policy=policy,
                          audit_path=os.path.join(d, "audit.json"))


# ---------------------------------------------------------------------------
# Executor: sandbox + allowlist
# ---------------------------------------------------------------------------

@test("write confined to workspace; outside denied")
def test_write_sandbox():
    ex = _tmp_executor()
    ok = ex.act("file.write", path="notes/a.txt", content="hello")
    assert ok.ok and "wrote" in ok.output
    denied = ex.act("file.write", path="/tmp/escape.txt", content="nope")
    assert not denied.ok and "denied" in denied.error


@test("read confined to roots; outside denied")
def test_read_roots():
    ex = _tmp_executor()
    ex.act("file.write", path="a.txt", content="data123")
    r = ex.act("file.read", path=str(Path(ex.policy.workspace) / "a.txt"))
    assert r.ok and "data123" in r.output
    denied = ex.act("file.read", path="/etc/passwd")
    assert not denied.ok and "denied" in denied.error


@test("shell allowlist: allowed runs, others refused, no shell features")
def test_shell_allowlist():
    ex = _tmp_executor()
    ok = ex.act("shell.run", command="echo synapse")
    assert ok.ok and "synapse" in ok.output
    denied = ex.act("shell.run", command="rm -rf /")
    assert not denied.ok and "allowlist" in denied.error
    # pipes are not interpreted — argv only
    piped = ex.act("shell.run", command="echo a | cat /etc/passwd")
    assert piped.ok and "|" in piped.output     # echoed literally, not executed


@test("trust tiers gate tools; raising trust unlocks them")
def test_trust_tiers():
    # OBSERVE can't write
    obs = _tmp_executor(trust=OBSERVE)
    denied = obs.act("file.write", path="a.txt", content="x")
    assert not denied.ok and "trust too low" in denied.error
    # STANDARD can write but not run code (TRUSTED) or delete (FULL)
    std = _tmp_executor(trust=STANDARD)
    assert std.act("file.write", path="a.txt", content="x").ok
    assert not std.act("code.python", code="print(1)").ok
    # TRUSTED unlocks code execution
    tr = _tmp_executor(trust=TRUSTED)
    r = tr.act("code.python", code="print(6*7)")
    assert r.ok and "42" in r.output
    # FULL unlocks unrestricted shell + delete-anywhere
    full = _tmp_executor(trust=FULL)
    assert full.act("shell.run", command="echo full-power").ok


@test("FULL trust allows shell features and any command")
def test_full_shell():
    full = _tmp_executor(trust=FULL)
    r = full.act("shell.run", command="echo a && echo b")
    assert r.ok and "a" in r.output and "b" in r.output   # && works under shell=True


@test("audit trail records everything and persists")
def test_audit():
    d = tempfile.mkdtemp(prefix="actions_test_")
    policy = PermissionPolicy(workspace=os.path.join(d, "ws"), read_roots=[d])
    path = os.path.join(d, "audit.json")
    ex = ActionExecutor(policy=policy, audit_path=path)
    ex.act("file.write", path="x.txt", content="1")
    ex.act("shell.run", command="not-allowed-cmd")
    assert len(ex.audit_trail()) == 2
    ex2 = ActionExecutor(policy=policy, audit_path=path)
    trail = ex2.audit_trail()
    assert len(trail) == 2
    assert trail[0]["tool"] == "file.write" and trail[0]["ok"]
    assert trail[1]["tool"] == "shell.run" and not trail[1]["ok"]


@test("dry run logs but does nothing")
def test_dry_run():
    ex = _tmp_executor(dry_run=True)
    r = ex.act("file.write", path="ghost.txt", content="boo")
    assert r.ok and "dry-run" in r.output
    assert not (Path(ex.policy.workspace) / "ghost.txt").exists()


@test("dir list and file search work within roots")
def test_list_search():
    ex = _tmp_executor()
    ex.act("file.write", path="findme_special.txt", content="x")
    listing = ex.act("dir.list", path=ex.policy.workspace)
    assert listing.ok and "findme_special.txt" in listing.output
    found = ex.act("file.search", pattern="findme_special",
                   root=ex.policy.workspace)
    assert found.ok and "findme_special" in found.output


# ---------------------------------------------------------------------------
# Expanded tools: code, web, registry
# ---------------------------------------------------------------------------

@test("code.python runs and returns output")
def test_code_python():
    ex = _tmp_executor(trust=TRUSTED)
    r = ex.act("code.python", code="print(sum(range(10)))")
    assert r.ok and "45" in r.output


@test("registry exposes the broad toolset")
def test_registry():
    ex = _tmp_executor(trust=TRUSTED)
    names = ex.tools()
    for t in ("file.read", "shell.run", "http.get", "http.request",
              "email.send", "browser.open", "code.python", "clipboard.read",
              "notify", "screenshot"):
        assert t in names, f"missing tool {t}"
    assert len(names) >= 28


@test("email.send fails clearly without credentials")
def test_email_no_creds():
    ex = _tmp_executor(trust=TRUSTED)
    r = ex.act("email.send", to="x@y.com", subject="hi", body="test")
    assert not r.ok and "credentials" in r.error


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

@test("scheduler: one-shot runs once, recurring repeats, persists")
def test_scheduler():
    d = tempfile.mkdtemp(prefix="actions_test_")
    policy = PermissionPolicy(workspace=os.path.join(d, "ws"), read_roots=[d])
    ex = ActionExecutor(policy=policy, audit_path=os.path.join(d, "audit.json"))
    sched = TaskScheduler(ex, store_path=os.path.join(d, "sched.json"))

    once = sched.add("once", "file.append", {"path": "log.txt", "content": "A"},
                     at_ts=time.time() - 1)
    rec = sched.add("rec", "file.append", {"path": "log.txt", "content": "B"},
                    every_s=0.01)
    assert sched.tick() == 2
    time.sleep(0.02)
    assert sched.tick() == 1                      # only the recurring one again
    content = (Path(policy.workspace) / "log.txt").read_text()
    assert content == "ABB"
    # persistence
    sched2 = TaskScheduler(ex, store_path=os.path.join(d, "sched.json"))
    assert {t.name for t in sched2.tasks()} == {"once", "rec"}
    assert not [t for t in sched2.tasks() if t.name == "once"][0].enabled


@test("scheduler background thread starts and stops cleanly")
def test_scheduler_thread():
    d = tempfile.mkdtemp(prefix="actions_test_")
    policy = PermissionPolicy(workspace=os.path.join(d, "ws"), read_roots=[d])
    ex = ActionExecutor(policy=policy, audit_path=os.path.join(d, "audit.json"))
    sched = TaskScheduler(ex, store_path=os.path.join(d, "sched.json"))
    sched.add("beat", "file.append", {"path": "hb.txt", "content": "."}, every_s=0.01)
    sched.start_background(interval=0.03)
    time.sleep(0.2)
    sched.stop()
    assert sched._thread is None
    assert (Path(policy.workspace) / "hb.txt").read_text().count(".") >= 2


# ---------------------------------------------------------------------------
# KARMA agent
# ---------------------------------------------------------------------------

def _karma():
    from synapseos.phase5.agents.karma import KARMAAgent
    d = tempfile.mkdtemp(prefix="actions_test_")
    policy = PermissionPolicy(workspace=os.path.join(d, "ws"), read_roots=[d])
    ex = ActionExecutor(policy=policy, audit_path=os.path.join(d, "audit.json"))
    return KARMAAgent(executor=ex,
                      scheduler=TaskScheduler(ex, store_path=os.path.join(d, "s.json")))


@test("KARMA routes action queries, not creative ones")
def test_karma_routing():
    agent = _karma()
    assert agent.can_handle("run command echo hello") > 0.55
    assert agent.can_handle("list the files in data") > 0.55
    assert agent.can_handle("sing a song about files") == 0.0   # anti-keyword


@test("KARMA executes allowed command and shows output")
def test_karma_command():
    result = _karma().handle("run command echo synapse-hands")
    assert "synapse-hands" in result.output
    assert result.structured["ok"]


@test("KARMA refuses non-allowlisted command with the reason")
def test_karma_refusal():
    result = _karma().handle("run command sudo rm -rf /")
    assert not result.structured["ok"]
    assert "allowlist" in result.output


@test("KARMA writes files into the sandbox")
def test_karma_write():
    agent = _karma()
    result = agent.handle("create a file notes.txt with remember the milk")
    assert result.structured["ok"], result.output
    assert (Path(agent.executor.policy.workspace) / "notes.txt").read_text() \
        == "remember the milk"


@test("KARMA schedules reminders")
def test_karma_schedule():
    agent = _karma()
    result = agent.handle("remind me to stretch every 1 hour")
    assert result.structured["kind"] == "schedule"
    assert len(agent.scheduler.tasks()) == 1


@test("KARMA shows its audit trail")
def test_karma_audit():
    agent = _karma()
    agent.handle("run command echo first")
    result = agent.handle("show me your audit trail")
    assert result.structured["kind"] == "audit"
    assert "shell.run" in result.output


@test("controller routes action query to KARMA end to end")
def test_controller_karma():
    from synapseos.phase5.cognitive_controller import CognitiveController
    d = tempfile.mkdtemp(prefix="actions_test_")
    c = CognitiveController(
        memory_episodic_path=os.path.join(d, "epi.json"),
        memory_semantic_path=os.path.join(d, "sem.json"),
        kg_path=os.path.join(d, "missing_kg.json"),
        rules_path=os.path.join(d, "missing_rules.json"),
        enable_world_model=False,
        affect_state_path=os.path.join(d, "affect.json"),
    )
    result = c.think("run command echo cognitive-hands")
    assert result.routed_to_agent == "KARMA", result.routed_to_agent
    assert "cognitive-hands" in result.primary_output
    c.shutdown()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_write_sandbox, test_read_roots, test_shell_allowlist,
        test_audit, test_dry_run, test_list_search,
        test_scheduler, test_scheduler_thread,
        test_karma_routing, test_karma_command, test_karma_refusal,
        test_karma_write, test_karma_schedule, test_karma_audit,
        test_controller_karma,
    ]

    for t in tests:
        t()

    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed

    print(f"\n{'='*60}")
    print(f"Actions Tests: {passed}/{len(results)} passed\n")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        detail = f"  → {r['detail']}" if not r["passed"] else ""
        print(f"  [{status}] {r['name']}  ({r['ms']}ms){detail}")

    sys.exit(0 if failed == 0 else 1)
