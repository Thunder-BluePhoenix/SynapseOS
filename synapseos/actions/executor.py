"""
Actions — tiered, audited executor. The companion's hands, at full power.

Design that makes it MORE than a generic agent runtime, not just as risky:
  - every tool declares a required TRUST LEVEL; the policy's level gates it
  - every action is appended to a persistent audit trail before returning
  - the toolset is broad (files, shell, HTTP, browser, email, messaging,
    code execution, macOS control) but each capability is unlocked
    deliberately by raising the trust level — never by accident

Trust tiers (each includes the ones below it):
  0 OBSERVE   read files, list dirs, web fetch, system info
  1 STANDARD  + write to workspace, allowlisted shell, HTTP GET, clipboard
  2 TRUSTED   + HTTP any method, send email/messages, browser control,
              code execution, notifications, scheduling, app control
  3 FULL      + unrestricted shell, file ops anywhere, delete/move

The user sets the level. "I want it all" = TRUSTED or FULL. The audit trail
is always on, at every level — that is the SynapseOS signature.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Trust levels
OBSERVE, STANDARD, TRUSTED, FULL = 0, 1, 2, 3
TRUST_NAMES = {0: "OBSERVE", 1: "STANDARD", 2: "TRUSTED", 3: "FULL"}

DEFAULT_ALLOWED_COMMANDS = [
    "ls", "cat", "head", "tail", "wc", "grep", "find", "echo", "date",
    "python --version", "python3 --version", "git status", "git log", "git diff",
    "git add", "git commit", "df", "du", "uptime", "whoami", "uname", "pwd",
    "curl --version", "node --version", "npm --version", "pip list", "which",
]


@dataclass
class PermissionPolicy:
    trust_level: int = STANDARD
    workspace: str = "data/workspace"
    read_roots: list[str] = field(default_factory=lambda: ["."])
    allowed_commands: list[str] = field(default_factory=lambda: list(DEFAULT_ALLOWED_COMMANDS))
    max_read_bytes: int = 2_000_000
    max_write_bytes: int = 5_000_000
    command_timeout: float = 60.0
    dry_run: bool = False
    credentials: dict[str, str] = field(default_factory=dict)

    @property
    def unrestricted(self) -> bool:
        return self.trust_level >= FULL

    def allows(self, required: int) -> bool:
        return self.trust_level >= required

    def can_read(self, path: Path) -> bool:
        if self.unrestricted:
            return True
        p = path.resolve()
        return any(p.is_relative_to(Path(r).resolve()) for r in self.read_roots)

    def can_write(self, path: Path) -> bool:
        if self.unrestricted:
            return True
        return path.resolve().is_relative_to(Path(self.workspace).resolve())

    def command_allowed(self, command: str) -> bool:
        if self.unrestricted:
            return True
        cmd = command.strip()
        return any(cmd == a or cmd.startswith(a + " ") for a in self.allowed_commands)

    def credential(self, key: str) -> str | None:
        """Env var first, then policy credentials dict. Never hardcoded."""
        return os.environ.get(key) or self.credentials.get(key)


@dataclass
class ActionResult:
    tool: str = ""
    args: dict = field(default_factory=dict)
    ok: bool = False
    output: str = ""
    error: str = ""
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["output"] = (d["output"] or "")[:2000]
        return d


def ok(output: str = "") -> ActionResult:
    return ActionResult(ok=True, output=output)


def fail(error: str) -> ActionResult:
    return ActionResult(ok=False, error=error)


class ActionExecutor:
    """Registry of (tool → fn, required_trust). Gate, audit, execute, audit."""

    def __init__(self, policy: PermissionPolicy | None = None,
                 audit_path: str = "data/actions/audit.json",
                 max_audit: int = 4000):
        self.policy = policy or PermissionPolicy()
        Path(self.policy.workspace).mkdir(parents=True, exist_ok=True)
        self.audit_path = Path(audit_path)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_audit = max_audit
        self._lock = threading.Lock()
        self._audit: list[dict] = []
        self._load_audit()

        # registry: name -> (fn(policy, **args) -> ActionResult, required_trust)
        self.registry: dict[str, tuple] = {}
        self._register_builtins()
        self._register_modules()

    # -- registration -------------------------------------------------------------

    def register(self, name: str, fn, required_trust: int) -> None:
        self.registry[name] = (fn, required_trust)

    def _register_builtins(self) -> None:
        self.register("file.read", _file_read, OBSERVE)
        self.register("file.write", _file_write, STANDARD)
        self.register("file.append", _file_append, STANDARD)
        self.register("file.search", _file_search, OBSERVE)
        self.register("file.delete", _file_delete, FULL)
        self.register("file.move", _file_move, FULL)
        self.register("dir.list", _dir_list, OBSERVE)
        self.register("dir.make", _dir_make, STANDARD)
        self.register("shell.run", _shell_run, STANDARD)

    def _register_modules(self) -> None:
        for module in ("web_tools", "comms", "system_tools", "browser", "code_tools"):
            try:
                mod = __import__(f"synapseos.actions.{module}", fromlist=["TOOLS"])
                for name, (fn, trust) in getattr(mod, "TOOLS", {}).items():
                    self.register(name, fn, trust)
            except Exception as e:
                print(f"[ActionExecutor] tool module '{module}' unavailable: {e}")

    def tools(self) -> dict[str, int]:
        return {name: trust for name, (_, trust) in self.registry.items()}

    # -- audit ----------------------------------------------------------------------

    def _load_audit(self) -> None:
        if self.audit_path.exists():
            try:
                self._audit = json.loads(self.audit_path.read_text()).get("actions", [])
            except Exception:
                self._audit = []

    def save(self) -> None:
        with self._lock:
            self.audit_path.write_text(json.dumps(
                {"actions": self._audit[-self.max_audit:]}, indent=2))

    def audit_trail(self, limit: int = 20) -> list[dict]:
        return self._audit[-limit:]

    def _log(self, result: ActionResult) -> ActionResult:
        with self._lock:
            self._audit.append(result.to_dict())
            if len(self._audit) > self.max_audit:
                self._audit = self._audit[-self.max_audit:]
        return result

    # -- dispatch ----------------------------------------------------------------

    def act(self, tool: str, **args) -> ActionResult:
        t0 = time.time()
        entry = self.registry.get(tool)
        if entry is None:
            return self._log(ActionResult(tool=tool, args=args, ok=False,
                                          error=f"unknown tool '{tool}'"))
        fn, required = entry
        if not self.policy.allows(required):
            return self._log(ActionResult(
                tool=tool, args=args, ok=False,
                error=f"trust too low: '{tool}' needs {TRUST_NAMES[required]}, "
                      f"policy is {TRUST_NAMES[self.policy.trust_level]}"))
        if self.policy.dry_run:
            return self._log(ActionResult(tool=tool, args=args, ok=True,
                                          output=f"[dry-run] would run {tool}"))
        try:
            result = fn(self.policy, **args)
        except TypeError as e:
            result = fail(f"bad arguments: {e}")
        except Exception as e:
            result = fail(f"{type(e).__name__}: {e}")
        result.tool, result.args = tool, args
        result.duration_ms = (time.time() - t0) * 1000
        self._log(result)
        self.save()
        return result

    def stats(self) -> dict:
        by_tool: dict[str, int] = {}
        for a in self._audit:
            by_tool[a.get("tool", "?")] = by_tool.get(a.get("tool", "?"), 0) + 1
        return {"actions_logged": len(self._audit),
                "trust_level": TRUST_NAMES[self.policy.trust_level],
                "tools_available": len(self.registry),
                "by_tool": by_tool}


# ---------------------------------------------------------------------------
# Builtin file/shell tools — fn(policy, **args) -> ActionResult
# ---------------------------------------------------------------------------

def _resolve_write(policy: PermissionPolicy, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path(policy.workspace) / path


def _file_read(policy, path: str) -> ActionResult:
    p = Path(path)
    if not policy.can_read(p):
        return fail(f"read denied outside roots: {path}")
    if not p.exists():
        return fail(f"not found: {path}")
    return ok(p.read_text(errors="replace")[: policy.max_read_bytes])


def _file_write(policy, path: str, content: str = "") -> ActionResult:
    p = _resolve_write(policy, path)
    if not policy.can_write(p):
        return fail(f"write denied outside workspace: {p}")
    if len(content) > policy.max_write_bytes:
        return fail("content exceeds write limit")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return ok(f"wrote {len(content)} chars to {p}")


def _file_append(policy, path: str, content: str = "") -> ActionResult:
    p = _resolve_write(policy, path)
    if not policy.can_write(p):
        return fail(f"write denied outside workspace: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(content)
    return ok(f"appended {len(content)} chars to {p}")


def _file_delete(policy, path: str) -> ActionResult:
    p = Path(path)
    if not policy.can_write(p):
        return fail(f"delete denied: {p}")
    if not p.exists():
        return fail(f"not found: {path}")
    p.unlink()
    return ok(f"deleted {p}")


def _file_move(policy, src: str, dst: str) -> ActionResult:
    sp, dp = Path(src), _resolve_write(policy, dst)
    if not policy.can_read(sp) or not policy.can_write(dp):
        return fail("move denied by policy")
    dp.parent.mkdir(parents=True, exist_ok=True)
    sp.rename(dp)
    return ok(f"moved {sp} → {dp}")


def _file_search(policy, pattern: str, root: str = ".",
                 max_results: int = 100) -> ActionResult:
    rp = Path(root)
    if not policy.can_read(rp):
        return fail(f"search denied: {root}")
    hits = []
    for p in rp.rglob("*"):
        if len(hits) >= max_results:
            break
        if p.is_file() and pattern.lower() in p.name.lower():
            hits.append(str(p))
    return ok("\n".join(hits) or "(no matches)")


def _dir_list(policy, path: str = ".") -> ActionResult:
    p = Path(path)
    if not policy.can_read(p):
        return fail(f"list denied: {path}")
    entries = sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir())
    return ok("\n".join(entries[:300]))


def _dir_make(policy, path: str) -> ActionResult:
    p = _resolve_write(policy, path)
    if not policy.can_write(p):
        return fail(f"mkdir denied: {p}")
    p.mkdir(parents=True, exist_ok=True)
    return ok(f"created {p}")


def _shell_run(policy, command: str) -> ActionResult:
    if not policy.command_allowed(command):
        return fail(f"command not in allowlist (raise trust to FULL for any "
                    f"command): '{command}'")
    if policy.unrestricted:
        proc = subprocess.run(command, shell=True, capture_output=True, text=True,
                              timeout=policy.command_timeout)
    else:
        proc = subprocess.run(shlex.split(command), capture_output=True, text=True,
                              timeout=policy.command_timeout)
    out = proc.stdout[:6000] + (proc.stderr[:2000] if proc.returncode else "")
    r = ActionResult(ok=proc.returncode == 0, output=out)
    if proc.returncode:
        r.error = f"exit {proc.returncode}"
    return r
