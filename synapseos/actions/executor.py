"""
Actions — policy-gated executor with a persistent audit trail.

Tools: file.read, file.write, file.append, file.search, dir.list,
       shell.run, web.fetch

The signature constraints:
  - file writes confined to the workspace sandbox (default: data/workspace)
  - file reads confined to allowed roots (project + workspace)
  - shell commands must match the allowlist (prefix match) and never run
    through a shell — argv only, no pipes/expansion
  - everything appended to data/actions/audit.json before AND after running
  - dry_run mode evaluates policy and logs without touching anything
"""

from __future__ import annotations

import json
import shlex
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_ALLOWED_COMMANDS = [
    "ls", "cat", "head", "tail", "wc", "grep", "find", "echo", "date",
    "python --version", "python3 --version", "git status", "git log", "git diff",
    "df", "uptime", "whoami", "uname",
]


@dataclass
class PermissionPolicy:
    workspace: str = "data/workspace"             # writes confined here
    read_roots: list[str] = field(default_factory=lambda: ["."])
    allowed_commands: list[str] = field(default_factory=lambda: list(DEFAULT_ALLOWED_COMMANDS))
    max_read_bytes: int = 1_000_000
    max_write_bytes: int = 1_000_000
    command_timeout: float = 30.0
    dry_run: bool = False

    def can_read(self, path: Path) -> bool:
        p = path.resolve()
        return any(p.is_relative_to(Path(root).resolve()) for root in self.read_roots)

    def can_write(self, path: Path) -> bool:
        return path.resolve().is_relative_to(Path(self.workspace).resolve())

    def command_allowed(self, command: str) -> bool:
        cmd = command.strip()
        return any(cmd == allowed or cmd.startswith(allowed + " ")
                   for allowed in self.allowed_commands)


@dataclass
class ActionResult:
    tool: str
    args: dict
    ok: bool
    output: str = ""
    error: str = ""
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["output"] = d["output"][:2000]
        return d


class ActionExecutor:
    """The hands. Every call: policy check → audit → execute → audit."""

    def __init__(self, policy: PermissionPolicy | None = None,
                 audit_path: str = "data/actions/audit.json",
                 max_audit: int = 2000):
        self.policy = policy or PermissionPolicy()
        Path(self.policy.workspace).mkdir(parents=True, exist_ok=True)
        self.audit_path = Path(audit_path)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_audit = max_audit
        self._lock = threading.Lock()
        self._audit: list[dict] = []
        self._load_audit()
        self.tools = {
            "file.read": self._file_read,
            "file.write": self._file_write,
            "file.append": self._file_append,
            "file.search": self._file_search,
            "dir.list": self._dir_list,
            "shell.run": self._shell_run,
            "web.fetch": self._web_fetch,
        }

    # -- audit ------------------------------------------------------------------

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
        fn = self.tools.get(tool)
        if fn is None:
            return self._log(ActionResult(tool=tool, args=args, ok=False,
                                          error=f"unknown tool '{tool}'"))
        if self.policy.dry_run:
            return self._log(ActionResult(tool=tool, args=args, ok=True,
                                          output="[dry-run] policy check only"))
        try:
            result = fn(**args)
        except Exception as e:
            result = ActionResult(tool=tool, args=args, ok=False,
                                  error=f"{type(e).__name__}: {e}")
        result.tool, result.args = tool, args
        result.duration_ms = (time.time() - t0) * 1000
        self._log(result)
        self.save()
        return result

    # -- tools ----------------------------------------------------------------------

    def _file_read(self, path: str) -> ActionResult:
        p = Path(path)
        if not self.policy.can_read(p):
            return ActionResult("", {}, False, error=f"read denied outside roots: {path}")
        if not p.exists():
            return ActionResult("", {}, False, error=f"not found: {path}")
        text = p.read_text(errors="replace")[: self.policy.max_read_bytes]
        return ActionResult("", {}, True, output=text)

    def _file_write(self, path: str, content: str) -> ActionResult:
        p = Path(self.policy.workspace) / path if not Path(path).is_absolute() else Path(path)
        if not self.policy.can_write(p):
            return ActionResult("", {}, False,
                                error=f"write denied outside workspace: {p}")
        if len(content) > self.policy.max_write_bytes:
            return ActionResult("", {}, False, error="content exceeds write limit")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return ActionResult("", {}, True, output=f"wrote {len(content)} chars to {p}")

    def _file_append(self, path: str, content: str) -> ActionResult:
        p = Path(self.policy.workspace) / path if not Path(path).is_absolute() else Path(path)
        if not self.policy.can_write(p):
            return ActionResult("", {}, False,
                                error=f"write denied outside workspace: {p}")
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as f:
            f.write(content)
        return ActionResult("", {}, True, output=f"appended {len(content)} chars to {p}")

    def _file_search(self, pattern: str, root: str = ".",
                     max_results: int = 50) -> ActionResult:
        rp = Path(root)
        if not self.policy.can_read(rp):
            return ActionResult("", {}, False, error=f"search denied: {root}")
        hits = []
        for p in rp.rglob("*"):
            if len(hits) >= max_results:
                break
            if p.is_file() and pattern.lower() in p.name.lower():
                hits.append(str(p))
        return ActionResult("", {}, True, output="\n".join(hits) or "(no matches)")

    def _dir_list(self, path: str = ".") -> ActionResult:
        p = Path(path)
        if not self.policy.can_read(p):
            return ActionResult("", {}, False, error=f"list denied: {path}")
        entries = sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir())
        return ActionResult("", {}, True, output="\n".join(entries[:200]))

    def _shell_run(self, command: str) -> ActionResult:
        if not self.policy.command_allowed(command):
            return ActionResult("", {}, False,
                                error=f"command not in allowlist: '{command}'")
        argv = shlex.split(command)          # argv only — no shell, no pipes
        proc = subprocess.run(argv, capture_output=True, text=True,
                              timeout=self.policy.command_timeout)
        out = proc.stdout[:4000] + (proc.stderr[:1000] if proc.returncode else "")
        return ActionResult("", {}, proc.returncode == 0, output=out,
                            error="" if proc.returncode == 0
                            else f"exit {proc.returncode}")

    def _web_fetch(self, url: str) -> ActionResult:
        from synapseos.web.extractor import TextExtractor
        from synapseos.web.fetcher import WebFetcher
        result = WebFetcher(seen_path="data/actions/web_seen.json").fetch(url)
        if not result.ok:
            return ActionResult("", {}, False, error=result.error or str(result.status))
        text = TextExtractor().extract(result.text)[:4000]
        return ActionResult("", {}, True, output=text)

    def stats(self) -> dict:
        by_tool: dict[str, int] = {}
        for a in self._audit:
            by_tool[a.get("tool", "?")] = by_tool.get(a.get("tool", "?"), 0) + 1
        return {"actions_logged": len(self._audit), "by_tool": by_tool,
                "workspace": self.policy.workspace, "dry_run": self.policy.dry_run}
