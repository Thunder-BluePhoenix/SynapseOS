"""
Phase 5 — KARMA: the action agent. Karma (कर्म) = action.

The companion's hands, exposed as an agent: file operations, sandboxed shell
commands, web fetches, and scheduling — all through the policy-gated,
audit-logged ActionExecutor. KARMA never improvises: it maps the request to
an explicit tool call, shows exactly what it did, and the audit trail is one
question away ("what have you done lately").
"""

from __future__ import annotations

import re

from synapseos.phase5.agents.advisors import AdvisorAgent
from synapseos.phase5.agents.base import AgentCapability, AgentResult


class KARMAAgent(AdvisorAgent):
    capability = AgentCapability(
        name="KARMA",
        keywords=["run command", "execute", "list files", "list the files",
                  "read file", "read the file", "write a file", "create a file",
                  "save to file", "search files", "find file", "fetch url",
                  "fetch the page", "schedule", "remind me", "every hour",
                  "every day", "audit trail", "what have you done"],
        anti_keywords=["sing", "compose", "draw", "teach"],
        base_confidence=0.6,
        description="Actions: files, sandboxed shell, web fetch, scheduling",
    )
    domain = "action"
    seed_topics = []

    def __init__(self, executor=None, scheduler=None) -> None:
        super().__init__()
        self._executor = executor
        self._scheduler = scheduler

    @property
    def executor(self):
        if self._executor is None:
            from synapseos.actions import ActionExecutor
            self._executor = ActionExecutor()
        return self._executor

    @property
    def scheduler(self):
        if self._scheduler is None:
            from synapseos.actions import TaskScheduler
            self._scheduler = TaskScheduler(self.executor)
        return self._scheduler

    # -- main ------------------------------------------------------------------

    def _handle(self, query: str, context=None) -> AgentResult:
        ql = query.lower()
        trace = ["KARMA action agent engaged"]

        if "audit" in ql or "what have you done" in ql:
            rows = self.executor.audit_trail(limit=8)
            lines = ["[KARMA — audit trail]"]
            for r in rows:
                status = "ok" if r.get("ok") else f"FAILED ({r.get('error', '')[:40]})"
                lines.append(f"  {r.get('tool')}  {status}")
            if not rows:
                lines.append("  (no actions yet)")
            return self._result(query, "\n".join(lines), 0.8, trace,
                                {"kind": "audit", "count": len(rows)})

        # schedule: "remind me / schedule ... every N minutes|hours|seconds"
        m = re.search(r"every\s+(\d+)?\s*(second|minute|hour|day)s?", ql)
        if ("schedule" in ql or "remind" in ql) and m:
            n = int(m.group(1) or 1)
            mult = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}[m.group(2)]
            note = re.sub(r"(schedule|remind me to|remind me|every.*)", "", query,
                          flags=re.IGNORECASE).strip() or "scheduled note"
            task = self.scheduler.add(
                name=note[:50], tool="file.append",
                args={"path": "reminders.log", "content": f"{note}\n"},
                every_s=n * mult)
            trace.append(f"scheduled task {task.id} every {n} {m.group(2)}(s)")
            return self._result(
                query,
                f"[KARMA — scheduled]\nTask '{task.name}' every {n} {m.group(2)}(s) "
                f"(id {task.id}). It appends to the workspace reminders.log and is "
                f"audit-logged on every run.",
                0.8, trace, {"kind": "schedule", "task_id": task.id})

        # tool mapping
        call = self._map_tool(query, ql)
        if call is None:
            return self._result(
                query,
                "[KARMA]\nI can: read/write/search files (workspace-sandboxed), "
                "list directories, run allowlisted commands, fetch a URL, schedule "
                "tasks, and show my audit trail. Phrase the action explicitly.",
                0.4, trace, {"kind": "help"})

        tool, args = call
        result = self.executor.act(tool, **args)
        trace.append(f"{tool} → {'ok' if result.ok else 'denied/failed'}")
        status = "done" if result.ok else f"refused/failed: {result.error}"
        body = result.output if result.ok else result.error
        return self._result(
            query,
            f"[KARMA — {tool}] {status}\n{body[:1200]}",
            0.8 if result.ok else 0.5, trace,
            {"kind": "action", "tool": tool, "ok": result.ok})

    # -- request → explicit tool call ------------------------------------------------

    def _map_tool(self, query: str, ql: str) -> tuple[str, dict] | None:
        m = re.search(r"run command\s+[:\"']?(.+?)[\"']?\s*$", query, re.IGNORECASE)
        if m or ql.startswith("execute "):
            cmd = m.group(1) if m else query[8:].strip()
            return "shell.run", {"command": cmd}
        m = re.search(r"read (?:the )?file\s+([\w./\- ]+)", query, re.IGNORECASE)
        if m:
            return "file.read", {"path": m.group(1).strip()}
        m = re.search(r"(?:write|create|save)(?: a)?(?: to)? file\s+([\w./\-]+)"
                      r"(?:\s+with\s+(.+))?$", query, re.IGNORECASE | re.DOTALL)
        if m:
            return "file.write", {"path": m.group(1),
                                  "content": (m.group(2) or "").strip()}
        m = re.search(r"(?:search|find) files?\s+(?:named\s+|for\s+)?([\w.\-]+)",
                      query, re.IGNORECASE)
        if m:
            return "file.search", {"pattern": m.group(1)}
        m = re.search(r"list (?:the )?files(?:\s+in\s+([\w./\-]+))?", query,
                      re.IGNORECASE)
        if m:
            return "dir.list", {"path": m.group(1) or "."}
        m = re.search(r"fetch (?:url|the page|page)?\s*(https?://\S+)", query,
                      re.IGNORECASE)
        if m:
            return "web.fetch", {"url": m.group(1)}
        return None

    def _result(self, query: str, output: str, confidence: float,
                trace: list[str], structured: dict) -> AgentResult:
        return AgentResult(
            agent_name="KARMA", query=query, output=output,
            structured={"domain": self.domain, **structured},
            confidence=confidence, latency_ms=0.0, traceable=True, trace=trace,
        )
