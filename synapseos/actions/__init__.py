"""
SynapseOS Actions — the hands of the companion.

The capability set of an agent framework (file ops, shell, web actions,
scheduling) rebuilt with the SynapseOS signature:

  - every action passes a PermissionPolicy BEFORE it runs
    (workspace sandbox, command allowlist, size limits, dry-run mode)
  - every action is appended to a persistent audit trail
    (data/actions/audit.json) — what ran, with what args, what happened
  - no LLM decides anything: actions are explicit, deterministic calls
    routed by the same keyword/capability system as every other agent

This is the "Action Layer — the hands of the system" from the original
JARVIS vision docs, grown natively instead of bolted on.
"""

from synapseos.actions.executor import ActionExecutor, ActionResult, PermissionPolicy
from synapseos.actions.scheduler import ScheduledTask, TaskScheduler

__all__ = ["ActionExecutor", "ActionResult", "PermissionPolicy",
           "ScheduledTask", "TaskScheduler"]
