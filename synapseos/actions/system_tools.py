"""
Actions — macOS system control. The companion's reach into the machine.

  clipboard.read  (STANDARD) — pbpaste
  clipboard.write (TRUSTED)  — pbcopy
  notify          (TRUSTED)  — macOS notification banner
  say             (TRUSTED)  — speak text aloud
  screenshot      (TRUSTED)  — capture screen to workspace
  app.open        (TRUSTED)  — open an app or URL
  process.list    (OBSERVE)  — running processes (ps)
  system.info     (OBSERVE)  — uname/uptime/disk snapshot

All via stdlib subprocess to standard macOS binaries — no install.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from synapseos.actions.executor import (OBSERVE, STANDARD, TRUSTED, ActionResult,
                                        fail, ok)


def _run(argv: list[str], timeout: float = 20.0) -> tuple[int, str, str]:
    p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def clipboard_read(policy) -> ActionResult:
    try:
        _, out, _ = _run(["pbpaste"])
        return ok(out[:4000])
    except Exception as e:
        return fail(str(e))


def clipboard_write(policy, text: str) -> ActionResult:
    try:
        p = subprocess.run(["pbcopy"], input=text, text=True, timeout=10)
        return ok("copied to clipboard") if p.returncode == 0 else fail("pbcopy failed")
    except Exception as e:
        return fail(str(e))


def notify(policy, text: str, title: str = "SynapseOS") -> ActionResult:
    script = f'display notification "{text}" with title "{title}"'
    rc, _, err = _run(["osascript", "-e", script])
    return ok("notification shown") if rc == 0 else fail(err.strip() or "failed")


def say(policy, text: str, voice: str = "") -> ActionResult:
    argv = ["say"] + (["-v", voice] if voice else []) + [text]
    rc, _, err = _run(argv, timeout=60)
    return ok("spoke aloud") if rc == 0 else fail(err.strip() or "failed")


def screenshot(policy, filename: str = "") -> ActionResult:
    name = filename or f"screen_{int(time.time())}.png"
    dest = Path(policy.workspace) / name
    if not policy.can_write(dest):
        return fail(f"screenshot target outside workspace: {dest}")
    rc, _, err = _run(["screencapture", "-x", str(dest)])
    return ok(f"screenshot → {dest}") if rc == 0 else fail(err.strip() or "failed")


def app_open(policy, target: str) -> ActionResult:
    rc, _, err = _run(["open", target])
    return ok(f"opened {target}") if rc == 0 else fail(err.strip() or "failed")


def process_list(policy, filter: str = "") -> ActionResult:
    rc, out, _ = _run(["ps", "-eo", "pid,pcpu,pmem,comm"])
    lines = out.splitlines()
    if filter:
        lines = [lines[0]] + [l for l in lines[1:] if filter.lower() in l.lower()]
    return ok("\n".join(lines[:60]))


def system_info(policy) -> ActionResult:
    parts = []
    for argv in (["uname", "-a"], ["uptime"], ["df", "-h", "/"]):
        try:
            _, out, _ = _run(argv)
            parts.append(out.strip())
        except Exception:
            pass
    return ok("\n".join(parts))


TOOLS = {
    "clipboard.read": (clipboard_read, STANDARD),
    "clipboard.write": (clipboard_write, TRUSTED),
    "notify": (notify, TRUSTED),
    "say": (say, TRUSTED),
    "screenshot": (screenshot, TRUSTED),
    "app.open": (app_open, TRUSTED),
    "process.list": (process_list, OBSERVE),
    "system.info": (system_info, OBSERVE),
}
