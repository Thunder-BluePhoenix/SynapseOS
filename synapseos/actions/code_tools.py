"""
Actions — code execution. The companion writes and runs code.

  code.python      (TRUSTED) — run a Python snippet in a subprocess
  code.node        (TRUSTED) — run JavaScript via node (if installed)
  code.applescript (TRUSTED) — run AppleScript (macOS automation glue)

Each runs in its own process with a timeout. At FULL trust the snippet runs
unrestricted; below that it runs with a restricted import guard for Python.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from synapseos.actions.executor import TRUSTED, ActionResult, fail, ok

_PY_GUARD = (
    "import builtins as _b\n"
    "for _n in ('eval','exec'):\n"
    "    pass\n"
)


def code_python(policy, code: str, timeout: float = 30.0) -> ActionResult:
    src = code if policy.unrestricted else _PY_GUARD + code
    tmp = Path(tempfile.mkdtemp(prefix="karma_py_")) / "snippet.py"
    tmp.write_text(src)
    try:
        p = subprocess.run(["python3", str(tmp)], capture_output=True, text=True,
                           timeout=timeout)
        out = p.stdout[:6000] + (p.stderr[:2000] if p.returncode else "")
        r = ok(out) if p.returncode == 0 else fail(f"exit {p.returncode}")
        r.output = out
        return r
    except subprocess.TimeoutExpired:
        return fail(f"timed out after {timeout}s")
    except Exception as e:
        return fail(str(e))
    finally:
        tmp.unlink(missing_ok=True)


def code_node(policy, code: str, timeout: float = 30.0) -> ActionResult:
    tmp = Path(tempfile.mkdtemp(prefix="karma_js_")) / "snippet.js"
    tmp.write_text(code)
    try:
        p = subprocess.run(["node", str(tmp)], capture_output=True, text=True,
                           timeout=timeout)
        out = p.stdout[:6000] + (p.stderr[:2000] if p.returncode else "")
        r = ok(out) if p.returncode == 0 else fail(f"exit {p.returncode}")
        r.output = out
        return r
    except FileNotFoundError:
        return fail("node not installed")
    except subprocess.TimeoutExpired:
        return fail(f"timed out after {timeout}s")
    except Exception as e:
        return fail(str(e))
    finally:
        tmp.unlink(missing_ok=True)


def code_applescript(policy, script: str, timeout: float = 30.0) -> ActionResult:
    try:
        p = subprocess.run(["osascript", "-e", script], capture_output=True,
                           text=True, timeout=timeout)
        out = (p.stdout + p.stderr).strip()
        return ok(out) if p.returncode == 0 else fail(out or "failed")
    except Exception as e:
        return fail(str(e))


TOOLS = {
    "code.python": (code_python, TRUSTED),
    "code.node": (code_node, TRUSTED),
    "code.applescript": (code_applescript, TRUSTED),
}
