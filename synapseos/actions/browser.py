"""
Actions — browser automation. Real, install-free, via macOS AppleScript.

Drives Safari or Chrome through osascript — navigate, read page text, run
JavaScript, and screenshot — with no playwright/selenium dependency. If
playwright IS installed, a headless path is used instead for reliability.

  browser.open   (TRUSTED) — open a URL in the browser
  browser.read   (TRUSTED) — get the visible text / DOM text of the page
  browser.js     (TRUSTED) — run JavaScript in the page, return its result
  browser.screenshot (TRUSTED) — capture the browser window

This is the "computer use / browser" capability, native and audited.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from synapseos.actions.executor import TRUSTED, ActionResult, fail, ok

try:
    from playwright.sync_api import sync_playwright  # type: ignore
    _HAS_PLAYWRIGHT = True
except ImportError:
    _HAS_PLAYWRIGHT = False


def _osa(script: str, timeout: float = 30.0) -> tuple[int, str, str]:
    p = subprocess.run(["osascript", "-e", script], capture_output=True,
                       text=True, timeout=timeout)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def _browser_app(policy) -> str:
    return policy.credential("BROWSER_APP") or "Safari"


def browser_open(policy, url: str) -> ActionResult:
    app = _browser_app(policy)
    if app.lower() == "safari":
        script = f'tell application "Safari" to open location "{url}"'
    else:
        script = (f'tell application "{app}" to open location "{url}"')
    rc, _, err = _osa(script)
    if rc:
        return fail(err or "browser open failed")
    return ok(f"opened {url} in {app}")


def browser_read(policy, url: str = "") -> ActionResult:
    app = _browser_app(policy)
    if _HAS_PLAYWRIGHT and url:
        try:
            with sync_playwright() as pw:
                browser = pw.webkit.launch()
                page = browser.new_page()
                page.goto(url, timeout=30000)
                text = page.inner_text("body")
                browser.close()
            return ok(text[:6000])
        except Exception as e:
            return fail(f"playwright: {e}")
    if url:
        browser_open(policy, url)
        time.sleep(2.0)
    if app.lower() == "safari":
        script = ('tell application "Safari" to get text of '
                  'document 1')
    else:
        script = (f'tell application "{app}" to execute front window\'s '
                  f'active tab javascript "document.body.innerText"')
    rc, out, err = _osa(script, timeout=40)
    if rc:
        return fail(err or "browser read failed (enable 'Allow JavaScript from "
                    "Apple Events' in the browser's Develop menu)")
    return ok(out[:6000])


def browser_js(policy, script: str, url: str = "") -> ActionResult:
    app = _browser_app(policy)
    if url:
        browser_open(policy, url)
        time.sleep(2.0)
    safe = script.replace('"', '\\"')
    if app.lower() == "safari":
        osa = f'tell application "Safari" to do JavaScript "{safe}" in document 1'
    else:
        osa = (f'tell application "{app}" to execute front window\'s active tab '
               f'javascript "{safe}"')
    rc, out, err = _osa(osa, timeout=40)
    if rc:
        return fail(err or "JavaScript execution failed")
    return ok(out[:6000])


def browser_screenshot(policy, filename: str = "") -> ActionResult:
    name = filename or f"browser_{int(time.time())}.png"
    dest = Path(policy.workspace) / name
    if not policy.can_write(dest):
        return fail(f"screenshot target outside workspace: {dest}")
    # capture the frontmost window
    rc, out, err = _osa('tell application "System Events" to get name of '
                        'first application process whose frontmost is true')
    p = subprocess.run(["screencapture", "-x", "-o", str(dest)],
                       capture_output=True, text=True, timeout=20)
    if p.returncode:
        return fail(p.stderr.strip() or "capture failed")
    return ok(f"browser screenshot → {dest}")


TOOLS = {
    "browser.open": (browser_open, TRUSTED),
    "browser.read": (browser_read, TRUSTED),
    "browser.js": (browser_js, TRUSTED),
    "browser.screenshot": (browser_screenshot, TRUSTED),
}
