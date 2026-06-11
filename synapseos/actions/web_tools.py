"""
Actions — web/API tools. Full HTTP, not just fetch.

  http.get      (STANDARD) — GET a URL, return body
  http.request  (TRUSTED)  — any method (POST/PUT/DELETE...), headers, JSON body
  web.download   (TRUSTED) — save a URL to the workspace
  api.call       (TRUSTED) — http.request + JSON parse convenience

stdlib urllib; optional `requests` backend used automatically if installed.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from synapseos.actions.executor import STANDARD, TRUSTED, ActionResult, fail, ok

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


def _http(policy, url: str, method: str = "GET", headers: dict | None = None,
          body=None, timeout: float = 30.0) -> ActionResult:
    headers = headers or {}
    headers.setdefault("User-Agent", "SynapseOS-KARMA/1.0")
    data = None
    if body is not None:
        if isinstance(body, (dict, list)):
            data = json.dumps(body).encode()
            headers.setdefault("Content-Type", "application/json")
        else:
            data = str(body).encode()
    try:
        if _HAS_REQUESTS:
            resp = _requests.request(method, url, headers=headers, data=data,
                                     timeout=timeout)
            text, status = resp.text, resp.status_code
        else:
            req = urllib.request.Request(url, data=data, headers=headers,
                                         method=method.upper())
            with urllib.request.urlopen(req, timeout=timeout) as r:
                text = r.read(2_000_000).decode("utf-8", errors="replace")
                status = r.status
        result = ok(f"[{status}] {text[:6000]}")
        result.error = "" if 200 <= status < 400 else f"HTTP {status}"
        result.ok = 200 <= status < 400
        return result
    except Exception as e:
        return fail(f"{type(e).__name__}: {e}")


def http_get(policy, url: str, headers: dict | None = None) -> ActionResult:
    return _http(policy, url, "GET", headers)


def http_request(policy, url: str, method: str = "GET", headers: dict | None = None,
                 body=None) -> ActionResult:
    return _http(policy, url, method, headers, body)


def web_download(policy, url: str, filename: str = "") -> ActionResult:
    name = filename or url.rstrip("/").split("/")[-1] or "download.bin"
    dest = Path(policy.workspace) / name
    if not policy.can_write(dest):
        return fail(f"download target outside workspace: {dest}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SynapseOS-KARMA/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read(50_000_000)
        dest.write_bytes(data)
        return ok(f"downloaded {len(data)} bytes → {dest}")
    except Exception as e:
        return fail(f"{type(e).__name__}: {e}")


def api_call(policy, url: str, method: str = "GET", headers: dict | None = None,
             body=None) -> ActionResult:
    r = _http(policy, url, method, headers, body)
    if r.ok:
        try:
            payload = r.output.split("] ", 1)[-1]
            parsed = json.loads(payload)
            r.output = json.dumps(parsed, indent=2)[:6000]
        except Exception:
            pass
    return r


TOOLS = {
    "http.get": (http_get, STANDARD),
    "http.request": (http_request, TRUSTED),
    "web.download": (web_download, TRUSTED),
    "api.call": (api_call, TRUSTED),
}
