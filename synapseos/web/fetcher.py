"""
Web Learning Loop — Fetcher.

Polite HTTP fetching with safety rails:
  - robots.txt respected (cached per host, fail-open with a log line)
  - per-domain rate limiting
  - session page budget
  - URL dedup via persistent seen-store
  - GET-only, size-capped, never executes JS
  - optional domain allowlist (None = open web)

Wikipedia is the preferred structured source for topic lookups (clean JSON,
no API key), but any HTTP(S) page can be fetched.

FakeFetcher is the injection point for zero-network tests.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from dataclasses import asdict, dataclass, field
from pathlib import Path

WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
WIKI_SEARCH_URL = (
    "https://en.wikipedia.org/w/api.php"
    "?action=opensearch&format=json&limit={limit}&search={query}"
)


@dataclass
class FetchResult:
    url: str
    status: int                 # HTTP status; 0 = transport error, -1 = blocked by policy
    content_type: str = ""
    text: str = ""
    fetched_at: float = field(default_factory=time.time)
    from_cache: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300 and bool(self.text)


@dataclass
class FetchPolicy:
    """Safety and scope configuration for web access."""
    allowed_domains: list[str] | None = None   # None = open web; else suffix match
    max_pages_per_session: int = 30
    min_interval_per_domain: float = 1.0       # seconds between hits to same host
    timeout: float = 10.0
    max_bytes: int = 2_000_000
    user_agent: str = "SynapseOS-LearningLoop/0.1 (research; local)"
    respect_robots: bool = True

    def domain_allowed(self, netloc: str) -> bool:
        if self.allowed_domains is None:
            return True
        host = netloc.split(":")[0].lower()
        return any(host == d or host.endswith("." + d) for d in self.allowed_domains)


def url_hash(url: str) -> str:
    """Stable short hash of a canonicalized URL."""
    parts = urllib.parse.urlsplit(url.strip())
    canon = urllib.parse.urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), parts.query, "")
    )
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


class BaseFetcher:
    """Interface — inject FakeFetcher in tests, WebFetcher in production."""

    def fetch(self, url: str) -> FetchResult:
        raise NotImplementedError

    def wiki_summary(self, title: str) -> FetchResult:
        raise NotImplementedError

    def wiki_search(self, query: str, limit: int = 3) -> list[str]:
        raise NotImplementedError


class WebFetcher(BaseFetcher):
    """Real HTTP fetcher. Stdlib urllib; no external dependencies."""

    def __init__(
        self,
        policy: FetchPolicy | None = None,
        seen_path: str = "data/web/seen_urls.json",
    ):
        self.policy = policy or FetchPolicy()
        self.seen_path = Path(seen_path)
        self.seen_path.parent.mkdir(parents=True, exist_ok=True)
        self._seen: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._robots_cache: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._last_hit: dict[str, float] = {}
        self._session_pages = 0
        self._load_seen()

    # -- seen store ---------------------------------------------------------

    def _load_seen(self) -> None:
        if self.seen_path.exists():
            try:
                self._seen = json.loads(self.seen_path.read_text())
            except Exception:
                self._seen = {}

    def save(self) -> None:
        with self._lock:
            self.seen_path.write_text(json.dumps(self._seen, indent=2))

    def seen(self, url: str) -> bool:
        return url_hash(url) in self._seen

    def mark_seen(self, url: str, status: int) -> None:
        with self._lock:
            self._seen[url_hash(url)] = {
                "url": url,
                "status": status,
                "fetched_at": time.time(),
            }

    # -- policy checks ------------------------------------------------------

    def _allowed(self, url: str, api: bool = False) -> tuple[bool, str]:
        parts = urllib.parse.urlsplit(url)
        if parts.scheme not in ("http", "https"):
            return False, f"scheme '{parts.scheme}' not allowed"
        if not self.policy.domain_allowed(parts.netloc):
            return False, f"domain '{parts.netloc}' not in allowlist"
        if self._session_pages >= self.policy.max_pages_per_session:
            return False, "session page budget exhausted"
        # robots.txt governs crawling, not sanctioned API access (e.g. Wikipedia
        # disallows /api/ for crawlers while documenting it as the API interface).
        if not api and self.policy.respect_robots and not self._robots_ok(url):
            return False, "disallowed by robots.txt"
        return True, ""

    def _robots_ok(self, url: str) -> bool:
        parts = urllib.parse.urlsplit(url)
        key = f"{parts.scheme}://{parts.netloc}"
        if key not in self._robots_cache:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(key + "/robots.txt")
            try:
                rp.read()
                self._robots_cache[key] = rp
            except Exception:
                # Fail-open: unreachable robots.txt should not block learning
                print(f"[Fetcher] robots.txt unreachable for {key} — allowing")
                self._robots_cache[key] = None
        rp = self._robots_cache[key]
        if rp is None:
            return True
        try:
            return rp.can_fetch(self.policy.user_agent, url)
        except Exception:
            return True

    def _throttle(self, netloc: str) -> None:
        last = self._last_hit.get(netloc, 0.0)
        wait = self.policy.min_interval_per_domain - (time.time() - last)
        if wait > 0:
            time.sleep(wait)
        self._last_hit[netloc] = time.time()

    # -- fetching -----------------------------------------------------------

    def fetch(self, url: str, api: bool = False) -> FetchResult:
        ok, reason = self._allowed(url, api=api)
        if not ok:
            return FetchResult(url=url, status=-1, error=reason)

        netloc = urllib.parse.urlsplit(url).netloc
        self._throttle(netloc)

        req = urllib.request.Request(
            url, headers={"User-Agent": self.policy.user_agent}, method="GET"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.policy.timeout) as resp:
                final_url = resp.geturl()
                # Redirects must also satisfy policy
                if final_url != url:
                    final_netloc = urllib.parse.urlsplit(final_url).netloc
                    if not self.policy.domain_allowed(final_netloc):
                        return FetchResult(
                            url=url, status=-1,
                            error=f"redirected to disallowed domain '{final_netloc}'",
                        )
                raw = resp.read(self.policy.max_bytes)
                ctype = resp.headers.get("Content-Type", "")
                charset = resp.headers.get_content_charset() or "utf-8"
                text = raw.decode(charset, errors="replace")
                self._session_pages += 1
                self.mark_seen(url, resp.status)
                return FetchResult(
                    url=final_url, status=resp.status, content_type=ctype, text=text
                )
        except urllib.error.HTTPError as e:
            self.mark_seen(url, e.code)
            return FetchResult(url=url, status=e.code, error=str(e))
        except Exception as e:
            return FetchResult(url=url, status=0, error=str(e))

    # -- Wikipedia helpers --------------------------------------------------

    def wiki_summary(self, title: str) -> FetchResult:
        quoted = urllib.parse.quote(title.replace(" ", "_"))
        return self.fetch(WIKI_SUMMARY_URL.format(title=quoted), api=True)

    def wiki_search(self, query: str, limit: int = 3) -> list[str]:
        url = WIKI_SEARCH_URL.format(limit=limit, query=urllib.parse.quote(query))
        result = self.fetch(url, api=True)
        if not result.ok:
            return []
        try:
            data = json.loads(result.text)
            return list(data[1])[:limit]
        except Exception:
            return []

    def stats(self) -> dict:
        return {
            "seen_urls": len(self._seen),
            "session_pages": self._session_pages,
            "budget": self.policy.max_pages_per_session,
        }


class FakeFetcher(BaseFetcher):
    """
    Test double — returns canned pages, records every call. Zero network.

    pages: url → body (HTML or raw text)
    wiki:  title → summary dict (mirrors Wikipedia REST summary JSON)
    searches: query → list of titles
    """

    def __init__(
        self,
        pages: dict[str, str] | None = None,
        wiki: dict[str, dict] | None = None,
        searches: dict[str, list[str]] | None = None,
    ):
        self.pages = pages or {}
        self.wiki = wiki or {}
        self.searches = searches or {}
        self.calls: list[tuple[str, str]] = []

    def fetch(self, url: str) -> FetchResult:
        self.calls.append(("fetch", url))
        if url in self.pages:
            return FetchResult(url=url, status=200, content_type="text/html",
                               text=self.pages[url])
        return FetchResult(url=url, status=404, error="not in fixture set")

    def wiki_summary(self, title: str) -> FetchResult:
        self.calls.append(("wiki_summary", title))
        if title in self.wiki:
            return FetchResult(
                url=WIKI_SUMMARY_URL.format(title=title),
                status=200, content_type="application/json",
                text=json.dumps(self.wiki[title]),
            )
        return FetchResult(url=title, status=404, error="not in fixture set")

    def wiki_search(self, query: str, limit: int = 3) -> list[str]:
        self.calls.append(("wiki_search", query))
        if query in self.searches:
            return self.searches[query][:limit]
        # Default: any fixture title whose words overlap the query
        q = query.lower()
        hits = [t for t in self.wiki if any(w in q for w in t.lower().split())]
        return hits[:limit]

    def save(self) -> None:
        pass

    def stats(self) -> dict:
        return {"calls": len(self.calls)}
