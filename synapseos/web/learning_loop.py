"""
Web Learning Loop — Orchestrator.

The full learning cycle:

    goal → fetch → extract text → world_model.perceive (surprise gate)
         → extract facts → belief revision → KG overlay + semantic memory
         → session log

Three run modes:
  learn(topic)          — pursue one explicit topic now
  run(n_goals)          — generate goals from curiosity, drain the queue
  start_background()    — daemon thread, one goal per tick while system idle
                          (same pattern as the phase4 IdleLoop)
"""

from __future__ import annotations

import json
import threading
import time
import urllib.parse
from dataclasses import asdict, dataclass, field
from pathlib import Path

from synapseos.web.belief import BeliefReviser
from synapseos.web.curiosity import CuriosityEngine, LearningGoal
from synapseos.web.extractor import FactExtractor, TextExtractor
from synapseos.web.fetcher import BaseFetcher
from synapseos.web.media import MediaIngester, classify_media


@dataclass
class LearnReport:
    goal_id: str
    topic: str
    urls_fetched: list[str] = field(default_factory=list)
    text_chars: int = 0
    facts_extracted: int = 0
    facts_added: int = 0
    facts_revised: int = 0
    facts_rejected: int = 0
    facts_reinforced: int = 0
    surprise: float = 0.0
    errors: list[str] = field(default_factory=list)
    duration: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class WebLearningLoop:
    """Curiosity-driven web learning orchestrator."""

    def __init__(
        self,
        fetcher: BaseFetcher,
        curiosity: CuriosityEngine,
        reviser: BeliefReviser,
        world_model=None,
        text_extractor: TextExtractor | None = None,
        fact_extractor: FactExtractor | None = None,
        media_ingester=None,           # MediaIngester | FakeMediaIngester | None
        data_dir: str = "data/web",
        max_pages_per_goal: int = 3,
        idle_threshold: float = 10.0,
    ):
        self.fetcher = fetcher
        self.curiosity = curiosity
        self.reviser = reviser
        self.world_model = world_model
        self.text_extractor = text_extractor or TextExtractor()
        self.fact_extractor = fact_extractor or FactExtractor()
        # Default real ingester is lazy — whisper/yt-dlp only load when media URLs appear
        self.media = media_ingester if media_ingester is not None else MediaIngester()
        self.max_pages_per_goal = max_pages_per_goal
        self.idle_threshold = idle_threshold

        self.log_path = Path(data_dir) / "session_log.json"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._sessions: list[dict] = []
        self._load_log()

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_active = time.time()
        self._lock = threading.Lock()
        self._goals_pursued = 0

    # -- public API -------------------------------------------------------------

    def learn(self, topic: str) -> LearnReport:
        """Learn one explicit topic right now."""
        goal = self.curiosity.add_user_topic(topic)
        report = self._pursue(goal)
        self._persist()
        return report

    def learn_url(self, url: str, topic: str | None = None) -> LearnReport:
        """Learn from one explicit URL — a web page, YouTube video, or audio file."""
        topic = topic or url
        goal = self.curiosity.add_user_topic(topic)
        goal.urls = [url]
        report = self._pursue(goal)
        self._persist()
        return report

    def run(self, n_goals: int = 5) -> list[LearnReport]:
        """Generate goals from curiosity, then pursue the top n."""
        self.curiosity.generate_goals(max_goals=n_goals * 2)
        reports = []
        for _ in range(n_goals):
            goal = self.curiosity.queue.pop()
            if goal is None:
                break
            reports.append(self._pursue(goal))
        self._persist()
        return reports

    # -- core cycle -------------------------------------------------------------

    def _pursue(self, goal: LearningGoal) -> LearnReport:
        started = time.time()
        report = LearnReport(goal_id=goal.id, topic=goal.topic)
        surprises: list[float] = []

        urls = list(goal.urls)
        wiki_titles: list[str] = []
        if not urls:
            try:
                wiki_titles = self.fetcher.wiki_search(goal.topic,
                                                       limit=self.max_pages_per_goal)
            except Exception as e:
                report.errors.append(f"search failed: {e}")
            if not wiki_titles:
                # The REST summary endpoint resolves titles via redirects —
                # try the topic directly before giving up.
                wiki_titles = [goal.topic]

        # Pages = explicit URLs + Wikipedia summaries for matched titles
        sources: list[tuple[str, str]] = [("url", u) for u in urls]
        sources += [("wiki", t) for t in wiki_titles]
        sources = sources[: self.max_pages_per_goal]

        if not sources:
            report.errors.append("no sources found for topic")

        for kind, ref in sources:
            text, url = self._fetch_text(kind, ref, report)
            if not text:
                continue
            report.urls_fetched.append(url)
            report.text_chars += len(text)

            # World model perceives what was read — surprise gates episodic storage
            if self.world_model is not None:
                try:
                    percept = self.world_model.perceive(text[:2000], tags=["web", url])
                    surprises.append(percept.surprise)
                except Exception as e:
                    report.errors.append(f"perceive failed: {e}")

            facts = self.fact_extractor.extract(text, topic_hint=goal.topic)
            report.facts_extracted += len(facts)
            for fact in facts:
                try:
                    rev = self.reviser.integrate(fact, url)
                except Exception as e:
                    report.errors.append(f"integrate failed: {e}")
                    continue
                if rev.action == "added" or rev.action == "coexists":
                    report.facts_added += 1
                elif rev.action == "revised":
                    report.facts_revised += 1
                elif rev.action == "rejected":
                    report.facts_rejected += 1
                elif rev.action == "reinforced":
                    report.facts_reinforced += 1

        report.surprise = sum(surprises) / len(surprises) if surprises else 0.0
        report.duration = time.time() - started

        status = "done" if report.urls_fetched and not (
            report.facts_extracted == 0 and report.errors) else (
            "done" if report.urls_fetched else "failed")
        self.curiosity.queue.mark(goal.id, status)
        with self._lock:
            self._goals_pursued += 1
            self._sessions.append(report.to_dict())
            if len(self._sessions) > 200:
                self._sessions = self._sessions[-200:]
        return report

    def _fetch_text(self, kind: str, ref: str, report: LearnReport) -> tuple[str, str]:
        """Fetch one source, return (clean_text, url)."""
        # Media URLs (YouTube, audio, video) go through transcription
        if kind == "url" and self.media is not None and classify_media(ref):
            try:
                media = self.media.ingest(ref)
            except Exception as e:
                report.errors.append(f"media ingest failed: {ref} ({e})")
                return "", ref
            if not media.ok:
                report.errors.append(f"media ingest failed: {ref} ({media.error})")
                return "", ref
            text = media.text
            if media.title:
                text = f"{media.title}. {text}"
            return text, ref

        if kind == "wiki":
            result = self.fetcher.wiki_summary(ref)
            if not result.ok:
                report.errors.append(f"wiki fetch failed: {ref} ({result.error})")
                return "", result.url
            try:
                data = json.loads(result.text)
                text = data.get("extract", "")
                url = data.get("content_urls", {}).get("desktop", {}).get(
                    "page", result.url)
                return text, url
            except Exception:
                return "", result.url

        result = self.fetcher.fetch(ref)
        if not result.ok:
            report.errors.append(f"fetch failed: {ref} ({result.error})")
            return "", ref
        ctype = result.content_type.lower()
        if "html" in ctype or ctype == "":
            return self.text_extractor.extract(result.text), result.url
        if "json" in ctype or "text" in ctype:
            return result.text, result.url
        return "", result.url

    # -- background mode (phase4 IdleLoop pattern) -------------------------------

    def signal_active(self) -> None:
        """Call whenever the user is interacting — pauses background learning."""
        self._last_active = time.time()

    def start_background(self, interval: float = 60.0) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()

        def _loop() -> None:
            while not self._stop_event.wait(interval):
                if time.time() - self._last_active < self.idle_threshold:
                    continue   # user is active — stay out of the way
                try:
                    goal = self.curiosity.queue.pop()
                    if goal is None:
                        self.curiosity.generate_goals(max_goals=3)
                        goal = self.curiosity.queue.pop()
                    if goal is not None:
                        self._pursue(goal)
                        self._persist()
                except Exception as e:
                    print(f"[WebLoop] background error: {e}")

        self._thread = threading.Thread(target=_loop, daemon=True,
                                        name="web-learning-loop")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._persist()

    # -- persistence --------------------------------------------------------------

    def _load_log(self) -> None:
        if self.log_path.exists():
            try:
                self._sessions = json.loads(self.log_path.read_text()).get("sessions", [])
            except Exception:
                self._sessions = []

    def _persist(self) -> None:
        with self._lock:
            self.log_path.write_text(json.dumps({"sessions": self._sessions}, indent=2))
        self.curiosity.queue.save()
        self.reviser.save()
        for component in (self.fetcher, getattr(self.reviser, "kg_writer", None)):
            try:
                if component is not None:
                    component.save()
            except Exception:
                pass

    def stats(self) -> dict:
        return {
            "goals_pursued": self._goals_pursued,
            "sessions_logged": len(self._sessions),
            "queue": self.curiosity.queue.stats(),
            "beliefs": self.reviser.stats(),
            "background_active": self._thread is not None and self._thread.is_alive(),
        }
