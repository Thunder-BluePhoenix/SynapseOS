"""
Soul — inner life. The questions the companion is genuinely wondering about.

The idle loop walks the graph and makes weak hypotheses; that is background
processing, not an inner life. An inner life means the system holds questions
it actually cares about — some for weeks — and what it chooses to think about
reflects genuine prioritization, not a random walk (per the companion docs).

A Wonder is a question with a life of its own:
  - born from a surprise (something it couldn't predict), a contradiction
    (two beliefs that clash), a frontier (a thinly-connected concept), or
    the user
  - carries an intensity that decays slowly when ignored but never to zero —
    a real question nags
  - is revisited, and revisiting can DEEPEN it into sub-questions
  - resolves when an answer arrives, or goes dormant (still there, quieter)
  - the oldest unresolved ones are what make curiosity feel real:
    "I've been wondering about X for three weeks now."

The journal feeds the web learning loop (questions → learning goals) and
surfaces in conversation.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

DAY = 86400.0


@dataclass
class Wonder:
    id: str
    question: str
    topic: str
    source: str                 # surprise | contradiction | frontier | user | deepened
    intensity: float            # 0..1 — how much it nags right now
    created_at: float = field(default_factory=time.time)
    last_visited: float = field(default_factory=time.time)
    visits: int = 0
    status: str = "open"        # open | dormant | resolved
    parent_id: str = ""         # set if this is a sub-question
    answer: str = ""
    notes: list[str] = field(default_factory=list)

    def age_days(self, now: float | None = None) -> float:
        return ((now or time.time()) - self.created_at) / DAY

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Wonder":
        return cls(**d)


def _wid(question: str) -> str:
    return hashlib.sha1(question.lower().strip().encode()).hexdigest()[:12]


class WonderJournal:
    """Persistent open questions with a genuine attention economy."""

    def __init__(self, path: str = "data/soul/wonder.json",
                 decay_per_day: float = 0.08, dormant_after_days: float = 14.0,
                 max_open: int = 60):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.decay_per_day = decay_per_day
        self.dormant_after_days = dormant_after_days
        self.max_open = max_open
        self._lock = threading.Lock()
        self._wonders: dict[str, Wonder] = {}
        self._load()

    # -- persistence ----------------------------------------------------------------

    def _load(self) -> None:
        if self.path.exists():
            try:
                for d in json.loads(self.path.read_text()).get("wonders", []):
                    w = Wonder.from_dict(d)
                    self._wonders[w.id] = w
            except Exception:
                pass

    def save(self) -> None:
        with self._lock:
            self.path.write_text(json.dumps(
                {"wonders": [w.to_dict() for w in self._wonders.values()]}, indent=2))

    # -- forming questions ----------------------------------------------------------

    def wonder_about(self, question: str, topic: str = "", source: str = "user",
                     intensity: float = 0.6, parent_id: str = "") -> Wonder:
        """Form (or reinforce) a question. Reinforcing an existing one stokes it."""
        wid = _wid(question)
        with self._lock:
            if wid in self._wonders:
                w = self._wonders[wid]
                w.intensity = min(1.0, w.intensity + 0.15)
                if w.status == "dormant":
                    w.status = "open"           # something rekindled it
                return w
            w = Wonder(id=wid, question=question.strip(), topic=topic or question[:40],
                       source=source, intensity=max(0.1, min(1.0, intensity)),
                       parent_id=parent_id)
            self._wonders[wid] = w
            return w

    def from_surprise(self, text: str, surprise: float) -> Wonder | None:
        if surprise < 0.5:
            return None
        topic = " ".join(text.split()[:6])
        return self.wonder_about(f"Why did this surprise me: \"{topic}\"?",
                                 topic=topic, source="surprise",
                                 intensity=min(1.0, 0.4 + surprise / 2))

    def from_contradiction(self, subject: str, predicate: str,
                           a: str, b: str) -> Wonder:
        q = f"Is {subject} really {predicate.replace('_', ' ')} {a}, or {b}?"
        return self.wonder_about(q, topic=subject, source="contradiction",
                                 intensity=0.75)

    def from_frontier(self, concept: str, degree: int) -> Wonder:
        return self.wonder_about(f"What is {concept} connected to that I don't know yet?",
                                 topic=concept, source="frontier",
                                 intensity=0.3 + 0.1 * max(0, 3 - degree))

    # -- the attention economy ------------------------------------------------------

    def _decay(self, now: float) -> None:
        """Intensity fades when ignored, but never to zero — questions nag."""
        for w in self._wonders.values():
            if w.status != "open":
                continue
            idle_days = (now - w.last_visited) / DAY
            if idle_days > 0:
                w.intensity = max(0.08, w.intensity - self.decay_per_day * idle_days)
            idle_since_visit = (now - w.last_visited) / DAY
            if idle_since_visit > self.dormant_after_days:
                w.status = "dormant"            # not revisited within window → quietens

    def ponder(self, now: float | None = None) -> Wonder | None:
        """
        Pick the question to think about now — NOT randomly. Liveliness blends
        intensity with how long it's gone unexamined: a long-ignored intense
        question wins, which is what makes the curiosity feel real.
        """
        now = now or time.time()
        with self._lock:
            self._decay(now)
            open_qs = [w for w in self._wonders.values() if w.status == "open"]
            if not open_qs:
                return None

            def liveliness(w: Wonder) -> float:
                neglect = min(2.0, (now - w.last_visited) / DAY)
                return w.intensity * (1.0 + 0.3 * neglect)

            chosen = max(open_qs, key=liveliness)
            chosen.visits += 1
            chosen.last_visited = now
            return chosen

    def deepen(self, wonder_id: str, sub_question: str) -> Wonder | None:
        """Revisiting a question can spawn a sharper sub-question."""
        if wonder_id not in self._wonders:
            return None
        return self.wonder_about(sub_question, topic=self._wonders[wonder_id].topic,
                                 source="deepened", intensity=0.6,
                                 parent_id=wonder_id)

    def note(self, wonder_id: str, note: str) -> None:
        with self._lock:
            if wonder_id in self._wonders:
                self._wonders[wonder_id].notes.append(note[:200])

    def resolve(self, wonder_id: str, answer: str) -> Wonder | None:
        with self._lock:
            w = self._wonders.get(wonder_id)
            if w:
                w.status = "resolved"
                w.answer = answer
            return w

    # -- expression -----------------------------------------------------------------

    def open_questions(self, limit: int = 10) -> list[Wonder]:
        now = time.time()
        return sorted([w for w in self._wonders.values() if w.status == "open"],
                      key=lambda w: w.intensity, reverse=True)[:limit]

    def longest_held(self) -> Wonder | None:
        now = time.time()
        open_qs = [w for w in self._wonders.values() if w.status == "open"]
        return max(open_qs, key=lambda w: w.age_days(now), default=None)

    def surface(self) -> str:
        """A natural 'what's on my mind' line, grounded in real held questions."""
        held = self.longest_held()
        if held is None:
            return "Nothing pressing on my mind right now."
        days = held.age_days()
        when = (f"for {int(days)} days now" if days >= 1
                else "since earlier today")
        return f"I've been wondering, {when}: {held.question}"

    def feed_curiosity(self, curiosity_engine, max_goals: int = 5) -> int:
        """Turn the liveliest open questions into web-learning goals."""
        fed = 0
        for w in self.open_questions(limit=max_goals):
            try:
                goal = curiosity_engine.add_user_topic(w.topic, priority=w.intensity)
                goal.source = "wonder"
                goal.reason = f"wondering: {w.question}"
                fed += 1
            except Exception:
                break
        return fed

    def stats(self) -> dict:
        by_status: dict[str, int] = {}
        by_source: dict[str, int] = {}
        for w in self._wonders.values():
            by_status[w.status] = by_status.get(w.status, 0) + 1
            by_source[w.source] = by_source.get(w.source, 0) + 1
        held = self.longest_held()
        return {"total": len(self._wonders), "by_status": by_status,
                "by_source": by_source,
                "longest_held_days": round(held.age_days(), 1) if held else 0}
