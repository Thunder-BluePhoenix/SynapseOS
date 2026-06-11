"""
Web Learning Loop — Curiosity engine.

Decides WHAT the system wants to learn next. Four sources of curiosity,
in priority order:

  user        (1.0)        — explicit topics from the user
  hypothesis  (0.5–0.9)    — weak idle-loop hypotheses worth verifying
  surprise    (0.4–0.9)    — recent world-model episodes it didn't predict
  kg_frontier (0.3–0.6)    — knowledge-graph entities with few connections

Goals live in a persistent, deduplicated priority queue.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class LearningGoal:
    id: str
    topic: str                  # used as the web search query
    reason: str                 # why the system is curious
    source: str                 # user | hypothesis | surprise | kg_frontier
    priority: float             # 0..1
    created_at: float = field(default_factory=time.time)
    status: str = "pending"     # pending | done | failed
    attempts: int = 0
    urls: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LearningGoal":
        return cls(**d)


def _goal_id(source: str, topic: str) -> str:
    return hashlib.sha1(f"{source}|{topic.lower()}".encode()).hexdigest()[:12]


class GoalQueue:
    """Persistent priority queue of learning goals. Thread-safe, dedup by id."""

    def __init__(self, path: str = "data/web/goals.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._goals: dict[str, LearningGoal] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                for g in data.get("goals", []):
                    goal = LearningGoal.from_dict(g)
                    self._goals[goal.id] = goal
            except Exception:
                pass

    def save(self) -> None:
        with self._lock:
            data = {"goals": [g.to_dict() for g in self._goals.values()]}
            self.path.write_text(json.dumps(data, indent=2))

    def add(self, goal: LearningGoal) -> bool:
        with self._lock:
            if goal.id in self._goals:
                return False
            self._goals[goal.id] = goal
            return True

    def pop(self) -> LearningGoal | None:
        with self._lock:
            pending = [g for g in self._goals.values() if g.status == "pending"]
            if not pending:
                return None
            best = max(pending, key=lambda g: g.priority)
            best.attempts += 1
            return best

    def mark(self, goal_id: str, status: str) -> None:
        with self._lock:
            if goal_id in self._goals:
                self._goals[goal_id].status = status

    def pending(self) -> list[LearningGoal]:
        return sorted(
            (g for g in self._goals.values() if g.status == "pending"),
            key=lambda g: g.priority, reverse=True,
        )

    def stats(self) -> dict:
        by_status: dict[str, int] = {}
        for g in self._goals.values():
            by_status[g.status] = by_status.get(g.status, 0) + 1
        return {"total": len(self._goals), "by_status": by_status}

    def __len__(self) -> int:
        return len(self._goals)


class CuriosityEngine:
    """Generates learning goals from internal signals of not-knowing."""

    def __init__(
        self,
        hypothesis_store=None,    # phase4 HypothesisStore | None
        world_model=None,         # WorldModel | None
        knowledge_graph=None,     # phase3 KnowledgeGraph | None
        queue: GoalQueue | None = None,
        surprise_threshold: float = 0.5,
    ):
        self.hypothesis_store = hypothesis_store
        self.world_model = world_model
        self.kg = knowledge_graph
        # `is not None` matters: an empty GoalQueue is falsy via __len__
        self.queue = queue if queue is not None else GoalQueue()
        self.surprise_threshold = surprise_threshold

    # -- public API -------------------------------------------------------------

    def add_user_topic(self, topic: str, priority: float = 1.0) -> LearningGoal:
        goal = LearningGoal(
            id=_goal_id("user", topic), topic=topic,
            reason="user requested", source="user", priority=priority,
        )
        self.queue.add(goal)
        return goal

    def generate_goals(self, max_goals: int = 10) -> list[LearningGoal]:
        """Pull from every curiosity source, enqueue the best candidates."""
        candidates: list[LearningGoal] = []
        candidates.extend(self._from_hypotheses())
        candidates.extend(self._from_surprises())
        candidates.extend(self._from_kg_frontier())
        candidates.sort(key=lambda g: g.priority, reverse=True)

        added: list[LearningGoal] = []
        for goal in candidates[: max_goals * 2]:
            if self.queue.add(goal):
                added.append(goal)
            if len(added) >= max_goals:
                break
        return added

    # -- curiosity sources --------------------------------------------------------

    def _from_hypotheses(self) -> list[LearningGoal]:
        """Weak hypotheses (conf 0.2–0.6) are worth checking against the web."""
        if self.hypothesis_store is None:
            return []
        goals = []
        try:
            for hyp in self.hypothesis_store.get_all():
                if not (0.2 <= hyp.confidence < 0.6):
                    continue
                topic = f"{hyp.concept_a} {hyp.concept_b}"
                goals.append(LearningGoal(
                    id=_goal_id("hypothesis", topic), topic=topic,
                    reason=f"weak hypothesis ({hyp.confidence:.2f}): {hyp.description}",
                    source="hypothesis",
                    priority=min(0.9, 0.5 + (0.6 - hyp.confidence)),
                ))
        except Exception:
            return []
        return goals

    def _from_surprises(self) -> list[LearningGoal]:
        """High-surprise episodes = things the world model failed to predict."""
        if self.world_model is None:
            return []
        goals = []
        try:
            memory = getattr(self.world_model, "memory", None)
            episodes = getattr(memory, "episodes", None) if memory else None
            if episodes is None:
                return []
            items = list(episodes.values()) if isinstance(episodes, dict) else list(episodes)
            for ep in items[-50:]:
                surprise = getattr(ep, "surprise", 0.0)
                text = getattr(ep, "text", "")
                if surprise < self.surprise_threshold or not text:
                    continue
                topic = self._topic_from_text(text)
                if not topic:
                    continue
                goals.append(LearningGoal(
                    id=_goal_id("surprise", topic), topic=topic,
                    reason=f"high surprise ({surprise:.2f}) — did not understand this",
                    source="surprise",
                    priority=min(0.9, 0.4 + surprise / 2.0),
                ))
        except Exception:
            return []
        return goals

    def _from_kg_frontier(self, max_sample: int = 5) -> list[LearningGoal]:
        """Entities with few edges sit at the boundary of what the system knows."""
        if self.kg is None:
            return []
        goals = []
        try:
            adjacency = getattr(self.kg, "_adjacency", {})
            nodes = getattr(self.kg, "_nodes", {})
            for name in nodes:
                if len(goals) >= max_sample:
                    break
                if "__" in name or len(name) < 3:
                    continue
                degree = len(adjacency.get(name, []))
                if degree <= 2:
                    goals.append(LearningGoal(
                        id=_goal_id("kg_frontier", name), topic=name,
                        reason=f"knowledge frontier — only {degree} connections",
                        source="kg_frontier",
                        priority=0.3 + 0.1 * (3 - degree),
                    ))
        except Exception:
            return []
        return goals

    # -- helpers ----------------------------------------------------------------

    @staticmethod
    def _topic_from_text(text: str, max_words: int = 5) -> str:
        """First capitalized/content words of an episode → search topic."""
        words = re.findall(r"[A-Za-z][\w-]+", text)
        content = [w for w in words if len(w) > 2][:max_words]
        return " ".join(content)
