"""
Affect — emotional state tracking over time.

Per-message detection is what every chatbot does; this layer is what makes
SynapseOS different: it remembers. The tracker maintains a persistent
emotional state (EMA over valence/arousal, mirroring the world model's
surprise_ema), detects shifts, and learns durable patterns —
"user frustrated_by web scraping" — written to semantic memory through the
same fact store as web knowledge, so reasoning can retrieve them later.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from synapseos.affect.text_affect import AffectResult, TextAffectDetector

_STOPWORDS = {"the", "a", "an", "this", "that", "with", "and", "but", "for",
              "not", "is", "are", "was", "were", "have", "has", "had", "its",
              "it's", "i", "you", "we", "they", "my", "your", "what", "why",
              "how", "when", "again", "still", "just", "very", "really", "so"}


@dataclass
class EmotionalState:
    """Persistent slow-moving estimate of the user's emotional state."""
    valence_ema: float = 0.0
    arousal_ema: float = 0.3
    dominant_emotion: str = "neutral"
    observations: int = 0
    last_shift: str = ""          # description of the most recent big change
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        if self.observations == 0:
            return "no emotional observations yet"
        mood = ("positive" if self.valence_ema > 0.15 else
                "negative" if self.valence_ema < -0.15 else "neutral")
        energy = "high" if self.arousal_ema > 0.55 else "low" if self.arousal_ema < 0.3 else "moderate"
        return (f"recent mood {mood} (v={self.valence_ema:+.2f}), "
                f"{energy} energy, last dominant: {self.dominant_emotion}")


class EmotionTracker:
    """Detect → update state → notice shifts → learn durable patterns."""

    def __init__(
        self,
        semantic_memory=None,            # SemanticMemory | None — learned patterns land here
        state_path: str = "data/affect/state.json",
        alpha: float = 0.25,             # EMA smoothing; higher = more reactive
        shift_threshold: float = 0.5,    # |Δvalence| that counts as a mood shift
        pattern_min_count: int = 2,      # repeats before a trigger becomes a fact
    ):
        self.detector = TextAffectDetector()
        self.memory = semantic_memory
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.alpha = alpha
        self.shift_threshold = shift_threshold
        self.pattern_min_count = pattern_min_count
        self.state = EmotionalState()
        # emotion -> {topic -> count}: candidate triggers awaiting confirmation
        self._trigger_counts: dict[str, dict[str, int]] = {}
        self._load()

    # -- persistence ----------------------------------------------------------

    def _load(self) -> None:
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text())
                self.state = EmotionalState(**data.get("state", {}))
                self._trigger_counts = data.get("trigger_counts", {})
            except Exception:
                pass

    def save(self) -> None:
        self.state_path.write_text(json.dumps({
            "state": self.state.to_dict(),
            "trigger_counts": self._trigger_counts,
        }, indent=2))

    # -- main entry --------------------------------------------------------------

    def observe(self, text: str, prosody_arousal: float | None = None) -> AffectResult:
        """
        Detect affect in one message and fold it into the running state.
        prosody_arousal (from analyze_wav) overrides text arousal when the
        voice carries more signal than the words — "I'm fine" said tensely.
        """
        result = self.detector.detect(text)

        arousal = result.arousal
        if prosody_arousal is not None:
            # Voice is the stronger arousal channel; words disambiguate valence
            arousal = 0.4 * result.arousal + 0.6 * prosody_arousal
            result.arousal = round(arousal, 3)

        if result.is_neutral and prosody_arousal is None:
            return result   # nothing to learn from silence

    # -- state update --
        prev_valence = self.state.valence_ema
        a = self.alpha
        self.state.valence_ema = round((1 - a) * self.state.valence_ema + a * result.valence, 3)
        self.state.arousal_ema = round((1 - a) * self.state.arousal_ema + a * arousal, 3)
        if not result.is_neutral:
            self.state.dominant_emotion = result.dominant_emotion
        self.state.observations += 1
        self.state.updated_at = time.time()

        # -- shift detection --
        delta = result.valence - prev_valence
        if abs(delta) >= self.shift_threshold and self.state.observations > 1:
            direction = "improved" if delta > 0 else "dropped"
            self.state.last_shift = (
                f"mood {direction} ({prev_valence:+.2f} → {result.valence:+.2f}) "
                f"on: \"{text[:60]}\""
            )

        # -- pattern learning --
        if not result.is_neutral and result.confidence >= 0.4:
            self._learn_trigger(result.dominant_emotion, text)

        return result

    # -- pattern learning ---------------------------------------------------------

    def _topic_of(self, text: str) -> str:
        """Crude topic: most prominent content words of the message."""
        words = [w for w in re.findall(r"[A-Za-z][\w-]+", text.lower())
                 if w not in _STOPWORDS and len(w) > 2]
        return " ".join(words[:3])

    def _learn_trigger(self, emotion: str, text: str) -> None:
        topic = self._topic_of(text)
        if not topic:
            return
        counts = self._trigger_counts.setdefault(emotion, {})
        counts[topic] = counts.get(topic, 0) + 1
        if counts[topic] == self.pattern_min_count and self.memory is not None:
            # Promoted: repeated association becomes a retrievable fact
            predicate = {
                "frustration": "frustrated_by", "anger": "angered_by",
                "fear": "anxious_about", "sadness": "saddened_by",
                "joy": "enjoys", "excitement": "excited_by",
                "gratitude": "grateful_for", "hesitation": "uncertain_about",
            }.get(emotion, f"feels_{emotion}_about")
            self.memory.add_fact(
                subject="user", predicate=predicate, obj=topic,
                confidence=0.6, source="affect_observation",
            )

    def stats(self) -> dict:
        return {
            "state": self.state.to_dict(),
            "summary": self.state.describe(),
            "tracked_triggers": {e: len(t) for e, t in self._trigger_counts.items()},
        }
