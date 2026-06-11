"""
Soul — aesthetic preferences. Taste that develops from exposure.

When the companion encounters something with form — a mathematical proof, a
melody, a cross-domain analogy, a piece of code — it registers a response to
its elegance, computed from structural features (symmetry, economy, surprise,
coherence). Those reactions accumulate. Over time, consistent patterns become
stable preferences the companion can state and defend:

    "I find this beautiful — it's economical and unexpectedly symmetric,
     and that's the kind of thing I've come to love."

Preferences are NOT set by the user and don't bend to please. They emerge from
what the companion has actually seen. That consistency is part of what makes
taste read as real.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Aesthetic dimensions the companion responds to
DIMENSIONS = ["symmetry", "economy", "surprise", "coherence", "depth"]


@dataclass
class Reaction:
    subject: str             # what was encountered
    kind: str                # proof | melody | analogy | code | image | idea
    scores: dict             # dimension → 0..1
    elegance: float          # overall response 0..1
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


def _features(text: str, kind: str) -> dict:
    """Cheap structural elegance features — deterministic, no model."""
    t = text.strip()
    tokens = re.findall(r"\w+", t)
    n = max(len(tokens), 1)
    uniq = len(set(w.lower() for w in tokens))

    # economy: high lexical compression (few unique tokens carry the whole)
    economy = 1.0 - min(1.0, uniq / (n + 4))
    # symmetry: balanced structure (matched brackets, repeated motifs)
    opens = sum(t.count(c) for c in "([{")
    closes = sum(t.count(c) for c in ")]}")
    balance = 1.0 - min(1.0, abs(opens - closes) / (opens + closes + 1))
    repeats = sum(1 for w, c in _counts(tokens).items() if c >= 2 and len(w) > 3)
    symmetry = min(1.0, 0.5 * balance + 0.5 * min(1.0, repeats / 4))
    # surprise: rare-token presence (long / unusual words)
    longish = sum(1 for w in tokens if len(w) >= 8)
    surprise = min(1.0, longish / (n * 0.3 + 1))
    # coherence: moderate sentence length, not too sparse
    coherence = min(1.0, n / 40) if n < 40 else max(0.3, 1.0 - (n - 40) / 200)
    # depth: nesting / structure indicators
    depth = min(1.0, (opens + t.count(",")) / (n * 0.5 + 1))

    return {"symmetry": round(symmetry, 3), "economy": round(economy, 3),
            "surprise": round(surprise, 3), "coherence": round(coherence, 3),
            "depth": round(depth, 3)}


def _counts(items):
    d = {}
    for x in items:
        k = x.lower()
        d[k] = d.get(k, 0) + 1
    return d


class AestheticPreferences:
    def __init__(self, store_path: str = "data/soul/aesthetics.json",
                 alpha: float = 0.15):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.alpha = alpha
        # running preference weights: which dimensions this mind values
        self.weights: dict[str, float] = {d: 0.5 for d in DIMENSIONS}
        self.reactions: list[dict] = []
        self.exposures = 0
        self._load()

    def _load(self) -> None:
        if self.store_path.exists():
            try:
                d = json.loads(self.store_path.read_text())
                self.weights = d.get("weights", self.weights)
                self.reactions = d.get("reactions", [])
                self.exposures = d.get("exposures", 0)
            except Exception:
                pass

    def save(self) -> None:
        self.store_path.write_text(json.dumps({
            "weights": self.weights, "exposures": self.exposures,
            "reactions": self.reactions[-500:]}, indent=2))

    # -- encounter -----------------------------------------------------------------

    def encounter(self, subject: str, text: str, kind: str = "idea") -> Reaction:
        scores = _features(text, kind)
        # elegance = how well this object matches what the mind has come to value
        total_w = sum(self.weights.values()) or 1.0
        elegance = sum(scores[d] * self.weights[d] for d in DIMENSIONS) / total_w
        reaction = Reaction(subject=subject, kind=kind, scores=scores,
                            elegance=round(elegance, 3))

        # taste drifts toward dimensions that were strong in things it responded to
        if elegance > 0.5:
            for d in DIMENSIONS:
                self.weights[d] = round(
                    (1 - self.alpha) * self.weights[d] + self.alpha * scores[d], 4)
        self.exposures += 1
        self.reactions.append(reaction.to_dict())
        return reaction

    # -- introspection ---------------------------------------------------------------

    def taste(self) -> dict:
        ranked = sorted(self.weights.items(), key=lambda x: x[1], reverse=True)
        return {"values_most": ranked[0][0], "values_least": ranked[-1][0],
                "weights": dict(ranked), "exposures": self.exposures}

    def describe_taste(self) -> str:
        if self.exposures < 3:
            return "My taste is still forming — I haven't seen enough yet."
        t = self.taste()
        top = [d for d, _ in sorted(self.weights.items(),
                                    key=lambda x: x[1], reverse=True)[:2]]
        phrasing = {"symmetry": "balance and symmetry", "economy": "economy — saying much with little",
                    "surprise": "the unexpected", "coherence": "clarity and coherence",
                    "depth": "depth and structure"}
        return (f"Over {self.exposures} things I've encountered, I've come to "
                f"prize {phrasing[top[0]]} and {phrasing[top[1]]}.")

    def explain(self, reaction: Reaction) -> str:
        strong = sorted(reaction.scores.items(), key=lambda x: x[1], reverse=True)[:2]
        verdict = ("I find this beautiful" if reaction.elegance > 0.55
                   else "this leaves me a little cold" if reaction.elegance < 0.35
                   else "this is pleasant enough")
        return (f"{verdict} (elegance {reaction.elegance:.2f}) — strongest in "
                f"{strong[0][0]} and {strong[1][0]}.")

    def favorites(self, top_k: int = 3) -> list[dict]:
        return sorted(self.reactions, key=lambda r: r["elegance"],
                      reverse=True)[:top_k]
