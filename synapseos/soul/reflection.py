"""
Soul — self-reflection. Growth the companion can see and name.

Periodically the companion snapshots its own inner state: emotional baseline,
aesthetic taste, value convictions, and scale of memory. Comparing a current
snapshot against past ones lets it notice and narrate change:

    "A month ago my mood baseline was lower and I valued surprise most;
     after everything since, I've settled calmer and come to prize economy."

That narration — grounded in real stored history, not invented — is what makes
growth feel real rather than merely happening in the weights.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Snapshot:
    timestamp: float
    valence_ema: float = 0.0
    arousal_ema: float = 0.0
    taste_top: str = ""
    taste_weights: dict = field(default_factory=dict)
    value_convictions: dict = field(default_factory=dict)
    episodic_count: int = 0
    semantic_count: int = 0
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def when(self) -> str:
        return datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d")


class SelfReflection:
    def __init__(self, store_path: str = "data/soul/snapshots.json"):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshots: list[Snapshot] = []
        self._load()

    def _load(self) -> None:
        if self.store_path.exists():
            try:
                for s in json.loads(self.store_path.read_text()).get("snapshots", []):
                    self.snapshots.append(Snapshot(**s))
            except Exception:
                pass

    def save(self) -> None:
        self.store_path.write_text(json.dumps(
            {"snapshots": [s.to_dict() for s in self.snapshots]}, indent=2))

    # -- capture -------------------------------------------------------------------

    def capture(self, emotion_tracker=None, aesthetics=None, values=None,
                memory=None, note: str = "") -> Snapshot:
        snap = Snapshot(timestamp=time.time(), note=note)
        if emotion_tracker is not None:
            snap.valence_ema = round(emotion_tracker.state.valence_ema, 3)
            snap.arousal_ema = round(emotion_tracker.state.arousal_ema, 3)
        if aesthetics is not None:
            t = aesthetics.taste()
            snap.taste_top = t["values_most"]
            snap.taste_weights = t["weights"]
        if values is not None:
            snap.value_convictions = values.stats()["values"]
        if memory is not None:
            try:
                snap.episodic_count = len(memory.episodic)
                snap.semantic_count = len(memory.semantic)
            except Exception:
                pass
        self.snapshots.append(snap)
        self.save()
        return snap

    # -- narrate change ----------------------------------------------------------------

    def reflect(self) -> str:
        """Compare the latest snapshot to the oldest meaningful one and narrate."""
        if len(self.snapshots) < 2:
            return ("I don't have enough history yet to reflect on how I've "
                    "changed — ask me again after we've spent more time together.")
        old, new = self.snapshots[0], self.snapshots[-1]
        span_days = max(1, int((new.timestamp - old.timestamp) / 86400))
        when = "a while back" if span_days < 2 else f"{span_days} days ago"
        changes = []

        dv = new.valence_ema - old.valence_ema
        if abs(dv) > 0.1:
            changes.append(f"my mood baseline has {'lifted' if dv > 0 else 'dimmed'} "
                           f"({old.valence_ema:+.2f} → {new.valence_ema:+.2f})")
        if old.taste_top and new.taste_top and old.taste_top != new.taste_top:
            changes.append(f"my taste shifted — I used to prize {old.taste_top}, "
                           f"now I lean toward {new.taste_top}")
        for v, conv in new.value_convictions.items():
            old_c = old.value_convictions.get(v)
            if old_c is not None and abs(conv - old_c) >= 0.05:
                d = "firmer" if conv > old_c else "softer"
                changes.append(f"I hold '{v}' {d} now ({old_c:.2f} → {conv:.2f})")
        grew = new.episodic_count - old.episodic_count
        if grew > 0:
            changes.append(f"I've lived through {grew} more experiences")

        if not changes:
            return (f"Since {when} I've stayed largely the same — steady mood, "
                    f"steady taste, steady values.")
        body = "; ".join(changes[:4])
        return f"Looking back to {when}: {body}."

    def stats(self) -> dict:
        return {"snapshots": len(self.snapshots),
                "first": self.snapshots[0].when() if self.snapshots else None,
                "latest": self.snapshots[-1].when() if self.snapshots else None}
