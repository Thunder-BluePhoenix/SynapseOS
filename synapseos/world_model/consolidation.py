"""
Consolidation — the "sleep" phase of the world model.

During idle time the brain replays salient episodes and distills them from
hippocampal (episodic) into neocortical (semantic) memory. This module does
the computational analogue:

  1. Replay      — sample high-salience (high-surprise) episodes
  2. Distill     — extract recurring patterns; promote stable ones into the
                   SemanticMemory KG as facts ("X frequently precedes Y")
  3. Re-train    — gradient steps so the predictor reproduces replayed episodes,
                   permanently lowering future surprise for those patterns
                   (this is how episodic experience becomes procedural skill)
  4. Mark        — flag consolidated episodes so they aren't replayed forever

This is what lets the system reason over months: experience doesn't just pile
up as episodes, it compresses into the world model's weights and the semantic
graph.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from synapseos.world_model.predictive_memory import PredictiveMemory, Episode


@dataclass
class ConsolidationReport:
    episodes_replayed: int
    patterns_found: int
    facts_distilled: int
    predictor_loss_before: float
    predictor_loss_after: float
    episodes_marked: int
    duration_s: float

    def summary(self) -> str:
        return (
            f"Consolidation: replayed {self.episodes_replayed}, "
            f"distilled {self.facts_distilled} facts, "
            f"predictor loss {self.predictor_loss_before:.3f}→{self.predictor_loss_after:.3f}, "
            f"marked {self.episodes_marked} ({self.duration_s:.1f}s)"
        )

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


class Consolidator:
    """
    Runs consolidation over the world model's memory + net.
    Designed to be called during idle time (hook into phase4 idle_loop).
    """

    def __init__(
        self,
        world_model,                 # WorldModel (avoid circular import via duck-typing)
        semantic_memory=None,        # optional synapseos.phase5.memory.SemanticMemory
        replay_batch: int = 16,
        replay_epochs: int = 3,
        min_pattern_count: int = 2,
    ):
        self.wm = world_model
        self.semantic = semantic_memory
        self.replay_batch  = replay_batch
        self.replay_epochs = replay_epochs
        self.min_pattern_count = min_pattern_count

    # ------------------------------------------------------------------
    # Pattern distillation
    # ------------------------------------------------------------------

    def _distill_patterns(self, episodes: list[Episode]) -> list[tuple[str, str, int]]:
        """
        Find recurring temporal patterns: which kinds of events tend to precede
        which others.

        Reads the world model's transition counter (updated on EVERY perception),
        so routines are captured even after surprise-gating stops storing them.
        Falls back to episode prev_id chains if the counter is unavailable.
        Returns list of (antecedent_concept, consequent_concept, count).
        """
        # Primary source: the always-on transition counter
        counts = {}
        if hasattr(self.wm, "transition_counts"):
            counts = self.wm.transition_counts()

        if not counts:
            # Fallback: reconstruct from stored episode chains
            def concept(text: str) -> str:
                import re
                words = [w for w in re.findall(r"[A-Za-z]+", text) if len(w) > 3]
                caps = [w for w in words if w[0].isupper()]
                return (caps[-1] if caps else (words[-1] if words else "event")).lower()
            by_id = {e.id: e for e in episodes}
            c: Counter = Counter()
            for e in episodes:
                if e.prev_id and e.prev_id in by_id:
                    a, b = concept(by_id[e.prev_id].text), concept(e.text)
                    if a != b:
                        c[(a, b)] += 1
            counts = dict(c)

        return [(a, b, n) for (a, b), n in counts.items() if n >= self.min_pattern_count]

    def _promote_to_semantic(self, patterns: list[tuple[str, str, int]]) -> int:
        if self.semantic is None:
            return 0
        n = 0
        for a, b, count in patterns:
            try:
                self.semantic.add_fact(
                    subject=a, predicate="frequently_precedes", obj=b,
                    confidence=min(0.95, 0.5 + 0.1 * count),
                    source="world_model_consolidation",
                )
                n += 1
            except Exception:
                pass
        return n

    # ------------------------------------------------------------------
    # Predictor re-training (episodic → procedural)
    # ------------------------------------------------------------------

    def _replay_train(self, episodes: list[Episode]) -> tuple[float, float]:
        """
        Train the predictor to reproduce replayed episodes:
        predictor(z_at_episode) → obs_of_episode.
        Lowers future surprise for these patterns (skill formation).
        """
        if not episodes:
            return 0.0, 0.0
        device = self.wm.device
        net    = self.wm.net
        opt    = self.wm.opt

        Z   = torch.stack([e.z_tensor() for e in episodes]).to(device)
        O   = torch.stack([e.obs_tensor() for e in episodes]).to(device)

        net.eval()
        with torch.no_grad():
            pred0 = net.predict(Z)
            loss_before = (1 - F.cosine_similarity(pred0, O)).mean().item()

        net.train()
        for _ in range(self.replay_epochs):
            pred = net.predict(Z)
            loss = (1 - F.cosine_similarity(pred, O)).mean()
            opt.zero_grad(); loss.backward(); opt.step()

        net.eval()
        with torch.no_grad():
            pred1 = net.predict(Z)
            loss_after = (1 - F.cosine_similarity(pred1, O)).mean().item()

        return loss_before, loss_after

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def consolidate(self, salience_threshold: float = 0.45) -> ConsolidationReport:
        t0 = time.time()
        mem = self.wm.memory

        replay = mem.high_salience(threshold=salience_threshold, limit=self.replay_batch)
        if not replay:
            return ConsolidationReport(0, 0, 0, 0.0, 0.0, 0, round(time.time()-t0, 2))

        # 1. distill patterns from the broader episode set (temporal structure)
        patterns = self._distill_patterns(mem.episodes)
        facts    = self._promote_to_semantic(patterns)

        # 2. replay-train the predictor on salient episodes
        loss_before, loss_after = self._replay_train(replay)

        # 3. mark replayed episodes consolidated (retain but stop replaying)
        for e in replay:
            e.consolidated = True

        # 4. persist
        self.wm.save()
        if self.semantic is not None:
            try:
                self.semantic.save()
            except Exception:
                pass

        return ConsolidationReport(
            episodes_replayed=len(replay),
            patterns_found=len(patterns),
            facts_distilled=facts,
            predictor_loss_before=round(loss_before, 4),
            predictor_loss_after=round(loss_after, 4),
            episodes_marked=len(replay),
            duration_s=round(time.time() - t0, 2),
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    import tempfile, os
    from synapseos.world_model.world_model import WorldModel
    from synapseos.phase5.memory import SemanticMemory

    d = tempfile.mkdtemp(prefix="consol_demo_")
    wm = WorldModel(
        state_path=os.path.join(d, "state.json"),
        episodes_path=os.path.join(d, "episodes.json"),
        snapshot_dir=os.path.join(d, "snaps"),
    )
    sem = SemanticMemory(store_path=os.path.join(d, "semantic.json"), kg_path="/tmp/nope.json")
    sem.facts.clear(); sem._fact_index.clear()

    # Feed a structured experience: A → B → C repeated, plus noise
    seq = [
        "User created a Sales Order",
        "System generated a Delivery Note",
        "Accounts posted a Sales Invoice",
    ]
    for _ in range(4):
        for ev in seq:
            wm.perceive(ev)
        wm.perceive("Random unrelated maintenance log entry " + str(time.time()))

    print(f"Before consolidation: {wm.memory.stats()}\n")

    consolidator = Consolidator(wm, semantic_memory=sem, replay_epochs=5)
    report = consolidator.consolidate()
    print(report.summary())

    print(f"\nDistilled semantic facts ({len(sem.facts)}):")
    for f in sem.facts[:8]:
        print(f"  {f.subject} --{f.predicate}--> {f.obj}  (conf={f.confidence:.2f})")

    print(f"\nAfter: {wm.memory.stats()}")
    print("  → temporal patterns compressed into semantic memory; predictor improved.")


if __name__ == "__main__":
    _demo()
