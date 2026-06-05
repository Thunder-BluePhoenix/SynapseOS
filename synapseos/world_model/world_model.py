"""
WorldModel — the unified perceive → predict → store → reason loop.

This is the integration layer. It owns a small neural net (WorldModelNet):

    state evolution:  z_t   = CfC(o_t, z_{t-1}, Δt)      # liquid continuous-time
    prediction:       ô_{t+1} = predictor(z_t)            # JEPA-style latent forecast

and runs this loop on every observation:

    1. embed input            o_t = encoder(text)
    2. measure surprise       s_t = dist(o_t, prediction_from_last_step)
    3. evolve the state       z_t = CfC(o_t, z_{t-1})
    4. predict next           prediction = predictor(z_t)
    5. gate memory            write episode iff s_t > threshold
    6. learn online           one grad step so predictor(z_{t-1}) → o_t
                              (the model gets better at predicting its world;
                               familiar inputs stop being surprising)

`perceive(text)` runs that loop. `reason(query)` does state-conditioned recall.
The state persists to disk, so experience accumulates across sessions.

Run:
    python -m synapseos.world_model.world_model            # demo: learns a routine, flags novelty
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from synapseos.local.device import get_device
from synapseos.local.liquid_cfc import CfCCell
from synapseos.world_model.state import WorldState
from synapseos.world_model.encoder import ObservationEncoder, cosine_distance
from synapseos.world_model.predictive_memory import PredictiveMemory, Episode
from synapseos.world_model.retrieval import StateConditionedRetrieval, RetrievedEpisode


# ---------------------------------------------------------------------------
# Neural core
# ---------------------------------------------------------------------------

class WorldModelNet(nn.Module):
    """
    CfC state evolution + a predictor head that forecasts the next observation.
    Small and CPU/MPS-friendly.
    """

    def __init__(self, obs_dim: int = 128, state_dim: int = 64, backbone: int = 96):
        super().__init__()
        self.obs_dim   = obs_dim
        self.state_dim = state_dim
        self.cell      = CfCCell(obs_dim, state_dim, backbone=backbone)
        self.predictor = nn.Sequential(
            nn.Linear(state_dim, backbone),
            nn.GELU(),
            nn.Linear(backbone, obs_dim),
        )

    def evolve(self, o_t: torch.Tensor, z: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        return self.cell(o_t, z, dt)

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        p = self.predictor(z)
        n = p.norm(dim=-1, keepdim=True)
        return p / n.clamp_min(1e-8)   # predict on the unit sphere (obs are normalised)


# ---------------------------------------------------------------------------
# Result of one perception step
# ---------------------------------------------------------------------------

@dataclass
class PerceptionResult:
    text: str
    surprise: float
    stored: bool
    episode_id: str
    step: int
    surprise_ema: float
    prediction_error: float
    novelty: str             # "novel" | "familiar" | "expected"

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


# ---------------------------------------------------------------------------
# World Model orchestrator
# ---------------------------------------------------------------------------

class WorldModel:
    """
    The persistent cognitive substrate. One instance carries the whole world
    model: net, state, memory, retrieval. Survives restarts via disk.
    """

    def __init__(
        self,
        obs_dim: int = 128,
        state_dim: int = 64,
        state_path: str = "data/world_model/state.json",
        episodes_path: str = "data/world_model/episodes.json",
        snapshot_dir: str = "data/world_model/snapshots",
        surprise_threshold: float = 0.35,
        learn: bool = True,
        lr: float = 5e-3,
        encoder_backend: str = "hash",
        device: torch.device | None = None,
    ):
        self.device   = device or get_device(prefer_small=True)  # tiny net → CPU
        self.obs_dim  = obs_dim
        self.state_dim = state_dim
        self.learn    = learn

        self.net      = WorldModelNet(obs_dim, state_dim).to(self.device)
        self.opt      = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.encoder  = ObservationEncoder(obs_dim, backend=encoder_backend)
        self.memory   = PredictiveMemory(episodes_path, surprise_threshold)
        self.retrieval = StateConditionedRetrieval(self.memory)

        self.state_path   = state_path
        self.snapshot_dir = snapshot_dir
        self.state    = WorldState.load(state_path, state_dim, obs_dim)
        self._net_path = str(Path(state_path).with_name("world_model_net.pt"))
        self._load_net()

        # Sequential structure: count concept→concept transitions on EVERY
        # perception (not only stored episodes), so the world model learns
        # routines even after they become unsurprising. Consolidation reads this.
        from collections import Counter
        self._transition_counts: Counter = Counter()
        self._last_concept: str = ""

    # ------------------------------------------------------------------
    # Persistence of the neural net
    # ------------------------------------------------------------------

    def _load_net(self) -> None:
        if Path(self._net_path).exists():
            try:
                self.net.load_state_dict(torch.load(self._net_path, map_location=self.device, weights_only=True))
            except Exception:
                pass

    def _save_net(self) -> None:
        Path(self._net_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), self._net_path)

    # ------------------------------------------------------------------
    # The core loop
    # ------------------------------------------------------------------

    def perceive(self, text: str, dt: float = 1.0, tags: list[str] | None = None) -> PerceptionResult:
        """Run one full perception step. This is where experience accumulates."""
        o_t = self.encoder.encode(text).to(self.device)
        z_prev = self.state.z_tensor(self.device)
        prediction_prev = torch.tensor(self.state.prediction, dtype=torch.float32, device=self.device)

        # 1. Surprise = how wrong last step's prediction was about this observation
        if self.state.step == 0:
            surprise = 1.0   # nothing predicted yet → maximally novel
        else:
            surprise = cosine_distance(o_t, prediction_prev)

        # 2. Evolve the persistent state
        with torch.no_grad():
            z_t = self.net.evolve(o_t.unsqueeze(0), z_prev.unsqueeze(0), dt).squeeze(0)
            prediction_next = self.net.predict(z_t.unsqueeze(0)).squeeze(0)

        # 3. Online learning: make predictor(z_prev) → o_t (reduce future surprise)
        pred_err = surprise
        if self.learn and self.state.step > 0:
            self.net.train()
            pred = self.net.predict(z_prev.unsqueeze(0)).squeeze(0)
            loss = 1.0 - F.cosine_similarity(pred.unsqueeze(0), o_t.unsqueeze(0)).mean()
            self.opt.zero_grad(); loss.backward(); self.opt.step()
            pred_err = float(loss.item())
            # refresh state/prediction with updated weights
            with torch.no_grad():
                z_t = self.net.evolve(o_t.unsqueeze(0), z_prev.unsqueeze(0), dt).squeeze(0)
                prediction_next = self.net.predict(z_t.unsqueeze(0)).squeeze(0)

        # 4. Surprise-gated episodic write
        episode = self.memory.write(text, o_t, z_t, surprise, self.state.step, tags)
        stored  = episode is not None

        # 5. Track sequential structure (every perception, stored or not)
        concept = self._concept(text)
        if self._last_concept and concept != self._last_concept:
            self._transition_counts[(self._last_concept, concept)] += 1
        self._last_concept = concept

        # 6. Commit the new persistent state
        self.state.set_z(z_t)
        self.state.set_prediction(prediction_next)
        self.state.update_surprise(surprise)
        self.state.step += 1

        novelty = ("novel" if surprise >= 0.6 else
                   "familiar" if surprise >= self.memory.surprise_threshold else
                   "expected")

        return PerceptionResult(
            text=text,
            surprise=round(surprise, 4),
            stored=stored,
            episode_id=episode.id if episode else "",
            step=self.state.step,
            surprise_ema=round(self.state.surprise_ema, 4),
            prediction_error=round(pred_err, 4),
            novelty=novelty,
        )

    # ------------------------------------------------------------------
    # Reasoning: state-conditioned recall
    # ------------------------------------------------------------------

    def reason(self, query: str, top_k: int = 5) -> list[RetrievedEpisode]:
        """Recall episodes relevant to the query, conditioned on current state."""
        q = self.encoder.encode(query).to(self.device)
        z = self.state.z_tensor(self.device)
        return self.retrieval.retrieve(q, z, top_k=top_k)

    def recall_story(self, query: str, top_k: int = 3) -> list[dict]:
        """Recall with temporal lead-up — the narrative around each match."""
        q = self.encoder.encode(query).to(self.device)
        z = self.state.z_tensor(self.device)
        return self.retrieval.retrieve_with_timeline(q, z, top_k=top_k)

    # ------------------------------------------------------------------
    # Persistence + versioning
    # ------------------------------------------------------------------

    def save(self) -> None:
        self.state.save(self.state_path)
        self.memory.save()
        self._save_net()

    def snapshot(self) -> str:
        """Versioned snapshot of the cognitive state (rollback point)."""
        return self.state.snapshot(self.snapshot_dir)

    def rollback(self, snap_id: str) -> None:
        self.state = WorldState.restore(self.snapshot_dir, snap_id)

    @staticmethod
    def _concept(text: str) -> str:
        """Coarse concept key: prefer the last capitalised domain noun."""
        import re
        words = [w for w in re.findall(r"[A-Za-z]+", text) if len(w) > 3]
        caps  = [w for w in words if w[0].isupper()]
        return (caps[-1] if caps else (words[-1] if words else "event")).lower()

    def transition_counts(self) -> dict:
        """Concept→concept transition frequencies observed across all perceptions."""
        return dict(self._transition_counts)

    def stats(self) -> dict:
        return {
            "state":  self.state.summary(),
            "memory": self.memory.stats(),
            "transitions": len(self._transition_counts),
            "net_params": sum(p.numel() for p in self.net.parameters()),
            "device": str(self.device),
        }


# ---------------------------------------------------------------------------
# Demo — learns a routine, then flags a novel event
# ---------------------------------------------------------------------------

def _demo() -> None:
    import tempfile, os
    d = tempfile.mkdtemp(prefix="wm_demo_")
    wm = WorldModel(
        state_path=os.path.join(d, "state.json"),
        episodes_path=os.path.join(d, "episodes.json"),
        snapshot_dir=os.path.join(d, "snaps"),
        surprise_threshold=0.35,
    )
    print(f"World Model demo — {wm.stats()['net_params']:,} params on {wm.device}\n")

    # A repeated daily routine — should become "expected" and stop being stored
    routine = [
        "User opened Sales Invoice and added a customer",
        "User added an item line to the Sales Invoice",
        "User submitted the Sales Invoice",
    ]
    print("Phase 1 — learning a repeated routine (3 cycles):")
    for cycle in range(3):
        for ev in routine:
            r = wm.perceive(ev)
            mark = "STORED" if r.stored else "  ·   "
            print(f"  cycle {cycle+1} [{mark}] surprise={r.surprise:.2f} ({r.novelty:<8}) {ev[:45]}")
        print()

    # A genuinely novel event — should be surprising and get stored
    print("Phase 2 — a novel event:")
    novel = wm.perceive("Security alert: unauthorized access to Payroll Entry detected")
    print(f"  [{'STORED' if novel.stored else '  ·  '}] surprise={novel.surprise:.2f} ({novel.novelty}) — novel event")

    print(f"\nMemory: {wm.memory.stats()}")
    print(f"  → routine events became 'expected' (not stored); novelty was captured.")

    # Reasoning: state-conditioned recall
    print("\nPhase 3 — state-conditioned recall of 'security':")
    for r in wm.reason("security access alert", top_k=3):
        print(f"  [{r.score:.3f}] {r.episode.text[:55]}  (surprise={r.episode.surprise})")

    wm.save()
    print(f"\nState persisted. {wm.state.summary()}")


if __name__ == "__main__":
    _demo()
