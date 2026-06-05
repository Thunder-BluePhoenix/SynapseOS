"""
Predictive Memory — surprise-gated episodic store.

This is the part that makes memory *experience* rather than a log. Standard
RAG stores everything and retrieves by string similarity. Here, an episode is
written only when the world model FAILS to predict the observation — i.e. when
something surprising happened. Expected, boring inputs update the state but
leave no trace, exactly like the hippocampus.

Each episode carries:
  - the observation embedding o_t (for associative recall)
  - the world-state latent z_t at the moment it happened (for state-conditioned recall)
  - surprise (= salience; drives consolidation priority)
  - prev_id (temporal link to the previous episode — the experience timeline)
  - causal_parent (the most similar prior episode — a learned causal/associative edge)
  - access_count + last_access (for usage-based forgetting)

Persisted to JSON so experience survives restarts.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import torch


@dataclass
class Episode:
    id: str
    text: str
    obs: list[float]            # observation embedding o_t
    z: list[float]              # world-state latent z_t at write time
    surprise: float             # prediction error that triggered the write
    timestamp: str
    step: int
    prev_id: str = ""           # temporal predecessor (the episode before this)
    causal_parent: str = ""     # most similar prior episode (associative edge)
    tags: list[str] = field(default_factory=list)
    access_count: int = 0
    last_access: str = ""
    consolidated: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Episode":
        return cls(**d)

    def obs_tensor(self) -> torch.Tensor:
        return torch.tensor(self.obs, dtype=torch.float32)

    def z_tensor(self) -> torch.Tensor:
        return torch.tensor(self.z, dtype=torch.float32)


class PredictiveMemory:
    """
    Surprise-gated episodic memory with temporal + causal structure.
    """

    def __init__(
        self,
        store_path: str = "data/world_model/episodes.json",
        surprise_threshold: float = 0.35,   # write only if surprise exceeds this
        max_episodes: int = 5000,
    ):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.surprise_threshold = surprise_threshold
        self.max_episodes = max_episodes
        self.episodes: list[Episode] = []
        self._last_id: str = ""
        self._load()

    def _load(self) -> None:
        if self.store_path.exists():
            try:
                raw = json.loads(self.store_path.read_text())
                self.episodes = [Episode.from_dict(e) for e in raw]
                if self.episodes:
                    self._last_id = self.episodes[-1].id
            except Exception:
                self.episodes = []

    def save(self) -> None:
        self.store_path.write_text(json.dumps([e.to_dict() for e in self.episodes], indent=2))

    # ------------------------------------------------------------------
    # Write (the surprise gate)
    # ------------------------------------------------------------------

    def should_write(self, surprise: float) -> bool:
        return surprise >= self.surprise_threshold

    def write(
        self,
        text: str,
        obs: torch.Tensor,
        z: torch.Tensor,
        surprise: float,
        step: int,
        tags: list[str] | None = None,
    ) -> Episode | None:
        """
        Write an episode iff the observation was surprising.
        Returns the Episode if written, else None.
        """
        if not self.should_write(surprise):
            return None

        ep_id = f"ep_{step}_{int(time.time()*1000) % 100000}"
        causal_parent = self._find_causal_parent(obs)

        ep = Episode(
            id=ep_id,
            text=text,
            obs=obs.detach().cpu().float().flatten().tolist(),
            z=z.detach().cpu().float().flatten().tolist(),
            surprise=round(surprise, 4),
            timestamp=datetime.now().isoformat(),
            step=step,
            prev_id=self._last_id,
            causal_parent=causal_parent,
            tags=tags or [],
            last_access=datetime.now().isoformat(),
        )
        self.episodes.append(ep)
        self._last_id = ep_id
        self._evict_if_needed()
        return ep

    def _find_causal_parent(self, obs: torch.Tensor, top: int = 1) -> str:
        """Most similar prior episode — an associative/causal edge."""
        if not self.episodes:
            return ""
        best_id, best_sim = "", -1.0
        for ep in self.episodes[-200:]:           # recent window for speed
            sim = float(torch.cosine_similarity(obs.flatten(), ep.obs_tensor(), dim=0))
            if sim > best_sim:
                best_sim, best_id = sim, ep.id
        return best_id if best_sim > 0.3 else ""

    def _evict_if_needed(self) -> None:
        """Usage- and salience-weighted forgetting when over capacity."""
        if len(self.episodes) <= self.max_episodes:
            return
        # score = salience (surprise) + usage (access_count), keep top-K
        def score(e: Episode) -> float:
            return e.surprise + 0.1 * e.access_count + (0.2 if e.consolidated else 0.0)
        self.episodes.sort(key=score, reverse=True)
        self.episodes = self.episodes[:self.max_episodes]
        # restore chronological order
        self.episodes.sort(key=lambda e: e.step)

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def get(self, ep_id: str) -> Episode | None:
        for e in self.episodes:
            if e.id == ep_id:
                return e
        return None

    def timeline(self, ep_id: str, back: int = 5) -> list[Episode]:
        """Walk the temporal prev_id chain backwards from an episode."""
        chain, cur, seen = [], self.get(ep_id), set()
        while cur and len(chain) < back and cur.id not in seen:
            chain.append(cur); seen.add(cur.id)
            cur = self.get(cur.prev_id) if cur.prev_id else None
        return chain

    def high_salience(self, threshold: float = 0.5, limit: int = 50) -> list[Episode]:
        eps = [e for e in self.episodes if e.surprise >= threshold and not e.consolidated]
        eps.sort(key=lambda e: e.surprise, reverse=True)
        return eps[:limit]

    def touch(self, ep_id: str) -> None:
        ep = self.get(ep_id)
        if ep:
            ep.access_count += 1
            ep.last_access = datetime.now().isoformat()

    def stats(self) -> dict:
        if not self.episodes:
            return {"episodes": 0}
        surprises = [e.surprise for e in self.episodes]
        return {
            "episodes":        len(self.episodes),
            "consolidated":    sum(1 for e in self.episodes if e.consolidated),
            "mean_surprise":   round(sum(surprises) / len(surprises), 3),
            "max_surprise":    round(max(surprises), 3),
            "has_temporal_links": sum(1 for e in self.episodes if e.prev_id),
            "has_causal_links":   sum(1 for e in self.episodes if e.causal_parent),
        }

    def __len__(self) -> int:
        return len(self.episodes)
