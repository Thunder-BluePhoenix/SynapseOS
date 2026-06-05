"""
State-conditioned retrieval.

Plain RAG retrieves by query-string similarity alone. A world model retrieves
relative to *where it currently is* — the same query means different things
depending on the system's current latent state z_t. This module scores each
episode by a blend of:

  - observation similarity : how well the episode matches the query embedding
  - state similarity       : how close the episode's z_t is to the current z_t
  - salience               : how surprising/important the episode was
  - recency                : a mild bias toward recent experience

The result is associative recall conditioned on context, not keyword lookup.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from synapseos.world_model.predictive_memory import PredictiveMemory, Episode


@dataclass
class RetrievedEpisode:
    episode: Episode
    score: float
    obs_sim: float
    state_sim: float

    def to_dict(self) -> dict:
        return {
            "id":        self.episode.id,
            "text":      self.episode.text[:120],
            "score":     round(self.score, 4),
            "obs_sim":   round(self.obs_sim, 4),
            "state_sim": round(self.state_sim, 4),
            "surprise":  self.episode.surprise,
            "step":      self.episode.step,
        }


class StateConditionedRetrieval:
    def __init__(
        self,
        memory: PredictiveMemory,
        w_obs: float = 0.5,
        w_state: float = 0.3,
        w_salience: float = 0.1,
        w_recency: float = 0.1,
    ):
        self.memory = memory
        self.w_obs      = w_obs
        self.w_state    = w_state
        self.w_salience = w_salience
        self.w_recency  = w_recency

    def retrieve(
        self,
        query_obs: torch.Tensor,
        current_z: torch.Tensor,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[RetrievedEpisode]:
        if not self.memory.episodes:
            return []

        q = query_obs.flatten()
        z = current_z.flatten()
        max_step = max(e.step for e in self.memory.episodes) or 1

        scored: list[RetrievedEpisode] = []
        for ep in self.memory.episodes:
            obs_sim   = float(torch.cosine_similarity(q, ep.obs_tensor(), dim=0))
            # state similarity (guard dim mismatch)
            ez = ep.z_tensor()
            if ez.shape == z.shape and z.norm() > 0:
                state_sim = float(torch.cosine_similarity(z, ez, dim=0))
            else:
                state_sim = 0.0
            salience  = min(1.0, ep.surprise)
            recency   = ep.step / max_step

            score = (
                self.w_obs      * obs_sim   +
                self.w_state    * state_sim +
                self.w_salience * salience  +
                self.w_recency  * recency
            )
            if score >= min_score:
                scored.append(RetrievedEpisode(ep, score, obs_sim, state_sim))

        scored.sort(key=lambda r: r.score, reverse=True)
        top = scored[:top_k]
        # mark retrieved episodes as accessed (usage-based retention)
        for r in top:
            self.memory.touch(r.episode.id)
        return top

    def retrieve_with_timeline(
        self,
        query_obs: torch.Tensor,
        current_z: torch.Tensor,
        top_k: int = 3,
        timeline_depth: int = 3,
    ) -> list[dict]:
        """Retrieve top episodes plus their temporal lead-up (the 'story')."""
        hits = self.retrieve(query_obs, current_z, top_k=top_k)
        out = []
        for r in hits:
            timeline = self.memory.timeline(r.episode.id, back=timeline_depth)
            out.append({
                "match":    r.to_dict(),
                "timeline": [e.text[:80] for e in timeline],
            })
        return out
