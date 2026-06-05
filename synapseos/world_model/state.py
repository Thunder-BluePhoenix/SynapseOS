"""
WorldState — the persistent latent z_t of the SynapseOS world model.

This is the single evolving state that everything reads from and writes to —
the thing that makes the system "feel like a brain" rather than a stateless
pipeline. It persists across sessions (serialized to disk), is versioned for
rollback (like a git for cognition), and carries:

  z            — the world-state latent vector (evolves via CfC liquid dynamics)
  prediction   — the model's expectation of the next observation
  step         — how many observations integrated so far
  surprise_ema — running average surprise (how well-modelled the world is now)

Versioning lets us snapshot a known-good cognitive state and roll back if
adaptation drifts (reuses the DriftMonitor philosophy at the state level).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import torch


@dataclass
class WorldState:
    z: list[float]                       # world-state latent (state_dim)
    prediction: list[float]              # predicted next observation (obs_dim)
    step: int = 0
    surprise_ema: float = 0.0            # exponential moving average of surprise
    created_at: str = ""
    updated_at: str = ""
    version: int = 0

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at

    # ------------------------------------------------------------------
    # Tensor views
    # ------------------------------------------------------------------

    def z_tensor(self, device=None, dtype=torch.float32) -> torch.Tensor:
        t = torch.tensor(self.z, dtype=dtype)
        return t.to(device) if device else t

    def set_z(self, z: torch.Tensor) -> None:
        self.z = z.detach().cpu().float().flatten().tolist()
        self.updated_at = datetime.now().isoformat()

    def set_prediction(self, p: torch.Tensor) -> None:
        self.prediction = p.detach().cpu().float().flatten().tolist()

    def update_surprise(self, surprise: float, alpha: float = 0.1) -> None:
        self.surprise_ema = (1 - alpha) * self.surprise_ema + alpha * surprise

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def fresh(cls, state_dim: int, obs_dim: int) -> "WorldState":
        return cls(
            z=[0.0] * state_dim,
            prediction=[0.0] * obs_dim,
            step=0,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WorldState":
        return cls(**d)

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = datetime.now().isoformat()
        p.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str, state_dim: int = 64, obs_dim: int = 128) -> "WorldState":
        p = Path(path)
        if p.exists():
            try:
                return cls.from_dict(json.loads(p.read_text()))
            except Exception:
                pass
        return cls.fresh(state_dim, obs_dim)

    # ------------------------------------------------------------------
    # Versioning (rollback support)
    # ------------------------------------------------------------------

    def snapshot(self, snapshot_dir: str) -> str:
        """Save a versioned snapshot; returns the snapshot id."""
        d = Path(snapshot_dir)
        d.mkdir(parents=True, exist_ok=True)
        snap_id = f"ws_v{self.version}_{int(time.time())}"
        self.version += 1
        (d / f"{snap_id}.json").write_text(json.dumps(self.to_dict(), indent=2))
        return snap_id

    @staticmethod
    def restore(snapshot_dir: str, snap_id: str) -> "WorldState":
        path = Path(snapshot_dir) / f"{snap_id}.json"
        return WorldState.from_dict(json.loads(path.read_text()))

    def summary(self) -> str:
        return (
            f"WorldState v{self.version} | step={self.step} | "
            f"surprise_ema={self.surprise_ema:.3f} | "
            f"z_dim={len(self.z)} | updated={self.updated_at[:19]}"
        )
