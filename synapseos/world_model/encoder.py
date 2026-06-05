"""
Observation encoder — perception layer of the world model.

Turns a raw input (text today; voice/vision later via the same interface)
into an observation embedding o_t. The key property for an associative world
model is: similar observations → similar embeddings, deterministically.

Two backends:
  - "hash"  : n-gram hashing (default) — deterministic, no training, gives
              genuine semantic-ish similarity (shared n-grams → high cosine).
              This is the same family as the existing EpisodicMemory embedding,
              chosen because an *untrained* neural encoder would give random
              (though consistent) vectors with no similarity structure.
  - "mamba" : pooled latent from the local MinimalMamba (synapseos/local).
              Structurally richer but only meaningful once trained — offered
              for when a trained Mamba checkpoint exists.

obs_dim defaults to 128 to match the rest of the memory stack.
"""

from __future__ import annotations

import hashlib
import math
import re

import torch


class ObservationEncoder:
    def __init__(self, obs_dim: int = 128, backend: str = "hash",
                 mamba_checkpoint: str | None = None):
        self.obs_dim = obs_dim
        self.backend = backend
        self._mamba  = None
        if backend == "mamba":
            self._load_mamba(mamba_checkpoint)

    def _load_mamba(self, checkpoint: str | None) -> None:
        try:
            from synapseos.local.mamba_cpu import MinimalMamba, MambaConfig
            cfg = MambaConfig(vocab_size=256, d_model=self.obs_dim, n_layers=2, d_state=8)
            self._mamba = MinimalMamba(cfg)
            if checkpoint:
                import torch as _t
                self._mamba.load_state_dict(_t.load(checkpoint, map_location="cpu", weights_only=True))
            self._mamba.eval()
        except Exception as e:
            print(f"[Encoder] Mamba backend unavailable ({e}) — using hash")
            self.backend = "hash"

    # ------------------------------------------------------------------

    def encode(self, text: str) -> torch.Tensor:
        """text → (obs_dim,) float tensor, L2-normalised."""
        if self.backend == "mamba" and self._mamba is not None:
            return self._encode_mamba(text)
        return self._encode_hash(text)

    def _encode_hash(self, text: str) -> torch.Tensor:
        vec = [0.0] * self.obs_dim
        tokens = re.findall(r"\w+", text.lower())
        for token in tokens:
            for n in (1, 2, 3):
                for i in range(len(token) - n + 1):
                    gram = token[i:i+n]
                    h = int(hashlib.md5(gram.encode()).hexdigest(), 16)
                    idx  = h % self.obs_dim
                    sign = 1 if (h >> 8) & 1 else -1
                    vec[idx] += sign
        t = torch.tensor(vec, dtype=torch.float32)
        norm = t.norm(2)
        return t / norm if norm > 0 else t

    def _encode_mamba(self, text: str) -> torch.Tensor:
        from synapseos.local.mamba_cpu import bytes_encode
        ids = bytes_encode(text, max_len=128)
        with torch.no_grad():
            latent = self._mamba.encode(ids).squeeze(0)
        norm = latent.norm(2)
        return latent / norm if norm > 0 else latent


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """1 - cosine similarity, clamped to [0, 2]. 0 = identical, 1 = orthogonal."""
    a = a.flatten(); b = b.flatten()
    na, nb = a.norm(2), b.norm(2)
    if na == 0 or nb == 0:
        return 1.0
    cos = float((a @ b) / (na * nb))
    return max(0.0, min(2.0, 1.0 - cos))
