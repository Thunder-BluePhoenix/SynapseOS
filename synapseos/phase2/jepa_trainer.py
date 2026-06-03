"""
Phase 2 — JEPA World State Model training.

Learns to predict the next abstract latent state from a sequence of states —
NOT next-token prediction. Forces the model to capture what's *happening*,
not what words appeared.

Training data: sequences of Frappe workflow/document state transitions.
If real event logs are available → use them.
If not → synth_data_generator.py produces synthetic sequences from the
knowledge graph + business rules extracted in Phase 1.

Install:
    pip install torch  (already installed)

Run:
    python -m synapseos.phase2.jepa_trainer
    python -m synapseos.phase2.jepa_trainer --data data/jepa/transitions.json --epochs 20
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class JEPAConfig:
    # Architecture
    input_dim: int = 128        # dimension of each state embedding
    hidden_dim: int = 128       # encoder hidden size
    latent_dim: int = 64        # world state latent dimension
    num_encoder_layers: int = 2
    num_heads: int = 4
    context_len: int = 3        # how many past states to condition on
    # Training
    data_path: str = "data/jepa/transitions.json"
    output_dir: str = "data/checkpoints/jepa"
    num_epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 3e-4
    val_split: float = 0.1
    ema_decay: float = 0.996    # target encoder EMA — standard JEPA trick
    save_every: int = 5

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class StateEncoder(nn.Module):
    """Encodes a sequence of states into a latent world representation."""

    def __init__(self, cfg: JEPAConfig):
        super().__init__()
        self.input_proj = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_encoder_layers)
        self.output_proj = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.output_norm = nn.LayerNorm(cfg.latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim)
        h = self.input_proj(x)
        h = self.transformer(h)
        return self.output_norm(self.output_proj(h[:, -1, :]))  # (B, latent_dim)


class WorldStatePredictor(nn.Module):
    """Predicts the target latent state from context latent (not from raw tokens)."""

    def __init__(self, cfg: JEPAConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
        )

    def forward(self, context_latent: torch.Tensor) -> torch.Tensor:
        return self.net(context_latent)


class JEPAModel(nn.Module):
    def __init__(self, cfg: JEPAConfig):
        super().__init__()
        self.context_encoder = StateEncoder(cfg)
        self.target_encoder  = StateEncoder(cfg)   # EMA copy — never trained directly
        self.predictor        = WorldStatePredictor(cfg)
        self.cfg = cfg

        # Initialize target encoder as copy of context encoder
        for p_ctx, p_tgt in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            p_tgt.data.copy_(p_ctx.data)
            p_tgt.requires_grad_(False)

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        decay = self.cfg.ema_decay
        for p_ctx, p_tgt in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            p_tgt.data = decay * p_tgt.data + (1 - decay) * p_ctx.data

    def forward(self, context: torch.Tensor, target: torch.Tensor):
        # context: (B, T_ctx, D)  — past states
        # target:  (B, 1,   D)    — next state (what to predict)
        ctx_latent = self.context_encoder(context)          # (B, latent_dim)
        predicted  = self.predictor(ctx_latent)             # (B, latent_dim)

        with torch.no_grad():
            tgt_latent = self.target_encoder(target)        # (B, latent_dim) — EMA target

        loss = F.mse_loss(predicted, tgt_latent)
        return loss, predicted, tgt_latent


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StateTransitionDataset(Dataset):
    """
    Each sample: a context window of states + the next state to predict.

    JSON format expected:
        [
          {"states": [[f1, f2, ...], [f1, f2, ...], ...], "doctype": "Sales Order"},
          ...
        ]
    Each states list is a sequence of state vectors (already embedded/encoded as floats).
    If vectors aren't available yet, synth_data_generator.py creates them.
    """

    def __init__(self, data_path: str, context_len: int):
        with open(data_path) as f:
            sequences = json.load(f)

        self.samples = []
        for seq in sequences:
            states = seq["states"]
            if len(states) < context_len + 1:
                continue
            for i in range(len(states) - context_len):
                ctx = states[i : i + context_len]
                tgt = states[i + context_len : i + context_len + 1]
                self.samples.append((ctx, tgt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ctx, tgt = self.samples[idx]
        return (
            torch.tensor(ctx, dtype=torch.float32),
            torch.tensor(tgt, dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: JEPAConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_path = Path(cfg.data_path)
    if not data_path.exists():
        print(f"Data not found at {data_path}")
        print("Run: python -m synapseos.phase2.synth_data_generator  to generate synthetic data")
        return

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    cfg.save(f"{cfg.output_dir}/jepa_config.json")

    dataset = StateTransitionDataset(cfg.data_path, cfg.context_len)
    print(f"Dataset: {len(dataset)} context→target pairs")

    val_size  = max(1, int(len(dataset) * cfg.val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=2)

    model = JEPAModel(cfg).to(device)
    optimizer = torch.optim.AdamW(
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()),
        lr=cfg.learning_rate, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)

    best_val_loss = float("inf")

    for epoch in range(1, cfg.num_epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for context, target in train_loader:
            context, target = context.to(device), target.to(device)
            optimizer.zero_grad()
            loss, _, _ = model(context, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.update_target_encoder()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for context, target in val_loader:
                context, target = context.to(device), target.to(device)
                loss, _, _ = model(context, target)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{cfg.num_epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if epoch % cfg.save_every == 0:
            ckpt_path = f"{cfg.output_dir}/jepa_epoch{epoch:03d}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Checkpoint → {ckpt_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{cfg.output_dir}/jepa_best.pt")

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Best model  → {cfg.output_dir}/jepa_best.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    default="data/jepa/transitions.json")
    parser.add_argument("--output",  default="data/checkpoints/jepa")
    parser.add_argument("--epochs",  type=int,   default=20)
    parser.add_argument("--batch",   type=int,   default=64)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--latent",  type=int,   default=128)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = JEPAConfig(
        data_path=args.data,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        latent_dim=args.latent,
    )

    print("=== JEPA World State Model Config ===")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")

    if args.dry_run:
        return

    train(cfg)


if __name__ == "__main__":
    main()
