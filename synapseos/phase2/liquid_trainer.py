"""
Phase 2 — Liquid Neural Network (LTC) training.

Trains on time-series operational data: CPU, memory, queue depth, DB latency,
event rates. The LTC layer continues adapting at inference time via ODE solver —
it learns project-specific patterns without retraining.

Data format expected (data/liquid/timeseries.json):
    [
      {
        "series": [[cpu, mem, queue, db_lat, event_rate], ...],  # T timesteps
        "target": [[cpu, mem, queue, db_lat, event_rate], ...],  # T next-step values
        "source": "frappe_monitor"
      },
      ...
    ]

Install:
    pip install torchdiffeq ncps

Run:
    python -m synapseos.phase2.liquid_trainer
    python -m synapseos.phase2.liquid_trainer --neurons 64 --dry-run
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split


@dataclass
class LiquidConfig:
    # Architecture
    input_dim: int = 5           # number of input features per timestep
    num_neurons: int = 32        # LTC neuron count
    num_outputs: int = 5         # output features (same as input for next-step prediction)
    # Training
    data_path: str = "data/liquid/timeseries.json"
    output_dir: str = "data/checkpoints/liquid"
    num_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 5e-4
    val_split: float = 0.1
    # ODE solver
    ode_method: str = "rk4"      # rk4 | euler | dopri5

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


# ---------------------------------------------------------------------------
# Model — ncps LTC or fallback GRU
# ---------------------------------------------------------------------------

def get_liquid_model(cfg: LiquidConfig) -> tuple[nn.Module, bool]:
    try:
        from ncps import wirings
        from ncps.torch import LTC

        wiring = wirings.AutoNCP(cfg.num_neurons, cfg.num_outputs)
        model = LTC(cfg.input_dim, wiring, return_sequences=True, batch_first=True)
        print(f"Using LTC (ncps) — {cfg.num_neurons} neurons → {cfg.num_outputs} outputs")
        return model, True

    except ImportError:
        print("ncps not installed — using GRU fallback (install ncps for true LNN behavior)")
        model = LiquidFallbackGRU(cfg.input_dim, cfg.num_neurons, cfg.num_outputs)
        return model, False


class LiquidFallbackGRU(nn.Module):
    """
    GRU fallback for when ncps is not installed.
    Same interface as LTC. Does NOT adapt weights at inference — install ncps for that.
    """

    def __init__(self, input_dim: int, hidden: int, output_dim: int):
        super().__init__()
        self.gru  = nn.GRU(input_dim, hidden, batch_first=True, num_layers=2)
        self.head = nn.Linear(hidden, output_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None):
        out, h_new = self.gru(x, h)
        return self.head(out), h_new


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TimeSeriesDataset(Dataset):
    """
    Each sample: (input_sequence, target_sequence).
    Input and target are the same length — task is next-step prediction.
    """

    def __init__(self, data_path: str):
        with open(data_path) as f:
            records = json.load(f)

        self.inputs  = []
        self.targets = []

        for rec in records:
            series = rec["series"]
            target = rec.get("target", series[1:] + [series[-1]])
            min_len = min(len(series), len(target))
            if min_len < 4:
                continue
            self.inputs.append(torch.tensor(series[:min_len], dtype=torch.float32))
            self.targets.append(torch.tensor(target[:min_len], dtype=torch.float32))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def collate_fn(batch):
    """Pad variable-length sequences to batch max length."""
    inputs, targets = zip(*batch)
    max_len = max(x.shape[0] for x in inputs)
    feat_dim = inputs[0].shape[1]

    padded_inputs  = torch.zeros(len(inputs),  max_len, feat_dim)
    padded_targets = torch.zeros(len(targets), max_len, feat_dim)
    mask = torch.zeros(len(inputs), max_len, dtype=torch.bool)

    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        L = inp.shape[0]
        padded_inputs[i, :L]  = inp
        padded_targets[i, :L] = tgt
        mask[i, :L] = True

    return padded_inputs, padded_targets, mask


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: LiquidConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_path = Path(cfg.data_path)
    if not data_path.exists():
        print(f"Data not found at {data_path}")
        print("Run: python -m synapseos.phase2.synth_data_generator  to generate synthetic time-series")
        return

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    cfg.save(f"{cfg.output_dir}/liquid_config.json")

    dataset = TimeSeriesDataset(cfg.data_path)
    print(f"Dataset: {len(dataset)} time-series sequences")

    val_size   = max(1, int(len(dataset) * cfg.val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    model, is_ltc = get_liquid_model(cfg)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)

    best_val_loss = float("inf")

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()

            out, _ = model(x)
            # Only compute loss on non-padded positions
            loss = (nn.functional.mse_loss(out, y, reduction="none") * mask.unsqueeze(-1)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, mask in val_loader:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                out, _ = model(x)
                loss = (nn.functional.mse_loss(out, y, reduction="none") * mask.unsqueeze(-1)).mean()
                val_loss += loss.item()
        val_loss /= len(val_loader)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{cfg.num_epochs}  train={train_loss:.5f}  val={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{cfg.output_dir}/liquid_best.pt")

    print(f"\nBest val MSE : {best_val_loss:.5f}")
    print(f"Best model  → {cfg.output_dir}/liquid_best.pt")

    if is_ltc:
        print("\nNOTE: At inference, this LTC model continues adapting weights via ODE solver.")
        print("Each new input shifts the internal state — no retraining needed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    default="data/liquid/timeseries.json")
    parser.add_argument("--output",  default="data/checkpoints/liquid")
    parser.add_argument("--epochs",  type=int,   default=50)
    parser.add_argument("--batch",   type=int,   default=32)
    parser.add_argument("--neurons", type=int,   default=32)
    parser.add_argument("--lr",      type=float, default=5e-4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = LiquidConfig(
        data_path=args.data,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch,
        num_neurons=args.neurons,
        learning_rate=args.lr,
    )

    print("=== Liquid NN (LTC) Config ===")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")

    if args.dry_run:
        return

    train(cfg)


if __name__ == "__main__":
    main()
