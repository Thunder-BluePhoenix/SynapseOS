"""
Phase 2 — KAN (Kolmogorov-Arnold Network) Reasoning Layer.

Learns interpretable symbolic relationships from domain metric pairs.
After training, call model.auto_symbolic() to extract readable formulas
like: "approval_time = 8.3 / approver_count^1.8 + 0.4 * amount_log"

These formulas become auditable business rules fed to the Neurosymbolic core.

Data format expected (data/kan/metric_pairs.json):
    [{"inputs": [v1, v2, v3, v4], "output": v5, "label": "approval_time"}, ...]

Install:
    pip install pykan

Run:
    python -m synapseos.phase2.kan_trainer
    python -m synapseos.phase2.kan_trainer --data data/kan/metric_pairs.json --dry-run
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
class KANConfig:
    # Architecture — width defines layer sizes, e.g. [4, 8, 1]
    width: list[int] = None       # set in __post_init__
    grid: int = 5                 # spline grid points
    k: int = 3                    # spline order
    # Training
    data_path: str = "data/kan/metric_pairs.json"
    output_dir: str = "data/checkpoints/kan"
    num_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    val_split: float = 0.15
    # Symbolic extraction
    run_auto_symbolic: bool = True
    symbolic_threshold: float = 0.95   # R² threshold to accept a symbolic formula

    def __post_init__(self):
        if self.width is None:
            self.width = [4, 8, 1]

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


# ---------------------------------------------------------------------------
# Fallback MLP (same interface as KAN) when pykan is unavailable
# ---------------------------------------------------------------------------

class FallbackMLP(nn.Module):
    """
    Drop-in replacement for KAN when pykan is not installed.
    Trains identically but cannot extract symbolic formulas.
    Used for rapid iteration before GPU setup.
    """

    def __init__(self, width: list[int], **kwargs):
        super().__init__()
        layers = []
        for i in range(len(width) - 1):
            layers.append(nn.Linear(width[i], width[i + 1]))
            if i < len(width) - 2:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def auto_symbolic(self, *args, **kwargs):
        print("  [FallbackMLP] auto_symbolic() not available — install pykan for formula extraction")
        return {}

    def plot(self, *args, **kwargs):
        pass


def get_kan_model(cfg: KANConfig):
    try:
        from kan import KAN
        model = KAN(width=cfg.width, grid=cfg.grid, k=cfg.k)
        print("Using KAN (pykan installed)")
        return model, True
    except ImportError:
        print("pykan not installed — using FallbackMLP (install pykan for symbolic extraction)")
        model = FallbackMLP(width=cfg.width)
        return model, False


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MetricPairsDataset(Dataset):
    """
    Loads domain metric pairs.
    Each item: (input_vector, output_scalar).
    """

    def __init__(self, data_path: str):
        with open(data_path) as f:
            records = json.load(f)

        self.inputs  = []
        self.outputs = []
        self.labels  = []

        for rec in records:
            self.inputs.append(rec["inputs"])
            self.outputs.append([rec["output"]])
            self.labels.append(rec.get("label", "unknown"))

        self.inputs  = torch.tensor(self.inputs,  dtype=torch.float32)
        self.outputs = torch.tensor(self.outputs, dtype=torch.float32)

        # Normalize inputs to [0, 1] for KAN stability
        self.input_min  = self.inputs.min(dim=0).values
        self.input_max  = self.inputs.max(dim=0).values
        eps = 1e-8
        self.inputs = (self.inputs - self.input_min) / (self.input_max - self.input_min + eps)

        # Normalize outputs
        self.output_mean = self.outputs.mean()
        self.output_std  = self.outputs.std() + eps
        self.outputs = (self.outputs - self.output_mean) / self.output_std

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

    def denormalize_output(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.output_std + self.output_mean

    def save_normalization(self, path: str) -> None:
        norm = {
            "input_min":   self.input_min.tolist(),
            "input_max":   self.input_max.tolist(),
            "output_mean": float(self.output_mean),
            "output_std":  float(self.output_std),
        }
        with open(path, "w") as f:
            json.dump(norm, f, indent=2)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: KANConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_path = Path(cfg.data_path)
    if not data_path.exists():
        print(f"Data not found at {data_path}")
        print("Run: python -m synapseos.phase2.synth_data_generator  to generate synthetic KAN data")
        return

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    cfg.save(f"{cfg.output_dir}/kan_config.json")

    dataset = MetricPairsDataset(cfg.data_path)
    dataset.save_normalization(f"{cfg.output_dir}/normalization.json")
    print(f"Dataset: {len(dataset)} metric pairs  input_dim={dataset.inputs.shape[1]}")

    # Update width to match actual input dim
    if cfg.width[0] != dataset.inputs.shape[1]:
        cfg.width[0] = dataset.inputs.shape[1]
        print(f"  Auto-adjusted KAN input width to {cfg.width[0]}")

    val_size   = max(1, int(len(dataset) * cfg.val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False)

    model, is_kan = get_kan_model(cfg)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_loss = float("inf")

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = nn.functional.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += nn.functional.mse_loss(pred, y).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{cfg.num_epochs}  train={train_loss:.5f}  val={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{cfg.output_dir}/kan_best.pt")

    print(f"\nBest val MSE: {best_val_loss:.5f}")

    # Symbolic extraction
    if cfg.run_auto_symbolic and is_kan:
        print("\nExtracting symbolic formulas (auto_symbolic)...")
        model.load_state_dict(torch.load(f"{cfg.output_dir}/kan_best.pt", map_location=device))
        formulas = model.auto_symbolic(lib=["x", "x^2", "x^3", "sqrt", "log", "exp", "1/x", "1/x^2"])
        formula_path = f"{cfg.output_dir}/symbolic_formulas.json"
        with open(formula_path, "w") as f:
            json.dump({"formulas": str(formulas)}, f, indent=2)
        print(f"Symbolic formulas → {formula_path}")
        print(f"  {formulas}")

    print(f"\nBest model → {cfg.output_dir}/kan_best.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    default="data/kan/metric_pairs.json")
    parser.add_argument("--output",  default="data/checkpoints/kan")
    parser.add_argument("--epochs",  type=int,   default=100)
    parser.add_argument("--batch",   type=int,   default=64)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--grid",    type=int,   default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = KANConfig(
        data_path=args.data,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        grid=args.grid,
    )

    print("=== KAN Reasoning Layer Config ===")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")

    if args.dry_run:
        return

    train(cfg)


if __name__ == "__main__":
    main()
