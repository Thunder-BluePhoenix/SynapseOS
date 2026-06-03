"""
CfC (Closed-form Continuous-time) Liquid cell — pure PyTorch.

Replaces the `ncps` package (Neural Circuit Policies / Liquid Time-Constant
networks). ncps is the reference implementation; this is a faithful pure-PyTorch
CfC cell from Hasani et al. "Closed-form Continuous-time Neural Networks" (2022),
so it needs no install and runs on CPU/MPS.

Why CfC: the Liquid layer is the only SynapseOS component that keeps adapting
after deployment. The CfC cell approximates a continuous-time ODE in closed
form (no ODE solver needed), making it fast and stable on CPU.

Core equation (per the paper):
    h(t) = σ(-f(x,h,θ_f)·t) ⊙ g(x,h,θ_g) + [1 - σ(-f(x,h,θ_f)·t)] ⊙ h(x,h,θ_h)

where f, g, h are small MLPs and t is the elapsed time between inputs.
The time-gate σ(-f·t) gives each neuron its own input-dependent time constant —
the "liquid" property.

Run:
    python -m synapseos.local.liquid_cfc                # smoke test
    python -m synapseos.local.liquid_cfc --train-demo   # fit a noisy sine, MSE drops
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn as nn

from synapseos.local.device import get_device


# ---------------------------------------------------------------------------
# CfC cell
# ---------------------------------------------------------------------------

class CfCCell(nn.Module):
    """
    A single Closed-form Continuous-time recurrent cell.

    Args:
        input_size:  dimension of x_t
        hidden_size: dimension of h_t
        backbone:    hidden width of the shared backbone MLP
    """

    def __init__(self, input_size: int, hidden_size: int, backbone: int = 64):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size

        # Shared backbone over [x, h]
        self.backbone = nn.Sequential(
            nn.Linear(input_size + hidden_size, backbone),
            nn.Tanh(),
            nn.Linear(backbone, backbone),
            nn.Tanh(),
        )

        # Three heads: f (time-constant), g and h (candidate states)
        self.ff_f = nn.Linear(backbone, hidden_size)   # input-dependent time constant
        self.ff_g = nn.Linear(backbone, hidden_size)   # "fast" candidate
        self.ff_h = nn.Linear(backbone, hidden_size)   # "slow" candidate

        # Learnable per-neuron time-scale and bias on the gate
        self.w_tau = nn.Parameter(torch.ones(hidden_size))
        self.A     = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor, h: torch.Tensor, ts: float | torch.Tensor = 1.0):
        """
        One step.
        x:  (B, input_size)
        h:  (B, hidden_size)
        ts: elapsed time since last step (scalar or (B,1)) — the continuous-time part.
        Returns new hidden state (B, hidden_size).
        """
        feat = self.backbone(torch.cat([x, h], dim=-1))

        f = self.ff_f(feat)
        g = self.ff_g(feat)
        h_cand = self.ff_h(feat)

        if not torch.is_tensor(ts):
            ts = torch.tensor(float(ts), device=x.device, dtype=x.dtype)

        # Closed-form continuous-time gate: σ(-(w_tau·f)·t) per neuron
        gate = torch.sigmoid(-(self.w_tau * f) * ts)

        # h(t) = gate ⊙ g + (1 - gate) ⊙ (A ⊙ tanh(h_cand))
        new_h = gate * g + (1.0 - gate) * (self.A * torch.tanh(h_cand))
        return new_h


# ---------------------------------------------------------------------------
# Sequence wrapper
# ---------------------------------------------------------------------------

class LiquidNet(nn.Module):
    """
    CfC-based liquid network for time-series — drop-in replacement for the
    ncps-backed Liquid layer in Phase 2 when ncps isn't installed.

    Processes a sequence and (optionally) predicts a target at each step.
    Supports irregular time steps via the `timestamps` argument.
    """

    def __init__(self, input_size: int, hidden_size: int = 32,
                 output_size: int = 1, backbone: int = 64):
        super().__init__()
        self.cell    = CfCCell(input_size, hidden_size, backbone)
        self.readout = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, x_seq: torch.Tensor,
                timestamps: torch.Tensor | None = None,
                h0: torch.Tensor | None = None):
        """
        x_seq:      (B, L, input_size)
        timestamps: (B, L) elapsed-time per step, or None for unit steps
        Returns:    outputs (B, L, output_size), final_hidden (B, hidden_size)
        """
        B, L, _ = x_seq.shape
        h = h0 if h0 is not None else torch.zeros(B, self.hidden_size,
                                                  device=x_seq.device, dtype=x_seq.dtype)
        outs = []
        for t in range(L):
            ts = 1.0 if timestamps is None else timestamps[:, t:t+1]
            h  = self.cell(x_seq[:, t], h, ts)
            outs.append(self.readout(h))
        return torch.stack(outs, dim=1), h

    @torch.no_grad()
    def adapt_step(self, x: torch.Tensor, h: torch.Tensor, ts: float = 1.0):
        """
        Single online adaptation step — what runs in production between requests.
        Returns (prediction, new_hidden). No gradient: the liquid state evolves
        continuously without backprop, the defining property of the layer.
        """
        new_h = self.cell(x, h, ts)
        return self.readout(new_h), new_h

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Smoke test + train demo
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    device = get_device(prefer_small=True)
    net    = LiquidNet(input_size=3, hidden_size=16, output_size=1).to(device)
    print(f"[CfC] params: {net.num_params():,}  device: {device}")

    x = torch.randn(4, 20, 3, device=device)
    ts = torch.rand(4, 20, device=device) + 0.5      # irregular time steps
    out, h = net(x, timestamps=ts)
    assert out.shape == (4, 20, 1)
    assert h.shape == (4, 16)

    # Online adaptation (no grad)
    h_state = torch.zeros(1, 16, device=device)
    pred, h_state = net.adapt_step(torch.randn(1, 3, device=device), h_state, ts=0.7)
    assert pred.shape == (1, 1)
    print(f"[CfC] forward {tuple(out.shape)} | adapt_step {tuple(pred.shape)} | irregular Δt OK")
    print("[CfC] smoke test PASSED")


def _train_demo(steps: int = 300) -> None:
    """Fit a noisy sine wave — a classic liquid-network time-series task."""
    device = get_device(prefer_small=True)
    net    = LiquidNet(input_size=1, hidden_size=32, output_size=1).to(device)
    opt    = torch.optim.Adam(net.parameters(), lr=5e-3)
    lossf  = nn.MSELoss()

    # Build sequence: input = sin(t), target = sin(t + phase)  (predict ahead)
    L = 50
    t = torch.linspace(0, 4 * math.pi, L + 1)
    seq_in  = torch.sin(t[:-1]).reshape(1, L, 1).to(device)
    seq_tgt = torch.sin(t[1:]).reshape(1, L, 1).to(device)
    seq_in += 0.05 * torch.randn_like(seq_in)        # noise

    print(f"[CfC] train demo: {net.num_params():,} params on {device}, {steps} steps")
    t0 = time.time()
    first = None
    for step in range(steps):
        out, _ = net(seq_in)
        loss   = lossf(out, seq_tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        if first is None:
            first = loss.item()
        if (step + 1) % 100 == 0:
            print(f"  step {step+1:>3}  mse {loss.item():.5f}")
    elapsed = time.time() - t0
    print(f"[CfC] mse {first:.5f} → {loss.item():.5f} in {elapsed:.1f}s")
    assert loss.item() < first * 0.5, "MSE should at least halve"
    print("[CfC] train demo PASSED — liquid net is learning the dynamics")


def main():
    parser = argparse.ArgumentParser(description="CfC Liquid cell (pure PyTorch)")
    parser.add_argument("--train-demo", action="store_true")
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()

    _smoke_test()
    if args.train_demo:
        print()
        _train_demo(args.steps)


if __name__ == "__main__":
    main()
