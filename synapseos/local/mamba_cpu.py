"""
Minimal Mamba — pure-PyTorch selective-scan SSM.

Replaces the CUDA-only `mamba-ssm` package. The official package needs a
custom CUDA kernel for the selective scan; this implements the same math
with a sequential scan in plain PyTorch, so it runs on CPU and Apple MPS.

It is slower than the fused CUDA kernel (O(L) python-level loop over the
sequence) but numerically equivalent and fully trainable. For SynapseOS's
domain corpus at small scale this trains in minutes on MPS.

Architecture (one MambaBlock):
    x → in_proj → (x, z)
    x → conv1d (causal, depthwise) → SiLU
    x → x_proj → (Δ, B, C)        (input-dependent SSM params — the "selective" part)
    selective_scan(x, Δ, A, B, C, D) → y
    y = y * SiLU(z)               (gating)
    y → out_proj

MinimalMamba stacks N MambaBlocks with residual connections + RMSNorm,
plus an embedding and an LM head — a drop-in tiny language model.

Run:
    python -m synapseos.local.mamba_cpu                 # smoke test forward/backward
    python -m synapseos.local.mamba_cpu --train-demo    # train a few steps on toy data
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from synapseos.local.device import get_device


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MambaConfig:
    vocab_size: int = 256        # byte-level by default
    d_model:    int = 128        # embedding / residual width
    n_layers:   int = 4
    d_state:    int = 16         # SSM state dimension (N)
    d_conv:     int = 4          # causal conv kernel size
    expand:     int = 2          # inner expansion factor
    dt_rank:    int | None = None  # rank of Δ projection (auto = ceil(d_model/16))

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        if self.dt_rank is None:
            self.dt_rank = math.ceil(self.d_model / 16)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


# ---------------------------------------------------------------------------
# Mamba block (selective SSM)
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    def __init__(self, cfg: MambaConfig):
        super().__init__()
        self.cfg = cfg
        d_inner, d_state, dt_rank = cfg.d_inner, cfg.d_state, cfg.dt_rank

        self.in_proj  = nn.Linear(cfg.d_model, d_inner * 2, bias=False)
        self.conv1d   = nn.Conv1d(
            d_inner, d_inner, kernel_size=cfg.d_conv,
            groups=d_inner, padding=cfg.d_conv - 1, bias=True,
        )
        self.x_proj   = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj  = nn.Linear(dt_rank, d_inner, bias=True)

        # A is parameterised in log space (stable, negative real part).
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D     = nn.Parameter(torch.ones(d_inner))

        self.out_proj = nn.Linear(d_inner, cfg.d_model, bias=False)

    def forward(self, x):
        """x: (B, L, d_model) → (B, L, d_model)"""
        B, L, _ = x.shape
        d_inner = self.cfg.d_inner

        x_and_z = self.in_proj(x)                       # (B, L, 2*d_inner)
        x_in, z = x_and_z.chunk(2, dim=-1)              # each (B, L, d_inner)

        # Causal depthwise conv over the sequence dim
        x_conv = x_in.transpose(1, 2)                   # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :L]          # trim padding → (B, d_inner, L)
        x_conv = x_conv.transpose(1, 2)                 # (B, L, d_inner)
        x_conv = F.silu(x_conv)

        y = self._selective_scan(x_conv)                # (B, L, d_inner)
        y = y * F.silu(z)                               # gating
        return self.out_proj(y)

    def _selective_scan(self, u):
        """
        Selective scan — the core of Mamba.
        u: (B, L, d_inner)
        Returns: (B, L, d_inner)

        Discretised SSM:  h_t = exp(Δ_t · A) h_{t-1} + Δ_t · B_t · u_t
                          y_t = C_t · h_t + D · u_t
        Input-dependent Δ, B, C make it "selective".
        """
        B_sz, L, d_inner = u.shape
        d_state = self.cfg.d_state

        x_dbl = self.x_proj(u)                          # (B, L, dt_rank + 2*d_state)
        delta, B_mat, C_mat = torch.split(
            x_dbl, [self.cfg.dt_rank, d_state, d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))         # (B, L, d_inner), positive

        A = -torch.exp(self.A_log.float())              # (d_inner, d_state)

        # Discretise: dA = exp(Δ · A), dB·u = Δ · B · u
        # delta: (B,L,d_inner) ; A: (d_inner,d_state)
        deltaA = torch.exp(delta.unsqueeze(-1) * A)     # (B, L, d_inner, d_state)
        deltaB_u = (
            delta.unsqueeze(-1)                         # (B, L, d_inner, 1)
            * B_mat.unsqueeze(2)                        # (B, L, 1, d_state)
            * u.unsqueeze(-1)                           # (B, L, d_inner, 1)
        )                                               # (B, L, d_inner, d_state)

        # Sequential scan over L (pure PyTorch — works on CPU/MPS)
        h  = torch.zeros(B_sz, d_inner, d_state, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(L):
            h = deltaA[:, t] * h + deltaB_u[:, t]       # (B, d_inner, d_state)
            y_t = torch.einsum("bdn,bn->bd", h, C_mat[:, t])  # (B, d_inner)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)                      # (B, L, d_inner)
        y = y + u * self.D                              # skip connection
        return y


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class ResidualMambaLayer(nn.Module):
    def __init__(self, cfg: MambaConfig):
        super().__init__()
        self.norm  = RMSNorm(cfg.d_model)
        self.mixer = MambaBlock(cfg)

    def forward(self, x):
        return x + self.mixer(self.norm(x))


class MinimalMamba(nn.Module):
    """
    Tiny Mamba language model — drop-in replacement for the Falcon-Mamba
    TextEncoder when mamba-ssm/CUDA is unavailable.

    Use `.encode(token_ids)` to get a pooled latent vector for SynapseOS,
    or call forward for next-token logits (training).
    """

    def __init__(self, cfg: MambaConfig | None = None):
        super().__init__()
        self.cfg   = cfg or MambaConfig()
        self.embed = nn.Embedding(self.cfg.vocab_size, self.cfg.d_model)
        self.layers = nn.ModuleList([ResidualMambaLayer(self.cfg) for _ in range(self.cfg.n_layers)])
        self.norm_f = RMSNorm(self.cfg.d_model)
        self.lm_head = nn.Linear(self.cfg.d_model, self.cfg.vocab_size, bias=False)
        # weight tying
        self.lm_head.weight = self.embed.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: (B, L) long → logits (B, L, vocab)"""
        x = self.embed(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode a sequence into a single pooled latent (mean over positions).
        token_ids: (B, L) → (B, d_model). Used as the SynapseOS TextEncoder.
        """
        x = self.embed(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return x.mean(dim=1)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Byte tokenizer (no external deps)
# ---------------------------------------------------------------------------

def bytes_encode(text: str, max_len: int = 256) -> torch.Tensor:
    ids = list(text.encode("utf-8"))[:max_len]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


# ---------------------------------------------------------------------------
# Smoke test + train demo
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    device = get_device(prefer_small=True)   # tiny model → CPU is fine
    cfg    = MambaConfig(vocab_size=256, d_model=64, n_layers=2, d_state=8)
    model  = MinimalMamba(cfg).to(device)
    print(f"[Mamba] params: {model.num_params():,}  device: {device}")

    x = bytes_encode("Sales Invoice validation rules", max_len=32).to(device)
    t0 = time.time()
    logits = model(x)
    latent = model.encode(x)
    fwd_ms = (time.time() - t0) * 1000

    loss = F.cross_entropy(logits[:, :-1].reshape(-1, 256), x[:, 1:].reshape(-1))
    loss.backward()

    print(f"[Mamba] forward {fwd_ms:.0f}ms | logits {tuple(logits.shape)} | latent {tuple(latent.shape)}")
    print(f"[Mamba] loss {loss.item():.3f} | backward OK")
    assert logits.shape == (1, x.shape[1], 256)
    assert latent.shape == (1, cfg.d_model)
    print("[Mamba] smoke test PASSED")


def _train_demo(steps: int = 60) -> None:
    """Train next-byte prediction on a tiny repeated corpus — loss should drop."""
    device = get_device(prefer_small=True)
    cfg    = MambaConfig(vocab_size=256, d_model=96, n_layers=3, d_state=12)
    model  = MinimalMamba(cfg).to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=3e-3)

    corpus = (
        "def validate(doc, method=None):\n"
        "    if not doc.customer:\n"
        "        frappe.throw('Customer is required')\n"
    ) * 4
    data = bytes_encode(corpus, max_len=200).to(device)

    print(f"[Mamba] train demo: {model.num_params():,} params on {device}, {steps} steps")
    t0 = time.time()
    first_loss = None
    for step in range(steps):
        logits = model(data)
        loss   = F.cross_entropy(logits[:, :-1].reshape(-1, 256), data[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if first_loss is None:
            first_loss = loss.item()
        if (step + 1) % 20 == 0:
            print(f"  step {step+1:>3}  loss {loss.item():.3f}")
    elapsed = time.time() - t0
    print(f"[Mamba] loss {first_loss:.3f} → {loss.item():.3f} in {elapsed:.1f}s "
          f"({steps/elapsed:.1f} steps/s)")
    assert loss.item() < first_loss, "Loss should decrease"
    print("[Mamba] train demo PASSED — model is learning")


def main():
    parser = argparse.ArgumentParser(description="Minimal Mamba (pure PyTorch)")
    parser.add_argument("--train-demo", action="store_true")
    parser.add_argument("--steps", type=int, default=60)
    args = parser.parse_args()

    _smoke_test()
    if args.train_demo:
        print()
        _train_demo(args.steps)


if __name__ == "__main__":
    main()
