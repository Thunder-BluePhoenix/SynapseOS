"""
Local SFT Trainer — pure-PyTorch supervised fine-tuning, no `trl`.

Replaces the `trl` (Transformer Reinforcement Learning) library's SFTTrainer
for the output synthesizer. trl wraps HuggingFace + accelerate + a big model;
this trains a small from-scratch byte-level decoder transformer on the
(instruction, output) pairs produced by `synapseos/phase3/output_synth_data.py`.

It runs on CPU/MPS with no installs. The model is intentionally small
(a few M params) — SynapseOS does the hard reasoning upstream in the
neurosymbolic core, so the synthesizer only needs to render structured
facts into fluent text.

Pipeline:
  1. Load SFT pairs (Alpaca JSONL: instruction/input/output)
  2. Format as  "### Instruction:\n{q}\n\n### Response:\n{a}<eos>"
  3. Byte-tokenize, pack to fixed length, mask the prompt in the loss
  4. Train a small causal transformer with next-token cross-entropy
  5. Save checkpoint; provide .generate() for inference

Run:
    python -m synapseos.local.sft_trainer --demo                  # synthetic 50-pair demo
    python -m synapseos.local.sft_trainer --data data/sft/sft_pairs.jsonl --steps 200
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from synapseos.local.device import get_device


# ---------------------------------------------------------------------------
# Tiny causal transformer
# ---------------------------------------------------------------------------

@dataclass
class SFTConfig:
    vocab_size: int = 256        # byte-level
    d_model:    int = 192
    n_heads:    int = 6
    n_layers:   int = 4
    max_len:    int = 320
    dropout:    float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: SFTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.d_head  = cfg.d_model // cfg.n_heads
        self.qkv     = nn.Linear(cfg.d_model, cfg.d_model * 3)
        self.proj    = nn.Linear(cfg.d_model, cfg.d_model)
        self.drop    = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.d_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)          # each (B, n_heads, L, d_head)
        # scaled dot-product attention with causal mask
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).reshape(B, L, D)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, cfg: SFTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 4),
            nn.GELU(),
            nn.Linear(cfg.d_model * 4, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyDecoder(nn.Module):
    def __init__(self, cfg: SFTConfig):
        super().__init__()
        self.cfg   = cfg
        self.tok   = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos   = nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f  = nn.LayerNorm(cfg.d_model)
        self.head  = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, idx):
        B, L = idx.shape
        pos = torch.arange(L, device=idx.device).unsqueeze(0)
        x = self.tok(idx) + self.pos(pos)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new: int = 120,
                 temperature: float = 0.8, eos_id: int = 0) -> torch.Tensor:
        self.eval()
        idx = prompt_ids
        for _ in range(max_new):
            idx_cond = idx[:, -self.cfg.max_len:]
            logits = self(idx_cond)[:, -1, :] / max(temperature, 1e-5)
            probs  = F.softmax(logits, dim=-1)
            nxt    = torch.multinomial(probs, 1)
            idx    = torch.cat([idx, nxt], dim=1)
            if nxt.item() == eos_id:
                break
        return idx


# ---------------------------------------------------------------------------
# Data formatting
# ---------------------------------------------------------------------------

EOS = 0   # use byte 0 as end-of-sequence

def format_pair(instruction: str, output: str) -> str:
    return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"


def encode_example(instruction: str, output: str, max_len: int):
    """
    Returns (input_ids, target_ids) with the prompt portion masked (-100) in targets.
    """
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    full   = prompt + output

    p_bytes = list(prompt.encode("utf-8"))
    f_bytes = list(full.encode("utf-8"))[:max_len - 1] + [EOS]

    ids = torch.tensor(f_bytes, dtype=torch.long)
    tgt = ids.clone()
    # mask the prompt tokens so loss is only on the response
    mask_len = min(len(p_bytes), len(tgt))
    tgt[:mask_len] = -100
    return ids, tgt


def collate(batch, max_len: int, pad_id: int = EOS):
    """Pad a batch of (ids, tgt) to the longest sequence."""
    maxlen = min(max(len(ids) for ids, _ in batch), max_len)
    input_ids, targets = [], []
    for ids, tgt in batch:
        ids = ids[:maxlen]; tgt = tgt[:maxlen]
        pad = maxlen - len(ids)
        input_ids.append(torch.cat([ids, torch.full((pad,), pad_id)]))
        targets.append(torch.cat([tgt, torch.full((pad,), -100)]))
    return torch.stack(input_ids), torch.stack(targets)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class SFTTrainer:
    """
    Pure-PyTorch supervised fine-tuning for the output synthesizer.
    Drop-in alternative to trl.SFTTrainer that runs on CPU/MPS.
    """

    def __init__(self, cfg: SFTConfig | None = None, device: torch.device | None = None):
        self.cfg    = cfg or SFTConfig()
        self.device = device or get_device(prefer_small=False)   # MPS helps here
        self.model  = TinyDecoder(self.cfg).to(self.device)
        self.opt    = torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=0.01)

    def load_pairs(self, jsonl_path: str, limit: int | None = None) -> list[tuple[str, str]]:
        pairs = []
        with open(jsonl_path) as f:
            for line in f:
                rec = json.loads(line)
                instr = rec.get("instruction", "")
                out   = rec.get("output", "")
                if instr and out:
                    pairs.append((instr, out))
                if limit and len(pairs) >= limit:
                    break
        return pairs

    def train(self, pairs: list[tuple[str, str]], steps: int = 200,
              batch_size: int = 8, log_every: int = 50) -> dict:
        encoded = [encode_example(i, o, self.cfg.max_len) for i, o in pairs]
        self.model.train()

        import random
        rng = random.Random(42)
        t0  = time.time()
        first_loss = None
        losses = []

        for step in range(steps):
            batch = rng.sample(encoded, min(batch_size, len(encoded)))
            input_ids, targets = collate(batch, self.cfg.max_len)
            input_ids = input_ids.to(self.device)
            targets   = targets.to(self.device)

            logits = self.model(input_ids)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, self.cfg.vocab_size),
                targets[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            self.opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()

            losses.append(loss.item())
            if first_loss is None:
                first_loss = loss.item()
            if (step + 1) % log_every == 0:
                avg = sum(losses[-log_every:]) / len(losses[-log_every:])
                print(f"  step {step+1:>4}  loss {loss.item():.3f}  (avg {avg:.3f})")

        elapsed = time.time() - t0
        return {
            "first_loss":  round(first_loss, 4),
            "final_loss":  round(loss.item(), 4),
            "steps":       steps,
            "elapsed_s":   round(elapsed, 1),
            "steps_per_s": round(steps / elapsed, 2),
            "device":      str(self.device),
            "params":      self.model.num_params(),
        }

    @torch.no_grad()
    def synthesize(self, instruction: str, max_new: int = 150, temperature: float = 0.7) -> str:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        ids    = torch.tensor([list(prompt.encode("utf-8"))], dtype=torch.long, device=self.device)
        out    = self.model.generate(ids, max_new=max_new, temperature=temperature, eos_id=EOS)
        gen_bytes = out[0].tolist()[len(ids[0]):]
        # stop at EOS
        if EOS in gen_bytes:
            gen_bytes = gen_bytes[:gen_bytes.index(EOS)]
        try:
            return bytes(b for b in gen_bytes if b != EOS).decode("utf-8", errors="replace")
        except Exception:
            return "[decode error]"

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict(), "cfg": self.cfg.__dict__}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    """Train on a tiny synthetic set so loss visibly drops — proves it works."""
    pairs = [
        ("What fields are required for a Sales Invoice?",
         "A Sales Invoice requires Customer, Posting Date, and at least one Item."),
        ("What fields are required for a Purchase Order?",
         "A Purchase Order requires Supplier, Schedule Date, and at least one Item."),
        ("How do I submit a Journal Entry?",
         "Fill all accounts so debit equals credit, then click Submit to post to the ledger."),
    ] * 12   # repeat for a small but learnable set

    cfg = SFTConfig(d_model=128, n_heads=4, n_layers=3, max_len=256)
    trainer = SFTTrainer(cfg)
    print(f"[SFT] model: {trainer.model.num_params():,} params on {trainer.device}")

    stats = trainer.train(pairs, steps=150, batch_size=6, log_every=50)
    print(f"[SFT] loss {stats['first_loss']} → {stats['final_loss']} "
          f"in {stats['elapsed_s']}s ({stats['steps_per_s']} steps/s on {stats['device']})")
    assert stats["final_loss"] < stats["first_loss"], "Loss should decrease"

    print("\n[SFT] sample generation after training:")
    out = trainer.synthesize("What fields are required for a Sales Invoice?", max_new=80)
    print(f"  → {out[:160]}")
    print("\n[SFT] demo PASSED — synthesizer trains and generates")


def main():
    parser = argparse.ArgumentParser(description="Local SFT Trainer (replaces trl)")
    parser.add_argument("--demo",   action="store_true")
    parser.add_argument("--data",   default=None, help="Alpaca JSONL from output_synth_data.py")
    parser.add_argument("--steps",  type=int, default=200)
    parser.add_argument("--limit",  type=int, default=500, help="Max pairs to load")
    parser.add_argument("--save",   default="data/checkpoints/synth/local_sft.pt")
    args = parser.parse_args()

    if args.demo or not args.data:
        _demo()
        return

    trainer = SFTTrainer()
    print(f"[SFT] loading pairs from {args.data} ...")
    pairs = trainer.load_pairs(args.data, limit=args.limit)
    print(f"[SFT] {len(pairs)} pairs | model {trainer.model.num_params():,} params on {trainer.device}")

    stats = trainer.train(pairs, steps=args.steps, batch_size=8)
    print(f"[SFT] loss {stats['first_loss']} → {stats['final_loss']} in {stats['elapsed_s']}s")

    trainer.save(args.save)
    print(f"[SFT] checkpoint → {args.save}")

    print("\n[SFT] sample:")
    print(trainer.synthesize("What fields are required for a Sales Invoice?")[:200])


if __name__ == "__main__":
    main()
