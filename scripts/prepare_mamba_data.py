"""
Phase 1 — Mamba LoRA Fine-Tuning Data Preparation.

Packages the domain corpus into HuggingFace-format training data ready for
Mamba LoRA fine-tuning when GPU is available. Runs fully on CPU.

What this does:
  1. Analyses corpus_all.txt (character counts, token estimates, quality stats)
  2. Tokenizes the corpus using the AutoTokenizer for Falcon Mamba (or fast fallback)
  3. Packs tokens into fixed-length windows (2048 tokens each)
  4. Saves to data/mamba_train/ as:
       - train.jsonl       — training split (90%)
       - val.jsonl         — validation split (10%)
       - corpus_stats.json — full statistics for training plan
  5. Prints a GPU session checklist

Tokenizer:
  - Prefers: tiiuae/falcon-mamba-7b (requires `transformers` and internet access)
  - Fallback: simple whitespace tokenizer (estimates only, still produces valid windows)

Run:
    python scripts/prepare_mamba_data.py
    python scripts/prepare_mamba_data.py --corpus data/corpus/corpus_all.txt
    python scripts/prepare_mamba_data.py --dry-run   # stats only, no output files
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Corpus analysis
# ---------------------------------------------------------------------------

@dataclass
class CorpusStats:
    total_chars: int
    total_lines: int
    non_empty_lines: int
    estimated_tokens: int       # chars / 4 (rough heuristic)
    python_lines: int
    doctype_lines: int
    doc_lines: int
    unique_words: int
    avg_line_length: float
    quality_score: float        # 0–1: fraction of non-trivial lines
    token_target: int           # how many tokens needed for good Mamba fine-tune
    coverage_pct: float         # estimated_tokens / token_target × 100


def analyse_corpus(corpus_path: str) -> CorpusStats:
    text = Path(corpus_path).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    non_empty = [l for l in lines if l.strip()]
    python_lines   = sum(1 for l in non_empty if re.match(r'\s*(def |class |import |from |@)', l))
    doctype_lines  = sum(1 for l in non_empty if re.search(r'(fieldname|fieldtype|doctype|DocType)', l, re.I))
    doc_lines      = sum(1 for l in non_empty if len(l.split()) > 5 and not l.strip().startswith('#'))

    # Quality: lines with meaningful content (>10 chars, not just whitespace/comments)
    quality_lines  = sum(1 for l in non_empty if len(l.strip()) > 10 and not l.strip().startswith('#'))
    quality_score  = quality_lines / max(len(non_empty), 1)

    words = re.findall(r'\b[a-zA-Z_]\w*\b', text)
    unique_words   = len(set(w.lower() for w in words))
    estimated_tokens = len(text) // 4       # ~4 chars per token for code
    token_target   = 500_000_000            # 500M for Mamba LoRA
    coverage_pct   = min(100.0, estimated_tokens / token_target * 100)

    return CorpusStats(
        total_chars=len(text),
        total_lines=len(lines),
        non_empty_lines=len(non_empty),
        estimated_tokens=estimated_tokens,
        python_lines=python_lines,
        doctype_lines=doctype_lines,
        doc_lines=doc_lines,
        unique_words=unique_words,
        avg_line_length=round(sum(len(l) for l in non_empty) / max(len(non_empty), 1), 1),
        quality_score=round(quality_score, 3),
        token_target=token_target,
        coverage_pct=round(coverage_pct, 2),
    )


# ---------------------------------------------------------------------------
# Tokenizer (with graceful fallback)
# ---------------------------------------------------------------------------

class TokenizerWrapper:
    """Wraps HuggingFace tokenizer with a whitespace-split fallback."""

    def __init__(self, model_name: str = "tiiuae/falcon-mamba-7b", cache_dir: str | None = None):
        self._hf  = None
        self.name = "whitespace_fallback"
        self._try_load(model_name, cache_dir)

    def _try_load(self, model_name: str, cache_dir: str | None) -> None:
        try:
            from transformers import AutoTokenizer
            print(f"[Tokenizer] Loading {model_name} ...")
            self._hf = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
            self.name = model_name
            print(f"[Tokenizer] Loaded: vocab_size={self._hf.vocab_size}")
        except ImportError:
            print("[Tokenizer] `transformers` not installed — using whitespace fallback")
        except Exception as e:
            print(f"[Tokenizer] Load failed ({e}) — using whitespace fallback")

    def encode(self, text: str) -> list[int]:
        if self._hf:
            return self._hf.encode(text, add_special_tokens=False)
        # Fallback: character-level bigram hash (consistent, no model needed)
        tokens = []
        for i in range(0, len(text), 2):
            chunk = text[i:i+2]
            tokens.append(hash(chunk) % 50000)
        return tokens

    def decode(self, ids: list[int]) -> str:
        if self._hf:
            return self._hf.decode(ids)
        return f"[{len(ids)} tokens]"

    @property
    def is_real(self) -> bool:
        return self._hf is not None

    @property
    def eos_token_id(self) -> int:
        if self._hf and self._hf.eos_token_id is not None:
            return self._hf.eos_token_id
        return 0


# ---------------------------------------------------------------------------
# Windowing and dataset creation
# ---------------------------------------------------------------------------

def pack_windows(
    tokens: list[int],
    window_size: int = 2048,
    eos_id: int = 0,
    stride: int | None = None,
) -> list[list[int]]:
    """
    Pack tokens into fixed-length windows.
    Uses sliding window with stride = window_size (no overlap by default).
    Pads last window with eos_token_id if short.
    """
    stride   = stride or window_size
    windows  = []
    for start in range(0, len(tokens), stride):
        window = tokens[start:start + window_size]
        if len(window) < window_size:
            window = window + [eos_id] * (window_size - len(window))
        windows.append(window)
    return windows


def tokenize_corpus(
    corpus_path: str,
    tokenizer: TokenizerWrapper,
    chunk_chars: int = 100_000,
) -> list[int]:
    """
    Tokenize a large corpus file in chunks to avoid OOM.
    Returns flat list of token IDs.
    """
    text = Path(corpus_path).read_text(encoding="utf-8", errors="replace")
    all_tokens: list[int] = []
    total = len(text)
    n_chunks = math.ceil(total / chunk_chars)
    print(f"[Tokenize] {total:,} chars → {n_chunks} chunks of {chunk_chars:,} chars")

    t0 = time.time()
    for i in range(n_chunks):
        chunk = text[i * chunk_chars:(i + 1) * chunk_chars]
        ids   = tokenizer.encode(chunk)
        all_tokens.extend(ids)
        if (i + 1) % 10 == 0 or i == n_chunks - 1:
            elapsed = time.time() - t0
            rate    = (i + 1) * chunk_chars / elapsed / 1000
            print(f"  chunk {i+1}/{n_chunks}  tokens_so_far={len(all_tokens):,}  {rate:.0f}k chars/s")

    return all_tokens


def write_jsonl(windows: list[list[int]], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for win in windows:
            f.write(json.dumps({"input_ids": win}) + "\n")


# ---------------------------------------------------------------------------
# GPU session checklist
# ---------------------------------------------------------------------------

def print_gpu_checklist(stats: CorpusStats, n_windows: int, val_windows: int) -> None:
    multiplier = math.ceil(stats.token_target / max(stats.estimated_tokens, 1))
    print(f"""
{'='*64}
  GPU SESSION CHECKLIST — Mamba LoRA Fine-Tuning
{'='*64}

  Corpus:
    Available tokens : {stats.estimated_tokens:>12,}  ({stats.coverage_pct:.1f}% of 500M target)
    Training windows : {n_windows:>12,}  (2048 tokens each)
    Validation windows: {val_windows:>11,}

  To reach 500M tokens → repeat corpus ~{multiplier}× during training
  (set `num_train_epochs = {min(multiplier, 10)}` in fine-tune config)

  Install commands (run on GPU machine):
    pip install mamba-ssm peft transformers datasets accelerate bitsandbytes
    pip install trl   # for output synthesizer SFT later

  Fine-tuning command:
    python -m synapseos.phase2.mamba_trainer \\
      --data    data/mamba_train/train.jsonl \\
      --val     data/mamba_train/val.jsonl \\
      --epochs  {min(multiplier, 5)} \\
      --lr      2e-4 \\
      --lora-r  16

  Estimated GPU time:
    A100 40GB × 1 : ~{max(1, n_windows // 5000)}h  (500M tokens, r=16)
    A100 80GB × 2 : ~{max(1, n_windows // 10000)}h  (parallel)
    H100 × 4      : ~{max(1, n_windows // 20000)}h  (optimal)

  After fine-tuning:
    Checkpoint → data/checkpoints/mamba/
    Next: run synapseos.phase2.liquid_trainer on GPU for ODE adaptation
{'='*64}
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mamba Training Data Preparation")
    parser.add_argument("--corpus",      default="data/corpus/corpus_all.txt")
    parser.add_argument("--output-dir",  default="data/mamba_train")
    parser.add_argument("--window-size", type=int, default=2048)
    parser.add_argument("--val-split",   type=float, default=0.10)
    parser.add_argument("--model",       default="tiiuae/falcon-mamba-7b",
                        help="HuggingFace tokenizer model (requires internet + transformers)")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Analyse corpus and print checklist without writing files")
    parser.add_argument("--max-chars",   type=int, default=None,
                        help="Limit corpus to first N chars (for testing)")
    args = parser.parse_args()

    corpus_path = args.corpus
    if not Path(corpus_path).exists():
        print(f"[Error] Corpus not found: {corpus_path}")
        return

    # 1. Analyse corpus
    print(f"\n{'='*64}")
    print(f"  Mamba Training Data Preparation")
    print(f"{'='*64}")
    print(f"  Corpus: {corpus_path}  ({Path(corpus_path).stat().st_size // 1024}KB)")

    print("\n[1/4] Analysing corpus ...")
    stats = analyse_corpus(corpus_path)
    print(f"  Total chars     : {stats.total_chars:,}")
    print(f"  Lines           : {stats.total_lines:,}  ({stats.non_empty_lines:,} non-empty)")
    print(f"  Estimated tokens: {stats.estimated_tokens:,}  ({stats.coverage_pct:.1f}% of 500M target)")
    print(f"  Python lines    : {stats.python_lines:,}")
    print(f"  DocType lines   : {stats.doctype_lines:,}")
    print(f"  Unique words    : {stats.unique_words:,}")
    print(f"  Quality score   : {stats.quality_score:.3f}")

    if args.dry_run:
        print("\n[Dry run] Skipping tokenization and file output.")
        print_gpu_checklist(stats, 0, 0)
        json.dump(asdict(stats), open(f"{args.output_dir}/corpus_analysis.json", "w"), indent=2) if not args.dry_run else None
        return

    # 2. Tokenize
    print("\n[2/4] Tokenizing ...")
    tokenizer = TokenizerWrapper(model_name=args.model)

    corpus_text = Path(corpus_path).read_text(encoding="utf-8", errors="replace")
    if args.max_chars:
        corpus_text = corpus_text[:args.max_chars]
        print(f"  [Limited to {args.max_chars:,} chars for testing]")
    tmp_path = "/tmp/synapseos_corpus_prep.txt"
    Path(tmp_path).write_text(corpus_text)

    all_tokens = tokenize_corpus(tmp_path, tokenizer, chunk_chars=100_000)
    print(f"  Total tokens: {len(all_tokens):,}")
    print(f"  Using tokenizer: {tokenizer.name}")

    # 3. Pack windows
    print(f"\n[3/4] Packing {len(all_tokens):,} tokens into {args.window_size}-token windows ...")
    all_windows = pack_windows(all_tokens, args.window_size, tokenizer.eos_token_id)

    n_val   = max(1, int(len(all_windows) * args.val_split))
    n_train = len(all_windows) - n_val
    train_windows = all_windows[:n_train]
    val_windows   = all_windows[n_train:]

    print(f"  Total windows : {len(all_windows):,}")
    print(f"  Train windows : {n_train:,}  ({n_train * args.window_size:,} tokens)")
    print(f"  Val   windows : {n_val:,}   ({n_val * args.window_size:,} tokens)")

    # 4. Write output
    print(f"\n[4/4] Writing to {args.output_dir}/ ...")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    write_jsonl(train_windows, f"{args.output_dir}/train.jsonl")
    write_jsonl(val_windows,   f"{args.output_dir}/val.jsonl")

    # Save stats
    full_stats = {
        **asdict(stats),
        "tokenizer":      tokenizer.name,
        "window_size":    args.window_size,
        "total_windows":  len(all_windows),
        "train_windows":  n_train,
        "val_windows":    n_val,
        "train_tokens":   n_train * args.window_size,
        "val_tokens":     n_val * args.window_size,
        "is_real_tokenizer": tokenizer.is_real,
    }
    stats_path = f"{args.output_dir}/corpus_stats.json"
    Path(stats_path).write_text(json.dumps(full_stats, indent=2))

    print(f"  train.jsonl   → {n_train} windows")
    print(f"  val.jsonl     → {n_val} windows")
    print(f"  corpus_stats  → {stats_path}")

    print_gpu_checklist(stats, n_train, n_val)


if __name__ == "__main__":
    main()
