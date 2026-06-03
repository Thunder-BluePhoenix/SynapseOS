"""
Phase 2 — Mamba Sequence Engine: LoRA fine-tuning on domain corpus.

Model  : tiiuae/falcon-mamba-7b  (or state-spaces/mamba-2.8b for lighter runs)
Method : LoRA (r=16) on x_proj + dt_proj — Parameter-Efficient Fine-Tuning
Loss   : Causal language modeling (next token prediction)
Data   : data/corpus/corpus_all.txt  (~7M local tokens + external after augmentation)

Install before running:
    pip install mamba-ssm peft transformers datasets accelerate

Run from project root:
    python -m synapseos.phase2.mamba_trainer
    # or with config overrides:
    python -m synapseos.phase2.mamba_trainer --model state-spaces/mamba-2.8b --epochs 3
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    # Model
    model_name: str = "tiiuae/falcon-mamba-7b"
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_target_modules: list[str] = None  # set in __post_init__
    lora_dropout: float = 0.05
    # Training
    corpus_path: str = "data/corpus/corpus_all.txt"
    output_dir: str = "data/checkpoints/mamba"
    num_epochs: int = 2
    batch_size: int = 4
    grad_accum_steps: int = 8        # effective batch = 32
    learning_rate: float = 2e-4
    max_seq_length: int = 4096
    warmup_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 50
    fp16: bool = True
    gradient_checkpointing: bool = True
    # Data
    val_split: float = 0.02          # 2% held out for validation

    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["x_proj", "dt_proj", "out_proj"]

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_file(cls, path: str) -> TrainingConfig:
        with open(path) as f:
            return cls(**json.load(f))


def check_dependencies() -> bool:
    missing = []
    for pkg in ["mamba_ssm", "peft", "transformers", "datasets", "accelerate"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print("Missing dependencies. Install with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    return True


def build_dataset(corpus_path: str, tokenizer, max_seq_length: int, val_split: float):
    from datasets import Dataset
    import torch

    print(f"Loading corpus from {corpus_path}...")
    text = Path(corpus_path).read_text(encoding="utf-8", errors="ignore")
    print(f"Corpus size: {len(text) / 1_000_000:.1f}M chars")

    print("Tokenizing...")
    tokens = tokenizer.encode(text)
    print(f"Total tokens: {len(tokens) / 1_000_000:.2f}M")

    # Chunk into fixed-length sequences
    chunks = []
    for i in range(0, len(tokens) - max_seq_length, max_seq_length):
        chunk = tokens[i : i + max_seq_length]
        chunks.append({"input_ids": chunk, "labels": chunk})

    print(f"Sequences: {len(chunks)}")

    split_idx = int(len(chunks) * (1 - val_split))
    train_chunks = chunks[:split_idx]
    val_chunks = chunks[split_idx:]

    def to_tensors(batch):
        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "labels":    torch.tensor(batch["labels"],    dtype=torch.long),
        }

    train_ds = Dataset.from_list(train_chunks)
    val_ds   = Dataset.from_list(val_chunks)
    return train_ds, val_ds


def train(config: TrainingConfig) -> None:
    if not check_dependencies():
        return

    from mamba_ssm import MambaLMHeadModel
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoTokenizer, TrainingArguments, Trainer

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    config.save(f"{config.output_dir}/training_config.json")

    # Load tokenizer
    print(f"\nLoading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model: {config.model_name}")
    model = MambaLMHeadModel.from_pretrained(
        config.model_name,
        dtype="float16" if config.fp16 else "float32",
        device_map="auto",
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if config.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    # Build dataset
    train_ds, val_ds = build_dataset(
        config.corpus_path, tokenizer, config.max_seq_length, config.val_split
    )

    # Training arguments
    args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        fp16=config.fp16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        evaluation_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    print("\nStarting training...")
    trainer.train()

    # Save final LoRA adapter
    adapter_path = f"{config.output_dir}/lora_adapter_final"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\nLoRA adapter saved → {adapter_path}")


def main():
    parser = argparse.ArgumentParser(description="Mamba LoRA fine-tuning")
    parser.add_argument("--model", default="tiiuae/falcon-mamba-7b")
    parser.add_argument("--corpus", default="data/corpus/corpus_all.txt")
    parser.add_argument("--output", default="data/checkpoints/mamba")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    args = parser.parse_args()

    config = TrainingConfig(
        model_name=args.model,
        corpus_path=args.corpus,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_len,
        lora_r=args.lora_r,
    )

    print("=== Mamba LoRA Training Config ===")
    for k, v in asdict(config).items():
        print(f"  {k}: {v}")

    if args.dry_run:
        print("\nDry run — exiting.")
        return

    train(config)


if __name__ == "__main__":
    main()
