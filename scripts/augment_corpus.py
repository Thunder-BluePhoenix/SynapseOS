#!/usr/bin/env python3
"""
Phase 1 — Step 3b: Augment corpus with external Frappe/ERPNext sources.

The local bench gives ~7M tokens. Target is 500M for Mamba LoRA fine-tuning.
This script pulls additional sources:
  1. Frappe + ERPNext full GitHub repos (source code + docs)
  2. Frappe community forum export (if available)
  3. India Compliance, HRMS, Payments repos

Run from the SynapseOS project root:
    python scripts/augment_corpus.py

Outputs appended to:
    data/corpus/corpus_all.txt
    data/corpus/augment_stats.json
"""

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from synapseos.phase1.corpus_builder import CorpusBuilder

OUTPUT_DIR = Path("data/corpus")
EXTERNAL_DIR = Path("data/external_repos")

# Public Frappe ecosystem repos to clone
EXTERNAL_REPOS = [
    # Core framework + ERP
    ("https://github.com/frappe/frappe.git",            "frappe_full"),
    ("https://github.com/frappe/erpnext.git",           "erpnext_full"),
    # Payroll, HR
    ("https://github.com/frappe/hrms.git",              "frappe_hrms"),
    # Payments
    ("https://github.com/frappe/payments.git",          "frappe_payments"),
    # India compliance (full upstream)
    ("https://github.com/resilient-tech/india-compliance.git", "india_compliance_full"),
    # Frappe docs
    ("https://github.com/frappe/frappe_docs.git",       "frappe_docs"),
    # ERPNext user guide (markdown)
    ("https://github.com/frappe/erpnext_documentation.git", "erpnext_docs"),
]


def clone_or_update(url: str, dest: Path) -> bool:
    if dest.exists():
        print(f"  Updating  {dest.name} ...")
        result = subprocess.run(
            ["git", "-C", str(dest), "pull", "--quiet"],
            capture_output=True, text=True
        )
        return result.returncode == 0
    else:
        print(f"  Cloning   {url} → {dest.name} ...")
        result = subprocess.run(
            ["git", "clone", "--depth=1", "--quiet", url, str(dest)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"    FAILED: {result.stderr.strip()}")
            return False
        return True


def build_external_corpus(repo_path: Path, app_name: str) -> dict[str, int]:
    builder = CorpusBuilder(
        apps_dir=str(repo_path.parent),
        output_dir=str(OUTPUT_DIR / "external"),
        app_filter=[repo_path.name],
    )
    builder.build()
    return builder.stats


def merge_into_main_corpus() -> None:
    ext_dir = OUTPUT_DIR / "external"
    if not ext_dir.exists():
        return

    main_corpus = OUTPUT_DIR / "corpus_all.txt"
    with open(main_corpus, "a") as merged:
        for txt_file in sorted(ext_dir.glob("*.txt")):
            if txt_file.name == "corpus_all.txt":
                continue
            size = txt_file.stat().st_size
            if size < 1000:
                continue
            merged.write(f"\n\n### EXTERNAL: {txt_file.stem} ###\n\n")
            merged.write(txt_file.read_text(encoding="utf-8", errors="ignore"))
            print(f"  Merged {txt_file.name} ({size / 1_000_000:.1f} MB)")


def estimate_tokens(path: Path) -> float:
    if not path.exists():
        return 0.0
    return path.stat().st_size / 4 / 1_000_000  # ~4 bytes per token


def main():
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "external").mkdir(parents=True, exist_ok=True)

    print("=== Phase 1 — Corpus Augmentation ===\n")

    # Check current corpus size
    current_tokens = estimate_tokens(OUTPUT_DIR / "corpus_all.txt")
    print(f"Current corpus: ~{current_tokens:.1f}M tokens")
    print(f"Target:         ~500M tokens\n")

    print("Cloning external repos...")
    success_count = 0
    for url, name in EXTERNAL_REPOS:
        dest = EXTERNAL_DIR / name
        ok = clone_or_update(url, dest)
        if ok:
            success_count += 1

    print(f"\nCloned/updated {success_count}/{len(EXTERNAL_REPOS)} repos")

    print("\nBuilding corpus from external repos...")
    for url, name in EXTERNAL_REPOS:
        repo_path = EXTERNAL_DIR / name
        if not repo_path.exists():
            continue
        print(f"\n  Processing {name}...")
        # Treat as a standalone app directory
        builder = CorpusBuilder(
            apps_dir=str(EXTERNAL_DIR),
            output_dir=str(OUTPUT_DIR / "external"),
            app_filter=[name],
        )
        builder.build()

    print("\nMerging external corpus into corpus_all.txt...")
    merge_into_main_corpus()

    # Final stats
    final_tokens = estimate_tokens(OUTPUT_DIR / "corpus_all.txt")
    stats = {
        "before_augmentation_tokens_m": round(current_tokens, 2),
        "after_augmentation_tokens_m": round(final_tokens, 2),
        "target_tokens_m": 500,
        "gap_tokens_m": round(max(0, 500 - final_tokens), 2),
    }

    with open(OUTPUT_DIR / "augment_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n=== Augmentation Complete ===")
    print(f"  Before : ~{current_tokens:.1f}M tokens")
    print(f"  After  : ~{final_tokens:.1f}M tokens")
    print(f"  Target : ~500M tokens")

    if final_tokens < 500:
        gap = 500 - final_tokens
        print(f"\n  Still {gap:.0f}M tokens short. Additional sources to consider:")
        print("    - Frappe community forum export (discuss.frappe.io)")
        print("    - Python standard library + documentation")
        print("    - Stack Overflow Frappe/ERPNext Q&A dump")
        print("    - Internal company documentation / wiki export")
        print("    - GitHub Issues / PRs from frappe/erpnext repos")


if __name__ == "__main__":
    main()
