#!/usr/bin/env python3
"""
Phase 1 — Step 3: Build Mamba training corpus from Frappe apps.

Run from the SynapseOS project root:
    python scripts/build_corpus.py

Outputs:
    data/corpus/<app>_python.txt    — Python source files
    data/corpus/<app>_doctype.txt   — DocType JSON as structured text
    data/corpus/<app>_docs.txt      — Markdown/RST documentation
    data/corpus/corpus_stats.json   — size and token estimates
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from synapseos.phase1.corpus_builder import CorpusBuilder

APPS_DIR = "/Users/bluephoenix/frappe-bench/bench-r2/apps"
OUTPUT_DIR = "data/corpus"

# All apps — None means every app in the bench gets included.
# Custom apps (f_genbi, frappe_flowbridge, nextai, frappe_rpg, f_icecore, etc.)
# are automatically picked up since APP_FILTER = None.
APP_FILTER = None


def main():
    from pathlib import Path

    apps_path = Path(APPS_DIR)
    all_apps = sorted(p.name for p in apps_path.iterdir() if p.is_dir())
    print(f"Building corpus from: {APPS_DIR}")
    print(f"\nApps found ({len(all_apps)}):")
    for app in all_apps:
        print(f"  - {app}")
    print()

    builder = CorpusBuilder(apps_dir=APPS_DIR, output_dir=OUTPUT_DIR, app_filter=APP_FILTER)
    builder.build()


if __name__ == "__main__":
    main()
