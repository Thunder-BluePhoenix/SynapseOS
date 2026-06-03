#!/usr/bin/env python3
"""
Phase 1 — Step 2: Extract business rules from Frappe app Python controllers.

Run from the SynapseOS project root:
    python scripts/extract_rules.py

Outputs:
    data/rules/rules.json   — all extracted symbolic rules
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from synapseos.phase1.rules_extractor import RulesExtractor

APPS_DIR = "/Users/bluephoenix/frappe-bench/bench-r2/apps"
OUTPUT_PATH = "data/rules/rules.json"

APP_FILTER = None  # None = all apps


def main():
    print(f"Extracting business rules from: {APPS_DIR}\n")

    extractor = RulesExtractor(apps_dir=APPS_DIR, app_filter=APP_FILTER)
    result = extractor.extract()
    extractor.save(OUTPUT_PATH)

    print("\n--- Rule Breakdown by Type ---")
    for rule_type, count in result.stats.get("by_type", {}).items():
        print(f"  {rule_type:30s}  {count}")

    print("\n--- Rule Breakdown by Event ---")
    for event, count in result.stats.get("by_event", {}).items():
        print(f"  {event:30s}  {count}")

    print("\n--- Sample Rules ---")
    for rule in result.rules[:5]:
        print(f"  [{rule.rule_type}] {rule.doctype} / {rule.event}")
        print(f"    IF:   {rule.condition[:80]}")
        print(f"    THEN: {rule.consequence[:80]}")
        print()


if __name__ == "__main__":
    main()
