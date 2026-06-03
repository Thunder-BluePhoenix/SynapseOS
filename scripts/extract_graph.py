#!/usr/bin/env python3
"""
Phase 1 — Step 1: Extract Frappe knowledge graph.

Run from the SynapseOS project root:
    python scripts/extract_graph.py

Outputs:
    data/graph/knowledge_graph.json   — NetworkX node-link format
    data/graph/graph_stats.json       — extraction stats
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from synapseos.phase1.graph_extractor import FrappeGraphExtractor

APPS_DIR = "/Users/bluephoenix/frappe-bench/bench-r2/apps"
OUTPUT_PATH = "data/graph/knowledge_graph.json"

# Set to None to scan all apps, or list specific ones:
# APP_FILTER = ["frappe", "erpnext", "f_icecore"]
APP_FILTER = None


def main():
    print(f"Extracting knowledge graph from: {APPS_DIR}")
    print(f"App filter: {APP_FILTER or 'all apps'}\n")

    extractor = FrappeGraphExtractor(apps_dir=APPS_DIR, app_filter=APP_FILTER)
    G = extractor.build()
    extractor.save(OUTPUT_PATH)

    # Print top connected nodes (most relationships)
    print("\n--- Top 20 Most Connected DocTypes ---")
    degree_sorted = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    for node, degree in degree_sorted[:20]:
        node_data = G.nodes[node]
        if node_data.get("type") == "doctype":
            print(f"  {node:40s}  degree={degree}  app={node_data.get('app', '?')}")


if __name__ == "__main__":
    main()
