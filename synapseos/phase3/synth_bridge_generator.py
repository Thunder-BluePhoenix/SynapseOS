"""
Phase 3 — Synthetic bridge training data generator.

Generates training pairs for both bridges using the knowledge graph from Phase 1
and the JEPA encoder from Phase 2 (or random vectors if JEPA isn't trained yet).

Outputs:
    data/bridges/n2s_pairs.json   — (latent_vector, query_type, doctype) pairs
    data/bridges/s2n_pairs.json   — (symbolic_feature_vector, target_embed) pairs

Run:
    python -m synapseos.phase3.synth_bridge_generator
    python -m synapseos.phase3.synth_bridge_generator --samples 10000
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch


random.seed(42)
torch.manual_seed(42)

QUERY_TYPES = ["retrieve", "reason", "plan", "compute", "validate"]

# Roughly which DocTypes map to which query pattern
QUERY_TYPE_HINTS = {
    "retrieve":  ["Customer", "Item", "Supplier", "Employee", "Account", "User",
                  "Contact", "Address", "Currency", "UOM"],
    "reason":    ["Sales Invoice", "Purchase Invoice", "Payment Entry",
                  "Journal Entry", "Stock Entry", "Payroll Entry"],
    "plan":      ["Sales Order", "Purchase Order", "Production Plan",
                  "Work Order", "Project", "Task", "Leave Application"],
    "compute":   ["Sales Invoice Item", "Purchase Invoice Item", "Stock Ledger Entry",
                  "GL Entry", "Salary Slip", "Expense Claim"],
    "validate":  ["Leave Policy", "Leave Allocation", "Attendance",
                  "Appraisal", "Asset", "Maintenance Schedule"],
}

# Event types
EVENTS = ["validate", "on_submit", "on_cancel", "before_submit",
          "before_cancel", "on_update", "before_save", "autoname", "render"]

RULE_TYPES = ["constraint", "conditional_required", "workflow_guard", "computed"]


def load_graph_doctypes() -> list[str]:
    graph_path = Path("data/graph/knowledge_graph.json")
    if graph_path.exists():
        try:
            data = json.loads(graph_path.read_text())
            doctypes = [
                n["id"] for n in data.get("nodes", [])
                if n.get("type") == "doctype" and not n["id"].startswith("__")
            ]
            print(f"Loaded {len(doctypes)} DocTypes from knowledge graph")
            return doctypes
        except Exception:
            pass
    # Fallback list
    return list({dt for dts in QUERY_TYPE_HINTS.values() for dt in dts})


def get_query_type_for_doctype(doctype: str) -> str:
    for qt, doctypes in QUERY_TYPE_HINTS.items():
        if doctype in doctypes:
            return qt
    # Hash-based deterministic assignment for unseen doctypes
    return QUERY_TYPES[hash(doctype) % len(QUERY_TYPES)]


def generate_n2s_pairs(n_samples: int, latent_dim: int, doctypes: list[str]) -> list[dict]:
    """
    For each query_type, simulate what a latent vector for that intent looks like.
    Each query type has a characteristic direction in latent space (synthetic clusters).
    """
    # Create a prototype vector per query type
    prototypes = {qt: torch.randn(latent_dim) for qt in QUERY_TYPES}

    pairs = []
    for _ in range(n_samples):
        doctype = random.choice(doctypes)
        qt = get_query_type_for_doctype(doctype)

        # Sample latent vector = prototype + noise (simulates JEPA output for this intent)
        proto = prototypes[qt]
        noise = torch.randn(latent_dim) * 0.3
        latent = (proto + noise).tolist()

        pairs.append({
            "latent":     latent,
            "query_type": qt,
            "doctype":    doctype,
        })

    return pairs


def encode_rule_result(
    rule_type: str,
    event: str,
    confidence: float,
    value: float,
    doctype_hash: float,
) -> list[float]:
    """
    Encode a symbolic rule result as a fixed-size feature vector.
    Dimension = 32:
      [0]     : confidence
      [1-4]   : rule_type one-hot (4 types)
      [5-13]  : event one-hot (9 events)
      [14]    : value scalar (normalized)
      [15]    : doctype_hash (normalized)
      [16-31] : small noise padding
    """
    vec = [0.0] * 32
    vec[0] = confidence

    if rule_type in RULE_TYPES:
        vec[1 + RULE_TYPES.index(rule_type)] = 1.0

    if event in EVENTS:
        vec[5 + EVENTS.index(event)] = 1.0

    vec[14] = max(-3, min(3, value)) / 3.0
    vec[15] = (doctype_hash % 1000) / 1000.0

    for i in range(16, 32):
        vec[i] = random.gauss(0, 0.05)

    return vec


def generate_s2n_pairs(n_samples: int, output_embed_dim: int) -> list[dict]:
    """
    For each symbolic rule result, generate a target embedding.
    The target is a structured vector in the same space the KAN layer consumes.
    Rule results cluster by type — similar rules should map to nearby embeddings.
    """
    # Prototype embeddings per rule type (what the KAN + output synthesizer expects)
    prototypes = {rt: torch.randn(output_embed_dim) for rt in RULE_TYPES}

    pairs = []
    for _ in range(n_samples):
        rule_type  = random.choice(RULE_TYPES)
        event      = random.choice(EVENTS)
        confidence = random.uniform(0.5, 1.0)
        value      = random.gauss(0, 2)
        dt_hash    = random.uniform(0, 1000)

        features = encode_rule_result(rule_type, event, confidence, value, dt_hash)

        # Target embedding = prototype for this rule type + small noise
        proto  = prototypes[rule_type]
        noise  = torch.randn(output_embed_dim) * 0.2
        target = (proto + noise).tolist()

        pairs.append({
            "symbolic_features": features,
            "target_embed":      target,
            "rule_type":         rule_type,
            "event":             event,
        })

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples",     type=int, default=10000)
    parser.add_argument("--latent-dim",  type=int, default=64)
    parser.add_argument("--embed-dim",   type=int, default=64)
    parser.add_argument("--output-dir",  default="data/bridges")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.samples} bridge pairs each...\n")

    doctypes = load_graph_doctypes()

    # Neural→Symbolic pairs
    n2s_pairs = generate_n2s_pairs(args.samples, args.latent_dim, doctypes)
    n2s_path = out / "n2s_pairs.json"
    with open(n2s_path, "w") as f:
        json.dump(n2s_pairs, f)
    qt_counts = {}
    for p in n2s_pairs:
        qt_counts[p["query_type"]] = qt_counts.get(p["query_type"], 0) + 1
    print(f"Neural→Symbolic: {len(n2s_pairs)} pairs → {n2s_path}")
    for qt, cnt in sorted(qt_counts.items()):
        print(f"  {qt:20s}: {cnt}")

    # Symbolic→Neural pairs
    s2n_pairs = generate_s2n_pairs(args.samples, args.embed_dim)
    s2n_path = out / "s2n_pairs.json"
    with open(s2n_path, "w") as f:
        json.dump(s2n_pairs, f)
    print(f"\nSymbolic→Neural: {len(s2n_pairs)} pairs → {s2n_path}")

    print("\nDone. Run bridge trainer:")
    print("  python -m synapseos.phase3.bridge_trainer")


if __name__ == "__main__":
    main()
