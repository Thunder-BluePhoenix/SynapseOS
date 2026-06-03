"""
Synthetic training data generator for Phase 2.

Produces realistic synthetic data for JEPA, KAN, and Liquid NN trainers
using the knowledge graph and business rules from Phase 1 as the schema.

Generates:
  data/jepa/transitions.json   — document state transition sequences
  data/kan/metric_pairs.json   — domain metric input/output pairs
  data/liquid/timeseries.json  — operational time-series

Run:
    python -m synapseos.phase2.synth_data_generator
    python -m synapseos.phase2.synth_data_generator --samples 10000
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


random.seed(42)


# ---------------------------------------------------------------------------
# JEPA — document state transition sequences
# ---------------------------------------------------------------------------

# Frappe document lifecycle states (common across ERPNext)
DOC_STATES = {
    "Sales Order":      ["Draft", "To Deliver and Bill", "To Bill", "To Deliver", "Completed", "Cancelled"],
    "Purchase Order":   ["Draft", "To Receive and Bill", "To Bill", "To Receive", "Completed", "Cancelled"],
    "Sales Invoice":    ["Draft", "Unpaid", "Overdue", "Paid", "Return", "Cancelled"],
    "Purchase Invoice": ["Draft", "Unpaid", "Overdue", "Paid", "Return", "Cancelled"],
    "Delivery Note":    ["Draft", "To Bill", "Completed", "Cancelled"],
    "Payment Entry":    ["Draft", "Submitted", "Cancelled"],
    "Stock Entry":      ["Draft", "Submitted", "Cancelled"],
    "Journal Entry":    ["Draft", "Submitted", "Cancelled"],
    "Leave Application": ["Open", "Approved", "Rejected", "Cancelled"],
    "Expense Claim":    ["Draft", "Submitted", "Approved", "Rejected", "Paid"],
}

# Feature vector: one-hot state + numeric metadata (amount, item_count, age_days, priority)
STATE_FEATURE_DIM = 128


def encode_state(
    doctype: str,
    state: str,
    amount: float,
    item_count: int,
    age_days: float,
    priority: float,
) -> list[float]:
    states = DOC_STATES.get(doctype, ["Draft", "Submitted", "Cancelled"])
    state_idx = states.index(state) if state in states else 0
    # One-hot for state (max 10 states) + 4 numeric features, padded to STATE_FEATURE_DIM
    vec = [0.0] * STATE_FEATURE_DIM
    vec[state_idx] = 1.0
    vec[10] = amount / 1_000_000       # normalize to millions
    vec[11] = item_count / 100.0
    vec[12] = age_days / 365.0
    vec[13] = priority
    # Add small noise
    for i in range(14, 30):
        vec[i] = random.gauss(0, 0.01)
    return vec


def generate_jepa_sequences(n_sequences: int) -> list[dict]:
    sequences = []
    doctypes = list(DOC_STATES.keys())

    for _ in range(n_sequences):
        doctype = random.choice(doctypes)
        states  = DOC_STATES[doctype]

        # Simulate a realistic document lifecycle
        amount     = random.lognormvariate(math.log(50_000), 1.2)
        item_count = random.randint(1, 50)
        priority   = random.random()

        # Walk through states (mostly forward, small chance of cancel)
        state_sequence = ["Draft"]
        age = 0.0
        for i in range(1, len(states)):
            age += random.uniform(0.5, 30)
            if states[i] == "Cancelled" and random.random() > 0.1:
                continue   # most docs don't get cancelled
            state_sequence.append(states[i])
            if states[i] in ("Completed", "Paid", "Cancelled", "Rejected"):
                break

        if len(state_sequence) < 3:
            continue

        # Expand: each real state gets 2-4 substeps with small noise (simulates event ticks)
        encoded = []
        expanded_labels = []
        for s in state_sequence:
            substeps = random.randint(2, 5)
            for _ in range(substeps):
                vec = encode_state(doctype, s, amount, item_count, age, priority)
                encoded.append(vec)
                expanded_labels.append(s)

        sequences.append({
            "doctype": doctype,
            "states": encoded,
            "state_labels": expanded_labels,
        })

    return sequences


# ---------------------------------------------------------------------------
# KAN — domain metric input/output pairs
# ---------------------------------------------------------------------------

# Synthetic relationships derived from common ERPNext business logic
# These mirror real relationships KAN should discover

def _approval_time(approver_count: float, amount: float, urgency: float, backlog: float) -> float:
    """Synthetic: approval_time = 8.3 / approver_count^1.8 + 0.4 * log(amount) + 2 * backlog"""
    base = 8.3 / max(approver_count, 0.1) ** 1.8
    amt_factor = 0.4 * math.log(max(amount, 1))
    return base + amt_factor + 2.0 * backlog + urgency * -1.5 + random.gauss(0, 0.3)


def _delivery_delay(distance_km: float, stock_level: float, order_priority: float, carrier_load: float) -> float:
    """Synthetic: days_delay = sqrt(distance/100) * (1 + carrier_load) - stock_level * 0.5"""
    return (
        math.sqrt(distance_km / 100) * (1 + carrier_load)
        - stock_level * 0.5
        + (1 - order_priority) * 2
        + random.gauss(0, 0.2)
    )


def _invoice_payment_days(amount: float, customer_score: float, outstanding_ratio: float, terms_days: float) -> float:
    """Synthetic: payment_days ~ terms_days * (1 + outstanding_ratio) / customer_score"""
    return (
        terms_days * (1 + outstanding_ratio * 0.5) / max(customer_score, 0.1)
        + random.gauss(0, 2)
    )


def _stock_reorder_qty(avg_daily_use: float, lead_time_days: float, safety_factor: float, current_stock: float) -> float:
    """Synthetic: reorder_qty = avg_daily_use * lead_time_days * safety_factor - current_stock"""
    return max(0.0, avg_daily_use * lead_time_days * safety_factor - current_stock + random.gauss(0, 1))


METRIC_GENERATORS = [
    {
        "label": "approval_time_hours",
        "input_names": ["approver_count", "amount_log", "urgency", "backlog"],
        "fn": lambda: _approval_time(
            random.uniform(1, 10),
            random.lognormvariate(math.log(50000), 1),
            random.random(),
            random.uniform(0, 5),
        ),
        "inputs_fn": lambda: [
            random.uniform(1, 10),
            math.log(random.lognormvariate(math.log(50000), 1)),
            random.random(),
            random.uniform(0, 5),
        ],
    },
    {
        "label": "delivery_delay_days",
        "input_names": ["distance_km", "stock_level", "order_priority", "carrier_load"],
        "fn": None,
        "inputs_fn": lambda: [
            random.uniform(10, 5000),
            random.uniform(0, 1),
            random.uniform(0, 1),
            random.uniform(0, 1),
        ],
    },
    {
        "label": "payment_days",
        "input_names": ["amount_log", "customer_score", "outstanding_ratio", "terms_days"],
        "inputs_fn": lambda: [
            math.log(max(random.lognormvariate(math.log(50000), 1), 1)),
            random.uniform(0.3, 1.0),
            random.uniform(0, 0.8),
            random.choice([15, 30, 45, 60, 90]),
        ],
        "fn": None,
    },
    {
        "label": "reorder_quantity",
        "input_names": ["avg_daily_use", "lead_time_days", "safety_factor", "current_stock"],
        "inputs_fn": lambda: [
            random.uniform(1, 100),
            random.uniform(3, 60),
            random.uniform(1.2, 2.5),
            random.uniform(0, 200),
        ],
        "fn": None,
    },
]

_FUNCTIONS = [_approval_time, _delivery_delay, _invoice_payment_days, _stock_reorder_qty]


def generate_kan_pairs(n_samples: int) -> list[dict]:
    pairs = []
    fns = [_approval_time, _delivery_delay, _invoice_payment_days, _stock_reorder_qty]

    for i, gen in enumerate(METRIC_GENERATORS):
        fn = fns[i]
        for _ in range(n_samples // len(METRIC_GENERATORS)):
            inputs = gen["inputs_fn"]()
            output = fn(*inputs)
            pairs.append({
                "label": gen["label"],
                "input_names": gen["input_names"],
                "inputs": inputs,
                "output": float(output),
            })

    random.shuffle(pairs)
    return pairs


# ---------------------------------------------------------------------------
# Liquid NN — operational time-series
# ---------------------------------------------------------------------------

def generate_timeseries(n_series: int, series_len: int = 60) -> list[dict]:
    """
    Simulate Frappe server operational metrics over time.
    Features: [cpu_pct, mem_pct, queue_depth, db_latency_ms, events_per_min]
    """
    series_list = []

    for _ in range(n_series):
        # Each series represents one server over `series_len` minutes
        base_cpu  = random.uniform(20, 70)
        base_mem  = random.uniform(40, 80)
        base_q    = random.uniform(0, 20)
        base_db   = random.uniform(10, 100)
        base_evt  = random.uniform(10, 500)

        # Random event: spike at a random timestep
        spike_t = random.randint(10, series_len - 10)
        spike_mag = random.uniform(1.5, 3.0)

        series = []
        for t in range(series_len):
            spike = spike_mag if abs(t - spike_t) < 5 else 1.0
            cpu = min(100, base_cpu * spike + random.gauss(0, 3))
            mem = min(100, base_mem + (cpu - base_cpu) * 0.3 + random.gauss(0, 2))
            q   = max(0, base_q * spike + random.gauss(0, 1))
            db  = max(1,  base_db * (1 + q * 0.05) + random.gauss(0, 5))
            evt = max(0, base_evt * spike + random.gauss(0, 10))
            series.append([
                cpu / 100.0,
                mem / 100.0,
                min(q, 100) / 100.0,
                min(db, 1000) / 1000.0,
                min(evt, 1000) / 1000.0,
            ])

        # Target = next step
        target = series[1:] + [series[-1]]

        series_list.append({
            "series": series,
            "target": target,
            "source": "synthetic_frappe_monitor",
            "spike_at": spike_t,
        })

    return series_list


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples",    type=int, default=5000,  help="Samples per dataset")
    parser.add_argument("--series-len", type=int, default=60,    help="Time-series length")
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()

    base = Path(args.output_dir)
    (base / "jepa").mkdir(parents=True, exist_ok=True)
    (base / "kan").mkdir(parents=True, exist_ok=True)
    (base / "liquid").mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.samples} samples per dataset...\n")

    # JEPA
    print("JEPA state transitions...")
    jepa_data = generate_jepa_sequences(args.samples)
    jepa_path = base / "jepa" / "transitions.json"
    with open(jepa_path, "w") as f:
        json.dump(jepa_data, f)
    print(f"  {len(jepa_data)} sequences → {jepa_path}")

    # KAN
    print("KAN metric pairs...")
    kan_data = generate_kan_pairs(args.samples)
    kan_path = base / "kan" / "metric_pairs.json"
    with open(kan_path, "w") as f:
        json.dump(kan_data, f)
    print(f"  {len(kan_data)} pairs → {kan_path}")

    # Liquid
    print("Liquid NN time-series...")
    liquid_data = generate_timeseries(args.samples, series_len=args.series_len)
    liquid_path = base / "liquid" / "timeseries.json"
    with open(liquid_path, "w") as f:
        json.dump(liquid_data, f)
    print(f"  {len(liquid_data)} series → {liquid_path}")

    print("\nDone. Run Phase 2 trainers:")
    print("  python -m synapseos.phase2.jepa_trainer")
    print("  python -m synapseos.phase2.kan_trainer")
    print("  python -m synapseos.phase2.liquid_trainer")


if __name__ == "__main__":
    main()
