"""
Phase 4 — Idle Exploration Loop (Mode 5: Default Mode Network).

When no active task is running, SynapseOS enters idle mode — mirroring the
human brain's default mode network during rest.

What happens during idle:
  1. Background graph walk — explore low-confidence knowledge edges
  2. Pattern consolidation — compress weak hypotheses into stronger facts
  3. Weak hypothesis store — record low-confidence new connections
  4. Cross-domain bridge discovery — find analogies across module boundaries
  5. Liquid layer adaptation hook — signal that weights should update

The system becomes smarter during idle without explicit retraining.

Run standalone:
    python -m synapseos.phase4.idle_loop --duration 30

Or integrate into the pipeline:
    loop = IdleLoop(graph_path=..., hypothesis_store=...)
    loop.start_background()
    # ... active inference happens ...
    loop.stop()
    report = loop.get_session_report()
"""

from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Hypothesis store
# ---------------------------------------------------------------------------

@dataclass
class WeakHypothesis:
    id: str
    type: str                   # cross_domain_link | pattern | consolidation
    concept_a: str
    concept_b: str
    description: str
    confidence: float           # starts low, increases if reinforced
    observed_at: str
    reinforcement_count: int = 0

    def reinforce(self, delta: float = 0.1) -> None:
        self.confidence = min(1.0, self.confidence + delta)
        self.reinforcement_count += 1

    def to_dict(self) -> dict:
        return asdict(self)


class HypothesisStore:
    def __init__(self, store_path: str = "data/idle/hypotheses.json"):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.hypotheses: dict[str, WeakHypothesis] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if self.store_path.exists():
            try:
                data = json.loads(self.store_path.read_text())
                for h in data:
                    self.hypotheses[h["id"]] = WeakHypothesis(**h)
            except Exception:
                pass

    def save(self) -> None:
        with self._lock:
            data = [h.to_dict() for h in self.hypotheses.values()]
            self.store_path.write_text(json.dumps(data, indent=2))

    def add(self, hyp: WeakHypothesis) -> None:
        with self._lock:
            # If already exists, reinforce rather than duplicate
            key = f"{hyp.concept_a}|{hyp.concept_b}|{hyp.type}"
            if key in self.hypotheses:
                self.hypotheses[key].reinforce()
            else:
                hyp.id = key
                self.hypotheses[key] = hyp

    def get_strong(self, threshold: float = 0.7) -> list[WeakHypothesis]:
        return [h for h in self.hypotheses.values() if h.confidence >= threshold]

    def get_all(self) -> list[WeakHypothesis]:
        return list(self.hypotheses.values())

    def prune(self, max_size: int = 1000) -> int:
        with self._lock:
            if len(self.hypotheses) <= max_size:
                return 0
            # Keep strongest
            sorted_hyps = sorted(self.hypotheses.values(), key=lambda h: h.confidence, reverse=True)
            pruned = len(self.hypotheses) - max_size
            self.hypotheses = {h.id: h for h in sorted_hyps[:max_size]}
            return pruned


# ---------------------------------------------------------------------------
# Idle loop
# ---------------------------------------------------------------------------

@dataclass
class IdleSessionReport:
    duration_seconds: float
    walks_completed: int
    hypotheses_generated: int
    hypotheses_reinforced: int
    strong_patterns_found: int
    domains_explored: list[str]
    top_hypotheses: list[dict]


class IdleLoop:
    """
    Background idle exploration — Mode 5.

    Runs in a daemon thread. Can be stopped cleanly.
    Does not modify any neural weights directly — provides signals for
    the Liquid NN adapter to consume on its next update cycle.
    """

    def __init__(
        self,
        graph_path: str = "data/graph/knowledge_graph.json",
        hypothesis_path: str = "data/idle/hypotheses.json",
        tick_interval: float = 2.0,    # seconds between idle ticks
        idle_threshold: float = 5.0,   # seconds of inactivity before idle triggers
    ):
        self.tick_interval  = tick_interval
        self.idle_threshold = idle_threshold
        self.hypothesis_store = HypothesisStore(hypothesis_path)

        self._nodes: dict[str, dict] = {}
        self._adjacency: dict[str, list[str]] = {}
        self._domain_index: dict[str, list[str]] = {}
        self._load_graph(graph_path)

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_active = time.time()

        # Session stats
        self._walks = 0
        self._hyps_generated = 0
        self._hyps_reinforced = 0
        self._domains_visited: set[str] = set()
        self._started_at: float | None = None

    def _load_graph(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text())
            for node in data.get("nodes", []):
                nid = node.get("id", "")
                if not nid or "__" in nid:
                    continue
                self._nodes[nid] = node
                domain = node.get("module", "general")
                if domain not in self._domain_index:
                    self._domain_index[domain] = []
                self._domain_index[domain].append(nid)

            for edge in data.get("links", data.get("edges", [])):
                src = edge.get("source", "")
                tgt = edge.get("target", "")
                if src and tgt:
                    if src not in self._adjacency:
                        self._adjacency[src] = []
                    self._adjacency[src].append(tgt)
        except Exception as e:
            print(f"[Idle] Graph load error: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def signal_active(self) -> None:
        """Call this whenever the system handles an active query."""
        self._last_active = time.time()

    def start_background(self) -> None:
        self._stop_event.clear()
        self._started_at = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True, name="SynapseOS-IdleLoop")
        self._thread.start()
        print(f"[Idle] Background loop started (tick={self.tick_interval}s, threshold={self.idle_threshold}s)")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        self.hypothesis_store.save()
        print(f"[Idle] Loop stopped. Hypotheses saved → {self.hypothesis_store.store_path}")

    def get_session_report(self) -> IdleSessionReport:
        duration = time.time() - (self._started_at or time.time())
        strong = self.hypothesis_store.get_strong(threshold=0.6)
        return IdleSessionReport(
            duration_seconds=round(duration, 1),
            walks_completed=self._walks,
            hypotheses_generated=self._hyps_generated,
            hypotheses_reinforced=self._hyps_reinforced,
            strong_patterns_found=len(strong),
            domains_explored=sorted(self._domains_visited),
            top_hypotheses=[h.to_dict() for h in sorted(strong, key=lambda h: h.confidence, reverse=True)[:5]],
        )

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            idle_for = time.time() - self._last_active
            if idle_for >= self.idle_threshold:
                self._idle_tick()
            self._stop_event.wait(timeout=self.tick_interval)

    def _idle_tick(self) -> None:
        if not self._nodes:
            return

        # 1. Random walk — explore low-confidence edges
        start = random.choice(list(self._nodes.keys()))
        walk = self._random_walk(start, steps=6)
        self._walks += 1

        # 2. Find cross-domain connections along walk
        walk_domains = set()
        for node_id in walk:
            if node_id in self._nodes:
                domain = self._nodes[node_id].get("module", "general")
                walk_domains.add(domain)
                self._domains_visited.add(domain)

        # 3. If walk crossed domains, generate a weak hypothesis
        if len(walk_domains) > 1 and len(walk) >= 3:
            source = walk[0]
            target = walk[-1]
            source_domain = self._nodes.get(source, {}).get("module", "?")
            target_domain = self._nodes.get(target, {}).get("module", "?")

            if source_domain != target_domain:
                hyp = WeakHypothesis(
                    id="",
                    type="cross_domain_link",
                    concept_a=source,
                    concept_b=target,
                    description=(
                        f"Potential cross-domain pattern: {source} ({source_domain}) "
                        f"→ {target} ({target_domain}) via {len(walk)-2} hops"
                    ),
                    confidence=0.2 + 0.05 * len(walk_domains),
                    observed_at=datetime.now().isoformat(),
                )
                prev_count = len(self.hypothesis_store.hypotheses)
                self.hypothesis_store.add(hyp)
                new_count = len(self.hypothesis_store.hypotheses)
                if new_count > prev_count:
                    self._hyps_generated += 1
                else:
                    self._hyps_reinforced += 1

        # 4. Pattern consolidation — promote high-reinforcement hypotheses
        for hyp in self.hypothesis_store.get_all():
            if hyp.reinforcement_count >= 3 and hyp.confidence < 0.6:
                hyp.reinforce(delta=0.15)

        # 5. Prune if too large
        self.hypothesis_store.prune(max_size=500)

        # 6. Persist every 10 walks
        if self._walks % 10 == 0:
            self.hypothesis_store.save()

    def _random_walk(self, start: str, steps: int) -> list[str]:
        path = [start]
        current = start
        for _ in range(steps):
            neighbors = self._adjacency.get(current, [])
            valid = [n for n in neighbors if n in self._nodes and n not in path]
            if not valid:
                # Jump to random node
                current = random.choice(list(self._nodes.keys()))
            else:
                current = random.choice(valid)
            path.append(current)
        return path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=15.0, help="Seconds to run idle loop")
    parser.add_argument("--tick",     type=float, default=1.0,  help="Tick interval in seconds")
    args = parser.parse_args()

    print(f"Starting idle loop for {args.duration}s...\n")

    loop = IdleLoop(tick_interval=args.tick, idle_threshold=0.0)   # threshold=0 → always idle
    loop.start_background()

    time.sleep(args.duration)
    loop.stop()

    report = loop.get_session_report()
    print(f"\n=== Idle Session Report ===")
    print(f"Duration          : {report.duration_seconds}s")
    print(f"Graph walks       : {report.walks_completed}")
    print(f"Hypotheses new    : {report.hypotheses_generated}")
    print(f"Hypotheses reinf. : {report.hypotheses_reinforced}")
    print(f"Strong patterns   : {report.strong_patterns_found}")
    print(f"Domains explored  : {', '.join(report.domains_explored[:8])}")

    if report.top_hypotheses:
        print(f"\nTop hypotheses:")
        for h in report.top_hypotheses:
            print(f"  [{h['confidence']:.2f}] {h['description'][:80]}")


if __name__ == "__main__":
    main()
