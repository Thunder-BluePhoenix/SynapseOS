"""
Phase 5 — Liquid Layer Drift Monitor.

The Liquid Neural Network from Phase 2 continuously adapts its weights
based on incoming inputs (ODE solver updates hidden state between calls).
In production this is a feature — but unchecked drift can degrade outputs.

This module provides:
  1. WeightSnapshot   — immutable point-in-time record of model parameter norms
  2. DriftMetric      — measured distance between two snapshots
  3. AdaptationLog    — time-series of drift measurements (persisted to JSON)
  4. DriftMonitor     — full lifecycle: capture → measure → alert → rollback → consolidate
  5. ConsolidationScheduler — periodic distillation of adaptations into the base weights

Design constraints:
  - CPU-only (no CUDA required for drift measurement)
  - Works with any torch.nn.Module or with saved .pt state_dict files
  - Threshold is ±10% relative drift (from Phase 5 success criteria)
  - Rollback restores from a named snapshot checkpoint

Run standalone:
    python -m synapseos.phase5.drift_monitor
    python -m synapseos.phase5.drift_monitor --snapshot data/checkpoints/liquid/liquid_best.pt
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DRIFT_THRESHOLD = 0.10      # 10% relative drift triggers alert
DEFAULT_ROLLBACK_THRESHOLD = 0.25   # 25% → force rollback
CHECKPOINT_DIR = "data/checkpoints/liquid"
ADAPTATION_LOG_PATH = "data/drift/adaptation_log.json"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class WeightSnapshot:
    """Immutable snapshot of model parameter L2-norms at a point in time."""
    snapshot_id: str
    model_name: str
    timestamp: str
    checkpoint_path: str             # where the full state_dict was saved
    param_norms: dict[str, float]    # parameter_name → L2 norm
    total_params: int
    label: str = "manual"            # manual | auto | consolidation

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WeightSnapshot":
        return cls(**d)


@dataclass
class DriftMetric:
    """Measured drift between a baseline snapshot and a current snapshot."""
    baseline_id: str
    current_id: str
    timestamp: str
    param_drifts: dict[str, float]   # parameter_name → relative drift (|Δnorm| / baseline_norm)
    overall_drift: float             # max single-parameter drift
    mean_drift: float                # mean across all parameters
    within_threshold: bool
    alert: bool                      # True when overall_drift > DRIFT_THRESHOLD
    force_rollback: bool             # True when overall_drift > ROLLBACK_THRESHOLD
    drift_threshold: float
    rollback_threshold: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DriftMetric":
        return cls(**d)

    def summary(self) -> str:
        status = "OK"
        if self.force_rollback:
            status = "ROLLBACK REQUIRED"
        elif self.alert:
            status = "ALERT"
        return (
            f"Drift [{status}] overall={self.overall_drift:.3f} "
            f"mean={self.mean_drift:.3f} "
            f"threshold={self.drift_threshold:.2f}"
        )


@dataclass
class ConsolidationRecord:
    """Records a consolidation event (adaption distilled into base weights)."""
    timestamp: str
    source_snapshot_id: str
    target_snapshot_id: str          # new baseline after consolidation
    params_consolidated: int
    mean_drift_before: float


# ---------------------------------------------------------------------------
# Adaptation Log
# ---------------------------------------------------------------------------

class AdaptationLog:
    """
    Persisted time-series of drift measurements.
    Allows plotting the drift trajectory over a production session.
    """

    def __init__(self, log_path: str = ADAPTATION_LOG_PATH):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.measurements: list[DriftMetric] = []
        self.consolidations: list[ConsolidationRecord] = []
        self._load()

    def _load(self) -> None:
        if self.log_path.exists():
            try:
                data = json.loads(self.log_path.read_text())
                self.measurements = [DriftMetric.from_dict(m) for m in data.get("measurements", [])]
                self.consolidations = [ConsolidationRecord(**c) for c in data.get("consolidations", [])]
            except Exception:
                pass

    def save(self) -> None:
        data = {
            "measurements":   [m.to_dict() for m in self.measurements],
            "consolidations": [asdict(c) for c in self.consolidations],
        }
        self.log_path.write_text(json.dumps(data, indent=2))

    def record(self, metric: DriftMetric) -> None:
        self.measurements.append(metric)
        # Keep last 1000 measurements
        if len(self.measurements) > 1000:
            self.measurements = self.measurements[-1000:]

    def record_consolidation(self, record: ConsolidationRecord) -> None:
        self.consolidations.append(record)

    def recent(self, n: int = 10) -> list[DriftMetric]:
        return list(reversed(self.measurements[-n:]))

    def trend(self) -> dict:
        """Compute drift trend statistics over the last 20 measurements."""
        recent = self.measurements[-20:]
        if not recent:
            return {"count": 0}
        overall_drifts = [m.overall_drift for m in recent]
        alerts = sum(1 for m in recent if m.alert)
        return {
            "count":       len(recent),
            "min":         round(min(overall_drifts), 4),
            "max":         round(max(overall_drifts), 4),
            "mean":        round(sum(overall_drifts) / len(overall_drifts), 4),
            "alerts":      alerts,
            "alert_rate":  round(alerts / len(recent), 3),
            "increasing":  overall_drifts[-1] > overall_drifts[0] if len(overall_drifts) > 1 else False,
        }

    def stats(self) -> dict:
        return {
            "total_measurements":   len(self.measurements),
            "total_consolidations": len(self.consolidations),
            "trend":                self.trend(),
        }


# ---------------------------------------------------------------------------
# Drift Monitor
# ---------------------------------------------------------------------------

class DriftMonitor:
    """
    Full lifecycle manager for Liquid NN weight drift in production.

    Usage:
        monitor = DriftMonitor()

        # Capture baseline at startup
        baseline_id = monitor.capture(model, label="startup")

        # ... model runs inference, adapts weights ...

        # Measure drift
        metric = monitor.measure(model, baseline_id)
        if metric.force_rollback:
            monitor.rollback(model, baseline_id)
    """

    def __init__(
        self,
        checkpoint_dir: str = CHECKPOINT_DIR,
        log_path: str = ADAPTATION_LOG_PATH,
        drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
        rollback_threshold: float = DEFAULT_ROLLBACK_THRESHOLD,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.drift_threshold = drift_threshold
        self.rollback_threshold = rollback_threshold
        self.log = AdaptationLog(log_path)
        self._snapshots: dict[str, WeightSnapshot] = {}
        self._load_snapshot_index()

    # ------------------------------------------------------------------
    # Snapshot index
    # ------------------------------------------------------------------

    @property
    def _index_path(self) -> Path:
        return self.checkpoint_dir / "snapshot_index.json"

    def _load_snapshot_index(self) -> None:
        if self._index_path.exists():
            try:
                data = json.loads(self._index_path.read_text())
                self._snapshots = {k: WeightSnapshot.from_dict(v) for k, v in data.items()}
            except Exception:
                self._snapshots = {}

    def _save_snapshot_index(self) -> None:
        data = {k: v.to_dict() for k, v in self._snapshots.items()}
        self._index_path.write_text(json.dumps(data, indent=2))

    # ------------------------------------------------------------------
    # Snapshot capture
    # ------------------------------------------------------------------

    def capture(
        self,
        model: nn.Module,
        label: str = "manual",
        model_name: str = "liquid_nn",
    ) -> str:
        """
        Capture current model weights as a named snapshot.
        Saves full state_dict to disk and records parameter norms.
        Returns snapshot_id.
        """
        snap_id = f"{model_name}_{label}_{int(time.time())}"
        ckpt_path = self.checkpoint_dir / f"{snap_id}.pt"

        # Save full state dict
        torch.save(model.state_dict(), str(ckpt_path))

        # Compute parameter norms (CPU)
        param_norms = {}
        total_params = 0
        for name, param in model.named_parameters():
            norm = float(param.detach().cpu().float().norm(2).item())
            param_norms[name] = round(norm, 6)
            total_params += param.numel()

        snapshot = WeightSnapshot(
            snapshot_id=snap_id,
            model_name=model_name,
            timestamp=datetime.now().isoformat(),
            checkpoint_path=str(ckpt_path),
            param_norms=param_norms,
            total_params=total_params,
            label=label,
        )
        self._snapshots[snap_id] = snapshot
        self._save_snapshot_index()
        return snap_id

    def capture_from_file(
        self,
        checkpoint_path: str,
        label: str = "file",
        model_name: str = "liquid_nn",
    ) -> str:
        """
        Load a saved checkpoint and capture its parameter norms as a snapshot.
        No live model required — works from .pt files.
        """
        snap_id = f"{model_name}_{label}_{int(time.time())}"
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        param_norms: dict[str, float] = {}
        total_params = 0
        for name, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor):
                norm = float(tensor.float().norm(2).item())
                param_norms[name] = round(norm, 6)
                total_params += tensor.numel()

        snapshot = WeightSnapshot(
            snapshot_id=snap_id,
            model_name=model_name,
            timestamp=datetime.now().isoformat(),
            checkpoint_path=checkpoint_path,
            param_norms=param_norms,
            total_params=total_params,
            label=label,
        )
        self._snapshots[snap_id] = snapshot
        self._save_snapshot_index()
        return snap_id

    # ------------------------------------------------------------------
    # Drift measurement
    # ------------------------------------------------------------------

    def measure(
        self,
        model_or_path: nn.Module | str,
        baseline_id: str,
    ) -> DriftMetric:
        """
        Measure parameter drift between current model state and a baseline snapshot.

        model_or_path: live nn.Module OR path to a .pt checkpoint file.
        baseline_id:   snapshot_id returned by capture().
        """
        if baseline_id not in self._snapshots:
            raise KeyError(f"Baseline snapshot '{baseline_id}' not found")

        baseline = self._snapshots[baseline_id]

        # Get current parameter norms
        if isinstance(model_or_path, str):
            state_dict = torch.load(model_or_path, map_location="cpu", weights_only=True)
            current_norms = {
                name: float(t.float().norm(2).item())
                for name, t in state_dict.items()
                if isinstance(t, torch.Tensor)
            }
        else:
            current_norms = {
                name: float(p.detach().cpu().float().norm(2).item())
                for name, p in model_or_path.named_parameters()
            }

        # Compute per-parameter relative drift
        param_drifts: dict[str, float] = {}
        for name, baseline_norm in baseline.param_norms.items():
            current_norm = current_norms.get(name, 0.0)
            if baseline_norm > 1e-8:
                drift = abs(current_norm - baseline_norm) / baseline_norm
            else:
                drift = 0.0
            param_drifts[name] = round(drift, 6)

        overall_drift = max(param_drifts.values()) if param_drifts else 0.0
        mean_drift = (
            sum(param_drifts.values()) / len(param_drifts)
            if param_drifts else 0.0
        )

        metric = DriftMetric(
            baseline_id=baseline_id,
            current_id=f"live_{int(time.time())}",
            timestamp=datetime.now().isoformat(),
            param_drifts=param_drifts,
            overall_drift=round(overall_drift, 6),
            mean_drift=round(mean_drift, 6),
            within_threshold=overall_drift <= self.drift_threshold,
            alert=overall_drift > self.drift_threshold,
            force_rollback=overall_drift > self.rollback_threshold,
            drift_threshold=self.drift_threshold,
            rollback_threshold=self.rollback_threshold,
        )
        self.log.record(metric)
        return metric

    def measure_between(self, snapshot_a_id: str, snapshot_b_id: str) -> DriftMetric:
        """Compare two saved snapshots without a live model."""
        if snapshot_a_id not in self._snapshots:
            raise KeyError(f"Snapshot '{snapshot_a_id}' not found")
        if snapshot_b_id not in self._snapshots:
            raise KeyError(f"Snapshot '{snapshot_b_id}' not found")

        snap_a = self._snapshots[snapshot_a_id]
        snap_b = self._snapshots[snapshot_b_id]

        param_drifts: dict[str, float] = {}
        for name, norm_a in snap_a.param_norms.items():
            norm_b = snap_b.param_norms.get(name, 0.0)
            drift = abs(norm_b - norm_a) / norm_a if norm_a > 1e-8 else 0.0
            param_drifts[name] = round(drift, 6)

        overall_drift = max(param_drifts.values()) if param_drifts else 0.0
        mean_drift = sum(param_drifts.values()) / len(param_drifts) if param_drifts else 0.0

        metric = DriftMetric(
            baseline_id=snapshot_a_id,
            current_id=snapshot_b_id,
            timestamp=datetime.now().isoformat(),
            param_drifts=param_drifts,
            overall_drift=round(overall_drift, 6),
            mean_drift=round(mean_drift, 6),
            within_threshold=overall_drift <= self.drift_threshold,
            alert=overall_drift > self.drift_threshold,
            force_rollback=overall_drift > self.rollback_threshold,
            drift_threshold=self.drift_threshold,
            rollback_threshold=self.rollback_threshold,
        )
        self.log.record(metric)
        return metric

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def rollback(self, model: nn.Module, snapshot_id: str) -> None:
        """
        Restore model weights from a saved snapshot checkpoint.
        In-place operation — modifies model.
        """
        if snapshot_id not in self._snapshots:
            raise KeyError(f"Snapshot '{snapshot_id}' not found")

        snap = self._snapshots[snapshot_id]
        ckpt = snap.checkpoint_path

        if not Path(ckpt).exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        state_dict = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        print(f"[DriftMonitor] Rolled back to snapshot '{snapshot_id}' ({snap.label} @ {snap.timestamp})")

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------

    def consolidate(
        self,
        adapted_model: nn.Module,
        base_model: nn.Module,
        alpha: float = 0.1,
        label: str = "consolidated",
    ) -> str:
        """
        Distill Liquid NN adaptations into the base model weights via weighted averaging:
            new_base = (1 - alpha) * base + alpha * adapted

        This preserves most of the base model while incorporating the learned adaptations.
        Returns the new baseline snapshot_id.
        """
        # Measure drift before consolidation
        pre_metric = self.measure(adapted_model, self._latest_snapshot_id())
        mean_drift_before = pre_metric.mean_drift if pre_metric else 0.0

        source_id = self._latest_snapshot_id()

        # Blend weights
        with torch.no_grad():
            for (name, base_param), (_, adapted_param) in zip(
                base_model.named_parameters(),
                adapted_model.named_parameters(),
            ):
                base_param.data.copy_(
                    (1.0 - alpha) * base_param.data + alpha * adapted_param.data
                )

        # Capture new consolidated baseline
        new_id = self.capture(base_model, label=label)

        record = ConsolidationRecord(
            timestamp=datetime.now().isoformat(),
            source_snapshot_id=source_id,
            target_snapshot_id=new_id,
            params_consolidated=sum(p.numel() for p in base_model.parameters()),
            mean_drift_before=mean_drift_before,
        )
        self.log.record_consolidation(record)
        self.log.save()

        print(f"[DriftMonitor] Consolidated: alpha={alpha}, new baseline='{new_id}'")
        return new_id

    def _latest_snapshot_id(self) -> str:
        if not self._snapshots:
            return ""
        return max(self._snapshots.values(), key=lambda s: s.timestamp).snapshot_id

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def list_snapshots(self) -> list[WeightSnapshot]:
        return sorted(self._snapshots.values(), key=lambda s: s.timestamp, reverse=True)

    def get_snapshot(self, snap_id: str) -> WeightSnapshot | None:
        return self._snapshots.get(snap_id)

    def stats(self) -> dict:
        return {
            "snapshots":      len(self._snapshots),
            "drift_threshold": self.drift_threshold,
            "rollback_threshold": self.rollback_threshold,
            "log":            self.log.stats(),
        }


# ---------------------------------------------------------------------------
# Consolidation Scheduler
# ---------------------------------------------------------------------------

class ConsolidationScheduler:
    """
    Schedules periodic consolidation of Liquid NN adaptations.
    Runs as a background thread — checks drift every N seconds and
    consolidates when mean_drift exceeds the consolidation threshold.
    """

    def __init__(
        self,
        monitor: DriftMonitor,
        check_interval: float = 300.0,   # 5 minutes
        consolidation_threshold: float = 0.08,
        alpha: float = 0.1,
    ):
        import threading
        self.monitor = monitor
        self.check_interval = check_interval
        self.consolidation_threshold = consolidation_threshold
        self.alpha = alpha
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._adapted_model: nn.Module | None = None
        self._base_model: nn.Module | None = None
        self._baseline_id: str = ""

    def set_models(
        self,
        adapted: nn.Module,
        base: nn.Module,
        baseline_id: str,
    ) -> None:
        self._adapted_model = adapted
        self._base_model    = base
        self._baseline_id   = baseline_id

    def start(self) -> None:
        import threading
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="SynapseOS-ConsolidationScheduler"
        )
        self._thread.start()
        print(f"[ConsolidationScheduler] Started (interval={self.check_interval}s)")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while not self._stop.wait(timeout=self.check_interval):
            if not (self._adapted_model and self._base_model and self._baseline_id):
                continue
            try:
                metric = self.monitor.measure(self._adapted_model, self._baseline_id)
                print(f"[ConsolidationScheduler] {metric.summary()}")
                if metric.mean_drift >= self.consolidation_threshold:
                    new_id = self.monitor.consolidate(
                        self._adapted_model,
                        self._base_model,
                        alpha=self.alpha,
                    )
                    self._baseline_id = new_id
                    print(f"[ConsolidationScheduler] Auto-consolidated → {new_id}")
            except Exception as e:
                print(f"[ConsolidationScheduler] Error: {e}")


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

def _make_demo_model(hidden: int = 32) -> nn.Module:
    return nn.Sequential(
        nn.Linear(16, hidden),
        nn.Tanh(),
        nn.Linear(hidden, 8),
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Liquid Layer Drift Monitor")
    parser.add_argument("--snapshot", default=None, help="Path to a .pt checkpoint to use as baseline")
    parser.add_argument("--threshold", type=float, default=DEFAULT_DRIFT_THRESHOLD)
    args = parser.parse_args()

    monitor = DriftMonitor(drift_threshold=args.threshold)

    if args.snapshot and Path(args.snapshot).exists():
        print(f"[Demo] Capturing baseline from file: {args.snapshot}")
        baseline_id = monitor.capture_from_file(args.snapshot, label="cli")
    else:
        print("[Demo] No checkpoint provided — using synthetic model for demo")
        model = _make_demo_model()
        baseline_id = monitor.capture(model, label="demo_baseline")
        print(f"[Demo] Baseline captured: {baseline_id}")

        # Simulate adaptation (perturb weights)
        adapted = _make_demo_model()
        with torch.no_grad():
            for param in adapted.parameters():
                param.data += torch.randn_like(param.data) * 0.15

        adapted_id = monitor.capture(adapted, label="demo_adapted")
        metric = monitor.measure_between(baseline_id, adapted_id)
        print(f"\n{metric.summary()}")

        if metric.alert:
            print("ALERT: Drift exceeds threshold!")
        if metric.force_rollback:
            print("ROLLBACK REQUIRED — restoring baseline weights")
            monitor.rollback(adapted, baseline_id)
            print("Rollback complete.")

    print(f"\nMonitor stats: {monitor.stats()}")
    print(f"Trend: {monitor.log.trend()}")


if __name__ == "__main__":
    main()
