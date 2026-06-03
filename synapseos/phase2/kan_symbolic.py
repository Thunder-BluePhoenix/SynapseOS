"""
Phase 2 — KAN Symbolic Formula Extractor.

Extracts interpretable symbolic formulas from the trained KAN/MLP checkpoint
without requiring `pykan`. Uses regression-based curve fitting on model outputs.

Method:
  1. Load trained model + normalization parameters
  2. For each input dimension: sweep across [0,1] while holding others at median
  3. Fit candidate symbolic functions (linear, power, log, sqrt, quadratic, inverse)
  4. Report best-fit formula per dimension (ranked by R²)
  5. Compose global formula approximation
  6. Back-transform from normalized to original feature space

The resulting formulas become auditable business rules fed into the
NeurosymbolicCore — e.g.:
  approval_time_hours ≈ 3.2 / approver_count^0.9 + 0.4 × log(amount) + base_time

Run:
    python -m synapseos.phase2.kan_symbolic
    python -m synapseos.phase2.kan_symbolic --checkpoint data/checkpoints/kan/kan_best.pt
    python -m synapseos.phase2.kan_symbolic --save data/rules/kan_formulas.json
"""

from __future__ import annotations

import json
import math
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Candidate symbolic functions (operate on 1D normalized input x ∈ [0, 1])
# ---------------------------------------------------------------------------

class SymbolicFunction:
    """A named symbolic function type with feature transform for linear fitting."""

    def __init__(self, name: str, transform, formula_template: str):
        self.name = name
        self.transform = transform         # x (numpy array) → transformed features matrix
        self.formula_template = formula_template

    def fit(self, x: np.ndarray, y: np.ndarray) -> tuple[float, list[float]]:
        """
        Least-squares fit: solve y ≈ coefficients @ transform(x).
        Returns (R², coefficients).
        """
        try:
            X = self.transform(x)
            # Add bias column
            X = np.column_stack([X, np.ones(len(X))])
            # Least squares
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            y_pred = X @ coeffs
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = float(1 - ss_res / (ss_tot + 1e-10))
            return r2, coeffs.tolist()
        except Exception:
            return -1.0, []

    def describe(self, feature_name: str, coeffs: list[float]) -> str:
        """Produce human-readable formula string."""
        if not coeffs:
            return f"[{self.name}: fit failed]"
        c = coeffs[:-1]   # slope coefficients
        b = coeffs[-1]    # bias
        return self.formula_template.format(
            feature=feature_name,
            c=round(c[0], 4) if c else "?",
            c2=round(c[1], 4) if len(c) > 1 else "?",
            b=round(b, 4),
        )


EPSILON = 1e-6

SYMBOLIC_CANDIDATES = [
    SymbolicFunction(
        "linear",
        lambda x: x.reshape(-1, 1),
        "{c:.4g} × {feature} + {b:.4g}",
    ),
    SymbolicFunction(
        "quadratic",
        lambda x: np.column_stack([x**2, x]),
        "{c:.4g} × {feature}² + {c2:.4g} × {feature} + {b:.4g}",
    ),
    SymbolicFunction(
        "log",
        lambda x: np.log(x + EPSILON).reshape(-1, 1),
        "{c:.4g} × log({feature}) + {b:.4g}",
    ),
    SymbolicFunction(
        "sqrt",
        lambda x: np.sqrt(np.abs(x)).reshape(-1, 1),
        "{c:.4g} × √{feature} + {b:.4g}",
    ),
    SymbolicFunction(
        "power_neg",
        lambda x: (1.0 / (x + EPSILON)).reshape(-1, 1),
        "{c:.4g} / {feature} + {b:.4g}",
    ),
    SymbolicFunction(
        "exp_decay",
        lambda x: np.exp(-x).reshape(-1, 1),
        "{c:.4g} × exp(-{feature}) + {b:.4g}",
    ),
    SymbolicFunction(
        "power_half",
        lambda x: (x ** 0.5 * np.log(x + EPSILON)).reshape(-1, 1),
        "{c:.4g} × √{feature} × log({feature}) + {b:.4g}",
    ),
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DimensionFormula:
    dim_index: int
    feature_name: str
    best_function: str
    r2: float
    coefficients: list[float]
    formula_normalized: str   # in normalized [0,1] space
    formula_original: str     # back-transformed to original feature space

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SymbolicFormulaSet:
    """Complete symbolic approximation of the KAN model."""
    label: str                        # what metric is being predicted
    input_names: list[str]
    dimension_formulas: list[DimensionFormula]
    global_formula: str               # combined readable formula
    global_r2: float                  # overall fit quality
    model_checkpoint: str

    def to_dict(self) -> dict:
        return {
            "label":              self.label,
            "input_names":        self.input_names,
            "global_formula":     self.global_formula,
            "global_r2":          round(self.global_r2, 4),
            "model_checkpoint":   self.model_checkpoint,
            "dimension_formulas": [d.to_dict() for d in self.dimension_formulas],
        }

    def display(self) -> str:
        lines = [
            f"\nSymbolic Formula — {self.label}",
            f"{'─'*50}",
            f"  {self.global_formula}",
            f"  Global R²  : {self.global_r2:.4f}",
            f"\nDimension breakdown:",
        ]
        for d in sorted(self.dimension_formulas, key=lambda x: x.r2, reverse=True):
            lines.append(f"  [{d.r2:+.3f}] {d.feature_name:20s} → {d.best_function:12s}: {d.formula_original}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model_and_norm(
    checkpoint_path: str,
    norm_path: str,
    width: list[int] = None,
) -> tuple[nn.Module, dict]:
    """Load trained MLP and normalization parameters."""
    from synapseos.phase2.kan_trainer import FallbackMLP, KANConfig

    width = width or [4, 8, 1]
    model = FallbackMLP(width=width)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    norm = json.loads(Path(norm_path).read_text())
    return model, norm


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

class KANSymbolicExtractor:
    """
    Extracts symbolic formulas from a trained KAN/MLP model via regression.
    Works CPU-only — no pykan required.
    """

    def __init__(
        self,
        model: nn.Module,
        norm: dict,
        input_names: list[str],
        label: str,
        n_sweep: int = 200,
        checkpoint_path: str = "",
    ):
        self.model         = model
        self.norm          = norm
        self.input_names   = input_names
        self.label         = label
        self.n_sweep       = n_sweep
        self.checkpoint_path = checkpoint_path

        self.input_min  = np.array(norm["input_min"])
        self.input_max  = np.array(norm["input_max"])
        self.output_mean = float(norm["output_mean"])
        self.output_std  = float(norm["output_std"])
        self.input_dim   = len(input_names)

    def _predict_normalized(self, x_norm: np.ndarray) -> np.ndarray:
        """Run model on normalized inputs, return normalized outputs."""
        with torch.no_grad():
            t = torch.tensor(x_norm, dtype=torch.float32)
            out = self.model(t).squeeze(-1).numpy()
        return out

    def _sweep_dimension(self, dim: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Vary dimension `dim` across [0, 1] while holding others at 0.5.
        Returns (sweep_x_normalized, model_output_normalized).
        """
        sweep = np.linspace(0.0, 1.0, self.n_sweep)
        inputs = np.full((self.n_sweep, self.input_dim), 0.5)
        inputs[:, dim] = sweep
        outputs = self._predict_normalized(inputs)
        return sweep, outputs

    def _fit_dimension(self, dim: int) -> DimensionFormula:
        """Fit symbolic function to dimension `dim`."""
        x_norm, y_norm = self._sweep_dimension(dim)
        feature_name   = self.input_names[dim]

        best_r2   = -np.inf
        best_fn   = None
        best_coeffs = []

        for fn in SYMBOLIC_CANDIDATES:
            r2, coeffs = fn.fit(x_norm, y_norm)
            if r2 > best_r2:
                best_r2     = r2
                best_fn     = fn
                best_coeffs = coeffs

        if best_fn is None:
            best_fn = SYMBOLIC_CANDIDATES[0]

        formula_norm = best_fn.describe(f"{feature_name}_norm", best_coeffs)

        # Back-transform: x_norm = (x_orig - min) / (max - min + ε)
        x_min  = float(self.input_min[dim])
        x_max  = float(self.input_max[dim])
        x_range = x_max - x_min + 1e-8

        # For linear case: formula_orig uses raw x
        # For other cases we note the back-transform relationship
        if best_fn.name == "linear" and best_coeffs:
            c_norm  = best_coeffs[0]
            b_norm  = best_coeffs[-1]
            # y_norm ≈ c_norm * x_norm + b_norm
            # x_norm = (x - x_min) / x_range → substitute
            c_orig  = round(c_norm / x_range, 5)
            b_orig  = round(b_norm - c_norm * x_min / x_range, 5)
            formula_orig = f"{c_orig:.4g} × {feature_name} + {b_orig:.4g}"
        else:
            # Provide the normalized formula with a note
            formula_orig = (
                f"{formula_norm}  "
                f"[x_norm = ({feature_name} − {x_min:.3g}) / {x_range:.3g}]"
            )

        return DimensionFormula(
            dim_index=dim,
            feature_name=feature_name,
            best_function=best_fn.name,
            r2=round(float(best_r2), 4),
            coefficients=best_coeffs,
            formula_normalized=formula_norm,
            formula_original=formula_orig,
        )

    def _global_multivariate_fit(self) -> tuple[float, dict[str, Any]]:
        """
        Global multivariate regression: fit a linear combination of transformed
        features across all dimensions to the model output on a random grid.

        Selects the best transform per dimension independently, then fits
        them jointly. Returns (global_R², dict of selected feature info).
        """
        n_samples = 1000
        rng = np.random.default_rng(42)
        x_rand = rng.uniform(0.01, 0.99, size=(n_samples, self.input_dim))  # avoid 0/1 for log/inverse
        y_true = self._predict_normalized(x_rand)

        # Build design matrix: best transform per dimension (found via dimension sweeps)
        feature_cols = []
        feature_names_out = []
        selected_fns: list[SymbolicFunction] = []

        for dim in range(self.input_dim):
            fname = self.input_names[dim]
            x_col = x_rand[:, dim]

            # Try each function, pick best single-dim R² with global random samples
            best_r2_dim = -np.inf
            best_fn_dim = SYMBOLIC_CANDIDATES[0]
            for fn in SYMBOLIC_CANDIDATES:
                try:
                    X_dim = fn.transform(x_col)
                    if X_dim.ndim == 1:
                        X_dim = X_dim.reshape(-1, 1)
                    X_aug = np.column_stack([X_dim, np.ones(n_samples)])
                    coeffs, _, _, _ = np.linalg.lstsq(X_aug, y_true, rcond=None)
                    y_hat = X_aug @ coeffs
                    ss_res = np.sum((y_true - y_hat) ** 2)
                    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
                    r2 = float(1 - ss_res / (ss_tot + 1e-10))
                    if r2 > best_r2_dim:
                        best_r2_dim = r2
                        best_fn_dim = fn
                except Exception:
                    continue

            # Add the transformed column(s)
            X_best = best_fn_dim.transform(x_col)
            if X_best.ndim == 1:
                X_best = X_best.reshape(-1, 1)
            for col_i in range(X_best.shape[1]):
                feature_cols.append(X_best[:, col_i])
                feature_names_out.append(f"{best_fn_dim.name}({fname})[{col_i}]")
            selected_fns.append(best_fn_dim)

        # Global joint fit
        X_joint = np.column_stack(feature_cols + [np.ones(n_samples)])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_joint, y_true, rcond=None)
            y_pred = X_joint @ coeffs
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - y_true.mean()) ** 2)
            global_r2 = float(max(0.0, 1 - ss_res / (ss_tot + 1e-10)))
        except Exception:
            global_r2 = 0.0
            coeffs = np.zeros(X_joint.shape[1])

        return global_r2, {
            "selected_fns":   [f.name for f in selected_fns],
            "feature_names":  feature_names_out,
            "coefficients":   coeffs.tolist(),
        }

    def _global_r2(self, formulas: list[DimensionFormula]) -> float:
        r2, _ = self._global_multivariate_fit()
        return r2

    def _compose_global_formula(self, formulas: list[DimensionFormula]) -> str:
        """
        Compose a human-readable global formula using multivariate regression.
        Uses the best per-dimension transform with fitted global coefficients.
        """
        global_r2, fit_info = self._global_multivariate_fit()
        selected_fns = fit_info["selected_fns"]
        coeffs       = fit_info["coefficients"]

        out_scale = round(self.output_std, 3)
        out_bias  = round(self.output_mean, 3)

        # Build term for each dimension using selected function and global coefficient
        terms = []
        coeff_idx = 0
        for dim, (fname, fn_name) in enumerate(zip(self.input_names, selected_fns)):
            fn = next((f for f in SYMBOLIC_CANDIDATES if f.name == fn_name), SYMBOLIC_CANDIDATES[0])
            # Number of columns this function produces (quadratic → 2, others → 1)
            sample_x = np.array([0.5])
            n_cols = fn.transform(sample_x).reshape(1, -1).shape[1]

            c_vals = coeffs[coeff_idx:coeff_idx + n_cols]
            coeff_idx += n_cols
            c_main = round(float(c_vals[0]), 4)

            if abs(c_main) < 0.001:
                continue

            if fn_name == "linear":
                terms.append(f"{c_main:.4g}×{fname}")
            elif fn_name == "log":
                terms.append(f"{c_main:.4g}×log({fname})")
            elif fn_name == "sqrt":
                terms.append(f"{c_main:.4g}×√{fname}")
            elif fn_name == "power_neg":
                terms.append(f"{c_main:.4g}/{fname}")
            elif fn_name == "quadratic":
                c2 = round(float(c_vals[1]), 4) if len(c_vals) > 1 else 0
                terms.append(f"{c_main:.4g}×{fname}²+{c2:.4g}×{fname}")
            elif fn_name == "exp_decay":
                terms.append(f"{c_main:.4g}×exp(-{fname})")
            else:
                terms.append(f"{c_main:.4g}×f({fname})")

        bias = round(float(coeffs[-1]) * out_scale + out_bias, 3) if len(coeffs) > 0 else out_bias
        body = " + ".join(terms) if terms else "constant"
        return f"{self.label} ≈ ({body}) × {out_scale} + {bias}  [R²={global_r2:.4f}]"

    def extract(self) -> SymbolicFormulaSet:
        """Run full symbolic extraction. Returns a SymbolicFormulaSet."""
        print(f"[KANSymbolic] Extracting formulas for '{self.label}' ({self.input_dim}D) ...")
        formulas = [self._fit_dimension(d) for d in range(self.input_dim)]
        global_r2 = self._global_r2(formulas)
        global_formula = self._compose_global_formula(formulas)

        return SymbolicFormulaSet(
            label=self.label,
            input_names=self.input_names,
            dimension_formulas=formulas,
            global_formula=global_formula,
            global_r2=global_r2,
            model_checkpoint=self.checkpoint_path,
        )


# ---------------------------------------------------------------------------
# Multi-label extraction
# ---------------------------------------------------------------------------

def extract_all_labels(
    checkpoint_path: str = "data/checkpoints/kan/kan_best.pt",
    norm_path: str = "data/checkpoints/kan/normalization.json",
    data_path: str = "data/kan/metric_pairs.json",
    config_path: str = "data/checkpoints/kan/kan_config.json",
    save_path: str | None = "data/rules/kan_formulas.json",
    n_sweep: int = 200,
) -> list[SymbolicFormulaSet]:
    """
    Load the trained KAN, detect unique labels and their input names,
    extract symbolic formulas for each label, optionally save results.
    """
    # Load data to get label/input_names mapping
    records = json.loads(Path(data_path).read_text())
    label_to_inputs: dict[str, list[str]] = {}
    for rec in records:
        lbl = rec["label"]
        if lbl not in label_to_inputs:
            label_to_inputs[lbl] = rec["input_names"]

    # Load config to get model width
    config = json.loads(Path(config_path).read_text()) if Path(config_path).exists() else {}
    width  = config.get("width", [4, 8, 1])

    # Load model
    model, norm = load_model_and_norm(checkpoint_path, norm_path, width)

    results = []
    for label, input_names in sorted(label_to_inputs.items()):
        extractor = KANSymbolicExtractor(
            model=model,
            norm=norm,
            input_names=input_names,
            label=label,
            n_sweep=n_sweep,
            checkpoint_path=checkpoint_path,
        )
        formula_set = extractor.extract()
        results.append(formula_set)
        print(formula_set.display())

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = [fs.to_dict() for fs in results]
        out.write_text(json.dumps(data, indent=2))
        print(f"\n[KANSymbolic] Saved {len(results)} formula sets → {save_path}")

    return results


# ---------------------------------------------------------------------------
# Integration with NeurosymbolicCore
# ---------------------------------------------------------------------------

def formulas_to_rules(formula_sets: list[SymbolicFormulaSet]) -> list[dict]:
    """
    Convert extracted symbolic formulas into the rules engine format.
    These are appended to data/rules/rules.json as 'symbolic_formula' type rules.
    """
    rules = []
    for fs in formula_sets:
        rules.append({
            "doctype":     fs.label.replace("_", " ").title(),
            "rule_type":   "symbolic_formula",
            "event":       "compute",
            "condition":   "always",
            "consequence": fs.global_formula,
            "source":      "kan_extraction",
            "r2":          fs.global_r2,
            "features":    fs.input_names,
        })
    return rules


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="KAN Symbolic Formula Extractor")
    parser.add_argument("--checkpoint", default="data/checkpoints/kan/kan_best.pt")
    parser.add_argument("--norm",       default="data/checkpoints/kan/normalization.json")
    parser.add_argument("--data",       default="data/kan/metric_pairs.json")
    parser.add_argument("--config",     default="data/checkpoints/kan/kan_config.json")
    parser.add_argument("--save",       default="data/rules/kan_formulas.json")
    parser.add_argument("--n-sweep",    type=int, default=200)
    parser.add_argument("--to-rules",   action="store_true",
                        help="Also convert formulas to rules engine format")
    args = parser.parse_args()

    formula_sets = extract_all_labels(
        checkpoint_path=args.checkpoint,
        norm_path=args.norm,
        data_path=args.data,
        config_path=args.config,
        save_path=args.save,
        n_sweep=args.n_sweep,
    )

    if args.to_rules:
        rules = formulas_to_rules(formula_sets)
        rules_path = "data/rules/kan_symbolic_rules.json"
        Path(rules_path).parent.mkdir(parents=True, exist_ok=True)
        Path(rules_path).write_text(json.dumps({"rules": rules}, indent=2))
        print(f"[KANSymbolic] Exported {len(rules)} rules → {rules_path}")

    print(f"\nSummary: {len(formula_sets)} formula sets extracted")
    for fs in formula_sets:
        print(f"  {fs.label:30s}  global_R²={fs.global_r2:.4f}")


if __name__ == "__main__":
    main()
