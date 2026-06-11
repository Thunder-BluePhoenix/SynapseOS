"""
Skills — Math solver. Computed, never recalled. Zero hallucination by design.

Covers:
  arithmetic     safe AST-based expression evaluation (sin, log, sqrt, ...)
  algebra        linear & quadratic equations parsed from text ("x^2-5x+6=0")
  calculus       symbolic polynomial derivatives, numeric derivative at a
                 point, numeric definite integrals (Simpson)
  stats/prob     mean/median/mode/std/variance, nCr/nPr, binomial probability
  geometry &     formula registry: areas, volumes, perimeters for the
  mensuration    standard shapes, with the formula shown
  trigonometry   degree-aware evaluation ("sin(30 deg)" → 0.5)
  constants      physics/quantum constants (c, h, ħ, G, k_B, N_A, e, m_e)

Every MathAnswer carries the steps taken — the teacher agent shows its work.
"""

from __future__ import annotations

import ast
import math
import re
import statistics
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Safe expression evaluation
# ---------------------------------------------------------------------------

_ALLOWED_FUNCS = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "log2": math.log2,
    "exp": math.exp, "abs": abs, "round": round, "floor": math.floor,
    "ceil": math.ceil, "factorial": math.factorial, "degrees": math.degrees,
    "radians": math.radians,
}
_ALLOWED_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}

PHYSICS_CONSTANTS = {
    "speed of light":      (299_792_458.0, "m/s", "c"),
    "planck constant":     (6.62607015e-34, "J·s", "h"),
    "reduced planck":      (1.054571817e-34, "J·s", "ħ"),
    "gravitational constant": (6.67430e-11, "m³/(kg·s²)", "G"),
    "boltzmann constant":  (1.380649e-23, "J/K", "k_B"),
    "avogadro":            (6.02214076e23, "1/mol", "N_A"),
    "elementary charge":   (1.602176634e-19, "C", "e"),
    "electron mass":       (9.1093837015e-31, "kg", "m_e"),
    "proton mass":         (1.67262192369e-27, "kg", "m_p"),
    "gas constant":        (8.31446261815324, "J/(mol·K)", "R"),
}


class _SafeEval(ast.NodeVisitor):
    _BINOPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
               ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
               ast.Pow: lambda a, b: a ** b, ast.Mod: lambda a, b: a % b,
               ast.FloorDiv: lambda a, b: a // b}

    def eval(self, expr: str) -> float:
        tree = ast.parse(expr, mode="eval")
        return self._ev(tree.body)

    def _ev(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in _ALLOWED_CONSTS:
                return _ALLOWED_CONSTS[node.id]
            raise ValueError(f"unknown name '{node.id}'")
        if isinstance(node, ast.BinOp) and type(node.op) in self._BINOPS:
            return self._BINOPS[type(node.op)](self._ev(node.left), self._ev(node.right))
        if isinstance(node, ast.UnaryOp):
            v = self._ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else +v
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = _ALLOWED_FUNCS.get(node.func.id)
            if fn is None:
                raise ValueError(f"function '{node.func.id}' not allowed")
            return fn(*[self._ev(a) for a in node.args])
        raise ValueError(f"unsupported syntax: {ast.dump(node)[:40]}")


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class MathAnswer:
    topic: str                  # arithmetic | algebra | calculus | stats | geometry | trig | constant
    answer: str                 # human-readable result
    value: float | list | None = None
    steps: list[str] = field(default_factory=list)
    exact: bool = True          # computed (True) vs numeric approximation

    def to_dict(self) -> dict:
        return {"topic": self.topic, "answer": self.answer,
                "steps": self.steps, "exact": self.exact}


# ---------------------------------------------------------------------------
# Polynomial helpers (symbolic calculus on polynomials)
# ---------------------------------------------------------------------------

_TERM_RE = re.compile(
    r"([+-]?\s*\d*\.?\d*)\s*\*?\s*x(?:\s*[\^]\s*|\s*\*\*\s*)?(\d+)?|([+-]?\s*\d+\.?\d*)")


def _parse_poly(expr: str) -> dict[int, float] | None:
    """'3x^2 + 2x - 5' → {2: 3, 1: 2, 0: -5}. None if not a polynomial."""
    s = expr.replace(" ", "").replace("**", "^").replace("·", "*")
    if not s or not re.fullmatch(r"[0-9x+\-.^*]+", s):
        return None
    coeffs: dict[int, float] = {}
    pos = 0
    for m in _TERM_RE.finditer(s):
        if m.start() != pos:
            return None
        pos = m.end()
        if m.group(3) is not None:                       # constant term
            coeffs[0] = coeffs.get(0, 0.0) + float(m.group(3).replace(" ", ""))
        else:                                            # x term
            c = m.group(1).replace(" ", "")
            coef = 1.0 if c in ("", "+") else -1.0 if c == "-" else float(c)
            power = int(m.group(2)) if m.group(2) else 1
            coeffs[power] = coeffs.get(power, 0.0) + coef
    return coeffs if pos == len(s) and coeffs else None


def _poly_str(coeffs: dict[int, float]) -> str:
    parts = []
    for p in sorted(coeffs, reverse=True):
        c = coeffs[p]
        if abs(c) < 1e-12:
            continue
        cs = f"{c:g}"
        if p == 0:
            parts.append(f"{'+ ' if c > 0 else '- '}{abs(c):g}")
        else:
            xs = "x" if p == 1 else f"x^{p}"
            mag = "" if abs(c) == 1 else f"{abs(c):g}"
            parts.append(f"{'+ ' if c > 0 else '- '}{mag}{xs}")
    out = " ".join(parts) or "0"
    return out[2:] if out.startswith("+ ") else ("-" + out[2:] if out.startswith("- ") else out)


# ---------------------------------------------------------------------------
# Geometry / mensuration formula registry
# ---------------------------------------------------------------------------

# (shape, measure) -> (formula text, fn(params...), param names)
GEOMETRY: dict[tuple[str, str], tuple[str, callable, list[str]]] = {
    ("circle", "area"):          ("π·r²", lambda r: math.pi * r * r, ["radius"]),
    ("circle", "circumference"): ("2π·r", lambda r: 2 * math.pi * r, ["radius"]),
    ("square", "area"):          ("a²", lambda a: a * a, ["side"]),
    ("square", "perimeter"):     ("4a", lambda a: 4 * a, ["side"]),
    ("rectangle", "area"):       ("l·w", lambda l, w: l * w, ["length", "width"]),
    ("rectangle", "perimeter"):  ("2(l+w)", lambda l, w: 2 * (l + w), ["length", "width"]),
    ("triangle", "area"):        ("½·b·h", lambda b, h: 0.5 * b * h, ["base", "height"]),
    ("sphere", "volume"):        ("4/3·π·r³", lambda r: 4/3 * math.pi * r**3, ["radius"]),
    ("sphere", "area"):          ("4π·r²", lambda r: 4 * math.pi * r * r, ["radius"]),
    ("cylinder", "volume"):      ("π·r²·h", lambda r, h: math.pi * r * r * h, ["radius", "height"]),
    ("cone", "volume"):          ("⅓·π·r²·h", lambda r, h: math.pi * r * r * h / 3, ["radius", "height"]),
    ("cube", "volume"):          ("a³", lambda a: a ** 3, ["side"]),
    ("cube", "area"):            ("6a²", lambda a: 6 * a * a, ["side"]),
}
_MEASURE_ALIASES = {"surface area": "area", "circumference": "circumference",
                    "perimeter": "perimeter", "volume": "volume", "area": "area"}


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class MathSolver:
    """Routes a natural-language math query to the right computation."""

    def __init__(self) -> None:
        self._eval = _SafeEval()

    # -- dispatcher --------------------------------------------------------------

    def solve(self, query: str) -> MathAnswer | None:
        """Try every recognizer; None if this isn't a math query we can do."""
        q = query.strip()
        # calculus before equations: "derivative of x^2 at x = 4" contains
        # an "x = 4" that the equation recognizer would otherwise grab
        for recognizer in (self._try_constant, self._try_derivative,
                           self._try_integral, self._try_equation,
                           self._try_stats, self._try_probability,
                           self._try_geometry, self._try_trig_degrees,
                           self._try_arithmetic):
            try:
                ans = recognizer(q)
            except Exception:
                ans = None
            if ans is not None:
                return ans
        return None

    # -- recognizers ---------------------------------------------------------------

    def _try_arithmetic(self, q: str) -> MathAnswer | None:
        m = re.search(r"(?:calculate|compute|evaluate|what is|solve)?\s*"
                      r"([\d().+\-*/^%\s]*[a-z]*\([\d().+\-*/^,\s]+\)[\d().+\-*/^%\s]*"
                      r"|[\d().+\-*/^%\s]{3,})\s*\??$", q.lower())
        if not m:
            return None
        expr = m.group(1).strip().replace("^", "**")
        if not re.search(r"\d", expr) or not re.search(r"[+\-*/(]|\*\*", expr):
            return None
        value = self._eval.eval(expr)
        return MathAnswer(topic="arithmetic", answer=f"{expr.replace('**', '^')} = {value:g}",
                          value=value, steps=[f"evaluated expression exactly"])

    def _try_equation(self, q: str) -> MathAnswer | None:
        m = re.search(r"([0-9x .+\-*/^]+)=([0-9x .+\-*/^]+)", q.lower())
        if not m or "x" not in m.group(0):
            return None
        left, right = _parse_poly(m.group(1)), _parse_poly(m.group(2))
        if left is None or right is None:
            return None
        # Move everything left: left - right = 0
        for p, c in right.items():
            left[p] = left.get(p, 0.0) - c
        deg = max((p for p, c in left.items() if abs(c) > 1e-12), default=0)
        steps = [f"normalized to: {_poly_str(left)} = 0"]
        if deg == 1:
            a, b = left.get(1, 0.0), left.get(0, 0.0)
            x = -b / a
            steps.append(f"linear: x = -({b:g})/({a:g})")
            return MathAnswer(topic="algebra", answer=f"x = {x:g}", value=x, steps=steps)
        if deg == 2:
            a, b, c = left.get(2, 0.0), left.get(1, 0.0), left.get(0, 0.0)
            disc = b * b - 4 * a * c
            steps.append(f"quadratic formula, discriminant = {disc:g}")
            if disc < 0:
                re_p, im_p = -b / (2 * a), math.sqrt(-disc) / (2 * a)
                return MathAnswer(topic="algebra",
                                  answer=f"x = {re_p:g} ± {im_p:g}i (complex roots)",
                                  value=None, steps=steps)
            r1 = (-b + math.sqrt(disc)) / (2 * a)
            r2 = (-b - math.sqrt(disc)) / (2 * a)
            return MathAnswer(topic="algebra",
                              answer=f"x = {r1:g} or x = {r2:g}" if abs(r1-r2) > 1e-12
                                     else f"x = {r1:g} (double root)",
                              value=[r1, r2], steps=steps)
        return None

    def _try_derivative(self, q: str) -> MathAnswer | None:
        m = re.search(r"(?:derivative|differentiate)\s+(?:of\s+)?([0-9x .+\-*/^]+?)"
                      r"(?:\s+at\s+x\s*=\s*([\-0-9.]+))?\s*\??$", q.lower())
        if not m:
            return None
        coeffs = _parse_poly(m.group(1))
        if coeffs is not None:
            deriv = {p - 1: c * p for p, c in coeffs.items() if p > 0}
            steps = [f"d/dx of {_poly_str(coeffs)} (power rule)"]
            if m.group(2) is not None:
                x = float(m.group(2))
                val = sum(c * x ** p for p, c in deriv.items())
                steps.append(f"evaluated at x = {x:g}")
                return MathAnswer(topic="calculus",
                                  answer=f"f'(x) = {_poly_str(deriv) or '0'}; f'({x:g}) = {val:g}",
                                  value=val, steps=steps)
            return MathAnswer(topic="calculus", answer=f"f'(x) = {_poly_str(deriv) or '0'}",
                              steps=steps)
        return None

    def _try_integral(self, q: str) -> MathAnswer | None:
        m = re.search(r"integra(?:te|l)(?:\s+of)?\s+([0-9x .+\-*/^a-z()]+?)\s+from\s+"
                      r"([\-0-9.]+)\s+to\s+([\-0-9.]+)", q.lower())
        if not m:
            return None
        expr = m.group(1).strip().replace("^", "**")
        a, b = float(m.group(2)), float(m.group(3))

        def f(x: float) -> float:
            return self._eval.eval(expr.replace("x", f"({x!r})"))

        n = 1000                                   # Simpson's rule
        h = (b - a) / n
        total = f(a) + f(b)
        for i in range(1, n):
            total += f(a + i * h) * (4 if i % 2 else 2)
        val = total * h / 3
        return MathAnswer(topic="calculus",
                          answer=f"∫ {m.group(1).strip()} dx from {a:g} to {b:g} ≈ {val:.6g}",
                          value=val, steps=[f"Simpson's rule, n={n}"], exact=False)

    def _try_stats(self, q: str) -> MathAnswer | None:
        m = re.search(r"(mean|average|median|mode|standard deviation|std|variance)\s+"
                      r"(?:of\s+)?\[?([\d.,\s\-]+)\]?", q.lower())
        if not m:
            return None
        nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", m.group(2))]
        if len(nums) < 2:
            return None
        op = m.group(1)
        fns = {"mean": statistics.mean, "average": statistics.mean,
               "median": statistics.median, "mode": statistics.mode,
               "standard deviation": statistics.stdev, "std": statistics.stdev,
               "variance": statistics.variance}
        val = fns[op](nums)
        return MathAnswer(topic="stats", answer=f"{op} of {nums} = {val:g}",
                          value=val, steps=[f"n = {len(nums)} values"])

    def _try_probability(self, q: str) -> MathAnswer | None:
        ql = q.lower()
        m = re.search(r"(\d+)\s*c\s*(\d+)|choose\s+(\d+)\s+from\s+(\d+)|"
                      r"combinations?\s+of\s+(\d+)\s+(?:items?\s+)?(?:taken|from|choose)\s+(\d+)", ql)
        if m:
            g = [x for x in m.groups() if x]
            n, k = int(g[0]), int(g[1])
            if "from" in ql and m.group(3):       # "choose K from N"
                n, k = int(g[1]), int(g[0])
            val = math.comb(n, k)
            return MathAnswer(topic="probability", answer=f"C({n},{k}) = {val}",
                              value=val, steps=[f"n!/(k!(n-k)!) with n={n}, k={k}"])
        m = re.search(r"(\d+)\s*p\s*(\d+)|permutations?\s+of\s+(\d+)\s+(?:taken|from|choose)\s+(\d+)", ql)
        if m and ("permutation" in ql or re.search(r"\dp\d", ql.replace(" ", ""))):
            g = [x for x in m.groups() if x]
            n, k = int(g[0]), int(g[1])
            val = math.perm(n, k)
            return MathAnswer(topic="probability", answer=f"P({n},{k}) = {val}",
                              value=val, steps=[f"n!/(n-k)! with n={n}, k={k}"])
        return None

    def _try_geometry(self, q: str) -> MathAnswer | None:
        ql = q.lower()
        measure = next((std for alias, std in _MEASURE_ALIASES.items() if alias in ql), None)
        shape = next((s for s in {s for s, _ in GEOMETRY} if s in ql), None)
        if measure is None or shape is None:
            return None
        entry = GEOMETRY.get((shape, measure))
        if entry is None:
            return None
        formula, fn, params = entry
        nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", ql)]
        if len(nums) < len(params):
            return MathAnswer(topic="geometry",
                              answer=f"{measure} of a {shape} = {formula} "
                                     f"(need: {', '.join(params)})",
                              steps=["formula lookup"])
        val = fn(*nums[: len(params)])
        given = ", ".join(f"{p}={v:g}" for p, v in zip(params, nums))
        return MathAnswer(topic="geometry",
                          answer=f"{measure} of {shape} ({given}) = {formula} = {val:.6g}",
                          value=val, steps=[f"formula: {formula}", f"substituted {given}"])

    def _try_trig_degrees(self, q: str) -> MathAnswer | None:
        m = re.search(r"\b(sin|cos|tan)\s*\(?\s*([\-0-9.]+)\s*(?:deg(?:rees)?|°)\s*\)?", q.lower())
        if not m:
            return None
        fn = {"sin": math.sin, "cos": math.cos, "tan": math.tan}[m.group(1)]
        deg = float(m.group(2))
        val = fn(math.radians(deg))
        if abs(val) < 1e-12:
            val = 0.0
        return MathAnswer(topic="trigonometry",
                          answer=f"{m.group(1)}({deg:g}°) = {round(val, 10):g}",
                          value=val, steps=[f"converted {deg:g}° to radians"])

    def _try_constant(self, q: str) -> MathAnswer | None:
        ql = q.lower()
        for name, (val, unit, sym) in PHYSICS_CONSTANTS.items():
            if name in ql:
                return MathAnswer(topic="constant",
                                  answer=f"{name} ({sym}) = {val:g} {unit}",
                                  value=val, steps=["physics constant (CODATA)"])
        return None
