"""
Business rules extractor for Frappe/ERPNext.

Extracts symbolic rules from:
  - Python validate() methods (field constraints, conditional requirements)
  - Custom scripts (server-side JS transpiled patterns)
  - Property setters (dynamic field properties)
  - Required/read-only conditions in DocType JSON

Output: list of Rule objects → written to data/rules/rules.json
These rules feed directly into the Neurosymbolic Core in Phase 3.
"""

import ast
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Rule:
    id: str
    doctype: str
    rule_type: str           # constraint | conditional_required | workflow_guard | computed
    event: str               # validate | on_submit | on_cancel | etc.
    condition: str           # human-readable condition string
    consequence: str         # what happens when condition is true
    source_file: str
    source_line: int = 0
    confidence: float = 1.0  # 1.0 = extracted from explicit code, <1.0 = inferred


@dataclass
class RulesExtractionResult:
    rules: list[Rule] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rules": [asdict(r) for r in self.rules],
            "stats": self.stats,
        }


class RulesExtractor:
    def __init__(self, apps_dir: str, app_filter: list[str] | None = None):
        self.apps_dir = Path(apps_dir)
        self.app_filter = app_filter
        self.result = RulesExtractionResult()
        self._rule_counter = 0

    def extract(self) -> RulesExtractionResult:
        apps = self._get_apps()
        for app_path in apps:
            self._process_app(app_path)

        self.result.stats = {
            "apps_scanned": len(apps),
            "total_rules": len(self.result.rules),
            "by_type": self._count_by_type(),
            "by_event": self._count_by_event(),
        }
        return self.result

    def save(self, output_path: str) -> None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(self.result.to_dict(), f, indent=2)
        print(f"Rules saved → {out}  ({len(self.result.rules)} rules)")

    def _get_apps(self) -> list[Path]:
        apps = []
        for item in sorted(self.apps_dir.iterdir()):
            if not item.is_dir():
                continue
            if self.app_filter and item.name not in self.app_filter:
                continue
            apps.append(item)
        return apps

    def _process_app(self, app_path: Path) -> None:
        # Scan all Python controllers (*.py files inside doctype directories)
        for py_file in app_path.rglob("*.py"):
            if "doctype" not in str(py_file):
                continue
            if py_file.name.startswith("test_"):
                continue
            self._extract_from_python(py_file)

        # Scan DocType JSON for mandatory/read-only conditions
        for json_file in app_path.rglob("*.json"):
            if "doctype" not in str(json_file):
                continue
            self._extract_from_doctype_json(json_file)

    def _extract_from_python(self, py_file: Path) -> None:
        try:
            source = py_file.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source)
        except (SyntaxError, OSError):
            return

        doctype = self._infer_doctype_from_path(py_file)

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue

            func_name = node.name
            if func_name not in {
                "validate", "before_save", "on_submit", "before_submit",
                "on_cancel", "before_cancel", "on_update", "autoname",
            }:
                continue

            # Extract frappe.throw() calls — these are hard constraints
            self._extract_throw_calls(node, doctype, func_name, str(py_file))

            # Extract frappe.msgprint with raise_exception — soft constraints
            self._extract_msgprint_calls(node, doctype, func_name, str(py_file))

            # Extract set_mandatory / df.reqd manipulations — dynamic requirements
            self._extract_mandatory_changes(node, doctype, func_name, str(py_file))

    def _extract_throw_calls(
        self, func_node: ast.FunctionDef, doctype: str, event: str, source_file: str
    ) -> None:
        for node in ast.walk(func_node):
            if not isinstance(node, ast.Call):
                continue

            # frappe.throw(_(\"message\"), condition_class)
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr == "throw"
                and isinstance(func.value, ast.Name)
                and func.value.id == "frappe"
            ):
                continue

            if not node.args:
                continue

            msg = self._extract_string_value(node.args[0])
            if not msg:
                continue

            # Walk up to find enclosing if-condition
            condition = self._find_enclosing_condition(func_node, node)

            rule_id = self._next_id()
            self.result.rules.append(Rule(
                id=rule_id,
                doctype=doctype,
                rule_type="constraint",
                event=event,
                condition=condition or "always",
                consequence=f"throw: {msg[:200]}",
                source_file=source_file,
                source_line=getattr(node, "lineno", 0),
            ))

    def _extract_msgprint_calls(
        self, func_node: ast.FunctionDef, doctype: str, event: str, source_file: str
    ) -> None:
        for node in ast.walk(func_node):
            if not isinstance(node, ast.Call):
                continue

            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr == "msgprint"
                and isinstance(func.value, ast.Name)
                and func.value.id == "frappe"
            ):
                continue

            # Only include if raise_exception=True is set
            is_exception = any(
                (isinstance(kw.value, ast.Constant) and kw.value.value is True)
                for kw in node.keywords
                if kw.arg == "raise_exception"
            )
            if not is_exception:
                continue

            msg = self._extract_string_value(node.args[0]) if node.args else ""
            if not msg:
                continue

            condition = self._find_enclosing_condition(func_node, node)
            rule_id = self._next_id()
            self.result.rules.append(Rule(
                id=rule_id,
                doctype=doctype,
                rule_type="constraint",
                event=event,
                condition=condition or "always",
                consequence=f"msgprint(raise): {msg[:200]}",
                source_file=source_file,
                source_line=getattr(node, "lineno", 0),
            ))

    def _extract_mandatory_changes(
        self, func_node: ast.FunctionDef, doctype: str, event: str, source_file: str
    ) -> None:
        for node in ast.walk(func_node):
            # Detect: self.flags.ignore_mandatory or toggle_reqd patterns
            if not isinstance(node, ast.Assign):
                continue

            for target in node.targets:
                target_str = ast.unparse(target) if hasattr(ast, "unparse") else ""
                if "reqd" in target_str or "mandatory" in target_str.lower():
                    value_str = ast.unparse(node.value) if hasattr(ast, "unparse") else ""
                    condition = self._find_enclosing_condition(func_node, node)
                    rule_id = self._next_id()
                    self.result.rules.append(Rule(
                        id=rule_id,
                        doctype=doctype,
                        rule_type="conditional_required",
                        event=event,
                        condition=condition or "always",
                        consequence=f"set {target_str} = {value_str}",
                        source_file=source_file,
                        source_line=getattr(node, "lineno", 0),
                        confidence=0.8,
                    ))

    def _extract_from_doctype_json(self, json_file: Path) -> None:
        try:
            with open(json_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return

        if not isinstance(data, dict) or data.get("doctype") != "DocType":
            return

        doctype = data.get("name", "")
        if not doctype:
            return

        for field_def in data.get("fields", []):
            fieldname = field_def.get("fieldname", "")
            label = field_def.get("label", fieldname)

            # mandatory_depends_on → conditional required rule
            if field_def.get("mandatory_depends_on"):
                rule_id = self._next_id()
                self.result.rules.append(Rule(
                    id=rule_id,
                    doctype=doctype,
                    rule_type="conditional_required",
                    event="validate",
                    condition=field_def["mandatory_depends_on"],
                    consequence=f"field '{label}' ({fieldname}) becomes required",
                    source_file=str(json_file),
                ))

            # read_only_depends_on → access control rule
            if field_def.get("read_only_depends_on"):
                rule_id = self._next_id()
                self.result.rules.append(Rule(
                    id=rule_id,
                    doctype=doctype,
                    rule_type="conditional_required",
                    event="render",
                    condition=field_def["read_only_depends_on"],
                    consequence=f"field '{label}' ({fieldname}) becomes read-only",
                    source_file=str(json_file),
                ))

            # reqd=1 → hard requirement
            if field_def.get("reqd") and field_def.get("fieldtype") not in {"Section Break", "Column Break", "Tab Break"}:
                rule_id = self._next_id()
                self.result.rules.append(Rule(
                    id=rule_id,
                    doctype=doctype,
                    rule_type="constraint",
                    event="validate",
                    condition="always",
                    consequence=f"field '{label}' ({fieldname}) must not be empty",
                    source_file=str(json_file),
                ))

    def _infer_doctype_from_path(self, py_file: Path) -> str:
        # Path pattern: .../doctype/<doctype_folder>/<doctype_name>.py
        parts = py_file.parts
        for i, part in enumerate(parts):
            if part == "doctype" and i + 2 < len(parts):
                return parts[i + 1].replace("_", " ").title()
        return py_file.stem.replace("_", " ").title()

    def _extract_string_value(self, node: ast.expr) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        # _("translated string")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_":
            if node.args and isinstance(node.args[0], ast.Constant):
                return str(node.args[0].value)
        if hasattr(ast, "unparse"):
            return ast.unparse(node)
        return ""

    def _find_enclosing_condition(
        self, func_node: ast.FunctionDef, target_node: ast.AST
    ) -> str | None:
        target_line = getattr(target_node, "lineno", -1)

        # Walk all If nodes in the function, find the innermost that contains target_line
        best: ast.If | None = None
        best_start = -1

        for node in ast.walk(func_node):
            if not isinstance(node, ast.If):
                continue
            start = getattr(node, "lineno", -1)
            end = getattr(node, "end_lineno", start)
            if start <= target_line <= end and start > best_start:
                best = node
                best_start = start

        if best is None:
            return None

        if hasattr(ast, "unparse"):
            try:
                return ast.unparse(best.test)
            except Exception:
                pass
        return "if <condition>"

    def _next_id(self) -> str:
        self._rule_counter += 1
        return f"rule_{self._rule_counter:05d}"

    def _count_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rule in self.result.rules:
            counts[rule.rule_type] = counts.get(rule.rule_type, 0) + 1
        return counts

    def _count_by_event(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rule in self.result.rules:
            counts[rule.event] = counts.get(rule.event, 0) + 1
        return counts
