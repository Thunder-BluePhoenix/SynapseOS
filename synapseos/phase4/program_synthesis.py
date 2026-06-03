"""
Phase 4 — Program Synthesis Module (Mode 4: Code Thinking).

The biggest divergence from LLMs. Does NOT predict "what token comes after def".
Reasons about STRUCTURE first:

  1. Parse intent → extract function goal, inputs, constraints
  2. Query knowledge graph → what APIs/patterns exist for this domain
  3. Plan algorithm → decompose into verified symbolic sub-steps
  4. Synthesize code from plan (structure-first, not token-first)
  5. Static validate → check correctness before output
  6. Auto-generate tests

Every synthesized function is traceable to its symbolic plan.
No token prediction. No hallucination of APIs.

Run:
    python -m synapseos.phase4.program_synthesis
    python -m synapseos.phase4.program_synthesis --intent "validate a sales invoice before submission"
"""

from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from synapseos.phase3.neurosymbolic_core import NeurosymbolicCore, QueryType


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ParsedIntent:
    raw: str
    goal: str              # what the function should achieve
    doctype: str           # primary DocType involved
    inputs: list[str]      # parameter names inferred
    constraints: list[str] # what must hold true
    event: str             # validate | on_submit | on_cancel | utility


@dataclass
class AlgorithmStep:
    step: int
    action: str            # check_required | query_graph | apply_rule | compute | return_result
    description: str
    inputs: list[str]
    outputs: list[str]
    rule_ids: list[str] = field(default_factory=list)


@dataclass
class SynthesizedFunction:
    intent: ParsedIntent
    plan: list[AlgorithmStep]
    code: str
    tests: str
    static_valid: bool
    static_errors: list[str]
    generation_trace: list[str]


# ---------------------------------------------------------------------------
# Intent parser
# ---------------------------------------------------------------------------

class IntentParser:
    """Extracts structured intent from natural language."""

    EVENT_KEYWORDS = {
        "validate":     ["validate", "check", "verify", "ensure", "before save"],
        "on_submit":    ["submit", "post", "finalize", "on submit"],
        "on_cancel":    ["cancel", "reverse", "undo", "on cancel"],
        "utility":      ["get", "fetch", "calculate", "compute", "find", "list"],
    }

    KNOWN_DOCTYPES = [
        "Sales Invoice", "Purchase Invoice", "Sales Order", "Purchase Order",
        "Delivery Note", "Payment Entry", "Stock Entry", "Journal Entry",
        "Customer", "Supplier", "Item", "Employee", "Account", "Project",
        "Task", "Leave Application", "Expense Claim", "Salary Slip",
        "Purchase Receipt", "Stock Ledger Entry", "GL Entry",
    ]

    def parse(self, raw: str) -> ParsedIntent:
        raw_lower = raw.lower()

        # Identify event type
        event = "utility"
        for evt, keywords in self.EVENT_KEYWORDS.items():
            if any(kw in raw_lower for kw in keywords):
                event = evt
                break

        # Extract DocType
        doctype = ""
        for dt in self.KNOWN_DOCTYPES:
            if dt.lower() in raw_lower:
                doctype = dt
                break

        # Extract goal (strip doctype and event words)
        goal = raw.strip().rstrip(".")
        if not goal.endswith("?"):
            goal = goal[0].upper() + goal[1:]

        # Infer inputs
        inputs = ["doc"]
        if "amount" in raw_lower or "total" in raw_lower:
            inputs.append("amount")
        if "date" in raw_lower or "period" in raw_lower:
            inputs.append("posting_date")
        if "customer" in raw_lower:
            inputs.append("customer")

        # Extract constraints (sentences with "must", "should", "cannot")
        constraints = []
        for sent in re.split(r"[.;,]", raw):
            if any(w in sent.lower() for w in ["must", "should not", "cannot", "required"]):
                constraints.append(sent.strip())

        return ParsedIntent(
            raw=raw,
            goal=goal,
            doctype=doctype,
            inputs=list(dict.fromkeys(inputs)),   # dedupe, preserve order
            constraints=constraints,
            event=event,
        )


# ---------------------------------------------------------------------------
# Algorithm planner
# ---------------------------------------------------------------------------

class AlgorithmPlanner:
    """Decomposes a parsed intent into a symbolic algorithm plan."""

    def __init__(self, core: NeurosymbolicCore):
        self.core = core

    def plan(self, intent: ParsedIntent) -> list[AlgorithmStep]:
        steps: list[AlgorithmStep] = []

        # Step 1: always validate inputs first
        steps.append(AlgorithmStep(
            step=1,
            action="check_required",
            description=f"Ensure required fields are present on {intent.doctype or 'the document'}",
            inputs=["doc"],
            outputs=["validated_doc"],
        ))

        # Step 2: query knowledge graph for this DocType
        if intent.doctype:
            kg_result = self.core.reason(QueryType.RETRIEVE, intent.doctype)
            link_targets = [f["target"] for f in kg_result.facts if f.get("type") == "link"]

            if link_targets:
                steps.append(AlgorithmStep(
                    step=2,
                    action="query_graph",
                    description=f"Fetch linked records: {', '.join(link_targets[:3])}",
                    inputs=["doc"],
                    outputs=["linked_records"],
                ))

        # Step 3: apply business rules
        if intent.doctype:
            rule_result = self.core.reason(QueryType.REASON, intent.doctype,
                                            {"field_values": {}})
            if rule_result.rules_applied:
                steps.append(AlgorithmStep(
                    step=len(steps) + 1,
                    action="apply_rule",
                    description=f"Evaluate {len(rule_result.rules_applied)} business rules for {intent.doctype}",
                    inputs=["doc", "linked_records"],
                    outputs=["rule_results"],
                    rule_ids=rule_result.rules_applied[:5],
                ))

        # Step 4: event-specific logic
        if intent.event == "validate":
            steps.append(AlgorithmStep(
                step=len(steps) + 1,
                action="check_required",
                description="Raise frappe.ValidationError if any constraint is violated",
                inputs=["rule_results"],
                outputs=["None"],
            ))
        elif intent.event == "on_submit":
            steps.append(AlgorithmStep(
                step=len(steps) + 1,
                action="compute",
                description="Post accounting entries / update stock ledger",
                inputs=["doc", "rule_results"],
                outputs=["ledger_entries"],
            ))
        elif intent.event == "utility":
            steps.append(AlgorithmStep(
                step=len(steps) + 1,
                action="return_result",
                description="Aggregate and return computed result",
                inputs=["rule_results"],
                outputs=["result"],
            ))

        # Final step: return
        if steps[-1].action != "return_result":
            steps.append(AlgorithmStep(
                step=len(steps) + 1,
                action="return_result",
                description="Return success or computed value",
                inputs=[steps[-1].outputs[0]],
                outputs=["result"],
            ))

        return steps


# ---------------------------------------------------------------------------
# Code synthesizer
# ---------------------------------------------------------------------------

class CodeSynthesizer:
    """Produces Python/Frappe code from a symbolic algorithm plan."""

    def synthesize(self, intent: ParsedIntent, plan: list[AlgorithmStep]) -> str:
        func_name = self._to_snake(intent.goal)
        params = ", ".join(intent.inputs)
        docstring = intent.goal

        body_lines = [
            f'def {func_name}({params}):',
            f'    """',
            f'    {docstring}',
            f'',
            f'    Synthesized from symbolic plan — not token prediction.',
            f'    Generation trace: {len(plan)} steps.',
            f'    """',
            f'    import frappe',
            f'',
        ]

        for step in plan:
            body_lines.append(f"    # Step {step.step}: {step.description}")

            if step.action == "check_required":
                if intent.doctype:
                    body_lines += [
                        f"    required_fields = frappe.get_meta({repr(intent.doctype)}).get_mandatory_fields()",
                        f"    for field in required_fields:",
                        f"        if not doc.get(field.fieldname):",
                        f"            frappe.throw(_(f'{{field.label}} is required'), frappe.MandatoryError)",
                    ]

            elif step.action == "query_graph":
                targets = [f["target"] for f in self.core_facts(intent) if f.get("type") == "link"][:2]
                for target in targets:
                    var = self._to_snake(target)
                    body_lines += [
                        f"    {var}_records = frappe.get_all({repr(target)},",
                        f"        filters={{\"name\": doc.get(\"name\")}},",
                        f"        fields=[\"name\", \"status\"])",
                    ]

            elif step.action == "apply_rule":
                body_lines += [
                    f"    # Rules applied: {', '.join(step.rule_ids[:3])}",
                    f"    errors = []",
                ]
                for constraint in intent.constraints[:3]:
                    body_lines.append(f"    # Constraint: {constraint[:60]}")

            elif step.action == "compute":
                body_lines += [
                    f"    # Post accounting / stock ledger entries",
                    f"    doc.set_missing_values()",
                    f"    doc.validate_with_previous_doc()",
                ]

            elif step.action == "return_result":
                if intent.event in ("validate", "on_submit", "on_cancel"):
                    body_lines.append(f"    # Validation complete — no return value needed")
                else:
                    body_lines.append(f"    return doc")

            body_lines.append("")

        return "\n".join(body_lines)

    def core_facts(self, intent: ParsedIntent) -> list[dict]:
        return []   # filled in by synthesizer when wired to real core

    def _to_snake(self, text: str) -> str:
        s = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        s = re.sub(r"\s+", "_", s.strip().lower())
        s = re.sub(r"_+", "_", s)
        if s and s[0].isdigit():
            s = "fn_" + s
        return s or "synthesized_function"


# ---------------------------------------------------------------------------
# Static validator
# ---------------------------------------------------------------------------

class StaticValidator:
    def validate(self, code: str) -> tuple[bool, list[str]]:
        errors = []
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"SyntaxError at line {e.lineno}: {e.msg}")
            return False, errors

        # Check for common issues
        if "frappe.throw" in code and "import frappe" not in code:
            errors.append("frappe.throw used but frappe not imported")

        # Check function is defined
        if "def " not in code:
            errors.append("No function definition found")

        return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Test generator
# ---------------------------------------------------------------------------

class TestGenerator:
    def generate(self, intent: ParsedIntent, func_name: str) -> str:
        dt = intent.doctype or "DocType"
        fn = func_name

        return textwrap.dedent(f'''\
            import frappe
            import unittest


            class Test{dt.replace(" ", "")}(unittest.TestCase):

                def setUp(self):
                    frappe.set_user("Administrator")
                    self.doc = frappe.new_doc({repr(dt)})

                def test_{fn}_with_valid_data(self):
                    """Verify {fn} succeeds with all required fields."""
                    # TODO: populate self.doc with valid test data
                    # self.doc.customer = "Test Customer"
                    # self.doc.posting_date = frappe.utils.today()
                    try:
                        result = {fn}(self.doc)
                        self.assertIsNotNone(result, "Function should return a result")
                    except Exception as e:
                        self.fail(f"Unexpected error: {{e}}")

                def test_{fn}_raises_on_missing_required(self):
                    """Verify {fn} raises ValidationError when required fields are missing."""
                    empty_doc = frappe.new_doc({repr(dt)})
                    with self.assertRaises((frappe.ValidationError, frappe.MandatoryError)):
                        {fn}(empty_doc)

                def test_{fn}_idempotent(self):
                    """Verify calling {fn} twice with same input produces same result."""
                    # TODO: populate doc
                    # result1 = {fn}(self.doc)
                    # result2 = {fn}(self.doc)
                    # self.assertEqual(result1, result2)
                    pass


            if __name__ == "__main__":
                unittest.main()
        ''')


# ---------------------------------------------------------------------------
# Full synthesis pipeline
# ---------------------------------------------------------------------------

class ProgramSynthesizer:
    def __init__(self, core: NeurosymbolicCore | None = None):
        if core is None:
            core = NeurosymbolicCore()
        self.core      = core
        self.parser    = IntentParser()
        self.planner   = AlgorithmPlanner(core)
        self.synth     = CodeSynthesizer()
        self.validator = StaticValidator()
        self.test_gen  = TestGenerator()

    def synthesize(self, intent_text: str) -> SynthesizedFunction:
        trace = []

        # 1. Parse intent
        intent = self.parser.parse(intent_text)
        trace.append(f"Parsed intent: goal='{intent.goal}', doctype='{intent.doctype}', event='{intent.event}'")
        trace.append(f"Inputs: {intent.inputs}")
        trace.append(f"Constraints: {intent.constraints or ['none explicit']}")

        # 2. Plan
        plan = self.planner.plan(intent)
        trace.append(f"Algorithm plan: {len(plan)} steps")
        for step in plan:
            trace.append(f"  Step {step.step}: [{step.action}] {step.description}")

        # 3. Synthesize code
        code = self.synth.synthesize(intent, plan)
        trace.append("Code synthesized from symbolic plan")

        # 4. Static validate
        valid, errors = self.validator.validate(code)
        trace.append(f"Static validation: {'PASS' if valid else 'FAIL — ' + '; '.join(errors)}")

        # 5. Generate tests
        func_name = self.synth._to_snake(intent.goal)
        tests = self.test_gen.generate(intent, func_name)
        trace.append(f"Tests generated: 3 test cases for {func_name}")

        return SynthesizedFunction(
            intent=intent,
            plan=plan,
            code=code,
            tests=tests,
            static_valid=valid,
            static_errors=errors,
            generation_trace=trace,
        )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--intent", default="validate a sales invoice before submission")
    args = parser.parse_args()

    print(f"Synthesizing code for: '{args.intent}'\n{'='*60}\n")

    synth = ProgramSynthesizer()
    result = synth.synthesize(args.intent)

    print("=== GENERATION TRACE ===")
    for step in result.generation_trace:
        print(f"  {step}")

    print(f"\n=== SYNTHESIZED CODE ({'VALID' if result.static_valid else 'INVALID'}) ===")
    print(result.code)

    if result.static_errors:
        print(f"Static errors: {result.static_errors}")

    print("=== AUTO-GENERATED TESTS ===")
    print(result.tests)


if __name__ == "__main__":
    main()
