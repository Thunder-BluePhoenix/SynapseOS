"""
Phase 5 — ARIEL: Business Logic Specialist.

ARIEL reasons about ERP workflows, DocType relationships, and operational
semantics. It is the agent that understands *what the system does* at the
business level, not just what data it holds.

Capabilities:
  1. Workflow analysis    — states, transitions, guards for a DocType
  2. Dependency mapping  — what this DocType links to / is linked from
  3. Business rule audit — all rules that apply to a DocType + event
  4. Operation planning  — step-by-step plan to submit/cancel/validate
  5. Field requirement   — which fields are mandatory and under what condition

All outputs are grounded in the knowledge graph and extracted rules —
ARIEL cannot hallucinate APIs that don't exist in the domain corpus.

Run:
    python -m synapseos.phase5.agents.ariel
    python -m synapseos.phase5.agents.ariel --query "plan the submission steps for a Sales Invoice"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any

from synapseos.phase5.agents.base import SubAgent, AgentResult, AgentCapability
from synapseos.phase3.neurosymbolic_core import KnowledgeGraph, RulesEngine


# ---------------------------------------------------------------------------
# Capability declaration
# ---------------------------------------------------------------------------

ARIEL_CAPABILITY = AgentCapability(
    name="ARIEL",
    keywords=[
        "workflow", "submit", "cancel", "amend", "approve", "reject",
        "business rule", "operation", "plan", "steps", "process",
        "dependency", "link", "related", "doctype", "document",
        "mandatory", "required field", "validate", "on submit",
        "on cancel", "before save", "automate", "automation",
        "erp", "frappe", "accounting", "invoice", "order", "stock",
    ],
    anti_keywords=["security", "permission", "password", "token", "exploit", "audit trail"],
    base_confidence=0.55,
    description="ERP business logic — workflows, rules, dependencies, operation planning",
)


# ---------------------------------------------------------------------------
# ARIEL internal data structures
# ---------------------------------------------------------------------------

@dataclass
class WorkflowAnalysis:
    doctype: str
    is_submittable: bool
    is_child_table: bool
    linked_doctypes: list[str]
    reverse_links: list[str]         # DocTypes that link TO this one
    validate_rules: list[str]
    submit_rules: list[str]
    cancel_rules: list[str]
    rule_count: int

    def summary(self) -> str:
        lines = [
            f"DocType: {self.doctype}",
            f"Submittable: {self.is_submittable} | Child table: {self.is_child_table}",
            f"Links to: {', '.join(self.linked_doctypes[:5]) or 'none'}",
            f"Linked from: {', '.join(self.reverse_links[:5]) or 'none'}",
            f"Total rules: {self.rule_count}",
        ]
        if self.validate_rules:
            lines.append(f"Validate rules ({len(self.validate_rules)}): {self.validate_rules[0][:80]}...")
        if self.submit_rules:
            lines.append(f"Submit rules ({len(self.submit_rules)}): {self.submit_rules[0][:80]}...")
        return "\n".join(lines)


@dataclass
class OperationPlan:
    doctype: str
    operation: str      # submit | cancel | validate | amend
    steps: list[str]
    rules_referenced: list[str]
    warnings: list[str]

    def to_text(self) -> str:
        lines = [f"Operation: {self.operation.upper()} — {self.doctype}", ""]
        for i, step in enumerate(self.steps, 1):
            lines.append(f"  {i}. {step}")
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ARIEL Agent
# ---------------------------------------------------------------------------

class ARIELAgent(SubAgent):
    """
    ARIEL — business logic specialist.

    Grounded entirely in the knowledge graph and extracted rules.
    No token prediction. Every output cites which rules or graph edges
    produced it.
    """

    capability = ARIEL_CAPABILITY

    # Workflow states — standard Frappe lifecycle
    WORKFLOW_STATES = {
        True:  ["Draft", "Pending Approval", "Approved", "Submitted", "Cancelled"],
        False: ["Draft", "Saved"],
    }

    OPERATION_EVENTS = {
        "submit":   ["validate", "before_submit", "on_submit"],
        "cancel":   ["before_cancel", "on_cancel"],
        "validate": ["validate", "before_save"],
        "amend":    ["validate", "before_submit", "on_submit"],
        "save":     ["validate", "before_save", "on_update"],
    }

    def __init__(
        self,
        kg_path: str = "data/graph/knowledge_graph.json",
        rules_path: str = "data/rules/rules.json",
    ):
        self._kg = KnowledgeGraph(kg_path)
        self._rules = RulesEngine(rules_path)
        self._reverse_index = self._build_reverse_index()

    def _build_reverse_index(self) -> dict[str, list[str]]:
        """DocType → list of DocTypes that link to it."""
        index: dict[str, list[str]] = {}
        for name in self._kg._nodes:
            for edge in self._kg._adjacency.get(name, []):
                tgt = edge.get("target", "")
                if tgt:
                    if tgt not in index:
                        index[tgt] = []
                    if name not in index[tgt]:
                        index[tgt].append(name)
        return index

    # ------------------------------------------------------------------
    # Core analysis methods
    # ------------------------------------------------------------------

    def analyze_workflow(self, doctype: str) -> WorkflowAnalysis:
        node = self._kg.get_entity(doctype) or {}
        is_submittable = bool(node.get("is_submittable", False))
        is_child = bool(node.get("is_child_table", False))

        linked = self._kg.get_related_doctypes(doctype)
        reverse = self._reverse_index.get(doctype, [])

        all_rules = self._rules.get_rules(doctype)
        validate_rules = [self._rule_summary(r) for r in self._rules.get_rules(doctype, "validate")]
        submit_rules   = [self._rule_summary(r) for r in self._rules.get_rules(doctype, "on_submit")]
        cancel_rules   = [self._rule_summary(r) for r in self._rules.get_rules(doctype, "on_cancel")]

        return WorkflowAnalysis(
            doctype=doctype,
            is_submittable=is_submittable,
            is_child_table=is_child,
            linked_doctypes=linked,
            reverse_links=reverse[:10],
            validate_rules=validate_rules,
            submit_rules=submit_rules,
            cancel_rules=cancel_rules,
            rule_count=len(all_rules),
        )

    def plan_operation(self, doctype: str, operation: str) -> OperationPlan:
        operation = operation.lower().strip()
        if operation not in self.OPERATION_EVENTS:
            operation = "validate"

        events = self.OPERATION_EVENTS[operation]
        steps: list[str] = []
        rules_referenced: list[str] = []
        warnings: list[str] = []

        node = self._kg.get_entity(doctype) or {}
        is_submittable = bool(node.get("is_submittable"))

        # Pre-operation checks
        steps.append(f"Verify {doctype} document exists and is in correct state")
        if operation in ("submit", "amend") and not is_submittable:
            warnings.append(f"{doctype} is not a submittable DocType — submit may not be valid")

        # Event-by-event steps from rules
        for event in events:
            rules = self._rules.get_rules(doctype, event)
            if rules:
                steps.append(f"Run {event} hooks ({len(rules)} rules):")
                for rule in rules[:4]:
                    summary = self._rule_summary(rule)
                    steps.append(f"  → {summary}")
                    rules_referenced.append(f"{doctype}:{event}:{rule.get('rule_type', '?')}")
                if len(rules) > 4:
                    steps.append(f"  → … and {len(rules)-4} more rules")
            else:
                steps.append(f"Run {event} hooks (no extracted rules — check controller logic)")

        # Post-operation
        if operation == "submit":
            linked = self._kg.get_related_doctypes(doctype)
            if linked:
                steps.append(f"Trigger linked document updates: {', '.join(linked[:3])}")
            steps.append(f"Set document status to 'Submitted'")
        elif operation == "cancel":
            steps.append(f"Reverse all GL/stock entries created on submit")
            steps.append(f"Set document status to 'Cancelled'")
        elif operation == "amend":
            steps.append(f"Create amended copy with docstatus=0")
            steps.append(f"Link amended doc to original via 'amended_from' field")

        if not rules_referenced:
            warnings.append(f"No rules extracted for {doctype} — manual controller review needed")

        return OperationPlan(
            doctype=doctype,
            operation=operation,
            steps=steps,
            rules_referenced=rules_referenced,
            warnings=warnings,
        )

    def get_mandatory_fields(self, doctype: str) -> list[dict[str, str]]:
        """Return fields that must not be empty, grounded in extracted rules."""
        fields = []
        for rule in self._rules.get_rules(doctype):
            if rule.get("rule_type") == "constraint":
                consequence = rule.get("consequence", "")
                if "must not be empty" in consequence:
                    fields.append({
                        "consequence": consequence,
                        "event":       rule.get("event", "?"),
                        "condition":   rule.get("condition", "always"),
                    })
        return fields

    def find_doctype(self, query: str) -> str:
        """Best-effort extraction of a DocType name from a query string."""
        # Try exact KG node name match (longest first)
        q_lower = query.lower()
        candidates = []
        for name in self._kg._nodes:
            if name.lower() in q_lower:
                candidates.append(name)
        if candidates:
            return max(candidates, key=len)

        # Fallback: search KG
        results = self._kg.search(query, limit=1)
        return results[0]["name"] if results else "Unknown"

    def find_operation(self, query: str) -> str:
        """Extract intended operation from query."""
        q = query.lower()
        for op in ("submit", "cancel", "amend", "validate", "save"):
            if op in q:
                return op
        return "validate"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rule_summary(rule: dict) -> str:
        rt = rule.get("rule_type", "?")
        cond = rule.get("condition", "always")
        cons = rule.get("consequence", "")[:80]
        return f"[{rt}] if {cond}: {cons}"

    # ------------------------------------------------------------------
    # SubAgent interface
    # ------------------------------------------------------------------

    def _handle(self, query: str, context: Any) -> AgentResult:
        trace: list[str] = [f"ARIEL received: {query[:80]}"]

        doctype = self.find_doctype(query)
        operation = self.find_operation(query)
        trace.append(f"Resolved DocType: {doctype} | Operation: {operation}")

        q_lower = query.lower()

        # Route to appropriate capability
        if any(kw in q_lower for kw in ("plan", "steps", "how to", "submit", "cancel", "amend")):
            plan = self.plan_operation(doctype, operation)
            output = plan.to_text()
            structured = {
                "doctype":    doctype,
                "operation":  operation,
                "steps":      plan.steps,
                "warnings":   plan.warnings,
                "rules_used": plan.rules_referenced,
            }
            trace.append(f"Generated operation plan: {len(plan.steps)} steps, {len(plan.rules_referenced)} rules")

        elif any(kw in q_lower for kw in ("dependency", "link", "related", "workflow", "analyze")):
            analysis = self.analyze_workflow(doctype)
            output = analysis.summary()
            structured = {
                "doctype":         doctype,
                "is_submittable":  analysis.is_submittable,
                "linked_doctypes": analysis.linked_doctypes,
                "reverse_links":   analysis.reverse_links,
                "rule_count":      analysis.rule_count,
            }
            trace.append(f"Workflow analysis: {analysis.rule_count} rules, {len(analysis.linked_doctypes)} links")

        elif any(kw in q_lower for kw in ("mandatory", "required", "field", "must")):
            fields = self.get_mandatory_fields(doctype)
            if fields:
                lines = [f"Mandatory fields for {doctype}:"]
                for f in fields[:10]:
                    lines.append(f"  [{f['event']}] {f['consequence']}")
                output = "\n".join(lines)
            else:
                output = f"No mandatory field constraints extracted for {doctype}. Check DocType definition directly."
            structured = {"doctype": doctype, "mandatory_fields": fields[:10]}
            trace.append(f"Mandatory field lookup: {len(fields)} constraints found")

        else:
            # Default: full workflow analysis + operation plan
            analysis = self.analyze_workflow(doctype)
            plan     = self.plan_operation(doctype, operation)
            output   = f"{analysis.summary()}\n\n{plan.to_text()}"
            structured = {
                "doctype":        doctype,
                "rule_count":     analysis.rule_count,
                "linked":         analysis.linked_doctypes,
                "operation_plan": plan.steps,
            }
            trace.append("Combined workflow analysis + operation plan")

        # Confidence: higher if we found the doctype and have rules
        all_rules = self._rules.get_rules(doctype)
        entity    = self._kg.get_entity(doctype)
        confidence = 0.5
        if entity:
            confidence += 0.2
        if all_rules:
            confidence += min(0.3, 0.05 * len(all_rules))

        trace.append(f"Confidence: {confidence:.2f} ({len(all_rules)} rules, entity_found={entity is not None})")

        return AgentResult(
            agent_name="ARIEL",
            query=query,
            output=output,
            structured=structured,
            confidence=min(1.0, confidence),
            latency_ms=0.0,   # set by base handle()
            traceable=True,
            trace=trace,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ARIEL — Business Logic Agent")
    parser.add_argument("--query", default="plan the submission steps for a Sales Invoice")
    args = parser.parse_args()

    agent = ARIELAgent()
    print(f"\nARIEL — Business Logic Specialist")
    print(f"Query: {args.query}\n{'='*60}\n")

    result = agent.handle(args.query)

    print(result.output)
    print(f"\nConfidence : {result.confidence:.3f}")
    print(f"Latency    : {result.latency_ms:.1f}ms")
    print(f"Traceable  : {result.traceable}")
    print(f"\nTrace:")
    for step in result.trace:
        print(f"  {step}")


if __name__ == "__main__":
    main()
