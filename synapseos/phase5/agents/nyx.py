"""
Phase 5 — NYX: Security Specialist.

NYX reasons about security surfaces, access controls, sensitive data
exposure, and audit requirements in the ERP domain.

Capabilities:
  1. Sensitive field detection  — fields likely containing PII/secrets/credentials
  2. Audit trail specification  — what must be logged for a given operation
  3. Access control review      — map permission model for a DocType
  4. Security surface analysis  — submittable docs without validation rules (risk)
  5. Anomaly detection          — DocTypes with high connectivity but no rules

NYX never modifies anything — it only observes and reports.
All findings are grounded in the knowledge graph and extracted rules.

Run:
    python -m synapseos.phase5.agents.nyx
    python -m synapseos.phase5.agents.nyx --query "audit the permission model for Sales Invoice"
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from typing import Any

from synapseos.phase5.agents.base import SubAgent, AgentResult, AgentCapability
from synapseos.phase3.neurosymbolic_core import KnowledgeGraph, RulesEngine


# ---------------------------------------------------------------------------
# Capability declaration
# ---------------------------------------------------------------------------

NYX_CAPABILITY = AgentCapability(
    name="NYX",
    keywords=[
        "security", "permission", "audit", "access", "sensitive",
        "password", "token", "secret", "key", "credential", "ssn",
        "pii", "personal data", "gdpr", "compliance", "vulnerability",
        "risk", "exposure", "trail", "log", "monitor", "harden",
        "control", "privilege", "role", "user", "restrict",
        "encrypt", "hash", "mask", "redact",
    ],
    anti_keywords=["workflow", "submit plan", "business rule"],
    base_confidence=0.60,
    description="Security audit — sensitive fields, permissions, audit trails, risk surface",
)


# ---------------------------------------------------------------------------
# NYX internal patterns
# ---------------------------------------------------------------------------

# Field name patterns that indicate sensitive data
SENSITIVE_PATTERNS: list[tuple[str, str, str]] = [
    # (regex pattern, risk level, category)
    (r"password",          "CRITICAL", "credential"),
    (r"api[_\s]?key",      "CRITICAL", "credential"),
    (r"secret",            "CRITICAL", "credential"),
    (r"token",             "HIGH",     "credential"),
    (r"private[_\s]?key",  "CRITICAL", "credential"),
    (r"ssn|social[_\s]?security", "CRITICAL", "PII"),
    (r"pan|card[_\s]?number",     "CRITICAL", "PII"),
    (r"bank[_\s]?account",        "HIGH",     "financial"),
    (r"salary|wage|compensation",  "HIGH",     "PII"),
    (r"dob|date[_\s]?of[_\s]?birth", "HIGH",  "PII"),
    (r"phone|mobile|contact",      "MEDIUM",   "PII"),
    (r"email",                     "MEDIUM",   "PII"),
    (r"address|street|city|zip",   "MEDIUM",   "PII"),
    (r"aadhaar|passport|national[_\s]?id", "CRITICAL", "identity"),
    (r"two[_\s]?factor|2fa|otp",   "HIGH",     "auth"),
    (r"webhook[_\s]?secret",       "HIGH",     "credential"),
    (r"encryption[_\s]?key",       "CRITICAL", "credential"),
]

# Events that always require audit logging
ALWAYS_AUDIT_EVENTS = {"on_submit", "on_cancel", "before_cancel"}

# Minimum rule count threshold — below this for a submittable doc is a risk
MIN_RULES_FOR_SUBMITTABLE = 2


@dataclass
class SensitiveFieldFinding:
    field_name: str
    risk_level: str      # CRITICAL | HIGH | MEDIUM | LOW
    category: str        # credential | PII | financial | auth | identity
    doctype: str
    matched_pattern: str


@dataclass
class AuditSpec:
    doctype: str
    operation: str
    must_log: list[str]       # fields / events that must be captured
    should_log: list[str]     # recommended
    risk_events: list[str]    # rules that could alter sensitive state


@dataclass
class SecurityReport:
    doctype: str
    sensitive_fields: list[SensitiveFieldFinding]
    risk_level: str           # CRITICAL | HIGH | MEDIUM | LOW | OK
    findings: list[str]
    recommendations: list[str]
    rule_count: int
    is_submittable: bool

    def summary(self) -> str:
        lines = [
            f"Security Report — {self.doctype}",
            f"Risk level    : {self.risk_level}",
            f"Is submittable: {self.is_submittable}",
            f"Rules found   : {self.rule_count}",
            f"Sensitive fields: {len(self.sensitive_fields)}",
            "",
        ]
        if self.findings:
            lines.append("Findings:")
            for f in self.findings:
                lines.append(f"  ⚠ {f}")
        if self.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for r in self.recommendations:
                lines.append(f"  → {r}")
        if self.sensitive_fields:
            lines.append("")
            lines.append("Sensitive fields:")
            for sf in self.sensitive_fields[:8]:
                lines.append(f"  [{sf.risk_level}] {sf.field_name} ({sf.category})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# NYX Agent
# ---------------------------------------------------------------------------

class NYXAgent(SubAgent):
    """
    NYX — security specialist.

    Analyses the ERP domain for security risks, access control gaps,
    sensitive data exposure, and audit logging requirements.
    All findings are deterministic and traceable to the KG + rules corpus.
    """

    capability = NYX_CAPABILITY

    def __init__(
        self,
        kg_path: str = "data/graph/knowledge_graph.json",
        rules_path: str = "data/rules/rules.json",
    ):
        self._kg    = KnowledgeGraph(kg_path)
        self._rules = RulesEngine(rules_path)
        # Compile patterns once
        self._compiled = [
            (re.compile(pat, re.IGNORECASE), risk, cat)
            for pat, risk, cat in SENSITIVE_PATTERNS
        ]

    # ------------------------------------------------------------------
    # Sensitive field detection
    # ------------------------------------------------------------------

    def detect_sensitive_fields(self, doctype: str) -> list[SensitiveFieldFinding]:
        """
        Scan all rules and the node's attribute keys for sensitive field names.
        """
        findings: list[SensitiveFieldFinding] = []
        seen: set[str] = set()

        # Scan from rules (consequence text often contains field names)
        for rule in self._rules.get_rules(doctype):
            consequence = rule.get("consequence", "")
            # Extract field names in parentheses — format: "field 'Label' (fieldname)"
            for match in re.finditer(r"\(([^)]+)\)", consequence):
                fname = match.group(1)
                if fname in seen:
                    continue
                hit = self._check_sensitive(fname, doctype)
                if hit:
                    findings.append(hit)
                    seen.add(fname)

        # Scan from KG node attributes
        node = self._kg.get_entity(doctype) or {}
        for key in node:
            if key in seen:
                continue
            hit = self._check_sensitive(key, doctype)
            if hit:
                findings.append(hit)
                seen.add(key)

        # Sort by risk level
        risk_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        findings.sort(key=lambda f: risk_order.get(f.risk_level, 4))
        return findings

    def _check_sensitive(self, field_name: str, doctype: str) -> SensitiveFieldFinding | None:
        for pattern, risk, cat in self._compiled:
            if pattern.search(field_name):
                return SensitiveFieldFinding(
                    field_name=field_name,
                    risk_level=risk,
                    category=cat,
                    doctype=doctype,
                    matched_pattern=pattern.pattern,
                )
        return None

    # ------------------------------------------------------------------
    # Audit trail specification
    # ------------------------------------------------------------------

    def generate_audit_spec(self, doctype: str, operation: str) -> AuditSpec:
        op = operation.lower()

        must_log = [
            f"user who performed {op}",
            "timestamp (UTC)",
            f"{doctype} document name/ID",
            "document status before and after",
        ]
        should_log = [
            "IP address of request origin",
            "session ID",
            "changed field values (before/after diff)",
        ]
        risk_events: list[str] = []

        # Add operation-specific mandatory items
        if op in ("submit", "cancel", "amend"):
            must_log.append(f"docstatus change ({op})")

        # Check for rules on sensitive events
        for event in ALWAYS_AUDIT_EVENTS:
            rules = self._rules.get_rules(doctype, event)
            for rule in rules:
                consequence = rule.get("consequence", "")
                if any(kw in consequence.lower() for kw in ("amount", "total", "balance", "ledger")):
                    risk_events.append(f"{event}: {consequence[:60]}")

        # If the DocType has sensitive fields, add them to must-log
        sens_fields = self.detect_sensitive_fields(doctype)
        critical = [sf.field_name for sf in sens_fields if sf.risk_level == "CRITICAL"]
        if critical:
            must_log.append(f"Access to sensitive fields: {', '.join(critical[:3])}")
            should_log.append("Mask sensitive field values in logs (never log raw)")

        return AuditSpec(
            doctype=doctype,
            operation=op,
            must_log=must_log,
            should_log=should_log,
            risk_events=risk_events[:5],
        )

    # ------------------------------------------------------------------
    # Full security report
    # ------------------------------------------------------------------

    def security_report(self, doctype: str) -> SecurityReport:
        node      = self._kg.get_entity(doctype) or {}
        all_rules = self._rules.get_rules(doctype)
        is_sub    = bool(node.get("is_submittable"))
        linked    = self._kg.get_related_doctypes(doctype)

        findings: list[str] = []
        recommendations: list[str] = []

        # Risk 1: submittable with no rules
        if is_sub and len(all_rules) < MIN_RULES_FOR_SUBMITTABLE:
            findings.append(
                f"{doctype} is submittable but has fewer than {MIN_RULES_FOR_SUBMITTABLE} "
                f"extracted validation rules — submit path may be unguarded"
            )
            recommendations.append("Add explicit validate + before_submit controllers")

        # Risk 2: sensitive fields
        sens_fields = self.detect_sensitive_fields(doctype)
        for sf in sens_fields:
            if sf.risk_level == "CRITICAL":
                findings.append(f"Critical sensitive field detected: '{sf.field_name}' ({sf.category})")
                recommendations.append(
                    f"Ensure '{sf.field_name}' is encrypted at rest and never included in API responses"
                )
            elif sf.risk_level == "HIGH":
                findings.append(f"High-risk field: '{sf.field_name}' ({sf.category})")
                recommendations.append(f"Restrict read access to '{sf.field_name}' to privileged roles only")

        # Risk 3: high connectivity without rules (large attack surface)
        if len(linked) > 10 and len(all_rules) < 3:
            findings.append(
                f"{doctype} has {len(linked)} outbound links but only {len(all_rules)} rules — "
                f"high connectivity creates large attack surface"
            )
            recommendations.append("Audit each linked DocType's access permissions independently")

        # Risk 4: no findings but submittable — basic recommendation
        if not findings and is_sub:
            recommendations.append("Enable amendment log for all submitted documents")
            recommendations.append("Ensure role-based permissions are set in DocType permissions table")

        # Determine overall risk level
        critical_fields = [sf for sf in sens_fields if sf.risk_level == "CRITICAL"]
        high_fields     = [sf for sf in sens_fields if sf.risk_level == "HIGH"]

        if critical_fields or (is_sub and len(all_rules) < MIN_RULES_FOR_SUBMITTABLE):
            risk_level = "CRITICAL"
        elif high_fields or len(linked) > 10 and len(all_rules) < 3:
            risk_level = "HIGH"
        elif sens_fields:
            risk_level = "MEDIUM"
        elif findings:
            risk_level = "LOW"
        else:
            risk_level = "OK"

        return SecurityReport(
            doctype=doctype,
            sensitive_fields=sens_fields,
            risk_level=risk_level,
            findings=findings,
            recommendations=recommendations,
            rule_count=len(all_rules),
            is_submittable=is_sub,
        )

    # ------------------------------------------------------------------
    # Surface scan — find highest-risk DocTypes globally
    # ------------------------------------------------------------------

    def scan_high_risk_doctypes(self, top_k: int = 5) -> list[tuple[str, str]]:
        """
        Scan all DocTypes and return those with the highest risk level.
        Returns list of (doctype, risk_level) sorted CRITICAL → OK.
        """
        risk_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "OK": 4}
        results: list[tuple[str, str]] = []

        submittable_nodes = [
            name for name, node in self._kg._nodes.items()
            if node.get("is_submittable")
        ]

        for doctype in submittable_nodes[:50]:   # sample to keep it fast
            report = self.security_report(doctype)
            if report.risk_level in ("CRITICAL", "HIGH"):
                results.append((doctype, report.risk_level))

        results.sort(key=lambda x: risk_order.get(x[1], 5))
        return results[:top_k]

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _find_doctype(self, query: str) -> str:
        q_lower = query.lower()
        candidates = []
        for name in self._kg._nodes:
            if name.lower() in q_lower:
                candidates.append(name)
        if candidates:
            return max(candidates, key=len)
        results = self._kg.search(query, limit=1)
        return results[0]["name"] if results else "Unknown"

    # ------------------------------------------------------------------
    # SubAgent interface
    # ------------------------------------------------------------------

    def _handle(self, query: str, context: Any) -> AgentResult:
        trace: list[str] = [f"NYX received: {query[:80]}"]

        q_lower = query.lower()

        # Surface-wide scan — no specific doctype needed
        if any(kw in q_lower for kw in ("scan", "surface", "high risk", "top risk", "overview")):
            high_risk = self.scan_high_risk_doctypes(top_k=8)
            if high_risk:
                lines = ["High-risk DocTypes (submittable with security gaps):"]
                for dt, level in high_risk:
                    lines.append(f"  [{level}] {dt}")
                output = "\n".join(lines)
            else:
                output = "No critical or high-risk DocTypes detected in surface scan."
            structured = {"high_risk_doctypes": [{"doctype": d, "risk": r} for d, r in high_risk]}
            trace.append(f"Surface scan: {len(high_risk)} high-risk DocTypes found")
            confidence = 0.80

        else:
            doctype = self._find_doctype(query)
            trace.append(f"Resolved DocType: {doctype}")

            if any(kw in q_lower for kw in ("audit", "log", "trail", "monitor")):
                op = "submit" if "submit" in q_lower else "save"
                spec = self.generate_audit_spec(doctype, op)
                lines = [f"Audit Trail Specification — {doctype} ({op.upper()})", ""]
                lines.append("MUST log:")
                for item in spec.must_log:
                    lines.append(f"  ✓ {item}")
                if spec.should_log:
                    lines.append("\nSHOULD log:")
                    for item in spec.should_log:
                        lines.append(f"  ○ {item}")
                if spec.risk_events:
                    lines.append("\nRisk events requiring special attention:")
                    for ev in spec.risk_events:
                        lines.append(f"  ⚠ {ev}")
                output = "\n".join(lines)
                structured = {
                    "doctype":     doctype,
                    "operation":   op,
                    "must_log":    spec.must_log,
                    "should_log":  spec.should_log,
                    "risk_events": spec.risk_events,
                }
                trace.append(f"Audit spec generated: {len(spec.must_log)} must-log items")

            elif any(kw in q_lower for kw in ("sensitive", "field", "pii", "password", "credential")):
                fields = self.detect_sensitive_fields(doctype)
                if fields:
                    lines = [f"Sensitive field analysis — {doctype}:"]
                    for sf in fields:
                        lines.append(f"  [{sf.risk_level}] {sf.field_name} ({sf.category})")
                else:
                    lines = [f"No sensitive fields detected for {doctype} in extracted rules/attributes."]
                output = "\n".join(lines)
                structured = {
                    "doctype":          doctype,
                    "sensitive_fields": [{"field": sf.field_name, "risk": sf.risk_level,
                                         "category": sf.category} for sf in fields],
                }
                trace.append(f"Sensitive field scan: {len(fields)} findings")

            else:
                # Full security report (default)
                report = self.security_report(doctype)
                output = report.summary()
                structured = {
                    "doctype":          doctype,
                    "risk_level":       report.risk_level,
                    "findings":         report.findings,
                    "recommendations":  report.recommendations,
                    "rule_count":       report.rule_count,
                    "sensitive_fields": len(report.sensitive_fields),
                }
                trace.append(f"Full security report: risk={report.risk_level}, {len(report.findings)} findings")

            # Confidence: higher when we found the doctype
            entity = self._kg.get_entity(doctype)
            confidence = 0.6 if entity else 0.35
            if doctype != "Unknown":
                confidence += 0.15
            confidence = min(1.0, confidence)

        trace.append(f"NYX complete | confidence={confidence:.2f}")
        return AgentResult(
            agent_name="NYX",
            query=query,
            output=output,
            structured=structured,
            confidence=confidence,
            latency_ms=0.0,
            traceable=True,
            trace=trace,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NYX — Security Agent")
    parser.add_argument("--query", default="audit the permission model for Sales Invoice")
    args = parser.parse_args()

    agent = NYXAgent()
    print(f"\nNYX — Security Specialist")
    print(f"Query: {args.query}\n{'='*60}\n")

    result = agent.handle(args.query)
    print(result.output)
    print(f"\nConfidence : {result.confidence:.3f}")
    print(f"Latency    : {result.latency_ms:.1f}ms")
    print(f"\nTrace:")
    for step in result.trace:
        print(f"  {step}")


if __name__ == "__main__":
    main()
