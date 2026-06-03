"""
Phase 3 — Output Synthesizer SFT Data Generator.

Generates (instruction, response) pairs for supervised fine-tuning (SFT)
of the output synthesizer (Mistral 1B / Llama 3B).

Uses the existing SynapseOS neurosymbolic pipeline as the teacher —
every response is grounded in the KG and extracted rules, not hallucinated.

Data format (Alpaca-style JSONL):
    {"instruction": "...", "input": "", "output": "..."}

Categories of pairs generated:
  1. Field requirement queries    — "What fields are required for X?"
  2. Validation rule queries      — "What validate rules apply to X?"
  3. Workflow/submission queries  — "What are the steps to submit X?"
  4. Relationship queries         — "What does X link to?"
  5. Code generation queries      — "Write a validate function for X."
  6. Business rule explanations   — "Explain the rule: [rule text]"
  7. Comparison queries           — "What is the difference between X and Y?"
  8. Error diagnosis queries      — "Why would X fail validation?"

Run:
    python -m synapseos.phase3.output_synth_data
    python -m synapseos.phase3.output_synth_data --n 1000 --out data/sft/sft_pairs.jsonl
    python -m synapseos.phase3.output_synth_data --dry-run   # 20 pairs, print only
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SFTPair:
    instruction: str
    input: str
    output: str
    category: str
    doctype: str
    quality: float      # 0–1 estimated quality of the output

    def to_dict(self) -> dict:
        return {
            "instruction": self.instruction,
            "input":       self.input,
            "output":      self.output,
            "metadata": {
                "category": self.category,
                "doctype":  self.doctype,
                "quality":  round(self.quality, 3),
            },
        }


# ---------------------------------------------------------------------------
# Query templates per category
# ---------------------------------------------------------------------------

FIELD_REQUIREMENT_TEMPLATES = [
    "What fields are required for a {doctype}?",
    "Which fields must not be empty in a {doctype} document?",
    "List the mandatory fields for the {doctype} doctype.",
    "What information is needed to create a {doctype}?",
    "What are the required inputs for submitting a {doctype}?",
]

VALIDATION_RULE_TEMPLATES = [
    "What validate rules apply to a {doctype}?",
    "What checks happen when a {doctype} is saved?",
    "What validations run before a {doctype} is submitted?",
    "Describe the business constraints on the {doctype} document.",
    "What conditions must be true for a {doctype} to pass validation?",
]

WORKFLOW_TEMPLATES = [
    "What are the steps to submit a {doctype}?",
    "Describe the lifecycle of a {doctype} document.",
    "How does a {doctype} move from draft to submitted?",
    "What happens on submission of a {doctype}?",
    "Explain the {doctype} workflow from creation to cancellation.",
]

RELATIONSHIP_TEMPLATES = [
    "What documents does a {doctype} link to?",
    "What other DocTypes depend on {doctype}?",
    "Describe the relationships between {doctype} and other documents.",
    "What child tables does {doctype} contain?",
    "Which DocTypes are linked from {doctype}?",
]

CODE_TEMPLATES = [
    "Write a validate function for the {doctype} doctype.",
    "Implement an on_submit handler for {doctype}.",
    "Generate a before_save hook for {doctype}.",
    "Write Python code to validate a {doctype} document.",
    "Create a controller method that enforces rules on {doctype}.",
]

RULE_EXPLANATION_TEMPLATES = [
    "Explain this business rule: {rule}",
    "What does this constraint mean? {rule}",
    "Describe the business intent of: {rule}",
    "When does this rule apply? {rule}",
]

COMPARISON_TEMPLATES = [
    "What is the difference between {doctype_a} and {doctype_b}?",
    "Compare {doctype_a} and {doctype_b} in terms of purpose and usage.",
    "When would you use a {doctype_a} instead of a {doctype_b}?",
    "How do {doctype_a} and {doctype_b} relate to each other?",
]

ERROR_TEMPLATES = [
    "Why would a {doctype} fail validation?",
    "What errors can occur when submitting a {doctype}?",
    "List common validation errors in a {doctype}.",
    "What causes a {doctype} to be rejected on submit?",
]


# ---------------------------------------------------------------------------
# Response generator (uses existing pipeline as teacher)
# ---------------------------------------------------------------------------

class SFTDataGenerator:
    """
    Generates SFT pairs using the neurosymbolic pipeline as teacher.
    All responses are grounded in KG facts and extracted rules.
    """

    def __init__(
        self,
        kg_path: str = "data/graph/knowledge_graph.json",
        rules_path: str = "data/rules/rules.json",
        seed: int = 42,
    ):
        random.seed(seed)
        self._kg      = None
        self._rules   = None
        self._synth   = None
        self._kg_path    = kg_path
        self._rules_path = rules_path
        self._doctype_list: list[str] = []
        self._rule_samples: list[dict] = []
        self._load_data()

    def _load_data(self) -> None:
        from synapseos.phase3.neurosymbolic_core import KnowledgeGraph, RulesEngine
        self._kg    = KnowledgeGraph(self._kg_path)
        self._rules = RulesEngine(self._rules_path)

        # Build doctype list (real named DocTypes from KG)
        self._doctype_list = [
            name for name, node in self._kg._nodes.items()
            if not name.startswith("__") and "scheduler" not in name.lower()
            and len(name) > 3
        ]
        # Sample of rules for explanation pairs
        all_rules = []
        for dt_rules in list(self._rules._by_doctype.values())[:50]:
            all_rules.extend(dt_rules[:3])
        self._rule_samples = all_rules[:200]

        print(f"[SFTData] {len(self._doctype_list)} doctypes, {len(self._rule_samples)} sample rules")

    def _get_doctype(self) -> str:
        return random.choice(self._doctype_list)

    def _get_rule_text(self) -> str:
        if not self._rule_samples:
            return "field must not be empty"
        rule = random.choice(self._rule_samples)
        return rule.get("consequence", "field must not be empty")[:120]

    # ------------------------------------------------------------------
    # Response generators per category
    # ------------------------------------------------------------------

    def _field_requirement_response(self, doctype: str) -> tuple[str, float]:
        rules = self._rules.get_rules(doctype)
        mandatory = [
            r for r in rules
            if r.get("rule_type") == "constraint" and "must not be empty" in r.get("consequence", "")
        ]
        links = self._kg.get_related_doctypes(doctype)
        node  = self._kg.get_entity(doctype) or {}

        parts = [f"For a **{doctype}** document, the following fields are required:"]
        if mandatory:
            for r in mandatory[:6]:
                cons = r.get("consequence", "")
                # Extract field name
                import re
                m = re.search(r"'([^']+)'", cons)
                field = m.group(1) if m else cons[:40]
                parts.append(f"- **{field}** — {r.get('condition', 'always')}")
        else:
            parts.append("- No mandatory fields extracted from rules. Check the DocType definition for `reqd=1` fields.")

        if links:
            parts.append(f"\nLinked documents that may be required: {', '.join(links[:4])}")

        quality = 0.85 if mandatory else 0.55
        return "\n".join(parts), quality

    def _validation_rule_response(self, doctype: str) -> tuple[str, float]:
        rules = self._rules.get_rules(doctype, "validate")
        all_rules = self._rules.get_rules(doctype)

        parts = [f"The following validation rules apply to **{doctype}**:"]
        if rules:
            for r in rules[:8]:
                rt   = r.get("rule_type", "constraint")
                cond = r.get("condition", "always")
                cons = r.get("consequence", "")[:80]
                parts.append(f"\n**[{rt}]** if `{cond}`:")
                parts.append(f"  → {cons}")
        else:
            parts.append(f"\nNo validate-event rules extracted for {doctype}.")
            if all_rules:
                parts.append(f"({len(all_rules)} rules exist across other events.)")

        quality = min(0.95, 0.5 + len(rules) * 0.08)
        return "\n".join(parts), quality

    def _workflow_response(self, doctype: str) -> tuple[str, float]:
        node      = self._kg.get_entity(doctype) or {}
        is_sub    = node.get("is_submittable", False)
        rules_val = self._rules.get_rules(doctype, "validate")
        rules_sub = self._rules.get_rules(doctype, "on_submit")
        rules_can = self._rules.get_rules(doctype, "on_cancel")
        links     = self._kg.get_related_doctypes(doctype)

        parts = [f"## {doctype} Workflow\n"]
        if is_sub:
            parts += [
                "**States:** Draft → Submitted → Cancelled (if applicable)\n",
                "### Submission steps:",
                f"1. Create {doctype} in **Draft** status",
                "2. Fill all mandatory fields",
            ]
            if rules_val:
                parts.append(f"3. Run validate hooks ({len(rules_val)} rules):")
                for r in rules_val[:3]:
                    parts.append(f"   - {r.get('consequence', '')[:60]}")
            else:
                parts.append("3. Run validate hooks (check controller for logic)")
            parts.append("4. Click **Submit** → docstatus changes from 0 → 1")
            if rules_sub:
                parts.append(f"5. Run on_submit hooks ({len(rules_sub)} rules):")
                for r in rules_sub[:2]:
                    parts.append(f"   - {r.get('consequence', '')[:60]}")
            if links:
                parts.append(f"6. Trigger updates in: {', '.join(links[:3])}")
            parts += [
                "\n### Cancellation:",
                f"- Cancel → docstatus changes from 1 → 2",
                f"- {len(rules_can)} on_cancel rules apply" if rules_can else "- Standard cancellation (no special rules extracted)",
            ]
        else:
            parts += [
                "**States:** Draft → Saved (non-submittable master)\n",
                "This is a master document — it does not have a submit/cancel workflow.",
                f"It has {len(self._rules.get_rules(doctype))} validation rules.",
            ]

        quality = 0.88 if is_sub else 0.65
        return "\n".join(parts), quality

    def _relationship_response(self, doctype: str) -> tuple[str, float]:
        links = self._kg.get_related_doctypes(doctype)
        # Find reverse links (who links TO this doctype)
        reverse = [
            name for name, node in self._kg._nodes.items()
            if doctype in self._kg.get_related_doctypes(name)
        ][:8]

        parts = [f"## Relationships — {doctype}\n"]
        if links:
            parts.append(f"**Links to ({len(links)} DocTypes):**")
            for lk in links[:8]:
                parts.append(f"  - {lk}")
        else:
            parts.append("**Links to:** None found in graph")

        if reverse:
            parts.append(f"\n**Linked from ({len(reverse)} DocTypes):**")
            for rv in reverse[:6]:
                parts.append(f"  - {rv}")

        quality = 0.80 if links else 0.45
        return "\n".join(parts), quality

    def _code_response(self, doctype: str) -> tuple[str, float]:
        from synapseos.phase4.program_synthesis import ProgramSynthesizer
        if self._synth is None:
            self._synth = ProgramSynthesizer()

        query  = f"validate the {doctype} document"
        result = self._synth.synthesize(query)

        if result.code and result.static_valid:
            output = f"```python\n{result.code}\n```"
            quality = 0.92
        elif result.code:
            output = f"```python\n{result.code}\n```\n\n⚠️ Note: static analysis found issues — review before using."
            quality = 0.70
        else:
            output = (
                f"```python\ndef validate(doc, method=None):\n"
                f"    # Validate {doctype}\n"
                f"    if not doc.docstatus:\n"
                f"        # Add validation logic here\n"
                f"        pass\n```"
            )
            quality = 0.40

        return output, quality

    def _rule_explanation_response(self, rule_text: str) -> tuple[str, float]:
        parts = [
            f"**Business Rule Explanation:**\n",
            f"> {rule_text}\n",
            "**Meaning:** This rule enforces a business constraint on the document.",
            "When this condition is met, the system ensures the specified behaviour is maintained.",
            "Violating this rule will prevent the document from being saved or submitted.",
            "\n**Practical impact:** Users must satisfy this constraint before they can proceed.",
            "The rule is evaluated automatically during the validate event.",
        ]
        return "\n".join(parts), 0.70

    def _comparison_response(self, doctype_a: str, doctype_b: str) -> tuple[str, float]:
        node_a   = self._kg.get_entity(doctype_a) or {}
        node_b   = self._kg.get_entity(doctype_b) or {}
        sub_a    = node_a.get("is_submittable", False)
        sub_b    = node_b.get("is_submittable", False)
        links_a  = self._kg.get_related_doctypes(doctype_a)
        links_b  = self._kg.get_related_doctypes(doctype_b)
        rules_a  = self._rules.get_rules(doctype_a)
        rules_b  = self._rules.get_rules(doctype_b)

        parts = [
            f"## {doctype_a} vs {doctype_b}\n",
            f"| Property | {doctype_a} | {doctype_b} |",
            f"|---|---|---|",
            f"| Submittable | {'Yes' if sub_a else 'No'} | {'Yes' if sub_b else 'No'} |",
            f"| Linked to   | {len(links_a)} DocTypes | {len(links_b)} DocTypes |",
            f"| Rules       | {len(rules_a)} rules | {len(rules_b)} rules |",
            "",
        ]

        shared_links = set(links_a) & set(links_b)
        if shared_links:
            parts.append(f"**Shared links:** {', '.join(list(shared_links)[:4])}")

        diff_a = set(links_a) - set(links_b)
        diff_b = set(links_b) - set(links_a)
        if diff_a:
            parts.append(f"**Only in {doctype_a}:** {', '.join(list(diff_a)[:3])}")
        if diff_b:
            parts.append(f"**Only in {doctype_b}:** {', '.join(list(diff_b)[:3])}")

        quality = 0.75 if (shared_links or diff_a or diff_b) else 0.50
        return "\n".join(parts), quality

    def _error_response(self, doctype: str) -> tuple[str, float]:
        rules    = self._rules.get_rules(doctype, "validate")
        sub_rules = self._rules.get_rules(doctype, "on_submit")

        parts = [f"## Common {doctype} Validation Errors\n"]
        errors = []
        import re
        for r in (rules + sub_rules)[:10]:
            cons = r.get("consequence", "")
            if "must not be empty" in cons:
                m = re.search(r"'([^']+)'", cons)
                field = m.group(1) if m else "a required field"
                errors.append(f"- **Missing {field}**: This field is mandatory and cannot be blank.")
            elif "throw" in cons.lower() or "frappe.throw" in cons.lower():
                msg = cons.replace("frappe.throw:", "").strip()[:70]
                errors.append(f"- **Rule violation**: {msg}")

        if errors:
            parts.extend(errors[:6])
        else:
            parts += [
                f"- **Missing mandatory fields**: Check that all `reqd=1` fields are filled.",
                f"- **Invalid field values**: Ensure linked documents (e.g. Customer, Supplier) exist and are valid.",
                f"- **Workflow state errors**: Some operations require the document to be in a specific state.",
            ]

        quality = min(0.90, 0.50 + len(errors) * 0.08)
        return "\n".join(parts), quality

    # ------------------------------------------------------------------
    # Generator pipeline
    # ------------------------------------------------------------------

    def generate(self) -> Iterator[SFTPair]:
        """Infinite iterator — call with itertools.islice(gen.generate(), N)."""

        CATEGORIES = [
            ("field_requirement",  0.18),
            ("validation_rules",   0.18),
            ("workflow",           0.15),
            ("relationship",       0.12),
            ("code_generation",    0.15),
            ("rule_explanation",   0.08),
            ("comparison",         0.07),
            ("error_diagnosis",    0.07),
        ]
        cat_names = [c for c, _ in CATEGORIES]
        cat_weights = [w for _, w in CATEGORIES]

        while True:
            cat = random.choices(cat_names, weights=cat_weights)[0]
            try:
                pair = self._generate_pair(cat)
                if pair:
                    yield pair
            except Exception:
                continue

    def _generate_pair(self, category: str) -> SFTPair | None:
        dt = self._get_doctype()
        try:
            if category == "field_requirement":
                instr  = random.choice(FIELD_REQUIREMENT_TEMPLATES).format(doctype=dt)
                out, q = self._field_requirement_response(dt)

            elif category == "validation_rules":
                instr  = random.choice(VALIDATION_RULE_TEMPLATES).format(doctype=dt)
                out, q = self._validation_rule_response(dt)

            elif category == "workflow":
                instr  = random.choice(WORKFLOW_TEMPLATES).format(doctype=dt)
                out, q = self._workflow_response(dt)

            elif category == "relationship":
                instr  = random.choice(RELATIONSHIP_TEMPLATES).format(doctype=dt)
                out, q = self._relationship_response(dt)

            elif category == "code_generation":
                instr  = random.choice(CODE_TEMPLATES).format(doctype=dt)
                out, q = self._code_response(dt)

            elif category == "rule_explanation":
                rule   = self._get_rule_text()
                instr  = random.choice(RULE_EXPLANATION_TEMPLATES).format(rule=rule)
                out, q = self._rule_explanation_response(rule)

            elif category == "comparison":
                dt2    = self._get_doctype()
                if dt2 == dt:
                    return None
                instr  = random.choice(COMPARISON_TEMPLATES).format(doctype_a=dt, doctype_b=dt2)
                out, q = self._comparison_response(dt, dt2)

            elif category == "error_diagnosis":
                instr  = random.choice(ERROR_TEMPLATES).format(doctype=dt)
                out, q = self._error_response(dt)

            else:
                return None

            if not out or not instr:
                return None

            return SFTPair(
                instruction=instr,
                input="",
                output=out,
                category=category,
                doctype=dt,
                quality=q,
            )
        except Exception:
            return None

    def generate_n(self, n: int, min_quality: float = 0.40) -> list[SFTPair]:
        """Generate exactly n pairs, filtering by quality threshold."""
        import itertools
        pairs = []
        for pair in itertools.islice(self.generate(), n * 3):
            if pair.quality >= min_quality:
                pairs.append(pair)
            if len(pairs) >= n:
                break
        return pairs[:n]


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_sft_jsonl(pairs: list[SFTPair], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for pair in pairs:
            f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")


def write_sft_stats(pairs: list[SFTPair], path: str) -> dict:
    from collections import Counter
    cat_counts   = Counter(p.category for p in pairs)
    dt_counts    = Counter(p.doctype for p in pairs)
    avg_quality  = sum(p.quality for p in pairs) / max(len(pairs), 1)
    avg_resp_len = sum(len(p.output) for p in pairs) / max(len(pairs), 1)

    stats = {
        "total_pairs":        len(pairs),
        "avg_quality":        round(avg_quality, 3),
        "avg_response_chars": round(avg_resp_len, 1),
        "category_counts":    dict(cat_counts),
        "unique_doctypes":    len(dt_counts),
        "top_doctypes":       dict(dt_counts.most_common(10)),
        "quality_distribution": {
            ">=0.90": sum(1 for p in pairs if p.quality >= 0.90),
            ">=0.75": sum(1 for p in pairs if p.quality >= 0.75),
            ">=0.60": sum(1 for p in pairs if p.quality >= 0.60),
            "<0.60":  sum(1 for p in pairs if p.quality < 0.60),
        },
    }
    Path(path).write_text(json.dumps(stats, indent=2))
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Output Synthesizer SFT Data Generator")
    parser.add_argument("--n",           type=int, default=500, help="Number of pairs to generate")
    parser.add_argument("--out",         default="data/sft/sft_pairs.jsonl")
    parser.add_argument("--kg",          default="data/graph/knowledge_graph.json")
    parser.add_argument("--rules",       default="data/rules/rules.json")
    parser.add_argument("--min-quality", type=float, default=0.40)
    parser.add_argument("--dry-run",     action="store_true", help="Generate 20 pairs, print only")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    n = 20 if args.dry_run else args.n

    print(f"\nOutput Synthesizer SFT Data Generator")
    print(f"Target : {n} pairs (min_quality={args.min_quality})")
    print(f"Output : {args.out}\n")

    gen = SFTDataGenerator(kg_path=args.kg, rules_path=args.rules, seed=args.seed)

    t0 = time.time()
    pairs = gen.generate_n(n, min_quality=args.min_quality)
    elapsed = time.time() - t0

    print(f"\nGenerated {len(pairs)} pairs in {elapsed:.1f}s ({len(pairs)/elapsed:.1f} pairs/s)")

    if args.dry_run:
        for i, p in enumerate(pairs[:5], 1):
            print(f"\n--- Pair {i} [{p.category}] [{p.doctype}] quality={p.quality:.2f} ---")
            print(f"INSTRUCTION: {p.instruction}")
            print(f"OUTPUT (first 200 chars): {p.output[:200]}")
        return

    write_sft_jsonl(pairs, args.out)
    stats = write_sft_stats(pairs, args.out.replace(".jsonl", "_stats.json"))

    print(f"\nSaved → {args.out}")
    print(f"Stats:")
    print(f"  Avg quality   : {stats['avg_quality']:.3f}")
    print(f"  Avg resp len  : {stats['avg_response_chars']:.0f} chars")
    print(f"  Unique doctypes: {stats['unique_doctypes']}")
    print(f"  Categories    : {stats['category_counts']}")
    print(f"\nGPU fine-tuning command (when ready):")
    print(f"  python -m synapseos.phase3.output_synthesizer_sft \\")
    print(f"    --data {args.out} --model mistralai/Mistral-7B-v0.1 --lora-r 16")


if __name__ == "__main__":
    main()
