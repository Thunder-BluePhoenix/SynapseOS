"""
Web Learning Loop — Belief revision.

When a new fact arrives from the web, it is not blindly stored. It is checked
against existing beliefs (semantic memory) and integrated through a
trust-weighted resolution:

  reinforced — same (s, p, o) already known → confidence bumped
  revised    — conflicting object on a functional predicate, new evidence
               clearly stronger → old belief degraded (kept for history),
               new belief stored
  rejected   — conflicting, but existing belief wins → new fact dropped
  coexists   — non-functional predicate (has_property, causes, used_for):
               multiple objects are legitimate → both kept
  added      — no conflict → stored

score = confidence × source_trust × recency_decay (half-life ~90 days).
Unknown web domains get 0.5 trust — they can add knowledge but rarely
overturn established beliefs. Every decision is appended to an audit log.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.parse
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from synapseos.web.extractor import ExtractedFact

SOURCE_TRUST = {
    "user_stated": 0.95,
    "knowledge_graph": 0.85,
    "kg_import": 0.8,
    "world_model_consolidation": 0.7,
    "inferred": 0.5,
}

DOMAIN_TRUST = {
    "en.wikipedia.org": 0.75,
    "wikipedia.org": 0.75,
    "arxiv.org": 0.7,
    "github.com": 0.6,
}
DEFAULT_WEB_TRUST = 0.5

# Predicates where a subject has (at most) one true object.
FUNCTIONAL_PREDICATES = {"is_a", "located_in", "created_by", "born_in", "died_in",
                         "part_of"}

RECENCY_HALF_LIFE_DAYS = 90.0


def _domain_of(url: str) -> str:
    return urllib.parse.urlsplit(url).netloc.split(":")[0].lower()


def source_trust(source: str) -> float:
    if source.startswith("web:"):
        return DOMAIN_TRUST.get(source[4:], DEFAULT_WEB_TRUST)
    return SOURCE_TRUST.get(source, 0.5)


@dataclass
class BeliefRevision:
    timestamp: float
    subject: str
    predicate: str
    old_obj: str | None
    new_obj: str
    old_confidence: float
    new_confidence: float
    action: str          # added | reinforced | revised | rejected | coexists
    reason: str
    source_url: str

    def to_dict(self) -> dict:
        return asdict(self)


class BeliefReviser:
    """Integrates extracted web facts into semantic memory + the KG overlay."""

    def __init__(
        self,
        semantic_memory,
        kg_writer=None,
        audit_path: str = "data/web/revisions.json",
        max_audit: int = 5000,
    ):
        self.memory = semantic_memory
        self.kg_writer = kg_writer
        self.audit_path = Path(audit_path)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_audit = max_audit
        self._lock = threading.Lock()
        self._audit: list[dict] = []
        self._load_audit()

    # -- audit log ------------------------------------------------------------

    def _load_audit(self) -> None:
        if self.audit_path.exists():
            try:
                self._audit = json.loads(self.audit_path.read_text()).get("revisions", [])
            except Exception:
                self._audit = []

    def save(self) -> None:
        with self._lock:
            self.audit_path.write_text(
                json.dumps({"revisions": self._audit[-self.max_audit:]}, indent=2)
            )

    def revisions(self, limit: int = 50) -> list[dict]:
        return self._audit[-limit:]

    def _log(self, rev: BeliefRevision) -> BeliefRevision:
        with self._lock:
            self._audit.append(rev.to_dict())
            if len(self._audit) > self.max_audit:
                self._audit = self._audit[-self.max_audit:]
        return rev

    # -- scoring ----------------------------------------------------------------

    def _score(self, confidence: float, source: str, timestamp: str) -> float:
        age_days = 0.0
        try:
            then = datetime.fromisoformat(timestamp)
            age_days = max(0.0, (datetime.now() - then).total_seconds() / 86400.0)
        except Exception:
            pass
        recency = 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)
        return confidence * source_trust(source) * max(recency, 0.1)

    def _find_conflicts(self, subject: str, predicate: str, obj: str) -> list:
        """Existing facts with same subject+predicate but a different object."""
        conflicts = []
        for fact in self.memory.retrieve(subject, top_k=50):
            if fact.subject.lower() != subject.lower():
                continue
            if fact.predicate != predicate:
                continue
            if fact.obj.strip().lower() == obj.strip().lower():
                continue
            conflicts.append(fact)
        return conflicts

    def _find_exact(self, subject: str, predicate: str, obj: str):
        for fact in self.memory.retrieve(subject, top_k=50):
            if (fact.subject.lower() == subject.lower()
                    and fact.predicate == predicate
                    and fact.obj.strip().lower() == obj.strip().lower()):
                return fact
        return None

    # -- integration ----------------------------------------------------------

    def integrate(self, fact: ExtractedFact, source_url: str,
                  source: str | None = None) -> BeliefRevision:
        """`source` overrides the domain-derived source tag (e.g. "mesh:<node_id>")."""
        domain = _domain_of(source_url)
        new_source = source or (f"web:{domain}" if domain else "web:unknown")
        now = time.time()

        # 1. Exact match → reinforce
        existing = self._find_exact(fact.subject, fact.predicate, fact.obj)
        if existing is not None:
            old_conf = existing.confidence
            existing.confidence = min(0.99, existing.confidence + 0.05)
            return self._log(BeliefRevision(
                timestamp=now, subject=fact.subject, predicate=fact.predicate,
                old_obj=existing.obj, new_obj=fact.obj,
                old_confidence=old_conf, new_confidence=existing.confidence,
                action="reinforced", reason="independent confirmation",
                source_url=source_url,
            ))

        conflicts = self._find_conflicts(fact.subject, fact.predicate, fact.obj)

        # 2. Non-functional predicates: multiple objects legitimately coexist
        if conflicts and fact.predicate not in FUNCTIONAL_PREDICATES:
            self._store(fact, new_source, source_url)
            return self._log(BeliefRevision(
                timestamp=now, subject=fact.subject, predicate=fact.predicate,
                old_obj=conflicts[0].obj, new_obj=fact.obj,
                old_confidence=conflicts[0].confidence, new_confidence=fact.confidence,
                action="coexists", reason="non-exclusive predicate",
                source_url=source_url,
            ))

        # 3. Functional conflict: trust-weighted resolution
        if conflicts:
            strongest = max(
                conflicts,
                key=lambda f: self._score(f.confidence, f.source, f.timestamp),
            )
            old_score = self._score(strongest.confidence, strongest.source,
                                    strongest.timestamp)
            new_score = self._score(fact.confidence, new_source,
                                    datetime.now().isoformat())

            if new_score > old_score * 1.1:
                # Revised — degrade the old belief, keep it for history
                old_conf = strongest.confidence
                strongest.confidence = 0.2
                if self.kg_writer is not None:
                    self.kg_writer.degrade_edge(
                        strongest.subject, strongest.predicate, strongest.obj, 0.2)
                self._store(fact, new_source, source_url)
                return self._log(BeliefRevision(
                    timestamp=now, subject=fact.subject, predicate=fact.predicate,
                    old_obj=strongest.obj, new_obj=fact.obj,
                    old_confidence=old_conf, new_confidence=fact.confidence,
                    action="revised",
                    reason=f"new evidence score {new_score:.2f} > old {old_score:.2f}",
                    source_url=source_url,
                ))

            # Rejected — existing belief stands
            return self._log(BeliefRevision(
                timestamp=now, subject=fact.subject, predicate=fact.predicate,
                old_obj=strongest.obj, new_obj=fact.obj,
                old_confidence=strongest.confidence, new_confidence=0.0,
                action="rejected",
                reason=f"existing belief stronger ({old_score:.2f} >= {new_score:.2f})",
                source_url=source_url,
            ))

        # 4. No conflict → added
        self._store(fact, new_source, source_url)
        return self._log(BeliefRevision(
            timestamp=now, subject=fact.subject, predicate=fact.predicate,
            old_obj=None, new_obj=fact.obj,
            old_confidence=0.0, new_confidence=fact.confidence,
            action="added", reason="no conflicting belief",
            source_url=source_url,
        ))

    def _store(self, fact: ExtractedFact, source: str, url: str) -> None:
        self.memory.add_fact(
            subject=fact.subject, predicate=fact.predicate, obj=fact.obj,
            confidence=fact.confidence, source=source,
        )
        if self.kg_writer is not None:
            self.kg_writer.add_fact(
                fact.subject, fact.predicate, fact.obj, fact.confidence, url)

    def stats(self) -> dict:
        counts: dict[str, int] = {}
        for r in self._audit:
            counts[r.get("action", "?")] = counts.get(r.get("action", "?"), 0) + 1
        return {"revisions_logged": len(self._audit), "by_action": counts}
