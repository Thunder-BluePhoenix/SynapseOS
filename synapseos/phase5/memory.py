"""
Phase 5 — Memory Architecture.

Three-tier memory mirroring human cognitive architecture:

  EpisodicMemory  — Hippocampus: recent interactions with vector similarity retrieval
  SemanticMemory  — Neocortex: long-term facts and knowledge graph facts
  WorkingMemory   — Prefrontal cortex: per-session scratchpad

All three are unified under MemoryController, which provides a single
get_context(query) call for use by the CognitiveController.

No GPU required — cosine similarity via numpy.

Run standalone:
    python -m synapseos.phase5.memory
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _text_to_vector(text: str, dim: int = 128) -> list[float]:
    """
    Deterministic text → float vector using character n-gram hashing.
    Consistent across processes (no random seed needed).
    Fast, CPU-only, no dependencies.
    """
    vec = [0.0] * dim
    tokens = re.findall(r"\w+", text.lower())
    for token in tokens:
        for n in (1, 2, 3):
            for i in range(len(token) - n + 1):
                gram = token[i:i+n]
                h = int(hashlib.md5(gram.encode()).hexdigest(), 16)
                idx  = h % dim
                sign = 1 if (h >> 8) & 1 else -1
                vec[idx] += sign

    # L2 normalise
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Episodic Memory
# ---------------------------------------------------------------------------

@dataclass
class EpisodicEntry:
    id: str
    query: str
    response: str
    mode_used: str
    timestamp: str
    tags: list[str]
    vector: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EpisodicEntry":
        return cls(**d)


class EpisodicMemory:
    """
    Stores every (query, response) interaction with vector embeddings.
    Retrieval: top-k cosine similarity search.
    Persistence: JSON file, loaded on init.
    """

    def __init__(
        self,
        store_path: str = "data/memory/episodic.json",
        max_entries: int = 2000,
        embed_dim: int = 128,
    ):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self.embed_dim = embed_dim
        self.entries: list[EpisodicEntry] = []
        self._load()

    def _load(self) -> None:
        if self.store_path.exists():
            try:
                raw = json.loads(self.store_path.read_text())
                self.entries = [EpisodicEntry.from_dict(e) for e in raw]
            except Exception:
                self.entries = []

    def save(self) -> None:
        data = [e.to_dict() for e in self.entries]
        self.store_path.write_text(json.dumps(data, indent=2))

    def add(self, query: str, response: str, mode_used: str, tags: list[str] | None = None) -> str:
        entry_id = hashlib.md5(f"{query}{time.time()}".encode()).hexdigest()[:12]
        vec = _text_to_vector(query, self.embed_dim)
        entry = EpisodicEntry(
            id=entry_id,
            query=query,
            response=response,
            mode_used=mode_used,
            timestamp=datetime.now().isoformat(),
            tags=tags or [],
            vector=vec,
        )
        self.entries.append(entry)

        # Evict oldest when over limit
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

        return entry_id

    def retrieve(self, query: str, top_k: int = 5, min_score: float = 0.1) -> list[tuple[float, EpisodicEntry]]:
        """Return top_k most similar past interactions, with similarity scores."""
        if not self.entries:
            return []
        qvec = _text_to_vector(query, self.embed_dim)
        scored = []
        for entry in self.entries:
            if entry.vector:
                score = _cosine(qvec, entry.vector)
                if score >= min_score:
                    scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def get_recent(self, n: int = 5) -> list[EpisodicEntry]:
        return list(reversed(self.entries[-n:]))

    def __len__(self) -> int:
        return len(self.entries)


# ---------------------------------------------------------------------------
# Semantic Memory
# ---------------------------------------------------------------------------

@dataclass
class SemanticFact:
    subject: str
    predicate: str
    obj: str
    confidence: float
    source: str          # "knowledge_graph" | "inferred" | "user_stated"
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SemanticFact":
        return cls(**d)

    def as_text(self) -> str:
        return f"{self.subject} {self.predicate} {self.obj}"


class SemanticMemory:
    """
    Long-term factual store — neocortex analogue.
    Wraps facts derived from the knowledge graph (Phase 1) plus
    any facts elevated from the episodic store or inferred by the idle loop.
    """

    def __init__(
        self,
        store_path: str = "data/memory/semantic.json",
        kg_path: str = "data/graph/knowledge_graph.json",
    ):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.facts: list[SemanticFact] = []
        self._fact_index: dict[str, list[int]] = {}   # subject → list of fact indices
        self._load()
        self._ingest_kg(kg_path)

    def _load(self) -> None:
        if self.store_path.exists():
            try:
                raw = json.loads(self.store_path.read_text())
                self.facts = [SemanticFact.from_dict(f) for f in raw]
                self._rebuild_index()
            except Exception:
                self.facts = []

    def _rebuild_index(self) -> None:
        self._fact_index = {}
        for i, fact in enumerate(self.facts):
            key = fact.subject.lower()
            self._fact_index.setdefault(key, []).append(i)

    def _ingest_kg(self, kg_path: str) -> None:
        """Load node→edge facts from the knowledge graph (one-time, idempotent)."""
        p = Path(kg_path)
        if not p.exists():
            return

        existing_subjects = {f.subject for f in self.facts if f.source == "knowledge_graph"}
        if len(existing_subjects) > 100:
            return  # Already ingested

        try:
            data = json.loads(p.read_text())
            added = 0
            for edge in data.get("links", data.get("edges", [])):
                src = edge.get("source", "")
                tgt = edge.get("target", "")
                rel = edge.get("relationship", edge.get("type", "relates_to"))
                if src and tgt:
                    self.facts.append(SemanticFact(
                        subject=src,
                        predicate=rel,
                        obj=tgt,
                        confidence=1.0,
                        source="knowledge_graph",
                        timestamp=datetime.now().isoformat(),
                    ))
                    added += 1
                    if added >= 5000:   # cap initial ingest to keep memory lean
                        break
            self._rebuild_index()
            print(f"[SemanticMemory] Ingested {added} facts from knowledge graph")
        except Exception as e:
            print(f"[SemanticMemory] KG ingest error: {e}")

    def save(self) -> None:
        data = [f.to_dict() for f in self.facts]
        self.store_path.write_text(json.dumps(data, indent=2))

    def add_fact(self, subject: str, predicate: str, obj: str,
                 confidence: float = 0.8, source: str = "inferred") -> None:
        fact = SemanticFact(
            subject=subject,
            predicate=predicate,
            obj=obj,
            confidence=confidence,
            source=source,
            timestamp=datetime.now().isoformat(),
        )
        self.facts.append(fact)
        key = subject.lower()
        self._fact_index.setdefault(key, []).append(len(self.facts) - 1)

    def retrieve(self, subject: str, top_k: int = 10) -> list[SemanticFact]:
        """Retrieve facts about a subject (exact + partial match)."""
        subject_lower = subject.lower()
        indices: set[int] = set()

        # Exact match
        if subject_lower in self._fact_index:
            indices.update(self._fact_index[subject_lower])

        # Partial match — subject as substring
        for key, idx_list in self._fact_index.items():
            if subject_lower in key or key in subject_lower:
                indices.update(idx_list)

        facts = [self.facts[i] for i in indices if i < len(self.facts)]
        facts.sort(key=lambda f: f.confidence, reverse=True)
        return facts[:top_k]

    def retrieve_text(self, query: str, top_k: int = 5) -> list[SemanticFact]:
        """Full-text search across fact text."""
        query_lower = query.lower()
        tokens = re.findall(r"\w+", query_lower)
        scored: list[tuple[int, SemanticFact]] = []
        for fact in self.facts:
            fact_text = fact.as_text().lower()
            hits = sum(1 for t in tokens if t in fact_text)
            if hits > 0:
                scored.append((hits, fact))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored[:top_k]]

    # Sources representing knowledge the system acquired (web learning,
    # consolidation, user statements) rather than the bulk KG import.
    LEARNED_SOURCES = ("web:", "world_model_consolidation", "user_stated", "inferred")

    def retrieve_learned(self, query: str, top_k: int = 3) -> list[SemanticFact]:
        """
        Retrieve learned facts only — a separate lane so the 5k KG-imported
        facts can't crowd acquired knowledge out of the context window.
        Content words (4+ chars) only, to avoid spurious substring hits.
        """
        tokens = [t for t in re.findall(r"\w+", query.lower()) if len(t) >= 4]
        if not tokens:
            return []
        scored: list[tuple[int, SemanticFact]] = []
        for fact in self.facts:
            if not any(fact.source.startswith(s) for s in self.LEARNED_SOURCES):
                continue
            fact_text = fact.as_text().lower()
            hits = sum(1 for t in tokens if t in fact_text)
            if hits > 0:
                scored.append((hits, fact))
        scored.sort(key=lambda x: (x[0], x[1].confidence), reverse=True)
        return [f for _, f in scored[:top_k]]

    def __len__(self) -> int:
        return len(self.facts)


# ---------------------------------------------------------------------------
# Working Memory
# ---------------------------------------------------------------------------

@dataclass
class WorkingMemorySlot:
    key: str
    value: Any
    written_at: float = field(default_factory=time.time)
    ttl: float = 300.0   # seconds

    def is_expired(self) -> bool:
        return time.time() - self.written_at > self.ttl


class WorkingMemory:
    """
    Per-session scratchpad — prefrontal cortex analogue.
    Holds intermediate results, active hypotheses, current task context.
    Expires after TTL. Not persisted to disk.
    """

    def __init__(self, default_ttl: float = 300.0):
        self.default_ttl = default_ttl
        self._slots: dict[str, WorkingMemorySlot] = {}

    def write(self, key: str, value: Any, ttl: float | None = None) -> None:
        self._slots[key] = WorkingMemorySlot(
            key=key,
            value=value,
            written_at=time.time(),
            ttl=ttl or self.default_ttl,
        )

    def read(self, key: str, default: Any = None) -> Any:
        slot = self._slots.get(key)
        if slot is None or slot.is_expired():
            return default
        return slot.value

    def flush_expired(self) -> int:
        before = len(self._slots)
        self._slots = {k: v for k, v in self._slots.items() if not v.is_expired()}
        return before - len(self._slots)

    def clear(self) -> None:
        self._slots.clear()

    def keys(self) -> list[str]:
        return [k for k, v in self._slots.items() if not v.is_expired()]

    def snapshot(self) -> dict[str, Any]:
        return {k: v.value for k, v in self._slots.items() if not v.is_expired()}

    def __len__(self) -> int:
        return sum(1 for v in self._slots.values() if not v.is_expired())


# ---------------------------------------------------------------------------
# Memory Controller
# ---------------------------------------------------------------------------

@dataclass
class MemoryContext:
    """Packed context snapshot returned to the cognitive controller."""
    recent_interactions: list[dict]     # last N episodic entries
    similar_past: list[dict]            # top-k similar past interactions
    semantic_facts: list[dict]          # relevant facts from semantic store
    working_state: dict[str, Any]       # current working memory snapshot
    summary: str                        # one-line context summary


class MemoryController:
    """
    Unified interface across all three memory tiers.
    Called by CognitiveController before each inference pass.
    """

    def __init__(
        self,
        episodic_path: str = "data/memory/episodic.json",
        semantic_path: str = "data/memory/semantic.json",
        kg_path: str = "data/graph/knowledge_graph.json",
    ):
        self.episodic  = EpisodicMemory(episodic_path)
        self.semantic  = SemanticMemory(semantic_path, kg_path)
        self.working   = WorkingMemory()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_context(self, query: str, top_k: int = 3) -> MemoryContext:
        recent    = [e.to_dict() for e in self.episodic.get_recent(3)]
        similar   = [(score, e.to_dict()) for score, e in self.episodic.retrieve(query, top_k=top_k)]
        sem_facts = [f.to_dict() for f in self.semantic.retrieve_text(query, top_k=top_k)]
        # Learned knowledge gets its own retrieval lane — merge, dedup
        seen_facts = {(f["subject"], f["predicate"], f["obj"]) for f in sem_facts}
        for fact in self.semantic.retrieve_learned(query, top_k=top_k):
            key = (fact.subject, fact.predicate, fact.obj)
            if key not in seen_facts:
                seen_facts.add(key)
                sem_facts.append(fact.to_dict())
        working   = self.working.snapshot()

        similar_dicts = [{"score": round(s, 3), **e} for s, e in similar]

        n_episodic = len(self.episodic)
        n_semantic = len(self.semantic)
        summary = (
            f"Episodic: {n_episodic} entries | "
            f"Semantic: {n_semantic} facts | "
            f"Working: {len(working)} slots | "
            f"Similar past: {len(similar_dicts)}"
        )

        return MemoryContext(
            recent_interactions=recent,
            similar_past=similar_dicts,
            semantic_facts=sem_facts,
            working_state=working,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record_interaction(
        self,
        query: str,
        response: str,
        mode_used: str,
        tags: list[str] | None = None,
    ) -> str:
        return self.episodic.add(query, response, mode_used, tags)

    def assert_fact(self, subject: str, predicate: str, obj: str,
                    confidence: float = 0.8, source: str = "inferred") -> None:
        self.semantic.add_fact(subject, predicate, obj, confidence, source)

    def set_working(self, key: str, value: Any, ttl: float | None = None) -> None:
        self.working.write(key, value, ttl)

    def get_working(self, key: str, default: Any = None) -> Any:
        return self.working.read(key, default)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        self.episodic.save()
        self.semantic.save()

    def stats(self) -> dict:
        return {
            "episodic_entries": len(self.episodic),
            "semantic_facts": len(self.semantic),
            "working_slots": len(self.working),
        }


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

def main():
    mem = MemoryController()
    print(f"Memory stats: {mem.stats()}")

    # Store some interactions
    mem.record_interaction("What fields are required for a Sales Invoice?", "Customer, Date, Items.", "mode1")
    mem.record_interaction("Generate a creative workflow for HR onboarding.", "...", "mode2")
    mem.record_interaction("Write a validate function for Purchase Order.", "def validate(...)", "mode4")

    # Retrieve context
    ctx = mem.get_context("Sales Invoice validation")
    print(f"\n{ctx.summary}")
    print(f"Recent: {len(ctx.recent_interactions)} | Similar: {len(ctx.similar_past)} | Facts: {len(ctx.semantic_facts)}")

    if ctx.similar_past:
        best = ctx.similar_past[0]
        print(f"Best match [{best['score']:.3f}]: {best['query']}")

    if ctx.semantic_facts:
        print(f"Top fact: {ctx.semantic_facts[0]['subject']} {ctx.semantic_facts[0]['predicate']} {ctx.semantic_facts[0]['obj']}")


if __name__ == "__main__":
    main()
