"""
SynapseOS Web Learning Loop.

Curiosity-driven web ingestion: the system decides what it wants to learn,
fetches pages politely, extracts facts, revises beliefs on contradiction,
and feeds everything through the world model's surprise gate.

Components:
  - fetcher        — polite HTTP (robots.txt, rate limits, dedup) + FakeFetcher for tests
  - extractor      — HTML → clean text, text → (subject, predicate, object) triples
  - curiosity      — learning goals from hypotheses, surprises, KG frontier, user topics
  - belief         — trust-weighted belief revision with audit trail
  - kg_writer      — web-knowledge overlay graph (curated KG stays pristine)
  - learning_loop  — orchestrator: learn(topic), run(n_goals), background daemon
"""

from synapseos.web.fetcher import (
    BaseFetcher,
    FakeFetcher,
    FetchPolicy,
    FetchResult,
    WebFetcher,
)
from synapseos.web.extractor import ExtractedFact, FactExtractor, TextExtractor
from synapseos.web.curiosity import CuriosityEngine, GoalQueue, LearningGoal
from synapseos.web.belief import BeliefReviser, BeliefRevision
from synapseos.web.kg_writer import KGWriter
from synapseos.web.learning_loop import LearnReport, WebLearningLoop
from synapseos.web.media import (
    FakeMediaIngester,
    MediaIngester,
    MediaResult,
    classify_media,
)

__all__ = [
    "BaseFetcher",
    "FakeFetcher",
    "FetchPolicy",
    "FetchResult",
    "WebFetcher",
    "ExtractedFact",
    "FactExtractor",
    "TextExtractor",
    "CuriosityEngine",
    "GoalQueue",
    "LearningGoal",
    "BeliefReviser",
    "BeliefRevision",
    "KGWriter",
    "LearnReport",
    "WebLearningLoop",
    "FakeMediaIngester",
    "MediaIngester",
    "MediaResult",
    "classify_media",
]
