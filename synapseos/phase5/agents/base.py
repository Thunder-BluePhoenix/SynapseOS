"""
Phase 5 — Sub-agent base class.

All Phase 5 cognitive entities (ARIEL, NYX, NOVA, ATLAS) implement SubAgent.
Each agent:
  - declares which query signals it owns (AgentCapability)
  - exposes can_handle(query) → float confidence
  - exposes handle(query, context) → AgentResult

The AgentRouter in the CognitiveController scores all registered agents
against each query and dispatches to the highest-confidence match(es).

Agents share the MemoryController — they read episodic context and write
new facts back to semantic memory.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AgentCapability:
    """Declares what signal patterns this agent owns."""
    name: str
    keywords: list[str]          # trigger words → raise agent confidence
    anti_keywords: list[str]     # lower confidence when these appear
    base_confidence: float       # floor confidence for this agent's domain
    description: str


@dataclass
class AgentResult:
    agent_name: str
    query: str
    output: str
    structured: dict[str, Any]
    confidence: float
    latency_ms: float
    traceable: bool
    trace: list[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "agent":       self.agent_name,
            "query":       self.query,
            "output":      self.output[:300],
            "confidence":  round(self.confidence, 3),
            "latency_ms":  round(self.latency_ms, 1),
            "traceable":   self.traceable,
            "trace_steps": len(self.trace),
            "error":       self.error,
        }


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class SubAgent(ABC):
    """
    Base class for all SynapseOS cognitive sub-agents.

    Subclasses override:
      - capability: AgentCapability  (class attribute)
      - _handle(query, context) → AgentResult

    The base provides can_handle() scoring and wraps _handle() with
    timing and error capture.
    """

    capability: AgentCapability  # must be set by subclass

    def can_handle(self, query: str) -> float:
        """
        Return a confidence score [0, 1] indicating how well this agent
        can handle the given query.
        """
        query_lower = query.lower()

        hits = sum(1 for kw in self.capability.keywords if kw in query_lower)
        anti_hits = sum(1 for kw in self.capability.anti_keywords if kw in query_lower)

        if hits == 0:
            return 0.0

        score = self.capability.base_confidence + 0.08 * hits - 0.15 * anti_hits
        return max(0.0, min(1.0, score))

    def handle(self, query: str, context: Any = None) -> AgentResult:
        """Public entry point — times execution and catches errors."""
        t0 = time.time()
        try:
            result = self._handle(query, context)
            result.latency_ms = (time.time() - t0) * 1000
            return result
        except Exception as e:
            return AgentResult(
                agent_name=self.capability.name,
                query=query,
                output=f"[{self.capability.name} error: {e}]",
                structured={},
                confidence=0.0,
                latency_ms=(time.time() - t0) * 1000,
                traceable=False,
                error=str(e),
            )

    @abstractmethod
    def _handle(self, query: str, context: Any) -> AgentResult:
        """Implement domain-specific logic here."""
        ...

    @property
    def name(self) -> str:
        return self.capability.name
