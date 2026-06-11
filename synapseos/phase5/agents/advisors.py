"""
Phase 5 — Advisor agents: domain personas of the companion.

Six advisors join ARIEL/NYX/NOVA/ATLAS, built on the same SubAgent contract:

  VITA    — health & wellbeing
  LEX     — legal awareness
  AURUM   — finance / accounting (CA)
  FORGE   — engineering & technology
  SAGE    — counsellor (emotion-aware: reads the tracked emotional state)
  ORACLE  — foresight (pattern-based, NOT psychic: projects from observed
            history and learned facts, always shows its evidence)

What makes them companions rather than chatbots:
  - Their knowledge is whatever SynapseOS has actually learned (semantic
    memory: web facts, consolidated patterns, affect observations) — answers
    cite source + confidence, never invented.
  - When an advisor doesn't know, it says so and queues curiosity goals so
    the web learning loop grows that expertise for next time.
  - Regulated domains (health/legal/finance) always carry a professional-
    consultation disclaimer. This is a hard rule, not a style choice.

Shared resources (semantic memory, emotion tracker, world model, web loop)
are injected by the CognitiveController after construction via wire().
"""

from __future__ import annotations

import time

from synapseos.phase5.agents.base import AgentCapability, AgentResult, SubAgent


class AdvisorAgent(SubAgent):
    """Base for all domain advisors. Subclasses set capability + domain config."""

    domain: str = ""
    disclaimer: str = ""              # "" = unregulated domain
    seed_topics: list[str] = []       # curiosity goals queued when ignorant

    def __init__(self) -> None:
        # Wired post-construction by the controller; all optional.
        self.semantic_memory = None
        self.emotion_tracker = None
        self.world_model = None
        self.web_loop = None

    def wire(self, semantic_memory=None, emotion_tracker=None,
             world_model=None, web_loop=None) -> None:
        self.semantic_memory = semantic_memory
        self.emotion_tracker = emotion_tracker
        self.world_model = world_model
        self.web_loop = web_loop

    # -- knowledge access -------------------------------------------------------

    def _knowledge(self, query: str, top_k: int = 5) -> list:
        if self.semantic_memory is None:
            return []
        facts = list(self.semantic_memory.retrieve_learned(query, top_k=top_k))
        seen = {(f.subject, f.predicate, f.obj) for f in facts}
        for f in self.semantic_memory.retrieve_text(query, top_k=top_k):
            if (f.subject, f.predicate, f.obj) not in seen:
                facts.append(f)
        return facts[:top_k]

    def _queue_learning(self, query: str) -> list[str]:
        """Ignorance → curiosity. Returns the queued topic names."""
        if self.web_loop is None:
            return []
        queued = []
        try:
            for topic in [query[:60]] + self.seed_topics[:2]:
                self.web_loop.curiosity.add_user_topic(topic, priority=0.8)
                queued.append(topic)
        except Exception:
            pass
        return queued

    # -- response assembly --------------------------------------------------------

    def _compose(self, query: str, facts: list, trace: list[str]) -> tuple[str, float]:
        lines = [f"[{self.capability.name} — {self.domain}]"]
        if facts:
            lines.append("What I know (with sources):")
            for f in facts:
                lines.append(f"  - {f.subject} {f.predicate.replace('_', ' ')} {f.obj} "
                             f"({f.source}, confidence {f.confidence:.2f})")
            confidence = min(0.85, 0.5 + 0.07 * len(facts))
            trace.append(f"retrieved {len(facts)} facts from semantic memory")
        else:
            queued = self._queue_learning(query)
            lines.append("I don't have learned knowledge on this yet.")
            if queued:
                lines.append(f"I've queued it for web learning: {', '.join(queued[:2])} "
                             f"— ask me again after the next learning cycle.")
                trace.append(f"queued {len(queued)} curiosity goals")
            confidence = 0.3
        if self.disclaimer:
            lines.append(f"⚠ {self.disclaimer}")
        return "\n".join(lines), confidence

    def _handle(self, query: str, context=None) -> AgentResult:
        trace = [f"{self.capability.name} advisor engaged"]
        facts = self._knowledge(query)
        output, confidence = self._compose(query, facts, trace)
        return AgentResult(
            agent_name=self.capability.name, query=query, output=output,
            structured={"domain": self.domain, "facts_used": len(facts)},
            confidence=confidence, latency_ms=0.0, traceable=True, trace=trace,
        )


# ---------------------------------------------------------------------------
# Domain advisors
# ---------------------------------------------------------------------------

class VITAAgent(AdvisorAgent):
    """Health & wellbeing advisor."""
    capability = AgentCapability(
        name="VITA",
        keywords=["health", "sleep", "diet", "exercise", "symptom", "pain",
                  "doctor", "medicine", "nutrition", "fitness", "headache",
                  "tired", "stress", "wellness", "injury", "fever"],
        anti_keywords=["server health", "health check", "healthcheck"],
        base_confidence=0.55,
        description="Health and wellbeing information, habit tracking",
    )
    domain = "health & wellbeing"
    disclaimer = ("Health information only — not a medical diagnosis. "
                  "Consult a qualified doctor for medical decisions.")
    seed_topics = ["sleep hygiene basics", "balanced nutrition fundamentals"]


class LEXAgent(AdvisorAgent):
    """Legal awareness advisor."""
    capability = AgentCapability(
        name="LEX",
        keywords=["legal", "law", "contract", "agreement", "liability",
                  "copyright", "license", "gdpr", "compliance", "lawsuit",
                  "terms", "regulation", "patent", "trademark", "sue"],
        anti_keywords=["business rule", "validation rule"],
        base_confidence=0.55,
        description="Legal awareness and risk flagging",
    )
    domain = "legal awareness"
    disclaimer = ("General legal information — not legal advice. "
                  "Consult a licensed lawyer for your jurisdiction.")
    seed_topics = ["contract law basics", "software licensing overview"]


class AURUMAgent(AdvisorAgent):
    """Finance / chartered-accountant advisor."""
    capability = AgentCapability(
        name="AURUM",
        keywords=["tax", "invoice", "accounting", "finance", "budget",
                  "investment", "profit", "loss", "audit", "gst", "expense",
                  "revenue", "cash flow", "balance sheet", "depreciation",
                  "savings", "loan"],
        anti_keywords=["sales invoice doctype", "invoice field"],
        base_confidence=0.55,
        description="Finance, accounting and tax awareness",
    )
    domain = "finance & accounting"
    disclaimer = ("Financial information only — not investment or tax advice. "
                  "Consult a chartered accountant or licensed advisor.")
    seed_topics = ["personal budgeting fundamentals", "income tax basics India"]


class FORGEAgent(AdvisorAgent):
    """Engineering & technology advisor."""
    capability = AgentCapability(
        name="FORGE",
        keywords=["architecture", "design pattern", "database", "scaling",
                  "algorithm", "framework", "protocol", "circuit",
                  "mechanical", "electrical", "build system", "infrastructure",
                  "technology choice", "stack"],
        anti_keywords=[],
        base_confidence=0.5,
        description="Engineering and technology guidance",
    )
    domain = "engineering & technology"
    seed_topics = ["software architecture principles", "systems design basics"]


class SAGEAgent(AdvisorAgent):
    """Counsellor — the emotion-aware advisor. Reads the tracked state."""
    capability = AgentCapability(
        name="SAGE",
        keywords=["feel", "feeling", "overwhelmed", "anxious", "worried",
                  "burnout", "motivation", "lonely", "sad", "depressed",
                  "stressed", "frustrated", "angry", "cope", "talk",
                  "struggling", "advice", "decision"],
        anti_keywords=["error", "bug", "field"],
        base_confidence=0.5,
        description="Emotional support and reflective counselling",
    )
    domain = "counselling"
    disclaimer = ("Supportive conversation — not therapy. If you're in crisis, "
                  "reach out to a mental-health professional or helpline.")
    seed_topics = ["stress management techniques", "cognitive reframing basics"]

    def _handle(self, query: str, context=None) -> AgentResult:
        trace = ["SAGE counsellor engaged"]
        lines = ["[SAGE — counsellor]"]

        # Ground the response in the *tracked* emotional history, not just
        # this message — that's what makes it a companion.
        if self.emotion_tracker is not None:
            state = self.emotion_tracker.state
            lines.append(f"Where you seem to be: {state.describe()}.")
            if state.last_shift:
                lines.append(f"I noticed: {state.last_shift}")
            triggers = self.emotion_tracker._trigger_counts.get("frustration", {})
            if triggers:
                top = max(triggers, key=triggers.get)
                lines.append(f"A recurring source of frustration for you: \"{top}\".")
            trace.append("grounded in tracked emotional state")
            confidence = 0.6
        else:
            lines.append("I'm listening. Tell me more about what's going on.")
            confidence = 0.4

        facts = self._knowledge(query, top_k=3)
        if facts:
            lines.append("Something I've learned that may help:")
            for f in facts[:2]:
                lines.append(f"  - {f.subject} {f.predicate.replace('_', ' ')} {f.obj}")
            trace.append(f"used {len(facts)} learned facts")

        lines.append(f"⚠ {self.disclaimer}")
        return AgentResult(
            agent_name="SAGE", query=query, output="\n".join(lines),
            structured={"domain": self.domain},
            confidence=confidence, latency_ms=0.0, traceable=True, trace=trace,
        )


class ORACLEAgent(AdvisorAgent):
    """
    Foresight — honest version of 'psychic': projects from observed history.
    Every projection cites the episodes/facts it's based on.
    """
    capability = AgentCapability(
        name="ORACLE",
        keywords=["predict", "forecast", "will it", "what happens if",
                  "future", "likely", "expect", "trend", "outcome",
                  "should i start", "going to"],
        anti_keywords=[],
        base_confidence=0.5,
        description="Pattern-based foresight from observed history",
    )
    domain = "foresight (pattern-based, not prophecy)"
    seed_topics = []

    def _handle(self, query: str, context=None) -> AgentResult:
        trace = ["ORACLE foresight engaged"]
        lines = ["[ORACLE — foresight]",
                 "I don't predict the future — I project from what I've observed:"]
        evidence = 0

        # Past experience: state-conditioned recall from the world model
        if self.world_model is not None:
            try:
                recalled = self.world_model.reason(query, top_k=3)
                strong = [r for r in recalled if r.score > 0.5]
                if strong:
                    lines.append("Similar past situations:")
                    for r in strong[:2]:
                        lines.append(f"  - \"{r.episode.text[:80]}\" "
                                     f"(relevance {r.score:.2f})")
                    evidence += len(strong)
                    trace.append(f"recalled {len(strong)} relevant episodes")
            except Exception:
                pass

        # Learned causal facts ("X causes Y", "X leads to Y")
        facts = [f for f in self._knowledge(query, top_k=8)
                 if f.predicate in ("causes", "used_for", "part_of")]
        if facts:
            lines.append("Learned cause/effect patterns:")
            for f in facts[:3]:
                lines.append(f"  - {f.subject} {f.predicate.replace('_', ' ')} {f.obj} "
                             f"(confidence {f.confidence:.2f})")
            evidence += len(facts)

        if evidence == 0:
            lines.append("Not enough observed history to project from yet — "
                         "the more we work together, the better this gets.")
            confidence = 0.3
        else:
            confidence = min(0.75, 0.4 + 0.08 * evidence)

        return AgentResult(
            agent_name="ORACLE", query=query, output="\n".join(lines),
            structured={"domain": self.domain, "evidence": evidence},
            confidence=confidence, latency_ms=0.0, traceable=True, trace=trace,
        )


ALL_ADVISORS = [VITAAgent, LEXAgent, AURUMAgent, FORGEAgent, SAGEAgent, ORACLEAgent]
