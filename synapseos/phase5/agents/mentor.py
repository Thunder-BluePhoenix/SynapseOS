"""
Phase 5 — MENTOR: the teacher agent.

What a teacher does, mapped to SynapseOS primitives:
  solve exactly      → MathSolver (computed, with worked steps — never recalled)
  explain a topic    → structured lesson from learned facts (definition first,
                       then properties/causes), sources shown
  quiz the learner   → questions generated from known facts, answers withheld
  handle languages   → script detection + phrasebook; honest about the
                       translation limit (needs the neural path)
  adapt to learner   → reads the emotion tracker: frustrated → shorter, simpler
  grow expertise     → unknown topic → seeds curriculum / queues web learning
  remember teaching  → "user was_taught X" facts accumulate in semantic memory
"""

from __future__ import annotations

from synapseos.phase5.agents.advisors import AdvisorAgent
from synapseos.phase5.agents.base import AgentCapability, AgentResult


class MENTORAgent(AdvisorAgent):
    """Teacher persona: exact math, fact-grounded lessons, quizzes."""

    capability = AgentCapability(
        name="MENTOR",
        keywords=["teach", "explain", "what is", "learn about", "lesson",
                  "calculate", "solve", "derivative", "integrate", "integral",
                  "probability", "mean of", "median", "area of", "volume of",
                  "quiz", "study", "homework", "speak in", "how do you say",
                  "translate", "language", "history of", "definition"],
        anti_keywords=["doctype", "invoice field", "deploy"],
        base_confidence=0.5,
        description="Teaching: exact math, structured lessons, quizzes, languages",
    )
    domain = "teaching"
    seed_topics = []

    def __init__(self) -> None:
        super().__init__()
        from synapseos.skills.math_solver import MathSolver
        self.math = MathSolver()
        self._solved_cache: tuple[str, object] | None = None

    def can_handle(self, query: str) -> float:
        """Keywords first; else 'can I actually solve it?' — bare math
        expressions like 'sin(30 deg)' carry no teaching keyword."""
        score = super().can_handle(query)
        if score >= 0.55:
            return score
        try:
            ans = self.math.solve(query)
        except Exception:
            ans = None
        if ans is not None:
            self._solved_cache = (query, ans)
            return 0.85
        return score

    # -- main ------------------------------------------------------------------

    def _handle(self, query: str, context=None) -> AgentResult:
        trace = ["MENTOR teacher engaged"]
        ql = query.lower()

        # Learner state: frustrated students get the short, gentle version
        brief = False
        if self.emotion_tracker is not None:
            state = self.emotion_tracker.state
            if state.valence_ema < -0.25 and state.observations >= 2:
                brief = True
                trace.append("learner seems frustrated — keeping it short")

        # 1. Exact math — computed, never recalled (reuse can_handle's solve)
        if self._solved_cache is not None and self._solved_cache[0] == query:
            math_ans = self._solved_cache[1]
        else:
            math_ans = self.math.solve(query)
        if math_ans is not None:
            lines = [f"[MENTOR — {math_ans.topic}]", math_ans.answer]
            if not brief and math_ans.steps:
                lines.append("Working:")
                lines.extend(f"  {i+1}. {s}" for i, s in enumerate(math_ans.steps))
            if not math_ans.exact:
                lines.append("(numeric approximation)")
            trace.append(f"math solver: {math_ans.topic}")
            self._record_teaching(math_ans.topic)
            return self._result(query, "\n".join(lines), 0.9, trace,
                                {"kind": "math", "topic": math_ans.topic})

        # 2. Language questions
        lang_out = self._try_language(query, ql, trace)
        if lang_out is not None:
            return lang_out

        # 3. Quiz mode
        if "quiz" in ql or "test me" in ql:
            return self._quiz(query, trace)

        # 4. Lesson from learned knowledge
        return self._lesson(query, trace, brief)

    # -- language ------------------------------------------------------------------

    def _try_language(self, query: str, ql: str, trace: list[str]) -> AgentResult | None:
        from synapseos.skills.language import PHRASEBOOK, detect_language, phrase

        # "how do you say thank you in bengali"
        import re
        m = re.search(r"how do you say\s+['\"]?([\w ]+?)['\"]?\s+in\s+(\w+)", ql)
        if m:
            word, lang = m.group(1).strip(), m.group(2).strip()
            p = phrase(lang, word)
            trace.append(f"phrasebook lookup: {lang}/{word}")
            if p:
                out = (f"[MENTOR — languages]\nIn {lang.title()}: \"{word}\" → {p}\n"
                       f"(Phrasebook entry. Full translation needs the neural "
                       f"language path — deeper vocabulary grows via web learning.)")
                self._record_teaching(f"{lang} phrase")
                return self._result(query, out, 0.75, trace, {"kind": "language"})
            queued = self._queue_learning(f"{lang} language common phrases")
            out = (f"[MENTOR — languages]\nI don't have \"{word}\" in my {lang.title()} "
                   f"phrasebook yet." +
                   (f" Queued for learning: {queued[0]}" if queued else ""))
            return self._result(query, out, 0.4, trace, {"kind": "language"})

        # "what language is <text>" / detection
        m = re.search(r"(?:what|which) language is\s+[:\"']?(.+?)[\"']?\s*\??$", query,
                      re.IGNORECASE | re.DOTALL)
        if m:
            guess = detect_language(m.group(1))
            trace.append(f"script/stopword detection: {guess.language}")
            out = (f"[MENTOR — languages]\nDetected: {guess.language} "
                   f"(confidence {guess.confidence:.2f}, via {guess.method})")
            return self._result(query, out, guess.confidence, trace,
                                {"kind": "language", **guess.to_dict()})
        return None

    # -- lessons --------------------------------------------------------------------

    def _lesson(self, query: str, trace: list[str], brief: bool) -> AgentResult:
        facts = self._knowledge(query, top_k=8)
        topic = query.lower()
        for lead in ("teach me about ", "teach me ", "explain ", "what is ",
                     "learn about ", "lesson on "):
            if topic.startswith(lead):
                topic = topic[len(lead):].strip(" ?")
                break

        if not facts:
            queued = self._queue_learning(topic or query)
            out = (f"[MENTOR — lesson]\nI haven't studied \"{topic}\" yet."
                   + (f"\nQueued for web learning: {', '.join(queued[:2])} — "
                      f"ask me again after the next learning cycle."
                      if queued else ""))
            trace.append("no knowledge — queued curriculum")
            return self._result(query, out, 0.3, trace, {"kind": "lesson", "facts": 0})

        # Structure: definitions first, then properties/relations
        definitions = [f for f in facts if f.predicate == "is_a"]
        others = [f for f in facts if f.predicate != "is_a"]
        lines = [f"[MENTOR — lesson: {topic}]"]
        if definitions:
            d = definitions[0]
            lines.append(f"Definition: {d.subject} is a {d.obj}.")
        keep = 2 if brief else 5
        if others:
            lines.append("Key points:")
            for f in others[:keep]:
                lines.append(f"  - {f.subject} {f.predicate.replace('_', ' ')} {f.obj} "
                             f"({f.source}, {f.confidence:.2f})")
        if not brief:
            lines.append(f"Sources: {len(facts)} facts from learned knowledge. "
                         f"Ask \"quiz me on {topic}\" to test yourself.")
        trace.append(f"lesson from {len(facts)} facts")
        self._record_teaching(topic)
        confidence = min(0.85, 0.5 + 0.06 * len(facts))
        return self._result(query, "\n".join(lines), confidence, trace,
                            {"kind": "lesson", "facts": len(facts)})

    def _quiz(self, query: str, trace: list[str]) -> AgentResult:
        topic = query.lower().replace("quiz me on", "").replace("quiz", "").strip(" ?")
        facts = [f for f in self._knowledge(topic or query, top_k=6)
                 if f.predicate in ("is_a", "located_in", "created_by", "causes",
                                    "part_of", "used_for")]
        if not facts:
            out = f"[MENTOR — quiz]\nI don't know enough about \"{topic}\" to quiz you yet."
            return self._result(query, out, 0.3, trace, {"kind": "quiz", "questions": 0})
        templates = {
            "is_a": "What is {s}?", "located_in": "Where is {s} located?",
            "created_by": "Who created {s}?", "causes": "What does {s} cause?",
            "part_of": "What is {s} part of?", "used_for": "What is {s} used for?",
        }
        lines = [f"[MENTOR — quiz: {topic}]"]
        answers = []
        for i, f in enumerate(facts[:4], 1):
            lines.append(f"  Q{i}. {templates[f.predicate].format(s=f.subject)}")
            answers.append(f"A{i}: {f.obj}")
        lines.append("(Answers on request — say 'show answers'.)")
        trace.append(f"generated {min(len(facts), 4)} questions")
        return self._result(query, "\n".join(lines), 0.7, trace,
                            {"kind": "quiz", "questions": min(len(facts), 4),
                             "answers": answers})

    # -- helpers -----------------------------------------------------------------

    def _record_teaching(self, topic: str) -> None:
        if self.semantic_memory is not None and topic:
            self.semantic_memory.add_fact(
                subject="user", predicate="was_taught", obj=topic[:60],
                confidence=0.7, source="mentor_teaching")

    def _result(self, query: str, output: str, confidence: float,
                trace: list[str], structured: dict) -> AgentResult:
        return AgentResult(
            agent_name="MENTOR", query=query, output=output,
            structured={"domain": self.domain, **structured},
            confidence=confidence, latency_ms=0.0, traceable=True, trace=trace,
        )
