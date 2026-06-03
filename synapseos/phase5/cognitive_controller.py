"""
Phase 5 — Unified Cognitive Controller.

The top-level orchestrator for SynapseOS. Accepts any input, routes it to
one or more thinking modes run in parallel, merges the results, and returns
a structured CognitionResult.

Architecture:
    INPUT
      ↓
    ModeRouter          — classifies input → which mode(s) to activate
      ↓  (parallel)
    ┌─────────────────────────────────────────┐
    │  Mode 1: Focused Reasoning (pipeline)   │
    │  Mode 2: Creative Generation (engine)   │
    │  Mode 3: New Thought Synthesis          │
    │  Mode 4: Code Thinking (synthesizer)    │
    └─────────────────────────────────────────┘
      ↓
    ResultSynthesizer   — merge, rank, deduplicate
      ↓
    OutputRouter        — format as language | code | both
      ↓
    MemoryController    — record interaction, update episodic store

Mode 5 (Idle Exploration) runs as a background daemon — always active
when no other modes are executing.

Run:
    python -m synapseos.phase5.cognitive_controller
    python -m synapseos.phase5.cognitive_controller --query "design an automated audit trail for Sales Invoices"
    python -m synapseos.phase5.cognitive_controller --query "write a validate function for Purchase Order" --mode mode4
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from synapseos.phase5.memory import MemoryController, MemoryContext


# ---------------------------------------------------------------------------
# Mode definitions
# ---------------------------------------------------------------------------

class ThinkingMode(str, Enum):
    MODE1_REASONING   = "mode1"   # Focused Reasoning — neurosymbolic pipeline
    MODE2_CREATIVE    = "mode2"   # Creative Generation
    MODE3_SYNTHESIS   = "mode3"   # New Thought Synthesis (cross-domain)
    MODE4_CODE        = "mode4"   # Code Thinking — program synthesis
    MODE5_IDLE        = "mode5"   # Idle Exploration (background only)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ModeResult:
    mode: ThinkingMode
    output: str
    structured: dict[str, Any]
    confidence: float
    latency_ms: float
    traceable: bool
    trace: list[str] = field(default_factory=list)
    error: str = ""


@dataclass
class CognitionResult:
    query: str
    modes_activated: list[ThinkingMode]
    primary_output: str
    output_type: str           # language | code | both
    mode_results: list[ModeResult]
    memory_context: dict       # summary of memory context used
    total_latency_ms: float
    confidence: float

    def to_dict(self) -> dict:
        return {
            "query":           self.query,
            "modes_activated": [m.value for m in self.modes_activated],
            "primary_output":  self.primary_output,
            "output_type":     self.output_type,
            "confidence":      round(self.confidence, 3),
            "latency_ms":      round(self.total_latency_ms, 1),
            "mode_results": [
                {
                    "mode":       r.mode.value,
                    "confidence": round(r.confidence, 3),
                    "latency_ms": round(r.latency_ms, 1),
                    "traceable":  r.traceable,
                    "output":     r.output[:200],
                }
                for r in self.mode_results
            ],
        }


# ---------------------------------------------------------------------------
# Mode Router
# ---------------------------------------------------------------------------

class ModeRouter:
    """
    Classifies an input query to determine which thinking modes to activate.

    Rules are deterministic and inspectable — no LLM required for routing.
    Modes can be combined (e.g. creative + code for "design and implement X").
    """

    # Keyword → mode signals
    _SIGNALS: list[tuple[list[str], ThinkingMode, float]] = [
        # Code signals
        (["write", "implement", "code", "function", "method", "script",
          "def ", "class ", "validate", "controller", "handler"],
         ThinkingMode.MODE4_CODE, 0.85),

        # Creative signals
        (["create", "design", "brainstorm", "generate", "idea", "novel",
          "creative", "invent", "imagine", "propose", "suggest"],
         ThinkingMode.MODE2_CREATIVE, 0.75),

        # Synthesis signals — cross-domain, analogy
        (["analogy", "cross-domain", "bridge", "insight", "synthesize",
          "combine", "hybrid", "what if", "new approach"],
         ThinkingMode.MODE3_SYNTHESIS, 0.70),

        # Reasoning signals — query, explain, describe, validate a fact
        (["what", "which", "how", "explain", "describe", "list",
          "required", "rules", "fields", "relationship", "dependency"],
         ThinkingMode.MODE1_REASONING, 0.80),
    ]

    def classify(self, query: str) -> list[tuple[ThinkingMode, float]]:
        """
        Returns (mode, confidence) pairs, ordered by confidence.
        Multiple modes may be returned when signals overlap.
        """
        query_lower = query.lower()
        scores: dict[ThinkingMode, float] = {}

        for keywords, mode, base_score in self._SIGNALS:
            hits = sum(1 for kw in keywords if kw in query_lower)
            if hits > 0:
                # Multiple keyword hits increase confidence
                conf = min(1.0, base_score + 0.05 * (hits - 1))
                if mode not in scores or conf > scores[mode]:
                    scores[mode] = conf

        if not scores:
            # Default: focused reasoning
            scores[ThinkingMode.MODE1_REASONING] = 0.5

        # Return modes that scored above threshold, highest first
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(mode, conf) for mode, conf in ranked if conf >= 0.4]

    def select_modes(self, query: str, max_modes: int = 2) -> list[ThinkingMode]:
        ranked = self.classify(query)
        return [mode for mode, _ in ranked[:max_modes]]


# ---------------------------------------------------------------------------
# Mode executors
# ---------------------------------------------------------------------------

def _run_mode1(query: str, context: MemoryContext) -> ModeResult:
    """Focused Reasoning via the neurosymbolic pipeline."""
    t0 = time.time()
    try:
        from synapseos.phase3.pipeline import SynapseOSPipeline
        pipeline = SynapseOSPipeline()
        result = pipeline.run(query)
        output = result.get("output", str(result))
        structured = result if isinstance(result, dict) else {"output": output}
        confidence = result.get("confidence", 0.7) if isinstance(result, dict) else 0.7
        return ModeResult(
            mode=ThinkingMode.MODE1_REASONING,
            output=output,
            structured=structured,
            confidence=float(confidence),
            latency_ms=(time.time() - t0) * 1000,
            traceable=True,
            trace=[f"Mode 1 via neurosymbolic pipeline: {query[:60]}"],
        )
    except Exception as e:
        return ModeResult(
            mode=ThinkingMode.MODE1_REASONING,
            output=f"[Mode 1 unavailable: {e}]",
            structured={},
            confidence=0.0,
            latency_ms=(time.time() - t0) * 1000,
            traceable=False,
            error=str(e),
        )


def _run_mode2(query: str, context: MemoryContext) -> ModeResult:
    """Creative Generation via the creativity engine."""
    t0 = time.time()
    try:
        from synapseos.phase4.creativity_engine import CreativityEngine
        engine = CreativityEngine()
        ideas = engine.generate(seed=query, apply_debate=True, top_k=2)
        if not ideas:
            raise ValueError("No ideas generated")
        top = ideas[0]
        output = f"{top.title}\n\n{top.description}"
        return ModeResult(
            mode=ThinkingMode.MODE2_CREATIVE,
            output=output,
            structured={
                "title":        top.title,
                "type":         top.idea_type,
                "novelty":      top.novelty_score,
                "confidence":   top.confidence_score,
                "concepts":     top.source_concepts,
                "all_ideas":    [i.title for i in ideas],
            },
            confidence=top.combined_score,
            latency_ms=(time.time() - t0) * 1000,
            traceable=True,
            trace=top.generation_trace,
        )
    except Exception as e:
        return ModeResult(
            mode=ThinkingMode.MODE2_CREATIVE,
            output=f"[Mode 2 unavailable: {e}]",
            structured={},
            confidence=0.0,
            latency_ms=(time.time() - t0) * 1000,
            traceable=False,
            error=str(e),
        )


def _run_mode3(query: str, context: MemoryContext) -> ModeResult:
    """
    New Thought Synthesis — cross-domain analogy discovery.
    Runs the creativity engine with higher cross-domain bias and returns
    the most novel idea (distinct from Mode 2 which returns the highest
    combined score).
    """
    t0 = time.time()
    try:
        from synapseos.phase4.creativity_engine import CreativityEngine
        engine = CreativityEngine()
        ideas = engine.generate(seed=query, apply_debate=False, top_k=5)
        if not ideas:
            raise ValueError("No synthesis ideas generated")

        # Select the most novel (not the highest combined score)
        most_novel = max(ideas, key=lambda i: i.novelty_score)
        output = (
            f"[Cross-domain synthesis]\n"
            f"{most_novel.title}\n\n"
            f"{most_novel.description}\n\n"
            f"Novelty: {most_novel.novelty_score:.2f}"
        )
        return ModeResult(
            mode=ThinkingMode.MODE3_SYNTHESIS,
            output=output,
            structured={
                "title":          most_novel.title,
                "type":           most_novel.idea_type,
                "novelty":        most_novel.novelty_score,
                "source_concepts": most_novel.source_concepts,
            },
            confidence=most_novel.novelty_score * 0.7,
            latency_ms=(time.time() - t0) * 1000,
            traceable=True,
            trace=most_novel.generation_trace,
        )
    except Exception as e:
        return ModeResult(
            mode=ThinkingMode.MODE3_SYNTHESIS,
            output=f"[Mode 3 unavailable: {e}]",
            structured={},
            confidence=0.0,
            latency_ms=(time.time() - t0) * 1000,
            traceable=False,
            error=str(e),
        )


def _run_mode4(query: str, context: MemoryContext) -> ModeResult:
    """Code Thinking via the program synthesizer."""
    t0 = time.time()
    try:
        from synapseos.phase4.program_synthesis import ProgramSynthesizer
        synth = ProgramSynthesizer()
        result = synth.synthesize(query)
        code_block = result.code if result.code else "[No code produced]"
        output = f"```python\n{code_block}\n```"
        return ModeResult(
            mode=ThinkingMode.MODE4_CODE,
            output=output,
            structured={
                "code":          result.code,
                "tests":         result.tests,
                "static_valid":  result.static_valid,
                "static_errors": result.static_errors,
                "doctype":       result.intent.doctype,
                "event":         result.intent.event,
            },
            confidence=0.9 if result.static_valid else 0.5,
            latency_ms=(time.time() - t0) * 1000,
            traceable=True,
            trace=result.generation_trace,
        )
    except Exception as e:
        return ModeResult(
            mode=ThinkingMode.MODE4_CODE,
            output=f"[Mode 4 unavailable: {e}]",
            structured={},
            confidence=0.0,
            latency_ms=(time.time() - t0) * 1000,
            traceable=False,
            error=str(e),
        )


_MODE_EXECUTORS = {
    ThinkingMode.MODE1_REASONING: _run_mode1,
    ThinkingMode.MODE2_CREATIVE:  _run_mode2,
    ThinkingMode.MODE3_SYNTHESIS: _run_mode3,
    ThinkingMode.MODE4_CODE:      _run_mode4,
}


# ---------------------------------------------------------------------------
# Parallel executor
# ---------------------------------------------------------------------------

class ParallelExecutor:
    """
    Runs selected thinking modes in parallel using a thread pool.
    Each mode has a per-mode timeout — a slow mode does not block the others.
    """

    def __init__(self, mode_timeout: float = 30.0, max_workers: int = 4):
        self.mode_timeout = mode_timeout
        self.max_workers = max_workers

    def run(self, query: str, modes: list[ThinkingMode], context: MemoryContext) -> list[ModeResult]:
        if not modes:
            return []

        results: list[ModeResult] = []
        with ThreadPoolExecutor(max_workers=min(len(modes), self.max_workers)) as pool:
            futures = {
                pool.submit(_MODE_EXECUTORS[mode], query, context): mode
                for mode in modes
                if mode in _MODE_EXECUTORS
            }

            for future in as_completed(futures, timeout=self.mode_timeout * len(modes)):
                mode = futures[future]
                try:
                    result = future.result(timeout=self.mode_timeout)
                    results.append(result)
                except TimeoutError:
                    results.append(ModeResult(
                        mode=mode,
                        output=f"[{mode.value} timed out after {self.mode_timeout}s]",
                        structured={},
                        confidence=0.0,
                        latency_ms=self.mode_timeout * 1000,
                        traceable=False,
                        error="timeout",
                    ))
                except Exception as e:
                    results.append(ModeResult(
                        mode=mode,
                        output=f"[{mode.value} error: {e}]",
                        structured={},
                        confidence=0.0,
                        latency_ms=0.0,
                        traceable=False,
                        error=str(e),
                    ))

        return results


# ---------------------------------------------------------------------------
# Result synthesizer
# ---------------------------------------------------------------------------

class ResultSynthesizer:
    """
    Merges outputs from multiple modes into a single coherent response.

    Strategy:
    - If Mode 4 (code) ran and succeeded, code output is primary for code-type queries.
    - Otherwise highest-confidence successful mode is primary.
    - Secondary modes contribute to the trace/context section.
    """

    def synthesize(
        self,
        query: str,
        mode_results: list[ModeResult],
        context: MemoryContext,
    ) -> tuple[str, str, float]:
        """
        Returns (primary_output, output_type, confidence).
        output_type: "language" | "code" | "both"
        """
        successful = [r for r in mode_results if not r.error]
        if not successful:
            return "[No mode produced a valid output.]", "language", 0.0

        # Separate code and language results
        code_results = [r for r in successful if r.mode == ThinkingMode.MODE4_CODE
                        and r.structured.get("static_valid", False)]
        language_results = [r for r in successful if r.mode != ThinkingMode.MODE4_CODE]
        language_results.sort(key=lambda r: r.confidence, reverse=True)

        has_code     = bool(code_results)
        has_language = bool(language_results)

        if has_code and has_language:
            output_type = "both"
            code_part = code_results[0].output
            lang_part = language_results[0].output
            primary_output = f"{lang_part}\n\n---\n\n{code_part}"
            confidence = (code_results[0].confidence + language_results[0].confidence) / 2

        elif has_code:
            output_type = "code"
            primary_output = code_results[0].output
            confidence = code_results[0].confidence

        else:
            output_type = "language"
            primary_output = language_results[0].output
            confidence = language_results[0].confidence

        # Enrich with context from similar past interactions
        if context.similar_past:
            best_past = context.similar_past[0]
            if best_past["score"] > 0.5:
                primary_output += (
                    f"\n\n[Memory context: similar past query at score {best_past['score']:.2f} — "
                    f"\"{best_past['query'][:80]}\"]"
                )

        return primary_output, output_type, confidence


# ---------------------------------------------------------------------------
# Cognitive Controller
# ---------------------------------------------------------------------------

class CognitiveController:
    """
    Main entry point for Phase 5 unified cognitive flow.

    Usage:
        controller = CognitiveController()
        result = controller.think("What validate rules apply to Sales Invoice?")
        print(result.primary_output)
    """

    def __init__(
        self,
        memory_episodic_path: str = "data/memory/episodic.json",
        memory_semantic_path: str = "data/memory/semantic.json",
        kg_path: str = "data/graph/knowledge_graph.json",
        mode_timeout: float = 30.0,
        max_parallel_modes: int = 2,
        enable_idle_loop: bool = False,
    ):
        self.router    = ModeRouter()
        self.executor  = ParallelExecutor(mode_timeout=mode_timeout)
        self.synthesizer = ResultSynthesizer()
        self.memory    = MemoryController(
            episodic_path=memory_episodic_path,
            semantic_path=memory_semantic_path,
            kg_path=kg_path,
        )
        self.max_parallel_modes = max_parallel_modes
        self._idle_loop = None

        if enable_idle_loop:
            self._start_idle_loop(kg_path)

    def _start_idle_loop(self, kg_path: str) -> None:
        try:
            from synapseos.phase4.idle_loop import IdleLoop
            self._idle_loop = IdleLoop(graph_path=kg_path, idle_threshold=10.0)
            self._idle_loop.start_background()
        except Exception as e:
            print(f"[CognitiveController] Idle loop start failed: {e}")

    def think(
        self,
        query: str,
        force_modes: list[str] | None = None,
    ) -> CognitionResult:
        """
        Main inference call. Routes to the correct mode(s) and returns a CognitionResult.

        Args:
            query:       Natural language input.
            force_modes: Optional list of mode names (e.g. ["mode1", "mode4"]) to bypass routing.
        """
        t0 = time.time()

        # Signal active to idle loop
        if self._idle_loop:
            self._idle_loop.signal_active()

        # 1. Retrieve memory context
        ctx = self.memory.get_context(query)
        self.memory.set_working("current_query", query)

        # 2. Route
        if force_modes:
            modes = []
            for m in force_modes:
                try:
                    modes.append(ThinkingMode(m))
                except ValueError:
                    pass
        else:
            modes = self.router.select_modes(query, max_modes=self.max_parallel_modes)

        if not modes:
            modes = [ThinkingMode.MODE1_REASONING]

        # 3. Execute in parallel
        mode_results = self.executor.run(query, modes, ctx)

        # 4. Synthesize
        primary_output, output_type, confidence = self.synthesizer.synthesize(
            query, mode_results, ctx
        )

        total_ms = (time.time() - t0) * 1000

        result = CognitionResult(
            query=query,
            modes_activated=modes,
            primary_output=primary_output,
            output_type=output_type,
            mode_results=mode_results,
            memory_context={"summary": ctx.summary},
            total_latency_ms=total_ms,
            confidence=confidence,
        )

        # 5. Record in episodic memory
        self.memory.record_interaction(
            query=query,
            response=primary_output[:500],
            mode_used="+".join(m.value for m in modes),
            tags=[output_type] + [m.value for m in modes],
        )

        return result

    def shutdown(self) -> None:
        if self._idle_loop:
            self._idle_loop.stop()
        self.memory.save()

    def stats(self) -> dict:
        s = self.memory.stats()
        s["idle_loop_active"] = self._idle_loop is not None
        return s


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SynapseOS Cognitive Controller")
    parser.add_argument("--query",  default="What fields are required for a Sales Invoice?")
    parser.add_argument("--mode",   default=None, help="Force a specific mode (mode1/mode2/mode3/mode4)")
    parser.add_argument("--idle",   action="store_true", help="Enable idle loop")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"SynapseOS Cognitive Controller")
    print(f"{'='*60}")
    print(f"Query: {args.query}\n")

    controller = CognitiveController(enable_idle_loop=args.idle)
    print(f"Memory: {controller.stats()}\n")

    force_modes = [args.mode] if args.mode else None
    result = controller.think(args.query, force_modes=force_modes)

    print(f"Modes activated  : {[m.value for m in result.modes_activated]}")
    print(f"Output type      : {result.output_type}")
    print(f"Confidence       : {result.confidence:.3f}")
    print(f"Total latency    : {result.total_latency_ms:.1f}ms")
    print(f"Memory context   : {result.memory_context['summary']}")
    print(f"\n--- Output ---")
    print(result.primary_output)

    print(f"\n--- Mode breakdown ---")
    for r in result.mode_results:
        status = "OK" if not r.error else f"ERR: {r.error[:50]}"
        print(f"  [{r.mode.value}] {status} | conf={r.confidence:.2f} | {r.latency_ms:.0f}ms | traceable={r.traceable}")

    controller.shutdown()


if __name__ == "__main__":
    main()
