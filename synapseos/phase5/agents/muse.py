"""
Phase 5 — MUSE: the creative agent.

Creative *thinking* lives in the phase4 CreativityEngine (analogies, graph
walks, idea mutation). MUSE adds creative *output* — it makes actual things:

  "compose a sad song"            → original WAV + the score
  "compose music for my mood"     → reads the affect tracker's state
  "draw/paint/generate an image of X" → procedural PNG seeded by the concept
  "make a video of X"             → animated fractal MP4, optionally scored
                                    with its own composed soundtrack
  "give me a new idea for X"      → CreativityEngine ideas (traceable)

Every creation is original by construction (algorithmic, seeded, no training
data) and is remembered: "user was_made <thing>" lands in semantic memory.
"""

from __future__ import annotations

import re

from synapseos.phase5.agents.advisors import AdvisorAgent
from synapseos.phase5.agents.base import AgentCapability, AgentResult

_MOOD_WORDS = {"happy", "sad", "calm", "energetic", "mysterious", "bluesy"}


class MUSEAgent(AdvisorAgent):
    capability = AgentCapability(
        name="MUSE",
        keywords=["compose", "music", "song", "melody", "tune", "draw",
                  "paint", "generate an image", "generate image", "create an image",
                  "make a video", "create a video", "animate", "artwork",
                  "new idea", "brainstorm", "invent", "imagine", "creative"],
        anti_keywords=["doctype", "invoice", "deploy", "rules"],
        base_confidence=0.55,
        description="Creative output: music composition, image/video generation, ideas",
    )
    domain = "creation"
    seed_topics = []

    def __init__(self) -> None:
        super().__init__()
        self._composer = None
        self._imager = None
        self._videographer = None
        self._creativity = None

    # Lazy builders — heavy imports only when actually creating
    @property
    def composer(self):
        if self._composer is None:
            from synapseos.create import MusicComposer
            self._composer = MusicComposer()
        return self._composer

    @property
    def imager(self):
        if self._imager is None:
            from synapseos.create import ImageGenerator
            self._imager = ImageGenerator()
        return self._imager

    @property
    def videographer(self):
        if self._videographer is None:
            from synapseos.create import VideoGenerator
            self._videographer = VideoGenerator()
        return self._videographer

    # -- main ------------------------------------------------------------------

    def _handle(self, query: str, context=None) -> AgentResult:
        ql = query.lower()
        trace = ["MUSE creative agent engaged"]

        if any(w in ql for w in ("compose", "music", "song", "melody", "tune")):
            return self._music(query, ql, trace)
        if any(w in ql for w in ("video", "animate", "animation")):
            return self._video(query, ql, trace)
        if any(w in ql for w in ("draw", "paint", "image", "picture", "artwork")):
            return self._image(query, ql, trace)
        return self._ideas(query, trace)

    # -- music ----------------------------------------------------------------------

    def _music(self, query: str, ql: str, trace: list[str]) -> AgentResult:
        mood = next((m for m in _MOOD_WORDS if m in ql), None)

        if mood is None and ("my mood" in ql or "how i feel" in ql or "for me" in ql) \
                and self.emotion_tracker is not None:
            state = self.emotion_tracker.state
            comp = self.composer.compose_for_state(state.valence_ema, state.arousal_ema)
            trace.append(f"mood from affect state (v={state.valence_ema:+.2f})")
        else:
            comp = self.composer.compose(mood=mood or "calm")
            trace.append(f"mood: {comp.mood}")

        self._record(f"composition:{comp.mood}")
        out = (f"[MUSE — composition]\n"
               f"Composed an original {comp.mood} piece: {comp.bars} bars in "
               f"{comp.key} {comp.scale.replace('_', ' ')}, {comp.tempo} BPM, "
               f"{comp.duration_s}s.\n"
               f"Audio: {comp.path}\n"
               f"Opening melody: {' '.join(comp.melody[:8])}\n"
               f"Chords: {' → '.join(comp.chords[:4])}\n"
               f"(Seed {comp.seed} — same seed reproduces the exact piece.)")
        return self._result(query, out, 0.85, trace,
                            {"kind": "music", **comp.to_dict()})

    # -- image ----------------------------------------------------------------------

    def _concept(self, ql: str) -> str:
        m = re.search(r"(?:of|about)\s+(.+?)\s*$", ql)
        concept = m.group(1) if m else ql
        for w in ("draw", "paint", "generate", "create", "make", "an image",
                  "a picture", "image", "picture", "video", "artwork", "a "):
            concept = concept.replace(w, " ")
        return re.sub(r"\s+", " ", concept).strip() or "abstract thought"

    def _image(self, query: str, ql: str, trace: list[str]) -> AgentResult:
        concept = self._concept(ql)
        style = next((s for s in ("flow", "fractal", "geo") if s in ql), "auto")
        img = self.imager.create(concept, style=style)
        trace.append(f"procedural {img.style} render, seed {img.seed}")
        self._record(f"image:{concept}")
        out = (f"[MUSE — image]\n"
               f"Created an original {img.style} piece for \"{concept}\" "
               f"({img.width}×{img.height}).\n"
               f"Image: {img.path}\n"
               f"Palette: {', '.join(img.palette)}\n"
               f"(Procedural art seeded by the concept — not photorealistic; "
               f"that needs a diffusion model on the neural path.)")
        return self._result(query, out, 0.8, trace, {"kind": "image", **img.to_dict()})

    # -- video ---------------------------------------------------------------------

    def _video(self, query: str, ql: str, trace: list[str]) -> AgentResult:
        concept = self._concept(ql)
        soundtrack = None
        if "music" in ql or "soundtrack" in ql or "with sound" in ql:
            soundtrack = self.composer.compose(mood="mysterious", bars=4).path
            trace.append("composed its own soundtrack")
        vid = self.videographer.create(concept, soundtrack=soundtrack)
        trace.append(f"{vid.frames} frames @ {vid.fps}fps → {vid.format}")
        self._record(f"video:{concept}")
        out = (f"[MUSE — video]\n"
               f"Created an animated piece for \"{concept}\": "
               f"{vid.duration_s}s, {vid.frames} frames, {vid.format}.\n"
               f"Video: {vid.path}"
               + (f"\nSoundtrack (also original): {vid.audio_path}" if vid.audio_path else ""))
        return self._result(query, out, 0.8, trace, {"kind": "video", **vid.to_dict()})

    # -- ideas (creative thinking via phase4) -----------------------------------------

    def _ideas(self, query: str, trace: list[str]) -> AgentResult:
        try:
            if self._creativity is None:
                from synapseos.phase4.creativity_engine import CreativityEngine
                self._creativity = CreativityEngine()
            ideas = self._creativity.generate(seed=query, top_k=3)
            trace.append(f"creativity engine produced {len(ideas)} ideas")
            lines = ["[MUSE — ideas]"]
            for i, idea in enumerate(ideas[:3], 1):
                title = getattr(idea, "title", "")
                desc = getattr(idea, "description", str(idea))
                lines.append(f"  {i}. {title}: {desc[:120]}")
            if len(lines) == 1:
                lines.append("The concept graph gave me nothing novel — feed me "
                             "more knowledge (study a subject) and ask again.")
            return self._result(query, "\n".join(lines),
                                0.6 if len(lines) > 1 else 0.35, trace, {"kind": "ideas"})
        except Exception as e:
            return self._result(query, f"[MUSE] idea generation failed: {e}", 0.2,
                                trace, {"kind": "ideas"})

    # -- helpers ----------------------------------------------------------------------

    def _record(self, what: str) -> None:
        if self.semantic_memory is not None:
            self.semantic_memory.add_fact(
                subject="user", predicate="was_made", obj=what[:60],
                confidence=0.7, source="muse_creation")

    def _result(self, query: str, output: str, confidence: float,
                trace: list[str], structured: dict) -> AgentResult:
        return AgentResult(
            agent_name="MUSE", query=query, output=output,
            structured={"domain": self.domain, **structured},
            confidence=confidence, latency_ms=0.0, traceable=True, trace=trace,
        )
