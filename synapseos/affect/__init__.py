"""
SynapseOS Affect — non-LLM emotion understanding.

Three layers, all deterministic and auditable (no neural language model):

  text_affect — lexicon + pattern detection on text: valence, arousal,
                discrete emotions (joy, anger, frustration, ...), with the
                matched words kept as evidence
  prosody     — acoustic analysis of speech WAVs: energy, pitch variance,
                speaking rate, pauses → arousal estimate (tone often
                contradicts words; this catches that)
  tracker     — the part LLM chatbots don't have: emotional state tracked
                over time (EMA), episodes tagged with affect, and learned
                patterns ("user frustrated_by X") written to semantic memory
                through the same belief pipeline as web facts
"""

from synapseos.affect.text_affect import AffectResult, TextAffectDetector
from synapseos.affect.prosody import ProsodyResult, analyze_wav
from synapseos.affect.tracker import EmotionalState, EmotionTracker
from synapseos.affect.responder import ModulationResult, ResponseModulator

__all__ = [
    "ModulationResult",
    "ResponseModulator",
    "AffectResult",
    "TextAffectDetector",
    "ProsodyResult",
    "analyze_wav",
    "EmotionalState",
    "EmotionTracker",
]
