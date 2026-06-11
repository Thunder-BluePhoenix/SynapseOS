"""
Affect — Level 3: respond appropriately.

Detection (level 1) and state tracking (level 2) feed this: the output of
think() is modulated by the user's current emotional state. Deterministic
rules, fully auditable — each adjustment is reported in the result.

Behaviors:
  frustrated/angry  → brief empathic acknowledgement first, trim verbosity
  stressed + risky decision → caution: defer the irreversible choice
                      ("he's been stressed all week; bad time for a risky idea")
  sad/subdued       → gentler phrasing, no exclamation-heavy cheer
  excited + risky decision → gentle "validate before committing" nudge
                      (impulse-buying guard, the flip side of stress-gating)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Words signalling a consequential / hard-to-reverse decision
_RISKY_RE = re.compile(
    r"\b(invest|buy|sell|quit|resign|delete|drop|launch|sign|borrow|loan|"
    r"bet|gamble|migrate|rewrite|fire|spend|commit|all.?in)\b", re.IGNORECASE)

_ACK = {
    "frustration": "I can tell this has been frustrating — let's sort it out.",
    "anger": "I hear the frustration. Let's fix the actual problem.",
    "sadness": "Sounds like a heavy moment. Here's what I have, gently:",
    "fear": "I sense some worry here — let's look at this calmly.",
    "hesitation": "No rush — let's think it through together.",
}


@dataclass
class ModulationResult:
    output: str
    adjustments: list[str] = field(default_factory=list)   # audit trail

    @property
    def modified(self) -> bool:
        return bool(self.adjustments)


class ResponseModulator:
    """Affect-conditioned output shaping. Stateless; reads tracker state."""

    def __init__(self, stress_valence: float = -0.25, stress_arousal: float = 0.5,
                 trim_threshold: int = 1200):
        self.stress_valence = stress_valence
        self.stress_arousal = stress_arousal
        self.trim_threshold = trim_threshold

    def modulate(self, output: str, affect, state, query: str) -> ModulationResult:
        """
        affect: AffectResult for THIS message (may be None/neutral)
        state:  EmotionalState — the running EMA (the week, not the moment)
        """
        adjustments: list[str] = []
        risky = bool(_RISKY_RE.search(query))
        emotion = getattr(affect, "dominant_emotion", "neutral") if affect else "neutral"
        valence_now = getattr(affect, "valence", 0.0) if affect else 0.0

        # 1. Empathic acknowledgement when this message carries clear negative affect
        ack = _ACK.get(emotion)
        if ack and valence_now < -0.2:
            output = f"{ack}\n\n{output}"
            adjustments.append(f"empathic_ack:{emotion}")

        # 2. Sustained stress + risky decision → suggest deferring
        stressed = (state is not None
                    and state.observations >= 3
                    and state.valence_ema < self.stress_valence
                    and state.arousal_ema > self.stress_arousal)
        if stressed and risky:
            output += (
                "\n\n[Companion note] You've seemed stressed lately "
                f"(mood {state.valence_ema:+.2f} over {state.observations} interactions). "
                "This looks like a consequential decision — consider sleeping on it. "
                "I'll still help you do it now if you want."
            )
            adjustments.append("risk_deferral:stress")

        # 3. High excitement + risky decision → validate-first nudge
        excited = (affect is not None and emotion == "excitement"
                   and getattr(affect, "arousal", 0.0) > 0.6)
        if excited and risky and "risk_deferral:stress" not in adjustments:
            output += (
                "\n\n[Companion note] Love the energy — before committing, "
                "want me to research this first so the decision is grounded?"
            )
            adjustments.append("validate_first:excitement")

        # 4. Frustrated users get the short version
        if emotion in ("frustration", "anger") and len(output) > self.trim_threshold:
            cut = output[: self.trim_threshold]
            cut = cut[: cut.rfind("\n")] if "\n" in cut else cut
            output = cut + "\n[Trimmed — you seem busy/frustrated. Ask for the full detail anytime.]"
            adjustments.append("trimmed:frustration")

        return ModulationResult(output=output, adjustments=adjustments)
