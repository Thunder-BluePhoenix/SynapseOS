"""
Affect — text emotion detection. Lexicon + pattern based, no LLM.

Outputs the valence/arousal model standard in affective computing plus
discrete emotion scores. Every detection keeps its evidence (which words
fired), so the result is fully auditable — same philosophy as the fact
extractor.

Handles: negation ("not happy"), intensifiers ("really angry"),
diminishers ("a bit sad"), ALL-CAPS emphasis, punctuation (!!, ?!, ...),
and common emoticons/emoji.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

# ---------------------------------------------------------------------------
# Lexicon: word -> (valence [-1..1], arousal [0..1], emotion label)
# ---------------------------------------------------------------------------

LEXICON: dict[str, tuple[float, float, str]] = {
    # joy
    "happy": (0.8, 0.5, "joy"), "glad": (0.7, 0.4, "joy"), "joy": (0.9, 0.6, "joy"),
    "delighted": (0.9, 0.7, "joy"), "wonderful": (0.9, 0.6, "joy"),
    "great": (0.7, 0.5, "joy"), "awesome": (0.9, 0.7, "joy"),
    "love": (0.9, 0.6, "joy"), "enjoy": (0.7, 0.5, "joy"), "fun": (0.7, 0.6, "joy"),
    "perfect": (0.8, 0.5, "joy"), "beautiful": (0.8, 0.5, "joy"),
    "proud": (0.8, 0.5, "joy"), "pleased": (0.7, 0.4, "joy"),
    # excitement
    "excited": (0.8, 0.9, "excitement"), "thrilled": (0.9, 0.9, "excitement"),
    "amazing": (0.9, 0.8, "excitement"), "incredible": (0.8, 0.8, "excitement"),
    "fantastic": (0.9, 0.8, "excitement"), "wow": (0.7, 0.8, "excitement"),
    "finally": (0.5, 0.7, "excitement"), "yes": (0.5, 0.6, "excitement"),
    # gratitude
    "thanks": (0.7, 0.3, "gratitude"), "thank": (0.7, 0.3, "gratitude"),
    "grateful": (0.8, 0.3, "gratitude"), "appreciate": (0.7, 0.3, "gratitude"),
    # sadness
    "sad": (-0.7, 0.3, "sadness"), "unhappy": (-0.7, 0.3, "sadness"),
    "depressed": (-0.9, 0.2, "sadness"), "miserable": (-0.9, 0.4, "sadness"),
    "disappointed": (-0.7, 0.4, "sadness"), "lonely": (-0.7, 0.3, "sadness"),
    "cry": (-0.7, 0.5, "sadness"), "lost": (-0.5, 0.3, "sadness"),
    "miss": (-0.5, 0.3, "sadness"), "hopeless": (-0.9, 0.3, "sadness"),
    "tired": (-0.4, 0.2, "sadness"), "exhausted": (-0.6, 0.3, "sadness"),
    # anger
    "angry": (-0.8, 0.8, "anger"), "furious": (-0.9, 0.9, "anger"),
    "hate": (-0.9, 0.8, "anger"), "mad": (-0.7, 0.8, "anger"),
    "outrageous": (-0.8, 0.8, "anger"), "unacceptable": (-0.7, 0.7, "anger"),
    "ridiculous": (-0.7, 0.7, "anger"), "terrible": (-0.8, 0.6, "anger"),
    "awful": (-0.8, 0.6, "anger"), "worst": (-0.8, 0.6, "anger"),
    "stupid": (-0.7, 0.7, "anger"), "damn": (-0.6, 0.7, "anger"),
    # frustration
    "frustrated": (-0.7, 0.7, "frustration"), "frustrating": (-0.7, 0.7, "frustration"),
    "annoying": (-0.6, 0.6, "frustration"), "annoyed": (-0.6, 0.6, "frustration"),
    "stuck": (-0.5, 0.5, "frustration"), "again": (-0.2, 0.4, "frustration"),
    "broken": (-0.6, 0.5, "frustration"), "fails": (-0.6, 0.5, "frustration"),
    "failed": (-0.6, 0.5, "frustration"), "failing": (-0.6, 0.5, "frustration"),
    "error": (-0.4, 0.4, "frustration"), "bug": (-0.4, 0.4, "frustration"),
    "wrong": (-0.5, 0.5, "frustration"), "impossible": (-0.6, 0.6, "frustration"),
    "useless": (-0.7, 0.6, "frustration"), "waste": (-0.6, 0.5, "frustration"),
    "ugh": (-0.6, 0.6, "frustration"), "argh": (-0.6, 0.7, "frustration"),
    # fear / anxiety
    "afraid": (-0.7, 0.7, "fear"), "scared": (-0.7, 0.8, "fear"),
    "worried": (-0.6, 0.6, "fear"), "anxious": (-0.6, 0.7, "fear"),
    "nervous": (-0.5, 0.7, "fear"), "panic": (-0.8, 0.9, "fear"),
    "terrified": (-0.9, 0.9, "fear"), "risk": (-0.3, 0.5, "fear"),
    "dangerous": (-0.6, 0.6, "fear"), "stress": (-0.6, 0.7, "fear"),
    "stressed": (-0.6, 0.7, "fear"), "overwhelmed": (-0.7, 0.7, "fear"),
    # hesitation / uncertainty
    "maybe": (0.0, 0.2, "hesitation"), "unsure": (-0.2, 0.3, "hesitation"),
    "confused": (-0.3, 0.4, "hesitation"), "doubt": (-0.3, 0.3, "hesitation"),
    "hmm": (0.0, 0.2, "hesitation"), "perhaps": (0.0, 0.2, "hesitation"),
    "dunno": (-0.1, 0.2, "hesitation"), "hesitant": (-0.2, 0.3, "hesitation"),
}

EMOTICONS: dict[str, tuple[float, float, str]] = {
    ":)": (0.7, 0.4, "joy"), ":-)": (0.7, 0.4, "joy"), ":d": (0.9, 0.7, "joy"),
    ":(": (-0.7, 0.3, "sadness"), ":-(": (-0.7, 0.3, "sadness"),
    ":/": (-0.4, 0.3, "frustration"), ":|": (-0.2, 0.2, "hesitation"),
    "😀": (0.8, 0.6, "joy"), "😊": (0.8, 0.5, "joy"), "😂": (0.9, 0.8, "joy"),
    "❤": (0.9, 0.6, "joy"), "🎉": (0.9, 0.8, "excitement"),
    "😢": (-0.8, 0.4, "sadness"), "😭": (-0.8, 0.6, "sadness"),
    "😡": (-0.9, 0.9, "anger"), "😤": (-0.6, 0.7, "frustration"),
    "😱": (-0.7, 0.9, "fear"), "😰": (-0.6, 0.7, "fear"),
}

NEGATORS = {"not", "no", "never", "isn't", "wasn't", "don't", "doesn't",
            "didn't", "can't", "won't", "wouldn't", "couldn't", "shouldn't",
            "aint", "ain't", "without", "hardly", "barely"}
INTENSIFIERS = {"very": 1.5, "really": 1.4, "so": 1.3, "extremely": 1.8,
                "incredibly": 1.7, "absolutely": 1.6, "totally": 1.5,
                "completely": 1.5, "super": 1.4, "too": 1.3}
DIMINISHERS = {"slightly": 0.5, "somewhat": 0.6, "bit": 0.5, "little": 0.6,
               "kinda": 0.6, "kind": 0.7, "fairly": 0.7, "barely": 0.4}

_WORD_RE = re.compile(r"[a-z']+|[:;][-]?[)(d/|]|[\U0001F300-\U0001FAFF❤]")


@dataclass
class AffectResult:
    valence: float                    # -1 (negative) .. +1 (positive)
    arousal: float                    # 0 (calm) .. 1 (agitated)
    dominant_emotion: str             # joy | anger | ... | neutral
    emotions: dict[str, float] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)   # words that fired
    confidence: float = 0.0           # how much signal was found

    @property
    def is_neutral(self) -> bool:
        return self.dominant_emotion == "neutral"

    def to_dict(self) -> dict:
        return asdict(self)

    def as_text(self) -> str:
        if self.is_neutral:
            return "neutral"
        v = "positive" if self.valence > 0 else "negative"
        a = "high-energy" if self.arousal > 0.55 else "low-energy"
        return f"{self.dominant_emotion} ({v}, {a})"


class TextAffectDetector:
    """Deterministic affect detection. Evidence-preserving, no LLM."""

    def detect(self, text: str) -> AffectResult:
        tokens = _WORD_RE.findall(text.lower())
        hits: list[tuple[float, float, str, str]] = []   # (val, aro, emotion, word)

        for i, tok in enumerate(tokens):
            entry = LEXICON.get(tok) or EMOTICONS.get(tok)
            if entry is None:
                continue
            val, aro, emotion = entry

            # Modifier window: up to 3 tokens back
            window = tokens[max(0, i - 3):i]
            mult = 1.0
            for w in window:
                if w in INTENSIFIERS:
                    mult *= INTENSIFIERS[w]
                elif w in DIMINISHERS:
                    mult *= DIMINISHERS[w]
            if any(w in NEGATORS for w in window):
                val = -val * 0.7          # "not happy" ≈ mildly negative
                aro *= 0.8
                emotion = self._flip_emotion(emotion, val)

            hits.append((max(-1, min(1, val * mult)),
                         max(0, min(1, aro * mult)), emotion, tok))

        # Non-lexical signals
        exclaims = text.count("!")
        caps_words = re.findall(r"\b[A-Z]{3,}\b", text)
        ellipsis = "..." in text or "…" in text
        arousal_boost = min(0.3, 0.1 * exclaims) + min(0.2, 0.1 * len(caps_words))

        if not hits:
            arousal = min(1.0, arousal_boost + (0.1 if ellipsis else 0.0))
            return AffectResult(valence=0.0, arousal=arousal,
                                dominant_emotion="neutral", confidence=0.0)

        valence = sum(h[0] for h in hits) / len(hits)
        arousal = min(1.0, sum(h[1] for h in hits) / len(hits) + arousal_boost)

        emotions: dict[str, float] = {}
        for val, aro, emotion, _ in hits:
            emotions[emotion] = emotions.get(emotion, 0.0) + abs(val)
        if ellipsis:
            emotions["hesitation"] = emotions.get("hesitation", 0.0) + 0.3
        total = sum(emotions.values()) or 1.0
        emotions = {k: round(v / total, 3) for k, v in emotions.items()}

        dominant = max(emotions, key=emotions.get)
        # signal strength: more evidence relative to text length = more confident
        confidence = min(0.95, 0.3 + 0.15 * len(hits))

        return AffectResult(
            valence=round(valence, 3),
            arousal=round(arousal, 3),
            dominant_emotion=dominant,
            emotions=emotions,
            evidence=[h[3] for h in hits],
            confidence=round(confidence, 3),
        )

    @staticmethod
    def _flip_emotion(emotion: str, new_valence: float) -> str:
        """Negated emotion words land in the opposite-valence bucket."""
        if new_valence < 0 and emotion in ("joy", "excitement", "gratitude"):
            return "sadness"
        if new_valence > 0 and emotion in ("sadness", "anger", "frustration", "fear"):
            return "joy"
        return emotion
