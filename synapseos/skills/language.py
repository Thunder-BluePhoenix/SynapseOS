"""
Skills — language detection + phrasebook. Deterministic, honest scope.

What this does:
  - detect the language/script of text via Unicode ranges (Bengali,
    Devanagari/Hindi, Japanese, Korean, Arabic, Cyrillic, CJK) and stopword
    profiles for Latin-script languages (English, Spanish, French, German)
  - a small built-in phrasebook for greetings/courtesy in each language

What this deliberately does NOT do: translation or fluent generation —
that genuinely requires the neural sequence model (Mamba GPU fine-tuning
path). Deeper vocabulary grows via the web learning loop instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Unicode script ranges → language guess
_SCRIPTS: list[tuple[str, str, range]] = [
    ("bengali",    "bn", range(0x0980, 0x09FF)),
    ("hindi",      "hi", range(0x0900, 0x097F)),   # Devanagari (also Marathi etc.)
    ("japanese",   "ja", range(0x3040, 0x30FF)),   # Hiragana + Katakana
    ("korean",     "ko", range(0xAC00, 0xD7AF)),
    ("arabic",     "ar", range(0x0600, 0x06FF)),
    ("russian",    "ru", range(0x0400, 0x04FF)),   # Cyrillic
    ("chinese",    "zh", range(0x4E00, 0x9FFF)),   # CJK (ja text may also hit; kana checked first)
    ("greek",      "el", range(0x0370, 0x03FF)),
    ("tamil",      "ta", range(0x0B80, 0x0BFF)),
    ("telugu",     "te", range(0x0C00, 0x0C7F)),
]

# Latin-script languages: distinctive stopwords
_LATIN_PROFILES: dict[str, set[str]] = {
    "english": {"the", "and", "is", "of", "to", "in", "that", "for", "with", "are"},
    "spanish": {"el", "la", "los", "las", "es", "de", "que", "y", "en", "por", "una", "está"},
    "french":  {"le", "la", "les", "est", "de", "que", "et", "dans", "pour", "une", "avec"},
    "german":  {"der", "die", "das", "und", "ist", "von", "mit", "ein", "für", "nicht"},
}

PHRASEBOOK: dict[str, dict[str, str]] = {
    "bengali":  {"hello": "নমস্কার (nomoshkar)", "thank you": "ধন্যবাদ (dhonnobad)",
                 "goodbye": "বিদায় (biday)", "yes": "হ্যাঁ (hyã)", "no": "না (na)"},
    "hindi":    {"hello": "नमस्ते (namaste)", "thank you": "धन्यवाद (dhanyavaad)",
                 "goodbye": "अलविदा (alvida)", "yes": "हाँ (haan)", "no": "नहीं (nahin)"},
    "spanish":  {"hello": "hola", "thank you": "gracias", "goodbye": "adiós",
                 "yes": "sí", "no": "no"},
    "japanese": {"hello": "こんにちは (konnichiwa)", "thank you": "ありがとう (arigatou)",
                 "goodbye": "さようなら (sayounara)", "yes": "はい (hai)", "no": "いいえ (iie)"},
    "french":   {"hello": "bonjour", "thank you": "merci", "goodbye": "au revoir",
                 "yes": "oui", "no": "non"},
    "english":  {"hello": "hello", "thank you": "thank you", "goodbye": "goodbye",
                 "yes": "yes", "no": "no"},
}


@dataclass
class LanguageGuess:
    language: str        # "bengali" | "hindi" | ... | "unknown"
    code: str            # ISO-ish short code
    confidence: float    # 0..1
    method: str          # "script" | "stopwords" | "none"

    def to_dict(self) -> dict:
        return {"language": self.language, "code": self.code,
                "confidence": round(self.confidence, 3), "method": self.method}


def detect_language(text: str) -> LanguageGuess:
    """Best-effort language identification. Script first, stopwords second."""
    if not text.strip():
        return LanguageGuess("unknown", "", 0.0, "none")

    # 1. Script detection — count chars per script
    counts: dict[str, int] = {}
    total_alpha = 0
    for ch in text:
        cp = ord(ch)
        if ch.isalpha():
            total_alpha += 1
        for name, code, rng in _SCRIPTS:
            if cp in rng:
                counts[name] = counts.get(name, 0) + 1
                break
    if counts and total_alpha:
        best = max(counts, key=counts.get)
        share = counts[best] / total_alpha
        if share > 0.3:
            code = next(c for n, c, _ in _SCRIPTS if n == best)
            return LanguageGuess(best, code, min(0.95, 0.5 + share / 2), "script")

    # 2. Latin-script stopword profiles
    words = set(re.findall(r"[a-záéíóúüñàèìòùâêîôûäöß]+", text.lower()))
    if not words:
        return LanguageGuess("unknown", "", 0.0, "none")
    scores = {lang: len(words & profile) for lang, profile in _LATIN_PROFILES.items()}
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return LanguageGuess("unknown", "", 0.0, "none")
    codes = {"english": "en", "spanish": "es", "french": "fr", "german": "de"}
    confidence = min(0.9, 0.3 + 0.12 * scores[best])
    return LanguageGuess(best, codes[best], confidence, "stopwords")


def phrase(language: str, key: str) -> str | None:
    """Look up a courtesy phrase: phrase('bengali', 'thank you')."""
    return PHRASEBOOK.get(language.lower(), {}).get(key.lower())
