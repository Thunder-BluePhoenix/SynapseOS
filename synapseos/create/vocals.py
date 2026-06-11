"""
Create — singing voice via formant synthesis.

A sung vowel = a pitched glottal source (harmonic-rich pulse at the melody
note) shaped by the resonances of the vocal tract (formants). Each vowel has
characteristic formant frequencies; sweeping the source pitch under fixed
formants is what singing IS, acoustically. Implemented with numpy + scipy
biquad resonators — no neural vocoder, fully deterministic.

Emotion shapes the voice the way it does in a human singer:
  sad      → slower deeper vibrato, darker formants, softer attack
  joyful   → brighter formants, light fast vibrato
  energetic→ stronger vibrato, harder onsets, louder
  calm     → minimal vibrato, gentle dynamics
  devotion → long sustained notes, ornamented onsets (meend-like slides)

Honest scope: this sings VOWELS beautifully enough to carry a melody
(vocalise / sargam style) and approximates syllables by their vowel core +
a brief consonant noise. Crisp intelligible lyrics need a neural vocal model
— the lyrics are therefore always returned as text alongside the audio.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

try:
    from scipy.signal import lfilter as _lfilter
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

    def _lfilter(b, a, x):
        """Pure-numpy 2nd-order IIR (direct form I) — scipy-free fallback."""
        y = np.zeros_like(x)
        b0 = b[0]
        a1, a2 = a[1], a[2]
        y1 = y2 = 0.0
        for i in range(len(x)):
            yi = b0 * x[i] - a1 * y1 - a2 * y2
            y2, y1 = y1, yi
            y[i] = yi
        return y

SAMPLE_RATE = 22050

# Vowel → (F1, F2, F3) formant frequencies (adult voice, Hz)
FORMANTS = {
    "a": (800, 1200, 2500),    # as in "father"
    "e": (450, 1950, 2600),
    "i": (300, 2300, 3000),
    "o": (450, 850, 2500),
    "u": (350, 700, 2400),
}

EMOTION_VOICE = {
    "sad":       {"vibrato_hz": 4.6, "vibrato_depth": 0.30, "brightness": 0.75,
                  "attack": 0.12, "amp": 0.8},
    "joyful":    {"vibrato_hz": 5.8, "vibrato_depth": 0.18, "brightness": 1.15,
                  "attack": 0.04, "amp": 1.0},
    "energetic": {"vibrato_hz": 6.2, "vibrato_depth": 0.35, "brightness": 1.2,
                  "attack": 0.02, "amp": 1.1},
    "calm":      {"vibrato_hz": 4.8, "vibrato_depth": 0.10, "brightness": 0.95,
                  "attack": 0.10, "amp": 0.85},
    "devotion":  {"vibrato_hz": 5.0, "vibrato_depth": 0.22, "brightness": 0.9,
                  "attack": 0.15, "amp": 0.9},
}


@dataclass
class SungPhrase:
    audio: "np.ndarray"
    syllables: list[str]


def _resonator(signal: np.ndarray, freq: float, bandwidth: float,
               rate: int) -> np.ndarray:
    """Two-pole bandpass resonator — one formant."""
    r = np.exp(-np.pi * bandwidth / rate)
    theta = 2 * np.pi * freq / rate
    a = [1.0, -2 * r * np.cos(theta), r * r]
    b = [1.0 - r]
    return _lfilter(b, a, signal)


def _glottal_source(freq_curve: np.ndarray, rate: int) -> np.ndarray:
    """Harmonic-rich source: integrated sawtooth-ish pulse train."""
    phase = 2 * np.pi * np.cumsum(freq_curve) / rate
    sig = np.zeros_like(phase)
    for k in range(1, 13):
        sig += np.sin(k * phase) / k          # sawtooth series = buzzy glottis
    return sig


def _vowel_of(syllable: str) -> str:
    m = re.search(r"[aeiou]", syllable.lower())
    if not m:
        return "a"
    return {"a": "a", "e": "e", "i": "i", "o": "o", "u": "u"}[m.group(0)]


def sing_note(freq: float, dur: float, syllable: str = "aa",
              emotion: str = "calm", rate: int = SAMPLE_RATE,
              slide_from: float | None = None) -> np.ndarray:
    """Sing one syllable on one note."""
    cfg = EMOTION_VOICE.get(emotion, EMOTION_VOICE["calm"])
    n = max(1, int(dur * rate))
    t = np.arange(n) / rate

    # Pitch curve: optional meend (slide in), then vibrato around the note
    vib = cfg["vibrato_depth"] * np.sin(2 * np.pi * cfg["vibrato_hz"] * t)
    # vibrato fades in after the onset, like a real singer
    vib *= np.clip((t - 0.12) / 0.2, 0, 1)
    freq_curve = freq * (2.0 ** (vib / 12.0))
    if slide_from is not None and n > 100:
        glide_n = min(n // 3, int(0.15 * rate))
        glide = np.linspace(slide_from, freq, glide_n)
        freq_curve[:glide_n] = glide * (2.0 ** (vib[:glide_n] / 12.0))

    source = _glottal_source(freq_curve, rate)

    # Consonant onset: short noise burst for plosives/fricatives
    cons = syllable.strip().lower()
    if cons and cons[0] not in "aeiou":
        burst_n = int(0.025 * rate)
        rng = np.random.default_rng(abs(hash(cons[0])) % 2**31)
        source[:burst_n] += 1.5 * rng.standard_normal(min(burst_n, n))

    # Vocal tract: sum of formant resonators
    vowel = _vowel_of(syllable)
    f1, f2, f3 = FORMANTS[vowel]
    b = cfg["brightness"]
    voiced = (1.0 * _resonator(source, f1 * b, 90, rate)
              + 0.7 * _resonator(source, f2 * b, 110, rate)
              + 0.3 * _resonator(source, f3 * b, 160, rate))

    a_len = max(1, int(n * cfg["attack"]))
    r_len = max(1, int(n * 0.25))
    env = np.ones(n)
    env[:a_len] = np.linspace(0, 1, a_len)
    env[-r_len:] = np.minimum(env[-r_len:], np.linspace(1, 0, r_len))
    peak = np.abs(voiced).max() or 1.0
    return cfg["amp"] * 0.5 * env * voiced / peak


def sing_melody(notes: list[tuple[float, float, float]],
                syllables: list[str], emotion: str = "calm",
                total_s: float | None = None,
                rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Sing syllables over a melody.
    notes: list of (freq_hz, start_s, dur_s). Syllables cycle if fewer than notes.
    """
    if not notes:
        return np.zeros(rate)
    end = total_s or max(s + d for _, s, d in notes) + 0.5
    buf = np.zeros(int(end * rate) + rate // 4)
    prev_freq: float | None = None
    for i, (freq, start, dur) in enumerate(notes):
        syl = syllables[i % len(syllables)] if syllables else "aa"
        # devotion/sad styles slide between notes (meend)
        slide = prev_freq if emotion in ("sad", "devotion") and prev_freq else None
        sig = sing_note(freq, dur * 0.98, syl, emotion, rate, slide_from=slide)
        idx = int(start * rate)
        buf[idx:idx + len(sig)] += sig[: len(buf) - idx]
        prev_freq = freq
    return buf
