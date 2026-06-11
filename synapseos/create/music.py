"""
Create — algorithmic music composition.

Composes original melodies over chord progressions in a chosen mood, then
renders to WAV with harmonic synthesis (fundamental + overtones + ADSR
envelope). Every composition is seeded — same seed, same piece — and the
"score" (notes, chords, structure) is returned alongside the audio, so the
work is fully inspectable.

Moods map to music theory, and to the affect tracker's state:
  happy     — major scale, brighter tempo
  sad       — natural minor, slow, lower register
  calm      — pentatonic, gentle tempo, soft envelope
  energetic — mixolydian, fast, driving rhythm
  mysterious— harmonic minor, sparse
"""

from __future__ import annotations

import random
import struct
import time
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

A4 = 440.0
SAMPLE_RATE = 22050

# Scale interval patterns (semitones from root)
SCALES = {
    "major":          [0, 2, 4, 5, 7, 9, 11],
    "natural_minor":  [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "pentatonic":     [0, 2, 4, 7, 9],
    "blues":          [0, 3, 5, 6, 7, 10],
    "mixolydian":     [0, 2, 4, 5, 7, 9, 10],
}

# Chord progressions as scale degrees (I = 0)
PROGRESSIONS = {
    "pop":       [0, 5, 3, 4],       # I  vi IV V
    "folk":      [0, 3, 4, 0],       # I  IV V  I
    "emotive":   [0, 4, 5, 3],       # I  V  vi IV
    "tension":   [5, 3, 0, 4],       # vi IV I  V
}

MOODS = {
    "happy":      {"scale": "major", "tempo": 132, "octave": 0, "progression": "pop"},
    "sad":        {"scale": "natural_minor", "tempo": 76, "octave": -1, "progression": "emotive"},
    "calm":       {"scale": "pentatonic", "tempo": 88, "octave": 0, "progression": "folk"},
    "energetic":  {"scale": "mixolydian", "tempo": 150, "octave": 0, "progression": "pop"},
    "mysterious": {"scale": "harmonic_minor", "tempo": 96, "octave": -1, "progression": "tension"},
    "bluesy":     {"scale": "blues", "tempo": 104, "octave": 0, "progression": "folk"},
}

# Rhythm vocabularies: note lengths in beats
_RHYTHMS = [[1, 1, 0.5, 0.5, 1], [0.5, 0.5, 1, 2], [1, 0.5, 0.5, 1, 1],
            [2, 1, 1], [0.5, 0.5, 0.5, 0.5, 1, 1]]


@dataclass
class Composition:
    path: str
    mood: str
    key: str
    scale: str
    tempo: int
    seed: int
    bars: int
    duration_s: float
    melody: list[str] = field(default_factory=list)   # note names, the "score"
    chords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


_NOTE_NAMES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]


def _freq(semitones_from_a4: float) -> float:
    return A4 * (2.0 ** (semitones_from_a4 / 12.0))


def _note_name(semis: int) -> str:
    octave = 4 + (semis + 9) // 12      # A4-relative → scientific-ish name
    return f"{_NOTE_NAMES[semis % 12]}{octave}"


class MusicComposer:
    """Original, seeded composition + harmonic WAV synthesis."""

    def __init__(self, out_dir: str = "data/creations/music"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # -- composition ---------------------------------------------------------------

    def compose(self, mood: str = "calm", bars: int = 8,
                seed: int | None = None, key_offset: int = 0) -> Composition:
        mood = mood if mood in MOODS else "calm"
        cfg = MOODS[mood]
        seed = seed if seed is not None else int(time.time() * 1000) % 999_983
        rng = random.Random(seed)

        scale = SCALES[cfg["scale"]]
        beats_per_bar = 4
        spb = 60.0 / cfg["tempo"]                       # seconds per beat
        root = key_offset + 12 * cfg["octave"]          # semitones from A4
        progression = PROGRESSIONS[cfg["progression"]]

        melody_events: list[tuple[float, float, float]] = []  # (freq, start, dur_s)
        chord_events: list[tuple[list[float], float, float]] = []
        melody_names, chord_names = [], []

        t = 0.0
        degree = 0
        for bar in range(bars):
            chord_degree = progression[bar % len(progression)]
            # Triad: degrees chord, chord+2, chord+4 of the scale
            triad_semis = [root + scale[(chord_degree + k) % len(scale)]
                           + 12 * ((chord_degree + k) // len(scale)) - 12
                           for k in (0, 2, 4)]
            chord_events.append(([_freq(s) for s in triad_semis], t, beats_per_bar * spb))
            chord_names.append(_note_name(triad_semis[0] % 12))

            # Melody: random walk over the scale, biased toward chord tones
            rhythm = rng.choice(_RHYTHMS)
            beat = 0.0
            for length in rhythm:
                if beat >= beats_per_bar:
                    break
                step = rng.choice([-2, -1, -1, 0, 1, 1, 2, 3])
                degree = max(-3, min(len(scale) * 2, degree + step))
                if rng.random() < 0.35:                  # pull toward chord tone
                    degree = chord_degree + rng.choice([0, 2, 4])
                semis = root + scale[degree % len(scale)] + 12 * (degree // len(scale))
                dur = min(length, beats_per_bar - beat) * spb
                melody_events.append((_freq(semis), t + beat * spb, dur))
                melody_names.append(_note_name(semis % 12 + 12))
                beat += length
            t += beats_per_bar * spb

        total = t + 0.5
        audio = self._render(melody_events, chord_events, total)

        fname = f"{mood}_{seed}.wav"
        path = self.out_dir / fname
        self._write_wav(path, audio)

        return Composition(
            path=str(path), mood=mood, key=_note_name(root % 12),
            scale=cfg["scale"], tempo=cfg["tempo"], seed=seed, bars=bars,
            duration_s=round(total, 2), melody=melody_names, chords=chord_names,
        )

    def compose_for_state(self, valence: float, arousal: float,
                          bars: int = 8, seed: int | None = None) -> Composition:
        """Map an emotional state (from the affect tracker) to a mood."""
        if valence >= 0.1:
            mood = "energetic" if arousal > 0.55 else "happy" if arousal > 0.35 else "calm"
        elif valence <= -0.1:
            mood = "mysterious" if arousal > 0.55 else "sad"
        else:
            mood = "bluesy" if arousal > 0.5 else "calm"
        return self.compose(mood=mood, bars=bars, seed=seed)

    # -- synthesis ----------------------------------------------------------------

    @staticmethod
    def _tone(freq: float, dur: float, amp: float, soft: bool = False) -> np.ndarray:
        n = max(1, int(dur * SAMPLE_RATE))
        t = np.arange(n) / SAMPLE_RATE
        # Fundamental + overtones = warmer than a bare sine
        wave_ = (np.sin(2 * np.pi * freq * t)
                 + 0.4 * np.sin(2 * np.pi * 2 * freq * t)
                 + 0.15 * np.sin(2 * np.pi * 3 * freq * t))
        # ADSR envelope
        a = int(n * (0.10 if soft else 0.04))
        r = int(n * 0.3)
        env = np.ones(n)
        if a > 0:
            env[:a] = np.linspace(0, 1, a)
        if r > 0:
            env[-r:] = np.linspace(1, 0, r)
        return amp * env * wave_

    def _render(self, melody, chords, total_s: float) -> np.ndarray:
        buf = np.zeros(int(total_s * SAMPLE_RATE) + SAMPLE_RATE // 2)

        def add(sig: np.ndarray, start_s: float) -> None:
            i = int(start_s * SAMPLE_RATE)
            buf[i:i + len(sig)] += sig[: len(buf) - i]

        for freqs, start, dur in chords:
            for f in freqs:
                add(self._tone(f, dur, amp=0.10, soft=True), start)
        for f, start, dur in melody:
            add(self._tone(f, dur * 0.95, amp=0.28), start)

        peak = np.abs(buf).max() or 1.0
        return buf / peak * 0.85

    @staticmethod
    def _write_wav(path: Path, audio: np.ndarray) -> None:
        pcm = (audio * 32000).astype("<i2").tobytes()
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm)
