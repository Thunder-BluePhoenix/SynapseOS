"""
Create — instrument timbres + tabla/tanpura synthesis.

A timbre is a recipe: harmonic amplitude profile + envelope shape + optional
vibrato/detune. The same melody played through different recipes becomes a
different instrument — all numpy, all deterministic.

Percussion (tabla) is synthesized physically-ish: a tuned resonant 'tabla'
strike (decaying sine with pitch bend) for the dayan, a deeper bend for the
bayan, and filtered noise for closed strokes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SAMPLE_RATE = 22050


@dataclass
class Timbre:
    name: str
    harmonics: list[float]          # amplitude per overtone (1st = fundamental)
    attack: float = 0.04            # fraction of note length
    release: float = 0.3
    vibrato_hz: float = 0.0
    vibrato_depth: float = 0.0      # semitone fraction
    breath: float = 0.0             # added filtered-noise level
    detune: float = 0.0             # second voice detune in Hz (chorus/jawari)


TIMBRES: dict[str, Timbre] = {
    "piano":  Timbre("piano", [1.0, 0.4, 0.15], attack=0.02, release=0.4),
    "sitar":  Timbre("sitar", [1.0, 0.72, 0.55, 0.4, 0.3, 0.22, 0.15],
                     attack=0.01, release=0.7, detune=1.5),
    "flute":  Timbre("flute", [1.0, 0.12, 0.05], attack=0.12, release=0.25,
                     vibrato_hz=5.0, vibrato_depth=0.06, breath=0.02),
    "violin": Timbre("violin", [1.0, 0.55, 0.35, 0.22, 0.12], attack=0.18,
                     release=0.2, vibrato_hz=5.5, vibrato_depth=0.1),
    "veena":  Timbre("veena", [1.0, 0.6, 0.45, 0.3, 0.18], attack=0.02,
                     release=0.6, detune=1.0),
    "synth":  Timbre("synth", [1.0, 0.5, 0.33, 0.25, 0.2, 0.16], attack=0.05,
                     release=0.2),
}


def render_note(freq: float, dur: float, amp: float, timbre: Timbre,
                rate: int = SAMPLE_RATE) -> np.ndarray:
    """One note through a timbre recipe."""
    n = max(1, int(dur * rate))
    t = np.arange(n) / rate

    if timbre.vibrato_hz > 0:
        vib = timbre.vibrato_depth * np.sin(2 * np.pi * timbre.vibrato_hz * t)
        inst_freq = freq * (2.0 ** (vib / 12.0))
        phase = 2 * np.pi * np.cumsum(inst_freq) / rate
    else:
        phase = 2 * np.pi * freq * t

    sig = np.zeros(n)
    for k, h in enumerate(timbre.harmonics, start=1):
        sig += h * np.sin(k * phase)
    if timbre.detune:
        sig += 0.5 * np.sin(2 * np.pi * (freq + timbre.detune) * t)
    if timbre.breath:
        noise = np.random.default_rng(int(freq)).standard_normal(n)
        sig += timbre.breath * noise

    a = max(1, int(n * timbre.attack))
    r = max(1, int(n * timbre.release))
    env = np.ones(n)
    env[:a] = np.linspace(0, 1, a)
    env[-r:] = np.minimum(env[-r:], np.linspace(1, 0, r))
    peak = np.abs(sig).max() or 1.0
    return amp * env * sig / peak


# ---------------------------------------------------------------------------
# Tabla + tanpura
# ---------------------------------------------------------------------------

def tabla_stroke(bol: str, dur: float = 0.18, rate: int = SAMPLE_RATE,
                 sa_freq: float = 220.0) -> np.ndarray:
    """One tabla bol. dha/dhin = both drums, na/ta/tin = dayan, ge/ke = bayan."""
    n = max(1, int(dur * rate))
    t = np.arange(n) / rate
    rng = np.random.default_rng(abs(hash(bol)) % 2**31)
    out = np.zeros(n)

    open_dayan = bol in ("na", "ta", "tin", "dha", "dhin", "ti", "tu")
    bayan = bol in ("ge", "ga", "ke", "dha", "dhin", "dhi")

    if open_dayan:
        # tuned strike near Sa with fast decay
        f = sa_freq * (2 if bol in ("tin", "ti") else 1)
        out += np.sin(2 * np.pi * f * t) * np.exp(-t * 28)
        out += 0.4 * np.sin(2 * np.pi * f * 1.5 * t) * np.exp(-t * 40)
    if bayan:
        # deep pitch-bent boom: frequency slides down
        f0 = 110.0
        inst = f0 * (1 - 0.35 * np.clip(t / dur, 0, 1))
        phase = 2 * np.pi * np.cumsum(inst) / rate
        out += 0.9 * np.sin(phase) * np.exp(-t * 12)
    if bol in ("ke", "ka", "ti", "te"):
        out += 0.3 * rng.standard_normal(n) * np.exp(-t * 60)   # crisp slap

    peak = np.abs(out).max() or 1.0
    return 0.8 * out / peak


def tanpura_drone(duration: float, sa_freq: float = 220.0,
                  rate: int = SAMPLE_RATE) -> np.ndarray:
    """Continuous Sa–Pa–Sa drone with jawari shimmer (slow harmonic beating)."""
    n = int(duration * rate)
    t = np.arange(n) / rate
    out = np.zeros(n)
    for f, a in ((sa_freq, 0.5), (sa_freq * 1.5, 0.35), (sa_freq / 2, 0.4)):
        for k in range(1, 7):
            # jawari: harmonics slowly swell and fade against each other
            shimmer = 0.5 + 0.5 * np.sin(2 * np.pi * (0.1 + 0.07 * k) * t)
            out += (a / k) * shimmer * np.sin(2 * np.pi * f * k * t + 0.3 * k)
    peak = np.abs(out).max() or 1.0
    return 0.25 * out / peak
