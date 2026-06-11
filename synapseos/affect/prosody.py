"""
Affect — prosody analysis. How something was said, from the audio itself.

Whisper gives us the words but discards tone. This module reads the same WAV
and extracts the acoustic signals that carry emotion:

  energy (RMS)        — loudness; raised voice vs quiet
  pitch (autocorr F0) — mean + variance; flat = low affect, varied = engaged
  speaking rate       — energy-peak density; fast = aroused, slow = tired/sad
  pause ratio         — silence fraction; long pauses = hesitation/sadness

Output is an arousal estimate + qualitative labels. Valence is hard to get
from acoustics alone (a shout can be joy or rage) — that disambiguation comes
from fusing with the text detector in the tracker.

Pure numpy + stdlib wave. No LLM, no cloud, no extra installs.
"""

from __future__ import annotations

import wave
from dataclasses import asdict, dataclass

import numpy as np

# Typical adult speech F0 ranges ~75–350 Hz
_F0_MIN, _F0_MAX = 60.0, 400.0


@dataclass
class ProsodyResult:
    duration_s: float
    energy_rms: float           # 0..1 normalized loudness
    pitch_mean_hz: float
    pitch_var: float            # coefficient of variation of F0
    speaking_rate: float        # energy peaks per second (syllable proxy)
    pause_ratio: float          # fraction of frames that are silence
    arousal: float              # fused 0..1 estimate
    label: str                  # calm | engaged | agitated | subdued

    def to_dict(self) -> dict:
        return asdict(self)


def _read_wav_mono(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        rate = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
        width = w.getsampwidth()
        channels = w.getnchannels()
    if width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, rate


def _frame(data: np.ndarray, rate: int, ms: float = 30.0, hop_ms: float = 15.0):
    size = int(rate * ms / 1000)
    hop = int(rate * hop_ms / 1000)
    if len(data) < size:
        return np.empty((0, size))
    n = 1 + (len(data) - size) // hop
    return np.stack([data[i * hop: i * hop + size] for i in range(n)])


def _pitch_autocorr(frame: np.ndarray, rate: int) -> float:
    """F0 of one frame via autocorrelation. 0.0 if unvoiced."""
    frame = frame - frame.mean()
    if np.abs(frame).max() < 1e-4:
        return 0.0
    corr = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
    if corr[0] <= 0:
        return 0.0
    lag_min = int(rate / _F0_MAX)
    lag_max = min(int(rate / _F0_MIN), len(corr) - 1)
    if lag_max <= lag_min:
        return 0.0
    segment = corr[lag_min:lag_max]
    peak = int(np.argmax(segment)) + lag_min
    # voiced check: autocorr peak must be a substantial fraction of energy
    if corr[peak] / corr[0] < 0.3:
        return 0.0
    return rate / peak


def analyze_wav(path: str) -> ProsodyResult:
    data, rate = _read_wav_mono(path)
    duration = len(data) / rate if rate else 0.0
    frames = _frame(data, rate)
    if len(frames) == 0:
        return ProsodyResult(duration, 0, 0, 0, 0, 1.0, 0.0, "subdued")

    # Energy per frame
    rms = np.sqrt((frames ** 2).mean(axis=1))
    peak_rms = float(rms.max()) or 1e-6
    energy = float(rms.mean() / peak_rms)

    # Silence / pauses
    silence_thresh = 0.1 * peak_rms
    voiced_mask = rms > silence_thresh
    pause_ratio = float(1.0 - voiced_mask.mean())

    # Pitch over voiced frames (subsample for speed)
    pitches = []
    for i in range(0, len(frames), 3):
        if voiced_mask[i]:
            f0 = _pitch_autocorr(frames[i], rate)
            if f0 > 0:
                pitches.append(f0)
    if pitches:
        pitch_mean = float(np.mean(pitches))
        pitch_var = float(np.std(pitches) / (pitch_mean or 1.0))
    else:
        pitch_mean, pitch_var = 0.0, 0.0

    # Speaking rate: energy peaks per second (rough syllable nuclei proxy)
    above = (rms > 0.5 * peak_rms).astype(int)
    onsets = int(np.maximum(np.diff(above), 0).sum())
    speaking_rate = onsets / duration if duration > 0 else 0.0

    # Fuse into arousal: loud + varied pitch + fast + few pauses = agitated
    arousal = float(np.clip(
        0.35 * energy
        + 0.25 * np.clip(pitch_var / 0.4, 0, 1)
        + 0.25 * np.clip(speaking_rate / 5.0, 0, 1)
        + 0.15 * (1.0 - pause_ratio),
        0, 1,
    ))

    if arousal > 0.65:
        label = "agitated"
    elif arousal > 0.45:
        label = "engaged"
    elif pause_ratio > 0.6:
        label = "subdued"
    else:
        label = "calm"

    return ProsodyResult(
        duration_s=round(duration, 2),
        energy_rms=round(energy, 3),
        pitch_mean_hz=round(pitch_mean, 1),
        pitch_var=round(pitch_var, 3),
        speaking_rate=round(speaking_rate, 2),
        pause_ratio=round(pause_ratio, 3),
        arousal=round(arousal, 3),
        label=label,
    )
