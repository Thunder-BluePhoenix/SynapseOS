"""
Create — the full song pipeline: write, compose, sing.

Brings everything together:
  LyricWriter  → original themed lyrics (verse + chorus)
  Raag/mood    → melody composed within raag grammar (aroha/avaroha/vadi)
                 or western mood scales (music.py)
  Instruments  → accompaniment timbre + tanpura drone + tabla taal
  Vocals       → formant-synthesized singing of the lyric syllables,
                 emotion shaping vibrato/brightness/meend

Output: a mixed WAV + the complete score + lyrics. Everything seeded.
"""

from __future__ import annotations

import random
import time
import wave as wave_mod
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from synapseos.create.instruments import (SAMPLE_RATE, TIMBRES, render_note,
                                          tabla_stroke, tanpura_drone)
from synapseos.create.lyrics import Lyrics, LyricWriter
from synapseos.create.raag import RAAGS, TAALS, Raag, raag_for_state, swara_to_semitone
from synapseos.create.vocals import sing_melody

SA = 220.0   # Sa = A3 — comfortable vocal register

RASA_EMOTION = {"shringar": "joyful", "karuna": "sad", "shanta": "calm",
                "veera": "energetic", "bhakti": "devotion", "gambhir": "sad"}
RASA_TEMPO = {"shringar": 100, "karuna": 72, "shanta": 84, "veera": 132,
              "bhakti": 80, "gambhir": 66}


@dataclass
class Song:
    path: str
    title: str
    raag: str
    taal: str
    rasa: str
    emotion: str
    tempo: int
    seed: int
    duration_s: float
    lyrics_text: str
    melody_swaras: list[str] = field(default_factory=list)
    instrument: str = "sitar"

    def to_dict(self) -> dict:
        return asdict(self)


class SongComposer:
    """Writes, composes and sings an original song."""

    def __init__(self, out_dir: str = "data/creations/songs"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.lyricist = LyricWriter()

    def sing(self, theme: str, raag: str | None = None,
             emotion: str | None = None, taal: str = "keherwa",
             instrument: str = "sitar", cycles: int = 4,
             seed: int | None = None,
             valence: float | None = None, arousal: float | None = None) -> Song:
        """
        Full pipeline. Pick raag explicitly, or let emotion/affect choose it.
        """
        seed = seed if seed is not None else int(time.time() * 1000) % 999_983
        rng = random.Random(seed)

        # 1. Choose the raag
        if raag and raag.lower() in RAAGS:
            r = RAAGS[raag.lower()]
        elif valence is not None and arousal is not None:
            r = raag_for_state(valence, arousal)
        elif emotion:
            r = next((x for x in RAAGS.values()
                      if RASA_EMOTION[x.rasa] == emotion), RAAGS["bhupali"])
        else:
            r = RAAGS["yaman"]
        emotion = emotion or RASA_EMOTION[r.rasa]
        tempo = RASA_TEMPO[r.rasa]
        t = TAALS.get(taal, TAALS["keherwa"])

        # 2. Write the lyrics
        lyrics = self.lyricist.write(theme, emotion=emotion, seed=seed)

        # 3. Compose the melody inside the raag grammar
        melody, swara_names = self._compose_raag_melody(r, t, tempo, cycles, rng)
        total = max(s + d for _, s, d in melody) + 1.0

        # 4. Render all layers
        buf = np.zeros(int(total * SAMPLE_RATE) + SAMPLE_RATE)

        def mix(sig: np.ndarray, start_s: float = 0.0, gain: float = 1.0) -> None:
            i = int(start_s * SAMPLE_RATE)
            buf[i:i + len(sig)] += gain * sig[: len(buf) - i]

        mix(tanpura_drone(total, SA), 0.0, 0.9)                       # drone bed
        timbre = TIMBRES.get(instrument, TIMBRES["sitar"])
        for freq, start, dur in melody:                                # instrument
            mix(render_note(freq, dur, 0.22, timbre), start)
        beat = 60.0 / tempo
        n_beats = int(total / beat)
        for b in range(n_beats):                                       # tabla taal
            bol = t.bols[b % t.beats]
            mix(tabla_stroke(bol, sa_freq=SA * 2), b * beat, 0.7)
        vocal = sing_melody(melody, lyrics.syllables, emotion, total)  # the voice
        mix(vocal, 0.0, 1.0)

        peak = np.abs(buf).max() or 1.0
        buf = buf / peak * 0.85

        # 5. Write out
        safe = "".join(c if c.isalnum() else "_" for c in theme.lower())[:36]
        path = self.out_dir / f"{safe}_{r.name.lower()}_{seed}.wav"
        pcm = (buf * 32000).astype("<i2").tobytes()
        with wave_mod.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm)

        return Song(
            path=str(path), title=lyrics.title, raag=r.name, taal=t.name,
            rasa=r.rasa, emotion=emotion, tempo=tempo, seed=seed,
            duration_s=round(total, 2), lyrics_text=lyrics.full_text(),
            melody_swaras=swara_names, instrument=timbre.name,
        )

    # -- melody within the raag ---------------------------------------------------

    def _compose_raag_melody(self, r: Raag, taal, tempo: int, cycles: int,
                             rng: random.Random):
        """
        Walk the raag: ascending phrases use only aroha swaras, descending
        only avaroha; the vadi is emphasized (longer, more frequent); phrases
        resolve toward Sa; the pakad (signature) opens the piece.
        """
        beat = 60.0 / tempo
        events: list[tuple[float, float, float]] = []
        names: list[str] = []
        t_cur = 0.0

        def add(swara: str, beats: float) -> None:
            nonlocal t_cur
            semis = swara_to_semitone(swara)
            events.append((SA * (2 ** (semis / 12.0)), t_cur, beats * beat))
            names.append(swara)
            t_cur += beats * beat

        # Open with the pakad — the raag's signature
        for s in r.pakad:
            add(s, 1.0)

        aroha, avaroha = r.aroha, r.avaroha
        beats_per_cycle = taal.beats
        for _ in range(cycles):
            ascending = rng.random() < 0.5
            seq = aroha if ascending else avaroha
            idx = rng.randrange(len(seq))
            used = 0.0
            while used < beats_per_cycle - 1:
                step = rng.choice([1, 1, 1, 2, -1]) * (1 if ascending else 1)
                idx = max(0, min(len(seq) - 1, idx + step))
                swara = seq[idx]
                # vadi and samvadi get weight: longer and revisited
                if rng.random() < 0.3:
                    swara = rng.choice([r.vadi, r.samvadi])
                dur = 2.0 if swara.rstrip("'") in (r.vadi, r.samvadi) and rng.random() < 0.5 else 1.0
                dur = min(dur, beats_per_cycle - used)
                add(swara, dur)
                used += dur
            add("S", 1.0)                       # phrase resolves on Sa
        return events, names
