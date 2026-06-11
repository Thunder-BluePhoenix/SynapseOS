"""
Create — Hindustani raag system. Music as mathematics.

A raag is a formal object: a set of allowed swaras (notes), separate ascent
(aroha) and descent (avaroha) constraints, emphasized notes (vadi/samvadi),
and an emotional register (rasa). A taal is a cyclic group of beats. That
makes Indian classical music unusually computable — exactly the kind of
structured domain this architecture is built for.

Swara → semitone (from Sa):
  S=0  r=1 R=2  g=3 G=4  m=5 M=6(tivra)  P=7  d=8 D=9  n=10 N=11
  Lowercase = komal (flat), M = tivra Ma. Apostrophe ' = upper octave.

Rasa → affect mapping lets the emotion tracker choose a raag:
  shringar (love/joy), karuna (pathos/sadness), shanta (peace),
  veera (valor/energy), bhakti (devotion), gambhir (gravity/depth)
"""

from __future__ import annotations

from dataclasses import dataclass, field

SWARA_SEMITONES = {"S": 0, "r": 1, "R": 2, "g": 3, "G": 4, "m": 5, "M": 6,
                   "P": 7, "d": 8, "D": 9, "n": 10, "N": 11}


def swara_to_semitone(swara: str) -> int:
    """'S' → 0, "S'" (upper octave) → 12, ",P" (lower octave) → -5."""
    base = swara.strip()
    octave = 0
    while base.endswith("'"):
        octave += 12
        base = base[:-1]
    while base.startswith(","):
        octave -= 12
        base = base[1:]
    return SWARA_SEMITONES[base] + octave


@dataclass
class Raag:
    name: str
    thaat: str
    aroha: list[str]            # ascending swara sequence
    avaroha: list[str]          # descending swara sequence
    vadi: str                   # king note — most emphasized
    samvadi: str                # minister note — second emphasis
    rasa: str                   # emotional register
    time_of_day: str
    pakad: list[str] = field(default_factory=list)   # signature phrase

    def aroha_semitones(self) -> list[int]:
        return [swara_to_semitone(s) for s in self.aroha]

    def avaroha_semitones(self) -> list[int]:
        return [swara_to_semitone(s) for s in self.avaroha]


RAAGS: dict[str, Raag] = {
    "yaman": Raag(
        name="Yaman", thaat="Kalyan",
        aroha=["S", "R", "G", "M", "P", "D", "N", "S'"],
        avaroha=["S'", "N", "D", "P", "M", "G", "R", "S"],
        vadi="G", samvadi="N", rasa="shringar", time_of_day="evening",
        pakad=["N", "R", "G", "R", "S"],
    ),
    "bhairav": Raag(
        name="Bhairav", thaat="Bhairav",
        aroha=["S", "r", "G", "m", "P", "d", "N", "S'"],
        avaroha=["S'", "N", "d", "P", "m", "G", "r", "S"],
        vadi="d", samvadi="r", rasa="bhakti", time_of_day="dawn",
        pakad=["G", "m", "d", "P", "G", "m", "r", "S"],
    ),
    "kafi": Raag(
        name="Kafi", thaat="Kafi",
        aroha=["S", "R", "g", "m", "P", "D", "n", "S'"],
        avaroha=["S'", "n", "D", "P", "m", "g", "R", "S"],
        vadi="P", samvadi="R", rasa="shringar", time_of_day="late evening",
        pakad=["S", "S", "R", "R", "g", "g", "m", "m", "P"],
    ),
    "bhupali": Raag(
        name="Bhupali", thaat="Kalyan",
        aroha=["S", "R", "G", "P", "D", "S'"],
        avaroha=["S'", "D", "P", "G", "R", "S"],
        vadi="G", samvadi="D", rasa="shanta", time_of_day="evening",
        pakad=["G", "R", "S", "D", "S", "R", "G"],
    ),
    "darbari": Raag(
        name="Darbari", thaat="Asavari",
        aroha=["S", "R", "g", "m", "P", "d", "n", "S'"],
        avaroha=["S'", "d", "n", "P", "m", "g", "R", "S"],
        vadi="R", samvadi="P", rasa="gambhir", time_of_day="midnight",
        pakad=["g", "R", "S", "d", "n", "P"],
    ),
    "desh": Raag(
        name="Desh", thaat="Khamaj",
        aroha=["S", "R", "m", "P", "N", "S'"],
        avaroha=["S'", "n", "D", "P", "m", "G", "R", "S"],
        vadi="R", samvadi="P", rasa="veera", time_of_day="late evening",
        pakad=["R", "m", "P", "N", "S'", "n", "D", "P"],
    ),
    "malkauns": Raag(
        name="Malkauns", thaat="Bhairavi",
        aroha=["S", "g", "m", "d", "n", "S'"],
        avaroha=["S'", "n", "d", "m", "g", "S"],
        vadi="m", samvadi="S", rasa="karuna", time_of_day="deep night",
        pakad=["m", "g", "m", "d", "n", "d", "m"],
    ),
}

# rasa → (valence range hint) for choosing raag from emotional state
RASA_FOR_STATE = [
    # (min_valence, max_valence, high_arousal, rasa)
    (0.1, 1.0, True, "veera"),       # positive + energetic → valor
    (0.1, 1.0, False, "shringar"),   # positive + soft → love/beauty
    (-0.1, 0.1, False, "shanta"),    # neutral calm → peace
    (-0.1, 0.1, True, "bhakti"),     # neutral but stirred → devotion
    (-1.0, -0.1, True, "gambhir"),   # negative + agitated → gravity
    (-1.0, -0.1, False, "karuna"),   # negative + low → pathos
]


def raag_for_state(valence: float, arousal: float) -> Raag:
    """Pick a raag whose rasa matches an emotional state."""
    high = arousal > 0.5
    for lo, hi, needs_high, rasa in RASA_FOR_STATE:
        if lo <= valence <= hi and needs_high == high:
            for raag in RAAGS.values():
                if raag.rasa == rasa:
                    return raag
    return RAAGS["bhupali"]


@dataclass
class Taal:
    name: str
    beats: int
    divisions: list[int]            # vibhag structure
    bols: list[str]                 # tabla syllables per beat


TAALS: dict[str, Taal] = {
    "teentaal": Taal("Teentaal", 16, [4, 4, 4, 4],
                     ["dha", "dhin", "dhin", "dha"] * 2
                     + ["dha", "tin", "tin", "ta"]
                     + ["ta", "dhin", "dhin", "dha"]),
    "keherwa": Taal("Keherwa", 8, [4, 4],
                    ["dha", "ge", "na", "ti", "na", "ke", "dhi", "na"]),
    "dadra": Taal("Dadra", 6, [3, 3],
                  ["dha", "dhi", "na", "dha", "tu", "na"]),
}
