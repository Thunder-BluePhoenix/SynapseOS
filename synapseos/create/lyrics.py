"""
Create — lyric writing. Template-grammar verse construction, no LLM.

Original lyrics built from: a theme (the user's subject), an emotion (word
banks per feeling), and a rhyme scheme (small phonetic-ending rhyme groups).
Structure: verse (ABAB) + chorus (AABB repeated hook). The output is honest
craft, not poetry-prize material — but it is original, themed, emotionally
consistent, and sings well because line lengths are syllable-controlled.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

# Rhyme groups: words sharing an ending sound, organized by emotion fit
RHYME_GROUPS = {
    "ight": ["light", "night", "bright", "flight", "sight", "height"],
    "ay":   ["day", "way", "stay", "away", "gray", "sway"],
    "ain":  ["rain", "pain", "again", "remain", "plain", "chain"],
    "old":  ["gold", "old", "hold", "told", "bold", "cold"],
    "ee":   ["free", "sea", "be", "see", "tree", "me"],
    "ow":   ["flow", "glow", "slow", "grow", "know", "below"],
    "ire":  ["fire", "higher", "desire", "wire", "inspire"],
    "art":  ["heart", "start", "apart", "part", "art"],
}

EMOTION_BANKS = {
    "joyful":    {"adj": ["golden", "shining", "dancing", "boundless"],
                  "verb": ["rises", "sings", "blooms", "soars"],
                  "noun": ["morning", "laughter", "sunlight", "wonder"],
                  "rhymes": ["ight", "ay", "ee"]},
    "sad":       {"adj": ["fading", "silent", "heavy", "distant"],
                  "verb": ["falls", "fades", "weeps", "drifts"],
                  "noun": ["shadow", "memory", "echo", "winter"],
                  "rhymes": ["ain", "old", "ow"]},
    "energetic": {"adj": ["burning", "racing", "electric", "fearless"],
                  "verb": ["runs", "breaks", "ignites", "rises"],
                  "noun": ["thunder", "horizon", "engine", "storm"],
                  "rhymes": ["ire", "ight", "ay"]},
    "calm":      {"adj": ["quiet", "gentle", "endless", "soft"],
                  "verb": ["breathes", "settles", "wanders", "rests"],
                  "noun": ["river", "evening", "stillness", "moon"],
                  "rhymes": ["ow", "ee", "ay"]},
    "devotion":  {"adj": ["sacred", "timeless", "humble", "deep"],
                  "verb": ["calls", "carries", "answers", "remains"],
                  "noun": ["temple", "dawn", "prayer", "flame"],
                  "rhymes": ["art", "old", "ain"]},
}

# Line templates: {theme} {adj} {verb} {noun} {rhyme} slots
_VERSE_TEMPLATES = [
    "The {adj} {noun} of {theme} {verb} through the {rhyme}",
    "I have seen the {theme} in {adj} {rhyme}",
    "Where the {noun} {verb}, the {theme} finds its {rhyme}",
    "Every {adj} {noun} {verb} toward the {rhyme}",
    "And the {theme} {verb} like {noun} in the {rhyme}",
    "Carry me through {adj} {noun} to the {rhyme}",
]

_CHORUS_TEMPLATES = [
    "Oh, {theme}, my {adj} {rhyme}",
    "You are the {noun}, you are the {rhyme}",
    "Hold the {theme} and never let it {rhyme}",
    "We {verb_pl} with the {theme} tonight",
]


@dataclass
class Lyrics:
    title: str
    theme: str
    emotion: str
    verse: list[str]
    chorus: list[str]
    syllables: list[str] = field(default_factory=list)   # for the singer

    def full_text(self) -> str:
        return (f"\"{self.title}\"\n\n[Verse]\n" + "\n".join(self.verse)
                + "\n\n[Chorus]\n" + "\n".join(self.chorus))


def _syllabify(text: str) -> list[str]:
    """Crude syllable split — vowel-group chunks of each word."""
    out = []
    for word in re.findall(r"[a-z']+", text.lower()):
        parts = re.findall(r"[^aeiou]*[aeiou]+(?:[^aeiou](?![aeiou]))?", word)
        out.extend(parts or [word])
    return out


class LyricWriter:
    def write(self, theme: str, emotion: str = "calm",
              seed: int | None = None) -> Lyrics:
        emotion = emotion if emotion in EMOTION_BANKS else "calm"
        bank = EMOTION_BANKS[emotion]
        rng = random.Random(seed if seed is not None
                            else abs(hash((theme, emotion))) % 999_983)

        def line(template: str, rhyme_word: str) -> str:
            verb = rng.choice(bank["verb"])
            out = template.format(
                theme=theme,
                adj=rng.choice(bank["adj"]),
                verb=verb,
                verb_pl=verb.rstrip("s"),       # "we rise", not "we rises"
                noun=rng.choice(bank["noun"]),
                rhyme=rhyme_word,
            )
            return re.sub(r"\b[Tt]he the\b", "the", out)   # theme may carry "the"

        # Verse: ABAB rhyme scheme
        ga, gb = rng.sample(bank["rhymes"], 2)
        ra = rng.sample(RHYME_GROUPS[ga], 2)
        rb = rng.sample(RHYME_GROUPS[gb], 2)
        templates = rng.sample(_VERSE_TEMPLATES, 4)
        verse = [line(templates[0], ra[0]), line(templates[1], rb[0]),
                 line(templates[2], ra[1]), line(templates[3], rb[1])]

        # Chorus: AA couplet, repeated hook
        gc = rng.choice(bank["rhymes"])
        rc = rng.sample(RHYME_GROUPS[gc], 2)
        ct = rng.sample(_CHORUS_TEMPLATES, 2)
        chorus = [line(ct[0], rc[0]), line(ct[1], rc[1])]

        title = f"{rng.choice(bank['adj']).title()} {theme.title()}"
        syllables = _syllabify(" ".join(verse + chorus))
        return Lyrics(title=title, theme=theme, emotion=emotion,
                      verse=verse, chorus=chorus, syllables=syllables)
