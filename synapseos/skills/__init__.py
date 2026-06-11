"""
SynapseOS Skills — deterministic, computable expertise.

Unlike learned knowledge (which grows via the web loop), skills are exact:
  math_solver — arithmetic, algebra, calculus, statistics & probability,
                geometry/trigonometry/mensuration, physics constants.
                Every answer is computed, never recalled — zero hallucination.
  language    — script/language detection (Bengali, Devanagari, Japanese,
                Latin, ...) + basic phrasebook. Honest scope: real fluency
                and translation need the neural (Mamba GPU) path.
"""

from synapseos.skills.math_solver import MathSolver, MathAnswer
from synapseos.skills.language import detect_language, LanguageGuess, PHRASEBOOK

__all__ = ["MathSolver", "MathAnswer", "detect_language", "LanguageGuess",
           "PHRASEBOOK"]
