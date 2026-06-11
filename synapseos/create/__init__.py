"""
SynapseOS Create — generative output: music, images, video.

Deterministic, seeded, original-by-construction (no training data to copy,
no LLM). Algorithmic composition and procedural art are the classic non-neural
creative domains — every piece is parameterized by a seed + concept, so the
same idea is reproducible and the provenance is the algorithm itself.

  music  — scale/chord-progression composition, harmonic synthesis → WAV
  image  — flow fields, fractals, geometric palettes → PNG
  video  — animated parameter sweeps of the image generators → MP4 (ffmpeg)

The MUSE agent (phase5/agents/muse.py) routes natural-language requests here
and can compose to the user's tracked emotional state.
"""

from synapseos.create.music import MusicComposer, Composition
from synapseos.create.image import ImageGenerator, GeneratedImage
from synapseos.create.video import VideoGenerator, GeneratedVideo
from synapseos.create.raag import RAAGS, TAALS, Raag, raag_for_state
from synapseos.create.lyrics import LyricWriter, Lyrics
from synapseos.create.song import Song, SongComposer

__all__ = ["MusicComposer", "Composition", "ImageGenerator", "GeneratedImage",
           "VideoGenerator", "GeneratedVideo", "RAAGS", "TAALS", "Raag",
           "raag_for_state", "LyricWriter", "Lyrics", "Song", "SongComposer"]
