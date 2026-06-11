"""
Create — procedural video generation.

Animates the image generators: a parameter sweep (fractal constant drift,
flow-field phase rotation) rendered frame by frame with Pillow, then encoded
to MP4 by ffmpeg. Optionally muxes a composed soundtrack (music.py) in —
SynapseOS scoring its own film.

Degrades gracefully: without ffmpeg the frames are still produced and an
animated GIF is written via Pillow instead.
"""

from __future__ import annotations

import math
import random
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from synapseos.create.image import ImageGenerator, _concept_seed, _palette


@dataclass
class GeneratedVideo:
    path: str
    concept: str
    seed: int
    frames: int
    fps: int
    duration_s: float
    format: str            # mp4 | gif
    audio_path: str = ""   # soundtrack, if muxed

    def to_dict(self) -> dict:
        return asdict(self)


class VideoGenerator:
    def __init__(self, out_dir: str = "data/creations/videos"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.ffmpeg = shutil.which("ffmpeg")

    def create(self, concept: str, seconds: float = 4.0, fps: int = 12,
               width: int = 480, height: int = 360,
               seed: int | None = None, soundtrack: str | None = None) -> GeneratedVideo:
        seed = _concept_seed(concept, seed)
        rng = random.Random(seed)
        palette = _palette(rng)
        n_frames = max(2, int(seconds * fps))

        # Julia constant drifts along a small circle → evolving fractal
        c0 = complex(-0.8 + 0.6 * rng.random(), 0.1 + 0.25 * rng.random())
        radius = 0.04 + 0.05 * rng.random()
        pal = np.array(palette + [(12, 10, 24)])

        frames: list[Image.Image] = []
        for f in range(n_frames):
            theta = math.tau * f / n_frames
            c = c0 + radius * complex(math.cos(theta), math.sin(theta))
            frames.append(self._julia_frame(c, pal, width, height))

        safe = "".join(ch if ch.isalnum() else "_" for ch in concept.lower())[:40]
        if self.ffmpeg:
            path = self.out_dir / f"{safe}_{seed}.mp4"
            self._encode_mp4(frames, path, fps, soundtrack)
            fmt = "mp4"
        else:
            path = self.out_dir / f"{safe}_{seed}.gif"
            frames[0].save(path, save_all=True, append_images=frames[1:],
                           duration=int(1000 / fps), loop=0)
            fmt = "gif"

        return GeneratedVideo(path=str(path), concept=concept, seed=seed,
                              frames=n_frames, fps=fps,
                              duration_s=round(n_frames / fps, 2), format=fmt,
                              audio_path=soundtrack or "")

    # -- internals ------------------------------------------------------------------

    @staticmethod
    def _julia_frame(c: complex, pal: np.ndarray, w: int, h: int) -> Image.Image:
        zoom = 1.4
        xs = np.linspace(-zoom * w / h, zoom * w / h, w)
        ys = np.linspace(-zoom, zoom, h)
        z = xs[None, :] + 1j * ys[:, None]
        counts = np.zeros(z.shape, dtype=np.int32)
        alive = np.ones(z.shape, dtype=bool)
        for i in range(48):
            z[alive] = z[alive] ** 2 + c
            escaped = np.abs(z) > 2.0
            counts[alive & escaped] = i
            alive &= ~escaped
        norm = counts / counts.max() if counts.max() else counts
        idx = (norm * (len(pal) - 1)).astype(int)
        rgb = pal[idx].astype(np.uint8)
        rgb[alive] = (8, 6, 18)
        return Image.fromarray(rgb, "RGB")

    def _encode_mp4(self, frames: list[Image.Image], path: Path, fps: int,
                    soundtrack: str | None) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="synapse_video_"))
        try:
            for i, frame in enumerate(frames):
                frame.save(tmp / f"f_{i:05d}.png")
            cmd = [self.ffmpeg, "-y", "-framerate", str(fps),
                   "-i", str(tmp / "f_%05d.png")]
            if soundtrack and Path(soundtrack).exists():
                cmd += ["-i", soundtrack, "-shortest", "-c:a", "aac"]
            cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
            subprocess.run(cmd, capture_output=True, timeout=120, check=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
