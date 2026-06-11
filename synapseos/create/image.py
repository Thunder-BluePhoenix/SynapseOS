"""
Create — procedural image generation.

Original visual art, generated from algorithms rather than recalled from
training data — so every image is new by construction and reproducible from
(concept, style, seed). The concept string is hashed into the parameters
(palette, geometry, turbulence), so "ocean storm" and "quiet forest" genuinely
look different, deterministically.

Styles:
  flow    — particle traces through a sine/cosine flow field
  fractal — Julia set with concept-derived complex constant + palette
  geo     — layered translucent geometric forms

Honest scope: this is generative/procedural art, not photorealistic
text-to-image — that would need a diffusion model (possible later on MPS,
heavy install). For a companion that *makes original things*, procedural is
the no-LLM-ethos answer.
"""

from __future__ import annotations

import colorsys
import hashlib
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class GeneratedImage:
    path: str
    concept: str
    style: str
    seed: int
    width: int
    height: int
    palette: list[str]          # hex colors used — provenance of the look

    def to_dict(self) -> dict:
        return asdict(self)


def _concept_seed(concept: str, seed: int | None) -> int:
    if seed is not None:
        return seed
    return int(hashlib.sha256(concept.lower().encode()).hexdigest()[:8], 16)


def _palette(rng: random.Random, n: int = 5) -> list[tuple[int, int, int]]:
    """Harmonious palette: base hue + analogous/complementary offsets."""
    base = rng.random()
    offsets = [0, 0.08, -0.08, 0.5, 0.04]
    cols = []
    for i in range(n):
        h = (base + offsets[i % len(offsets)]) % 1.0
        s = 0.45 + 0.5 * rng.random()
        v = 0.55 + 0.4 * rng.random()
        cols.append(tuple(int(c * 255) for c in colorsys.hsv_to_rgb(h, s, v)))
    return cols


def _hex(c: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*c)


class ImageGenerator:
    def __init__(self, out_dir: str = "data/creations/images"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def create(self, concept: str, style: str = "auto",
               width: int = 640, height: int = 480,
               seed: int | None = None) -> GeneratedImage:
        seed = _concept_seed(concept, seed)
        rng = random.Random(seed)
        if style == "auto":
            style = rng.choice(["flow", "fractal", "geo"])
        palette = _palette(rng)

        if style == "fractal":
            img = self._fractal(rng, palette, width, height)
        elif style == "geo":
            img = self._geo(rng, palette, width, height)
        else:
            style = "flow"
            img = self._flow(rng, palette, width, height)

        safe = "".join(c if c.isalnum() else "_" for c in concept.lower())[:40]
        path = self.out_dir / f"{safe}_{style}_{seed}.png"
        img.save(path)
        return GeneratedImage(path=str(path), concept=concept, style=style,
                              seed=seed, width=width, height=height,
                              palette=[_hex(c) for c in palette])

    # -- styles --------------------------------------------------------------------

    def _flow(self, rng: random.Random, palette, w: int, h: int) -> Image.Image:
        """Particles traced through a procedural vector field."""
        img = Image.new("RGB", (w, h), self._bg(palette))
        draw = ImageDraw.Draw(img, "RGBA")
        k1, k2 = 2 + 4 * rng.random(), 2 + 4 * rng.random()
        phase = rng.random() * math.tau
        for _ in range(450):
            x, y = rng.random() * w, rng.random() * h
            color = (*rng.choice(palette), 36)
            pts = []
            for _ in range(60):
                angle = (math.sin(k1 * x / w + phase) + math.cos(k2 * y / h)) * math.pi
                x += 2.2 * math.cos(angle)
                y += 2.2 * math.sin(angle)
                if not (0 <= x < w and 0 <= y < h):
                    break
                pts.append((x, y))
            if len(pts) > 2:
                draw.line(pts, fill=color, width=2)
        return img

    def _fractal(self, rng: random.Random, palette, w: int, h: int) -> Image.Image:
        """Julia set, constant + coloring derived from the concept."""
        c = complex(-0.8 + 0.6 * rng.random(), 0.1 + 0.25 * rng.random())
        zoom = 1.4
        xs = np.linspace(-zoom * w / h, zoom * w / h, w)
        ys = np.linspace(-zoom, zoom, h)
        z = xs[None, :] + 1j * ys[:, None]
        counts = np.zeros(z.shape, dtype=np.int32)
        alive = np.ones(z.shape, dtype=bool)
        for i in range(80):
            z[alive] = z[alive] ** 2 + c
            escaped = np.abs(z) > 2.0
            counts[alive & escaped] = i
            alive &= ~escaped
        norm = counts / counts.max() if counts.max() else counts
        # Map iteration count through the palette
        pal = np.array(palette + [(12, 10, 24)])
        idx = (norm * (len(pal) - 1)).astype(int)
        rgb = pal[idx].astype(np.uint8)
        rgb[alive] = (8, 6, 18)
        return Image.fromarray(rgb, "RGB")

    def _geo(self, rng: random.Random, palette, w: int, h: int) -> Image.Image:
        """Layered translucent geometry."""
        img = Image.new("RGB", (w, h), self._bg(palette))
        draw = ImageDraw.Draw(img, "RGBA")
        for _ in range(rng.randint(18, 32)):
            color = (*rng.choice(palette), rng.randint(40, 110))
            cx, cy = rng.random() * w, rng.random() * h
            r = (0.05 + 0.22 * rng.random()) * min(w, h)
            kind = rng.choice(["circle", "poly", "rect"])
            if kind == "circle":
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
            elif kind == "rect":
                ang = rng.random() * math.tau
                dx, dy = r * math.cos(ang), r * math.sin(ang)
                draw.polygon([(cx - dx, cy - dy), (cx + dy, cy - dx),
                              (cx + dx, cy + dy), (cx - dy, cy + dx)], fill=color)
            else:
                n = rng.randint(3, 6)
                rot = rng.random() * math.tau
                draw.polygon([(cx + r * math.cos(rot + i * math.tau / n),
                               cy + r * math.sin(rot + i * math.tau / n))
                              for i in range(n)], fill=color)
        return img

    @staticmethod
    def _bg(palette) -> tuple[int, int, int]:
        r, g, b = palette[0]
        return (r // 6, g // 6, b // 6)
