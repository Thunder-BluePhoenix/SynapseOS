"""
Create — structured text-to-image (typography engine).

Ideogram's signature capability — crisp text rendered INTO images with layout
and palette control — rebuilt the SynapseOS way: deterministic typesetting
over procedural backgrounds instead of a 9B diffusion model. A poster is a
structured spec, not a prompt lottery:

    PosterSpec(
        title="SYNAPSE",
        subtitle="a mind that grows",
        body=["learns from the web", "remembers everything"],
        palette=["#1f83c0", "#d56c2c"],     # optional — else concept-derived
        background="flow" | "fractal" | "geo" | "solid",
        size=(800, 1000),
    )

Text is guaranteed legible: contrast band behind text, auto font sizing to
fit boxes, real system fonts (Avenir/Helvetica on macOS, DejaVu elsewhere).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFont

from synapseos.create.image import ImageGenerator, _concept_seed

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Avenir Next.ttc",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def _load_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                # index 1 in .ttc files is typically a bolder face
                return ImageFont.truetype(path, size, index=1 if bold else 0)
            except Exception:
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    continue
    return ImageFont.load_default()


@dataclass
class PosterSpec:
    title: str
    subtitle: str = ""
    body: list[str] = field(default_factory=list)
    palette: list[str] | None = None          # hex colors; None = from title
    background: str = "flow"                  # flow | fractal | geo | solid
    size: tuple[int, int] = (800, 1000)
    footer: str = ""


@dataclass
class GeneratedPoster:
    path: str
    spec: PosterSpec
    seed: int
    fonts_used: str

    def to_dict(self) -> dict:
        return {"path": self.path, "title": self.spec.title, "seed": self.seed,
                "background": self.spec.background, "fonts": self.fonts_used}


class PosterGenerator:
    """Deterministic poster/banner/cover typesetting."""

    def __init__(self, out_dir: str = "data/creations/posters"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._bg_gen = ImageGenerator(out_dir=str(self.out_dir / "_bg"))

    def create(self, spec: PosterSpec, seed: int | None = None) -> GeneratedPoster:
        seed = _concept_seed(spec.title, seed)
        w, h = spec.size

        # 1. Background — procedural art seeded by the title, darkened for contrast
        if spec.background == "solid":
            base = Image.new("RGB", (w, h), (16, 14, 28))
        else:
            bg = self._bg_gen.create(spec.title, style=spec.background,
                                     width=w, height=h, seed=seed)
            base = Image.open(bg.path).convert("RGB")
            base = ImageEnhance.Brightness(base).enhance(0.55)

        draw = ImageDraw.Draw(base, "RGBA")
        rng = random.Random(seed)
        accent = self._accent(spec, rng)

        # 2. Layout bands: title ~22% down, subtitle under it, body mid, footer low
        y = int(h * 0.16)
        y = self._text_block(draw, spec.title.upper(), y, w, max_px=int(h * 0.11),
                             color=(255, 255, 255), tracking=True)
        if spec.subtitle:
            draw.line([(w * 0.2, y + 8), (w * 0.8, y + 8)], fill=(*accent, 220), width=3)
            y = self._text_block(draw, spec.subtitle, y + 26, w,
                                 max_px=int(h * 0.038), color=(*accent, 255))
        if spec.body:
            y = max(y + int(h * 0.10), int(h * 0.52))
            for line in spec.body[:5]:
                y = self._text_block(draw, line, y, w, max_px=int(h * 0.027),
                                     color=(235, 235, 235), band=True) + int(h * 0.012)
        if spec.footer:
            self._text_block(draw, spec.footer, int(h * 0.92), w,
                             max_px=int(h * 0.02), color=(*accent, 230))

        safe = "".join(c if c.isalnum() else "_" for c in spec.title.lower())[:36]
        path = self.out_dir / f"{safe}_{seed}.png"
        base.save(path)
        fonts = next((Path(p).stem for p in _FONT_CANDIDATES if Path(p).exists()),
                     "default")
        return GeneratedPoster(path=str(path), spec=spec, seed=seed, fonts_used=fonts)

    # -- helpers ------------------------------------------------------------------

    @staticmethod
    def _accent(spec: PosterSpec, rng: random.Random) -> tuple[int, int, int]:
        if spec.palette:
            hexcol = spec.palette[0].lstrip("#")
            return tuple(int(hexcol[i:i + 2], 16) for i in (0, 2, 4))
        return tuple(rng.randint(140, 255) for _ in range(3))

    def _text_block(self, draw: ImageDraw.ImageDraw, text: str, y: int, w: int,
                    max_px: int, color, band: bool = False,
                    tracking: bool = False) -> int:
        """Centered text, auto-sized to fit width. Returns next y."""
        if tracking:
            text = " ".join(text)            # letterspaced display title
        size = max_px
        font = _load_font(size)
        while size > 10:
            box = draw.textbbox((0, 0), text, font=font)
            if box[2] - box[0] <= w * 0.86:
                break
            size = int(size * 0.92)
            font = _load_font(size)
        box = draw.textbbox((0, 0), text, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
        x = (w - tw) // 2
        if band:                              # translucent legibility band
            draw.rectangle([x - 14, y - 6, x + tw + 14, y + th + 10],
                           fill=(0, 0, 0, 110))
        draw.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0, 160))  # shadow
        draw.text((x, y), text, font=font, fill=color)
        return y + th + 12
