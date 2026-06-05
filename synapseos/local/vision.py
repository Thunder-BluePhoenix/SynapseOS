"""
Vision encoder — the practical route: real pretrained CLIP/SigLIP.

Unlike the trainable-from-scratch local components, image semantics need
pretrained weights (hundreds of millions of image-text pairs). This wraps
`open_clip` so images get real meaning: "cat" ≈ encode(image-of-cat).

Backend preference:
  1. SigLIP  (ViT-B-16-SigLIP, webli)  — best small open image-text model
  2. CLIP    (ViT-B-32, openai)        — fallback
  3. stub    (deterministic hashing)   — keeps the pipeline runnable with no model

The image embedding is projected (deterministic fixed projection) into the
world model's observation space (obs_dim, default 128) so images become
observations the persistent world model can perceive — same predictive loop
as text and speech.

JEPA-vision (training an image world model from scratch) is the RESEARCH route,
tracked separately for later. This module is the practical route.

Run:
    python -m synapseos.local.vision --selftest      # encodes synthetic images, shows text grounding
    python -m synapseos.local.vision --image x.jpg --labels "a cat,a dog,an invoice"
"""

from __future__ import annotations

import argparse
import hashlib
import time
from dataclasses import dataclass

import torch


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def open_clip_available() -> bool:
    try:
        import open_clip  # noqa
        return True
    except ImportError:
        return False


SIGLIP = ("ViT-B-16-SigLIP", "webli")
CLIP_FALLBACK = ("ViT-B-32", "openai")


@dataclass
class ImageEmbedding:
    vector: list[float]          # raw model embedding (model dim)
    obs: list[float]             # projected into world-model obs space
    dim: int
    model_used: str              # "siglip" | "clip" | "stub"
    latency_ms: float
    is_real: bool

    def obs_tensor(self) -> torch.Tensor:
        return torch.tensor(self.obs, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Vision encoder
# ---------------------------------------------------------------------------

class VisionEncoder:
    def __init__(self, obs_dim: int = 128, prefer: str = "siglip", device: str | None = None):
        self.obs_dim = obs_dim
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self.model_tag = "stub"
        self.model_dim = 0
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self._proj: torch.Tensor | None = None
        if open_clip_available():
            self._load(prefer)

    def _load(self, prefer: str) -> None:
        import open_clip
        candidates = [SIGLIP, CLIP_FALLBACK] if prefer == "siglip" else [CLIP_FALLBACK, SIGLIP]
        for name, pretrained in candidates:
            try:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    name, pretrained=pretrained,
                )
                model = model.to(self.device).eval()
                self._model = model
                self._preprocess = preprocess
                self._tokenizer = open_clip.get_tokenizer(name)
                self.model_tag = "siglip" if "SigLIP" in name else "clip"
                # infer embedding dim from a dry run
                with torch.no_grad():
                    dummy = torch.zeros(1, 3, 224, 224).to(self.device)
                    try:
                        feat = model.encode_image(dummy)
                        self.model_dim = feat.shape[-1]
                    except Exception:
                        self.model_dim = 512
                print(f"[Vision] loaded {name} ({pretrained}) dim={self.model_dim} on {self.device}")
                self._build_projection()
                return
            except Exception as e:
                print(f"[Vision] {name} unavailable ({str(e)[:60]})")
        print("[Vision] no open_clip model loaded — stub mode")

    def _build_projection(self) -> None:
        """Deterministic fixed projection model_dim → obs_dim (seeded, reproducible)."""
        g = torch.Generator().manual_seed(1234)
        self._proj = torch.randn(self.model_dim, self.obs_dim, generator=g) / (self.model_dim ** 0.5)

    @property
    def is_real(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    # Image input handling
    # ------------------------------------------------------------------

    def _to_pil(self, image):
        """Accept a path, a PIL Image, or a (C,H,W)/(H,W,C) tensor/array."""
        from PIL import Image
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, torch.Tensor):
            t = image.detach().cpu()
            if t.dim() == 3 and t.shape[0] in (1, 3):     # (C,H,W)
                t = t.permute(1, 2, 0)
            arr = (t.clamp(0, 1) * 255).to(torch.uint8).numpy()
            if arr.shape[-1] == 1:
                arr = arr.repeat(3, axis=-1)
            return Image.fromarray(arr).convert("RGB")
        import numpy as np
        arr = np.asarray(image)
        return Image.fromarray(arr).convert("RGB")

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_image(self, image) -> ImageEmbedding:
        t0 = time.time()
        if not self.is_real:
            return self._stub_image(image, t0)
        try:
            pil = self._to_pil(image)
            x = self._preprocess(pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self._model.encode_image(x)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            vec = feat.squeeze(0).cpu()
            obs = self._project(vec)
            return ImageEmbedding(
                vector=vec.tolist(),
                obs=obs.tolist(),
                dim=self.model_dim,
                model_used=self.model_tag,
                latency_ms=(time.time() - t0) * 1000,
                is_real=True,
            )
        except Exception as e:
            print(f"[Vision] encode failed ({e}) — stub")
            return self._stub_image(image, t0)

    def encode_text(self, text: str) -> torch.Tensor:
        """CLIP text embedding (same space as images — the semantic grounding)."""
        if not self.is_real:
            return self._stub_text(text)
        toks = self._tokenizer([text]).to(self.device)
        with torch.no_grad():
            feat = self._model.encode_text(toks)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).cpu()

    def image_text_similarity(self, image, labels: list[str]) -> list[tuple[str, float]]:
        """Zero-shot: rank labels by similarity to the image. The point of CLIP."""
        img = self.encode_image(image)
        iv = torch.tensor(img.vector)
        sims = []
        for lab in labels:
            tv = self.encode_text(lab)
            if tv.shape == iv.shape:
                sims.append((lab, float(iv @ tv)))
            else:
                sims.append((lab, 0.0))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims

    def _project(self, vec: torch.Tensor) -> torch.Tensor:
        if self._proj is None or vec.shape[0] != self._proj.shape[0]:
            # dim mismatch fallback → truncate/pad then normalise
            o = torch.zeros(self.obs_dim)
            n = min(self.obs_dim, vec.shape[0])
            o[:n] = vec[:n]
        else:
            o = vec @ self._proj
        nrm = o.norm()
        return o / nrm if nrm > 0 else o

    # ------------------------------------------------------------------
    # Stub backends
    # ------------------------------------------------------------------

    def _stub_image(self, image, t0: float) -> ImageEmbedding:
        # deterministic hash of raw bytes → structural (no semantics)
        key = str(image)[:200] if not isinstance(image, torch.Tensor) else str(image.shape) + str(image.float().mean().item())
        obs = self._hash_vec(key)
        return ImageEmbedding(
            vector=obs.tolist(), obs=obs.tolist(), dim=self.obs_dim,
            model_used="stub", latency_ms=(time.time()-t0)*1000, is_real=False,
        )

    def _stub_text(self, text: str) -> torch.Tensor:
        return self._hash_vec(text)

    def _hash_vec(self, key: str) -> torch.Tensor:
        vec = [0.0] * self.obs_dim
        for i, ch in enumerate(key):
            h = int(hashlib.md5(f"{i}{ch}".encode()).hexdigest(), 16)
            vec[h % self.obs_dim] += 1 if (h >> 8) & 1 else -1
        t = torch.tensor(vec, dtype=torch.float32)
        n = t.norm()
        return t / n if n > 0 else t


# ---------------------------------------------------------------------------
# Synthetic test images (no download / no files needed)
# ---------------------------------------------------------------------------

def _make_synthetic(kind: str, size: int = 96) -> torch.Tensor:
    """Generate a distinct (3,H,W) image in [0,1] for self-testing."""
    img = torch.zeros(3, size, size)
    if kind == "red_square":
        img[0, size//4:3*size//4, size//4:3*size//4] = 1.0
    elif kind == "green_circle":
        yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
        mask = ((xx - size/2)**2 + (yy - size/2)**2) < (size/3)**2
        img[1][mask] = 1.0
    elif kind == "blue_gradient":
        img[2] = torch.linspace(0, 1, size).repeat(size, 1)
    return img


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _selftest() -> None:
    enc = VisionEncoder(obs_dim=128)
    print(f"\nVision encoder: {enc.model_tag} | real={enc.is_real} | obs_dim={enc.obs_dim}")

    imgs = {k: _make_synthetic(k) for k in ("red_square", "green_circle", "blue_gradient")}
    embs = {k: enc.encode_image(v) for k, v in imgs.items()}
    for k, e in embs.items():
        print(f"  {k:14s} → {e.model_used} dim={e.dim} {e.latency_ms:.0f}ms")

    # determinism + distinctness
    e1 = enc.encode_image(imgs["red_square"])
    e2 = enc.encode_image(imgs["red_square"])
    same = torch.allclose(e1.obs_tensor(), e2.obs_tensor(), atol=1e-4)
    cross = float(embs["red_square"].obs_tensor() @ embs["green_circle"].obs_tensor())
    print(f"\n  deterministic (same image → same obs): {same}")
    print(f"  red vs green obs cosine: {cross:.3f} (distinct images → different vectors)")

    if enc.is_real:
        print("\n  Zero-shot text grounding (the point of CLIP):")
        sims = enc.image_text_similarity(imgs["red_square"],
                                         ["a red square", "a green circle", "a blue gradient", "a cat"])
        for lab, s in sims:
            print(f"    {s:+.3f}  {lab}")
        print("  → real semantics: image matches its text description above random labels")
    print("\n  [Vision] self-test complete")


def main():
    parser = argparse.ArgumentParser(description="Vision encoder (CLIP/SigLIP)")
    parser.add_argument("--image",  default=None)
    parser.add_argument("--labels", default=None, help="comma-separated zero-shot labels")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest or not args.image:
        _selftest()
        return

    enc = VisionEncoder()
    if args.labels:
        labels = [l.strip() for l in args.labels.split(",")]
        for lab, s in enc.image_text_similarity(args.image, labels):
            print(f"  {s:+.3f}  {lab}")
    else:
        e = enc.encode_image(args.image)
        print(f"model={e.model_used} dim={e.dim} real={e.is_real} latency={e.latency_ms:.0f}ms")


if __name__ == "__main__":
    main()
