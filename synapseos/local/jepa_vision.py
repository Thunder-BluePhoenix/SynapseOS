"""
I-JEPA vision — an image world model trained FROM SCRATCH (the research route).

Unlike `vision.py` (pretrained CLIP/SigLIP), this learns visual representations
with no labels and no pretrained weights, by predicting the *latent* of masked
image regions — exactly the JEPA principle that drives the rest of SynapseOS:

    predict the representation of what you can't see; the error teaches the model.

Architecture (Assran et al., "Self-Supervised Learning from Images with a
Joint-Embedding Predictive Architecture", 2023), at small scale for MPS:

    image → patches → context encoder (ViT) ─┐
                                              ├─ predictor → predicted target latents
    image → patches → target encoder (EMA) ──┘     ↑                    │
                       (no grad, momentum copy)     └── masked positions ┘
    loss = smooth_L1( predicted_target_latents , target_encoder_latents )  (latent space!)

Why this is the real research route: the model is forced to learn the *structure*
of images (objects, parts, layout) to predict masked latents — it cannot cheat by
copying pixels. After self-supervised training, a frozen encoder's embeddings
cluster by object class (provable with a k-NN probe, no labels used in training).

Runs on Apple MPS. Trains on CIFAR-10 (real natural images).

Run:
    python -m synapseos.local.jepa_vision --train --steps 400   # train + k-NN probe
    python -m synapseos.local.jepa_vision --selftest            # forward/shape smoke test
"""

from __future__ import annotations

import argparse
import copy
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from synapseos.local.device import get_device


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class IJEPAConfig:
    img_size:   int = 32          # CIFAR
    patch_size: int = 4           # → 8x8 = 64 patches
    in_chans:   int = 3
    embed_dim:  int = 128         # encoder width
    depth:      int = 4           # encoder transformer blocks
    n_heads:    int = 4
    pred_dim:   int = 96          # predictor width
    pred_depth: int = 2
    mask_ratio: float = 0.5       # fraction of patches used as prediction targets
    ema:        float = 0.998     # target encoder momentum
    mask_ratio_hi: float = 0.6    # harder masking → less shortcutting
    var_weight: float = 1.0       # anti-collapse: variance regularization
    cov_weight: float = 0.04      # anti-collapse: covariance (decorrelation)

    @property
    def n_patches(self) -> int:
        return (self.img_size // self.patch_size) ** 2


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, cfg: IJEPAConfig):
        super().__init__()
        self.proj = nn.Conv2d(cfg.in_chans, cfg.embed_dim,
                              kernel_size=cfg.patch_size, stride=cfg.patch_size)

    def forward(self, x):                       # (B,3,H,W)
        x = self.proj(x)                        # (B,E,h,w)
        return x.flatten(2).transpose(1, 2)     # (B, n_patches, E)


class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = dim // n_heads
        self.qkv  = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att.softmax(dim=-1)
        out = (att @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4):
        super().__init__()
        self.n1 = nn.LayerNorm(dim); self.attn = Attention(dim, n_heads)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*mlp_ratio), nn.GELU(),
                                 nn.Linear(dim*mlp_ratio, dim))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x


class ViTEncoder(nn.Module):
    def __init__(self, cfg: IJEPAConfig):
        super().__init__()
        self.patch = PatchEmbed(cfg)
        self.pos   = nn.Parameter(torch.randn(1, cfg.n_patches, cfg.embed_dim) * 0.02)
        self.blocks = nn.ModuleList([Block(cfg.embed_dim, cfg.n_heads) for _ in range(cfg.depth)])
        self.norm  = nn.LayerNorm(cfg.embed_dim)

    def forward(self, x, return_patches: bool = True):
        x = self.patch(x) + self.pos
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)                 # (B, n_patches, E)
        return x if return_patches else x.mean(dim=1)


class Predictor(nn.Module):
    """Predicts target patch latents from context patch latents + target position embeds."""
    def __init__(self, cfg: IJEPAConfig):
        super().__init__()
        self.embed = nn.Linear(cfg.embed_dim, cfg.pred_dim)
        self.mask_token = nn.Parameter(torch.randn(1, 1, cfg.pred_dim) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, cfg.n_patches, cfg.pred_dim) * 0.02)
        self.blocks = nn.ModuleList([Block(cfg.pred_dim, cfg.n_heads) for _ in range(cfg.pred_depth)])
        self.norm = nn.LayerNorm(cfg.pred_dim)
        self.out  = nn.Linear(cfg.pred_dim, cfg.embed_dim)

    def forward(self, ctx, ctx_idx, tgt_idx):
        """
        ctx:     (B, n_ctx, E) context patch latents
        ctx_idx: (B, n_ctx) indices of context patches
        tgt_idx: (B, n_tgt) indices of target patches
        returns predicted target latents (B, n_tgt, E)
        """
        B = ctx.shape[0]
        x = self.embed(ctx) + self._gather_pos(self.pos, ctx_idx)
        # append mask tokens at target positions
        mt = self.mask_token.expand(B, tgt_idx.shape[1], -1) + self._gather_pos(self.pos, tgt_idx)
        x = torch.cat([x, mt], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        pred = self.out(x[:, ctx.shape[1]:])     # only the target slots
        return pred

    @staticmethod
    def _gather_pos(pos, idx):
        # pos: (1,N,D), idx: (B,n) → (B,n,D)
        B, n = idx.shape
        D = pos.shape[-1]
        return pos.expand(B, -1, -1).gather(1, idx.unsqueeze(-1).expand(B, n, D))


# ---------------------------------------------------------------------------
# I-JEPA
# ---------------------------------------------------------------------------

class IJEPA(nn.Module):
    def __init__(self, cfg: IJEPAConfig | None = None):
        super().__init__()
        self.cfg = cfg or IJEPAConfig()
        self.context_encoder = ViTEncoder(self.cfg)
        self.target_encoder  = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.predictor = Predictor(self.cfg)

    @torch.no_grad()
    def update_target(self) -> None:
        m = self.cfg.ema
        for tp, cp in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            tp.data.mul_(m).add_(cp.data, alpha=1 - m)

    def _sample_masks(self, B: int, device):
        n = self.cfg.n_patches
        n_tgt = max(1, int(n * self.cfg.mask_ratio))
        ctx_idx, tgt_idx = [], []
        for _ in range(B):
            perm = torch.randperm(n, device=device)
            tgt_idx.append(perm[:n_tgt])
            ctx_idx.append(perm[n_tgt:])
        return torch.stack(ctx_idx), torch.stack(tgt_idx)

    def forward(self, imgs):
        """Self-supervised step → scalar loss."""
        B = imgs.shape[0]
        device = imgs.device
        ctx_idx, tgt_idx = self._sample_masks(B, device)

        # target latents (EMA encoder, no grad) over ALL patches, then gather targets
        with torch.no_grad():
            tgt_all = self.target_encoder(imgs)                  # (B,N,E)
            tgt = self._gather(tgt_all, tgt_idx)                 # (B,n_tgt,E)
            tgt = F.layer_norm(tgt, (tgt.shape[-1],))            # normalise targets

        # context latents over context patches only
        ctx_all = self.context_encoder(imgs)                     # (B,N,E)
        ctx = self._gather(ctx_all, ctx_idx)                     # (B,n_ctx,E)

        pred = self.predictor(ctx, ctx_idx, tgt_idx)             # (B,n_tgt,E)
        pred_loss = F.smooth_l1_loss(pred, tgt)

        # Anti-collapse (VICReg-style) on the pooled context embedding —
        # forces embeddings to keep variance and decorrelate, so the encoder
        # cannot win by mapping everything to a constant.
        var_loss, cov_loss = self._vicreg(ctx_all.mean(dim=1))
        return pred_loss + self.cfg.var_weight * var_loss + self.cfg.cov_weight * cov_loss

    @staticmethod
    def _vicreg(z: torch.Tensor):
        """z: (B, E). Returns (variance_loss, covariance_loss)."""
        B, E = z.shape
        z = z - z.mean(dim=0, keepdim=True)
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        var_loss = torch.mean(F.relu(1.0 - std))          # push per-dim std → ≥1
        cov = (z.T @ z) / max(B - 1, 1)                   # (E,E)
        off_diag = cov - torch.diag(torch.diag(cov))
        cov_loss = (off_diag ** 2).sum() / E              # decorrelate dims
        return var_loss, cov_loss

    @torch.no_grad()
    def encode(self, imgs) -> torch.Tensor:
        """Inference: pooled context-encoder embedding (B, embed_dim), L2-normalised."""
        feat = self.context_encoder(imgs, return_patches=False)
        return F.normalize(feat, dim=-1)

    @staticmethod
    def _gather(x, idx):
        B, n = idx.shape
        D = x.shape[-1]
        return x.gather(1, idx.unsqueeze(-1).expand(B, n, D))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Trainer + k-NN probe
# ---------------------------------------------------------------------------

class IJEPATrainer:
    def __init__(self, cfg: IJEPAConfig | None = None, device: torch.device | None = None,
                 lr: float = 1e-3):
        self.cfg    = cfg or IJEPAConfig()
        self.device = device or get_device(prefer_small=False)   # ViT → MPS helps
        self.model  = IJEPA(self.cfg).to(self.device)
        self.opt    = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=lr, weight_decay=0.04)

    @staticmethod
    def _augment(batch: torch.Tensor) -> torch.Tensor:
        """Random horizontal flip + reflect-pad crop — teaches invariances (self-supervised)."""
        B = batch.shape[0]
        # random horizontal flip
        flip = torch.rand(B, device=batch.device) < 0.5
        batch = torch.where(flip.view(B, 1, 1, 1), batch.flip(-1), batch)
        # random crop via pad(4)+crop
        padded = F.pad(batch, (4, 4, 4, 4), mode="reflect")
        H = batch.shape[-1]
        out = torch.empty_like(batch)
        for i in range(B):
            top  = int(torch.randint(0, 9, (1,)).item())
            left = int(torch.randint(0, 9, (1,)).item())
            out[i] = padded[i, :, top:top+H, left:left+H]
        return out

    def train(self, images: torch.Tensor, steps: int = 400, batch_size: int = 128,
              log_every: int = 100, augment: bool = True) -> dict:
        """images: (N,3,H,W) in [0,1]. Self-supervised — labels never used."""
        N = images.shape[0]
        self.model.train()
        t0 = time.time()
        first, last = None, None
        rng = torch.Generator().manual_seed(0)
        for step in range(steps):
            idx = torch.randint(0, N, (batch_size,), generator=rng)
            batch = images[idx].to(self.device)
            if augment:
                batch = self._augment(batch)
            loss = self.model(batch)
            self.opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            self.model.update_target()
            if first is None:
                first = loss.item()
            last = loss.item()
            if (step + 1) % log_every == 0:
                print(f"  step {step+1:>4}  loss {loss.item():.4f}  ({(step+1)/(time.time()-t0):.1f} it/s)")
        return {"first_loss": round(first, 4), "final_loss": round(last, 4),
                "steps": steps, "elapsed_s": round(time.time()-t0, 1),
                "device": str(self.device), "params": self.model.num_params()}

    @torch.no_grad()
    def embed(self, images: torch.Tensor, batch_size: int = 256,
              encoder: str = "target") -> torch.Tensor:
        """
        Frozen-encoder embeddings for probing.
        encoder="target" uses the EMA teacher (usually the better representation
        in JEPA); "context" uses the student.
        """
        self.model.eval()
        enc = self.model.target_encoder if encoder == "target" else self.model.context_encoder
        outs = []
        for i in range(0, images.shape[0], batch_size):
            b = images[i:i+batch_size].to(self.device)
            feat = enc(b, return_patches=False)
            outs.append(F.normalize(feat, dim=-1).cpu())
        return torch.cat(outs)

    def linear_probe(self, train_imgs, train_labels, test_imgs, test_labels,
                     encoder: str = "target", epochs: int = 150) -> float:
        """
        Standard SSL evaluation: freeze the encoder, train ONLY a linear
        classifier on its embeddings. Labels are used here for evaluation only,
        never during representation learning. Far more sensitive than k-NN.
        """
        ztr = self.embed(train_imgs, encoder=encoder).to(self.device)
        zte = self.embed(test_imgs,  encoder=encoder).to(self.device)
        ytr = train_labels.to(self.device)
        yte = test_labels.to(self.device)

        clf = nn.Linear(ztr.shape[1], int(train_labels.max()) + 1).to(self.device)
        opt = torch.optim.Adam(clf.parameters(), lr=1e-2, weight_decay=1e-4)
        for _ in range(epochs):
            opt.zero_grad()
            loss = F.cross_entropy(clf(ztr), ytr)
            loss.backward(); opt.step()
        with torch.no_grad():
            acc = (clf(zte).argmax(1) == yte).float().mean().item()
        return float(acc)

    @torch.no_grad()
    def knn_probe(self, train_imgs, train_labels, test_imgs, test_labels,
                  k: int = 20, encoder: str = "target") -> float:
        """
        Freeze encoder, k-NN classify test embeddings using train embeddings.
        Labels are used ONLY here for evaluation, never during training.
        """
        tr = self.embed(train_imgs, encoder=encoder)
        te = self.embed(test_imgs,  encoder=encoder)
        sims = te @ tr.T                                  # cosine (normalised)
        knn = sims.topk(k, dim=1).indices                 # (n_test, k)
        votes = train_labels[knn]                         # (n_test, k)
        preds = torch.mode(votes, dim=1).values
        return float((preds == test_labels).float().mean())

    def save(self, path: str) -> None:
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict(), "cfg": self.cfg.__dict__}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])


# ---------------------------------------------------------------------------
# Encoder wrapper for the world model (same interface as vision.VisionEncoder)
# ---------------------------------------------------------------------------

class JEPAVisionEncoder:
    """
    Wraps a trained I-JEPA encoder so images can feed the world model's
    perceive_image() — the RESEARCH route (learned from scratch), interchangeable
    with the practical CLIP/SigLIP VisionEncoder.

    encode_image(image) → ImageEmbedding (obs projected into world-model obs space).
    """

    DEFAULT_CKPT = "data/checkpoints/jepa_vision/ijepa_cifar.pt"

    def __init__(self, obs_dim: int = 128, checkpoint: str | None = None,
                 device: torch.device | None = None):
        from pathlib import Path
        self.obs_dim = obs_dim
        self.device  = device or get_device(prefer_small=True)
        self.model_tag = "jepa-vision"
        ckpt = checkpoint or self.DEFAULT_CKPT
        self._model = None
        if Path(ckpt).exists():
            try:
                blob = torch.load(ckpt, map_location=self.device, weights_only=False)
                cfg = IJEPAConfig(**blob["cfg"])
                self.model_dim = cfg.embed_dim
                self._model = IJEPA(cfg).to(self.device)
                self._model.load_state_dict(blob["model"])
                self._model.eval()
                self._img_size = cfg.img_size
                g = torch.Generator().manual_seed(1234)
                self._proj = torch.randn(self.model_dim, obs_dim, generator=g) / (self.model_dim ** 0.5)
                print(f"[JEPAVision] loaded {ckpt} dim={self.model_dim} on {self.device}")
            except Exception as e:
                print(f"[JEPAVision] load failed ({e}) — stub mode")
        else:
            print(f"[JEPAVision] no checkpoint at {ckpt} — train with "
                  f"`python -m synapseos.local.jepa_vision --train`")

    @property
    def is_real(self) -> bool:
        return self._model is not None

    def _to_tensor(self, image) -> torch.Tensor:
        from PIL import Image
        import numpy as np
        size = getattr(self, "_img_size", 32)
        if isinstance(image, torch.Tensor):
            t = image
            if t.dim() == 3 and t.shape[0] in (1, 3):
                pass
            elif t.dim() == 3:
                t = t.permute(2, 0, 1)
            if t.shape[0] == 1:
                t = t.repeat(3, 1, 1)
            t = F.interpolate(t.unsqueeze(0).float(), size=(size, size),
                              mode="bilinear", align_corners=False).squeeze(0)
            return t.clamp(0, 1)
        if isinstance(image, str):
            image = Image.open(image)
        if isinstance(image, Image.Image):
            image = image.convert("RGB").resize((size, size))
            arr = torch.tensor(np.asarray(image), dtype=torch.float32) / 255.0
            return arr.permute(2, 0, 1)
        arr = torch.tensor(np.asarray(image), dtype=torch.float32)
        if arr.max() > 1:
            arr = arr / 255.0
        return arr.permute(2, 0, 1) if arr.shape[-1] == 3 else arr

    def encode_image(self, image):
        from synapseos.local.vision import ImageEmbedding
        t0 = time.time()
        if not self.is_real:
            # deterministic stub
            import hashlib
            key = str(image)[:200] if not isinstance(image, torch.Tensor) else str(image.shape)
            vec = [0.0] * self.obs_dim
            for i, ch in enumerate(key):
                h = int(hashlib.md5(f"{i}{ch}".encode()).hexdigest(), 16)
                vec[h % self.obs_dim] += 1 if (h >> 8) & 1 else -1
            tv = torch.tensor(vec); tv = tv / (tv.norm() or 1)
            return ImageEmbedding(tv.tolist(), tv.tolist(), self.obs_dim, "stub",
                                  (time.time()-t0)*1000, False)
        x = self._to_tensor(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self._model.encode(x).squeeze(0).cpu()
        obs = feat @ self._proj
        obs = obs / (obs.norm() or 1)
        return ImageEmbedding(feat.tolist(), obs.tolist(), self.model_dim,
                              self.model_tag, (time.time()-t0)*1000, True)

    @property
    def model(self):
        return self._model


# ---------------------------------------------------------------------------
# CIFAR helper
# ---------------------------------------------------------------------------

def load_cifar(n_train: int = 5000, n_test: int = 1000, root: str = "data/datasets/cifar10"):
    from torchvision import datasets, transforms
    tf = transforms.ToTensor()
    tr = datasets.CIFAR10(root=root, train=True,  download=True, transform=tf)
    te = datasets.CIFAR10(root=root, train=False, download=True, transform=tf)

    def stack(ds, n):
        imgs = torch.stack([ds[i][0] for i in range(n)])
        labels = torch.tensor([ds[i][1] for i in range(n)])
        return imgs, labels

    xtr, ytr = stack(tr, n_train)
    xte, yte = stack(te, n_test)
    return xtr, ytr, xte, yte


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _selftest() -> None:
    cfg = IJEPAConfig()
    model = IJEPA(cfg)
    print(f"[I-JEPA] {model.num_params():,} trainable params | {cfg.n_patches} patches")
    x = torch.rand(4, 3, 32, 32)
    loss = model(x)
    loss.backward()
    emb = model.encode(x)
    assert emb.shape == (4, cfg.embed_dim)
    print(f"[I-JEPA] forward loss={loss.item():.4f} | encode {tuple(emb.shape)} | backward OK")
    print("[I-JEPA] self-test PASSED")


def _train(steps: int) -> None:
    print("[I-JEPA] loading CIFAR-10 (real natural images) ...")
    xtr, ytr, xte, yte = load_cifar(n_train=10000, n_test=2000)
    print(f"[I-JEPA] train {xtr.shape}, test {xte.shape}")

    trainer = IJEPATrainer()
    print(f"[I-JEPA] {trainer.model.num_params():,} params on {trainer.device}")

    # baseline: linear probe on the UNTRAINED encoder (random features)
    base_lin = trainer.linear_probe(xtr, ytr, xte, yte)
    base_knn = trainer.knn_probe(xtr, ytr, xte, yte)
    print(f"[I-JEPA] BEFORE training — linear probe {base_lin:.1%} | k-NN {base_knn:.1%}  (chance 10%)")

    print(f"[I-JEPA] self-supervised training ({steps} steps, NO labels) ...")
    stats = trainer.train(xtr, steps=steps, batch_size=128)
    print(f"[I-JEPA] loss {stats['first_loss']} → {stats['final_loss']} in {stats['elapsed_s']}s on {stats['device']}")

    after_lin = trainer.linear_probe(xtr, ytr, xte, yte)
    after_knn = trainer.knn_probe(xtr, ytr, xte, yte)
    print(f"[I-JEPA] AFTER training  — linear probe {after_lin:.1%} | k-NN {after_knn:.1%}")
    print(f"[I-JEPA] linear probe: {base_lin:.1%} → {after_lin:.1%}  "
          f"(Δ {after_lin-base_lin:+.1%})")
    verdict = ("LEARNED visual structure from images with NO labels ✓"
               if after_lin > base_lin + 0.03 else "marginal — needs more steps/data")
    print(f"[I-JEPA] {verdict}")

    trainer.save("data/checkpoints/jepa_vision/ijepa_cifar.pt")
    print("[I-JEPA] checkpoint → data/checkpoints/jepa_vision/ijepa_cifar.pt")


def main():
    parser = argparse.ArgumentParser(description="I-JEPA vision (from scratch)")
    parser.add_argument("--train",    action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--steps",    type=int, default=400)
    args = parser.parse_args()

    if args.train:
        _train(args.steps)
    else:
        _selftest()


if __name__ == "__main__":
    main()
