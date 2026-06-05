"""
Tests for the vision encoder (practical CLIP/SigLIP route) and image perception
in the world model.

The real open_clip model is loaded once (cached weights). Tests that need real
semantics are guarded by `enc.is_real`; structural tests run regardless.

Run:
    python tests/test_vision.py
    python -m pytest tests/test_vision.py -v
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from synapseos.local.vision import (
    VisionEncoder, ImageEmbedding, open_clip_available, _make_synthetic,
)

results = []


def test(name: str):
    def decorator(fn):
        def wrapper():
            t0 = time.time()
            try:
                fn()
                r = {"name": name, "passed": True,  "ms": round((time.time()-t0)*1000, 1), "detail": ""}
            except AssertionError as e:
                r = {"name": name, "passed": False, "ms": round((time.time()-t0)*1000, 1), "detail": str(e)}
            except Exception as e:
                r = {"name": name, "passed": False, "ms": round((time.time()-t0)*1000, 1), "detail": f"{type(e).__name__}: {e}"}
            results.append(r)
        wrapper.__name__ = fn.__name__
        return wrapper
    return decorator


# Shared encoder (loaded once)
_enc = None
def get_enc():
    global _enc
    if _enc is None:
        _enc = VisionEncoder(obs_dim=128)
    return _enc


# ---------------------------------------------------------------------------

@test("vision: open_clip_available returns bool")
def test_available():
    assert isinstance(open_clip_available(), bool)


@test("vision: encoder initialises")
def test_init():
    enc = get_enc()
    assert enc is not None
    assert isinstance(enc.is_real, bool)
    assert enc.model_tag in ("siglip", "clip", "stub")


@test("vision: synthetic image generator produces (3,H,W)")
def test_synthetic():
    img = _make_synthetic("red_square", size=64)
    assert img.shape == (3, 64, 64)
    assert img.max() <= 1.0 and img.min() >= 0.0


@test("vision: encode_image returns ImageEmbedding with obs in obs_dim")
def test_encode_image():
    enc = get_enc()
    img = _make_synthetic("red_square")
    emb = enc.encode_image(img)
    assert isinstance(emb, ImageEmbedding)
    assert len(emb.obs) == 128
    assert emb.latency_ms >= 0


@test("vision: encoding is deterministic (same image → same obs)")
def test_deterministic():
    enc = get_enc()
    img = _make_synthetic("green_circle")
    a = enc.encode_image(img)
    b = enc.encode_image(img)
    assert torch.allclose(a.obs_tensor(), b.obs_tensor(), atol=1e-4)


@test("vision: obs vector is L2-normalised")
def test_normalised():
    enc = get_enc()
    emb = enc.encode_image(_make_synthetic("blue_gradient"))
    assert abs(emb.obs_tensor().norm().item() - 1.0) < 1e-4


@test("vision: distinct images produce distinct obs vectors")
def test_distinct():
    enc = get_enc()
    red   = enc.encode_image(_make_synthetic("red_square")).obs_tensor()
    green = enc.encode_image(_make_synthetic("green_circle")).obs_tensor()
    assert not torch.allclose(red, green, atol=1e-2)


@test("vision: encode_image accepts a PIL image")
def test_pil_input():
    from PIL import Image
    import numpy as np
    enc = get_enc()
    arr = (np.random.RandomState(0).rand(64, 64, 3) * 255).astype("uint8")
    pil = Image.fromarray(arr)
    emb = enc.encode_image(pil)
    assert len(emb.obs) == 128


@test("vision: REAL zero-shot grounding — matching label scores highest")
def test_real_grounding():
    enc = get_enc()
    if not enc.is_real:
        return  # skip if no model
    img = _make_synthetic("red_square")
    sims = enc.image_text_similarity(img, ["a red square", "a green circle", "a cat", "a blue gradient"])
    top_label = sims[0][0]
    assert top_label == "a red square", f"Expected 'a red square' top, got '{top_label}'"


@test("vision: REAL encode_text returns same-dim embedding as image")
def test_real_text_embed():
    enc = get_enc()
    if not enc.is_real:
        return
    tv = enc.encode_text("a photograph of a cat")
    assert tv.shape[0] == enc.model_dim


@test("vision: stub fallback works when forced (no model)")
def test_stub_fallback():
    enc = VisionEncoder(obs_dim=64)
    enc._model = None   # force stub path
    emb = enc.encode_image(_make_synthetic("red_square"))
    assert emb.model_used == "stub"
    assert len(emb.obs) == 64
    assert not emb.is_real


# ---------------------------------------------------------------------------
# World model image perception
# ---------------------------------------------------------------------------

@test("world model: perceive_image runs the predictive loop")
def test_wm_perceive_image():
    import tempfile, os
    from synapseos.world_model.world_model import WorldModel
    d = tempfile.mkdtemp()
    wm = WorldModel(state_path=os.path.join(d, "s.json"),
                    episodes_path=os.path.join(d, "e.json"),
                    snapshot_dir=os.path.join(d, "snap"))
    r = wm.perceive_image(_make_synthetic("red_square"), caption="a red square")
    assert r.surprise >= 0.0
    assert r.step == 1
    assert r.novelty in ("novel", "familiar", "expected")


@test("world model: image perception advances same state as text")
def test_wm_image_text_shared_state():
    import tempfile, os
    from synapseos.world_model.world_model import WorldModel
    d = tempfile.mkdtemp()
    wm = WorldModel(state_path=os.path.join(d, "s.json"),
                    episodes_path=os.path.join(d, "e.json"),
                    snapshot_dir=os.path.join(d, "snap"))
    wm.perceive("a text observation about invoices")
    wm.perceive_image(_make_synthetic("green_circle"), caption="a green circle")
    # both modalities advanced the one persistent state
    assert wm.state.step == 2
    assert len(wm.memory) >= 1


@test("world model: repeated image becomes less surprising")
def test_wm_image_learns():
    import tempfile, os
    from synapseos.world_model.world_model import WorldModel
    d = tempfile.mkdtemp()
    wm = WorldModel(state_path=os.path.join(d, "s.json"),
                    episodes_path=os.path.join(d, "e.json"),
                    snapshot_dir=os.path.join(d, "snap"))
    img = _make_synthetic("blue_gradient")
    first = wm.perceive_image(img, caption="blue gradient").surprise
    last = first
    for _ in range(5):
        last = wm.perceive_image(img, caption="blue gradient").surprise
    assert last < first, f"Repeated image surprise should drop: {first:.3f} → {last:.3f}"


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_available, test_init, test_synthetic, test_encode_image,
        test_deterministic, test_normalised, test_distinct, test_pil_input,
        test_real_grounding, test_real_text_embed, test_stub_fallback,
        test_wm_perceive_image, test_wm_image_text_shared_state, test_wm_image_learns,
    ]
    for t in tests:
        t()

    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed
    print(f"\n{'='*60}")
    print(f"Vision Tests: {passed}/{len(results)} passed\n")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        detail = f"  → {r['detail']}" if not r["passed"] else ""
        print(f"  [{status}] {r['name']}  ({r['ms']}ms){detail}")
    sys.exit(0 if failed == 0 else 1)
