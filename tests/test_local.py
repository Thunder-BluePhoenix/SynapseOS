"""
Tests for synapseos/local — install-free CPU/MPS alternatives to GPU blockers.

  mamba_cpu  → pure-PyTorch selective-scan SSM (replaces mamba-ssm)
  liquid_cfc → pure-PyTorch CfC liquid cell (replaces ncps)
  sft_trainer→ pure-PyTorch SFT trainer (replaces trl)
  tts_macos  → macOS `say` real TTS (replaces pyttsx3)

These verify each component actually runs and learns on this machine.

Run:
    python tests/test_local.py
    python -m pytest tests/test_local.py -v
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from synapseos.local.device import get_device, device_info
from synapseos.local.mamba_cpu import (
    MinimalMamba, MambaConfig, bytes_encode, MambaBlock,
)
from synapseos.local.liquid_cfc import CfCCell, LiquidNet
from synapseos.local.sft_trainer import (
    SFTTrainer, SFTConfig, TinyDecoder, encode_example, format_pair,
)
from synapseos.local.tts_macos import MacTTS, say_available, TTSResult


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


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

@test("device: get_device returns a torch.device")
def test_device():
    dev = get_device()
    assert isinstance(dev, torch.device)


@test("device: prefer_small returns CPU")
def test_device_small():
    dev = get_device(prefer_small=True)
    assert dev.type == "cpu"


@test("device: device_info has required keys")
def test_device_info():
    info = device_info()
    for k in ("mps_available", "cuda_available", "cpu_threads", "selected", "torch_version"):
        assert k in info


# ---------------------------------------------------------------------------
# Mamba (pure PyTorch)
# ---------------------------------------------------------------------------

@test("mamba: config computes d_inner and dt_rank")
def test_mamba_config():
    cfg = MambaConfig(d_model=128, expand=2)
    assert cfg.d_inner == 256
    assert cfg.dt_rank == 8   # ceil(128/16)


@test("mamba: bytes_encode produces (1, L) long tensor")
def test_mamba_bytes_encode():
    x = bytes_encode("hello", max_len=10)
    assert x.shape[0] == 1
    assert x.dtype == torch.long
    assert x.shape[1] == 5


@test("mamba: forward produces correct logit shape")
def test_mamba_forward():
    cfg   = MambaConfig(vocab_size=256, d_model=64, n_layers=2, d_state=8)
    model = MinimalMamba(cfg)
    x = bytes_encode("Sales Invoice", max_len=16)
    logits = model(x)
    assert logits.shape == (1, x.shape[1], 256)


@test("mamba: encode produces pooled latent")
def test_mamba_encode():
    cfg   = MambaConfig(vocab_size=256, d_model=48, n_layers=2, d_state=8)
    model = MinimalMamba(cfg)
    x = bytes_encode("Purchase Order", max_len=16)
    latent = model.encode(x)
    assert latent.shape == (1, 48)


@test("mamba: selective scan is causal (future tokens don't affect past)")
def test_mamba_causal():
    torch.manual_seed(0)
    cfg   = MambaConfig(vocab_size=256, d_model=32, n_layers=1, d_state=8)
    model = MinimalMamba(cfg)
    model.eval()
    x = bytes_encode("abcdefgh", max_len=8)
    with torch.no_grad():
        out_full = model(x)
        # Change the last token — earlier positions' logits should be unchanged
        x2 = x.clone(); x2[0, -1] = (x2[0, -1] + 1) % 256
        out_mod = model(x2)
    # positions 0..L-2 should match (causality)
    assert torch.allclose(out_full[0, :-1], out_mod[0, :-1], atol=1e-4), \
        "Changing last token altered earlier positions — not causal"


@test("mamba: backward pass produces gradients")
def test_mamba_backward():
    cfg   = MambaConfig(vocab_size=256, d_model=32, n_layers=1, d_state=8)
    model = MinimalMamba(cfg)
    x = bytes_encode("test sequence", max_len=16)
    logits = model(x)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, 256), x[:, 1:].reshape(-1))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "Expected gradients after backward"


@test("mamba: trains — loss decreases on repeated corpus")
def test_mamba_trains():
    torch.manual_seed(0)
    cfg   = MambaConfig(vocab_size=256, d_model=64, n_layers=2, d_state=8)
    model = MinimalMamba(cfg)
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-3)
    data  = bytes_encode("frappe.throw('Customer required')" * 3, max_len=96)

    first, last = None, None
    for _ in range(40):
        logits = model(data)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 256), data[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if first is None:
            first = loss.item()
        last = loss.item()
    assert last < first * 0.5, f"Loss should at least halve: {first:.2f} → {last:.2f}"


# ---------------------------------------------------------------------------
# Liquid CfC (pure PyTorch)
# ---------------------------------------------------------------------------

@test("cfc: cell forward produces correct hidden shape")
def test_cfc_cell():
    cell = CfCCell(input_size=4, hidden_size=16)
    x = torch.randn(3, 4)
    h = torch.zeros(3, 16)
    new_h = cell(x, h, ts=1.0)
    assert new_h.shape == (3, 16)


@test("cfc: cell handles irregular time steps")
def test_cfc_irregular_ts():
    cell = CfCCell(input_size=2, hidden_size=8)
    x = torch.randn(1, 2)
    h = torch.zeros(1, 8)
    h1 = cell(x, h, ts=0.5)
    h2 = cell(x, h, ts=2.0)
    # Different time steps should give different states (continuous-time property)
    assert not torch.allclose(h1, h2), "Different Δt should produce different states"


@test("cfc: LiquidNet processes a sequence")
def test_cfc_net_forward():
    net = LiquidNet(input_size=3, hidden_size=16, output_size=1)
    x = torch.randn(4, 20, 3)
    out, h = net(x)
    assert out.shape == (4, 20, 1)
    assert h.shape == (4, 16)


@test("cfc: LiquidNet supports timestamps argument")
def test_cfc_net_timestamps():
    net = LiquidNet(input_size=2, hidden_size=8, output_size=1)
    x  = torch.randn(2, 10, 2)
    ts = torch.rand(2, 10) + 0.5
    out, h = net(x, timestamps=ts)
    assert out.shape == (2, 10, 1)


@test("cfc: adapt_step performs online update without grad")
def test_cfc_adapt_step():
    net = LiquidNet(input_size=3, hidden_size=12, output_size=1)
    h = torch.zeros(1, 12)
    pred, new_h = net.adapt_step(torch.randn(1, 3), h, ts=0.8)
    assert pred.shape == (1, 1)
    assert new_h.shape == (1, 12)
    # adapt_step is under no_grad — new_h should not require grad
    assert not new_h.requires_grad


@test("cfc: trains — MSE decreases fitting a sine")
def test_cfc_trains():
    import math
    torch.manual_seed(0)
    net = LiquidNet(input_size=1, hidden_size=32, output_size=1)
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    L = 40
    t = torch.linspace(0, 4 * math.pi, L + 1)
    x   = torch.sin(t[:-1]).reshape(1, L, 1)
    tgt = torch.sin(t[1:]).reshape(1, L, 1)

    first, last = None, None
    for _ in range(150):
        out, _ = net(x)
        loss = F.mse_loss(out, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        if first is None:
            first = loss.item()
        last = loss.item()
    assert last < first * 0.3, f"MSE should drop substantially: {first:.4f} → {last:.4f}"


# ---------------------------------------------------------------------------
# SFT Trainer (pure PyTorch, replaces trl)
# ---------------------------------------------------------------------------

@test("sft: format_pair builds instruction/response template")
def test_sft_format():
    s = format_pair("What is X?", "X is Y.")
    assert "### Instruction:" in s
    assert "### Response:" in s
    assert "What is X?" in s


@test("sft: encode_example masks the prompt in targets")
def test_sft_encode_masks_prompt():
    ids, tgt = encode_example("Question here", "Answer here", max_len=128)
    assert ids.shape == tgt.shape
    # Some target positions should be masked (-100) for the prompt
    assert (tgt == -100).sum() > 0, "Expected prompt tokens to be masked"
    # Not all should be masked (response remains)
    assert (tgt != -100).sum() > 0, "Expected response tokens to be unmasked"


@test("sft: TinyDecoder forward produces logits")
def test_sft_decoder_forward():
    cfg = SFTConfig(d_model=64, n_heads=4, n_layers=2, max_len=64)
    model = TinyDecoder(cfg)
    idx = torch.randint(0, 256, (2, 32))
    logits = model(idx)
    assert logits.shape == (2, 32, 256)


@test("sft: SFTTrainer initialises with a device")
def test_sft_trainer_init():
    cfg = SFTConfig(d_model=64, n_heads=4, n_layers=2, max_len=64)
    trainer = SFTTrainer(cfg)
    assert trainer.model.num_params() > 0
    assert trainer.device is not None


@test("sft: trainer learns synthetic pairs (loss drops)")
def test_sft_trains():
    pairs = [
        ("What fields are required for a Sales Invoice?",
         "Customer, Posting Date, and at least one Item."),
        ("What fields are required for a Purchase Order?",
         "Supplier, Schedule Date, and at least one Item."),
    ] * 8
    cfg = SFTConfig(d_model=96, n_heads=4, n_layers=2, max_len=192)
    trainer = SFTTrainer(cfg)
    stats = trainer.train(pairs, steps=80, batch_size=4, log_every=1000)
    assert stats["final_loss"] < stats["first_loss"], \
        f"Loss should drop: {stats['first_loss']} → {stats['final_loss']}"


@test("sft: trained model generates non-empty text")
def test_sft_generates():
    pairs = [("Q: required fields?", "A: Customer and Date.")] * 16
    cfg = SFTConfig(d_model=96, n_heads=4, n_layers=2, max_len=128)
    trainer = SFTTrainer(cfg)
    trainer.train(pairs, steps=120, batch_size=4, log_every=1000)
    out = trainer.synthesize("Q: required fields?", max_new=40)
    assert isinstance(out, str)
    assert len(out) > 0


@test("sft: save and load round-trip")
def test_sft_save_load():
    cfg = SFTConfig(d_model=64, n_heads=4, n_layers=2, max_len=64)
    trainer = SFTTrainer(cfg)
    path = "/tmp/test_sft_local.pt"
    trainer.save(path)
    assert Path(path).exists()
    trainer2 = SFTTrainer(cfg)
    trainer2.load(path)   # should not raise


@test("sft: load_pairs reads Alpaca JSONL")
def test_sft_load_pairs():
    import json
    path = "/tmp/test_sft_pairs.jsonl"
    with open(path, "w") as f:
        f.write(json.dumps({"instruction": "Q1", "input": "", "output": "A1"}) + "\n")
        f.write(json.dumps({"instruction": "Q2", "input": "", "output": "A2"}) + "\n")
    trainer = SFTTrainer(SFTConfig(d_model=64, n_heads=4, n_layers=1, max_len=64))
    pairs = trainer.load_pairs(path)
    assert len(pairs) == 2
    assert pairs[0] == ("Q1", "A1")


# ---------------------------------------------------------------------------
# macOS TTS
# ---------------------------------------------------------------------------

@test("tts: say_available returns a bool")
def test_tts_available():
    assert isinstance(say_available(), bool)


@test("tts: MacTTS initialises")
def test_tts_init():
    tts = MacTTS(output_dir="/tmp/test_tts_local")
    assert tts is not None
    assert isinstance(tts.is_real, bool)


@test("tts: synthesize returns TTSResult")
def test_tts_synthesize():
    tts = MacTTS(output_dir="/tmp/test_tts_local")
    result = tts.synthesize("Hello SynapseOS", filename="test_out.aiff")
    assert isinstance(result, TTSResult)
    assert result.text == "Hello SynapseOS"
    assert result.chars == len("Hello SynapseOS")
    assert Path(result.audio_path).exists(), "Audio/stub file should exist"


@test("tts: produces real WAV audio on macOS (or stub elsewhere)")
def test_tts_real_audio():
    tts = MacTTS(output_dir="/tmp/test_tts_local")
    result = tts.synthesize("Sales Invoice validation complete", filename="test_real.aiff")
    if say_available():
        assert result.is_real_audio, "Expected real audio on macOS"
        assert result.model_used == "macos_say"
        # File should be a non-trivial size (real waveform)
        assert Path(result.audio_path).stat().st_size > 1000, "Real audio should be > 1KB"
    else:
        assert result.model_used == "stub"


@test("tts: to_dict is serialisable")
def test_tts_to_dict():
    import json
    tts = MacTTS(output_dir="/tmp/test_tts_local")
    result = tts.synthesize("test", filename="test_dict.aiff")
    json.dumps(result.to_dict())


@test("tts: VoicePipeline uses real TTS when available")
def test_voice_pipeline_real_tts():
    from synapseos.phase5.voice import VoicePipeline
    vp = VoicePipeline(
        kg_path="data/graph/knowledge_graph.json",
        rules_path="data/rules/rules.json",
        prefer_macos_tts=True,
    )
    if say_available():
        assert vp.synthesizer.is_real, "VoicePipeline should use real macOS TTS"
    result = vp.process_text("What fields are required for a Sales Invoice?")
    assert Path(result.synthesis.audio_path).exists()
    vp.shutdown()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_device, test_device_small, test_device_info,
        # Mamba
        test_mamba_config, test_mamba_bytes_encode, test_mamba_forward,
        test_mamba_encode, test_mamba_causal, test_mamba_backward, test_mamba_trains,
        # CfC
        test_cfc_cell, test_cfc_irregular_ts, test_cfc_net_forward,
        test_cfc_net_timestamps, test_cfc_adapt_step, test_cfc_trains,
        # SFT
        test_sft_format, test_sft_encode_masks_prompt, test_sft_decoder_forward,
        test_sft_trainer_init, test_sft_trains, test_sft_generates,
        test_sft_save_load, test_sft_load_pairs,
        # TTS
        test_tts_available, test_tts_init, test_tts_synthesize,
        test_tts_real_audio, test_tts_to_dict, test_voice_pipeline_real_tts,
    ]

    for t in tests:
        t()

    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed

    print(f"\n{'='*60}")
    print(f"Local (CPU/MPS alternatives) Tests: {passed}/{len(results)} passed\n")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        detail = f"  → {r['detail']}" if not r["passed"] else ""
        print(f"  [{status}] {r['name']}  ({r['ms']}ms){detail}")

    sys.exit(0 if failed == 0 else 1)
