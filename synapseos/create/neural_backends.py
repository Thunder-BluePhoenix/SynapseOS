"""
Create — optional neural generation backends (local, graceful).

The procedural generators are the always-available core. When heavier local
neural tools exist on this machine, MUSE can upgrade:

  ComfyUI    — http://localhost:8188 (user has it at ~/projs/comfyui).
               If the server is running AND a checkpoint is installed, real
               diffusion images. JSON workflow over HTTP, no SDK.
  Ideogram4  — ~/projs/ideogram4 local 9.3B DiT (weights gated on HF;
               import-path detection only, invoked manually when downloaded).

Everything degrades silently to the procedural generators — these backends
are upgrades, never dependencies.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

COMFYUI_URL = "http://127.0.0.1:8188"

_BASIC_WORKFLOW = {
    "1": {"class_type": "CheckpointLoaderSimple",
          "inputs": {"ckpt_name": "{ckpt}"}},
    "2": {"class_type": "CLIPTextEncode",
          "inputs": {"text": "{prompt}", "clip": ["1", 1]}},
    "3": {"class_type": "CLIPTextEncode",
          "inputs": {"text": "blurry, low quality", "clip": ["1", 1]}},
    "4": {"class_type": "EmptyLatentImage",
          "inputs": {"width": 512, "height": 512, "batch_size": 1}},
    "5": {"class_type": "KSampler",
          "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                     "latent_image": ["4", 0], "seed": 0, "steps": 20,
                     "cfg": 7.0, "sampler_name": "euler", "scheduler": "normal",
                     "denoise": 1.0}},
    "6": {"class_type": "VAEDecode",
          "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
    "7": {"class_type": "SaveImage",
          "inputs": {"images": ["6", 0], "filename_prefix": "synapseos"}},
}


def comfyui_status(url: str = COMFYUI_URL, timeout: float = 1.5) -> dict:
    """Is ComfyUI running, and does it have any checkpoint to use?"""
    try:
        with urllib.request.urlopen(f"{url}/object_info/CheckpointLoaderSimple",
                                    timeout=timeout) as resp:
            info = json.loads(resp.read())
        ckpts = (info.get("CheckpointLoaderSimple", {}).get("input", {})
                 .get("required", {}).get("ckpt_name", [[]])[0])
        return {"running": True, "checkpoints": list(ckpts)}
    except Exception:
        return {"running": False, "checkpoints": []}


def generate_via_comfyui(prompt: str, seed: int = 0,
                         url: str = COMFYUI_URL,
                         timeout: float = 120.0) -> str | None:
    """
    Submit a basic txt2img workflow. Returns the output image path, or None
    (caller falls back to procedural generation).
    """
    status = comfyui_status(url)
    if not status["running"] or not status["checkpoints"]:
        return None
    workflow = json.loads(json.dumps(_BASIC_WORKFLOW)
                          .replace("{ckpt}", status["checkpoints"][0])
                          .replace("{prompt}", prompt.replace('"', "")))
    workflow["5"]["inputs"]["seed"] = seed
    try:
        req = urllib.request.Request(
            f"{url}/prompt", data=json.dumps({"prompt": workflow}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            prompt_id = json.loads(resp.read())["prompt_id"]
        deadline = time.time() + timeout
        while time.time() < deadline:
            with urllib.request.urlopen(f"{url}/history/{prompt_id}",
                                        timeout=10) as resp:
                hist = json.loads(resp.read())
            if prompt_id in hist:
                outputs = hist[prompt_id].get("outputs", {})
                for node in outputs.values():
                    for img in node.get("images", []):
                        return str(Path("~/projs/comfyui/output").expanduser()
                                   / img["filename"])
                return None
            time.sleep(1.0)
    except Exception:
        return None
    return None


def ideogram4_available() -> bool:
    """Source is installed; weights are HF-gated and download on first run."""
    return (Path("~/projs/ideogram4").expanduser() / "run_inference.py").exists()
