"""
Device selection — prefer Apple MPS, fall back to CUDA, then CPU.

On this machine MPS (Metal GPU) is available, giving real GPU acceleration
without CUDA. Small models often run faster on CPU due to MPS kernel-launch
overhead, so `get_device(prefer_small=True)` returns CPU for tiny workloads.
"""

from __future__ import annotations

import torch


def get_device(prefer_small: bool = False) -> torch.device:
    """
    Return the best available torch device.

    prefer_small=True  → CPU for tiny models (MPS overhead not worth it)
    prefer_small=False → MPS > CUDA > CPU
    """
    if prefer_small:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_info() -> dict:
    """Return a dict describing available compute."""
    return {
        "mps_available":  torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "cpu_threads":    torch.get_num_threads(),
        "selected":       str(get_device()),
        "torch_version":  torch.__version__,
    }


def to_device_safe(tensor_or_module, device: torch.device):
    """
    Move a tensor/module to device, falling back to CPU if the op is
    unsupported on MPS (some ops still lack MPS kernels in older torch).
    """
    try:
        return tensor_or_module.to(device)
    except Exception:
        return tensor_or_module.to("cpu")


if __name__ == "__main__":
    import json
    print(json.dumps(device_info(), indent=2))
