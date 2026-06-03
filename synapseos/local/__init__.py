"""
SynapseOS Local — install-free, CPU/MPS alternatives to GPU-blocked components.

This package replaces the CUDA/library dependencies in the GPU blocker list
with pure-PyTorch and native-macOS implementations that run on this machine:

  mamba-ssm (CUDA)  → mamba_cpu.MinimalMamba   (pure-PyTorch selective scan, MPS)
  ncps              → liquid_cfc.CfCCell       (pure-PyTorch closed-form continuous cell)
  trl               → sft_trainer.SFTTrainer   (pure-PyTorch supervised fine-tuning loop)
  pyttsx3 / TTS     → tts_macos.MacTTS         (macOS `say` command, real audio)

Everything here runs on Apple Silicon (MPS) or CPU with no pip installs.
"""

from synapseos.local.device import get_device, device_info

__all__ = ["get_device", "device_info"]
