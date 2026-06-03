"""
macOS TTS — real speech synthesis via the built-in `say` command.

Replaces `pyttsx3` (and the GPU-blocked neural TTS) with the native macOS
`say` binary. This produces REAL audio files (.aiff / .wav) on this machine
with zero installs and no GPU — it is not a stub.

`say` ships on every macOS install at /usr/bin/say. It supports:
  - dozens of voices (`say -v ?`)
  - adjustable rate (words per minute)
  - direct file output (`say -o out.aiff "text"`)

This wrapper exposes the same interface shape as the stub TTSSynthesizer in
synapseos/phase5/voice.py, so the VoicePipeline can use real audio output.

Run:
    python -m synapseos.local.tts_macos                          # synth a demo line
    python -m synapseos.local.tts_macos --text "Sales Invoice"   # custom text
    python -m synapseos.local.tts_macos --list-voices
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def say_available() -> bool:
    """True if the macOS `say` binary is present."""
    return platform.system() == "Darwin" and shutil.which("say") is not None


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class TTSResult:
    audio_path: str
    text: str
    duration_seconds: float
    chars: int
    voice: str
    rate_wpm: int
    model_used: str       # "macos_say" | "stub"
    latency_ms: float
    is_real_audio: bool

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


# ---------------------------------------------------------------------------
# macOS TTS
# ---------------------------------------------------------------------------

class MacTTS:
    """
    Real text-to-speech using the macOS `say` command.

    Falls back to a .txt stub on non-macOS systems so callers never crash.
    """

    def __init__(
        self,
        output_dir: str = "data/voice/output",
        voice: str = "Samantha",     # clear US English voice present on macOS
        rate_wpm: int = 175,
        to_wav: bool = True,         # convert .aiff → .wav via afconvert if available
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.voice    = voice
        self.rate_wpm = rate_wpm
        self.to_wav   = to_wav and shutil.which("afconvert") is not None
        self._real    = say_available()
        if self._real:
            print(f"[MacTTS] Using macOS `say` (voice={voice}, rate={rate_wpm}wpm)")
        else:
            print("[MacTTS] `say` not available — stub mode (.txt output)")

    @property
    def is_real(self) -> bool:
        return self._real

    def list_voices(self) -> list[str]:
        if not self._real:
            return []
        out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True)
        voices = []
        for line in out.stdout.splitlines():
            # format: "Samantha           en_US    # comment"
            name = line.split()[0] if line.split() else ""
            if name:
                voices.append(name)
        return voices

    def synthesize(self, text: str, filename: str | None = None) -> TTSResult:
        t0 = time.time()
        if filename is None:
            digest   = hashlib.md5(text.encode()).hexdigest()[:8]
            filename = f"say_{digest}.aiff"

        out_path = self.output_dir / filename

        if not self._real:
            return self._stub(text, out_path, t0)

        # Real synthesis via `say -o`
        try:
            cmd = ["say", "-v", self.voice, "-r", str(self.rate_wpm),
                   "-o", str(out_path), text]
            subprocess.run(cmd, check=True, capture_output=True, timeout=60)

            final_path = str(out_path)
            if self.to_wav:
                wav_path = str(out_path.with_suffix(".wav"))
                conv = subprocess.run(
                    ["afconvert", "-f", "WAVE", "-d", "LEI16", str(out_path), wav_path],
                    capture_output=True, timeout=30,
                )
                if conv.returncode == 0 and Path(wav_path).exists():
                    final_path = wav_path

            words    = len(text.split())
            duration = words / (self.rate_wpm / 60.0)

            return TTSResult(
                audio_path=final_path,
                text=text,
                duration_seconds=round(duration, 2),
                chars=len(text),
                voice=self.voice,
                rate_wpm=self.rate_wpm,
                model_used="macos_say",
                latency_ms=(time.time() - t0) * 1000,
                is_real_audio=True,
            )
        except Exception as e:
            print(f"[MacTTS] say failed ({e}) — falling back to stub")
            return self._stub(text, out_path, t0)

    def speak(self, text: str) -> bool:
        """Speak text aloud immediately through the speakers (no file)."""
        if not self._real:
            return False
        try:
            subprocess.run(["say", "-v", self.voice, "-r", str(self.rate_wpm), text],
                           check=True, timeout=60)
            return True
        except Exception:
            return False

    def _stub(self, text: str, out_path: Path, t0: float) -> TTSResult:
        txt_path = out_path.with_suffix(".txt")
        txt_path.write_text(text)
        words = len(text.split())
        return TTSResult(
            audio_path=str(txt_path),
            text=text,
            duration_seconds=round(words / (self.rate_wpm / 60.0), 2),
            chars=len(text),
            voice="none",
            rate_wpm=self.rate_wpm,
            model_used="stub",
            latency_ms=(time.time() - t0) * 1000,
            is_real_audio=False,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="macOS TTS (real `say` audio)")
    parser.add_argument("--text", default="SynapseOS voice output is now real audio on this machine.")
    parser.add_argument("--voice", default="Samantha")
    parser.add_argument("--rate",  type=int, default=175)
    parser.add_argument("--speak", action="store_true", help="Speak aloud instead of saving")
    parser.add_argument("--list-voices", action="store_true")
    args = parser.parse_args()

    tts = MacTTS(voice=args.voice, rate_wpm=args.rate)

    if args.list_voices:
        voices = tts.list_voices()
        print(f"Available voices ({len(voices)}):")
        for v in voices[:30]:
            print(f"  {v}")
        return

    if args.speak:
        ok = tts.speak(args.text)
        print(f"Spoke aloud: {ok}")
        return

    result = tts.synthesize(args.text)
    print(f"\nText      : {result.text}")
    print(f"Audio     : {result.audio_path}")
    print(f"Real audio: {result.is_real_audio}")
    print(f"Voice     : {result.voice}")
    print(f"Duration  : {result.duration_seconds}s")
    print(f"Latency   : {result.latency_ms:.0f}ms")
    if result.is_real_audio and Path(result.audio_path).exists():
        size_kb = Path(result.audio_path).stat().st_size // 1024
        print(f"File size : {size_kb}KB  (real waveform)")


if __name__ == "__main__":
    main()
