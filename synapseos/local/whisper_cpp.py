"""
whisper.cpp STT — real offline speech-to-text via the whisper.cpp `whisper-cli`.

This is the missing perception input: Human Speech → text. Unlike the
trainable-from-scratch components (Mamba/CfC/SFT), ASR needs a pretrained
acoustic model, so this wraps the whisper.cpp binary + a GGML model file.

On this machine:
  binary : /opt/homebrew/bin/whisper-cli   (brew install whisper-cpp)
  model  : data/models/whisper/ggml-base.en.bin  (Metal-accelerated on Apple Silicon)

It is fully offline, local, no API — fits the SynapseOS philosophy. Falls back
to the text-passthrough stub when the binary or model is absent, so callers
never crash.

whisper.cpp needs 16 kHz mono WAV; this auto-converts other formats/rates via
the macOS `afconvert` (or ffmpeg if present).

Run:
    python -m synapseos.local.whisper_cpp --selftest   # say→wav→transcribe round-trip
    python -m synapseos.local.whisper_cpp --audio path/to/audio.wav
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATHS = [
    "data/models/whisper/ggml-base.en.bin",
    "data/models/whisper/ggml-tiny.en.bin",
    "data/models/whisper/ggml-small.en.bin",
    os.path.expanduser("~/.cache/whisper.cpp/ggml-base.en.bin"),
]

BINARY_CANDIDATES = ["whisper-cli", "whisper-cpp", "main"]


def find_binary() -> str | None:
    for name in BINARY_CANDIDATES:
        path = shutil.which(name)
        if path:
            return path
    # Homebrew cellar fallback
    for cellar in Path("/opt/homebrew/Cellar/whisper-cpp").glob("*/bin/whisper-cli"):
        return str(cellar)
    return None


def find_model(explicit: str | None = None) -> str | None:
    if explicit and Path(explicit).exists():
        return explicit
    for p in DEFAULT_MODEL_PATHS:
        if Path(p).exists():
            return p
    return None


def whisper_cpp_available(model: str | None = None) -> bool:
    return find_binary() is not None and find_model(model) is not None


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class STTResult:
    text: str
    confidence: float
    duration_seconds: float
    language: str
    model_used: str          # "whisper.cpp:base.en" | "stub"
    latency_ms: float
    is_real: bool

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


# ---------------------------------------------------------------------------
# Transcriber
# ---------------------------------------------------------------------------

class WhisperCppSTT:
    """
    Real speech-to-text using whisper.cpp.

    Same surface as the stub WhisperTranscriber in phase5/voice.py:
      .transcribe(audio_path) -> STTResult
      .transcribe_text(text)  -> STTResult   (passthrough, for testing)
    """

    def __init__(
        self,
        model_path: str | None = None,
        binary: str | None = None,
        threads: int = 8,
        language: str = "en",
    ):
        self.binary    = binary or find_binary()
        self.model     = find_model(model_path)
        self.threads   = threads
        self.language  = language
        self._real     = self.binary is not None and self.model is not None
        if self._real:
            model_name = Path(self.model).stem.replace("ggml-", "")
            self.model_tag = f"whisper.cpp:{model_name}"
            print(f"[whisper.cpp] {self.binary} | model={Path(self.model).name}")
        else:
            self.model_tag = "stub"
            reason = "binary" if not self.binary else "model"
            print(f"[whisper.cpp] not available (missing {reason}) — stub mode")

    @property
    def is_real(self) -> bool:
        return self._real

    # ------------------------------------------------------------------
    # Audio prep
    # ------------------------------------------------------------------

    def _ensure_wav_16k(self, audio_path: str) -> tuple[str, bool]:
        """
        whisper.cpp wants 16 kHz mono 16-bit WAV. Convert if needed.
        Returns (path, is_temp). is_temp=True means caller should delete it.
        """
        afconvert = shutil.which("afconvert")
        ffmpeg    = shutil.which("ffmpeg")
        tmp = tempfile.NamedTemporaryFile(suffix="_16k.wav", delete=False)
        tmp.close()

        if afconvert:
            r = subprocess.run(
                ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                 audio_path, tmp.name],
                capture_output=True, timeout=60,
            )
            if r.returncode == 0 and Path(tmp.name).stat().st_size > 0:
                return tmp.name, True
        if ffmpeg:
            r = subprocess.run(
                ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1",
                 "-f", "wav", tmp.name],
                capture_output=True, timeout=60,
            )
            if r.returncode == 0 and Path(tmp.name).stat().st_size > 0:
                return tmp.name, True

        # No converter — try the file as-is (works if already 16k wav)
        os.unlink(tmp.name)
        return audio_path, False

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def transcribe(self, audio_path: str) -> STTResult:
        t0 = time.time()
        if not self._real:
            return self._stub(audio_path, t0)
        # Missing or empty audio → fall back to sidecar/placeholder
        if not Path(audio_path).exists() or Path(audio_path).stat().st_size < 1024:
            return self._stub(audio_path, t0)

        wav_path, is_temp = self._ensure_wav_16k(audio_path)
        try:
            cmd = [
                self.binary, "-m", self.model, "-f", wav_path,
                "-t", str(self.threads), "-l", self.language,
                "-nt", "-np",          # no timestamps, no prints
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            raw = proc.stdout.strip()
            text = self._clean(raw)

            # If transcription produced nothing, fall back to a sidecar if present
            if not text:
                fallback = self._stub(audio_path, t0)
                if fallback.text and not fallback.text.startswith("[stub"):
                    return fallback

            return STTResult(
                text=text,
                confidence=0.9 if text else 0.0,
                duration_seconds=self._wav_duration(wav_path),
                language=self.language,
                model_used=self.model_tag,
                latency_ms=(time.time() - t0) * 1000,
                is_real=True,
            )
        except Exception as e:
            print(f"[whisper.cpp] transcription failed: {e}")
            return self._stub(audio_path, t0)
        finally:
            if is_temp and Path(wav_path).exists():
                os.unlink(wav_path)

    def transcribe_text(self, text: str) -> STTResult:
        """Passthrough for pipeline testing without audio."""
        return STTResult(text, 1.0, len(text) / 15.0, self.language,
                         "text_passthrough", 0.0, False)

    @staticmethod
    def _clean(raw: str) -> str:
        # whisper.cpp -nt output is plain text lines; strip stray bracket noise
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        text  = " ".join(lines)
        text  = re.sub(r"\[[^\]]*\]", "", text)   # remove any [..] markers
        text  = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _wav_duration(wav_path: str) -> float:
        try:
            import wave
            with wave.open(wav_path, "rb") as w:
                return round(w.getnframes() / float(w.getframerate()), 2)
        except Exception:
            return 0.0

    def _stub(self, audio_path: str, t0: float) -> STTResult:
        # try a .txt sidecar (lets pipeline tests run without a model)
        sidecar = Path(audio_path).with_suffix(".txt")
        if sidecar.exists():
            text = sidecar.read_text().strip()
            return STTResult(text, 1.0, len(text)/15.0, self.language,
                             "stub_sidecar", (time.time()-t0)*1000, False)
        return STTResult(
            f"[stub transcription — whisper.cpp unavailable: {Path(audio_path).name}]",
            0.0, 0.0, self.language, "stub", (time.time()-t0)*1000, False,
        )


# ---------------------------------------------------------------------------
# Self-test: say → wav → transcribe round-trip
# ---------------------------------------------------------------------------

def _selftest() -> None:
    """Generate speech with macOS `say`, transcribe it back, check the text matches."""
    from synapseos.local.tts_macos import MacTTS, say_available

    stt = WhisperCppSTT()
    if not stt.is_real:
        print("[selftest] whisper.cpp not available — skipping round-trip")
        return
    if not say_available():
        print("[selftest] macOS `say` not available — cannot generate test audio")
        return

    phrase = "What fields are required for a sales invoice"
    print(f"[selftest] phrase: \"{phrase}\"")

    tts = MacTTS(output_dir="/tmp/whisper_selftest")
    synth = tts.synthesize(phrase, filename="selftest.aiff")
    print(f"[selftest] generated audio: {synth.audio_path} ({synth.duration_seconds}s)")

    t0 = time.time()
    result = stt.transcribe(synth.audio_path)
    print(f"[selftest] transcribed : \"{result.text}\"")
    print(f"[selftest] model={result.model_used} latency={result.latency_ms:.0f}ms real={result.is_real}")

    # word-overlap score
    want = set(re.findall(r"\w+", phrase.lower()))
    got  = set(re.findall(r"\w+", result.text.lower()))
    overlap = len(want & got) / len(want) if want else 0.0
    print(f"[selftest] word overlap: {overlap:.0%}")
    if overlap >= 0.6:
        print("[selftest] PASSED — real STT round-trip works on this machine")
    else:
        print("[selftest] LOW OVERLAP — transcription differs (check audio/model)")


def main():
    parser = argparse.ArgumentParser(description="whisper.cpp STT")
    parser.add_argument("--audio",    default=None)
    parser.add_argument("--model",    default=None)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest or not args.audio:
        _selftest()
        return

    stt = WhisperCppSTT(model_path=args.model)
    result = stt.transcribe(args.audio)
    print(f"\nText      : {result.text}")
    print(f"Model     : {result.model_used}")
    print(f"Real      : {result.is_real}")
    print(f"Duration  : {result.duration_seconds}s")
    print(f"Latency   : {result.latency_ms:.0f}ms")


if __name__ == "__main__":
    main()
