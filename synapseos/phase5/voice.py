"""
Phase 5 — Voice Interface.

Provides the full voice pipeline interface for SynapseOS:
  INPUT (audio file or stream)
      ↓
  WhisperTranscriber   — speech → text  (GPU stub; real Whisper when A100 available)
      ↓
  CognitiveController  — text → CognitionResult
      ↓
  TTSSynthesizer       — text → speech  (GPU stub; real TTS when A100 available)
      ↓
  OUTPUT (audio file or stream)

The interface is production-ready. The stubs produce correct-shaped outputs
so downstream code can be written and tested now. Swap in the real
`openai-whisper` and a TTS backend (Coqui / pyttsx3 / ElevenLabs) when
GPU is available — only the `_transcribe_impl` and `_synthesize_impl`
methods change.

Success criterion from Phase 5: voice round-trip <2s latency.
Current stub latency: <5ms (no actual model inference).
Production latency target with Whisper tiny + FastSpeech2: ~300–800ms.

Run:
    python -m synapseos.phase5.voice
    python -m synapseos.phase5.voice --text "What fields are required for a Sales Invoice?"
    python -m synapseos.phase5.voice --audio path/to/audio.wav
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TranscriptionResult:
    text: str
    confidence: float          # 0–1, model confidence in transcription
    duration_seconds: float    # audio clip duration
    language: str              # detected language code (e.g. "en")
    segments: list[dict]       # word-level segments if available
    model_used: str            # "whisper-tiny" | "whisper-base" | "stub"
    latency_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SynthesisResult:
    audio_path: str            # path to output audio file
    text: str                  # input text that was synthesised
    duration_seconds: float    # estimated audio duration
    chars: int                 # character count of input
    model_used: str            # "fastspeech2" | "pyttsx3" | "stub"
    latency_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VoiceRoundTrip:
    """Full voice → cognitive → voice result."""
    audio_input:    str | None             # input audio path (None if text-in)
    transcription:  TranscriptionResult
    cognition_output: str                  # CognitionResult.primary_output
    synthesis:      SynthesisResult
    total_latency_ms: float
    within_target: bool                    # <2000ms?
    timestamp: str

    def summary(self) -> str:
        status = "✓ within target" if self.within_target else "✗ exceeds 2s target"
        return (
            f"Voice round-trip [{status}]\n"
            f"  Transcription : {self.transcription.latency_ms:.0f}ms → \"{self.transcription.text[:60]}\"\n"
            f"  Cognition     : {self.total_latency_ms - self.transcription.latency_ms - self.synthesis.latency_ms:.0f}ms\n"
            f"  Synthesis     : {self.synthesis.latency_ms:.0f}ms\n"
            f"  TOTAL         : {self.total_latency_ms:.0f}ms"
        )

    def to_dict(self) -> dict:
        return {
            "audio_input":    self.audio_input,
            "transcription":  self.transcription.to_dict(),
            "cognition":      self.cognition_output[:200],
            "synthesis":      self.synthesis.to_dict(),
            "total_ms":       round(self.total_latency_ms, 1),
            "within_target":  self.within_target,
            "timestamp":      self.timestamp,
        }


# ---------------------------------------------------------------------------
# Whisper Transcriber
# ---------------------------------------------------------------------------

class WhisperTranscriber:
    """
    Speech-to-text interface.

    Production: `pip install openai-whisper` then set `model_size` to
    "tiny", "base", "small", "medium", "large".

    Stub mode (no GPU / no library): reads .txt sidecar files or generates
    a placeholder transcription. All method signatures are production-identical.
    """

    WHISPER_AVAILABLE = False

    def __init__(self, model_size: str = "tiny", device: str = "cpu"):
        self.model_size = model_size
        self.device     = device
        self._model     = None
        self._cpp       = None    # whisper.cpp backend (preferred, real offline STT)
        self._try_load()

    def _try_load(self) -> None:
        # Preferred backend: whisper.cpp (real, offline, Metal-accelerated on Apple Silicon)
        try:
            from synapseos.local.whisper_cpp import WhisperCppSTT, whisper_cpp_available
            if whisper_cpp_available():
                self._cpp = WhisperCppSTT()
                WhisperTranscriber.WHISPER_AVAILABLE = True
                return
        except Exception as e:
            print(f"[Whisper] whisper.cpp probe failed ({e})")

        # Fallback backend: openai-whisper (Python)
        try:
            import whisper  # type: ignore
            self._model = whisper.load_model(self.model_size, device=self.device)
            WhisperTranscriber.WHISPER_AVAILABLE = True
            print(f"[Whisper] Loaded openai-whisper {self.model_size} on {self.device}")
        except ImportError:
            print(f"[Whisper] no whisper.cpp / openai-whisper — running in stub mode")
        except Exception as e:
            print(f"[Whisper] Load failed ({e}) — running in stub mode")

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcribe an audio file to text."""
        t0 = time.time()

        if self._cpp is not None:
            return self._transcribe_cpp(audio_path, t0)
        if self._model is not None:
            return self._transcribe_real(audio_path, t0)
        return self._transcribe_stub(audio_path, t0)

    def _transcribe_cpp(self, audio_path: str, t0: float) -> TranscriptionResult:
        r = self._cpp.transcribe(audio_path)
        return TranscriptionResult(
            text=r.text,
            confidence=r.confidence,
            duration_seconds=r.duration_seconds,
            language=r.language,
            segments=[],
            model_used=r.model_used,
            latency_ms=(time.time() - t0) * 1000,
        )

    def _transcribe_real(self, audio_path: str, t0: float) -> TranscriptionResult:
        result = self._model.transcribe(audio_path)
        return TranscriptionResult(
            text=result["text"].strip(),
            confidence=0.95,
            duration_seconds=result.get("duration", 0.0),
            language=result.get("language", "en"),
            segments=result.get("segments", []),
            model_used=f"whisper-{self.model_size}",
            latency_ms=(time.time() - t0) * 1000,
        )

    def _transcribe_stub(self, audio_path: str, t0: float) -> TranscriptionResult:
        """
        Stub: if a .txt sidecar exists alongside the audio file, use it.
        Otherwise return a placeholder that signals stub mode.
        """
        p = Path(audio_path)
        sidecar = p.with_suffix(".txt")
        if sidecar.exists():
            text = sidecar.read_text().strip()
            duration = len(text) / 15.0   # rough: ~15 chars/second speech
        else:
            text = f"[Stub transcription — no Whisper model loaded. Audio: {p.name}]"
            duration = 0.0

        return TranscriptionResult(
            text=text,
            confidence=1.0 if sidecar.exists() else 0.0,
            duration_seconds=duration,
            language="en",
            segments=[],
            model_used="stub",
            latency_ms=(time.time() - t0) * 1000,
        )

    def transcribe_text(self, text: str) -> TranscriptionResult:
        """
        Convenience: treat a text string as already-transcribed audio.
        Useful for testing the voice pipeline without an audio file.
        """
        return TranscriptionResult(
            text=text,
            confidence=1.0,
            duration_seconds=len(text) / 15.0,
            language="en",
            segments=[],
            model_used="text_passthrough",
            latency_ms=0.0,
        )

    @property
    def is_real(self) -> bool:
        return self._cpp is not None or self._model is not None


# ---------------------------------------------------------------------------
# TTS Synthesizer
# ---------------------------------------------------------------------------

class TTSSynthesizer:
    """
    Text-to-speech interface.

    Production options (swap in _synthesize_impl):
    - `pip install pyttsx3` for CPU TTS (quality: OK)
    - Coqui TTS / FastSpeech2 for high-quality neural TTS (GPU)
    - ElevenLabs API for production-grade voice

    Stub mode: writes text to a .txt file and estimates duration.
    All method signatures are production-identical.
    """

    TTS_AVAILABLE = False

    def __init__(
        self,
        output_dir: str = "data/voice/output",
        voice_id: str = "default",
        speed: float = 1.0,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.voice_id = voice_id
        self.speed    = speed
        self._engine  = None
        self._try_load()

    def _try_load(self) -> None:
        try:
            import pyttsx3   # type: ignore
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", int(175 * self.speed))
            TTSSynthesizer.TTS_AVAILABLE = True
            print("[TTS] pyttsx3 loaded — CPU TTS available")
        except ImportError:
            print("[TTS] pyttsx3 not installed — running in stub mode")
        except Exception as e:
            print(f"[TTS] Load failed ({e}) — running in stub mode")

    def synthesize(self, text: str, filename: str | None = None) -> SynthesisResult:
        """Convert text to speech, save to file, return result."""
        t0 = time.time()
        if filename is None:
            digest   = hashlib.md5(text.encode()).hexdigest()[:8]
            filename = f"synth_{digest}.wav"

        output_path = str(self.output_dir / filename)

        if self._engine is not None:
            return self._synthesize_real(text, output_path, t0)
        else:
            return self._synthesize_stub(text, output_path, t0)

    def _synthesize_real(self, text: str, output_path: str, t0: float) -> SynthesisResult:
        self._engine.save_to_file(text, output_path)
        self._engine.runAndWait()
        words = len(text.split())
        duration = words / (175 / 60 * self.speed)   # words per minute → seconds
        return SynthesisResult(
            audio_path=output_path,
            text=text,
            duration_seconds=round(duration, 2),
            chars=len(text),
            model_used="pyttsx3",
            latency_ms=(time.time() - t0) * 1000,
        )

    def _synthesize_stub(self, text: str, output_path: str, t0: float) -> SynthesisResult:
        # Write .txt stub in place of audio
        txt_path = output_path.replace(".wav", ".txt")
        Path(txt_path).write_text(text)
        words    = len(text.split())
        duration = words / (175 / 60)
        return SynthesisResult(
            audio_path=txt_path,
            text=text,
            duration_seconds=round(duration, 2),
            chars=len(text),
            model_used="stub",
            latency_ms=(time.time() - t0) * 1000,
        )

    @property
    def is_real(self) -> bool:
        return self._engine is not None


# ---------------------------------------------------------------------------
# Voice Activity Detection stub
# ---------------------------------------------------------------------------

class VoiceActivityDetector:
    """
    Detects speech vs background noise in an audio stream.
    Production: use WebRTC VAD or SileroVAD.
    Stub: always returns True (speech detected).
    """

    def __init__(self):
        self._real = False
        try:
            import webrtcvad   # type: ignore
            self._vad = webrtcvad.Vad(2)
            self._real = True
        except ImportError:
            pass

    def is_speech(self, audio_frame: bytes, sample_rate: int = 16000) -> bool:
        if self._real:
            return self._vad.is_speech(audio_frame, sample_rate)
        return True   # stub: always speech

    @property
    def is_real(self) -> bool:
        return self._real


# ---------------------------------------------------------------------------
# Voice Pipeline
# ---------------------------------------------------------------------------

class VoicePipeline:
    """
    Full voice round-trip: audio → transcribe → think → synthesize.

    Integrates WhisperTranscriber, CognitiveController, and TTSSynthesizer.
    Target latency: <2 seconds end-to-end (Phase 5 success criterion).

    Usage:
        pipeline = VoicePipeline()

        # From audio file
        result = pipeline.process_audio("query.wav")

        # From text (bypasses transcription, useful for testing)
        result = pipeline.process_text("What fields does Sales Invoice need?")

        print(result.summary())
    """

    LATENCY_TARGET_MS = 2000.0

    def __init__(
        self,
        kg_path: str = "data/graph/knowledge_graph.json",
        rules_path: str = "data/rules/rules.json",
        whisper_model: str = "tiny",
        tts_output_dir: str = "data/voice/output",
        enable_agents: bool = True,
        prefer_macos_tts: bool = True,
    ):
        self.transcriber = WhisperTranscriber(model_size=whisper_model)
        # Prefer real macOS `say` TTS when available (produces real audio, no GPU);
        # fall back to the pyttsx3/stub TTSSynthesizer otherwise.
        self.synthesizer = self._select_tts(tts_output_dir, prefer_macos_tts)
        self.vad         = VoiceActivityDetector()
        self._controller = None
        self._kg_path    = kg_path
        self._rules_path = rules_path
        self._enable_agents = enable_agents
        self._session_log: list[VoiceRoundTrip] = []

    def _select_tts(self, output_dir: str, prefer_macos: bool):
        """Use real macOS `say` TTS if available, else the stub/pyttsx3 synthesizer."""
        if prefer_macos:
            try:
                from synapseos.local.tts_macos import MacTTS, say_available
                if say_available():
                    return MacTTS(output_dir=output_dir)
            except Exception:
                pass
        return TTSSynthesizer(output_dir=output_dir)

    def _get_controller(self):
        if self._controller is None:
            from synapseos.phase5.cognitive_controller import CognitiveController
            self._controller = CognitiveController(
                kg_path=self._kg_path,
                rules_path=self._rules_path,
                enable_agents=self._enable_agents,
            )
        return self._controller

    def process_audio(self, audio_path: str) -> VoiceRoundTrip:
        """Full pipeline: audio file → transcribe → think → speak."""
        t0 = time.time()
        transcription = self.transcriber.transcribe(audio_path)
        result        = self._run_cognition(transcription, audio_path, t0)
        return result

    def process_text(self, text: str) -> VoiceRoundTrip:
        """Bypass transcription: text → think → speak. Used for testing."""
        t0 = time.time()
        transcription = self.transcriber.transcribe_text(text)
        result        = self._run_cognition(transcription, None, t0)
        return result

    def _run_cognition(
        self,
        transcription: TranscriptionResult,
        audio_input: str | None,
        t0: float,
    ) -> VoiceRoundTrip:
        controller = self._get_controller()
        cognition  = controller.think(transcription.text)
        spoken_text = self._trim_for_speech(cognition.primary_output)
        synthesis  = self.synthesizer.synthesize(spoken_text)
        total_ms   = (time.time() - t0) * 1000

        trip = VoiceRoundTrip(
            audio_input=audio_input,
            transcription=transcription,
            cognition_output=cognition.primary_output,
            synthesis=synthesis,
            total_latency_ms=total_ms,
            within_target=total_ms <= self.LATENCY_TARGET_MS,
            timestamp=datetime.now().isoformat(),
        )
        self._session_log.append(trip)
        return trip

    @staticmethod
    def _trim_for_speech(text: str, max_chars: int = 500) -> str:
        """
        Strip markdown, code blocks, and memory context annotations
        to produce clean speech-friendly text.
        """
        import re
        # Remove code blocks
        text = re.sub(r"```[^`]*```", "[code omitted]", text, flags=re.DOTALL)
        # Remove memory context annotations
        text = re.sub(r"\[Memory context:.*?\]", "", text)
        # Remove markdown links
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + " …"
        return text

    def session_stats(self) -> dict:
        if not self._session_log:
            return {"sessions": 0}
        latencies = [r.total_latency_ms for r in self._session_log]
        within    = sum(1 for r in self._session_log if r.within_target)
        return {
            "sessions":        len(self._session_log),
            "within_target":   within,
            "target_rate":     round(within / len(self._session_log), 3),
            "mean_latency_ms": round(sum(latencies) / len(latencies), 1),
            "max_latency_ms":  round(max(latencies), 1),
            "whisper_real":    self.transcriber.is_real,
            "tts_real":        self.synthesizer.is_real,
        }

    def shutdown(self) -> None:
        if self._controller:
            self._controller.shutdown()


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SynapseOS Voice Interface")
    parser.add_argument("--text",  default=None, help="Text input (bypasses transcription)")
    parser.add_argument("--audio", default=None, help="Path to .wav audio file")
    parser.add_argument("--whisper-model", default="tiny", choices=["tiny","base","small","medium","large"])
    args = parser.parse_args()

    pipeline = VoicePipeline(whisper_model=args.whisper_model)

    print(f"\nSynapseOS Voice Interface")
    print(f"Whisper: {'real (' + args.whisper_model + ')' if pipeline.transcriber.is_real else 'stub mode'}")
    print(f"TTS    : {'pyttsx3' if pipeline.synthesizer.is_real else 'stub mode'}")
    print(f"{'='*60}\n")

    if args.text:
        result = pipeline.process_text(args.text)
    elif args.audio:
        result = pipeline.process_audio(args.audio)
    else:
        print("No input provided. Running demo with synthetic text query.")
        result = pipeline.process_text("What fields are required for a Sales Invoice?")

    print(result.summary())
    print(f"\nCognition output:\n{result.cognition_output[:300]}")
    print(f"\nSynthesized to: {result.synthesis.audio_path}")
    print(f"\nSession stats: {pipeline.session_stats()}")

    pipeline.shutdown()


if __name__ == "__main__":
    main()
