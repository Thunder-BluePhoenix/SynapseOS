"""
Web Learning Loop — Media ingestion (audio, video, YouTube).

Turns non-text sources into text so they flow through the same fact pipeline
as web pages:

  YouTube URL ──> captions via yt-dlp (no audio download)        [fast path]
             ──> yt-dlp audio download → whisper.cpp STT         [fallback]
  audio/video URL (.mp3/.wav/.m4a/.mp4/...) ──> download → whisper.cpp

Dependencies (all optional, graceful degradation):
  - yt-dlp        (installed via pip/brew) — YouTube access
  - whisper.cpp   (synapseos.local.whisper_cpp wrapper) — transcription
  - ffmpeg        (used by the whisper wrapper for format conversion)

FakeMediaIngester is the zero-network/zero-binary test double.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".opus"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}
YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be",
                 "music.youtube.com"}

_SRT_TIMESTAMP = re.compile(r"^\d{2}:\d{2}:\d{2}[.,]\d{3}\s+-->")
_VTT_TAG = re.compile(r"<[^>]+>")


@dataclass
class MediaResult:
    url: str
    kind: str               # youtube | audio | video
    text: str = ""
    title: str = ""
    method: str = ""        # captions | whisper | fake
    duration_s: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.text) and not self.error

    def to_dict(self) -> dict:
        return asdict(self)


def classify_media(url: str) -> str | None:
    """Return media kind for a URL, or None if it's a regular page."""
    parts = urllib.parse.urlsplit(url)
    host = parts.netloc.split(":")[0].lower()
    if host in YOUTUBE_HOSTS:
        return "youtube"
    ext = Path(parts.path).suffix.lower()
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    return None


def _parse_subtitles(raw: str) -> str:
    """SRT/VTT → plain text. Drops timestamps, indices, tags; dedups repeats."""
    lines: list[str] = []
    for line in raw.splitlines():
        line = _VTT_TAG.sub("", line).strip()
        if not line or line.isdigit() or line == "WEBVTT":
            continue
        if _SRT_TIMESTAMP.match(line) or "-->" in line:
            continue
        if line.startswith(("Kind:", "Language:", "NOTE")):
            continue
        # Auto-captions repeat lines heavily — keep only new content
        if lines and (line == lines[-1] or line in lines[-1]):
            continue
        lines.append(line)
    return " ".join(lines)


class MediaIngester:
    """Real media → text. yt-dlp for YouTube, whisper.cpp for audio."""

    def __init__(
        self,
        stt=None,                      # WhisperCppSTT-like (.transcribe(path).text)
        yt_dlp: str | None = None,
        max_download_bytes: int = 60_000_000,
        max_duration_s: int = 1800,    # skip media longer than 30 min
        timeout: float = 120.0,
    ):
        self._stt = stt
        self.yt_dlp = yt_dlp or shutil.which("yt-dlp")
        self.max_download_bytes = max_download_bytes
        self.max_duration_s = max_duration_s
        self.timeout = timeout

    @property
    def stt(self):
        # Lazy: whisper model load is heavy; only pay for it when audio shows up
        if self._stt is None:
            from synapseos.local.whisper_cpp import WhisperCppSTT
            self._stt = WhisperCppSTT()
        return self._stt

    def classify(self, url: str) -> str | None:
        return classify_media(url)

    # -- public entry ----------------------------------------------------------

    def ingest(self, url: str) -> MediaResult:
        kind = self.classify(url)
        if kind is None:
            return MediaResult(url=url, kind="", error="not a media URL")
        if kind == "youtube":
            return self._ingest_youtube(url)
        return self._ingest_audio_url(url, kind)

    # -- YouTube -----------------------------------------------------------------

    def _ingest_youtube(self, url: str) -> MediaResult:
        result = MediaResult(url=url, kind="youtube")
        if self.yt_dlp is None:
            result.error = "yt-dlp not installed (pip install yt-dlp)"
            return result

        result.title = self._yt_title(url)

        # Fast path: captions only, no audio download
        text = self._yt_captions(url)
        if text:
            result.text, result.method = text, "captions"
            return result

        # Fallback: download audio, transcribe with whisper.cpp
        text, duration, err = self._yt_audio_whisper(url)
        if text:
            result.text, result.method, result.duration_s = text, "whisper", duration
        else:
            result.error = err or "no captions and transcription failed"
        return result

    def _yt_title(self, url: str) -> str:
        try:
            proc = subprocess.run(
                [self.yt_dlp, "--no-download", "--print", "%(title)s", url],
                capture_output=True, text=True, timeout=30,
            )
            return proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
        except Exception:
            return ""

    def _yt_captions(self, url: str) -> str:
        workdir = tempfile.mkdtemp(prefix="yt_subs_")
        try:
            subprocess.run(
                [self.yt_dlp, "--skip-download", "--write-subs", "--write-auto-subs",
                 "--sub-langs", "en.*", "--convert-subs", "srt",
                 "-o", f"{workdir}/subs", url],
                capture_output=True, timeout=self.timeout,
            )
            for sub in sorted(Path(workdir).glob("subs*.srt")):
                text = _parse_subtitles(sub.read_text(errors="replace"))
                if len(text) > 100:
                    return text
            return ""
        except Exception:
            return ""
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def _yt_audio_whisper(self, url: str) -> tuple[str, float, str]:
        workdir = tempfile.mkdtemp(prefix="yt_audio_")
        try:
            proc = subprocess.run(
                [self.yt_dlp, "-x", "--audio-format", "mp3",
                 "--max-filesize", str(self.max_download_bytes),
                 "--match-filter", f"duration < {self.max_duration_s}",
                 "-o", f"{workdir}/audio.%(ext)s", url],
                capture_output=True, text=True, timeout=600,
            )
            files = list(Path(workdir).glob("audio.*"))
            if not files:
                detail = (proc.stderr or "").strip().splitlines()
                return "", 0.0, f"audio download failed: {detail[-1] if detail else 'unknown'}"
            stt_result = self.stt.transcribe(str(files[0]))
            if not getattr(self.stt, "is_real", True):
                return "", 0.0, "whisper.cpp not available for transcription"
            return stt_result.text, stt_result.duration_seconds, ""
        except Exception as e:
            return "", 0.0, str(e)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    # -- direct audio/video URLs -----------------------------------------------------

    def _ingest_audio_url(self, url: str, kind: str) -> MediaResult:
        result = MediaResult(url=url, kind=kind)
        suffix = Path(urllib.parse.urlsplit(url).path).suffix or ".mp3"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "SynapseOS-LearningLoop/0.1"}, method="GET")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = resp.read(self.max_download_bytes)
            tmp.write(data)
            tmp.close()
            stt_result = self.stt.transcribe(tmp.name)
            if not getattr(self.stt, "is_real", True):
                result.error = "whisper.cpp not available for transcription"
                return result
            result.text = stt_result.text
            result.method = "whisper"
            result.duration_s = stt_result.duration_seconds
            if not result.text:
                result.error = "transcription produced no text"
            return result
        except Exception as e:
            result.error = str(e)
            return result
        finally:
            Path(tmp.name).unlink(missing_ok=True)


class FakeMediaIngester:
    """Test double: url → canned MediaResult fields. Records calls."""

    def __init__(self, transcripts: dict[str, str] | None = None):
        self.transcripts = transcripts or {}
        self.calls: list[str] = []

    def classify(self, url: str) -> str | None:
        return classify_media(url)

    def ingest(self, url: str) -> MediaResult:
        self.calls.append(url)
        kind = self.classify(url) or "audio"
        if url in self.transcripts:
            return MediaResult(url=url, kind=kind, text=self.transcripts[url],
                               title="fixture", method="fake")
        return MediaResult(url=url, kind=kind, error="not in fixture set")
