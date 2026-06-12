"""
Identity — credential store. One secure surface for all secrets.

Precedence: environment variable > credentials file > None. The file lives at
data/identity/credentials.json and should be gitignored. Values are never
printed in full — `describe()` masks them.

Conventional keys (set whichever accounts you give the companion):
  GMAIL_ADDRESS, GMAIL_APP_PASSWORD          (email send/read)
  SMTP_HOST, SMTP_PORT, IMAP_HOST            (optional overrides)
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
  GIT_USER_NAME, GIT_USER_EMAIL, GIT_TOKEN   (commits/pushes as the companion)
  YOUTUBE_API_KEY, YOUTUBE_CHANNEL_ID
  X_API_KEY / INSTAGRAM_TOKEN / ...          (social, as you provide them)
"""

from __future__ import annotations

import json
import os
from pathlib import Path


class CredentialStore:
    def __init__(self, path: str = "data/identity/credentials.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._file = json.loads(self.path.read_text())
            except Exception:
                self._file = {}

    def get(self, key: str) -> str | None:
        return os.environ.get(key) or self._file.get(key)

    def set(self, key: str, value: str) -> None:
        self._file[key] = value
        self.path.write_text(json.dumps(self._file, indent=2))
        try:
            os.chmod(self.path, 0o600)
        except Exception:
            pass

    def has(self, key: str) -> bool:
        return self.get(key) is not None

    def as_policy_dict(self) -> dict[str, str]:
        """Snapshot for PermissionPolicy.credentials (env still wins at lookup)."""
        return dict(self._file)

    def describe(self) -> dict[str, str]:
        """Masked view — which credentials exist, never their values."""
        out = {}
        keys = set(self._file) | {k for k in os.environ
                                  if k in _KNOWN_KEYS and os.environ.get(k)}
        for k in sorted(keys):
            v = self.get(k) or ""
            out[k] = (v[:2] + "…" + v[-2:]) if len(v) > 6 else "set"
        return out


_KNOWN_KEYS = {
    "GMAIL_ADDRESS", "GMAIL_APP_PASSWORD", "SMTP_HOST", "SMTP_PORT", "SMTP_USER",
    "SMTP_PASSWORD", "IMAP_HOST", "IMAP_USER", "IMAP_PASSWORD",
    "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "GIT_USER_NAME", "GIT_USER_EMAIL",
    "GIT_TOKEN", "YOUTUBE_API_KEY", "YOUTUBE_CHANNEL_ID", "X_API_KEY",
    "INSTAGRAM_TOKEN", "BROWSER_APP", "SLACK_WEBHOOK", "DISCORD_WEBHOOK",
    "DISCORD_BOT_TOKEN",
}
