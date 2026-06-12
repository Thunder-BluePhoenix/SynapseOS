"""
Gateway — the message gateway. Wires the bots to the companion's mind.

Each inbound message becomes a CognitiveController.think() call, so the entire
mind — moral courage, affect, agents, memory — applies to a chat exactly as it
does to a local query. Replies are kept to chat length. Every exchange is
audited (data/gateway/messages.json) and gated by a per-platform chat allowlist.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from synapseos.gateway.discord_bot import DiscordBot
from synapseos.gateway.telegram_bot import TelegramBot


@dataclass
class GatewayConfig:
    telegram_token: str = ""
    telegram_allowed_chats: list = field(default_factory=list)
    discord_webhook: str = ""
    discord_token: str = ""
    discord_allowed_channels: list = field(default_factory=list)
    reply_max_chars: int = 3500


class MessageGateway:
    def __init__(self, controller, config: GatewayConfig | None = None,
                 log_path: str = "data/gateway/messages.json",
                 telegram_transport=None):
        self.controller = controller
        self.config = config or GatewayConfig()
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._log: list[dict] = []
        self._load_log()

        self.telegram = TelegramBot(
            token=self.config.telegram_token, handler=self._handle,
            transport=telegram_transport,
            allowed_chats=self.config.telegram_allowed_chats or None,
            on_log=self._record)
        self.discord = DiscordBot(
            webhook_url=self.config.discord_webhook,
            token=self.config.discord_token, handler=self._handle,
            allowed_channels=self.config.discord_allowed_channels or None,
            on_log=self._record)

    # -- the mind behind the chat ----------------------------------------------------

    def _handle(self, text: str, chat_id) -> str:
        """Route an inbound message through the full cognitive pipeline."""
        result = self.controller.think(text)
        reply = result.primary_output
        return reply[: self.config.reply_max_chars]

    # -- audit ------------------------------------------------------------------------

    def _load_log(self) -> None:
        if self.log_path.exists():
            try:
                self._log = json.loads(self.log_path.read_text()).get("messages", [])
            except Exception:
                self._log = []

    def _record(self, platform: str, chat_id, inbound: str, reply: str) -> None:
        with self._lock:
            self._log.append({"t": time.time(), "platform": platform,
                              "chat": str(chat_id), "in": inbound[:300],
                              "out": reply[:300]})
            self._log = self._log[-2000:]
            self.log_path.write_text(json.dumps({"messages": self._log}, indent=2))

    def history(self, limit: int = 20) -> list[dict]:
        return self._log[-limit:]

    # -- lifecycle --------------------------------------------------------------------

    def start(self) -> dict:
        status = {}
        if self.config.telegram_token or self.telegram.transport is not None:
            self.telegram.start()
            status["telegram"] = "listening"
        else:
            status["telegram"] = "no token"
        status["discord_outbound"] = "ready" if self.config.discord_webhook else "no webhook"
        status["discord_inbound"] = self.discord.start()
        return status

    def stop(self) -> None:
        self.telegram.stop()
        self.discord.stop()

    def post_discord(self, text: str) -> bool:
        return self.discord.post(text)

    def stats(self) -> dict:
        return {"telegram_handled": self.telegram.messages_handled,
                "discord_handled": self.discord.messages_handled,
                "discord_inbound": self.discord.inbound_available,
                "logged": len(self._log)}
