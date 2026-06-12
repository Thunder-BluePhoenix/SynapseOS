"""
Gateway — Discord bot.

Outbound (always available): post to a channel via an incoming webhook URL —
pure stdlib. Inbound (the companion listening and replying in Discord) needs
the Discord Gateway websocket, which has no stdlib client; it activates only
when `discord.py` is installed, and degrades to a clear message otherwise.

  DiscordBot(webhook_url=..., token=..., handler=..., allowed_channels=[...])
    .post(text)         — outbound via webhook (works now)
    .start() / .stop()  — inbound listener (needs discord.py + bot token)
"""

from __future__ import annotations

import json
import threading
import urllib.request

try:
    import discord  # type: ignore
    _HAS_DISCORD = True
except ImportError:
    _HAS_DISCORD = False


class DiscordBot:
    def __init__(self, webhook_url: str = "", token: str = "", handler=None,
                 allowed_channels: list | None = None, on_log=None):
        self.webhook_url = webhook_url
        self.token = token
        self.handler = handler
        self.allowed_channels = set(str(c) for c in allowed_channels) \
            if allowed_channels else None
        self.on_log = on_log
        self._client = None
        self._thread: threading.Thread | None = None
        self.messages_handled = 0

    # -- outbound (stdlib) ---------------------------------------------------------

    def post(self, text: str, username: str = "Synapse") -> bool:
        if not self.webhook_url:
            return False
        body = json.dumps({"content": text[:1900], "username": username}).encode()
        req = urllib.request.Request(self.webhook_url, data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=20):
                return True
        except Exception:
            return False

    # -- inbound (optional discord.py) ----------------------------------------------

    @property
    def inbound_available(self) -> bool:
        return _HAS_DISCORD and bool(self.token)

    def start(self) -> str:
        if not self.inbound_available:
            reason = "discord.py not installed" if not _HAS_DISCORD else "no bot token"
            return (f"Discord inbound disabled ({reason}). Outbound webhook still "
                    f"works. `pip install discord.py` + set a bot token to enable "
                    f"two-way chat.")
        intents = discord.Intents.default()
        intents.message_content = True
        client = discord.Client(intents=intents)
        self._client = client

        @client.event
        async def on_message(message):  # noqa: ANN001
            if message.author == client.user:
                return
            chan = str(message.channel.id)
            if self.allowed_channels and chan not in self.allowed_channels:
                return
            text = message.content
            reply = self._reply_for(text, chan)
            await message.channel.send(reply)
            self.messages_handled += 1
            if self.on_log:
                self.on_log("discord", chan, text, reply)

        def _run():
            client.run(self.token)

        self._thread = threading.Thread(target=_run, daemon=True, name="discord-bot")
        self._thread.start()
        return "Discord inbound started."

    def _reply_for(self, text: str, channel) -> str:
        if self.handler is None:
            return "(Synapse, an AI companion) I'm online."
        try:
            return str(self.handler(text, channel))
        except Exception as e:
            return f"I hit an error: {e}"

    def stop(self) -> None:
        if self._client is not None:
            try:
                import asyncio
                asyncio.run_coroutine_threadsafe(self._client.close(),
                                                 self._client.loop)
            except Exception:
                pass
        self._thread = None
