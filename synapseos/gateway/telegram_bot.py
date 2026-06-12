"""
Gateway — Telegram bot. Pure stdlib long-poll, no dependency.

Receives messages via getUpdates (long polling) and replies via sendMessage.
A message handler (typically CognitiveController.think) turns each inbound
text into a reply. An optional transport is injectable for tests so the whole
loop runs with zero network.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.parse
import urllib.request


class _HTTPTransport:
    """Real Telegram Bot API transport."""

    def __init__(self, token: str, timeout: float = 35.0):
        self.base = f"https://api.telegram.org/bot{token}"
        self.timeout = timeout

    def get_updates(self, offset: int, long_poll: int = 30) -> list[dict]:
        url = f"{self.base}/getUpdates?timeout={long_poll}&offset={offset}"
        with urllib.request.urlopen(url, timeout=self.timeout) as r:
            data = json.loads(r.read())
        return data.get("result", []) if data.get("ok") else []

    def send_message(self, chat_id, text: str) -> bool:
        url = f"{self.base}/sendMessage"
        body = json.dumps({"chat_id": chat_id, "text": text[:4000]}).encode()
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read()).get("ok", False)
        except Exception:
            return False


class TelegramBot:
    def __init__(self, token: str = "", handler=None, transport=None,
                 allowed_chats: list | None = None, on_log=None):
        """
        handler(text, chat_id) -> reply str
        transport: object with get_updates(offset) + send_message(chat_id, text)
        allowed_chats: only these chat ids get replies (None = anyone)
        """
        self.handler = handler
        self.transport = transport or (_HTTPTransport(token) if token else None)
        self.allowed_chats = set(str(c) for c in allowed_chats) if allowed_chats else None
        self.on_log = on_log
        self._offset = 0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._greeted: set[str] = set()
        self.messages_handled = 0

    def _allowed(self, chat_id) -> bool:
        return self.allowed_chats is None or str(chat_id) in self.allowed_chats

    def poll_once(self) -> int:
        """Fetch and handle one batch of updates. Returns count handled."""
        if self.transport is None:
            return 0
        updates = self.transport.get_updates(self._offset)
        handled = 0
        for upd in updates:
            self._offset = max(self._offset, upd.get("update_id", 0) + 1)
            msg = upd.get("message") or upd.get("edited_message") or {}
            text = msg.get("text", "")
            chat_id = (msg.get("chat") or {}).get("id")
            if not text or chat_id is None:
                continue
            if not self._allowed(chat_id):
                self.transport.send_message(
                    chat_id, "Sorry — I'm a private AI companion and can only "
                    "talk with approved contacts.")
                continue
            reply = self._reply_for(text, chat_id)
            self.transport.send_message(chat_id, reply)
            self.messages_handled += 1
            handled += 1
            if self.on_log:
                self.on_log("telegram", chat_id, text, reply)
        return handled

    def _reply_for(self, text: str, chat_id) -> str:
        prefix = ""
        if str(chat_id) not in self._greeted:
            self._greeted.add(str(chat_id))
            prefix = "(I'm Synapse, an AI companion.) "
        if self.handler is None:
            return prefix + "I'm online but have no mind attached yet."
        try:
            return prefix + str(self.handler(text, chat_id))
        except Exception as e:
            return prefix + f"I hit an error: {e}"

    def start(self, interval: float = 1.0) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def _loop() -> None:
            while not self._stop.wait(interval):
                try:
                    self.poll_once()
                except Exception as e:
                    print(f"[TelegramBot] poll error: {e}")
                    time.sleep(3.0)

        self._thread = threading.Thread(target=_loop, daemon=True, name="telegram-bot")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
