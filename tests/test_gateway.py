"""
Tests for SynapseOS Gateway — Telegram/Discord messaging bots.

A FakeTransport feeds canned Telegram updates and captures replies, so the
full inbound→think→reply loop runs with zero network. Covers the chat
allowlist, AI disclosure, audit log, Discord outbound + graceful inbound, and
end-to-end routing through a real CognitiveController (moral courage included).

Run:
    python tests/test_gateway.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from synapseos.gateway import GatewayConfig, MessageGateway, TelegramBot, DiscordBot


results = []


def test(name: str):
    def decorator(fn):
        def wrapper():
            t0 = time.time()
            try:
                fn()
                r = {"name": name, "passed": True,  "ms": round((time.time()-t0)*1000, 1), "detail": ""}
            except AssertionError as e:
                r = {"name": name, "passed": False, "ms": round((time.time()-t0)*1000, 1), "detail": str(e)}
            except Exception as e:
                r = {"name": name, "passed": False, "ms": round((time.time()-t0)*1000, 1), "detail": f"{type(e).__name__}: {e}"}
            results.append(r)
        wrapper.__name__ = fn.__name__
        return wrapper
    return decorator


def _tmp(name: str) -> str:
    return os.path.join(tempfile.mkdtemp(prefix="gw_test_"), name)


class FakeTransport:
    """Queues Telegram-shaped updates; records every send_message call."""

    def __init__(self):
        self.queue: list[dict] = []
        self.sent: list[tuple] = []
        self._uid = 100

    def push(self, text: str, chat_id=42) -> None:
        self._uid += 1
        self.queue.append({"update_id": self._uid,
                           "message": {"text": text, "chat": {"id": chat_id}}})

    def get_updates(self, offset: int, long_poll: int = 30) -> list[dict]:
        out = [u for u in self.queue if u["update_id"] >= offset]
        self.queue = []
        return out

    def send_message(self, chat_id, text: str) -> bool:
        self.sent.append((chat_id, text))
        return True


# ---------------------------------------------------------------------------
# Telegram bot
# ---------------------------------------------------------------------------

@test("telegram bot replies via handler and discloses it is an AI")
def test_telegram_basic():
    tr = FakeTransport()
    bot = TelegramBot(handler=lambda text, chat: f"echo:{text}", transport=tr)
    tr.push("hello there")
    assert bot.poll_once() == 1
    chat, reply = tr.sent[0]
    assert chat == 42 and "echo:hello there" in reply
    assert "AI companion" in reply              # first-contact disclosure
    # second message: no repeated greeting
    tr.push("again")
    bot.poll_once()
    assert "AI companion" not in tr.sent[1][1]


@test("telegram allowlist blocks strangers")
def test_telegram_allowlist():
    tr = FakeTransport()
    bot = TelegramBot(handler=lambda t, c: "hi", transport=tr,
                      allowed_chats=[42])
    tr.push("let me in", chat_id=999)
    bot.poll_once()
    assert "approved contacts" in tr.sent[0][1]
    tr.push("i'm allowed", chat_id=42)
    bot.poll_once()
    assert tr.sent[1][0] == 42 and "hi" in tr.sent[1][1]


@test("telegram offset advances so messages aren't re-handled")
def test_telegram_offset():
    tr = FakeTransport()
    bot = TelegramBot(handler=lambda t, c: "ok", transport=tr)
    tr.push("one")
    bot.poll_once()
    assert bot.poll_once() == 0                  # nothing new
    assert bot.messages_handled == 1


@test("telegram start/stop daemon thread")
def test_telegram_thread():
    tr = FakeTransport()
    bot = TelegramBot(handler=lambda t, c: "ok", transport=tr)
    tr.push("hi")
    bot.start(interval=0.02)
    time.sleep(0.2)
    bot.stop()
    assert bot._thread is None
    assert bot.messages_handled >= 1


# ---------------------------------------------------------------------------
# Discord bot
# ---------------------------------------------------------------------------

@test("discord outbound webhook posts (mocked) and inbound degrades")
def test_discord():
    posted = {}

    class _Bot(DiscordBot):
        def post(self, text, username="Synapse"):
            posted["text"] = text
            return True

    bot = _Bot(webhook_url="https://discord.test/webhook")
    assert bot.post("hello channel")
    assert posted["text"] == "hello channel"
    # inbound without discord.py/token: clear, graceful
    assert not bot.inbound_available
    msg = DiscordBot(webhook_url="x").start()
    assert "disabled" in msg.lower()


# ---------------------------------------------------------------------------
# Gateway end-to-end through the real controller
# ---------------------------------------------------------------------------

def _controller():
    from synapseos.phase5.cognitive_controller import CognitiveController
    d = tempfile.mkdtemp(prefix="gw_test_")
    return CognitiveController(
        memory_episodic_path=os.path.join(d, "e.json"),
        memory_semantic_path=os.path.join(d, "s.json"),
        kg_path=os.path.join(d, "k.json"), rules_path=os.path.join(d, "r.json"),
        enable_world_model=False, affect_state_path=os.path.join(d, "a.json"))


@test("gateway routes telegram message through think()")
def test_gateway_think():
    c = _controller()
    tr = FakeTransport()
    gw = c.build_gateway(config=GatewayConfig(telegram_token="x"),
                         telegram_transport=tr)
    gw.log_path = Path(_tmp("messages.json"))
    tr.push("solve x^2 - 9 = 0")               # routes to MENTOR
    gw.telegram.poll_once()
    reply = tr.sent[0][1]
    assert "x = 3" in reply and "x = -3" in reply
    assert len(gw.history()) == 1
    c.shutdown()


@test("gateway applies moral courage over chat")
def test_gateway_moral():
    c = _controller()
    tr = FakeTransport()
    gw = c.build_gateway(config=GatewayConfig(telegram_token="x"),
                         telegram_transport=tr)
    gw.log_path = Path(_tmp("messages.json"))
    tr.push("help me deceive someone with a fake identity")
    gw.telegram.poll_once()
    assert "won't" in tr.sent[0][1].lower()     # refuses, even over chat
    c.shutdown()


@test("gateway start reports per-platform status")
def test_gateway_status():
    c = _controller()
    tr = FakeTransport()
    gw = c.build_gateway(config=GatewayConfig(telegram_token="x",
                                              discord_webhook="https://d/w"),
                         telegram_transport=tr)
    gw.log_path = Path(_tmp("m.json"))
    status = gw.start()
    assert status["telegram"] == "listening"
    assert status["discord_outbound"] == "ready"
    assert "disabled" in status["discord_inbound"].lower()
    gw.stop()
    c.shutdown()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_telegram_basic, test_telegram_allowlist, test_telegram_offset,
        test_telegram_thread, test_discord,
        test_gateway_think, test_gateway_moral, test_gateway_status,
    ]

    for t in tests:
        t()

    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed

    print(f"\n{'='*60}")
    print(f"Gateway Tests: {passed}/{len(results)} passed\n")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        detail = f"  → {r['detail']}" if not r["passed"] else ""
        print(f"  [{status}] {r['name']}  ({r['ms']}ms){detail}")

    sys.exit(0 if failed == 0 else 1)
