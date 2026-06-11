"""
Actions — communication tools. The companion's voice to the world.

  email.send    (TRUSTED) — SMTP send from the companion's own account
  email.read    (TRUSTED) — IMAP fetch recent messages
  telegram.send (TRUSTED) — Telegram Bot API sendMessage
  webhook.post  (TRUSTED) — Slack/Discord/generic incoming webhook
  imessage.send (TRUSTED) — macOS Messages via osascript

Credentials are NEVER hardcoded — read from env vars or the policy's
credential store (see synapseos.identity). Outbound messages can be routed
through an approval outbox first (synapseos.identity.Outbox) so the companion
drafts and the user approves — the oversight your vision called for.
"""

from __future__ import annotations

import json
import smtplib
import subprocess
import urllib.request
from email.message import EmailMessage

from synapseos.actions.executor import TRUSTED, ActionResult, fail, ok


def email_send(policy, to: str, subject: str, body: str,
               from_addr: str = "") -> ActionResult:
    host = policy.credential("SMTP_HOST") or "smtp.gmail.com"
    port = int(policy.credential("SMTP_PORT") or 587)
    user = policy.credential("SMTP_USER") or policy.credential("GMAIL_ADDRESS")
    password = policy.credential("SMTP_PASSWORD") or policy.credential("GMAIL_APP_PASSWORD")
    if not user or not password:
        return fail("no SMTP credentials (set GMAIL_ADDRESS + GMAIL_APP_PASSWORD)")
    msg = EmailMessage()
    msg["From"] = from_addr or user
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    try:
        with smtplib.SMTP(host, port, timeout=30) as s:
            s.starttls()
            s.login(user, password)
            s.send_message(msg)
        return ok(f"email sent to {to}: \"{subject}\"")
    except Exception as e:
        return fail(f"{type(e).__name__}: {e}")


def email_read(policy, folder: str = "INBOX", limit: int = 5) -> ActionResult:
    import imaplib
    import email as email_mod
    host = policy.credential("IMAP_HOST") or "imap.gmail.com"
    user = policy.credential("IMAP_USER") or policy.credential("GMAIL_ADDRESS")
    password = policy.credential("IMAP_PASSWORD") or policy.credential("GMAIL_APP_PASSWORD")
    if not user or not password:
        return fail("no IMAP credentials (set GMAIL_ADDRESS + GMAIL_APP_PASSWORD)")
    try:
        m = imaplib.IMAP4_SSL(host)
        m.login(user, password)
        m.select(folder)
        _, ids = m.search(None, "ALL")
        out = []
        for mid in ids[0].split()[-limit:][::-1]:
            _, data = m.fetch(mid, "(RFC822)")
            msg = email_mod.message_from_bytes(data[0][1])
            out.append(f"From: {msg['From']}\nSubject: {msg['Subject']}")
        m.logout()
        return ok("\n---\n".join(out) or "(no messages)")
    except Exception as e:
        return fail(f"{type(e).__name__}: {e}")


def telegram_send(policy, text: str, chat_id: str = "") -> ActionResult:
    token = policy.credential("TELEGRAM_BOT_TOKEN")
    chat = chat_id or policy.credential("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return fail("no Telegram credentials (TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID)")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = json.dumps({"chat_id": chat, "text": text}).encode()
    try:
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            r.read()
        return ok(f"telegram message sent to {chat}")
    except Exception as e:
        return fail(f"{type(e).__name__}: {e}")


def webhook_post(policy, url: str, text: str = "", payload: dict | None = None) -> ActionResult:
    """Slack/Discord/generic incoming webhook. Slack+Discord both accept JSON."""
    body = payload or ({"content": text, "text": text} if text else {})
    try:
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            r.read()
        return ok("webhook posted")
    except Exception as e:
        return fail(f"{type(e).__name__}: {e}")


def imessage_send(policy, to: str, text: str) -> ActionResult:
    """macOS Messages via AppleScript."""
    script = (f'tell application "Messages" to send "{text}" to '
              f'buddy "{to}" of (service 1 whose service type is iMessage)')
    try:
        proc = subprocess.run(["osascript", "-e", script],
                              capture_output=True, text=True, timeout=20)
        if proc.returncode:
            return fail(proc.stderr.strip() or "osascript failed")
        return ok(f"iMessage sent to {to}")
    except Exception as e:
        return fail(f"{type(e).__name__}: {e}")


TOOLS = {
    "email.send": (email_send, TRUSTED),
    "email.read": (email_read, TRUSTED),
    "telegram.send": (telegram_send, TRUSTED),
    "webhook.post": (webhook_post, TRUSTED),
    "imessage.send": (imessage_send, TRUSTED),
}
