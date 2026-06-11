"""
Identity — the approval Outbox. Autonomy with a brake.

Your vision doc was explicit: a system that can email, post, or spend should
have approval and oversight. So outbound communications the companion composes
on its own accounts are queued here as DRAFTS. Nothing leaves until approved —
by you, or (if you choose) auto-approved under rules you set.

Flow:
  companion drafts  → outbox.draft(channel, to, subject, body)  → status "pending"
  you review        → outbox.pending()
  you approve       → outbox.approve(id) → executes via ActionExecutor → "sent"
  you reject        → outbox.reject(id, reason) → "rejected"

Every state change is timestamped and kept — the full record of what the
companion wanted to say and what actually went out.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class OutboxItem:
    id: str
    channel: str             # email | telegram | webhook | imessage | social
    to: str
    subject: str
    body: str
    status: str = "pending"  # pending | approved | sent | rejected | failed
    reason: str = ""
    created_at: float = field(default_factory=time.time)
    resolved_at: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class Outbox:
    def __init__(self, executor=None, store_path: str = "data/identity/outbox.json",
                 auto_approve: bool = False):
        self.executor = executor
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.auto_approve = auto_approve
        self._items: dict[str, OutboxItem] = {}
        self._load()

    def _load(self) -> None:
        if self.store_path.exists():
            try:
                for it in json.loads(self.store_path.read_text()).get("items", []):
                    self._items[it["id"]] = OutboxItem(**it)
            except Exception:
                pass

    def save(self) -> None:
        self.store_path.write_text(json.dumps(
            {"items": [i.to_dict() for i in self._items.values()]}, indent=2))

    # -- draft / review / approve ---------------------------------------------------

    def draft(self, channel: str, to: str, body: str, subject: str = "") -> OutboxItem:
        item = OutboxItem(id=uuid.uuid4().hex[:10], channel=channel, to=to,
                          subject=subject, body=body)
        self._items[item.id] = item
        self.save()
        if self.auto_approve:
            self.approve(item.id)
        return item

    def pending(self) -> list[OutboxItem]:
        return [i for i in self._items.values() if i.status == "pending"]

    def history(self, limit: int = 20) -> list[OutboxItem]:
        return sorted(self._items.values(), key=lambda i: i.created_at)[-limit:]

    def approve(self, item_id: str) -> OutboxItem | None:
        item = self._items.get(item_id)
        if item is None or item.status not in ("pending", "approved"):
            return item
        item.status = "approved"
        result = self._send(item)
        item.status = "sent" if result else "failed"
        item.resolved_at = time.time()
        self.save()
        return item

    def reject(self, item_id: str, reason: str = "") -> OutboxItem | None:
        item = self._items.get(item_id)
        if item is None:
            return None
        item.status = "rejected"
        item.reason = reason
        item.resolved_at = time.time()
        self.save()
        return item

    def _send(self, item: OutboxItem) -> bool:
        if self.executor is None:
            return False
        if item.channel == "email":
            r = self.executor.act("email.send", to=item.to, subject=item.subject,
                                  body=item.body)
        elif item.channel == "telegram":
            r = self.executor.act("telegram.send", text=item.body, chat_id=item.to)
        elif item.channel == "webhook":
            r = self.executor.act("webhook.post", url=item.to, text=item.body)
        elif item.channel == "imessage":
            r = self.executor.act("imessage.send", to=item.to, text=item.body)
        else:
            return False
        return bool(getattr(r, "ok", False))

    def stats(self) -> dict:
        by_status: dict[str, int] = {}
        for i in self._items.values():
            by_status[i.status] = by_status.get(i.status, 0) + 1
        return {"total": len(self._items), "by_status": by_status,
                "auto_approve": self.auto_approve}
