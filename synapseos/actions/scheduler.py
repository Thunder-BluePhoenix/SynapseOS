"""
Actions — task scheduler (Hermes's cron, the SynapseOS way).

Persistent scheduled tasks executed through the same policy-gated
ActionExecutor — a scheduled action obeys the same sandbox and lands in the
same audit trail as a manual one. Daemon thread, Event-based stop, same
pattern as the idle loop and web learner.

Schedule forms:
  every_s   — repeat every N seconds
  at_ts     — run once at a unix timestamp
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ScheduledTask:
    id: str
    name: str
    tool: str                   # ActionExecutor tool name
    args: dict
    every_s: float = 0.0        # 0 = one-shot
    at_ts: float = 0.0          # one-shot time (used when every_s == 0)
    last_run: float = 0.0
    runs: int = 0
    enabled: bool = True

    def due(self, now: float) -> bool:
        if not self.enabled:
            return False
        if self.every_s > 0:
            return now - self.last_run >= self.every_s
        return self.at_ts > 0 and now >= self.at_ts and self.runs == 0

    def to_dict(self) -> dict:
        return asdict(self)


class TaskScheduler:
    def __init__(self, executor, store_path: str = "data/actions/schedule.json"):
        self.executor = executor
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._tasks: dict[str, ScheduledTask] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._load()

    def _load(self) -> None:
        if self.store_path.exists():
            try:
                for t in json.loads(self.store_path.read_text()).get("tasks", []):
                    self._tasks[t["id"]] = ScheduledTask(**t)
            except Exception:
                pass

    def save(self) -> None:
        with self._lock:
            self.store_path.write_text(json.dumps(
                {"tasks": [t.to_dict() for t in self._tasks.values()]}, indent=2))

    # -- API ----------------------------------------------------------------------

    def add(self, name: str, tool: str, args: dict,
            every_s: float = 0.0, at_ts: float = 0.0) -> ScheduledTask:
        task = ScheduledTask(id=uuid.uuid4().hex[:10], name=name, tool=tool,
                             args=args, every_s=every_s, at_ts=at_ts)
        with self._lock:
            self._tasks[task.id] = task
        self.save()
        return task

    def remove(self, task_id: str) -> bool:
        with self._lock:
            removed = self._tasks.pop(task_id, None) is not None
        self.save()
        return removed

    def tasks(self) -> list[ScheduledTask]:
        return list(self._tasks.values())

    def tick(self, now: float | None = None) -> int:
        """Run due tasks once. Returns count executed. Callable without thread."""
        now = now or time.time()
        ran = 0
        for task in list(self._tasks.values()):
            if task.due(now):
                self.executor.act(task.tool, **task.args)
                task.last_run = now
                task.runs += 1
                if task.every_s == 0:
                    task.enabled = False        # one-shot complete
                ran += 1
        if ran:
            self.save()
        return ran

    def start_background(self, interval: float = 5.0) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()

        def _loop() -> None:
            while not self._stop.wait(interval):
                try:
                    self.tick()
                except Exception as e:
                    print(f"[Scheduler] tick error: {e}")

        self._thread = threading.Thread(target=_loop, daemon=True,
                                        name="action-scheduler")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self.save()
