from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from typing import Any

from .catalog import Catalog


class EventHub:
    def __init__(
        self,
        catalog: Catalog,
        snapshot: Callable[[], Any],
        sanitize: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ):
        self.catalog = catalog
        self.snapshot = snapshot
        self.sanitize = sanitize or (lambda payload: payload)
        self._condition = threading.Condition()

    def publish(self, kind: str, payload: dict[str, Any], job_id: str | None = None) -> int:
        event_id = self.catalog.add_event(kind, payload, job_id)
        with self._condition:
            self._condition.notify_all()
        return event_id

    def stream(self, last_event_id: int = 0) -> Iterator[str]:
        if last_event_id <= 0:
            last_event_id = self.catalog.latest_event_id()
        snapshot = {"jobs": self.snapshot()}
        # Give the initial snapshot a cursor as well.  EventSource then sends
        # Last-Event-ID on a reconnect even when no domain event arrived
        # before the first network interruption.
        yield (
            f"id: {last_event_id}\nevent: snapshot\n"
            f"data: {json.dumps(snapshot, ensure_ascii=False)}\n\n"
        )
        cursor = last_event_id
        while True:
            events = self.catalog.events_after(cursor)
            if events:
                for item in events:
                    cursor = int(item["id"])
                    data = self.sanitize(dict(item["payload"]))
                    if item["job_id"]:
                        data.setdefault("job_id", item["job_id"])
                    yield (
                        f"id: {cursor}\nevent: {item['kind']}\n"
                        f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                    )
                continue
            with self._condition:
                self._condition.wait(timeout=15)
            yield "event: heartbeat\ndata: {}\n\n"


class JobQueue:
    def __init__(
        self,
        process: Callable[[str], Any],
        on_state: Callable[[str, dict[str, Any], str | None], None],
    ):
        self.process = process
        self.on_state = on_state
        self._pending: list[str] = []
        self._pending_lock = threading.RLock()
        self._condition = threading.Condition(self._pending_lock)
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, name="paneltone-gpu-worker", daemon=True)
        self._worker.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the queue worker during local app shutdown."""
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._worker.is_alive():
            self._worker.join(timeout=max(0.0, timeout))

    def enqueue(self, job_id: str) -> int:
        with self._pending_lock:
            if job_id in self._pending:
                return self._pending.index(job_id) + 1
            self._pending.append(job_id)
            position = len(self._pending)
            self._notify_positions()
            self._condition.notify()
            return position

    def positions(self) -> dict[str, int]:
        with self._pending_lock:
            return {job_id: index + 1 for index, job_id in enumerate(self._pending)}

    def reorder(self, job_ids: list[str]) -> None:
        with self._pending_lock:
            if set(job_ids) != set(self._pending):
                raise ValueError("队列顺序必须包含全部等待任务")
            self._pending[:] = job_ids
            self._notify_positions()
            self._condition.notify()

    def remove(self, job_id: str) -> bool:
        with self._pending_lock:
            if job_id not in self._pending:
                return False
            self._pending.remove(job_id)
            self._notify_positions()
            return True

    def _notify_positions(self) -> None:
        for job_id, position in self.positions().items():
            self.on_state(
                "job_queued",
                {"status": "queued", "queue_position": position},
                job_id,
            )

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._condition:
                self._condition.wait_for(
                    lambda: self._stop.is_set() or bool(self._pending),
                    timeout=0.25,
                )
                if self._stop.is_set() or not self._pending:
                    continue
                job_id = self._pending.pop(0)
                self._notify_positions()
            try:
                self.process(job_id)
            except Exception as exc:
                self.on_state("job_error", {"message": str(exc)}, job_id)
