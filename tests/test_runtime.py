from __future__ import annotations

import threading

from manga_repaint.catalog import Catalog
from manga_repaint.runtime import EventHub, JobQueue


def test_event_stream_starts_with_snapshot_then_new_events(tmp_path) -> None:
    catalog = Catalog(tmp_path)
    catalog.add_event("old", {"value": 1})
    hub = EventHub(catalog, lambda: [{"id": "job"}])
    stream = hub.stream()
    assert "event: snapshot" in next(stream)
    hub.publish("job_progress", {"percent": 50}, "job")
    message = next(stream)
    assert "event: job_progress" in message
    assert '"job_id": "job"' in message


def test_job_queue_preserves_reordered_pending_jobs() -> None:
    first_started = threading.Event()
    release_first = threading.Event()
    processed: list[str] = []
    events: list[tuple[str, dict, str | None]] = []

    def process(job_id: str) -> None:
        processed.append(job_id)
        if job_id == "first":
            first_started.set()
            release_first.wait(timeout=2)

    queue = JobQueue(
        process,
        lambda kind, payload, job_id: events.append((kind, payload, job_id)),
    )
    queue.enqueue("first")
    assert first_started.wait(timeout=2)
    queue.enqueue("second")
    queue.enqueue("third")
    queue.reorder(["third", "second"])
    release_first.set()
    for _ in range(100):
        if len(processed) == 3:
            break
        threading.Event().wait(0.01)

    assert processed == ["first", "third", "second"]
    assert all(kind == "job_queued" for kind, _payload, _job_id in events)
