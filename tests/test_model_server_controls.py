from __future__ import annotations

import time

from manga_repaint.model_server import Flux2Runtime


def test_runtime_interrupt_marks_active_request() -> None:
    runtime = Flux2Runtime()
    runtime.state = "generating"
    runtime._active_requests = 1

    result = runtime.request_interrupt()

    assert result == {"status": "interrupt_requested", "active": True}
    assert runtime._cancel_event.is_set()


def test_runtime_release_clears_idle_pipeline() -> None:
    runtime = Flux2Runtime()
    runtime._pipeline = object()
    runtime.state = "ready"

    result = runtime.release()

    assert result["status"] == "released"
    assert runtime._pipeline is None
    assert runtime.state == "idle"


def test_runtime_release_refuses_active_pipeline() -> None:
    runtime = Flux2Runtime()
    runtime._pipeline = object()
    runtime.state = "generating"
    runtime._active_requests = 1

    result = runtime.release()

    assert result == {"status": "busy", "active": True, "state": "generating"}


def test_runtime_releases_pipeline_after_idle_grace_period() -> None:
    runtime = Flux2Runtime()
    runtime.idle_release_seconds = 1
    runtime._pipeline = object()
    runtime.state = "ready"
    runtime.last_activity = time.time() - 2

    result = runtime.release_if_idle()

    assert result["status"] == "released"
    assert runtime._pipeline is None
    runtime.shutdown()
