from __future__ import annotations

import time

from PIL import Image

from manga_repaint import __version__
from manga_repaint.model_server import Flux2Runtime, app


def test_model_service_reports_package_version() -> None:
    assert app.version == __version__


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


def test_runtime_forwards_negative_mode_and_style_metadata() -> None:
    class FakePipeline:
        def __init__(self) -> None:
            self.received = {}

        def __call__(
            self,
            image,
            prompt,
            height,
            width,
            guidance_scale,
            num_inference_steps,
            generator,
            negative_prompt,
        ):
            self.received = locals()
            return type("Output", (), {"images": [Image.new("RGB", (width, height), "white")]})()

    runtime = Flux2Runtime()
    runtime.device = "cpu"
    runtime._pipeline = FakePipeline()
    runtime.state = "ready"
    runtime.generate(
        Image.new("RGB", (32, 32), "white"),
        [],
        "base prompt",
        7,
        negative_prompt="bleeding colours",
        mode="style_full",
        metadata={
            "style_guidance": "graphic cel shading",
            "guidance_scale": 1.2,
            "num_inference_steps": 6,
        },
    )

    received = runtime._pipeline.received
    assert received["negative_prompt"] == "bleeding colours"
    assert "graphic cel shading" in received["prompt"]
    assert "clearly different rendering style" in received["prompt"]
    assert received["guidance_scale"] == 1.2
    assert received["num_inference_steps"] == 6
    runtime.shutdown()
