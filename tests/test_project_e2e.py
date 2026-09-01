from __future__ import annotations

import threading
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from PIL import Image

from manga_repaint.config import Settings
from manga_repaint.engines import EngineInterrupted, EngineRegistry
from manga_repaint.models import DetailMode, JobSpec
from manga_repaint.project import ProjectManager


def test_unconfigured_allowed_roots_accept_existing_local_path(
    tmp_path: Path, manga_pages: Path
) -> None:
    settings = Settings(data_root=tmp_path / "jobs")
    manager = ProjectManager(settings, EngineRegistry())

    job_id = manager.create_shell(
        JobSpec(source=manga_pages, workspace=settings.data_root, engine="palette")
    )

    assert manager.status(job_id)["status"] == "ingesting"


def test_configured_allowed_roots_accept_inside_and_reject_outside(
    tmp_path: Path, manga_pages: Path
) -> None:
    settings = Settings(
        data_root=tmp_path / "jobs",
        allowed_roots=[manga_pages],
    )
    manager = ProjectManager(settings, EngineRegistry())
    allowed_id = manager.create_shell(
        JobSpec(source=manga_pages, workspace=settings.data_root, engine="palette")
    )

    outside = tmp_path / "outside"
    outside.mkdir()
    Image.new("RGB", (16, 16), "white").save(outside / "page.png")

    assert manager.status(allowed_id)["status"] == "ingesting"
    with pytest.raises(PermissionError, match="outside configured allowed roots"):
        manager.create_shell(
            JobSpec(source=outside, workspace=settings.data_root, engine="palette")
        )


def test_whole_book_end_to_end_and_resume(tmp_path: Path, manga_pages: Path) -> None:
    settings = Settings(data_root=tmp_path / "jobs")
    manager = ProjectManager(settings, EngineRegistry())
    spec = JobSpec(source=manga_pages, workspace=settings.data_root, engine="palette")
    job_id = manager.create(spec)
    output = manager.process(job_id)

    assert output.is_file()
    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == [
            "page_00000.png",
            "page_00001.png",
            "page_00002.png",
        ]
    summary = manager.status(job_id)
    assert summary["status"] == "completed"
    assert summary["unit_counts"] == {"qa_passed": 3}

    first_output_mtime = output.stat().st_mtime_ns
    second_output = manager.process(job_id)
    assert second_output == output
    assert output.stat().st_mtime_ns >= first_output_mtime

    page = Path(manager._manifest(job_id).pages(job_id)[0]["output_path"])
    with Image.open(page) as image:
        assert image.size == (320, 420)


def test_page_ready_event_precedes_completed(tmp_path: Path, manga_pages: Path) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    manager = ProjectManager(
        Settings(data_root=tmp_path / "jobs"),
        EngineRegistry(),
        lambda kind, payload, _job_id: events.append((kind, payload)),
    )
    job_id = manager.create(
        JobSpec(source=manga_pages, workspace=tmp_path / "jobs", engine="palette")
    )
    manager.process(job_id)
    names = [kind for kind, _payload in events]
    assert "page_ready" in names
    first_ready = next(payload for kind, payload in events if kind == "page_ready")
    assert first_ready["status"] == "qa_passed"
    final_url = str(first_ready["final_url"])
    thumbnail_url = str(first_ready["thumbnail_url"])
    assert final_url.startswith(f"/api/jobs/{job_id}/pages/0/final")
    assert thumbnail_url.startswith(f"/api/jobs/{job_id}/pages/0/thumbnail")
    assert parse_qs(urlparse(final_url).query)["v"] == [str(first_ready["asset_revision"])]
    assert parse_qs(urlparse(thumbnail_url).query)["v"] == [str(first_ready["asset_revision"])]
    completed_index = max(
        index
        for index, (kind, payload) in enumerate(events)
        if kind == "job_status" and payload.get("status") == "completed"
    )
    assert names.index("page_ready") < completed_index


def test_page_preview_event_keeps_failed_output_inspectable(
    tmp_path: Path, manga_pages: Path
) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    settings = Settings(data_root=tmp_path / "jobs", qa_line_f1_min=1.1)
    manager = ProjectManager(
        settings,
        EngineRegistry(),
        lambda kind, payload, _job_id: events.append((kind, payload)),
    )
    job_id = manager.create(
        JobSpec(source=manga_pages, workspace=settings.data_root, engine="palette")
    )
    manager.process(job_id)

    preview_events = [payload for kind, payload in events if kind == "page_preview_ready"]
    assert preview_events
    preview_url = str(preview_events[0]["preview_url"])
    assert preview_url.startswith(f"/api/jobs/{job_id}/pages/0/preview")
    assert parse_qs(urlparse(preview_url).query)["v"] == [str(preview_events[0]["asset_revision"])]
    preview_path = manager._job_dir(job_id) / "preview" / "pages" / "page_00000.png"
    assert preview_path.is_file()
    page = manager._manifest(job_id).pages(job_id)[0]
    assert not page["output_path"]


def test_balanced_detail_mode_preserves_protected_edges(tmp_path: Path, manga_pages: Path) -> None:
    settings = Settings(data_root=tmp_path / "jobs")
    manager = ProjectManager(settings, EngineRegistry())
    job_id = manager.create(
        JobSpec(
            source=manga_pages,
            workspace=settings.data_root,
            engine="palette",
            detail_mode=DetailMode.BALANCED,
        )
    )

    output = manager.process(job_id)

    assert output.is_file()
    assert manager.status(job_id)["unit_counts"] == {"qa_passed": 3}


def test_pause_request_at_queue_boundary_does_not_start_job(
    tmp_path: Path, manga_pages: Path
) -> None:
    settings = Settings(data_root=tmp_path / "jobs")
    manager = ProjectManager(settings, EngineRegistry())
    job_id = manager.create(
        JobSpec(source=manga_pages, workspace=settings.data_root, engine="palette")
    )

    manager.pause(job_id)
    manager.process(job_id)
    assert manager.status(job_id)["status"] == "paused"

    # queue() is the explicit resume operation, so processing is allowed
    # after the pause request has been cleared at that boundary.
    manager.queue(job_id)
    assert manager.status(job_id)["status"] == "queued"


def test_pause_interrupts_active_engine_without_counting_failure(
    tmp_path: Path, manga_pages: Path
) -> None:
    started = threading.Event()
    interrupted = threading.Event()
    released = threading.Event()

    class InterruptibleEngine:
        name = "interruptible"

        def generate(self, _request):
            started.set()
            interrupted.wait(timeout=2)
            raise EngineInterrupted("测试中断")

        def interrupt(self):
            interrupted.set()
            return {"status": "interrupt_requested", "active": True}

        def release(self):
            released.set()
            return {"status": "released", "active": False, "state": "idle"}

        def healthcheck(self):
            return {"ok": True, "engine": self.name}

    manager = ProjectManager(
        Settings(data_root=tmp_path / "jobs"),
        EngineRegistry({"interruptible": InterruptibleEngine()}),
    )
    job_id = manager.create(
        JobSpec(source=manga_pages, workspace=tmp_path / "jobs", engine="interruptible")
    )
    worker = threading.Thread(target=manager.process, args=(job_id,), daemon=True)
    worker.start()
    assert started.wait(timeout=2)
    manager.pause(job_id)
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert manager.status(job_id)["status"] == "paused"
    assert manager.status(job_id)["unit_counts"] == {"pending": 3}
    assert released.is_set()
