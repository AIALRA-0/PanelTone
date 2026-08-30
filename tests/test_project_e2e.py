from __future__ import annotations

import zipfile
from pathlib import Path

from PIL import Image

from manga_repaint.config import Settings
from manga_repaint.engines import EngineRegistry
from manga_repaint.models import JobSpec
from manga_repaint.project import ProjectManager


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
    completed_index = max(
        index
        for index, (kind, payload) in enumerate(events)
        if kind == "job_status" and payload.get("status") == "completed"
    )
    assert names.index("page_ready") < completed_index
