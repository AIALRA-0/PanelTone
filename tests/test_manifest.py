from __future__ import annotations

from pathlib import Path

from manga_repaint.manifest import Manifest
from manga_repaint.models import JobSpec, JobStatus, PanelBox


def test_manifest_creates_and_summarizes_job(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path / "manifest.sqlite")
    spec = JobSpec(source=tmp_path, workspace=tmp_path)
    manifest.create_job("abc", spec)
    summary = manifest.summary("abc")
    assert summary["status"] == "created"
    assert summary["page_count"] == 0
    assert summary["unit_counts"] == {}


def test_manifest_recovers_interrupted_running_job(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path / "manifest.sqlite")
    spec = JobSpec(source=tmp_path, workspace=tmp_path)
    manifest.create_job("abc", spec)
    manifest.set_job_status("abc", JobStatus.RUNNING)
    assert manifest.recover_interrupted("abc")
    assert manifest.summary("abc")["status"] == "paused"


def test_manifest_resets_stale_units_for_paused_job(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path / "manifest.sqlite")
    manifest.create_job("paused", JobSpec(source=tmp_path, workspace=tmp_path))
    source = tmp_path / "source"
    source.mkdir()
    page_id = manifest.add_page("paused", 0, source / "page.png", "sha", 10, 10)
    unit_id = manifest.add_unit(
        page_id,
        0,
        PanelBox(0, 0, 10, 10),
        "palette",
        "params",
        source / "page.png",
        source / "mask.png",
    )
    manifest.mark_unit_running(unit_id)
    manifest.set_job_status("paused", JobStatus.PAUSED)

    assert manifest.reset_running_units("paused", "pause cleanup") == 1
    assert manifest.summary("paused")["unit_counts"] == {"pending": 1}
    assert manifest.reset_running_units("paused") == 0


def test_manifest_does_not_recover_a_fresh_worker_heartbeat(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path / "manifest.sqlite")
    spec = JobSpec(source=tmp_path, workspace=tmp_path)
    manifest.create_job("fresh", spec)
    manifest.set_job_status("fresh", JobStatus.RUNNING)

    assert not manifest.recover_interrupted("fresh", stale_after_seconds=120)
    assert manifest.summary("fresh")["status"] == "running"


def test_paused_job_with_stale_running_unit_reports_paused_stage(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path / "manifest.sqlite")
    spec = JobSpec(source=tmp_path, workspace=tmp_path)
    manifest.create_job("paused", spec)
    source = tmp_path / "source.png"
    mask = tmp_path / "mask.png"
    source.write_bytes(b"source")
    mask.write_bytes(b"mask")
    page_id = manifest.add_page("paused", 0, source, "sha", 10, 10)
    manifest.add_unit(page_id, 0, PanelBox(0, 0, 10, 10), "palette", "params", source, mask)
    unit_id = manifest.page_units(page_id)[0]["id"]
    manifest.mark_unit_running(unit_id)
    manifest.set_job_status("paused", JobStatus.PAUSED)

    assert manifest.progress("paused")["stage"] == "paused"


def test_manifest_bulk_page_reads_group_units_and_latest_semantic_rows(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path / "manifest.sqlite")
    manifest.create_job("bulk", JobSpec(source=tmp_path, workspace=tmp_path))
    source = tmp_path / "source.png"
    mask = tmp_path / "mask.png"
    source.write_bytes(b"source")
    mask.write_bytes(b"mask")
    first_page = manifest.add_page("bulk", 0, source, "sha0", 10, 10)
    second_page = manifest.add_page("bulk", 1, source, "sha1", 10, 10)
    manifest.add_unit(first_page, 0, PanelBox(0, 0, 10, 10), "palette", "p0", source, mask)
    manifest.add_unit(second_page, 0, PanelBox(0, 0, 10, 10), "palette", "p1", source, mask)
    manifest.save_semantic_mask(
        first_page,
        provider="deterministic-protection",
        version="1",
        descriptor={"status": "fallback"},
        confidence_path=None,
        uncertain_path=None,
    )
    manifest.save_semantic_mask(
        first_page,
        provider="semantic-manga-v1-compatible/koharu-yolo26s",
        version="koharu-yolo26s",
        descriptor={"status": "ready"},
        confidence_path=None,
        uncertain_path=None,
    )

    units = manifest.page_units_for_job("bulk")
    semantic = manifest.semantic_masks_for_job("bulk")

    assert [item["unit_index"] for item in units[0]] == [0]
    assert [item["unit_index"] for item in units[1]] == [0]
    assert semantic[0]["descriptor"] == {"status": "ready"}
