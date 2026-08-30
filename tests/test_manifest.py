from __future__ import annotations

from pathlib import Path

from manga_repaint.manifest import Manifest
from manga_repaint.models import JobSpec, JobStatus


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
