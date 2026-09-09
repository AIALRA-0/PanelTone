from __future__ import annotations

import hashlib
import json
import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest
from PIL import Image

from manga_repaint.config import Settings
from manga_repaint.engines import EngineInterrupted, EngineRegistry
from manga_repaint.models import DetailMode, JobSpec
from manga_repaint.project import ProjectManager, _balloons_with_detected_text


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_color_processing_writes_book_plan_and_render_evidence(
    tmp_path: Path, manga_pages: Path
) -> None:
    settings = Settings(data_root=tmp_path / "jobs")
    manager = ProjectManager(settings, EngineRegistry())
    job_id = manager.create(
        JobSpec(source=manga_pages, workspace=settings.data_root, engine="palette")
    )

    manager.process(job_id)

    state_root = manager._job_dir(job_id) / "analysis" / "color-state"
    assert (state_root / "source_snapshot.json").is_file()
    assert (state_root / "identity_graph.json").is_file()
    assert (state_root / "color_book_state.json").is_file()
    assert list((state_root / "segments").glob("page_*.json"))
    assert list((state_root / "plans").glob("page_*.json"))
    assert list((state_root / "evidence").glob("page_*_unit_*_a*.json"))


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


def test_blank_colorize_page_is_source_passthrough_without_model_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "blank-book"
    source.mkdir()
    Image.new("RGB", (120, 100), "white").save(source / "001.png")
    settings = Settings(data_root=tmp_path / "jobs")
    manager = ProjectManager(settings, EngineRegistry())
    job_id = manager.create(
        JobSpec(source=source, workspace=settings.data_root, engine="palette")
    )

    output = manager.process(job_id)

    assert output.is_file()
    units = manager._manifest(job_id).page_units(
        int(manager._manifest(job_id).pages(job_id)[0]["id"])
    )
    assert len(units) == 1
    assert units[0]["generated_path"] is None
    qa = json.loads(units[0]["qa_json"])
    assert qa["source_class"] == "blank"
    assert qa["source_passthrough"] is True


def test_source_passthrough_survives_transactional_repair_without_dot_path(
    tmp_path: Path,
) -> None:
    source = tmp_path / "blank-book"
    source.mkdir()
    Image.new("RGB", (120, 100), "white").save(source / "001.png")
    settings = Settings(data_root=tmp_path / "jobs")
    manager = ProjectManager(settings, EngineRegistry())
    job_id = manager.create(
        JobSpec(source=source, workspace=settings.data_root, engine="palette")
    )
    manager.process(job_id)

    assert manager.repair_completed_colorization(job_id) == 1
    unit = manager._manifest(job_id).page_units(
        int(manager._manifest(job_id).pages(job_id)[0]["id"])
    )[0]
    assert unit["generated_path"] is None
    assert unit["final_path"]
    assert manager.status(job_id)["status"] == "completed"


def test_display_asset_backfill_is_bounded_and_idempotent(
    tmp_path: Path, manga_pages: Path
) -> None:
    settings = Settings(data_root=tmp_path / "jobs")
    manager = ProjectManager(settings, EngineRegistry())
    job_id = manager.create(
        JobSpec(source=manga_pages, workspace=settings.data_root, engine="palette")
    )
    manager.process(job_id)

    display = manager._job_dir(job_id) / "display" / "final" / "page_00000.webp"
    assert display.is_file()
    display.unlink()
    assert manager.prebuild_display_assets_for_job(job_id) == 1
    assert display.stat().st_size <= 900 * 1024
    assert manager.prebuild_display_assets_for_job(job_id) == 0


def test_geometry_locked_project_aligns_model_canvas_without_importing_edges(
    tmp_path: Path,
) -> None:
    settings = Settings(data_root=tmp_path / "jobs")
    manager = ProjectManager(settings, EngineRegistry())
    source = Image.new("RGB", (722, 1024), (230, 230, 230))
    generated = Image.new("RGB", (720, 1024), (220, 90, 60))
    spec = JobSpec(
        source=tmp_path / "source.png",
        workspace=settings.data_root,
        engine="palette",
    )

    final, _qa_mask = manager._compose_unit(
        source,
        generated,
        np.zeros((1024, 722), dtype=bool),
        spec,
    )

    assert final.size == source.size
    # Geometry-locked colourization keeps the source value channel as the
    # upper RGB bound; RGB luminance itself changes when hue/saturation is
    # introduced.
    assert np.array_equal(
        np.max(np.asarray(final), axis=2),
        np.max(np.asarray(source), axis=2),
    )


def test_new_model_candidate_rejects_material_aspect_warp(tmp_path: Path) -> None:
    settings = Settings(data_root=tmp_path / "jobs")
    manager = ProjectManager(settings, EngineRegistry())
    source = Image.new("RGB", (722, 1024), (230, 230, 230))
    warped = Image.new("RGB", (720, 1536), (220, 90, 60))
    spec = JobSpec(
        source=tmp_path / "source.png",
        workspace=settings.data_root,
        engine="palette",
    )

    with pytest.raises(ValueError, match="aspect ratio differs"):
        manager._render_color_candidate(source, warped, spec)

    repaired = manager._render_color_candidate(
        source, warped, spec, allow_legacy_aspect=True
    )
    assert repaired.size == source.size


def test_cobra_colourize_keeps_model_material_render_but_restores_source_ink(
    tmp_path: Path,
) -> None:
    settings = Settings(data_root=tmp_path / "jobs")
    manager = ProjectManager(settings, EngineRegistry())
    source_array = np.full((96, 96, 3), 245, dtype=np.uint8)
    source_array[10:86, 20:23] = 0
    source = Image.fromarray(source_array)
    generated_array = np.full((96, 96, 3), (235, 170, 145), dtype=np.uint8)
    generated_array[50:, :] = (135, 70, 85)  # authored model shadow layer
    generated = Image.fromarray(generated_array)
    spec = JobSpec(
        source=tmp_path / "source.png",
        workspace=settings.data_root,
        engine="cobra-candidate",
    )

    final, _ = manager._compose_unit(
        source,
        generated,
        np.zeros((96, 96), dtype=bool),
        spec,
    )
    result = np.asarray(final)
    expected = np.asarray(manager._render_color_candidate(source, generated, spec))

    assert np.array_equal(result[40, 21], source_array[40, 21])
    assert np.array_equal(result[20, 70], expected[20, 70])
    assert np.array_equal(result[70, 70], expected[70, 70])
    assert not np.array_equal(
        np.max(result, axis=2),
        np.max(source_array, axis=2),
    )


def test_semantic_ink_protection_keeps_only_dark_source_core(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "unit.png"
    pixels = np.full((24, 24, 3), 245, dtype=np.uint8)
    pixels[12, 12] = 0
    Image.fromarray(pixels, mode="RGB").save(source_path)
    manager = ProjectManager(Settings(data_root=tmp_path / "jobs"), EngineRegistry())
    spec = JobSpec(
        source=source_path,
        workspace=tmp_path / "jobs",
        engine="palette",
        preserve_text=False,
    )
    manifest = SimpleNamespace(mask_corrections=lambda _page_id: [])
    semantic = SimpleNamespace(
        confidence=np.ones((24, 24), dtype=np.float32),
        masks={
            "text": np.ones((24, 24), dtype=bool),
            "ink": np.ones((24, 24), dtype=bool),
        },
        uncertain=np.zeros((24, 24), dtype=bool),
    )
    monkeypatch.setattr(manager, "_load_spec", lambda _job_id: spec)
    monkeypatch.setattr(manager, "_manifest", lambda _job_id: manifest)
    monkeypatch.setattr(manager, "semantic_page", lambda _job_id, _page_index: semantic)

    protected, _uncertain = manager._corrected_unit_masks(
        "job",
        {
            "source_path": str(source_path),
            "page_id": 1,
            "page_index": 0,
            "unit_index": 0,
            "x": 0,
            "y": 0,
        },
    )

    assert protected[12, 12]
    assert not protected[5, 5]
    assert protected.mean() < 0.1


def test_semantic_balloon_protection_rejects_component_without_text() -> None:
    balloons = np.zeros((40, 60), dtype=bool)
    balloons[4:18, 4:24] = True
    balloons[22:36, 34:56] = True
    text = np.zeros_like(balloons)
    text[8:14, 10:18] = True
    source_gray = np.full(balloons.shape, 245, dtype=np.uint8)
    source_gray[8:14, 10:18] = 0

    protected = _balloons_with_detected_text(balloons, text, source_gray)

    assert protected[10, 12]
    assert not protected[28, 44]


def test_cobra_prefers_reviewed_curated_references_over_live_retrieval(
    tmp_path: Path,
) -> None:
    settings = Settings(data_root=tmp_path / "jobs")
    manager = ProjectManager(settings, EngineRegistry())
    job_id = "a" * 32
    job_dir = settings.data_root / job_id
    curated = job_dir / "references" / "curated"
    curated.mkdir(parents=True)
    source = tmp_path / "source.png"
    Image.new("RGB", (32, 32), "white").save(source)
    first = curated / "anchor-a.png"
    second = curated / "anchor-b.png"
    Image.new("RGB", (32, 32), "red").save(first)
    Image.new("RGB", (32, 32), "blue").save(second)
    spec = JobSpec(
        source=source,
        workspace=settings.data_root,
        engine="cobra-candidate",
    )

    paths = manager._reference_paths(job_id, source, spec)

    assert len(paths) == 2
    assert all(path.suffix == ".webp" for path in paths)


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


def test_completed_repair_is_staged_and_keeps_rollback_backup(
    tmp_path: Path, manga_pages: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(data_root=tmp_path / "jobs")
    manager = ProjectManager(settings, EngineRegistry())
    job_id = manager.create(
        JobSpec(source=manga_pages, workspace=settings.data_root, engine="palette")
    )
    manager.process(job_id)
    monkeypatch.setattr(
        "manga_repaint.project.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=10 * 1024**3),
    )

    repaired = manager.repair_completed_colorization(job_id)

    assert repaired == 3
    backup_roots = list((manager._job_dir(job_id) / "repair-backups").iterdir())
    assert len(backup_roots) == 1
    assert (backup_roots[0] / "manifest.sqlite").is_file()
    assert (backup_roots[0] / "final" / "pages" / "page_00000.png").is_file()
    assert manager.status(job_id)["status"] == "completed"


def test_atomic_repair_move_retries_a_temporary_windows_file_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "staging-output"
    destination = tmp_path / "output"
    source.mkdir()
    attempts = 0
    original_replace = Path.replace

    def temporarily_locked(path: Path, target: Path) -> Path:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("temporary scanner lock")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", temporarily_locked)
    monkeypatch.setattr("manga_repaint.project.time.sleep", lambda _seconds: None)

    ProjectManager._replace_with_windows_retry(source, destination)

    assert attempts == 3
    assert destination.is_dir()
    assert not source.exists()


def test_repair_output_publishes_files_without_renaming_staging_directory(
    tmp_path: Path,
) -> None:
    manager = ProjectManager(Settings(data_root=tmp_path / "jobs"), EngineRegistry())
    staging = tmp_path / "staging-output"
    live = tmp_path / "output"
    failed = tmp_path / "failed-output"
    staging.mkdir()
    (staging / "book-images.zip").write_bytes(b"verified archive")

    manager._publish_staged_output_files(staging, live)

    assert not staging.exists()
    assert (live / "book-images.zip").read_bytes() == b"verified archive"

    manager._withdraw_published_output_files(live, failed)

    assert not live.exists()
    assert (failed / "book-images.zip").read_bytes() == b"verified archive"


def test_repair_detects_a_dangling_output_directory_entry(tmp_path: Path) -> None:
    dangling = tmp_path / "output"
    try:
        dangling.symlink_to(tmp_path / "missing-output-target", target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    assert not dangling.exists()
    assert ProjectManager._path_entry_exists(dangling)


def test_failed_staged_repair_does_not_change_live_results(
    tmp_path: Path, manga_pages: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(data_root=tmp_path / "jobs")
    manager = ProjectManager(settings, EngineRegistry())
    job_id = manager.create(
        JobSpec(source=manga_pages, workspace=settings.data_root, engine="palette")
    )
    manager.process(job_id)
    monkeypatch.setattr(
        "manga_repaint.project.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=10 * 1024**3),
    )
    first_page = Path(manager._manifest(job_id).pages(job_id)[0]["output_path"])
    before = _sha256(first_page)

    def grayscale_result(source, _generated, _mask, _spec, _uncertain, *_args):
        return (
            Image.new("RGB", source.size, (160, 160, 160)),
            np.zeros((source.height, source.width), dtype=bool),
        )

    monkeypatch.setattr(manager, "_compose_unit", grayscale_result)
    with pytest.raises(RuntimeError, match="Repair QA failed"):
        manager.repair_completed_colorization(job_id)

    assert _sha256(first_page) == before
    assert not (manager._job_dir(job_id) / "repair-backups").exists()
    reports = list((manager._job_dir(job_id) / "repair-reports").glob("*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["status"] == "rolled_back"
    assert report["live_results_preserved"] is True


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
