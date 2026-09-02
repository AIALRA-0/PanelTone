from __future__ import annotations

import sys
import threading
import time
import types
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from manga_repaint import __version__
from manga_repaint.api import create_app
from manga_repaint.config import Settings
from manga_repaint.engines import EngineRegistry
from manga_repaint.models import JobStatus


class OfflineEngine:
    name = "offline"

    def generate(self, request):
        raise RuntimeError("offline")

    def healthcheck(self):
        return {"ok": False, "engine": self.name, "error": "模型服务未连接"}


def test_api_reports_package_version(tmp_path: Path) -> None:
    app = create_app(Settings(data_root=tmp_path / "jobs"), EngineRegistry())

    assert app.version == __version__


def wait_operation(client: TestClient, operation_url: str, timeout: int = 200) -> dict:
    operation: dict = {}
    for _ in range(timeout):
        operation = client.get(operation_url).json()
        if operation.get("status") in {"completed", "failed", "cancelled", "paused"}:
            return operation
        time.sleep(0.01)
    return operation


def test_api_create_run_and_download(tmp_path: Path, manga_pages: Path) -> None:
    app = create_app(Settings(data_root=tmp_path / "jobs"), EngineRegistry())
    with TestClient(app) as client:
        response = client.post(
            "/api/jobs?auto_start=false",
            json={"source": str(manga_pages), "engine": "palette"},
        )
        assert response.status_code == 202
        payload = response.json()
        assert payload["status"] == "ingesting"
        assert wait_operation(client, payload["progress_url"])["status"] == "completed"
        job_id = payload["job_id"]

        output = app.state.manager.process(job_id)
        assert output.is_file()
        response = client.get(f"/api/jobs/{job_id}")
        assert response.json()["status"] == "completed"
        response = client.get(f"/api/jobs/{job_id}/download")
        assert response.status_code == 200
        assert response.headers["content-type"] in {
            "application/vnd.comicbook+zip",
            "application/zip",
        }


@pytest.mark.parametrize(
    ("output_format", "content_type"),
    [
        ("pdf", "application/pdf"),
        ("images", "application/zip"),
        ("jpeg", "application/zip"),
        ("webp", "application/zip"),
    ],
)
def test_api_download_supports_all_image_exports(
    tmp_path: Path, manga_pages: Path, output_format: str, content_type: str
) -> None:
    app = create_app(Settings(data_root=tmp_path / "jobs"), EngineRegistry())
    with TestClient(app) as client:
        response = client.post(
            "/api/jobs?auto_start=false",
            json={
                "source": str(manga_pages),
                "engine": "palette",
                "output_format": output_format,
            },
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        assert wait_operation(client, response.json()["progress_url"])["status"] == "completed"
        app.state.manager.process(job_id)

        download = client.get(f"/api/jobs/{job_id}/download")
        assert download.status_code == 200
        assert download.headers["content-type"] == content_type


def test_api_presets_and_local_upload(tmp_path: Path) -> None:
    app = create_app(Settings(data_root=tmp_path / "jobs"), EngineRegistry())
    client = TestClient(app)
    presets = client.get("/api/presets")
    assert presets.status_code == 200
    assert len(presets.json()["colors"]) >= 4
    assert len(presets.json()["styles"]) >= 4

    payload = Image.new("RGB", (16, 16), "white")
    source = tmp_path / "upload.png"
    payload.save(source)
    with source.open("rb") as stream:
        response = client.post("/api/import", files={"file": ("upload.png", stream, "image/png")})
    assert response.status_code == 201
    descriptor = response.json()
    assert descriptor["source_id"]
    assert descriptor["name"] == "upload.png"
    assert descriptor["kind"] == "image"
    assert "path" not in descriptor
    assert Path(app.state.catalog.source(descriptor["source_id"])["path"]).is_file()


def test_uploaded_source_stays_allowed_with_explicit_local_roots(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    app = create_app(
        Settings(data_root=tmp_path / "jobs", allowed_roots=[allowed]),
        EngineRegistry(),
    )
    source = tmp_path / "uploaded-outside-allow-list.png"
    Image.new("RGB", (16, 16), "white").save(source)

    with TestClient(app) as client, source.open("rb") as stream:
        imported = client.post(
            "/api/import",
            files={"file": (source.name, stream, "image/png")},
        )
        assert imported.status_code == 201
        created = client.post(
            "/api/jobs",
            params={"auto_start": "false"},
            json={"source_id": imported.json()["source_id"], "engine": "palette"},
        )
        assert created.status_code == 202
        assert wait_operation(client, created.json()["progress_url"])["status"] == "completed"


def test_resumable_upload_keeps_offset_and_sha256(tmp_path: Path) -> None:
    app = create_app(Settings(data_root=tmp_path / "jobs"), EngineRegistry())
    client = TestClient(app)
    upload_id = "resume-upload-001"
    headers = {
        "X-Upload-ID": upload_id,
        "X-Upload-Offset": "0",
        "X-Upload-Total": "10",
    }
    first = client.post(
        "/api/import",
        headers=headers,
        files={"file": ("book.zip", b"01234", "application/zip")},
    )
    assert first.status_code == 202
    assert first.json()["uploaded_bytes"] == 5
    assert first.json()["complete"] is False
    status = client.get(f"/api/import/{upload_id}")
    assert status.status_code == 200
    assert status.json()["uploaded_bytes"] == 5

    second = client.post(
        "/api/import",
        headers={
            "X-Upload-ID": upload_id,
            "X-Upload-Offset": "5",
            "X-Upload-Total": "10",
        },
        files={"file": ("book.zip", b"56789", "application/zip")},
    )
    assert second.status_code == 201
    descriptor = second.json()
    assert descriptor["complete"] is True
    assert descriptor["uploaded_bytes"] == 10
    assert descriptor["sha256"]
    assert client.get(f"/api/import/{upload_id}").json()["complete"] is True


def test_resumable_upload_rejects_mismatched_identity_and_offset(tmp_path: Path) -> None:
    app = create_app(Settings(data_root=tmp_path / "jobs"), EngineRegistry())
    client = TestClient(app)
    headers = {
        "X-Upload-ID": "resume-upload-002",
        "X-Upload-Offset": "0",
        "X-Upload-Total": "6",
    }
    assert client.post(
        "/api/import",
        headers=headers,
        files={"file": ("book.zip", b"123", "application/zip")},
    ).status_code == 202
    mismatched = client.post(
        "/api/import",
        headers=headers,
        data={"client_upload_id": "resume-upload-other"},
        files={"file": ("book.zip", b"456", "application/zip")},
    )
    assert mismatched.status_code == 400
    inconsistent = client.post(
        "/api/import",
        headers={**headers, "X-Upload-Offset": "0"},
        files={"file": ("book.zip", b"456", "application/zip")},
    )
    assert inconsistent.status_code == 409
    assert inconsistent.json()["expected_offset"] == 3


def test_batch_groups_images_and_keeps_books_separate(tmp_path: Path) -> None:
    app = create_app(Settings(data_root=tmp_path / "jobs"), EngineRegistry())
    client = TestClient(app)
    source_ids: list[str] = []

    image_paths: list[Path] = []
    for index in range(20):
        path = tmp_path / f"page_{20 - index}.png"
        Image.new("RGB", (24, 32), "white").save(path)
        image_paths.append(path)

    books: list[Path] = []
    for index in range(3):
        archive_path = tmp_path / f"book_{index}.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.write(image_paths[0], "001.png")
        books.append(archive_path)
    pdf_path = tmp_path / "book.pdf"
    Image.new("RGB", (24, 32), "white").save(pdf_path, "PDF")
    books.append(pdf_path)

    for path in [*image_paths, *books]:
        with path.open("rb") as stream:
            response = client.post(
                "/api/import",
                files={"file": (path.name, stream, "application/octet-stream")},
            )
        assert response.status_code == 201
        source_ids.append(response.json()["source_id"])

    response = client.post(
        "/api/jobs/batch",
        json={
            "source_ids": source_ids,
            "image_order": source_ids[:20],
            "image_book_name": "合成测试书",
            "engine": "palette",
        },
    )
    assert response.status_code == 202
    assert len(response.json()["jobs"]) == 5
    public_jobs = client.get("/api/jobs").json()
    assert len(public_jobs) == 5
    assert all("source" not in job["spec"] for job in public_jobs)
    assert all("workspace" not in job["spec"] for job in public_jobs)


def test_batch_rejects_duplicate_image_order_and_invalid_event_cursor(tmp_path: Path) -> None:
    app = create_app(Settings(data_root=tmp_path / "jobs"), EngineRegistry())
    client = TestClient(app)
    source_ids: list[str] = []
    for name in ("page_1.png", "page_2.png"):
        path = tmp_path / name
        Image.new("RGB", (16, 16), "white").save(path)
        with path.open("rb") as stream:
            response = client.post(
                "/api/import",
                files={"file": (path.name, stream, "image/png")},
            )
        source_ids.append(response.json()["source_id"])

    response = client.post(
        "/api/jobs/batch",
        json={
            "source_ids": source_ids,
            "image_order": [source_ids[0], source_ids[0], source_ids[1]],
            "engine": "palette",
        },
    )
    assert response.status_code == 400
    assert client.get("/api/events", headers={"Last-Event-ID": "not-a-number"}).status_code == 400


def test_preview_endpoint_recovers_missed_live_preview(
    tmp_path: Path, manga_pages: Path, monkeypatch
) -> None:
    app = create_app(
        Settings(data_root=tmp_path / "jobs", qa_line_f1_min=1.1), EngineRegistry()
    )
    client = TestClient(app)
    job_id = client.post(
        "/api/jobs", json={"source": str(manga_pages), "engine": "palette"}
    ).json()["job_id"]

    app.state.manager.process(job_id)
    preview_path = (
        tmp_path / "jobs" / job_id / "preview" / "pages" / "page_00000.png"
    )
    assert preview_path.is_file()
    # Simulate a browser reconnect after the one-shot page_preview_ready event
    # was missed.  The API should rebuild the inspectable preview from the
    # already generated unit output without turning it into an export result.
    preview_path.unlink()

    def fail_if_listing_writes(*_args, **_kwargs):
        raise AssertionError("page listing must not update the manifest")

    monkeypatch.setattr(
        type(app.state.manager._manifest(job_id)),
        "set_page_asset_revision",
        fail_if_listing_writes,
    )
    listing = client.get(f"/api/jobs/{job_id}/pages")
    assert listing.status_code == 200
    assert listing.json()[0]["preview_url"] is None

    monkeypatch.undo()
    response = client.get(f"/api/jobs/{job_id}/pages/0/preview")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    assert preview_path.is_file()


def test_invalid_job_paths_return_not_found(tmp_path: Path) -> None:
    app = create_app(Settings(data_root=tmp_path / "jobs"), EngineRegistry())
    client = TestClient(app)
    requests = [
        ("GET", "/api/jobs/not-valid!/pages"),
        ("GET", "/api/jobs/not-valid!/progress"),
        ("GET", "/api/jobs/not-valid!/download"),
        ("POST", "/api/jobs/not-valid!/archive"),
    ]
    for method, path in requests:
        assert client.request(method, path).status_code == 404


def test_model_download_requests_are_deduplicated(tmp_path: Path, monkeypatch) -> None:
    app = create_app(
        Settings(data_root=tmp_path / "jobs", model_root=tmp_path / "models"), EngineRegistry()
    )
    client = TestClient(app)
    started = threading.Event()
    release = threading.Event()

    def fake_snapshot_download(**_kwargs):
        started.set()
        release.wait(timeout=2)
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir(exist_ok=True)
        return str(snapshot)

    fake_huggingface = types.ModuleType("huggingface_hub")
    fake_huggingface.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_huggingface)
    try:
        first = client.post("/api/models/flux2-klein-4b/download")
        assert first.status_code == 202
        assert started.wait(timeout=2)
        second = client.post("/api/models/flux2-klein-4b/download")
        assert second.status_code == 202
        assert second.json()["status"] == "already_downloading"
    finally:
        release.set()
    for _ in range(100):
        if client.get("/api/models").json()[0]["status"] == "installed":
            break
        time.sleep(0.01)
    assert client.get("/api/models").json()[0]["status"] == "installed"


def test_model_release_uses_connected_engine(tmp_path: Path) -> None:
    class ReleasableEngine:
        name = "releasable"

        def healthcheck(self):
            return {
                "ok": True,
                "engine": self.name,
                "model_id": "black-forest-labs/FLUX.2-klein-4B",
                "state": "ready",
                "supports_release": True,
            }

        def release(self):
            return {"status": "released", "active": False, "state": "idle"}

    app = create_app(
        Settings(data_root=tmp_path / "jobs"),
        EngineRegistry({"releasable": ReleasableEngine()}),
    )
    response = TestClient(app).post("/api/models/flux2-klein-4b/release")

    assert response.status_code == 202
    assert response.json()["status"] == "released"


def test_model_release_explains_legacy_service_control_gap(tmp_path: Path) -> None:
    class LegacyEngine:
        name = "legacy"

        def healthcheck(self):
            return {
                "ok": True,
                "engine": self.name,
                "model_id": "black-forest-labs/FLUX.2-klein-4B",
                "state": "ready",
            }

        def release(self):
            raise AssertionError("legacy service must not receive an unsupported request")

    app = create_app(
        Settings(data_root=tmp_path / "jobs"),
        EngineRegistry({"legacy": LegacyEngine()}),
    )
    response = TestClient(app).post("/api/models/flux2-klein-4b/release")

    assert response.status_code == 409
    assert "维护重启" in response.json()["detail"]


def test_archive_restore_and_permanent_delete(tmp_path: Path, manga_pages: Path) -> None:
    app = create_app(Settings(data_root=tmp_path / "jobs"), EngineRegistry())
    client = TestClient(app)
    created = client.post(
        "/api/jobs",
        params={"auto_start": "false"},
        json={"source": str(manga_pages), "engine": "palette", "display_name": "待清理书籍"},
    ).json()
    job_id = created["job_id"]

    assert wait_operation(client, created["progress_url"])["status"] == "completed"

    assert client.post(f"/api/jobs/{job_id}/archive").status_code == 200
    archived = client.get("/api/jobs?include_archived=true").json()
    assert next(job for job in archived if job["id"] == job_id)["status"] == "archived"
    assert client.post(f"/api/jobs/{job_id}/restore").status_code == 200
    assert client.post(f"/api/jobs/{job_id}/archive").status_code == 200
    response = client.request(
        "DELETE",
        f"/api/jobs/{job_id}",
        json={"confirmation": "永久删除"},
    )
    assert response.status_code == 200
    assert not (tmp_path / "jobs" / job_id).exists()


def test_library_tree_nested_move_and_folder_archive_restore(
    tmp_path: Path, manga_pages: Path
) -> None:
    app = create_app(Settings(data_root=tmp_path / "jobs"), EngineRegistry())
    with TestClient(app) as client:
        created = [
            client.post(
                "/api/jobs",
                params={"auto_start": "false"},
                json={
                    "source": str(manga_pages),
                    "engine": "palette",
                    "display_name": f"目录任务 {index}",
                },
            ).json()
            for index in (1, 2)
        ]
        for shell in created:
            assert wait_operation(client, shell["progress_url"])["status"] == "completed"
        parent = client.post("/api/folders", json={"name": "系列"}).json()
        child = client.post(
            "/api/folders", json={"name": "卷一", "parent_id": parent["id"]}
        ).json()
        sibling = client.post(
            "/api/folders", json={"name": "卷二", "parent_id": parent["id"]}
        ).json()
        assert client.post(
            "/api/library/reorder",
            json={"parent_id": parent["id"], "folder_ids": [sibling["id"], child["id"]]},
        ).status_code == 200
        assert client.post(
            f"/api/library/jobs/{created[0]['job_id']}/move",
            json={"folder_id": child["id"]},
        ).status_code == 200
        tree = client.get("/api/library/tree").json()
        assert tree["folders"][0]["name"] == "系列"
        assert tree["folders"][0]["children"][1]["jobs"][0]["id"] == created[0]["job_id"]
        assert client.get("/api/library/search?q=卷一").json()["folders"][0]["name"] == "系列"
        assert client.post(f"/api/folders/{parent['id']}/archive").status_code == 200
        archived_tree = client.get("/api/library/tree?include_archived=true").json()
        assert archived_tree["folders"][0]["archived_at"]
        assert archived_tree["folders"][0]["children"][1]["jobs"][0]["status"] == "archived"
        assert client.post(f"/api/folders/{parent['id']}/restore").status_code == 200
        restored_tree = client.get("/api/library/tree").json()
        assert restored_tree["folders"][0]["children"][1]["jobs"][0]["status"] != "archived"


def test_relative_upload_path_rejects_drive_relative_traversal(tmp_path: Path) -> None:
    app = create_app(Settings(data_root=tmp_path / "jobs"), EngineRegistry())
    client = TestClient(app)
    response = client.post(
        "/api/import",
        data={"relative_path": "C:folder/page.png"},
        files={"file": ("page.png", b"not-an-image", "image/png")},
    )
    assert response.status_code == 400


def test_activity_log_is_readable_and_hides_internal_paths(
    tmp_path: Path, manga_pages: Path
) -> None:
    app = create_app(Settings(data_root=tmp_path / "jobs"), EngineRegistry())
    client = TestClient(app)
    created = client.post(
        "/api/jobs",
        json={"source": str(manga_pages), "engine": "palette", "display_name": "日志测试"},
    ).json()

    response = client.get(f"/api/logs?job_id={created['job_id']}&limit=20")

    assert response.status_code == 200
    entries = response.json()
    assert entries
    assert all(entry["message"] for entry in entries)
    assert str(tmp_path) not in response.text


def test_public_job_errors_hide_local_data_paths(tmp_path: Path, manga_pages: Path) -> None:
    app = create_app(Settings(data_root=tmp_path / "jobs"), EngineRegistry())
    client = TestClient(app)
    job_id = client.post("/api/jobs", json={"source": str(manga_pages)}).json()["job_id"]
    app.state.manager._manifest(job_id).set_job_status(
        job_id, JobStatus.FAILED, str(tmp_path / "private.log")
    )
    response = client.get(f"/api/jobs/{job_id}")
    assert response.status_code == 200
    assert str(tmp_path) not in response.text
    assert "<本地路径>" in response.json()["error"]


def test_event_stream_hides_local_data_paths(tmp_path: Path, manga_pages: Path) -> None:
    app = create_app(Settings(data_root=tmp_path / "jobs"), EngineRegistry())
    client = TestClient(app)
    job_id = client.post("/api/jobs", json={"source": str(manga_pages)}).json()["job_id"]
    cursor = app.state.catalog.latest_event_id()
    app.state.events.publish(
        "job_error",
        {"message": f"failed at {tmp_path / 'private.log'}", "nested": {"path": str(tmp_path)}},
        job_id,
    )

    stream = app.state.events.stream(cursor)
    next(stream)
    event = next(stream)

    assert str(tmp_path) not in event
    assert "<本地路径>" in event


def test_waiting_model_can_be_paused_or_cancelled(
    tmp_path: Path, manga_pages: Path
) -> None:
    app = create_app(
        Settings(data_root=tmp_path / "jobs"),
        EngineRegistry({"offline": OfflineEngine()}),
    )
    client = TestClient(app)

    paused = client.post(
        "/api/jobs",
        params={"auto_start": "false"},
        json={"source": str(manga_pages), "engine": "offline", "display_name": "等待暂停"},
    ).json()
    paused_id = paused["job_id"]
    assert wait_operation(client, paused["progress_url"])["status"] == "completed"
    app.state.manager.process(paused_id)
    assert client.get(f"/api/jobs/{paused_id}").json()["status"] == "waiting_model"
    assert client.post(f"/api/jobs/{paused_id}/pause").json()["status"] == "paused"

    cancelled = client.post(
        "/api/jobs",
        params={"auto_start": "false"},
        json={"source": str(manga_pages), "engine": "offline", "display_name": "等待取消"},
    ).json()
    cancelled_id = cancelled["job_id"]
    assert wait_operation(client, cancelled["progress_url"])["status"] == "completed"
    app.state.manager.process(cancelled_id)
    assert client.post(f"/api/jobs/{cancelled_id}/cancel").json()["status"] == "cancelled"


def test_queued_jobs_are_restored_after_app_restart(tmp_path: Path, manga_pages: Path) -> None:
    settings = Settings(data_root=tmp_path / "jobs")
    first_app = create_app(settings, EngineRegistry({"offline": OfflineEngine()}))
    first_client = TestClient(first_app)
    job_id = first_client.post(
        "/api/jobs",
        json={"source": str(manga_pages), "engine": "offline", "display_name": "重启恢复"},
    ).json()["job_id"]
    first_app.state.manager.queue(job_id)
    first_app.state.scheduler.stop()

    second_app = create_app(settings, EngineRegistry({"offline": OfflineEngine()}))
    second_client = TestClient(second_app)
    try:
        for _ in range(100):
            if second_client.get(f"/api/jobs/{job_id}").json()["status"] == "waiting_model":
                break
            time.sleep(0.01)
        assert second_client.get(f"/api/jobs/{job_id}").json()["status"] == "waiting_model"
    finally:
        second_app.state.scheduler.stop()


def test_async_job_returns_shell_and_reports_ingest_progress(
    tmp_path: Path, manga_pages: Path
) -> None:
    app = create_app(Settings(data_root=tmp_path / "jobs"), EngineRegistry())
    with TestClient(app) as client:
        response = client.post(
            "/api/jobs?async=true",
            json={"source": str(manga_pages), "engine": "palette", "display_name": "异步建书"},
        )
        assert response.status_code == 202
        payload = response.json()
        assert payload["operation_id"]
        assert payload["status"] == "ingesting"
        assert payload["progress_url"].endswith(payload["operation_id"])

        operation = None
        for _ in range(200):
            operation = client.get(payload["progress_url"]).json()
            if operation["status"] in {"completed", "failed", "cancelled", "paused"}:
                break
            time.sleep(0.01)
        assert operation is not None
        assert operation["status"] == "completed"
        assert operation["discovered_pages"] == 3
        job = client.get(f"/api/jobs/{payload['job_id']}").json()
        assert job["status"] in {"queued", "running", "completed"}
        assert job["page_count"] == 3


def test_start_during_async_ingest_requests_auto_start(
    tmp_path: Path, manga_pages: Path, monkeypatch
) -> None:
    app = create_app(Settings(data_root=tmp_path / "jobs"), EngineRegistry())
    entered = threading.Event()
    release = threading.Event()
    original_ingest = app.state.manager.ingest

    def delayed_ingest(job_id: str) -> str:
        entered.set()
        release.wait(timeout=2)
        return original_ingest(job_id)

    monkeypatch.setattr(app.state.manager, "ingest", delayed_ingest)
    with TestClient(app) as client:
        response = client.post(
            "/api/jobs?async=true&auto_start=false",
            json={"source": str(manga_pages), "engine": "palette", "display_name": "延迟建书"},
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        operation_id = response.json()["operation_id"]
        assert entered.wait(timeout=2)
        started = client.post(f"/api/jobs/{job_id}/start")
        assert started.status_code == 202
        assert "自动开始" in started.json()["message"]
        release.set()
        for _ in range(200):
            operation = client.get(f"/api/operations/{operation_id}").json()
            if operation["status"] in {"completed", "failed", "cancelled", "paused"}:
                break
            time.sleep(0.01)
        assert operation["status"] == "completed"
        for _ in range(200):
            status = client.get(f"/api/jobs/{job_id}").json()["status"]
            if status in {"running", "completed", "needs_attention", "failed"}:
                break
            time.sleep(0.01)
        assert status in {"running", "completed", "needs_attention", "failed"}
