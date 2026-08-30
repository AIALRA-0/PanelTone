from __future__ import annotations

import zipfile
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from manga_repaint.api import create_app
from manga_repaint.config import Settings
from manga_repaint.engines import EngineRegistry


def test_api_create_run_and_download(tmp_path: Path, manga_pages: Path) -> None:
    app = create_app(Settings(data_root=tmp_path / "jobs"), EngineRegistry())
    client = TestClient(app)
    response = client.post("/api/jobs", json={"source": str(manga_pages), "engine": "palette"})
    assert response.status_code == 201
    job_id = response.json()["job_id"]

    output = app.state.manager.process(job_id)
    assert output.is_file()
    response = client.get(f"/api/jobs/{job_id}")
    assert response.json()["status"] == "completed"
    response = client.get(f"/api/jobs/{job_id}/download")
    assert response.status_code == 200
    assert response.headers["content-type"] in {"application/vnd.comicbook+zip", "application/zip"}


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
    assert response.status_code == 201
    assert len(response.json()["jobs"]) == 5
    public_jobs = client.get("/api/jobs").json()
    assert len(public_jobs) == 5
    assert all("source" not in job["spec"] for job in public_jobs)
    assert all("workspace" not in job["spec"] for job in public_jobs)


def test_archive_restore_and_permanent_delete(tmp_path: Path, manga_pages: Path) -> None:
    app = create_app(Settings(data_root=tmp_path / "jobs"), EngineRegistry())
    client = TestClient(app)
    created = client.post(
        "/api/jobs",
        json={"source": str(manga_pages), "engine": "palette", "display_name": "待清理书籍"},
    ).json()
    job_id = created["job_id"]

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
