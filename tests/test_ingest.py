from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from PIL import Image

from manga_repaint.ingest import ingest_book


def test_cbz_uses_natural_page_order(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for name, value in (("10.png", 10), ("2.png", 2), ("1.png", 1)):
        Image.new("L", (8, 8), value).save(inputs / name)
    archive_path = tmp_path / "book.cbz"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for path in inputs.iterdir():
            archive.write(path, path.name)
    outputs = ingest_book(archive_path, tmp_path / "pages")
    pixels = [Image.open(path).convert("L").getpixel((0, 0)) for path in outputs]
    assert pixels == [1, 2, 10]


def test_cbz_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.cbz"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.png", b"not-an-image")
    with pytest.raises(ValueError, match="escapes extraction root"):
        ingest_book(archive_path, tmp_path / "pages")


def test_cbz_rejects_nested_archives(tmp_path: Path) -> None:
    archive_path = tmp_path / "nested.cbz"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("inside.zip", b"archive")
    with pytest.raises(ValueError, match="Nested archives"):
        ingest_book(archive_path, tmp_path / "pages")


def test_cbz_rejects_abnormal_compression_ratio(tmp_path: Path) -> None:
    archive_path = tmp_path / "ratio.cbz"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("page.png", b"0" * 100_000)
    with pytest.raises(ValueError, match="compression ratio"):
        ingest_book(archive_path, tmp_path / "pages", max_archive_ratio=2)
