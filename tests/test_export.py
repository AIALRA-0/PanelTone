from __future__ import annotations

import zipfile
from pathlib import Path

from PIL import Image

from manga_repaint.export import export_book


def test_jpeg_and_webp_exports_are_readable_archives(tmp_path: Path) -> None:
    pages = []
    for index, color in enumerate(((240, 80, 40), (40, 100, 220))):
        page = tmp_path / f"source-{index}.png"
        Image.new("RGB", (32, 48), color).save(page)
        pages.append(page)

    jpeg = export_book(pages, tmp_path / "jpeg", "jpeg")
    webp = export_book(pages, tmp_path / "webp", "webp")

    with zipfile.ZipFile(jpeg) as archive:
        assert archive.namelist() == ["page_00000.jpg", "page_00001.jpg"]
        with Image.open(archive.open("page_00000.jpg")) as image:
            assert image.size == (32, 48)
    with zipfile.ZipFile(webp) as archive:
        assert archive.namelist() == ["page_00000.webp", "page_00001.webp"]
        with Image.open(archive.open("page_00001.webp")) as image:
            assert image.size == (32, 48)


def test_png_images_export_is_a_downloadable_archive(tmp_path: Path) -> None:
    pages = []
    for index, color in enumerate(((240, 80, 40), (40, 100, 220))):
        page = tmp_path / f"source-{index}.png"
        Image.new("RGB", (32, 48), color).save(page)
        pages.append(page)

    output = export_book(pages, tmp_path / "images", "images")

    assert output.name == "book-images.zip"
    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == ["page_00000.png", "page_00001.png"]
        with Image.open(archive.open("page_00000.png")) as image:
            assert image.format == "PNG"
            assert image.size == (32, 48)
