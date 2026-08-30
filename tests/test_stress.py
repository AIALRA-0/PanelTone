from __future__ import annotations

import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

from manga_repaint.config import Settings
from manga_repaint.engines import EngineRegistry
from manga_repaint.models import JobSpec
from manga_repaint.project import ProjectManager


def test_three_hundred_page_book_keeps_order_and_count(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for index in range(300):
        image = Image.new("RGB", (32, 40), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((2, 2, 29, 37), outline="black", width=2)
        image.save(source / f"page_{index + 1:04d}.png")

    settings = Settings(
        data_root=tmp_path / "jobs",
        qa_line_f1_min=0.0,
        qa_luminance_mae_max=255.0,
    )
    manager = ProjectManager(settings, EngineRegistry())
    job_id = manager.create(JobSpec(source=source, workspace=settings.data_root, engine="palette"))
    output = manager.process(job_id)

    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
    assert len(names) == 300
    assert names[0] == "page_00000.png"
    assert names[-1] == "page_00299.png"
    assert len(set(names)) == 300
