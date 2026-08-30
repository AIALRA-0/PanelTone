from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw


@pytest.fixture
def manga_pages(tmp_path: Path) -> Path:
    source = tmp_path / "book"
    source.mkdir()
    for index in range(3):
        image = Image.new("RGB", (320, 420), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((18, 18, 302, 402), outline="black", width=5)
        draw.ellipse((70, 60, 250, 240), outline="black", width=4)
        draw.ellipse((115, 115, 130, 130), fill="black")
        draw.ellipse((190, 115, 205, 130), fill="black")
        draw.arc((120, 135, 200, 190), 10, 170, fill="black", width=3)
        draw.rounded_rectangle((60, 265, 260, 345), radius=26, outline="black", width=4)
        draw.text((98, 292), f"PAGE {index + 1}", fill="black")
        image.save(source / f"{index + 1:03d}.png")
    return source
