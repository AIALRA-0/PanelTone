from __future__ import annotations

from PIL import Image, ImageDraw

from manga_repaint.masks import text_region_mask


def test_text_region_mask_selects_glyphs_without_filling_white_box() -> None:
    image = Image.new("L", (240, 180), 210)
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 20, 130, 160), fill=255, outline=0, width=2)
    for y in range(36, 142, 22):
        draw.rectangle((72, y, 86, y + 13), fill=0)
    mask = text_region_mask(image.convert("RGB"))
    assert mask[42, 78]
    assert not mask[28, 52]
    assert float(mask.mean()) < 0.2
