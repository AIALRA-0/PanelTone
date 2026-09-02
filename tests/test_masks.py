from __future__ import annotations

from PIL import Image, ImageDraw

from manga_repaint.masks import bubble_mask, text_region_mask


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


def test_bubble_mask_rejects_enclosed_white_art_without_text() -> None:
    image = Image.new("L", (240, 180), 170)
    draw = ImageDraw.Draw(image)
    draw.ellipse((50, 30, 190, 150), fill=255, outline=0, width=4)

    mask = bubble_mask(image.convert("RGB"))

    assert not mask[90, 120]


def test_bubble_mask_rejects_face_like_region_with_one_dark_mark() -> None:
    image = Image.new("L", (240, 180), 170)
    draw = ImageDraw.Draw(image)
    draw.ellipse((50, 30, 190, 150), fill=255, outline=0, width=4)
    draw.rectangle((112, 85, 123, 95), fill=0)

    mask = bubble_mask(image.convert("RGB"))

    assert not mask[65, 120]


def test_bubble_mask_keeps_white_region_that_contains_detected_glyphs() -> None:
    image = Image.new("L", (240, 180), 170)
    draw = ImageDraw.Draw(image)
    draw.ellipse((70, 45, 170, 135), fill=255, outline=0, width=3)
    for y in range(62, 120, 18):
        draw.rectangle((112, y, 123, y + 10), fill=0)

    mask = bubble_mask(image.convert("RGB"))

    assert mask[90, 85]
