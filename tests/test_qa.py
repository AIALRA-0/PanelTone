from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from manga_repaint.color import composite_geometry_locked_colorization
from manga_repaint.qa import evaluate


def test_color_qa_rejects_page_that_drops_generated_color() -> None:
    source = Image.new("RGB", (120, 120), (220, 220, 220))
    generated = Image.new("RGB", source.size, (220, 80, 40))
    result = Image.new("RGB", source.size, (160, 160, 160))
    protected = np.zeros((120, 120), dtype=bool)

    qa = evaluate(
        source,
        result,
        protected,
        generated=generated,
        line_f1_min=0.0,
        luminance_mae_max=255.0,
        pure_black_preservation_min=0.0,
    )

    assert not qa.passed
    assert qa.color_retention_ratio == 0.0
    assert qa.color_dropout_tiles == 36
    assert "color_retention_below_threshold" in qa.reasons
    assert "regional_color_dropout" in qa.reasons


def test_color_qa_accepts_full_color_with_exact_protection() -> None:
    source_pixels = np.full((120, 120, 3), 220, dtype=np.uint8)
    source_pixels[10:30, 10:110] = 0
    source = Image.fromarray(source_pixels, mode="RGB")
    generated = Image.new("RGB", source.size, (220, 80, 40))
    result_pixels = np.asarray(generated).copy()
    protected = np.zeros((120, 120), dtype=bool)
    protected[10:30, 10:110] = True
    result_pixels[protected] = source_pixels[protected]
    result = Image.fromarray(result_pixels, mode="RGB")

    qa = evaluate(
        source,
        result,
        protected,
        generated=generated,
        line_f1_min=0.0,
        luminance_mae_max=255.0,
    )

    assert qa.passed
    assert qa.protected_pixel_diff == 0
    assert qa.color_retention_ratio == 1.0
    assert qa.color_dropout_tiles == 0


def test_geometry_qa_rejects_twelve_pixel_structure_shift() -> None:
    source_pixels = np.full((128, 128, 3), 238, dtype=np.uint8)
    source_pixels[52:56, 20:108] = 0
    source = Image.fromarray(source_pixels, mode="RGB")
    shifted_pixels = np.full_like(source_pixels, 238)
    shifted_pixels[64:68, 20:108] = 0
    shifted = Image.fromarray(shifted_pixels, mode="RGB")

    qa = evaluate(
        source,
        shifted,
        np.zeros((128, 128), dtype=bool),
        line_f1_min=0.0,
        luminance_mae_max=255.0,
        pure_black_preservation_min=0.0,
        generated=shifted,
        geometry_locked=True,
    )

    assert not qa.passed
    assert qa.source_edge_recall < 0.995
    assert "source_edge_recall_below_threshold" in qa.reasons


def test_geometry_locked_composition_does_not_import_shifted_edges() -> None:
    source_pixels = np.full((96, 96, 3), 232, dtype=np.uint8)
    source_pixels[38:42, 14:82] = 0
    source = Image.fromarray(source_pixels, mode="RGB")
    generated_pixels = np.full_like(source_pixels, (226, 88, 56))
    generated_pixels[50:54, 14:82] = 0
    generated = Image.fromarray(generated_pixels, mode="RGB")
    result = composite_geometry_locked_colorization(
        source,
        generated,
        np.zeros((96, 96), dtype=bool),
    )

    source_edges = cv2.Canny(np.asarray(source.convert("L")), 60, 150) > 0
    result_edges = cv2.Canny(np.asarray(result.convert("L")), 60, 150) > 0
    assert int(result_edges[50:54, 14:82].sum()) == 0
    assert int(np.logical_and(source_edges, result_edges).sum()) > 0
