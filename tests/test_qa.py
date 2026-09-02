from __future__ import annotations

import numpy as np
from PIL import Image

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
