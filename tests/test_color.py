from __future__ import annotations

import numpy as np
from PIL import Image

from manga_repaint.color import (
    composite_protected,
    lab_l,
    preserve_ink_overlay,
    preserve_luminance_lab,
)


def test_protected_pixels_are_exact() -> None:
    source = Image.new("RGB", (20, 20), (255, 255, 255))
    generated = Image.new("RGB", (20, 20), (30, 120, 220))
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    result = np.asarray(composite_protected(source, generated, mask))
    assert np.all(result[mask] == 255)
    assert np.all(result[~mask] == np.array([30, 120, 220]))


def test_lab_composition_keeps_luminance_close() -> None:
    source = Image.linear_gradient("L").resize((128, 128)).convert("RGB")
    generated = Image.new("RGB", source.size, (220, 40, 90))
    result = preserve_luminance_lab(source, generated)
    source_luma = lab_l(source)
    result_luma = lab_l(result)
    assert np.abs(source_luma - result_luma).mean() < 2.0


def test_ink_overlay_preserves_black_without_white_halo() -> None:
    source = Image.new("L", (24, 24), 255)
    pixels = np.asarray(source).copy()
    pixels[11:13, 4:20] = 0
    source = Image.fromarray(pixels, mode="L").convert("RGB")
    generated = Image.new("RGB", source.size, (220, 120, 70))
    result = np.asarray(preserve_ink_overlay(source, generated))
    assert np.all(result[11:13, 4:20] == 0)
    assert np.all(result[0, 0] == np.array([220, 120, 70]))
