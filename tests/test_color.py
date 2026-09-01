from __future__ import annotations

import numpy as np
from PIL import Image

from manga_repaint.color import (
    composite_protected,
    composite_strict_colorization,
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


def test_ink_overlay_preserves_near_black_print_pixels_exactly() -> None:
    source = Image.new("RGB", (12, 12), (255, 255, 255))
    pixels = np.asarray(source).copy()
    pixels[4:8, 4:8] = np.array([7, 7, 7])
    source = Image.fromarray(pixels, mode="RGB")
    generated = Image.new("RGB", source.size, (240, 160, 90))

    result = np.asarray(preserve_ink_overlay(source, generated))

    assert np.array_equal(result[4:8, 4:8], pixels[4:8, 4:8])


def test_strict_colorization_does_not_replace_screentones_with_grayscale() -> None:
    source_pixels = np.full((24, 24, 3), 180, dtype=np.uint8)
    source_pixels[2:5, 2:22] = 0
    source = Image.fromarray(source_pixels, mode="RGB")
    generated = Image.new("RGB", source.size, (220, 80, 40))
    protected = np.zeros((24, 24), dtype=bool)
    protected[16:20, 4:20] = True

    result = np.asarray(
        composite_strict_colorization(source, generated, protected, chroma_strength=1.0)
    )

    assert np.array_equal(result[2:5, 2:22], source_pixels[2:5, 2:22])
    assert np.array_equal(result[16:20, 4:20], source_pixels[16:20, 4:20])
    assert not np.array_equal(result[10, 10], source_pixels[10, 10])
    assert int(result[10, 10].max()) - int(result[10, 10].min()) > 20


def test_strict_colorization_preserves_antialiased_boundary_edges() -> None:
    source_pixels = np.full((32, 32, 3), 255, dtype=np.uint8)
    source_pixels[:, 15:17] = 132
    source_pixels[:, 16] = 82
    source = Image.fromarray(source_pixels, mode="RGB")
    generated = Image.new("RGB", source.size, (220, 80, 40))

    result = np.asarray(
        composite_strict_colorization(
            source,
            generated,
            np.zeros((32, 32), dtype=bool),
            chroma_strength=1.0,
            ink_core_threshold=64,
        )
    )

    assert np.array_equal(result[:, 15:17], source_pixels[:, 15:17])
