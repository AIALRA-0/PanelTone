from __future__ import annotations

import numpy as np
from PIL import Image

from manga_repaint.color import (
    composite_protected,
    composite_strict_colorization,
    is_already_colorized,
    lab_l,
    preserve_ink_overlay,
    preserve_luminance_lab,
    replace_masked,
    validated_colorization_protection,
)
from manga_repaint.masks import ink_edge_mask


def test_protected_pixels_are_exact() -> None:
    source = Image.new("RGB", (20, 20), (255, 255, 255))
    generated = Image.new("RGB", (20, 20), (30, 120, 220))
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    result = np.asarray(composite_protected(source, generated, mask))
    assert np.all(result[mask] == 255)
    assert np.all(result[~mask] == np.array([30, 120, 220]))


def test_replace_masked_preserves_base_outside_uncertain_region() -> None:
    base = Image.new("RGB", (20, 20), (20, 30, 40))
    replacement = Image.new("RGB", (20, 20), (220, 180, 90))
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True

    result = np.asarray(replace_masked(base, replacement, mask))

    assert np.all(result[mask] == np.array([220, 180, 90]))
    assert np.all(result[~mask] == np.array([20, 30, 40]))


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


def test_strict_colorization_does_not_overlay_unprotected_gray_edges() -> None:
    source_pixels = np.full((32, 32, 3), 255, dtype=np.uint8)
    source_pixels[:, 15:17] = 132
    source_pixels[:, 16] = 82
    source = Image.fromarray(source_pixels, mode="RGB")
    generated_pixels = np.full((32, 32, 3), (220, 80, 40), dtype=np.uint8)
    generated_pixels[:, 15:17] = (170, 60, 30)
    generated = Image.fromarray(generated_pixels, mode="RGB")

    result = np.asarray(
        composite_strict_colorization(
            source,
            generated,
            np.zeros((32, 32), dtype=bool),
            chroma_strength=1.0,
            ink_core_threshold=64,
        )
    )

    generated_pixels = np.asarray(generated)
    assert np.array_equal(result[:, 15:17], generated_pixels[:, 15:17])


def test_strict_colorization_keeps_protected_and_core_ink_exact() -> None:
    source_pixels = np.full((20, 20, 3), 245, dtype=np.uint8)
    source_pixels[2:5, :] = 4
    source_pixels[12:17, 5:15] = 210
    source = Image.fromarray(source_pixels, mode="RGB")
    generated = Image.new("RGB", source.size, (230, 95, 45))
    protected = np.zeros((20, 20), dtype=bool)
    protected[12:17, 5:15] = True

    result = np.asarray(composite_strict_colorization(source, generated, protected))

    assert np.array_equal(result[2:5], source_pixels[2:5])
    assert np.array_equal(result[protected], source_pixels[protected])
    assert np.all(result[8, 8] == np.array([230, 95, 45]))


def test_ink_edge_protection_keeps_antialias_without_protecting_gray_field() -> None:
    source_pixels = np.full((32, 32, 3), 220, dtype=np.uint8)
    source_pixels[14:18] = 120
    source_pixels[15:17] = 0
    source = Image.fromarray(source_pixels, mode="RGB")
    generated = Image.new("RGB", source.size, (220, 80, 40))

    mask = ink_edge_mask(source, core_threshold=64, edge_threshold=128)
    assert mask[14].any()
    assert not mask[0, 0]
    result = np.asarray(
        composite_strict_colorization(
            source,
            generated,
            np.zeros((32, 32), dtype=bool),
            ink_core_threshold=64,
            ink_edge_threshold=128,
        )
    )
    assert np.array_equal(result[14], source_pixels[14])
    assert np.all(result[0, 0] == np.array([220, 80, 40]))


def test_colorization_protection_rejects_white_mask_over_generated_skin() -> None:
    source_pixels = np.full((24, 24, 3), 255, dtype=np.uint8)
    source_pixels[4:8, 4:20] = 0
    source = Image.fromarray(source_pixels, mode="RGB")
    generated_pixels = np.full((24, 24, 3), (238, 174, 140), dtype=np.uint8)
    generated_pixels[14:20, 4:20] = (245, 245, 245)
    generated_pixels[16:18, 8:12] = (10, 10, 10)
    generated = Image.fromarray(generated_pixels, mode="RGB")
    requested = np.ones((24, 24), dtype=bool)

    validated = validated_colorization_protection(source, generated, requested)

    assert validated[5, 10]
    assert not validated[10, 10]
    assert validated[16, 10]


def test_already_colorized_source_is_detected() -> None:
    pixels = np.full((96, 96, 3), 255, dtype=np.uint8)
    pixels[16:80, 16:80] = np.array([60, 150, 220], dtype=np.uint8)
    source = Image.fromarray(pixels, mode="RGB")

    assert is_already_colorized(source)
    assert not is_already_colorized(Image.new("RGB", source.size, (220, 220, 220)))
