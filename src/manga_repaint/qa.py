from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .color import lab_l
from .models import QAResult


def _edge_map(image: Image.Image) -> np.ndarray:
    luminance = np.clip(np.rint(lab_l(image) * 2.55), 0, 255).astype(np.uint8)
    return cv2.Canny(luminance, 80, 160) > 0


def _f1(reference: np.ndarray, candidate: np.ndarray) -> float:
    if not reference.any() and not candidate.any():
        return 1.0
    # Use a scale-aware tolerance: a 4-6 pixel antialias/resize offset on a
    # full-resolution page is visually the same edge, while small unit-test
    # images retain a one-pixel tolerance.
    radius = max(1, int(round(min(reference.shape) * 0.01)))
    kernel_size = radius * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    reference_neighborhood = cv2.dilate(reference.astype(np.uint8), kernel).astype(bool)
    candidate_neighborhood = cv2.dilate(candidate.astype(np.uint8), kernel).astype(bool)
    matched_reference = np.logical_and(reference, candidate_neighborhood).sum()
    matched_candidate = np.logical_and(candidate, reference_neighborhood).sum()
    recall = float(matched_reference / max(1, reference.sum()))
    precision = float(matched_candidate / max(1, candidate.sum()))
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _color_coverage(rgb: np.ndarray, roi: np.ndarray) -> float:
    if not roi.any():
        return 0.0
    maximum = rgb.max(axis=-1).astype(np.float32)
    chroma = maximum - rgb.min(axis=-1).astype(np.float32)
    saturation = chroma / np.maximum(maximum, 1.0)
    # Relative saturation recognizes dark blue/brown ink as colour without
    # treating neutral gray or black screentone as coloured.
    colourful = np.logical_and(chroma >= 6.0, saturation >= 0.08)
    return float(colourful[roi].mean())


def _color_dropout_tiles(
    generated_rgb: np.ndarray,
    result_rgb: np.ndarray,
    roi: np.ndarray,
    grid_size: int = 6,
) -> int:
    height, width = roi.shape
    dropouts = 0
    for row in range(grid_size):
        y0, y1 = height * row // grid_size, height * (row + 1) // grid_size
        for column in range(grid_size):
            x0, x1 = width * column // grid_size, width * (column + 1) // grid_size
            tile_roi = roi[y0:y1, x0:x1]
            if tile_roi.sum() < 64:
                continue
            generated_coverage = _color_coverage(
                generated_rgb[y0:y1, x0:x1], tile_roi
            )
            result_coverage = _color_coverage(result_rgb[y0:y1, x0:x1], tile_roi)
            if generated_coverage >= 0.25 and result_coverage < max(
                0.10, generated_coverage * 0.50
            ):
                dropouts += 1
    return dropouts


def evaluate(
    source: Image.Image,
    result: Image.Image,
    protected_mask: np.ndarray,
    line_f1_min: float = 0.98,
    luminance_mae_max: float = 2.0,
    pure_black_preservation_min: float = 0.999,
    generated: Image.Image | None = None,
    color_retention_min: float = 0.70,
    color_dropout_tiles_max: int = 0,
) -> QAResult:
    dimension_match = source.size == result.size
    if not dimension_match:
        return QAResult(False, -1, 0.0, float("inf"), False, ["dimension_mismatch"])

    source_rgb = np.asarray(source.convert("RGB")).astype(np.int16)
    result_rgb = np.asarray(result.convert("RGB")).astype(np.int16)
    if protected_mask.any():
        protected_diff = int(np.abs(source_rgb[protected_mask] - result_rgb[protected_mask]).max())
    else:
        protected_diff = 0

    source_luma = lab_l(source)
    result_luma = lab_l(result)
    luminance_mae = float(np.abs(source_luma - result_luma).mean())
    source_edges = _edge_map(source)
    result_edges = _edge_map(result)
    if protected_mask.any():
        roi = cv2.dilate(protected_mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        source_edges = np.logical_and(source_edges, roi)
        result_edges = np.logical_and(result_edges, roi)
    source_edge_neighborhood = cv2.dilate(
        source_edges.astype(np.uint8), np.ones((5, 5), np.uint8)
    ).astype(bool)
    result_edges = np.logical_and(result_edges, source_edge_neighborhood)
    line_edge_f1 = _f1(source_edges, result_edges)

    pure_black = np.all(source_rgb <= 8, axis=-1)
    if pure_black.any():
        preserved_black = np.all(result_rgb[pure_black] <= 8, axis=-1)
        pure_black_preservation_rate = float(preserved_black.mean())
    else:
        pure_black_preservation_rate = 1.0
    protected_area_ratio = float(protected_mask.mean())

    generated_color_coverage = 0.0
    result_color_coverage = 0.0
    color_retention_ratio = 1.0
    color_dropout_tiles = 0
    if generated is not None:
        generated_rgb = np.asarray(
            generated.convert("RGB").resize(source.size, Image.Resampling.LANCZOS)
        ).astype(np.int16)
        generated_luma = generated_rgb.mean(axis=-1)
        color_roi = np.logical_and(~protected_mask, generated_luma > 8)
        color_roi = np.logical_and(color_roi, generated_luma < 248)
        generated_color_coverage = _color_coverage(generated_rgb, color_roi)
        result_color_coverage = _color_coverage(result_rgb, color_roi)
        if generated_color_coverage >= 0.05:
            color_retention_ratio = min(
                1.0, result_color_coverage / generated_color_coverage
            )
            color_dropout_tiles = _color_dropout_tiles(
                generated_rgb, result_rgb, color_roi
            )

    reasons: list[str] = []
    if protected_diff != 0:
        reasons.append("protected_pixels_changed")
    if line_edge_f1 < line_f1_min:
        reasons.append("line_edge_f1_below_threshold")
    if luminance_mae > luminance_mae_max:
        reasons.append("luminance_mae_above_threshold")
    if pure_black_preservation_rate < pure_black_preservation_min:
        reasons.append("pure_black_preservation_below_threshold")
    if color_retention_ratio < color_retention_min:
        reasons.append("color_retention_below_threshold")
    if color_dropout_tiles > color_dropout_tiles_max:
        reasons.append("regional_color_dropout")
    return QAResult(
        passed=not reasons,
        protected_pixel_diff=protected_diff,
        line_edge_f1=line_edge_f1,
        luminance_mae=luminance_mae,
        dimension_match=True,
        reasons=reasons,
        pure_black_preservation_rate=pure_black_preservation_rate,
        protected_area_ratio=protected_area_ratio,
        generated_color_coverage=generated_color_coverage,
        result_color_coverage=result_color_coverage,
        color_retention_ratio=color_retention_ratio,
        color_dropout_tiles=color_dropout_tiles,
    )
