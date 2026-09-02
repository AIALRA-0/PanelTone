from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .color import lab_l
from .models import QAResult


def _edge_map(image: Image.Image) -> np.ndarray:
    # Geometry is measured on the value channel. Geometry-locked colorization
    # deliberately copies this channel from the source, so hue changes cannot
    # be mistaken for a newly drawn edge.
    rgb = np.asarray(image.convert("RGB"))
    value = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 2]
    return cv2.Canny(value, 80, 160) > 0


def _f1(reference: np.ndarray, candidate: np.ndarray) -> float:
    if not reference.any() and not candidate.any():
        return 1.0
    # A fixed one-pixel tolerance is intentional. A page-sized tolerance
    # allowed visibly shifted lines and doubled faces to pass on large scans.
    kernel = np.ones((3, 3), dtype=np.uint8)
    reference_neighborhood = cv2.dilate(reference.astype(np.uint8), kernel).astype(bool)
    candidate_neighborhood = cv2.dilate(candidate.astype(np.uint8), kernel).astype(bool)
    matched_reference = np.logical_and(reference, candidate_neighborhood).sum()
    matched_candidate = np.logical_and(candidate, reference_neighborhood).sum()
    recall = float(matched_reference / max(1, reference.sum()))
    precision = float(matched_candidate / max(1, candidate.sum()))
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _geometry_metrics(
    source_edges: np.ndarray, result_edges: np.ndarray
) -> tuple[float, float, float]:
    """Return source recall, added-edge ratio and edge alignment."""
    source_neighborhood = cv2.dilate(source_edges.astype(np.uint8), np.ones((3, 3), np.uint8))
    result_neighborhood = cv2.dilate(result_edges.astype(np.uint8), np.ones((3, 3), np.uint8))
    source_edge_recall = (
        float(np.logical_and(source_edges, result_neighborhood > 0).sum() / source_edges.sum())
        if source_edges.any()
        else 1.0
    )
    result_count = int(result_edges.sum())
    added_edges = np.logical_and(result_edges, source_neighborhood == 0)
    added_edge_ratio = float(added_edges.sum() / max(1, result_count))
    chroma_edge_alignment = float(
        np.logical_and(result_edges, source_neighborhood > 0).sum() / max(1, result_count)
    )
    return source_edge_recall, added_edge_ratio, chroma_edge_alignment


def _neutral_island_metrics(
    generated_rgb: np.ndarray, result_rgb: np.ndarray, roi: np.ndarray
) -> tuple[float, float]:
    """Measure result neutral pixels where the candidate has stable colour."""
    if not roi.any():
        return 0.0, 0.0
    generated_max = generated_rgb.max(axis=-1).astype(np.float32)
    generated_chroma = generated_max - generated_rgb.min(axis=-1).astype(np.float32)
    generated_saturation = generated_chroma / np.maximum(generated_max, 1.0)
    candidate_colour = np.logical_and(
        roi, np.logical_and(generated_chroma >= 6.0, generated_saturation >= 0.08)
    )
    result_max = result_rgb.max(axis=-1).astype(np.float32)
    result_chroma = result_max - result_rgb.min(axis=-1).astype(np.float32)
    result_saturation = result_chroma / np.maximum(result_max, 1.0)
    neutral = np.logical_and(
        candidate_colour,
        np.logical_or(result_chroma < 6.0, result_saturation < 0.08),
    )
    ratio = float(neutral.sum() / max(1, candidate_colour.sum()))
    count, _, stats, _ = cv2.connectedComponentsWithStats(neutral.astype(np.uint8), 8)
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0
    largest_ratio = float(largest / max(1, candidate_colour.sum()))
    return ratio, largest_ratio


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
    source_class: str = "line_art",
    source_passthrough: bool = False,
    source_edge_recall_min: float = 0.995,
    added_edge_ratio_max: float = 0.005,
    neutral_island_ratio_max: float = 0.08,
    largest_neutral_island_ratio_max: float = 0.03,
    panel_boundary_mask: np.ndarray | None = None,
    geometry_locked: bool = False,
    bypass_reason: str | None = None,
    source_sha256: str | None = None,
    final_sha256: str | None = None,
) -> QAResult:
    dimension_match = source.size == result.size
    if not dimension_match:
        return QAResult(
            passed=False,
            protected_pixel_diff=-1,
            line_edge_f1=0.0,
            luminance_mae=float("inf"),
            dimension_match=False,
            reasons=["dimension_mismatch"],
            source_class=source_class,
            source_passthrough=source_passthrough,
            bypass_reason=bypass_reason,
            source_sha256=source_sha256,
            final_sha256=final_sha256,
        )
    if protected_mask.shape != (source.height, source.width):
        raise ValueError("protection mask shape does not match source image")

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
    line_edge_f1 = _f1(source_edges, result_edges)
    source_edge_recall, added_edge_ratio, chroma_edge_alignment = _geometry_metrics(
        source_edges, result_edges
    )

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
    neutral_island_ratio = 0.0
    largest_neutral_island_ratio = 0.0
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
            neutral_island_ratio, largest_neutral_island_ratio = _neutral_island_metrics(
                generated_rgb, result_rgb, color_roi
            )

    panel_boundary_bleed_ratio = 0.0
    if panel_boundary_mask is not None:
        if panel_boundary_mask.shape != protected_mask.shape:
            raise ValueError("panel boundary mask shape does not match source image")
        boundary = panel_boundary_mask.astype(bool)
        outside = cv2.dilate(boundary.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        outside &= ~boundary
        source_chroma = source_rgb.max(axis=-1) - source_rgb.min(axis=-1)
        # Only neutral source pixels are eligible bleed targets. Existing
        # coloured artwork immediately outside a crop is legitimate page
        # content, not evidence that the candidate crossed the boundary.
        eligible = np.logical_and(outside, source_chroma < 6)
        result_chroma = result_rgb.max(axis=-1) - result_rgb.min(axis=-1)
        panel_boundary_bleed_ratio = float(
            np.logical_and(eligible, result_chroma >= 6).sum() / max(1, eligible.sum())
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
    if geometry_locked:
        if source_edge_recall < source_edge_recall_min:
            reasons.append("source_edge_recall_below_threshold")
        if added_edge_ratio > added_edge_ratio_max:
            reasons.append("added_edge_ratio_above_threshold")
        if neutral_island_ratio > neutral_island_ratio_max:
            reasons.append("neutral_island_above_threshold")
        if largest_neutral_island_ratio > largest_neutral_island_ratio_max:
            reasons.append("largest_neutral_island_above_threshold")
        if panel_boundary_bleed_ratio > 0.0:
            reasons.append("panel_boundary_bleed")
    if source_passthrough:
        if not np.array_equal(source_rgb, result_rgb):
            reasons.append("source_passthrough_changed")
        # A bypass page is an exact source copy, so colour and geometry gates
        # do not reject it merely because no generated candidate exists.
        reasons = [reason for reason in reasons if reason == "source_passthrough_changed"]
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
        source_class=source_class,
        source_passthrough=source_passthrough,
        source_edge_recall=source_edge_recall,
        added_edge_ratio=added_edge_ratio,
        chroma_edge_alignment=chroma_edge_alignment,
        neutral_island_ratio=neutral_island_ratio,
        largest_neutral_island_ratio=largest_neutral_island_ratio,
        panel_boundary_bleed_ratio=panel_boundary_bleed_ratio,
        bypass_reason=bypass_reason,
        source_sha256=source_sha256,
        final_sha256=final_sha256,
    )
