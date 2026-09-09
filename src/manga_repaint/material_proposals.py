"""Validate model-suggested mask prompts; confidence never implies acceptance."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MaskPrompt:
    # Absolute source-image pixels, not normalized model output
    box: tuple[int, int, int, int]
    positive: tuple[tuple[int, int], ...]
    negative: tuple[tuple[int, int], ...] = ()

    def validate(self, shape: tuple[int, int]) -> None:
        height, width = shape
        if any(type(value) is not int for value in self.box):
            raise ValueError("proposal coordinates must be integer source pixels")
        left, top, right, bottom = self.box
        if not 0 <= left < right <= width or not 0 <= top < bottom <= height:
            raise ValueError("invalid or out-of-source proposal box")
        if not self.positive:
            raise ValueError("mask proposal needs positive source points")
        for x, y in (*self.positive, *self.negative):
            if type(x) is not int or type(y) is not int:
                raise ValueError("proposal coordinates must be integer source pixels")
            if not 0 <= x < width or not 0 <= y < height:
                raise ValueError("mask proposal point outside source")
        if set(self.positive) & set(self.negative):
            raise ValueError("contradictory proposal points")


def evaluate_mask_proposal(mask: np.ndarray, prompt: MaskPrompt) -> dict:
    """Reject obvious prompt/geometry violations without claiming semantic truth.

    Neither a rectangle nor the model's predicted IoU is a material mask.
    Report holes and box spill for inspection, never auto-fill holes: they may
    be legitimate holes in objects, a balloon, another material or a highlight.
    """
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("mask proposal must be a two-dimensional boolean raster")
    prompt.validate(mask.shape)
    missed = [list(point) for point in prompt.positive if not mask[point[1], point[0]]]
    included = [list(point) for point in prompt.negative if mask[point[1], point[0]]]
    area = int(mask.sum())
    left, top, right, bottom = prompt.box
    inside = int(mask[top:bottom, left:right].sum())
    # Invert *within a padded bounding box* to measure enclosed holes cheaply
    local = np.pad(mask[top:bottom, left:right], 1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats((~local).astype(np.uint8), 8)
    exterior = labels[0, 0]
    hole_areas = [
        int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, count) if index != exterior
    ]
    reasons = []
    if not area:
        reasons.append("empty_mask")
    if missed:
        reasons.append("positive_points_missing")
    if included:
        reasons.append("negative_points_included")
    return {
        "state": "proposed",
        "prompt_consistent": not reasons,
        "reasons": reasons,
        "missed_positive": missed,
        "included_negative": included,
        "area": area,
        "box_spill_ratio": (area - inside) / max(area, 1),
        "enclosed_hole_count": len(hole_areas),
        "enclosed_hole_pixels": sum(hole_areas),
        "semantic_status": "unverified",
        "auto_accept": False,
    }
