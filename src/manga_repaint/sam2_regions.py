"""Optional local SAM2 proposals for reviewable manga material regions.

SAM2 is a geometry helper, not a semantic authority.  The model is loaded only
inside the dedicated model environment and every returned mask remains a
proposal until it has a material label, palette evidence and review state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True, slots=True)
class Sam2Region:
    mask: np.ndarray
    score: float
    seed_point: tuple[int, int]

    def validate(self, shape: tuple[int, int]) -> None:
        if self.mask.shape != shape or self.mask.dtype != np.bool_:
            raise ValueError("SAM2 proposal must be a source-sized boolean mask")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("SAM2 score must be between zero and one")
        x, y = self.seed_point
        if not 0 <= x < shape[1] or not 0 <= y < shape[0]:
            raise ValueError("SAM2 seed point is outside the source")


def _thumbnail_mask(mask: np.ndarray, size: int = 128) -> np.ndarray:
    height, width = mask.shape
    scale = min(size / max(width, 1), size / max(height, 1), 1.0)
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(mask.astype(np.uint8), target, interpolation=cv2.INTER_NEAREST).astype(bool)


def deduplicate_regions(
    regions: list[Sam2Region],
    *,
    duplicate_iou: float = 0.86,
    containment: float = 0.96,
) -> tuple[Sam2Region, ...]:
    """Keep the best geometrically distinct masks without semantic guessing."""
    if not 0.0 < duplicate_iou <= 1.0 or not 0.0 < containment <= 1.0:
        raise ValueError("invalid SAM2 deduplication threshold")
    kept: list[tuple[Sam2Region, np.ndarray]] = []
    for region in sorted(regions, key=lambda item: item.score, reverse=True):
        thumb = _thumbnail_mask(region.mask)
        duplicate = False
        for _other, other_thumb in kept:
            intersection = int(np.logical_and(thumb, other_thumb).sum())
            union = int(np.logical_or(thumb, other_thumb).sum())
            smaller = min(int(thumb.sum()), int(other_thumb.sum()))
            if union and intersection / union >= duplicate_iou:
                duplicate = True
                break
            if smaller and intersection / smaller >= containment:
                duplicate = True
                break
        if not duplicate:
            kept.append((region, thumb))
    return tuple(item for item, _thumb in kept)


def render_region_overlay(image: Image.Image, regions: tuple[Sam2Region, ...]) -> Image.Image:
    """Render an inspectable numbered proposal overlay for human review."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    overlay = rgb.copy()
    for index, region in enumerate(regions):
        hue = (index * 47) % 180
        colour = cv2.cvtColor(np.uint8([[[hue, 170, 235]]]), cv2.COLOR_HSV2RGB)[0, 0]
        boundary = cv2.morphologyEx(
            region.mask.astype(np.uint8),
            cv2.MORPH_GRADIENT,
            np.ones((3, 3), np.uint8),
        ).astype(bool)
        overlay[region.mask] = overlay[region.mask] * 0.72 + colour.astype(np.float32) * 0.28
        overlay[boundary] = colour
    result = Image.fromarray(np.clip(np.rint(overlay), 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(result)
    for index, region in enumerate(regions):
        x, y = region.seed_point
        draw.ellipse((x - 12, y - 12, x + 12, y + 12), fill="black", outline="white", width=2)
        draw.text((x - 6, y - 7), str(index), fill="white")
    return result


class Sam2AutomaticMasker:
    """Small local SAM2 automatic-mask adapter with bounded memory use."""

    def __init__(self, model_root: Path, *, device: str = "cpu", batch_size: int = 8):
        if batch_size < 1 or batch_size > 32:
            raise ValueError("SAM2 prompt batch must be between 1 and 32")
        # Transformers and torch intentionally remain optional app dependencies
        # because inference runs in PanelTone's dedicated model environment
        try:
            import torch
            from transformers import Sam2Model, Sam2Processor
        except ImportError as error:  # pragma: no cover - environment-dependent
            raise RuntimeError("SAM2 requires the PanelTone model environment") from error
        self._torch = torch
        self.processor = Sam2Processor.from_pretrained(str(model_root), local_files_only=True)
        self.model = Sam2Model.from_pretrained(str(model_root), local_files_only=True).eval()
        self.device = torch.device(device)
        self.model.to(self.device)
        self.batch_size = batch_size

    def _to_device(self, values: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in values.items()
        }

    def propose(
        self,
        image: Image.Image,
        *,
        exclude: np.ndarray | None = None,
        grid_x: int = 10,
        grid_y: int = 14,
        minimum_score: float = 0.72,
        minimum_area_ratio: float = 0.00025,
        maximum_area_ratio: float = 0.45,
    ) -> tuple[Sam2Region, ...]:
        if grid_x < 2 or grid_y < 2:
            raise ValueError("SAM2 proposal grid must have at least two points per axis")
        if not 0.0 <= minimum_score <= 1.0:
            raise ValueError("invalid SAM2 minimum score")
        rgb = image.convert("RGB")
        height, width = rgb.height, rgb.width
        if exclude is None:
            exclude = np.zeros((height, width), dtype=bool)
        if exclude.shape != (height, width) or exclude.dtype != np.bool_:
            raise ValueError("SAM2 exclusion must be a source-sized boolean mask")
        area = height * width
        xs = np.linspace(width / (grid_x * 2), width - width / (grid_x * 2), grid_x)
        ys = np.linspace(height / (grid_y * 2), height - height / (grid_y * 2), grid_y)
        points = [
            (int(round(x)), int(round(y)))
            for y in ys
            for x in xs
            if not exclude[int(round(y)), int(round(x))]
        ]
        image_inputs = self._to_device(self.processor(images=rgb, return_tensors="pt"))
        with self._torch.inference_mode():
            embeddings = self.model.get_image_embeddings(image_inputs["pixel_values"])
        candidates: list[Sam2Region] = []
        for start in range(0, len(points), self.batch_size):
            batch = points[start : start + self.batch_size]
            prompt_inputs = self._to_device(
                self.processor(
                    original_sizes=image_inputs["original_sizes"].cpu(),
                    input_points=[[[[x, y]] for x, y in batch]],
                    input_labels=[[[1] for _point in batch]],
                    return_tensors="pt",
                )
            )
            with self._torch.inference_mode():
                output = self.model(
                    image_embeddings=embeddings,
                    input_points=prompt_inputs["input_points"],
                    input_labels=prompt_inputs["input_labels"],
                    multimask_output=True,
                )
            masks = self.processor.post_process_masks(
                output.pred_masks.cpu(), image_inputs["original_sizes"].cpu()
            )[0]
            scores = output.iou_scores.detach().cpu().numpy()[0]
            for index, point in enumerate(batch):
                best = int(np.argmax(scores[index]))
                score = float(scores[index, best])
                mask = masks[index, best].numpy().astype(bool)
                mask &= ~exclude
                ratio = float(mask.sum() / max(area, 1))
                if (
                    score >= minimum_score
                    and minimum_area_ratio <= ratio <= maximum_area_ratio
                    and mask[point[1], point[0]]
                ):
                    region = Sam2Region(mask, score, point)
                    region.validate((height, width))
                    candidates.append(region)
        return deduplicate_regions(candidates)
