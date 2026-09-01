from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .models import PanelBox


def _sort_reading_order(boxes: list[PanelBox], rtl: bool) -> list[PanelBox]:
    if not boxes:
        return boxes
    median_height = float(np.median([box.height for box in boxes]))
    row_tolerance = max(20.0, median_height * 0.35)

    rows: list[list[PanelBox]] = []
    for box in sorted(boxes, key=lambda item: item.y):
        for row in rows:
            center = sum(item.y + item.height / 2 for item in row) / len(row)
            if abs((box.y + box.height / 2) - center) <= row_tolerance:
                row.append(box)
                break
        else:
            rows.append([box])
    ordered: list[PanelBox] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda item: item.x, reverse=rtl))
    return ordered


def detect_panels(
    image: Image.Image,
    min_area_ratio: float = 0.02,
    padding: int = 0,
    rtl: bool = True,
) -> list[PanelBox]:
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    page_area = height * width

    _, dark = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)
    horizontal = cv2.morphologyEx(
        dark,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, width // 80), 3)),
    )
    vertical = cv2.morphologyEx(
        dark,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(9, height // 80))),
    )
    borders = cv2.bitwise_or(horizontal, vertical)
    contours, _ = cv2.findContours(borders, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[PanelBox] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        ratio = (box_width * box_height) / page_area
        if ratio < min_area_ratio or ratio > 0.95:
            continue
        if box_width < width * 0.12 or box_height < height * 0.08:
            continue
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(width, x + box_width + padding)
        y1 = min(height, y + box_height + padding)
        candidates.append(PanelBox(x0, y0, x1 - x0, y1 - y0))

    deduplicated: list[PanelBox] = []
    for box in sorted(candidates, key=lambda item: item.width * item.height, reverse=True):
        overlaps = False
        for kept in deduplicated:
            x0 = max(box.x, kept.x)
            y0 = max(box.y, kept.y)
            x1 = min(box.right, kept.right)
            y1 = min(box.bottom, kept.bottom)
            intersection = max(0, x1 - x0) * max(0, y1 - y0)
            smaller = min(box.width * box.height, kept.width * kept.height)
            if smaller and intersection / smaller > 0.85:
                overlaps = True
                break
        if not overlaps:
            deduplicated.append(box)

    if not deduplicated:
        return [PanelBox(0, 0, width, height)]
    return _sort_reading_order(deduplicated, rtl=rtl)


def extract_panels(
    page_path: Path,
    destination: Path,
    mode: str,
    min_area_ratio: float,
    padding: int,
) -> list[tuple[PanelBox, Path]]:
    destination.mkdir(parents=True, exist_ok=True)
    with Image.open(page_path) as image:
        rgb = image.convert("RGB")
        if mode == "detect":
            boxes = detect_panels(rgb, min_area_ratio=min_area_ratio, padding=padding)
        else:
            boxes = [PanelBox(0, 0, rgb.width, rgb.height)]
        outputs: list[tuple[PanelBox, Path]] = []
        for index, box in enumerate(boxes):
            path = destination / f"panel_{index:04d}.png"
            rgb.crop(box.to_tuple()).save(path, format="PNG")
            outputs.append((box, path))
    return outputs
