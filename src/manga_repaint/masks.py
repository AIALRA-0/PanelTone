from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .models import ProtectionMode


def line_art_mask(image: Image.Image, threshold: int = 245, dilation: int = 3) -> np.ndarray:
    gray = np.asarray(image.convert("L"))
    mask = gray < threshold
    if dilation > 0:
        kernel = np.ones((dilation * 2 + 1, dilation * 2 + 1), dtype=np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return mask


def pure_black_ink_mask(image: Image.Image, threshold: int = 8) -> np.ndarray:
    gray = np.asarray(image.convert("L"))
    return gray <= threshold


def bubble_mask(image: Image.Image) -> np.ndarray:
    gray = np.asarray(image.convert("L"))
    height, width = gray.shape
    page_area = height * width
    white = (gray >= 245).astype(np.uint8) * 255
    contours, _ = cv2.findContours(white, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(white)
    for contour in contours:
        area = cv2.contourArea(contour)
        if not page_area * 0.001 <= area <= page_area * 0.18:
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if x <= 1 or y <= 1 or x + box_width >= width - 1 or y + box_height >= height - 1:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        compactness = 4 * np.pi * area / (perimeter * perimeter)
        if compactness < 0.08:
            continue
        cv2.drawContours(result, [contour], -1, 255, thickness=cv2.FILLED)
    return result.astype(bool)


def border_mask(image: Image.Image) -> np.ndarray:
    gray = np.asarray(image.convert("L"))
    dark = (gray < 100).astype(np.uint8) * 255
    horizontal = cv2.morphologyEx(
        dark,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, image.width // 16), 2)),
    )
    vertical = cv2.morphologyEx(
        dark,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, max(15, image.height // 16))),
    )
    return cv2.dilate(cv2.bitwise_or(horizontal, vertical), np.ones((3, 3), np.uint8)).astype(bool)


def text_like_mask(image: Image.Image) -> np.ndarray:
    gray = np.asarray(image.convert("L"))
    dark = (gray < 120).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    result = np.zeros_like(dark)
    page_area = image.width * image.height
    for label in range(1, count):
        _, _, width, height, area = stats[label]
        if not 2 <= area <= max(24, page_area * 0.002):
            continue
        if height > image.height * 0.08 or width > image.width * 0.15:
            continue
        result[labels == label] = 1
    return cv2.dilate(result, np.ones((7, 7), np.uint8), iterations=1).astype(bool)


def text_region_mask(image: Image.Image) -> np.ndarray:
    """Find likely manga text blocks while rejecting most art linework."""
    gray = np.asarray(image.convert("L"))
    height, width = gray.shape
    page_area = height * width
    dark = (gray < 105).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    glyphs = np.zeros_like(dark)
    for label in range(1, count):
        x, y, component_width, component_height, area = stats[label]
        if not 3 <= area <= max(1800, page_area * 0.0015):
            continue
        if component_width > width * 0.08 or component_height > height * 0.08:
            continue
        if component_width < 2 or component_height < 3:
            continue
        glyphs[labels == label] = 1

    grouped = cv2.dilate(
        glyphs,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 15)),
        iterations=1,
    )
    grouped = cv2.morphologyEx(
        grouped,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (13, 21)),
    )
    contours, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidate_regions = np.zeros_like(dark)
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        area = box_width * box_height
        if not page_area * 0.00015 <= area <= page_area * 0.12:
            continue
        x0 = max(0, x - 5)
        y0 = max(0, y - 5)
        x1 = min(width, x + box_width + 5)
        y1 = min(height, y + box_height + 5)
        region = gray[y0:y1, x0:x1]
        white_ratio = float((region >= 225).mean())
        dark_ratio = float((region < 105).mean())
        if white_ratio < 0.68 or not 0.03 <= dark_ratio <= 0.55:
            continue
        candidate_regions[y0:y1, x0:x1] = 1

    protected_glyphs = np.logical_and(glyphs.astype(bool), candidate_regions.astype(bool))
    return cv2.dilate(
        protected_glyphs.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1
    ).astype(bool)


def deterministic_protection_mask(image: Image.Image, preserve_text: bool = True) -> np.ndarray:
    mask = border_mask(image)
    if preserve_text:
        mask = np.logical_or(mask, text_region_mask(image))
    return mask


def protection_mask(image: Image.Image, mode: ProtectionMode) -> np.ndarray:
    if mode == ProtectionMode.LUMINANCE:
        return np.zeros((image.height, image.width), dtype=bool)
    lines = line_art_mask(image)
    if mode == ProtectionMode.LINE_ART:
        return lines
    return lines | border_mask(image)


def save_mask(mask: np.ndarray, path: str) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path, format="PNG")
