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


def ink_detail_mask(image: Image.Image, threshold: int = 64) -> np.ndarray:
    """Return dark print detail while leaving gray screentone colorable.

    Manga scans often contain large gray screentone fields. A threshold near
    white treats those fields as line art and prevents any color from reaching
    them, while a very low threshold loses antialiased text and fine ink. The
    middle threshold is intentionally conservative and is used only for strict
    detail protection; speech bubbles and text regions are protected separately.
    """
    if not 0 <= threshold <= 255:
        raise ValueError("Ink threshold must be between 0 and 255")
    gray = np.asarray(image.convert("L"))
    return gray <= threshold


def ink_edge_mask(
    image: Image.Image,
    *,
    core_threshold: int = 64,
    edge_threshold: int = 128,
) -> np.ndarray:
    """Protect antialiased ink edges without restoring whole screentones.

    A grayscale cutoff alone misses the soft fringe around printed lines on
    high-resolution scans.  Blurring before Canny suppresses halftone dots;
    restricting detected edges to moderately dark source pixels keeps the
    protection local instead of turning broad gray fields monochrome.
    """
    if not 0 <= core_threshold <= edge_threshold <= 255:
        raise ValueError("Ink edge thresholds must be between 0 and 255")
    gray = np.asarray(image.convert("L"))
    core = gray <= core_threshold
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 20, 80) > 0
    return core | (edges & (gray <= edge_threshold))


def bubble_mask(image: Image.Image) -> np.ndarray:
    gray = np.asarray(image.convert("L"))
    height, width = gray.shape
    page_area = height * width
    white = (gray >= 245).astype(np.uint8) * 255
    contours, _ = cv2.findContours(white, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(white)
    text = text_region_mask(image)
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
        candidate = np.zeros_like(white)
        cv2.drawContours(candidate, [contour], -1, 255, thickness=cv2.FILLED)
        # White hands, faces and clothing regions can be compact and enclosed
        # by ink too. A deterministic bubble candidate is trusted only when it
        # contains multiple glyph components. A single mouth, eye highlight or
        # garment mark must not turn the surrounding body region monochrome.
        candidate_guard = cv2.dilate(candidate, np.ones((9, 9), np.uint8)) > 0
        text_inside = np.logical_and(candidate_guard, text).astype(np.uint8)
        component_count, _, component_stats, _ = cv2.connectedComponentsWithStats(
            text_inside, connectivity=8
        )
        glyph_components = sum(
            int(component_stats[label, cv2.CC_STAT_AREA]) >= 8
            for label in range(1, component_count)
        )
        if glyph_components < 2:
            continue
        result = cv2.bitwise_or(result, candidate)
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
    _, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    page_area = image.width * image.height
    valid = (
        (stats[:, cv2.CC_STAT_AREA] >= 2)
        & (stats[:, cv2.CC_STAT_AREA] <= max(24, page_area * 0.002))
        & (stats[:, cv2.CC_STAT_HEIGHT] <= image.height * 0.08)
        & (stats[:, cv2.CC_STAT_WIDTH] <= image.width * 0.15)
    )
    valid[0] = False
    result = valid[labels].astype(np.uint8)
    return cv2.dilate(result, np.ones((7, 7), np.uint8), iterations=1).astype(bool)


def text_region_mask(image: Image.Image) -> np.ndarray:
    """Find likely manga text blocks while rejecting most art linework."""
    gray = np.asarray(image.convert("L"))
    height, width = gray.shape
    page_area = height * width
    dark = (gray < 105).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    valid = (
        (stats[:, cv2.CC_STAT_AREA] >= 3)
        & (stats[:, cv2.CC_STAT_AREA] <= max(1800, page_area * 0.0015))
        & (stats[:, cv2.CC_STAT_WIDTH] <= width * 0.08)
        & (stats[:, cv2.CC_STAT_HEIGHT] <= height * 0.08)
        & (stats[:, cv2.CC_STAT_WIDTH] >= 2)
        & (stats[:, cv2.CC_STAT_HEIGHT] >= 3)
    )
    valid[0] = False
    glyphs = valid[labels].astype(np.uint8)

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
        # Protect the full bubble as well as glyphs. This prevents the model
        # from tinting or redrawing the white field around dialogue text.
        mask = np.logical_or(mask, bubble_mask(image))
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


def apply_mask_corrections(
    mask: np.ndarray,
    corrections: list[dict[str, object]],
    *,
    offset: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Apply non-destructive page-coordinate brush corrections to a mask.

    A correction uses ``op`` (``protect`` or ``release``), ``x``, ``y`` and
    ``radius`` in page pixels. Unknown or malformed records are ignored so a
    partially edited page can still be processed safely.
    """
    if not corrections:
        return mask
    result = mask.astype(bool, copy=True)
    height, width = result.shape
    offset_x, offset_y = offset
    yy, xx = np.ogrid[:height, :width]
    for correction in corrections:
        try:
            operation = str(correction.get("op", correction.get("action", ""))).casefold()
            x = float(correction["x"]) - offset_x
            y = float(correction["y"]) - offset_y
            radius = max(1.0, min(512.0, float(correction.get("radius", 12))))
        except (KeyError, TypeError, ValueError):
            continue
        if operation not in {"protect", "release", "add", "remove", "lock", "unlock"}:
            continue
        brush = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
        if operation in {"protect", "add", "lock"}:
            result[brush] = True
        else:
            result[brush] = False
    return result
