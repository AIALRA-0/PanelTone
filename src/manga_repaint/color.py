from __future__ import annotations

import hashlib
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .masks import ink_edge_mask


@dataclass(frozen=True, slots=True)
class SourceClassification:
    """Deterministic page/unit classification used before model inference."""

    source_class: str
    source_passthrough: bool
    bypass_reason: str
    foreground_ratio: float
    edge_ratio: float
    gray_std: float


def _composited_rgb(image: Image.Image) -> np.ndarray:
    """Return RGB pixels with transparent input composited over white."""
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32)
    alpha = rgba[..., 3:4] / 255.0
    rgb = rgba[..., :3] * alpha + 255.0 * (1.0 - alpha)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def image_sha256(image: Image.Image) -> str:
    """Hash normalized RGB pixels, independent of the source file encoding."""
    rgb = _composited_rgb(image)
    digest = hashlib.sha256()
    digest.update(f"{rgb.shape[1]}x{rgb.shape[0]}x{rgb.shape[2]}".encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def classify_source_page(image: Image.Image) -> SourceClassification:
    """Classify a page before inference so empty pages cannot be hallucinated.

    The classifier intentionally uses only source pixels.  Semantic models are
    not involved, which keeps normal processing, repair and resume decisions
    identical even when an optional semantic provider is unavailable.
    """
    rgb = _composited_rgb(image)
    normalized = Image.fromarray(rgb, mode="RGB")
    if is_already_colorized(normalized):
        return SourceClassification("already_color", True, "source_already_color", 0.0, 0.0, 0.0)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    border = np.concatenate((gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]))
    background = float(np.median(border)) if border.size else 255.0
    deviation = np.abs(gray.astype(np.float32) - background)
    foreground = deviation >= max(12.0, float(np.std(border)) * 2.0)
    edges = cv2.Canny(gray, 60, 150) > 0
    foreground_ratio = float(foreground.mean())
    edge_ratio = float(edges.mean())
    gray_std = float(gray.astype(np.float32).std())

    if (foreground_ratio <= 0.001 and edge_ratio <= 0.0005) or gray_std <= 1.0:
        return SourceClassification(
            "blank", True, "source_is_blank_or_uniform", foreground_ratio, edge_ratio, gray_std
        )

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground.astype(np.uint8), connectivity=8
    )
    largest_component = 0
    if component_count > 1:
        largest_component = int(stats[1:, cv2.CC_STAT_AREA].max())
    largest_ratio = largest_component / max(1, width * height)
    if (
        foreground_ratio <= 0.04
        # Canny reports both sides of a narrow printed stroke.  The measured
        # page ratio is therefore allowed a small raster margin above the
        # semantic 1.2% sparse-content target.
        and edge_ratio <= 0.020
        and largest_ratio < 0.10
    ):
        return SourceClassification(
            "sparse_text_or_logo",
            True,
            "source_is_sparse_text_or_logo",
            foreground_ratio,
            edge_ratio,
            gray_std,
        )
    return SourceClassification(
        "line_art", False, "source_requires_colorization", foreground_ratio, edge_ratio, gray_std
    )


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    values = rgb.astype(np.float64) / 255.0
    linear = np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)
    xyz = (
        linear
        @ np.array(
            [
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ]
        ).T
    )
    xyz /= np.array([0.95047, 1.0, 1.08883])
    epsilon = 216 / 24389
    kappa = 24389 / 27
    fxyz = np.where(xyz > epsilon, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    lab = np.empty_like(fxyz)
    lab[..., 0] = 116 * fxyz[..., 1] - 16
    lab[..., 1] = 500 * (fxyz[..., 0] - fxyz[..., 1])
    lab[..., 2] = 200 * (fxyz[..., 1] - fxyz[..., 2])
    return lab


def _lab_to_linear_rgb(lab: np.ndarray) -> np.ndarray:
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    epsilon = 216 / 24389
    kappa = 24389 / 27
    fxyz = np.stack([fx, fy, fz], axis=-1)
    cubed = fxyz**3
    xyz = np.where(cubed > epsilon, cubed, (116 * fxyz - 16) / kappa)
    xyz *= np.array([0.95047, 1.0, 1.08883])
    return (
        xyz
        @ np.array(
            [
                [3.2404542, -1.5371385, -0.4985314],
                [-0.9692660, 1.8760108, 0.0415560],
                [0.0556434, -0.2040259, 1.0572252],
            ]
        ).T
    )


def _linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    srgb = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(np.maximum(linear, 0), 1 / 2.4) - 0.055,
    )
    return np.clip(np.rint(srgb * 255), 0, 255).astype(np.uint8)


def _lab_to_rgb_gamut_mapped(lab: np.ndarray) -> np.ndarray:
    mapped = lab.copy()
    chroma_scale = np.ones(lab.shape[:2], dtype=np.float64)
    for _ in range(12):
        linear = _lab_to_linear_rgb(mapped)
        invalid = np.logical_or(linear < 0, linear > 1).any(axis=-1)
        if not invalid.any():
            return _linear_to_srgb(linear)
        chroma_scale[invalid] *= 0.72
        mapped[..., 1] = lab[..., 1] * chroma_scale
        mapped[..., 2] = lab[..., 2] * chroma_scale
    return _linear_to_srgb(_lab_to_linear_rgb(mapped))


def lab_l(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32) * (100.0 / 255.0)


def preserve_luminance_lab(
    source: Image.Image,
    generated: Image.Image,
    chroma_strength: float = 1.0,
) -> Image.Image:
    if not 0.0 <= chroma_strength <= 2.5:
        raise ValueError("Chroma strength must be between 0.0 and 2.5")
    source_rgb = np.asarray(source.convert("RGB"))
    generated_rgb = np.asarray(
        generated.convert("RGB").resize(source.size, Image.Resampling.LANCZOS)
    )
    source_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB)
    generated_lab = cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2LAB)
    target_l = source_lab[..., 0].astype(np.float32)
    base_chroma = 128.0 + (generated_lab[..., 1:].astype(np.float32) - 128.0) * chroma_strength
    chroma_scale = np.ones(target_l.shape, dtype=np.float32)
    result = generated_rgb

    # OpenCV clips out-of-gamut Lab colours during the inverse conversion.  A
    # single conversion therefore changes L substantially for saturated
    # colours at the ends of the tonal range.  Reduce chroma only for those
    # pixels and recheck the converted result; this keeps the fast vectorized
    # path while retaining the source luminance contract.
    for _ in range(8):
        composed_lab = np.empty_like(source_lab)
        composed_lab[..., 0] = source_lab[..., 0]
        composed_lab[..., 1:] = np.clip(
            128.0 + (base_chroma - 128.0) * chroma_scale[..., None],
            0,
            255,
        ).astype(np.uint8)
        result = cv2.cvtColor(composed_lab, cv2.COLOR_LAB2RGB)
        result_l = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
        out_of_tolerance = np.abs(result_l - target_l) > 1.0
        if not out_of_tolerance.any():
            break
        chroma_scale[out_of_tolerance] *= 0.72
    return Image.fromarray(result, mode="RGB")


def apply_render_profile(
    image: Image.Image,
    *,
    saturation: float = 1.0,
    contrast: float = 1.0,
    hue_shift: float = 0.0,
) -> Image.Image:
    """Apply a bounded style grade before source-protection composition."""
    if not 0.0 <= saturation <= 2.5:
        raise ValueError("Saturation must be between 0.0 and 2.5")
    if not 0.5 <= contrast <= 1.5:
        raise ValueError("Contrast must be between 0.5 and 1.5")
    if not -180.0 <= hue_shift <= 180.0:
        raise ValueError("Hue shift must be between -180 and 180 degrees")
    rgb = np.asarray(image.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
    hsv[..., 2] = np.clip((hsv[..., 2] - 127.5) * contrast + 127.5, 0, 255)
    hsv[..., 0] = (hsv[..., 0] + hue_shift / 2.0) % 180.0
    graded = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return Image.fromarray(graded, mode="RGB")


def apply_vibrance_grade(
    image: Image.Image,
    *,
    saturation_gain: float = 1.55,
    vibrance: float = 0.28,
    max_saturation: float = 0.90,
    neutral_threshold: float = 0.025,
) -> Image.Image:
    """Enrich an existing colour render without repainting neutral pixels.

    Cobra already decides which colour belongs to each material. This finishing
    grade strengthens only pixels that contain a reliable chroma signal, with
    the largest relative lift applied to muted colours. Hue and HSV value stay
    unchanged, so the operation cannot introduce geometry, lighting texture,
    coloured speech bubbles, or a new colour into truly neutral source areas.
    """
    if not 1.0 <= saturation_gain <= 2.5:
        raise ValueError("Saturation gain must be between 1.0 and 2.5")
    if not 0.0 <= vibrance <= 1.0:
        raise ValueError("Vibrance must be between 0.0 and 1.0")
    if not 0.0 < max_saturation <= 1.0:
        raise ValueError("Maximum saturation must be between 0.0 and 1.0")
    if not 0.0 <= neutral_threshold < 0.25:
        raise ValueError("Neutral threshold must be between 0.0 and 0.25")

    rgb = np.asarray(image.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation = hsv[..., 1] / 255.0
    # A smooth gate leaves paper, gray screentones and white balloons exactly
    # neutral. It reaches full strength only after the model has supplied a
    # meaningful colour signal, rather than inventing one from gray pixels.
    gate = np.clip((saturation - neutral_threshold) / 0.11, 0.0, 1.0)
    gate = gate * gate * (3.0 - 2.0 * gate)
    multiplier = 1.0 + gate * (
        (saturation_gain - 1.0) + vibrance * (1.0 - saturation)
    )
    hsv[..., 1] = np.minimum(saturation * multiplier, max_saturation) * 255.0
    graded = cv2.cvtColor(np.rint(hsv).astype(np.uint8), cv2.COLOR_HSV2RGB)
    return Image.fromarray(graded, mode="RGB")


def is_already_colorized(
    image: Image.Image,
    *,
    coverage_min: float = 0.80,
    pixel_min: int = 4096,
) -> bool:
    """Return whether a source page is already a substantially colour image.

    Covers, credits and publisher pages can be supplied as colour scans. They
    must not be sent through a monochrome-to-colour model, especially when a
    model invents structure that is absent from the source. White paper and
    black line art are excluded from the coverage calculation.  The threshold
    deliberately allows printed highlights, white lettering and neutral scan
    areas: a genuinely coloured cover can contain more than ten percent
    achromatic pixels without becoming a monochrome page.
    """
    if not 0.0 <= coverage_min <= 1.0:
        raise ValueError("Colour coverage minimum must be between 0.0 and 1.0")
    if pixel_min < 1:
        raise ValueError("Colour coverage pixel minimum must be positive")
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    luma = rgb.mean(axis=-1)
    roi = (luma > 8) & (luma < 248)
    if int(roi.sum()) < pixel_min:
        return False
    maximum = rgb.max(axis=-1)
    minimum = rgb.min(axis=-1)
    chroma = maximum - minimum
    saturation = chroma / np.maximum(maximum, 1)
    colourful = (chroma >= 6) & (saturation >= 0.08)
    return float(colourful[roi].mean()) >= coverage_min


def composite_protected(
    source: Image.Image,
    generated: Image.Image,
    mask: np.ndarray,
) -> Image.Image:
    source_array = np.asarray(source.convert("RGB"))
    generated_array = np.asarray(
        generated.convert("RGB").resize(source.size, Image.Resampling.LANCZOS)
    )
    result = generated_array.copy()
    result[mask] = source_array[mask]
    return Image.fromarray(result, mode="RGB")


def replace_masked(
    base: Image.Image,
    replacement: Image.Image,
    mask: np.ndarray,
) -> Image.Image:
    """Replace only the selected pixels while preserving the base elsewhere."""
    base_array = np.asarray(base.convert("RGB"))
    replacement_array = np.asarray(
        replacement.convert("RGB").resize(base.size, Image.Resampling.LANCZOS)
    )
    result = base_array.copy()
    result[mask] = replacement_array[mask]
    return Image.fromarray(result, mode="RGB")


def composite_strict_colorization(
    source: Image.Image,
    generated: Image.Image,
    protected_mask: np.ndarray,
    chroma_strength: float = 1.0,
    ink_core_threshold: int = 64,
    ink_edge_threshold: int | None = None,
) -> Image.Image:
    """Use the generated colour image while restoring protected manga structure.

    Older releases copied source luminance across the full page. That removed
    most generated chroma on screentones and produced partially gray output.
    The generated image is now the visual base; only explicit protection and
    core ink are exact, while antialiased line edges are blended softly.
    """
    if not 0 <= ink_core_threshold <= 255:
        raise ValueError("Ink core threshold must be between 0 and 255")
    if not 0.0 <= chroma_strength <= 2.5:
        raise ValueError("Chroma strength must be between 0.0 and 2.5")
    source_rgb = np.asarray(source.convert("RGB"), dtype=np.float32)
    generated_rgb = np.asarray(
        generated.convert("RGB").resize(source.size, Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    source_gray = np.asarray(source.convert("L"), dtype=np.uint8)
    if ink_edge_threshold is None:
        # Keep the explicit legacy cutoff useful to callers that need the
        # original narrow behavior. Production composition opts into the
        # scale-aware local edge mask below.
        ink_mask = source_gray <= ink_core_threshold
    else:
        ink_mask = ink_edge_mask(
            source,
            core_threshold=ink_core_threshold,
            edge_threshold=ink_edge_threshold,
        )
    strict_mask = np.logical_or(protected_mask, ink_mask)
    # Keep every non-protected pixel from the full-colour generation. Dense
    # manga screentones are not safe soft-edge hints: even alignment-gated
    # multiplication can turn a shifted coloured limb back into gray.
    result = generated_rgb.copy()
    result = np.clip(np.rint(result), 0, 255).astype(np.uint8)
    result[strict_mask] = source_rgb.astype(np.uint8)[strict_mask]
    return Image.fromarray(result, mode="RGB")


def composite_geometry_locked_colorization(
    source: Image.Image,
    generated: Image.Image,
    protected_mask: np.ndarray,
    chroma_strength: float = 1.0,
    ink_core_threshold: int = 64,
) -> Image.Image:
    """Apply model colour to source geometry without importing model edges.

    ``generated`` contributes only hue and saturation.  The value channel,
    edges, ink and all explicitly protected pixels come from ``source``.  A
    shifted or hallucinated model structure therefore cannot create a second
    face, line, balloon or panel in the result.
    """
    if not 0.0 <= chroma_strength <= 2.5:
        raise ValueError("Chroma strength must be between 0.0 and 2.5")
    if not 0 <= ink_core_threshold <= 255:
        raise ValueError("Ink core threshold must be between 0 and 255")
    if source.size != generated.size:
        raise ValueError("source and generated dimensions must match exactly")
    source_rgb = _composited_rgb(source)
    generated_rgb = _composited_rgb(generated)
    height, width = source_rgb.shape[:2]
    if protected_mask.shape != (height, width):
        raise ValueError("protection mask shape does not match source image")

    barrier = geometry_barrier_mask(
        Image.fromarray(source_rgb, mode="RGB"),
        protected_mask,
        ink_core_threshold=ink_core_threshold,
    )
    traversable = ~barrier

    generated_hsv = cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue = generated_hsv[..., 0] * (2.0 * np.pi / 180.0)
    saturation = generated_hsv[..., 1]
    valid_color = np.logical_and(saturation >= 18.0, generated_hsv[..., 2] >= 8.0)
    filtered_hue = generated_hsv[..., 0].copy()
    filtered_saturation = saturation.copy()
    # Keep the normalized blur barrier-aware, but never run a full-canvas blur
    # once per component.  Manga pages commonly contain thousands of small
    # source-connected regions; doing the old operation on the whole canvas
    # made repair time grow quadratically with the number of regions.
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        traversable.astype(np.uint8), connectivity=8
    )
    blur_margin = 4

    for component in range(1, component_count):
        x, y, component_width, component_height, area = stats[component]
        area = int(area)
        if area < 4:
            continue
        x0 = max(0, int(x) - blur_margin)
        y0 = max(0, int(y) - blur_margin)
        x1 = min(width, int(x + component_width) + blur_margin)
        y1 = min(height, int(y + component_height) + blur_margin)
        region = labels[y0:y1, x0:x1] == component
        local_hue = filtered_hue[y0:y1, x0:x1]
        local_saturation = filtered_saturation[y0:y1, x0:x1]
        local_generated_hue = hue[y0:y1, x0:x1]
        local_generated_saturation = saturation[y0:y1, x0:x1]
        local_valid_color = valid_color[y0:y1, x0:x1]
        valid = np.logical_and(region, local_valid_color)
        valid_count = int(valid.sum())
        if valid_count / area < 0.20:
            local_saturation[region] = local_generated_saturation[region] * chroma_strength
            continue

        weights = np.where(valid, local_generated_saturation / 255.0, 0.0).astype(np.float32)
        cos_values = np.cos(local_generated_hue) * weights
        sin_values = np.sin(local_generated_hue) * weights
        # Masked normalized convolution is performed on each source-connected
        # component, so no generated colour can cross a source barrier.
        region_float = region.astype(np.float32)
        local_cos = cv2.GaussianBlur(cos_values * region_float, (0, 0), 1.15)
        local_sin = cv2.GaussianBlur(sin_values * region_float, (0, 0), 1.15)
        local_sat = cv2.GaussianBlur(
            local_generated_saturation * valid * region_float, (0, 0), 1.15
        )
        local_weight = cv2.GaussianBlur(valid.astype(np.float32) * region_float, (0, 0), 1.15)
        stable_cos = float(cos_values[valid].sum())
        stable_sin = float(sin_values[valid].sum())
        stable_angle = float(np.arctan2(stable_sin, stable_cos))
        stable_strength = float(np.hypot(stable_cos, stable_sin) / max(1, valid_count))
        stable_hue = (stable_angle % (2.0 * np.pi)) * 180.0 / np.pi
        stable_saturation = float(np.median(local_generated_saturation[valid])) * chroma_strength
        good_local = np.logical_and(region, local_weight > 0.01)
        local_angle = np.mod(np.arctan2(local_sin, local_cos), 2.0 * np.pi) * 180.0 / np.pi
        local_hue[good_local] = local_angle[good_local] / 2.0
        local_saturation[good_local] = (
            local_sat[good_local] / np.maximum(local_weight[good_local], 1e-3)
        ) * chroma_strength
        # Fill neutral holes only when the component has a coherent colour
        # signal.  This avoids turning a genuinely neutral page into a flat
        # tint while fixing the half-white skin/scene islands.
        if stable_strength >= 0.12:
            holes = np.logical_and(region, ~good_local)
            local_hue[holes] = stable_hue / 2.0
            local_saturation[holes] = stable_saturation
        local_hue[region] %= 180.0
        local_saturation[region] = np.clip(local_saturation[region], 0, 255)

    composed_hsv = np.empty((height, width, 3), dtype=np.uint8)
    composed_hsv[..., 0] = np.mod(filtered_hue, 180.0).astype(np.uint8)
    composed_hsv[..., 1] = np.clip(filtered_saturation, 0, 255).astype(np.uint8)
    # The source value channel is the sole value channel.  This deliberately
    # ignores generated RGB, value, Lab lightness, edges and texture while
    # preserving the original brightness of scans that contain colour accents.
    composed_hsv[..., 2] = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2HSV)[..., 2]
    result_rgb = cv2.cvtColor(composed_hsv, cv2.COLOR_HSV2RGB)
    result_rgb[barrier] = source_rgb[barrier]
    return Image.fromarray(result_rgb, mode="RGB")


def clean_region_flats(
    source_gray: np.ndarray,
    hue: np.ndarray,
    saturation: np.ndarray,
    protected: np.ndarray,
    *,
    strength: float = 0.85,
) -> tuple[np.ndarray, np.ndarray]:
    """Separate coherent base colours from source shading, like a flats layer.

    Only source-enclosed, overwhelmingly single-colour regions are regularized.
    Multicolour patterns and uncoloured areas are not assigned invented colours.
    Hue statistics are circular. Neither shadows nor source ink are blurred.
    """
    if not 0 <= strength <= 1:
        raise ValueError("Flat colour strength must be between zero and one")
    edges = cv2.Canny(cv2.GaussianBlur(source_gray, (3, 3), 0.6), 48, 120) > 0
    barrier = cv2.dilate(
        (edges | protected | (source_gray < 40)).astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(1 - barrier, connectivity=4)
    inside_distance = cv2.distanceTransform(1 - barrier, cv2.DIST_L2, 3)
    result_hue, result_sat = hue.copy(), saturation.copy()
    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area < 64:
            continue
        x, y, width, height = (int(value) for value in stats[component, :4])
        region = labels[y:y + height, x:x + width] == component
        local_hue = hue[y:y + height, x:x + width]
        local_sat = saturation[y:y + height, x:x + width]
        coloured = region & (local_sat >= 18)
        if int(coloured.sum()) < area * 0.65:
            continue
        angles = local_hue[coloured] * np.pi / 90.0
        # Saturation weighting would let a few neon stains dictate the flat.
        centre = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
        distance = np.abs(np.angle(np.exp(1j * (angles - centre))))
        coherent = distance <= np.pi / 6
        if float(np.mean(coherent)) < 0.9:
            continue
        centre = np.arctan2(np.mean(np.sin(angles[coherent])),
                            np.mean(np.cos(angles[coherent])))
        target_sat = float(np.median(local_sat[coloured][coherent]))
        region_angles = local_hue[coloured] * np.pi / 90.0
        # Fade corrections into the original colour field at ink and neutral
        # boundaries. A hard flat-mask edge would itself become a coloured rim.
        amount = strength * np.minimum(inside_distance[y:y + height, x:x + width][coloured] / 8, 1)
        amount *= np.clip((local_sat[coloured] - 18) / 32, 0, 1)
        mixed = np.arctan2(
            (1 - amount) * np.sin(region_angles) + amount * np.sin(centre),
            (1 - amount) * np.cos(region_angles) + amount * np.cos(centre),
        )
        result_hue[y:y + height, x:x + width][coloured] = (mixed * 90 / np.pi) % 180
        result_sat[y:y + height, x:x + width][coloured] = (
            (1 - amount) * local_sat[coloured] + amount * target_sat
        )
    return result_hue, result_sat


def composite_reference_locked_colorization(
    source: Image.Image,
    generated: Image.Image,
    protected_mask: np.ndarray,
    *,
    chroma_strength: float = 1.0,
    ink_core_threshold: int = 64,
    palette_anchors: list[tuple[float, float, float]] | None = None,
    palette_strength: float = 0.28,
    clean_flats: bool = True,
) -> Image.Image:
    """Transfer reference colour while keeping source luminance and geometry.

    The candidate contributes only HSV hue and saturation.  The source
    contributes the complete value channel plus protected and near-black
    pixels.  Circular hue and saturation smoothing makes the colour field
    stable, while excluding candidate RGB, edges, texture, and geometry keeps
    shifted eyes, limbs, lettering, and panels out of the result.
    """
    if not 0.0 <= chroma_strength <= 2.5:
        raise ValueError("Chroma strength must be between 0.0 and 2.5")
    source_rgb = _composited_rgb(source)
    generated_rgb = _composited_rgb(generated)
    if generated_rgb.shape[:2] != source_rgb.shape[:2]:
        raise ValueError("source and generated images must have identical dimensions")
    if protected_mask.shape != source_rgb.shape[:2]:
        raise ValueError("protection mask shape does not match source image")
    if not 0.0 <= palette_strength <= 1.0:
        raise ValueError("Palette strength must be between 0.0 and 1.0")
    source_hsv = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2HSV)
    candidate_hsv = cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2HSV)
    source_gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY)
    # A generated candidate is a low-frequency colour hint, never a pixelwise
    # overlay.  Model redraws can shift an eye, limb or balloon by several
    # pixels; copying its H/S channels directly turned those shifted boundaries
    # into coloured double contours even while source luminance stayed exact.
    # Guided filtering aligns the candidate chroma field to source luminance
    # edges and suppresses candidate-only high-frequency structure.
    candidate_ycc = cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    valid_candidate_colour = np.logical_and(
        candidate_hsv[..., 1] >= 18, candidate_hsv[..., 2] >= 32
    )
    nearby_colour = cv2.dilate(
        valid_candidate_colour.astype(np.uint8), np.ones((13, 13), np.uint8)
    ).astype(bool)
    # Candidate black strokes and small neutral holes describe the model's
    # redrawn geometry, not colour.  Remove them from the chroma signal before
    # edge guidance; source ink and luminance are restored later from source.
    inpaint_mask = np.logical_and(candidate_hsv[..., 2] <= 48, nearby_colour)
    neutral = np.logical_and(candidate_hsv[..., 1] < 12, nearby_colour)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        neutral.astype(np.uint8), connectivity=8
    )
    maximum_hole = max(64, int(source_rgb.shape[0] * source_rgb.shape[1] * 0.015))
    for component in range(1, count):
        if int(stats[component, cv2.CC_STAT_AREA]) > maximum_hole:
            continue
        x = int(stats[component, cv2.CC_STAT_LEFT])
        y = int(stats[component, cv2.CC_STAT_TOP])
        width = int(stats[component, cv2.CC_STAT_WIDTH])
        height = int(stats[component, cv2.CC_STAT_HEIGHT])
        # A 13x13 dilation can reach only six pixels beyond a component.
        # Work in that bounded ROI instead of allocating and scanning a full
        # page-sized boolean array for every neutral speck on a screentone page.
        x0 = max(0, x - 6)
        y0 = max(0, y - 6)
        x1 = min(labels.shape[1], x + width + 6)
        y1 = min(labels.shape[0], y + height + 6)
        region = labels[y0:y1, x0:x1] == component
        colour_ring = np.logical_and(
            cv2.dilate(region.astype(np.uint8), np.ones((13, 13), np.uint8)).astype(bool),
            np.logical_and(
                ~region, valid_candidate_colour[y0:y1, x0:x1]
            ),
        )
        if int(colour_ring.sum()) < 32:
            continue
        ring_hue = candidate_hsv[y0:y1, x0:x1, 0][colour_ring].astype(np.float32)
        ring_saturation = candidate_hsv[y0:y1, x0:x1, 1][colour_ring].astype(np.float32)
        angles = ring_hue * np.pi / 90.0
        weights = np.maximum(ring_saturation, 1.0)
        concentration = float(
            np.hypot(
                np.sum(np.sin(angles) * weights),
                np.sum(np.cos(angles) * weights),
            )
            / np.sum(weights)
        )
        # Only a coherent surrounding colour may fill a neutral candidate
        # island.  This repairs missing skin/hair patches without painting a
        # deliberately white shirt or a balloon from unrelated neighbours.
        if concentration >= 0.78 and float(np.median(ring_saturation)) >= 24.0:
            inpaint_mask[y0:y1, x0:x1] |= region
    if inpaint_mask.any() and (~inpaint_mask).any():
        encoded_mask = inpaint_mask.astype(np.uint8) * 255
        for channel in (1, 2):
            candidate_ycc[..., channel] = cv2.inpaint(
                candidate_ycc[..., channel].astype(np.uint8),
                encoded_mask,
                5,
                cv2.INPAINT_TELEA,
            )
    guide = source_gray.astype(np.float32) / 255.0
    radius = max(2, min(12, round(min(source_rgb.shape[:2]) / 144)))
    sigma = max(0.8, radius * 0.22)
    filtered_ycc = np.empty_like(candidate_ycc)
    # A constant working luminance prevents screentone/shadow values from
    # modulating H/S through YCrCb clipping. Actual source shading is applied
    # once, below, as the exact source value channel.
    filtered_ycc[..., 0] = 192 if clean_flats else source_gray
    for channel in (1, 2):
        chroma = (candidate_ycc[..., channel] - 128.0) / 127.0
        chroma = cv2.GaussianBlur(chroma, (0, 0), sigma)
        filtered = _guided_filter(guide, chroma, radius=radius, epsilon=0.0025)
        filtered_ycc[..., channel] = np.clip(
            128.0 + filtered * 127.0 * float(chroma_strength), 0, 255
        )
    guided_rgb = cv2.cvtColor(
        np.clip(np.rint(filtered_ycc), 0, 255).astype(np.uint8),
        cv2.COLOR_YCrCb2RGB,
    )
    guided_hsv = cv2.cvtColor(guided_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    result_hue = guided_hsv[..., 0]
    # Reapplying a bright source value after YCrCb conversion naturally lowers
    # measured HSV saturation. Restore it only from the already source-guided
    # chroma field. Candidate-derived saturation hints are deliberately
    # forbidden here: even after smoothing they can restore a shifted contour
    # as a cyan or magenta rim.
    result_sat = np.clip(
        guided_hsv[..., 1] * (1.15 if clean_flats else 1.5) * float(chroma_strength),
        0,
        255,
    )
    if clean_flats:
        # Cleaning a faintly tinted area must not erase its existing colour.
        # Retain only a bounded pastel floor from the SAME guided chroma field,
        # never from candidate pixels or texture. Strong saturation still comes
        # exclusively from the constant-luminance base above.
        shaded_hint = filtered_ycc.copy()
        shaded_hint[..., 0] = source_gray
        shaded_rgb = cv2.cvtColor(shaded_hint.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
        shaded_sat = cv2.cvtColor(shaded_rgb, cv2.COLOR_RGB2HSV)[..., 1].astype(np.float32)
        result_sat = np.maximum(result_sat, np.minimum(48, shaded_sat * 1.5 * chroma_strength))

    # A reference library is a book-level colour vocabulary, not a semantic
    # segmentation model. Softly pull nearby candidate hues toward that
    # vocabulary so skin, hair, and clothing do not drift between pages while
    # preserving genuinely distinct colours that are absent from the book.
    if palette_anchors:
        result_hue, result_sat = _stabilize_palette(
            result_hue,
            result_sat,
            palette_anchors,
            strength=palette_strength,
        )

    if clean_flats:
        result_hue, result_sat = clean_region_flats(
            source_gray, result_hue, result_sat, protected_mask
        )

    # Do not run the legacy pixelwise neutral-hole pass here.  Its local hue
    # lookup can reintroduce the very candidate edges removed above.  Small
    # dark/neutral holes have already been inpainted before guided transfer;
    # larger neutral regions stay neutral unless a future source-region model
    # assigns them a reviewed colour.
    composed = np.dstack(
        (
            result_hue.astype(np.uint8),
            result_sat.astype(np.uint8),
            source_hsv[..., 2],
        )
    )
    result_rgb = cv2.cvtColor(composed, cv2.COLOR_HSV2RGB)
    # Semantic protection is the hard boundary.  Pure black source pixels are
    # also restored exactly so missed semantic ink cannot be recoloured.
    black_source = source_gray <= 18
    exact = protected_mask | black_source
    result_rgb[exact] = source_rgb[exact]
    return Image.fromarray(result_rgb, mode="RGB")


def _guided_filter(
    guide: np.ndarray,
    signal: np.ndarray,
    *,
    radius: int,
    epsilon: float,
) -> np.ndarray:
    """Edge-align a scalar signal to a normalized single-channel guide."""
    if guide.shape != signal.shape:
        raise ValueError("guided filter arrays must have identical dimensions")
    if radius < 1:
        raise ValueError("guided filter radius must be positive")
    window = (radius * 2 + 1, radius * 2 + 1)
    mean_guide = cv2.boxFilter(guide, cv2.CV_32F, window, normalize=True)
    mean_signal = cv2.boxFilter(signal, cv2.CV_32F, window, normalize=True)
    correlation = cv2.boxFilter(guide * signal, cv2.CV_32F, window, normalize=True)
    variance = (
        cv2.boxFilter(guide * guide, cv2.CV_32F, window, normalize=True)
        - mean_guide * mean_guide
    )
    covariance = correlation - mean_guide * mean_signal
    coefficient = covariance / (variance + float(epsilon))
    offset = mean_signal - coefficient * mean_guide
    mean_coefficient = cv2.boxFilter(
        coefficient, cv2.CV_32F, window, normalize=True
    )
    mean_offset = cv2.boxFilter(offset, cv2.CV_32F, window, normalize=True)
    return mean_coefficient * guide + mean_offset


def _stabilize_palette(
    hue: np.ndarray,
    saturation: np.ndarray,
    anchors: list[tuple[float, float, float]],
    *,
    strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Softly regularize candidate colours against a local book palette."""
    if not anchors or strength <= 0:
        return hue, saturation
    anchor_hue = np.asarray([item[0] for item in anchors], dtype=np.float32)
    anchor_sat = np.asarray([item[1] for item in anchors], dtype=np.float32)
    anchor_weight = np.asarray([max(0.0, item[2]) for item in anchors], dtype=np.float32)
    anchor_weight /= max(float(anchor_weight.max()), 1e-6)
    pixels = hue.reshape(-1)
    sat = saturation.reshape(-1)
    angle = pixels[:, None] * np.pi / 90.0
    anchor_angle = anchor_hue[None, :] * np.pi / 90.0
    distance = np.abs(pixels[:, None] - anchor_hue[None, :])
    distance = np.minimum(distance, 180.0 - distance)
    nearest = np.argmin(distance, axis=1)
    nearest_distance = distance[np.arange(distance.shape[0]), nearest]
    # Large hue jumps are more likely to be a meaningful distinct colour than
    # palette drift. Only regularize a nearby colour family.
    apply = (sat >= 20.0) & (nearest_distance <= 32.0)
    local_strength = strength * anchor_weight[nearest] * np.clip(
        1.0 - nearest_distance / 32.0, 0.0, 1.0
    )
    local_strength = np.where(apply, local_strength, 0.0).astype(np.float32)
    target = anchor_angle[0, nearest]
    sin_value = np.sin(angle[:, 0]) * (1.0 - local_strength) + np.sin(target) * local_strength
    cos_value = np.cos(angle[:, 0]) * (1.0 - local_strength) + np.cos(target) * local_strength
    pixels = (np.arctan2(sin_value, cos_value) * 90.0 / np.pi) % 180.0
    target_sat = anchor_sat[nearest]
    sat = np.where(
        apply,
        sat * (1.0 - local_strength * 0.45) + target_sat * local_strength * 0.45,
        sat,
    )
    return pixels.reshape(hue.shape), np.clip(sat, 0, 255).reshape(saturation.shape)


def _fill_neutral_holes(
    hue: np.ndarray,
    saturation: np.ndarray,
    candidate_value: np.ndarray,
    protected_mask: np.ndarray,
    source_gray: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill small neutral gaps surrounded by a coherent candidate colour."""
    coloured = (saturation >= 28.0) & (candidate_value >= 18.0)
    if not coloured.any():
        return hue, saturation
    kernel = np.ones((11, 11), dtype=np.uint8)
    near_colour = cv2.dilate(coloured.astype(np.uint8), kernel) > 0
    support = cv2.GaussianBlur(coloured.astype(np.float32), (0, 0), 3.0)
    distance = cv2.distanceTransform((~coloured).astype(np.uint8), cv2.DIST_L2, 3)
    # Do not fill exact protection, pure black ink, or pixels far from a local
    # colour signal. A 12px radius is enough for holes in hair/skin/clothing
    # without turning a neighbouring speech balloon into a painted block.
    holes = (
        near_colour
        & (support >= 0.08)
        & (distance <= 12.0)
        & (saturation < 18.0)
        & ~protected_mask
        & (source_gray > 8)
    )
    if not holes.any():
        return hue, saturation
    angle = hue * np.pi / 90.0
    weights = coloured.astype(np.float32)
    local_sin = cv2.GaussianBlur(np.sin(angle) * weights, (0, 0), 3.0)
    local_cos = cv2.GaussianBlur(np.cos(angle) * weights, (0, 0), 3.0)
    local_sat = cv2.GaussianBlur(saturation * weights, (0, 0), 3.0)
    local_weight = cv2.GaussianBlur(weights, (0, 0), 3.0)
    stable = local_weight > 0.08
    filled_hue = (np.arctan2(local_sin, local_cos) * 90.0 / np.pi) % 180.0
    filled_sat = local_sat / np.maximum(local_weight, 1e-4)
    apply = holes & stable
    hue = np.where(apply, filled_hue, hue)
    saturation = np.where(apply, np.maximum(filled_sat * 0.78, 22.0), saturation)
    return hue, np.clip(saturation, 0, 255)


def normalize_color_candidate_size(
    candidate: Image.Image,
    source_size: tuple[int, int],
    *,
    max_aspect_error: float = 0.01,
) -> Image.Image:
    """Put a colour-only candidate on the source pixel grid safely.

    A candidate model may use a smaller internal canvas.  Since only its
    colour channels cross the boundary, a small aspect-preserving resize is
    safe; a crop or visible distortion is rejected before compositing.
    """
    if tuple(candidate.size) == tuple(source_size):
        return candidate
    source_ratio = source_size[0] / max(1, source_size[1])
    candidate_ratio = candidate.size[0] / max(1, candidate.size[1])
    if abs(source_ratio - candidate_ratio) / max(source_ratio, 1e-6) > max_aspect_error:
        raise ValueError(
            f"candidate output aspect ratio mismatch: {candidate.size} vs {source_size}"
        )
    return candidate.resize(source_size, resample=Image.Resampling.LANCZOS)


def geometry_barrier_mask(
    source: Image.Image,
    protected_mask: np.ndarray,
    *,
    ink_core_threshold: int = 64,
) -> np.ndarray:
    """Return the source geometry barrier shared by composition and QA."""
    if not 0 <= ink_core_threshold <= 255:
        raise ValueError("Ink threshold must be between 0 and 255")
    if protected_mask.shape != (source.height, source.width):
        raise ValueError("protection mask shape does not match source image")
    source_rgb = _composited_rgb(source)
    source_gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY)
    # Blur before edge extraction so halftone dots do not become a near-solid
    # barrier across the page.
    blurred = cv2.GaussianBlur(source_gray, (5, 5), 0)
    source_edges = cv2.Canny(blurred, 60, 150) > 0
    # A raw grayscale cutoff treats screentone dots and dark shading as if
    # they were walls.  That strands most of a manga page in monochrome. Use
    # blurred structural edges for the diffusion barrier; exact dark ink is
    # restored separately after compositing, while semantic protection keeps
    # text, balloons and borders pixel exact.
    ink = (cv2.Canny(blurred, 20, 80) > 0) & (source_gray <= 128)
    barrier = np.logical_or.reduce((protected_mask, source_edges, ink))
    # A one-pixel guard prevents the colour diffusion kernel from sampling on
    # the opposite side of a source line or panel boundary.
    return cv2.dilate(barrier.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0


def validated_colorization_protection(
    source: Image.Image,
    generated: Image.Image,
    protected_mask: np.ndarray,
    *,
    dark_structure_threshold: int = 128,
    neutral_chroma_threshold: int = 24,
) -> np.ndarray:
    """Reject bright protection pixels that contradict generated colour.

    Text, borders and ink are dark in the source and remain protected. Bright
    bubble interiors are protected only when the generated image also sees a
    neutral field. This both erases model-redrawn text inside real bubbles and
    prevents a false balloon over coloured skin from copying monochrome pixels.
    """
    if protected_mask.shape != (source.height, source.width):
        raise ValueError("protection mask shape does not match source image")
    source_gray = np.asarray(source.convert("L"), dtype=np.uint8)
    generated_rgb = np.asarray(
        generated.convert("RGB").resize(source.size, Image.Resampling.LANCZOS),
        dtype=np.int16,
    )
    generated_chroma = generated_rgb.max(axis=-1) - generated_rgb.min(axis=-1)
    dark_structure = source_gray <= dark_structure_threshold
    neutral_field = generated_chroma <= neutral_chroma_threshold
    return np.logical_and(protected_mask, np.logical_or(dark_structure, neutral_field))


def preserve_ink_overlay(
    source: Image.Image,
    generated: Image.Image,
    protected_mask: np.ndarray | None = None,
    gamma: float = 0.42,
) -> Image.Image:
    """Apply generated color under the source ink without copying white halos."""
    if not 0.05 <= gamma <= 2.0:
        raise ValueError("Ink gamma must be between 0.05 and 2.0")
    source_rgb = np.asarray(source.convert("RGB"), dtype=np.float32)
    generated_rgb = np.asarray(
        generated.convert("RGB").resize(source.size, Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    source_gray = np.asarray(source.convert("L"), dtype=np.float32) / 255.0
    ink_factor = np.power(source_gray, gamma)[..., None]
    result = np.clip(generated_rgb * ink_factor, 0, 255).astype(np.uint8)
    pure_black_ink = np.asarray(source.convert("L")) <= 8
    result[pure_black_ink] = source_rgb.astype(np.uint8)[pure_black_ink]
    if protected_mask is not None and protected_mask.any():
        result[protected_mask] = source_rgb.astype(np.uint8)[protected_mask]
    return Image.fromarray(result, mode="RGB")
