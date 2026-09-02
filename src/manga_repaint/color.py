from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .masks import ink_edge_mask


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
    base_chroma = 128.0 + (
        generated_lab[..., 1:].astype(np.float32) - 128.0
    ) * chroma_strength
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
        result_l = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)[..., 0].astype(
            np.float32
        )
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


def is_already_colorized(
    image: Image.Image,
    *,
    coverage_min: float = 0.90,
    pixel_min: int = 4096,
) -> bool:
    """Return whether a source page is already a substantially colour image.

    Covers, credits and publisher pages can be supplied as colour scans. They
    must not be sent through a monochrome-to-colour model, especially when a
    model invents structure that is absent from the source. White paper and
    black line art are excluded from the coverage calculation.
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
