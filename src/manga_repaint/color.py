from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


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
    return _rgb_to_lab(np.asarray(image.convert("RGB")))[..., 0]


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
    source_lab = _rgb_to_lab(source_rgb)
    generated_lab = _rgb_to_lab(generated_rgb)
    generated_lab[..., 0] = source_lab[..., 0]
    generated_lab[..., 1:] *= chroma_strength
    return Image.fromarray(_lab_to_rgb_gamut_mapped(generated_lab), mode="RGB")


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


def composite_strict_colorization(
    source: Image.Image,
    generated: Image.Image,
    protected_mask: np.ndarray,
    chroma_strength: float = 1.0,
    ink_core_threshold: int = 64,
) -> Image.Image:
    """Keep source luminance while retaining generated color outside protected pixels.

    Only high-confidence dark ink cores are copied verbatim. Treating every gray
    screentone pixel as line art would overwrite nearly the entire colorized page.
    """
    if not 0 <= ink_core_threshold <= 255:
        raise ValueError("Ink core threshold must be between 0 and 255")
    colorized = preserve_luminance_lab(source, generated, chroma_strength)
    source_gray = np.asarray(source.convert("L"))
    strict_mask = np.logical_or(protected_mask, source_gray <= ink_core_threshold)
    # Scanned manga lines often have antialiased gray pixels above the ink
    # threshold.  Preserve a narrow edge guard as well, otherwise generated
    # chroma can bridge a light gray clothing/skin boundary and create a
    # visible color block on the neighboring region.
    edge_low = max(8, ink_core_threshold // 2)
    edge_high = min(255, max(edge_low + 1, ink_core_threshold * 2))
    edges = cv2.Canny(source_gray, edge_low, edge_high)
    edge_guard = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)
    strict_mask = np.logical_or(strict_mask, edge_guard.astype(bool))
    return composite_protected(source, colorized, strict_mask)


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
