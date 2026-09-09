"""Inspectable flat-colour rendering from source tone and reviewed material masks.

This is deliberately separate from candidate generation. The final renderer has
no candidate-image argument: a model may propose a *single* palette colour, but
its spatial chroma, texture and lighting cannot enter an approved render.
Label zero means UNKNOWN, not background or skin. Segmentation/identity accuracy
still requires independent review; a conforming render is not proof of either.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .color import image_sha256

VERSION = "material-plan-v1"
RENDERER_VERSION = "material-cel-v3"
MATERIALS = {
    "skin",
    "hair",
    "cloth",
    "hosiery",
    "wood",
    "metal",
    "glass",
    "background",
    "white",
    "patterned",
    "other",
}
STATES = {"proposed", "accepted", "rejected"}


def _linear(rgb: np.ndarray) -> np.ndarray:
    value = rgb.astype(np.float32) / 255.0
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def _encode(linear: np.ndarray) -> np.ndarray:
    value = np.clip(linear, 0, 1)
    srgb = np.where(value <= 0.0031308, value * 12.92, 1.055 * np.power(value, 1 / 2.4) - 0.055)
    return np.clip(np.rint(srgb * 255), 0, 255).astype(np.uint8)


def _chromaticity(rgb: np.ndarray) -> np.ndarray:
    linear = _linear(rgb)
    return linear / np.maximum(linear.sum(axis=-1, keepdims=True), 1e-6)


def _source_rgb(source: Image.Image) -> np.ndarray:
    rgba = source.convert("RGBA")
    white = Image.new("RGBA", source.size, (255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(white, rgba).convert("RGB"))


@dataclass(frozen=True)
class MaterialSlot:
    slot_id: str
    material: str
    rgb: tuple[int, int, int]
    state: str = "proposed"
    evidence: str = ""
    identity_id: str | None = None
    appearance_id: str | None = None
    shadow_rgb: tuple[int, int, int] | None = None
    highlight_rgb: tuple[int, int, int] | None = None


@dataclass(frozen=True)
class MaterialRegion:
    label: int
    slot_id: str
    state: str = "proposed"
    mask_evidence: str = ""


@dataclass(frozen=True)
class MaterialPlan:
    source_hash: str
    labels: np.ndarray
    protected: np.ndarray
    slots: tuple[MaterialSlot, ...]
    regions: tuple[MaterialRegion, ...]
    palette_revision: str

    def validate(self, source: Image.Image) -> None:
        shape = (source.height, source.width)
        if self.labels.shape != shape or self.protected.shape != shape:
            raise ValueError("material mask dimensions do not match source")
        if not np.issubdtype(self.labels.dtype, np.integer):
            raise ValueError("material labels must be integers")
        if self.labels.min() < 0 or self.labels.max() > 65535:
            raise ValueError("material labels must fit uint16; zero means UNKNOWN")
        if self.protected.dtype != np.bool_:
            raise ValueError("protected mask must be boolean")
        if image_sha256(source) != self.source_hash:
            raise ValueError("material plan is stale for this source")
        if not self.palette_revision:
            raise ValueError("material plan requires a palette revision")
        slots = {slot.slot_id: slot for slot in self.slots}
        if len(slots) != len(self.slots) or any(not key for key in slots):
            raise ValueError("palette slot identifiers must be unique and nonempty")
        for slot in self.slots:
            if slot.material not in MATERIALS or slot.state not in STATES:
                raise ValueError("unknown material or palette state")
            if len(slot.rgb) != 3 or any(type(c) is not int or not 0 <= c <= 255 for c in slot.rgb):
                raise ValueError("palette colour must be three integer sRGB channels")
            if slot.state == "accepted" and not slot.evidence.strip():
                raise ValueError("accepted palette colour requires review evidence")
            if slot.material == "white" and max(slot.rgb) - min(slot.rgb) > 12:
                raise ValueError("white material cannot use a saturated palette colour")
            for layer_name, layer_colour in (
                ("shadow", slot.shadow_rgb),
                ("highlight", slot.highlight_rgb),
            ):
                if layer_colour is not None and (
                    len(layer_colour) != 3
                    or any(type(c) is not int or not 0 <= c <= 255 for c in layer_colour)
                ):
                    raise ValueError(f"{layer_name} colour must be three integer sRGB channels")
        regions = {region.label: region for region in self.regions}
        if len(regions) != len(self.regions) or any(key <= 0 for key in regions):
            raise ValueError("region identifiers must be unique and positive")
        present = set(int(value) for value in np.unique(self.labels)) - {0}
        if present != set(regions):
            raise ValueError("every nonzero raster label needs exactly one region binding")
        for region in self.regions:
            if region.slot_id not in slots or region.state not in STATES:
                raise ValueError("unknown region slot or state")
            if region.state == "accepted" and not region.mask_evidence.strip():
                raise ValueError("accepted material region requires mask review evidence")


def _structural_ink_mask(source: Image.Image) -> np.ndarray:
    """Keep drawn contours while rejecting isolated halftone dots.

    A raw darkness threshold cannot distinguish a black contour from a black
    screentone dot. Connected print strokes are retained, while tiny isolated
    components are treated as tone samples and consumed by the cel-shadow
    estimator instead of being stamped back over the paint.
    """
    gray = cv2.cvtColor(_source_rgb(source), cv2.COLOR_RGB2GRAY)
    dark = (gray <= 96).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    area = stats[:, cv2.CC_STAT_AREA]
    width = stats[:, cv2.CC_STAT_WIDTH]
    height = stats[:, cv2.CC_STAT_HEIGHT]
    span = np.maximum(width, height)
    thin = np.minimum(width, height)
    keep = (area >= 12) & ((span >= 8) | (area >= 32) | (thin <= 2))
    keep[0] = False
    core = keep[labels]
    # Restore antialiased fringes only around a retained structural stroke
    # instead of protecting every mid-gray screentone pixel
    fringe = cv2.dilate(core.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return core | (fringe & (gray <= 192))


def protection_mask(source: Image.Image, supplied: np.ndarray) -> np.ndarray:
    """Preserve reviewed text/panel masks plus structural source ink."""
    rgb = _source_rgb(source)
    if supplied.shape != rgb.shape[:2]:
        raise ValueError("protection dimensions do not match source")
    return supplied.astype(bool) | _structural_ink_mask(source)


def _accepted_masks(plan: MaterialPlan, protected: np.ndarray):
    slots = {slot.slot_id: slot for slot in plan.slots}
    for region in plan.regions:
        slot = slots[region.slot_id]
        if region.state == "accepted" and slot.state == "accepted":
            yield region, slot, (plan.labels == region.label) & ~protected


def region_chroma_statistics(rgb: np.ndarray, mask: np.ndarray) -> dict:
    """Measure intra-region stains without assuming a particular palette colour.

    The region must be an independently reviewed single-albedo area. Applying
    this test to an entire multicolour panel would be meaningless. Normalizing
    in linear RGB removes scalar shading before measuring colour variation.
    """
    if rgb.shape[:2] != mask.shape or mask.dtype != np.bool_:
        raise ValueError("chroma statistics require a same-size boolean region mask")
    bright = mask & (np.max(rgb, axis=-1) >= 32)
    y, x = np.where(bright)
    if not len(x):
        return {"samples": 0, "dispersion_p95": 0.0, "low_frequency_residual_p95": 0.0}
    bounds = np.s_[y.min() : y.max() + 1, x.min() : x.max() + 1]
    local = bright[bounds]
    chroma = _chromaticity(rgb[bounds])
    center = np.median(chroma[local], axis=0)
    residual = np.linalg.norm(chroma[local] - center, axis=-1)
    low_frequency = []
    weights = local.astype(np.float32)
    for sigma in (3.0, 9.0, 21.0):
        denominator = cv2.GaussianBlur(weights, (0, 0), sigma)
        numerator = cv2.GaussianBlur(chroma * weights[..., None], (0, 0), sigma)
        blurred = numerator / np.maximum(denominator[..., None], 1e-6)
        error = np.linalg.norm(blurred[local] - center, axis=-1)
        low_frequency.append(float(np.quantile(error, 0.95)))
    return {
        "samples": len(x),
        "dispersion_p95": float(np.quantile(residual, 0.95)),
        "low_frequency_residual_p95": max(low_frequency),
    }


def _chroma_residual_statistics(rgb: np.ndarray, expected: np.ndarray, mask: np.ndarray) -> dict:
    """Measure unexplained colour structure after subtracting authored layers."""
    bright = mask & (np.max(rgb, axis=-1) >= 32) & (np.max(expected, axis=-1) >= 32)
    y, x = np.where(bright)
    if not len(x):
        return {"dispersion_p95": 0.0, "low_frequency_residual_p95": 0.0}
    bounds = np.s_[y.min() : y.max() + 1, x.min() : x.max() + 1]
    local = bright[bounds]
    residual = _chromaticity(rgb[bounds]) - _chromaticity(expected[bounds])
    magnitude = np.linalg.norm(residual[local], axis=-1)
    weights = local.astype(np.float32)
    low_frequency = []
    for sigma in (3.0, 9.0, 21.0):
        numerator = cv2.GaussianBlur(residual * weights[..., None], (0, 0), sigma)
        denominator = cv2.GaussianBlur(weights, (0, 0), sigma)
        blurred = numerator / np.maximum(denominator[..., None], 1e-6)
        low_frequency.append(float(np.quantile(np.linalg.norm(blurred[local], axis=-1), 0.95)))
    return {
        "dispersion_p95": float(np.quantile(magnitude, 0.95)),
        "low_frequency_residual_p95": max(low_frequency),
    }


def _derived_layers(slot: MaterialSlot) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return book-stable base, shadow and highlight colours in linear light."""
    base_rgb = np.asarray(slot.rgb, dtype=np.uint8)
    base = _linear(base_rgb)
    if slot.shadow_rgb is not None:
        shadow = _linear(np.asarray(slot.shadow_rgb, dtype=np.uint8))
    else:
        # A material-aware shadow is a separate paint layer, not black opacity
        shadow_tints = {
            "skin": np.array([0.32, 0.08, 0.06], dtype=np.float32),
            "hair": np.array([0.08, 0.06, 0.09], dtype=np.float32),
            "hosiery": np.array([0.07, 0.08, 0.12], dtype=np.float32),
            "wood": np.array([0.22, 0.09, 0.04], dtype=np.float32),
        }
        tint = shadow_tints.get(slot.material, base * 0.35)
        shadow = np.clip(base * 0.62 + tint * 0.16, 0, 1)
    if slot.highlight_rgb is not None:
        highlight = _linear(np.asarray(slot.highlight_rgb, dtype=np.uint8))
    else:
        # Highlights retain the material colour instead of becoming white holes
        highlight = np.clip(base * 0.82 + 0.18, 0, 1)
    return base, shadow, highlight


def _region_tone_layers(
    gray: np.ndarray,
    mask: np.ndarray,
    *,
    preserve_pattern: bool,
) -> tuple[tuple[slice, slice], np.ndarray, np.ndarray, np.ndarray]:
    """Convert print tone into discrete paint layers for one material.

    Ordinary materials do not retain per-dot opacity. The density of dots and
    hatching selects a coherent cel-shadow layer. Only a region explicitly
    declared as patterned may retain a weak achromatic source pattern.
    """
    y, x = np.where(mask)
    if not len(x):
        empty = np.empty((0, 0), dtype=np.float32)
        return (slice(0, 0), slice(0, 0)), empty, empty, empty
    padding = 12
    y0, y1 = max(0, y.min() - padding), min(gray.shape[0], y.max() + padding + 1)
    x0, x1 = max(0, x.min() - padding), min(gray.shape[1], x.max() + padding + 1)
    local_gray = gray[y0:y1, x0:x1]

    # Descreen before deciding paint layers. This makes a 50% dot field one
    # shadow decision instead of thousands of transparent black pinholes
    descreened = cv2.medianBlur(local_gray, 7)
    broad = cv2.GaussianBlur(descreened, (0, 0), 2.2).astype(np.float32) / 255.0
    raw = local_gray.astype(np.float32) / 255.0

    # Four stable steps represent paper/base, light shade, shadow and deep
    # shadow. These are authored colour layers, not source gray used as alpha
    shade = np.select(
        (broad >= 0.91, broad >= 0.80, broad >= 0.58),
        (0.0, 0.38, 0.72),
        default=1.0,
    ).astype(np.float32)

    # Highlights require an explicit region in a future review plan. Guessing
    # them from white gaps in a screentone would recreate the spotted mask look
    highlight = np.zeros_like(shade)
    if preserve_pattern:
        residual = np.clip((raw + 0.04) / np.maximum(broad + 0.04, 0.08), 0.78, 1.06)
        residual = 1.0 + (residual - 1.0) * 0.22
    else:
        residual = np.ones_like(raw)

    return (slice(y0, y1), slice(x0, x1)), shade, highlight, residual


def render_material_flats(source: Image.Image, plan: MaterialPlan) -> Image.Image:
    """Render an inspectable colour-guidance preview.

    This output proves that accepted regions and palette slots are coherent,
    but it is not the final illustrated page. The production colourizer must
    consume sparse hints derived from this plan and create the actual material,
    shadow and highlight rendering. Unknown or proposed pixels remain source
    gray and make publishable QA fail.
    """
    plan.validate(source)
    rgb = _source_rgb(source)
    protected = protection_mask(source, plan.protected)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    result = rgb.copy()
    for _, slot, mask in _accepted_masks(plan, protected):
        if not mask.any():
            continue
        base, shadow, highlight_colour = _derived_layers(slot)
        bounds, shade, highlight, detail = _region_tone_layers(
            gray,
            mask,
            preserve_pattern=slot.material == "patterned",
        )
        painted = base[None, None, :] * (1.0 - shade[..., None])
        painted += shadow[None, None, :] * shade[..., None]
        painted = painted * (1.0 - highlight[..., None])
        painted += highlight_colour[None, None, :] * highlight[..., None]
        painted *= detail[..., None]
        local_mask = mask[bounds]
        result[bounds][local_mask] = _encode(painted[local_mask])
    result[protected] = rgb[protected]
    return Image.fromarray(result)


def build_sparse_color_hint(
    source: Image.Image,
    plan: MaterialPlan,
    *,
    radius: int = 6,
    max_points_per_region: int = 8,
) -> Image.Image:
    """Create an RGBA point-hint layer for a reference-guided colourizer.

    Full masks encourage a model to behave like a paint overlay. This artifact
    instead places a few unambiguous palette samples deep inside each reviewed
    material region. Transparent pixels mean no hint. It is safe to send to a
    model that supports colour scribbles because neither source geometry nor a
    full-page opacity field is encoded in it.
    """
    if radius < 1 or max_points_per_region < 1:
        raise ValueError("hint radius and point limit must be positive")
    plan.validate(source)
    protected = protection_mask(source, plan.protected)
    hint = np.zeros((source.height, source.width, 4), dtype=np.uint8)
    yy, xx = np.ogrid[: source.height, : source.width]
    for _, slot, mask in _accepted_masks(plan, protected):
        available = mask.astype(np.uint8)
        component_count, components, stats, _ = cv2.connectedComponentsWithStats(
            available, connectivity=8
        )
        point_budget = max_points_per_region
        for component in range(1, component_count):
            if point_budget <= 0:
                break
            if int(stats[component, cv2.CC_STAT_AREA]) < 4:
                continue
            component_mask = components == component
            distance = cv2.distanceTransform(component_mask.astype(np.uint8), cv2.DIST_L2, 5)
            while point_budget > 0:
                flat_index = int(np.argmax(distance))
                clearance = float(distance.flat[flat_index])
                if clearance < 1.0:
                    break
                y, x = np.unravel_index(flat_index, distance.shape)
                dot_radius = max(1, min(radius, int(clearance)))
                dot = (xx - x) ** 2 + (yy - y) ** 2 <= dot_radius**2
                dot &= component_mask & ~protected
                hint[dot, :3] = np.asarray(slot.rgb, dtype=np.uint8)
                hint[dot, 3] = 255
                suppress_radius = max(radius * 6, dot_radius * 4)
                distance[(xx - x) ** 2 + (yy - y) ** 2 <= suppress_radius**2] = 0
                point_budget -= 1
    return Image.fromarray(hint, mode="RGBA")


def evaluate_material_render(
    source: Image.Image,
    result: Image.Image,
    reference: MaterialPlan,
) -> dict:
    """Check against reviewed masks/slots, never against model candidate colours.

    The report explicitly distinguishes renderer conformance from semantic
    correctness. A mask accepted in a file is recorded evidence, not a machine
    proof that the region really is skin, clothing or a particular identity.
    """
    reference.validate(source)
    if result.size != source.size:
        return {"passed": False, "reasons": ["dimension_mismatch"], "regions": []}
    original = _source_rgb(source)
    output = np.asarray(result.convert("RGB"))
    protected = protection_mask(source, reference.protected)
    diff = np.abs(output.astype(np.int16) - original.astype(np.int16))
    expected_output = np.asarray(render_material_flats(source, reference))
    maximum = int(diff[protected].max()) if protected.any() else 0
    covered = protected.copy()
    reasons = ["protected_pixels_changed"] if maximum else []
    rows = []
    for region, slot, mask in _accepted_masks(reference, protected):
        covered |= mask
        # Avoid chromaticity division near quantized black; tone is still
        # checked on *every* pixel by the source/albedo lightness residual
        bright = mask & (np.max(output, axis=2) >= 32)
        error = np.linalg.norm(
            _chromaticity(output[bright]) - _chromaticity(expected_output[bright]), axis=-1
        )
        p95 = float(np.quantile(error, 0.95)) if error.size else 0.0
        worst = float(error.max()) if error.size else 0.0
        stain_metrics = _chroma_residual_statistics(output, expected_output, mask)
        render_error = np.max(
            np.abs(output[mask].astype(np.int16) - expected_output[mask].astype(np.int16)), axis=1
        )
        render_error_p95 = float(np.quantile(render_error, 0.95)) if render_error.size else 0.0
        failed = (
            p95 > 0.04
            or worst > 0.12
            or render_error_p95 > 1.0
            or stain_metrics["low_frequency_residual_p95"] > 0.04
        )
        rows.append(
            {
                "label": region.label,
                "slot_id": slot.slot_id,
                "material": slot.material,
                "pixels": int(mask.sum()),
                "chromaticity_error_p95": p95,
                "chromaticity_error_max": worst,
                "layer_render_error_p95": render_error_p95,
                "chroma_statistics": stain_metrics,
                "passed": not failed,
            }
        )
        if failed:
            reasons.append(f"region_color_or_shading_mismatch:{region.label}")
    unknown = ~covered
    if unknown.any():
        reasons.append("unreviewed_material_regions")
    if np.any(diff[unknown]):
        reasons.append("unknown_pixels_changed")
    return {
        "passed": not reasons,
        "renderer": RENDERER_VERSION,
        "reasons": reasons,
        "protected_pixel_diff": maximum,
        "unknown_pixels": int(unknown.sum()),
        "unknown_ratio": float(np.mean(unknown)),
        "regions": rows,
        "semantic_status": "requires_independent_mask_and_palette_review",
        "palette_revision": reference.palette_revision,
    }


def propose_region_colour(candidate: Image.Image, mask: np.ndarray) -> dict:
    """Robust colour vote, always PROPOSED even if homogeneous/confident."""
    rgb = np.asarray(candidate.convert("RGB"))
    if mask.shape != rgb.shape[:2] or mask.dtype != np.bool_:
        raise ValueError("colour vote requires a same-size boolean mask")
    interior = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    pixels = rgb[interior & (np.max(rgb, axis=2) >= 40)]
    if len(pixels) < 16:
        return {"state": "proposed", "usable": False, "reason": "insufficient_interior"}
    normalized = _chromaticity(pixels)
    median = np.median(normalized, axis=0)
    spread = float(np.quantile(np.linalg.norm(normalized - median, axis=1), 0.90))
    return {
        "state": "proposed",
        "usable": spread <= 0.10,
        "rgb": np.rint(np.median(pixels, axis=0)).astype(int).tolist(),
        "chromaticity_spread_p90": spread,
        "reason": "needs_palette_review" if spread <= 0.10 else "conflicting_candidate_colours",
    }


def labels_from_masks(masks: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize proposals without letting later masks overwrite conflicts."""
    if not masks:
        raise ValueError("at least one mask is required")
    shape = next(iter(masks.values())).shape
    labels = np.zeros(shape, dtype=np.uint16)
    count = np.zeros(shape, dtype=np.uint32)
    for label, mask in masks.items():
        if type(label) is not int or not 1 <= label <= 65535:
            raise ValueError("invalid region label")
        if mask.shape != shape or mask.dtype != np.bool_ or mask.ndim != 2:
            raise ValueError("all masks must be same-size two-dimensional booleans")
        labels[mask] = label
        count[mask] += 1
    conflict = count > 1
    labels[conflict] = 0
    return labels, conflict


def _file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_material_plan(directory: Path, source: Image.Image, plan: MaterialPlan) -> Path:
    """Write a new immutable review bundle; never overwrite previous evidence."""
    plan.validate(source)
    directory.mkdir(parents=True, exist_ok=False)
    labels_path, protection_path = directory / "labels.png", directory / "protected.png"
    Image.fromarray(plan.labels.astype(np.uint16)).save(labels_path)
    Image.fromarray(plan.protected.astype(np.uint8) * 255).save(protection_path)
    payload = {
        "version": VERSION,
        "source_hash": plan.source_hash,
        "palette_revision": plan.palette_revision,
        "slots": [asdict(slot) for slot in plan.slots],
        "regions": [asdict(region) for region in plan.regions],
        "labels_sha256": _file_hash(labels_path),
        "protected_sha256": _file_hash(protection_path),
    }
    # Manifest is the completion marker; failed writes leave no readable bundle
    manifest = directory / "plan.json"
    temporary = directory / "plan.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(manifest)
    return manifest


def load_material_plan(manifest: Path, source: Image.Image) -> MaterialPlan:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data["version"] != VERSION:
        raise ValueError("unsupported material plan version")
    for name in ("labels", "protected"):
        if _file_hash(manifest.parent / f"{name}.png") != data[f"{name}_sha256"]:
            raise ValueError(f"material {name} raster changed after review")
    with Image.open(manifest.parent / "labels.png") as image:
        labels = np.asarray(image, dtype=np.uint16).copy()
    with Image.open(manifest.parent / "protected.png") as image:
        protected = np.asarray(image) > 0
    plan = MaterialPlan(
        data["source_hash"],
        labels,
        protected,
        tuple(
            MaterialSlot(
                **{
                    **slot,
                    "rgb": tuple(slot["rgb"]),
                    "shadow_rgb": (
                        tuple(slot["shadow_rgb"]) if slot.get("shadow_rgb") is not None else None
                    ),
                    "highlight_rgb": (
                        tuple(slot["highlight_rgb"])
                        if slot.get("highlight_rgb") is not None
                        else None
                    ),
                }
            )
            for slot in data["slots"]
        ),
        tuple(MaterialRegion(**region) for region in data["regions"]),
        data["palette_revision"],
    )
    plan.validate(source)
    return plan
