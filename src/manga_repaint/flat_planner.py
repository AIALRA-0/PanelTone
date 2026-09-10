"""Conservative source-topology regions and model-independent colour proposals."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

from .color import image_sha256
from .material_render import (
    MATERIALS,
    MaterialPlan,
    MaterialRegion,
    MaterialSlot,
    decompose_source,
)

if TYPE_CHECKING:
    from .sam2_regions import Sam2Region


@dataclass(frozen=True, slots=True)
class RegionProposal:
    label: int
    bbox: tuple[int, int, int, int]
    area: int
    rgb: tuple[int, int, int] | None
    material: str
    confidence: float
    state: str
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FlatPlanProposal:
    plan: MaterialPlan
    proposals: tuple[RegionProposal, ...]
    unknown_ratio: float
    conflict_ratio: float


def source_topology_barrier(source: Image.Image, protected: np.ndarray) -> np.ndarray:
    """Return a source-owned fill barrier without treating print dots as objects."""
    layers = decompose_source(source, protected)
    gray = cv2.cvtColor(np.asarray(source.convert("RGB")), cv2.COLOR_RGB2GRAY)
    raw_edge = cv2.Canny(gray, 48, 144, L2gradient=True)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw_edge, 8)
    component_area = stats[:, cv2.CC_STAT_AREA]
    span = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    keep = (component_area >= 20) & (span >= 10)
    keep[0] = False
    contour_edge = keep[labels]
    barrier = layers.protected | layers.structural_ink | contour_edge
    barrier = cv2.morphologyEx(
        barrier.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    return cv2.dilate(barrier, np.ones((3, 3), np.uint8), iterations=1).astype(bool)


def _candidate_rgb(candidate: Image.Image, size: tuple[int, int]) -> np.ndarray:
    rgb = candidate.convert("RGB")
    if rgb.size != size:
        source_aspect = size[0] / max(size[1], 1)
        candidate_aspect = rgb.width / max(rgb.height, 1)
        if abs(candidate_aspect / source_aspect - 1.0) > 0.01:
            raise ValueError("candidate aspect does not match source")
        rgb = rgb.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(rgb)


def source_regions(
    source: Image.Image,
    protected: np.ndarray,
    *,
    minimum_area: int = 48,
) -> tuple[np.ndarray, tuple[tuple[int, tuple[int, int, int, int], int], ...]]:
    """Build fillable regions from source-owned barriers only.

    Candidate pixels never affect labels. Tiny print islands remain unknown
    instead of becoming thousands of fake objects.
    """
    if minimum_area < 1:
        raise ValueError("minimum area must be positive")
    barrier = source_topology_barrier(source, protected)
    traversable = (~barrier).astype(np.uint8)
    count, raw, stats, _ = cv2.connectedComponentsWithStats(traversable, 8)
    labels = np.zeros(raw.shape, dtype=np.uint16)
    regions: list[tuple[int, tuple[int, int, int, int], int]] = []
    page_area = source.width * source.height
    threshold = max(minimum_area, int(page_area * 0.00001))
    next_label = 1
    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area < threshold:
            continue
        if next_label > 65535:
            raise ValueError("source contains too many fillable regions")
        x = int(stats[component, cv2.CC_STAT_LEFT])
        y = int(stats[component, cv2.CC_STAT_TOP])
        width = int(stats[component, cv2.CC_STAT_WIDTH])
        height = int(stats[component, cv2.CC_STAT_HEIGHT])
        labels[raw == component] = next_label
        regions.append((next_label, (x, y, width, height), area))
        next_label += 1
    return labels, tuple(regions)


def _robust_colour(
    candidate: np.ndarray, mask: np.ndarray, *, prefer_chromatic: bool = False
) -> tuple[tuple[int, int, int] | None, float]:
    interior = cv2.erode(
        mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1
    ).astype(bool)
    pixels = candidate[interior]
    if len(pixels) < 16:
        return None, 0.0
    hsv = cv2.cvtColor(pixels.reshape(1, -1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    usable = (hsv[:, 2] >= 24) & (hsv[:, 2] <= 250)
    if usable.sum() < 16:
        return None, 0.0
    pixels = pixels[usable]
    if prefer_chromatic:
        chromatic = hsv[usable, 1] >= 24
        if int(chromatic.sum()) >= max(16, int(len(pixels) * 0.08)):
            pixels = pixels[chromatic]
    # A spatially dirty candidate may contain blue/pink stains. Quantize colour
    # samples and vote for one dominant albedo rather than averaging the stains
    lab = cv2.cvtColor(pixels.reshape(1, -1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    quantized = (lab // np.asarray([16, 16, 16], dtype=np.uint8)).astype(np.int16)
    keys, counts = np.unique(quantized, axis=0, return_counts=True)
    winning = keys[int(np.argmax(counts))]
    selected = np.all(quantized == winning, axis=1)
    colour = tuple(int(value) for value in np.median(pixels[selected], axis=0))
    confidence = float(selected.mean())
    return colour, confidence


def infer_material(rgb: tuple[int, int, int], source_gray: float) -> tuple[str, float]:
    """Return a broad material proposal, never an accepted semantic fact."""
    sample = np.asarray(rgb, dtype=np.uint8).reshape(1, 1, 3)
    hue, saturation, value = (int(item) for item in cv2.cvtColor(sample, cv2.COLOR_RGB2HSV)[0, 0])
    if saturation <= 18 and value >= 220:
        return "white", 0.88
    if source_gray <= 72 and value <= 100:
        return "hair", 0.52
    if (hue <= 18 or hue >= 174) and 28 <= saturation <= 170 and value >= 90:
        return "skin", 0.68
    if 5 <= hue <= 25 and saturation >= 70 and source_gray >= 100:
        return "wood", 0.48
    if saturation <= 26:
        return "other", 0.35
    return "cloth", 0.38


def propose_material_plan(
    source: Image.Image,
    candidate: Image.Image,
    protected: np.ndarray,
    *,
    palette_revision: str,
    evidence: str,
    minimum_area: int = 48,
) -> FlatPlanProposal:
    """Create an inspectable proposal from source regions and candidate votes.

    Every slot and region remains ``proposed``. A caller must explicitly review
    and accept it before the normal compositor can publish the result.
    """
    if not evidence.strip() or not palette_revision.strip():
        raise ValueError("proposal evidence and palette revision are required")
    source_rgb = np.asarray(source.convert("RGB"))
    gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY)
    candidate_rgb = _candidate_rgb(candidate, source.size)
    labels, regions = source_regions(source, protected, minimum_area=minimum_area)
    slots: list[MaterialSlot] = []
    bindings: list[MaterialRegion] = []
    proposals: list[RegionProposal] = []
    for label, bbox, area in regions:
        mask = labels == label
        colour, colour_confidence = _robust_colour(candidate_rgb, mask)
        reasons: list[str] = []
        if colour is None:
            labels[mask] = 0
            proposals.append(
                RegionProposal(
                    label,
                    bbox,
                    area,
                    None,
                    "other",
                    0.0,
                    "proposed",
                    ("no_stable_colour",),
                )
            )
            continue
        material, material_confidence = infer_material(colour, float(np.median(gray[mask])))
        if material not in MATERIALS:
            raise AssertionError("material inference returned an unsupported class")
        if material == "white" and max(colour) - min(colour) > 12:
            neutral = int(round(sum(colour) / 3))
            colour = (neutral, neutral, neutral)
        confidence = min(colour_confidence, material_confidence)
        if colour_confidence < 0.35:
            reasons.append("mixed_candidate_colours")
        if material_confidence < 0.6:
            reasons.append("material_needs_review")
        slot_key = hashlib.sha256(
            f"{material}|{colour[0] // 12}|{colour[1] // 12}|{colour[2] // 12}".encode()
        ).hexdigest()[:12]
        slot_id = f"proposal.{material}.{slot_key}"
        if all(item.slot_id != slot_id for item in slots):
            slots.append(
                MaterialSlot(
                    slot_id=slot_id,
                    material=material,
                    rgb=colour,
                    state="proposed",
                    evidence=evidence,
                )
            )
        bindings.append(MaterialRegion(label, slot_id, "proposed", evidence))
        proposals.append(
            RegionProposal(
                label,
                bbox,
                area,
                colour,
                material,
                confidence,
                "proposed",
                tuple(reasons),
            )
        )
    plan = MaterialPlan(
        source_hash=image_sha256(source),
        labels=labels,
        protected=protected.astype(bool),
        slots=tuple(slots),
        regions=tuple(bindings),
        palette_revision=palette_revision,
    )
    plan.validate(source)
    assigned = labels > 0
    unknown = ~assigned & ~source_topology_barrier(source, protected)
    return FlatPlanProposal(
        plan=plan,
        proposals=tuple(proposals),
        unknown_ratio=float(unknown.mean()),
        conflict_ratio=0.0,
    )


def propose_material_plan_with_object_masks(
    source: Image.Image,
    candidate: Image.Image,
    protected: np.ndarray,
    object_masks: Sequence[Sam2Region],
    *,
    palette_revision: str,
    evidence: str,
    material_labels: Sequence[str | None] | None = None,
    minimum_overlap: float = 0.35,
    minimum_area: int = 48,
) -> FlatPlanProposal:
    """Project object-level colour evidence onto source-owned fill regions.

    SAM2 may say that several disconnected source regions belong to one visual
    object.  It never supplies the final boundary: every output label remains a
    region cut exclusively by source ink.  This lets an arm, face or garment
    share one clean albedo without importing a generated silhouette.
    """
    if not 0.0 < minimum_overlap <= 1.0:
        raise ValueError("minimum overlap must be between zero and one")
    if not evidence.strip() or not palette_revision.strip():
        raise ValueError("proposal evidence and palette revision are required")
    if material_labels is not None and len(material_labels) != len(object_masks):
        raise ValueError("material labels must match object mask count")
    shape = (source.height, source.width)
    for item in object_masks:
        item.validate(shape)

    candidate_rgb = _candidate_rgb(candidate, source.size)
    source_rgb = np.asarray(source.convert("RGB"))
    gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY)
    labels, regions = source_regions(source, protected, minimum_area=minimum_area)
    barrier = source_topology_barrier(source, protected)

    evidence_rows: list[dict] = []
    for index, item in enumerate(object_masks):
        usable = item.mask & ~barrier
        colour, colour_confidence = _robust_colour(
            candidate_rgb, usable, prefer_chromatic=True
        )
        if colour is None:
            continue
        inferred, material_confidence = infer_material(colour, float(np.median(gray[usable])))
        supplied = material_labels[index] if material_labels is not None else None
        material = supplied if supplied in MATERIALS else inferred
        evidence_rows.append(
            {
                "index": index,
                "mask": usable,
                "area": int(usable.sum()),
                "score": float(item.score),
                "colour": colour,
                "colour_confidence": colour_confidence,
                "material": material,
                "material_confidence": material_confidence if supplied is None else 0.7,
            }
        )

    region_areas = {label: area for label, _bbox, area in regions}
    matches_by_label: dict[int, list[tuple[float, dict]]] = {}
    maximum_label = int(labels.max())
    for row in evidence_rows:
        counts = np.bincount(labels[row["mask"]], minlength=maximum_label + 1)
        specificity = 1.0 - min(
            row["area"] / max(source.width * source.height, 1), 1.0
        )
        for label in np.flatnonzero(counts[1:]) + 1:
            region_area = region_areas.get(int(label), 0)
            overlap = float(counts[label] / max(region_area, 1))
            if overlap < minimum_overlap:
                continue
            priority = overlap * 0.65 + row["score"] * 0.25 + specificity * 0.10
            matches_by_label.setdefault(int(label), []).append((priority, row))

    slots: list[MaterialSlot] = []
    bindings: list[MaterialRegion] = []
    proposals: list[RegionProposal] = []
    conflict_pixels = 0
    for label, bbox, area in regions:
        region_mask = labels == label
        matches = matches_by_label.get(label, [])
        matches.sort(key=lambda value: value[0], reverse=True)
        reasons: list[str] = []
        if len(matches) > 1 and matches[0][0] - matches[1][0] < 0.04:
            conflict_pixels += area
            reasons.append("overlapping_object_masks")
        if matches:
            row = matches[0][1]
            colour = row["colour"]
            material = row["material"]
            confidence = min(
                float(row["score"]),
                float(row["colour_confidence"]),
                float(row["material_confidence"]),
            )
            reasons.append(f"object_mask:{row['index']}")
        else:
            colour, colour_confidence = _robust_colour(candidate_rgb, region_mask)
            if colour is None:
                labels[region_mask] = 0
                proposals.append(
                    RegionProposal(
                        label,
                        bbox,
                        area,
                        None,
                        "other",
                        0.0,
                        "proposed",
                        ("no_stable_colour",),
                    )
                )
                continue
            material, material_confidence = infer_material(
                colour, float(np.median(gray[region_mask]))
            )
            confidence = min(colour_confidence, material_confidence)
            reasons.append("local_colour_fallback")
        if material == "white" and max(colour) - min(colour) > 12:
            neutral = int(round(sum(colour) / 3))
            colour = (neutral, neutral, neutral)
        slot_key = hashlib.sha256(
            f"{material}|{colour[0] // 12}|{colour[1] // 12}|{colour[2] // 12}".encode()
        ).hexdigest()[:12]
        slot_id = f"proposal.{material}.{slot_key}"
        if all(item.slot_id != slot_id for item in slots):
            slots.append(MaterialSlot(slot_id, material, colour, "proposed", evidence))
        bindings.append(MaterialRegion(label, slot_id, "proposed", evidence))
        proposals.append(
            RegionProposal(
                label,
                bbox,
                area,
                colour,
                material,
                confidence,
                "proposed",
                tuple(reasons),
            )
        )

    plan = MaterialPlan(
        source_hash=image_sha256(source),
        labels=labels,
        protected=protected.astype(bool),
        slots=tuple(slots),
        regions=tuple(bindings),
        palette_revision=palette_revision,
    )
    plan.validate(source)
    unknown = ~(labels > 0) & ~barrier
    return FlatPlanProposal(
        plan=plan,
        proposals=tuple(proposals),
        unknown_ratio=float(unknown.mean()),
        conflict_ratio=float(conflict_pixels / max(source.width * source.height, 1)),
    )


def accept_proposals(
    proposal: FlatPlanProposal,
    *,
    evidence: str,
    minimum_confidence: float = 0.75,
) -> MaterialPlan:
    """Accept only independently reviewed, sufficiently stable proposals."""
    if not evidence.strip():
        raise ValueError("review evidence is required")
    accepted_labels = {
        item.label for item in proposal.proposals if item.confidence >= minimum_confidence
    }
    accepted_slot_ids = {
        region.slot_id for region in proposal.plan.regions if region.label in accepted_labels
    }
    slots = tuple(
        replace(
            slot,
            state="accepted" if slot.slot_id in accepted_slot_ids else "proposed",
            evidence=evidence if slot.slot_id in accepted_slot_ids else slot.evidence,
        )
        for slot in proposal.plan.slots
    )
    regions = tuple(
        MaterialRegion(
            region.label,
            region.slot_id,
            "accepted" if region.label in accepted_labels else "proposed",
            evidence if region.label in accepted_labels else region.mask_evidence,
        )
        for region in proposal.plan.regions
    )
    return MaterialPlan(
        proposal.plan.source_hash,
        proposal.plan.labels.copy(),
        proposal.plan.protected.copy(),
        slots,
        regions,
        proposal.plan.palette_revision,
    )


def preview_plan(proposal: FlatPlanProposal) -> MaterialPlan:
    """Create an in-memory plan for isolated visual review, never publication."""
    usable = {item.label for item in proposal.proposals if item.rgb is not None}
    slot_ids = {
        region.slot_id for region in proposal.plan.regions if region.label in usable
    }
    return MaterialPlan(
        proposal.plan.source_hash,
        proposal.plan.labels.copy(),
        proposal.plan.protected.copy(),
        tuple(
            replace(slot, state="accepted", evidence="unreviewed-isolated-preview")
            if slot.slot_id in slot_ids
            else slot
            for slot in proposal.plan.slots
        ),
        tuple(
            replace(region, state="accepted", mask_evidence="unreviewed-isolated-preview")
            if region.label in usable
            else region
            for region in proposal.plan.regions
        ),
        proposal.plan.palette_revision,
    )
