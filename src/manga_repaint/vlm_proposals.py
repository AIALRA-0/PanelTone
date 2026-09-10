"""Strict boundary for non-spatial VLM material suggestions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .material_render import MATERIALS


@dataclass(frozen=True, slots=True)
class MaterialLabelProposal:
    region_id: int
    material: str
    confidence: float
    reason: str
    reviewed: bool = False


def parse_material_labels(
    value: str | dict[str, Any], *, expected_ids: set[int]
) -> tuple[MaterialLabelProposal, ...]:
    """Parse labels only; any spatial claim is rejected at this boundary."""
    payload = json.loads(value) if isinstance(value, str) else value
    if not isinstance(payload, dict) or set(payload) != {"labels"}:
        raise ValueError("VLM material payload must contain labels only")
    rows = payload["labels"]
    if not isinstance(rows, list):
        raise ValueError("VLM material labels must be an array")
    result: list[MaterialLabelProposal] = []
    seen: set[int] = set()
    allowed = {"id", "material", "confidence", "reason", "reviewed"}
    for row in rows:
        if not isinstance(row, dict) or not set(row).issubset(allowed):
            raise ValueError("VLM coordinates and unknown fields are not accepted")
        region_id = row.get("id")
        material = row.get("material")
        confidence = row.get("confidence")
        if type(region_id) is not int or region_id in seen or region_id not in expected_ids:
            raise ValueError("VLM material id is missing, duplicate or unexpected")
        if material not in MATERIALS:
            raise ValueError("VLM proposed an unsupported material")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            raise ValueError("VLM confidence must be numeric")
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("VLM confidence must be between zero and one")
        seen.add(region_id)
        result.append(
            MaterialLabelProposal(
                region_id,
                str(material),
                confidence,
                str(row.get("reason") or "")[:240],
                row.get("reviewed") is True,
            )
        )
    if seen != expected_ids:
        raise ValueError("VLM material response must contain every expected region exactly once")
    return tuple(sorted(result, key=lambda item: item.region_id))


def reviewed_materials(
    proposals: tuple[MaterialLabelProposal, ...], *, minimum_confidence: float = 0.8
) -> tuple[str | None, ...]:
    """Only an explicit review can promote a VLM suggestion into a mask label."""
    if not proposals:
        return ()
    maximum = max(item.region_id for item in proposals)
    output: list[str | None] = [None] * (maximum + 1)
    for item in proposals:
        if item.reviewed and item.confidence >= minimum_confidence:
            output[item.region_id] = item.material
    return tuple(output)
