"""Versioned book-level facts that control deterministic manga colour rendering.

Models may create proposals, but only accepted or locked records can be used by
the publishable renderer. The store is sidecar-only so existing SQLite files do
not need a migration and old live results remain recoverable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

VERSION = "book-art-bible-v1"
STATES = {"proposed", "accepted", "locked", "rejected", "superseded", "stale"}
_TRANSITIONS = {
    "proposed": {"accepted", "locked", "rejected", "stale"},
    "accepted": {"locked", "rejected", "superseded", "stale"},
    "locked": {"superseded", "stale"},
    "rejected": {"proposed", "superseded"},
    "superseded": set(),
    "stale": {"proposed", "accepted", "rejected", "superseded"},
}


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _colour(value: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(value) != 3 or any(
        type(channel) is not int or not 0 <= channel <= 255 for channel in value
    ):
        raise ValueError("colour must contain three integer sRGB channels")
    return value


def _ranges(value: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    for first, last in value:
        if first < 0 or last < first:
            raise ValueError("invalid page range")
    return value


@dataclass(frozen=True, slots=True)
class ArtPaletteSlot:
    slot_id: str
    material: str
    base_rgb: tuple[int, int, int]
    shadow_rgb: tuple[int, int, int] | None = None
    highlight_rgb: tuple[int, int, int] | None = None
    state: str = "proposed"
    confidence: float = 0.0
    evidence: tuple[str, ...] = ()
    locked_by_user: bool = False
    revision: int = 1

    def validate(self) -> None:
        if not self.slot_id or self.state not in STATES:
            raise ValueError("invalid palette slot")
        _colour(self.base_rgb)
        if self.shadow_rgb is not None:
            _colour(self.shadow_rgb)
        if self.highlight_rgb is not None:
            _colour(self.highlight_rgb)
        if not 0.0 <= self.confidence <= 1.0 or self.revision < 1:
            raise ValueError("invalid palette slot confidence or revision")
        if self.state in {"accepted", "locked"} and not self.evidence:
            raise ValueError("accepted palette slot requires evidence")
        if self.locked_by_user and self.state != "locked":
            raise ValueError("user-locked palette slot must be in locked state")


@dataclass(frozen=True, slots=True)
class Appearance:
    appearance_id: str
    character_id: str
    valid_page_ranges: tuple[tuple[int, int], ...]
    palette_slots: tuple[tuple[str, str], ...]
    state: str = "proposed"
    evidence: tuple[str, ...] = ()
    revision: int = 1
    outfit_ids: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.appearance_id or not self.character_id or self.state not in STATES:
            raise ValueError("invalid appearance")
        _ranges(self.valid_page_ranges)
        if len(dict(self.palette_slots)) != len(self.palette_slots):
            raise ValueError("appearance roles must be unique")


@dataclass(frozen=True, slots=True)
class Outfit:
    outfit_id: str
    character_id: str
    label: str
    valid_page_ranges: tuple[tuple[int, int], ...]
    palette_slots: tuple[tuple[str, str], ...]
    state: str = "proposed"
    evidence: tuple[str, ...] = ()
    revision: int = 1

    def validate(self) -> None:
        if not self.outfit_id or not self.character_id or self.state not in STATES:
            raise ValueError("invalid outfit")
        _ranges(self.valid_page_ranges)
        if len(dict(self.palette_slots)) != len(self.palette_slots):
            raise ValueError("outfit roles must be unique")
        if self.state in {"accepted", "locked"} and not self.evidence:
            raise ValueError("accepted outfit requires evidence")


@dataclass(frozen=True, slots=True)
class Character:
    character_id: str
    label: str
    state: str
    evidence: tuple[str, ...]
    appearances: tuple[Appearance, ...]
    revision: int = 1

    def validate(self) -> None:
        if not self.character_id or self.state not in STATES:
            raise ValueError("invalid character")
        if self.state in {"accepted", "locked"} and not self.evidence:
            raise ValueError("accepted character requires evidence")
        for appearance in self.appearances:
            appearance.validate()
            if appearance.character_id != self.character_id:
                raise ValueError("appearance belongs to another character")


@dataclass(frozen=True, slots=True)
class Scene:
    scene_id: str
    label: str
    page_ranges: tuple[tuple[int, int], ...]
    light_direction: str = "upper-left"
    light_temperature: float = 0.0
    ambient_strength: float = 0.0
    state: str = "proposed"
    evidence: tuple[str, ...] = ()
    revision: int = 1

    def validate(self) -> None:
        if not self.scene_id or self.state not in STATES:
            raise ValueError("invalid scene")
        _ranges(self.page_ranges)
        if self.light_direction not in {"upper-left", "upper-right", "top", "flat"}:
            raise ValueError("invalid scene light direction")
        if not -0.12 <= self.light_temperature <= 0.12:
            raise ValueError("invalid scene light temperature")
        if not 0.0 <= self.ambient_strength <= 0.2:
            raise ValueError("invalid scene ambient strength")


@dataclass(frozen=True, slots=True)
class RegionFact:
    page_index: int
    region_id: str
    label: int
    material: str
    slot_id: str | None
    character_id: str | None = None
    appearance_id: str | None = None
    confidence: float = 0.0
    state: str = "proposed"
    evidence: tuple[str, ...] = ()
    occludes: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.page_index < 0 or not self.region_id or self.label <= 0 or self.state not in STATES:
            raise ValueError("invalid region fact")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("invalid region confidence")
        if self.state in {"accepted", "locked"} and (not self.slot_id or not self.evidence):
            raise ValueError("accepted region requires a slot and evidence")


@dataclass(frozen=True, slots=True)
class BookArtBible:
    book_id: str
    revision: int
    source_snapshot_hash: str
    characters: tuple[Character, ...]
    palette_slots: tuple[ArtPaletteSlot, ...]
    scenes: tuple[Scene, ...]
    regions: tuple[RegionFact, ...]
    dependencies: tuple[tuple[str, str], ...]
    outfits: tuple[Outfit, ...] = ()
    state_hash: str = ""
    version: str = VERSION

    def validate(self) -> None:
        if self.version != VERSION or not self.book_id or self.revision < 1:
            raise ValueError("invalid art bible header")
        if not self.source_snapshot_hash:
            raise ValueError("art bible requires a source snapshot")
        slots = {item.slot_id: item for item in self.palette_slots}
        if len(slots) != len(self.palette_slots):
            raise ValueError("palette slot ids must be unique")
        for item in self.palette_slots:
            item.validate()
        characters = {item.character_id: item for item in self.characters}
        if len(characters) != len(self.characters):
            raise ValueError("character ids must be unique")
        appearances: set[str] = set()
        for item in self.characters:
            item.validate()
            for appearance in item.appearances:
                if appearance.appearance_id in appearances:
                    raise ValueError("appearance ids must be unique")
                appearances.add(appearance.appearance_id)
                if any(slot_id not in slots for _role, slot_id in appearance.palette_slots):
                    raise ValueError("appearance references an unknown palette slot")
        outfits = {item.outfit_id: item for item in self.outfits}
        if len(outfits) != len(self.outfits):
            raise ValueError("outfit ids must be unique")
        for outfit in self.outfits:
            outfit.validate()
            if outfit.character_id not in characters:
                raise ValueError("outfit references an unknown character")
            if any(slot_id not in slots for _role, slot_id in outfit.palette_slots):
                raise ValueError("outfit references an unknown palette slot")
        for character in self.characters:
            for appearance in character.appearances:
                if any(outfit_id not in outfits for outfit_id in appearance.outfit_ids):
                    raise ValueError("appearance references an unknown outfit")
        for scene in self.scenes:
            scene.validate()
        region_ids: set[tuple[int, str]] = set()
        for region in self.regions:
            region.validate()
            key = (region.page_index, region.region_id)
            if key in region_ids:
                raise ValueError("region ids must be unique within a page")
            region_ids.add(key)
            if region.slot_id is not None and region.slot_id not in slots:
                raise ValueError("region references an unknown palette slot")
            if region.character_id is not None and region.character_id not in characters:
                raise ValueError("region references an unknown character")
            if region.appearance_id is not None and region.appearance_id not in appearances:
                raise ValueError("region references an unknown appearance")
        body = asdict(replace(self, state_hash=""))
        expected = canonical_digest(body)
        if self.state_hash and self.state_hash != expected:
            raise ValueError("art bible state hash does not match content")

    def with_hash(self) -> BookArtBible:
        value = replace(self, state_hash="")
        return replace(value, state_hash=canonical_digest(asdict(value)))

    def publishable(self, pages: set[int] | None = None) -> tuple[bool, tuple[str, ...]]:
        target = [item for item in self.regions if pages is None or item.page_index in pages]
        reasons: list[str] = []
        if any(item.state not in {"accepted", "locked"} for item in target):
            reasons.append("unreviewed_regions")
        if any(item.slot_id is None for item in target):
            reasons.append("regions_without_palette_slot")
        used = {item.slot_id for item in target if item.slot_id}
        slots = {item.slot_id: item for item in self.palette_slots}
        if any(slots[slot_id].state not in {"accepted", "locked"} for slot_id in used):
            reasons.append("unreviewed_palette_slots")
        return not reasons, tuple(reasons)


def transition_state(item: Any, target: str, *, evidence: str | None = None) -> Any:
    current = str(item.state)
    if target not in STATES or target not in _TRANSITIONS[current]:
        raise ValueError(f"invalid state transition: {current} -> {target}")
    payload: dict[str, Any] = {"state": target}
    if evidence:
        payload["evidence"] = tuple((*item.evidence, evidence))
    if hasattr(item, "revision"):
        payload["revision"] = int(item.revision) + 1
    if hasattr(item, "locked_by_user"):
        payload["locked_by_user"] = target == "locked"
    return replace(item, **payload)


def affected_pages(before: BookArtBible, after: BookArtBible) -> set[int]:
    """Return the narrow page set invalidated by changed facts."""
    before_slots = {item.slot_id: item for item in before.palette_slots}
    after_slots = {item.slot_id: item for item in after.palette_slots}
    changed_slots = {
        key
        for key in before_slots.keys() | after_slots.keys()
        if before_slots.get(key) != after_slots.get(key)
    }
    before_regions = {(item.page_index, item.region_id): item for item in before.regions}
    after_regions = {(item.page_index, item.region_id): item for item in after.regions}
    pages = {
        key[0]
        for key in before_regions.keys() | after_regions.keys()
        if before_regions.get(key) != after_regions.get(key)
    }
    pages.update(item.page_index for item in after.regions if item.slot_id in changed_slots)
    return pages


class ArtBibleStore:
    """Atomic revision store with immutable history and one active pointer."""

    def __init__(self, job_dir: Path):
        self.root = job_dir / "analysis" / "art-bible"

    @property
    def active_path(self) -> Path:
        return self.root / "active.json"

    def save(self, bible: BookArtBible) -> Path:
        bible = bible.with_hash()
        bible.validate()
        self.root.mkdir(parents=True, exist_ok=True)
        revision = self.root / f"revision-{bible.revision:04d}.json"
        if revision.exists():
            raise FileExistsError(f"art bible revision already exists: {bible.revision}")
        payload = json.dumps(asdict(bible), ensure_ascii=False, sort_keys=True, indent=2)
        temporary = revision.with_suffix(".json.tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(revision)
        pointer = self.active_path.with_suffix(".json.tmp")
        pointer.write_text(payload, encoding="utf-8")
        pointer.replace(self.active_path)
        return revision

    def load(self, path: Path | None = None) -> BookArtBible:
        payload = json.loads((path or self.active_path).read_text(encoding="utf-8"))
        characters = tuple(
            Character(
                **{
                    **item,
                    "evidence": tuple(item.get("evidence", ())),
                    "appearances": tuple(
                        Appearance(
                            **{
                                **appearance,
                                "valid_page_ranges": tuple(
                                    tuple(value) for value in appearance["valid_page_ranges"]
                                ),
                                "palette_slots": tuple(
                                    tuple(value) for value in appearance["palette_slots"]
                                ),
                                "evidence": tuple(appearance.get("evidence", ())),
                                "outfit_ids": tuple(appearance.get("outfit_ids", ())),
                            }
                        )
                        for appearance in item.get("appearances", ())
                    ),
                }
            )
            for item in payload.get("characters", ())
        )
        slots = tuple(
            ArtPaletteSlot(
                **{
                    **item,
                    "base_rgb": tuple(item["base_rgb"]),
                    "shadow_rgb": tuple(item["shadow_rgb"]) if item.get("shadow_rgb") else None,
                    "highlight_rgb": (
                        tuple(item["highlight_rgb"]) if item.get("highlight_rgb") else None
                    ),
                    "evidence": tuple(item.get("evidence", ())),
                }
            )
            for item in payload.get("palette_slots", ())
        )
        bible = BookArtBible(
            book_id=payload["book_id"],
            revision=int(payload["revision"]),
            source_snapshot_hash=payload["source_snapshot_hash"],
            characters=characters,
            palette_slots=slots,
            scenes=tuple(
                Scene(
                    **{
                        **item,
                        "page_ranges": tuple(tuple(value) for value in item["page_ranges"]),
                        "evidence": tuple(item.get("evidence", ())),
                    }
                )
                for item in payload.get("scenes", ())
            ),
            regions=tuple(
                RegionFact(
                    **{
                        **item,
                        "evidence": tuple(item.get("evidence", ())),
                        "occludes": tuple(item.get("occludes", ())),
                    }
                )
                for item in payload.get("regions", ())
            ),
            dependencies=tuple(tuple(value) for value in payload.get("dependencies", ())),
            outfits=tuple(
                Outfit(
                    **{
                        **item,
                        "valid_page_ranges": tuple(
                            tuple(value) for value in item["valid_page_ranges"]
                        ),
                        "palette_slots": tuple(tuple(value) for value in item["palette_slots"]),
                        "evidence": tuple(item.get("evidence", ())),
                    }
                )
                for item in payload.get("outfits", ())
            ),
            state_hash=payload.get("state_hash", ""),
            version=payload.get("version", VERSION),
        )
        bible.validate()
        return bible
