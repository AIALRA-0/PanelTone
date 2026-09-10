"""Source-owned atomic regions and reviewable open-contour closures."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .color import image_sha256
from .flat_planner import source_regions, source_topology_barrier

VERSION = "segment-graph-v1"
CLOSURE_STATES = {"proposed", "accepted", "rejected", "stale"}


@dataclass(frozen=True, slots=True)
class AtomicRegion:
    region_id: str
    label: int
    bbox: tuple[int, int, int, int]
    area: int
    neighbours: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class VirtualClosure:
    closure_id: str
    start: tuple[int, int]
    end: tuple[int, int]
    confidence: float
    state: str = "proposed"
    evidence: tuple[str, ...] = ()

    def validate(self, shape: tuple[int, int]) -> None:
        height, width = shape
        if not self.closure_id or self.state not in CLOSURE_STATES:
            raise ValueError("invalid virtual closure")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("invalid virtual closure confidence")
        for x, y in (self.start, self.end):
            if not 0 <= x < width or not 0 <= y < height:
                raise ValueError("virtual closure endpoint outside source")
        if self.start == self.end:
            raise ValueError("virtual closure endpoints must differ")
        if self.state == "accepted" and not self.evidence:
            raise ValueError("accepted virtual closure requires review evidence")


@dataclass(frozen=True, slots=True)
class SegmentGraph:
    page_index: int
    source_hash: str
    labels_hash: str
    width: int
    height: int
    regions: tuple[AtomicRegion, ...]
    closures: tuple[VirtualClosure, ...] = ()
    unknown_ratio: float = 0.0
    version: str = VERSION

    def validate(self) -> None:
        if self.version != VERSION or self.page_index < 0 or self.width < 1 or self.height < 1:
            raise ValueError("invalid segment graph header")
        if not self.source_hash or not self.labels_hash or not 0.0 <= self.unknown_ratio <= 1.0:
            raise ValueError("invalid segment graph evidence")
        labels = {item.label for item in self.regions}
        if len(labels) != len(self.regions) or any(label <= 0 for label in labels):
            raise ValueError("atomic region labels must be unique and positive")
        for item in self.regions:
            if item.area < 1 or item.region_id != f"p{self.page_index:05d}.r{item.label:05d}":
                raise ValueError("invalid atomic region")
            if any(value not in labels or value == item.label for value in item.neighbours):
                raise ValueError("invalid atomic region adjacency")
        closure_ids: set[str] = set()
        for closure in self.closures:
            closure.validate((self.height, self.width))
            if closure.closure_id in closure_ids:
                raise ValueError("virtual closure ids must be unique")
            closure_ids.add(closure.closure_id)

    def publishable(self, *, maximum_unknown_ratio: float = 0.0) -> tuple[bool, tuple[str, ...]]:
        reasons: list[str] = []
        if self.unknown_ratio > maximum_unknown_ratio:
            reasons.append("unknown_regions")
        if any(item.state in {"proposed", "stale"} for item in self.closures):
            reasons.append("unreviewed_virtual_closures")
        return not reasons, tuple(reasons)


@dataclass(frozen=True, slots=True)
class AtomicSegmentMap:
    graph: SegmentGraph
    labels: np.ndarray

    def validate(self, source: Image.Image) -> None:
        self.graph.validate()
        if self.labels.shape != (source.height, source.width):
            raise ValueError("atomic label map dimensions do not match source")
        if not np.issubdtype(self.labels.dtype, np.integer):
            raise ValueError("atomic labels must be integers")
        digest = hashlib.sha256(self.labels.astype(np.uint16).tobytes()).hexdigest()
        if digest != self.graph.labels_hash or image_sha256(source) != self.graph.source_hash:
            raise ValueError("atomic segment map is stale")


def _adjacency(labels: np.ndarray, gap: int = 3) -> dict[int, set[int]]:
    result: dict[int, set[int]] = {
        int(value): set() for value in np.unique(labels) if int(value) > 0
    }
    for distance in range(1, gap + 1):
        for first, second in (
            (labels[:, distance:], labels[:, :-distance]),
            (labels[distance:, :], labels[:-distance, :]),
        ):
            mask = (first > 0) & (second > 0) & (first != second)
            pairs = np.unique(np.stack((first[mask], second[mask]), axis=1), axis=0)
            for left, right in pairs:
                result[int(left)].add(int(right))
                result[int(right)].add(int(left))
    return result


def build_segment_graph(
    source: Image.Image,
    protected: np.ndarray,
    *,
    page_index: int,
    minimum_area: int = 48,
) -> AtomicSegmentMap:
    labels, regions = source_regions(source, protected, minimum_area=minimum_area)
    neighbours = _adjacency(labels)
    atomic = tuple(
        AtomicRegion(
            region_id=f"p{page_index:05d}.r{label:05d}",
            label=label,
            bbox=bbox,
            area=area,
            neighbours=tuple(sorted(neighbours.get(label, ()))),
        )
        for label, bbox, area in regions
    )
    barrier = source_topology_barrier(source, protected)
    unknown = ~(labels > 0) & ~barrier
    labels_hash = hashlib.sha256(labels.astype(np.uint16).tobytes()).hexdigest()
    graph = SegmentGraph(
        page_index=page_index,
        source_hash=image_sha256(source),
        labels_hash=labels_hash,
        width=source.width,
        height=source.height,
        regions=atomic,
        unknown_ratio=float(unknown.mean()),
    )
    result = AtomicSegmentMap(graph, labels)
    result.validate(source)
    return result


class SegmentGraphStore:
    def __init__(self, root: Path):
        self.root = root

    def save(self, value: AtomicSegmentMap) -> tuple[Path, Path]:
        value.graph.validate()
        self.root.mkdir(parents=True, exist_ok=True)
        stem = f"page-{value.graph.page_index:05d}"
        labels_path = self.root / f"{stem}-labels.png"
        graph_path = self.root / f"{stem}.json"
        if labels_path.exists() or graph_path.exists():
            raise FileExistsError("segment graph revision already exists")
        labels_temp = labels_path.with_suffix(".png.tmp")
        with labels_temp.open("wb") as stream:
            Image.fromarray(value.labels.astype(np.uint16), mode="I;16").save(stream, format="PNG")
        labels_temp.replace(labels_path)
        graph_temp = graph_path.with_suffix(".json.tmp")
        graph_temp.write_text(
            json.dumps(asdict(value.graph), ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        graph_temp.replace(graph_path)
        return graph_path, labels_path
