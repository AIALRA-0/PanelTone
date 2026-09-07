"""Book-level colour planning artifacts.

The renderer used to make a colour decision independently for every panel.
This module is deliberately model-agnostic: it stores the evidence graph and
the deterministic plan that every candidate renderer must consume.  A model
may propose colours, but it cannot silently create or mutate identity state.
Artifacts are JSON sidecars so existing manifests and live SQLite files do not
need a schema migration.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

RENDERER_VERSION = "book-color-plan-v1"
STATE_VERSION = 1
_VALID_STATES = {
    "observed",
    "proposed",
    "accepted",
    "locked",
    "rejected",
    "conflicted",
    "superseded",
    "stale",
}


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def artifact_digest(value: Any) -> str:
    """Return the canonical digest used by book-level sidecars.

    Keeping this public lets the job runner link QA and render evidence to the
    exact same canonical representation without reaching into private helpers.
    """
    return _digest(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _hex_rgb(value: str) -> tuple[int, int, int] | None:
    text = value.strip().lstrip("#")
    if len(text) != 6:
        return None
    try:
        return tuple(int(text[index : index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]
    except ValueError:
        return None


def rgb_to_oklch(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    """Convert sRGB to OKLCH for perceptual palette comparisons."""
    values = np.asarray(rgb, dtype=np.float64) / 255.0
    linear = np.where(
        values <= 0.04045,
        values / 12.92,
        ((values + 0.055) / 1.055) ** 2.4,
    )
    lms = np.asarray(
        [
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005],
        ],
        dtype=np.float64,
    ) @ linear
    lms_cbrt = np.cbrt(np.maximum(lms, 0.0))
    lab = np.asarray(
        [
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660],
        ],
        dtype=np.float64,
    ) @ lms_cbrt
    chroma = float(np.hypot(lab[1], lab[2]))
    hue = float(np.degrees(np.arctan2(lab[2], lab[1])) % 360.0)
    return float(lab[0]), chroma, hue


@dataclass(frozen=True, slots=True)
class PaletteEvidence:
    source_kind: str
    page_index: int | None
    region_id: str | None
    path: str | None
    confidence: float
    evidence_hash: str


@dataclass(frozen=True, slots=True)
class PaletteSlot:
    slot_id: str
    identity_id: str | None
    appearance_id: str | None
    scene_id: str | None
    role: str
    canonical_oklch: tuple[float, float, float]
    tolerance_delta_e: float
    state: str
    evidence: tuple[PaletteEvidence, ...] = ()
    revision: int = 1

    def __post_init__(self) -> None:
        if self.state not in _VALID_STATES:
            raise ValueError(f"unknown palette slot state: {self.state}")


@dataclass(frozen=True, slots=True)
class IdentityNode:
    node_id: str
    page_index: int
    unit_index: int
    bbox: tuple[int, int, int, int]
    visual_key: str | None
    confidence: float


@dataclass(frozen=True, slots=True)
class IdentityEdge:
    source: str
    target: str
    relation: str
    confidence: float


@dataclass(frozen=True, slots=True)
class IdentityGraph:
    version: int
    job_id: str
    nodes: tuple[IdentityNode, ...]
    edges: tuple[IdentityEdge, ...]
    cannot_link: tuple[tuple[str, str], ...]
    analysis_hash: str


@dataclass(frozen=True, slots=True)
class SegmentNode:
    segment_id: str
    page_index: int
    panel_index: int
    bbox: tuple[int, int, int, int]
    area: int
    neighbours: tuple[str, ...]
    semantic_candidates: tuple[str, ...]
    confidence: float


@dataclass(frozen=True, slots=True)
class SegmentGraph:
    version: int
    job_id: str
    page_index: int
    width: int
    height: int
    nodes: tuple[SegmentNode, ...]
    analysis_hash: str


@dataclass(frozen=True, slots=True)
class SceneTransform:
    scene_id: str = "scene.default"
    lightness_delta: float = 0.0
    chroma_scale: float = 1.0
    hue_shift: float = 0.0


@dataclass(frozen=True, slots=True)
class AppearanceState:
    """A character appearance variant, separate from identity."""

    appearance_id: str
    identity_id: str | None
    attributes: tuple[tuple[str, str], ...]
    state: str
    revision: int = 1


@dataclass(frozen=True, slots=True)
class SceneState:
    """A scene/illumination context that may change without recolouring a role."""

    scene_id: str
    label: str
    state: str
    transform: SceneTransform = SceneTransform()
    revision: int = 1


@dataclass(frozen=True, slots=True)
class ColorBookState:
    version: int
    job_id: str
    source_snapshot_hash: str
    identity_graph_hash: str
    segment_graph_hash: str
    palette_slots: tuple[PaletteSlot, ...]
    scene_transforms: tuple[SceneTransform, ...]
    observed_palette_anchors: tuple[tuple[float, float, float], ...]
    revision: int
    state_hash: str
    dependencies: tuple[tuple[str, str], ...]
    appearances: tuple[AppearanceState, ...] = ()
    scenes: tuple[SceneState, ...] = ()


@dataclass(frozen=True, slots=True)
class RegionColorPlan:
    version: int
    job_id: str
    page_index: int
    page_source_hash: str
    identity_graph_hash: str
    segment_graph_hash: str
    palette_state_hash: str
    segment_slots: tuple[tuple[str, str], ...]
    reference_pools: tuple[tuple[str, tuple[str, ...]], ...]
    seed: int
    plan_hash: str


@dataclass(frozen=True, slots=True)
class RenderEvidence:
    version: int
    job_id: str
    page_index: int
    unit_index: int
    plan_hash: str
    model_id: str
    model_revision: str | None
    renderer_version: str
    source_hash: str
    generated_hash: str | None
    final_hash: str | None
    reference_hashes: tuple[str, ...]
    seed: int
    metadata: tuple[tuple[str, str], ...]
    qa_hash: str | None = None


def _artifact_dict(value: Any) -> dict[str, Any]:
    payload = asdict(value)
    return payload


class ColorStateStore:
    """Atomic sidecar store for book analysis and rendering evidence."""

    def __init__(self, job_dir: Path):
        self.root = job_dir / "analysis" / "color-state"

    def path(self, name: str) -> Path:
        return self.root / name

    def write(self, name: str, value: Any) -> None:
        payload = _artifact_dict(value) if hasattr(value, "__dataclass_fields__") else value
        _write_json(self.path(name), payload)

    def read(self, name: str) -> Any:
        """Read a sidecar without exposing a partially written JSON file."""
        return json.loads(self.path(name).read_text(encoding="utf-8"))

    def write_page(self, directory: str, page_index: int, value: Any) -> None:
        target = self.root / directory / f"page_{page_index:05d}.json"
        self.write(str(target.relative_to(self.root)), value)


def build_segment_graph(
    image: Image.Image,
    *,
    job_id: str,
    page_index: int,
    panel_index: int = 0,
    protected_mask: np.ndarray | None = None,
    min_area: int = 24,
) -> SegmentGraph:
    """Build a conservative line-enclosed region graph.

    This is intentionally an evidence graph, not a claim that every region is
    a semantic object.  It gives the planner stable boundaries and adjacency;
    semantic labels can be added later without changing the rendering API.
    """
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ink = gray <= 96
    if protected_mask is not None and protected_mask.shape == ink.shape:
        ink |= protected_mask
    barrier = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
    traversable = (barrier == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(traversable, 8)
    nodes: list[SegmentNode] = []
    label_to_id: dict[int, str] = {}
    height, width = gray.shape
    area_threshold = max(min_area, int(width * height * 0.00001))
    for label in range(1, count):
        x, y, box_width, box_height, area = (int(value) for value in stats[label])
        if area < area_threshold:
            continue
        segment_id = f"p{page_index:04d}-s{len(nodes):05d}"
        label_to_id[label] = segment_id
        nodes.append(
            SegmentNode(
                segment_id=segment_id,
                page_index=page_index,
                panel_index=panel_index,
                bbox=(x, y, box_width, box_height),
                area=area,
                neighbours=(),
                semantic_candidates=("unresolved",),
                confidence=0.0,
            )
        )
    neighbours: dict[str, set[str]] = {node.segment_id: set() for node in nodes}
    for label in label_to_id:
        label_mask = (labels == label).astype(np.uint8)
        nearby = cv2.dilate(label_mask, np.ones((3, 3), np.uint8), iterations=1)
        adjacent = set(int(value) for value in np.unique(labels[nearby > 0]))
        for other in adjacent:
            if other in label_to_id and other != label:
                left = label_to_id[label]
                right = label_to_id[other]
                neighbours[left].add(right)
                neighbours[right].add(left)
    final_nodes = tuple(
        SegmentNode(
            segment_id=node.segment_id,
            page_index=node.page_index,
            panel_index=node.panel_index,
            bbox=node.bbox,
            area=node.area,
            neighbours=tuple(sorted(neighbours[node.segment_id])),
            semantic_candidates=node.semantic_candidates,
            confidence=node.confidence,
        )
        for node in nodes
    )
    body = {
        "version": 1,
        "job_id": job_id,
        "page_index": page_index,
        "width": width,
        "height": height,
        "nodes": [_artifact_dict(node) for node in final_nodes],
    }
    return SegmentGraph(
        version=1,
        job_id=job_id,
        page_index=page_index,
        width=width,
        height=height,
        nodes=final_nodes,
        analysis_hash=_digest(body),
    )


def build_identity_graph(
    job_id: str,
    observations: Iterable[IdentityNode] = (),
) -> IdentityGraph:
    """Create an evidence graph without making irreversible identity claims."""
    nodes = tuple(observations)
    edges: list[IdentityEdge] = []
    cannot_link: list[tuple[str, str]] = []
    for index, left in enumerate(nodes):
        for right in nodes[index + 1 :]:
            if left.page_index == right.page_index and left.unit_index == right.unit_index:
                cannot_link.append(tuple(sorted((left.node_id, right.node_id))))
            elif left.visual_key and left.visual_key == right.visual_key:
                edges.append(IdentityEdge(left.node_id, right.node_id, "same_identity", 0.5))
    body = {
        "version": 1,
        "job_id": job_id,
        "nodes": [_artifact_dict(node) for node in nodes],
        "edges": [_artifact_dict(edge) for edge in edges],
        "cannot_link": cannot_link,
    }
    return IdentityGraph(
        version=1,
        job_id=job_id,
        nodes=nodes,
        edges=tuple(edges),
        cannot_link=tuple(cannot_link),
        analysis_hash=_digest(body),
    )


def build_color_book_state(
    job_id: str,
    source_snapshot: dict[int, str],
    identity_graph: IdentityGraph,
    segment_graph_hash: str,
    identities: Iterable[dict[str, Any]] = (),
    observed_palette_anchors: Iterable[tuple[float, float, float]] = (),
) -> ColorBookState:
    slots: list[PaletteSlot] = []
    appearances: list[AppearanceState] = []
    for item in identities:
        color = str(item.get("color") or "")
        rgb = _hex_rgb(color)
        if rgb is None:
            continue
        locked = bool(item.get("locked"))
        state = "locked" if locked else (
            "accepted" if float(item.get("confidence") or 0) >= 0.7 else "proposed"
        )
        identity_id = str(item.get("identity_id") or "unknown")
        role = str(item.get("region") or "unknown")
        evidence = PaletteEvidence(
            source_kind="manifest_identity",
            page_index=None,
            region_id=None,
            path=None,
            confidence=float(item.get("confidence") or 0.0),
            evidence_hash=_digest(item),
        )
        slots.append(
            PaletteSlot(
                slot_id=f"{identity_id}.{role}.base",
                identity_id=identity_id,
                appearance_id=None,
                scene_id=None,
                role=role,
                canonical_oklch=rgb_to_oklch(rgb),
                tolerance_delta_e=10.0,
                state=state,
                evidence=(evidence,),
                revision=1,
            )
        )
        appearances.append(
            AppearanceState(
                appearance_id=f"{identity_id}.default",
                identity_id=identity_id,
                attributes=(("role", role),),
                state=state,
            )
        )
    snapshot_hash = _digest(source_snapshot)
    anchors = tuple(tuple(float(value) for value in anchor) for anchor in observed_palette_anchors)
    scenes = (
        SceneState(
            scene_id="scene.default",
            label="book_default",
            state="observed",
        ),
    )
    dependencies = (
        ("source_snapshot", snapshot_hash),
        ("identity_graph", identity_graph.analysis_hash),
        ("segment_graph", segment_graph_hash),
        ("renderer", RENDERER_VERSION),
    )
    body = {
        "version": STATE_VERSION,
        "job_id": job_id,
        "source_snapshot_hash": snapshot_hash,
        "identity_graph_hash": identity_graph.analysis_hash,
        "segment_graph_hash": segment_graph_hash,
        "palette_slots": [_artifact_dict(slot) for slot in slots],
        "scene_transforms": [],
        "appearances": [_artifact_dict(item) for item in appearances],
        "scenes": [_artifact_dict(item) for item in scenes],
        "observed_palette_anchors": anchors,
        "revision": 1,
        "dependencies": dependencies,
    }
    return ColorBookState(
        version=STATE_VERSION,
        job_id=job_id,
        source_snapshot_hash=snapshot_hash,
        identity_graph_hash=identity_graph.analysis_hash,
        segment_graph_hash=segment_graph_hash,
        palette_slots=tuple(slots),
        scene_transforms=(),
        observed_palette_anchors=anchors,
        revision=1,
        state_hash=_digest(body),
        dependencies=dependencies,
        appearances=tuple(appearances),
        scenes=scenes,
    )


def make_region_color_plan(
    state: ColorBookState,
    segment_graph: SegmentGraph,
    *,
    page_source_hash: str,
    page_index: int,
    references: Iterable[Path] = (),
    page_seed: int = 0,
) -> RegionColorPlan:
    reference_paths = tuple(
        str(Path(path).resolve()) for path in references if Path(path).is_file()
    )
    pools = classify_reference_pools(reference_paths)
    segment_slots = tuple(
        (node.segment_id, "unresolved") for node in segment_graph.nodes
    )
    seed_payload = {
        "job_id": state.job_id,
        "page_index": page_index,
        "source_hash": page_source_hash,
        "state_hash": state.state_hash,
        "renderer": RENDERER_VERSION,
        "page_seed": page_seed,
    }
    seed = int(_digest(seed_payload)[:12], 16) % (2**31 - 1)
    body = {
        "version": 1,
        "job_id": state.job_id,
        "page_index": page_index,
        "page_source_hash": page_source_hash,
        "identity_graph_hash": state.identity_graph_hash,
        "segment_graph_hash": segment_graph.analysis_hash,
        "palette_state_hash": state.state_hash,
        "segment_slots": segment_slots,
        "reference_pools": pools,
        "seed": seed,
    }
    return RegionColorPlan(
        version=1,
        job_id=state.job_id,
        page_index=page_index,
        page_source_hash=page_source_hash,
        identity_graph_hash=state.identity_graph_hash,
        segment_graph_hash=segment_graph.analysis_hash,
        palette_state_hash=state.state_hash,
        segment_slots=segment_slots,
        reference_pools=pools,
        seed=seed,
        plan_hash=_digest(body),
    )


def classify_reference_pools(
    references: Iterable[Path],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Assign references to explicit, conservative pools.

    A reference is ``unassigned`` unless its path is deliberately placed in a
    folder named ``identity``, ``role`` or ``scene``.  This prevents automatic
    retrieval from silently becoming a false character identity claim.  A
    future curator can move or annotate a file without changing the plan API.
    """
    pools: dict[str, list[str]] = {
        "identity": [],
        "role": [],
        "scene": [],
        "unassigned": [],
    }
    for reference in references:
        path = Path(reference)
        if not path.is_file():
            continue
        parts = {part.casefold() for part in path.parts}
        pool = next(
            (name for name in ("identity", "role", "scene") if name in parts),
            "unassigned",
        )
        pools[pool].append(str(path.resolve()))
    return tuple((name, tuple(sorted(values))) for name, values in pools.items() if values)


def dependency_status(
    *,
    source_snapshot_hash: str,
    current_source_snapshot_hash: str,
    dependencies: Iterable[tuple[str, str]],
    current: dict[str, str],
) -> str:
    """Return a publish-safe state for changed upstream evidence."""
    if source_snapshot_hash != current_source_snapshot_hash:
        return "stale"
    if any(current.get(name) != value for name, value in dependencies):
        return "stale"
    return "current"
