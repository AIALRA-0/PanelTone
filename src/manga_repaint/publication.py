"""Hard publication gate for reviewed deterministic colour renders."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .art_bible import BookArtBible
from .segment_graph import SegmentGraph

VERSION = "publication-manifest-v1"


@dataclass(frozen=True, slots=True)
class PagePublicationEvidence:
    page_index: int
    source_hash: str
    render_hash: str
    plan_hash: str
    qa_passed: bool
    human_reviewed: bool
    unresolved_regions: int = 0
    conflict_regions: int = 0


@dataclass(frozen=True, slots=True)
class PublicationManifest:
    job_id: str
    art_bible_hash: str
    renderer_version: str
    pages: tuple[PagePublicationEvidence, ...]
    passed: bool
    reasons: tuple[str, ...]
    manifest_hash: str = ""
    version: str = VERSION

    def with_hash(self) -> PublicationManifest:
        payload = asdict(self)
        payload["manifest_hash"] = ""
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return PublicationManifest(
            self.job_id,
            self.art_bible_hash,
            self.renderer_version,
            self.pages,
            self.passed,
            self.reasons,
            digest,
            self.version,
        )


def build_publication_manifest(
    *,
    job_id: str,
    art_bible: BookArtBible,
    segment_graphs: tuple[SegmentGraph, ...],
    page_evidence: tuple[PagePublicationEvidence, ...],
    renderer_version: str,
) -> PublicationManifest:
    reasons: set[str] = set()
    if not job_id or not renderer_version:
        raise ValueError("publication job and renderer are required")
    art_bible.validate()
    pages = {item.page_index for item in page_evidence}
    if len(pages) != len(page_evidence):
        raise ValueError("publication pages must be unique")
    bible_ok, bible_reasons = art_bible.publishable(pages)
    if not bible_ok:
        reasons.update(bible_reasons)
    graph_pages = {item.page_index for item in segment_graphs}
    if graph_pages != pages:
        reasons.add("segment_graph_page_mismatch")
    for graph in segment_graphs:
        graph.validate()
        graph_ok, graph_reasons = graph.publishable(maximum_unknown_ratio=0.0)
        if not graph_ok:
            reasons.update(graph_reasons)
    for page in page_evidence:
        if (
            page.page_index < 0
            or not page.source_hash
            or not page.render_hash
            or not page.plan_hash
        ):
            raise ValueError("invalid page publication evidence")
        if not page.qa_passed:
            reasons.add("qa_failed")
        if not page.human_reviewed:
            reasons.add("human_review_required")
        if page.unresolved_regions:
            reasons.add("unresolved_regions")
        if page.conflict_regions:
            reasons.add("conflicted_regions")
    return PublicationManifest(
        job_id=job_id,
        art_bible_hash=art_bible.state_hash,
        renderer_version=renderer_version,
        pages=page_evidence,
        passed=not reasons,
        reasons=tuple(sorted(reasons)),
    ).with_hash()


def save_publication_manifest(path: Path, manifest: PublicationManifest) -> None:
    if path.exists():
        raise FileExistsError("publication manifest already exists")
    if not manifest.passed:
        raise ValueError(f"publication blocked: {', '.join(manifest.reasons)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def publication_blockers(value: dict[str, Any]) -> tuple[str, ...]:
    """Small UI/API helper that never turns a warning into an override."""
    return tuple(str(item) for item in value.get("reasons", ()) if str(item))
