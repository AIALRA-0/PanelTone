from manga_repaint.art_bible import (
    ArtPaletteSlot,
    BookArtBible,
    RegionFact,
)
from manga_repaint.publication import PagePublicationEvidence, build_publication_manifest
from manga_repaint.segment_graph import SegmentGraph


def _bible() -> BookArtBible:
    slot = ArtPaletteSlot(
        "page.skin",
        "skin",
        (232, 174, 148),
        state="accepted",
        confidence=1.0,
        evidence=("review",),
    )
    return BookArtBible(
        book_id="book",
        revision=1,
        source_snapshot_hash="source",
        characters=(),
        palette_slots=(slot,),
        scenes=(),
        regions=(
            RegionFact(0, "r1", 1, "skin", "page.skin", state="accepted", evidence=("review",)),
        ),
        dependencies=(),
    ).with_hash()


def _graph() -> SegmentGraph:
    return SegmentGraph(0, "source", "labels", 10, 10, (), unknown_ratio=0.0)


def _page(reviewed=False, qa=True):
    return PagePublicationEvidence(0, "source", "render", "plan", qa, reviewed)


def test_publication_gate_blocks_unreviewed_or_failed_page() -> None:
    manifest = build_publication_manifest(
        job_id="book",
        art_bible=_bible(),
        segment_graphs=(_graph(),),
        page_evidence=(_page(),),
        renderer_version="renderer",
    )
    assert not manifest.passed
    assert "human_review_required" in manifest.reasons


def test_publication_gate_accepts_fully_reviewed_evidence() -> None:
    manifest = build_publication_manifest(
        job_id="book",
        art_bible=_bible(),
        segment_graphs=(_graph(),),
        page_evidence=(_page(reviewed=True),),
        renderer_version="renderer",
    )
    assert manifest.passed
    assert manifest.manifest_hash
