from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from manga_repaint.color_state import (
    ColorStateStore,
    IdentityNode,
    build_color_book_state,
    build_identity_graph,
    build_segment_graph,
    classify_reference_pools,
    dependency_status,
    make_region_color_plan,
)


def test_segment_graph_is_deterministic_and_preserves_neighbour_evidence() -> None:
    image = Image.new("RGB", (96, 96), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 40, 40), outline="black", width=3)
    draw.rectangle((48, 8, 88, 40), outline="black", width=3)
    first = build_segment_graph(image, job_id="book", page_index=0)
    second = build_segment_graph(image, job_id="book", page_index=0)

    assert first.analysis_hash == second.analysis_hash
    assert first.nodes
    assert all(node.semantic_candidates == ("unresolved",) for node in first.nodes)


def test_segment_neighbours_match_original_dilation_without_per_region_scans(monkeypatch):
    import cv2
    import numpy as np

    image = Image.new("RGB", (180, 180), "white")
    draw = ImageDraw.Draw(image)
    for x in range(10, 170, 14):
        for y in range(10, 170, 14):
            draw.rectangle((x, y, x + 10, y + 10), outline="black", width=2)
    gray = np.asarray(image.convert("L"))
    barrier = cv2.dilate((gray <= 96).astype(np.uint8), np.ones((3, 3), np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats((barrier == 0).astype(np.uint8), 8)
    eligible = [label for label in range(1, len(stats)) if stats[label, cv2.CC_STAT_AREA] >= 24]
    mapping = {label: f"p0000-s{index:05d}" for index, label in enumerate(eligible)}
    expected = {}
    for label in eligible:
        nearby = cv2.dilate((labels == label).astype(np.uint8), np.ones((3, 3), np.uint8))
        expected[mapping[label]] = tuple(sorted(
            mapping[int(other)] for other in np.unique(labels[nearby > 0])
            if other in mapping and other != label
        ))
    original = cv2.dilate
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(cv2, "dilate", counted)
    graph = build_segment_graph(image, job_id="book", page_index=0)
    assert {node.segment_id: node.neighbours for node in graph.nodes} == expected
    assert len(calls) == 1


def test_identity_graph_keeps_same_panel_nodes_cannot_linked() -> None:
    observations = [
        IdentityNode("p0-a", 0, 0, (0, 0, 10, 10), "same", 0.8),
        IdentityNode("p0-b", 0, 0, (12, 0, 10, 10), "same", 0.8),
        IdentityNode("p1-a", 1, 0, (0, 0, 10, 10), "same", 0.8),
    ]
    graph = build_identity_graph("book", observations)

    assert ("p0-a", "p0-b") in graph.cannot_link
    assert any(edge.relation == "same_identity" for edge in graph.edges)


def test_color_state_plan_and_dependency_invalidation(tmp_path: Path) -> None:
    image = Image.new("RGB", (64, 64), "white")
    segment = build_segment_graph(image, job_id="book", page_index=0)
    graph = build_identity_graph("book")
    state = build_color_book_state(
        "book",
        {0: "source-hash"},
        graph,
        segment.analysis_hash,
        identities=[
            {
                "identity_id": "main_female_01",
                "region": "skin",
                "color": "#e0a080",
                "confidence": 0.9,
                "locked": True,
            }
        ],
    )
    anchor = tmp_path / "anchor.png"
    image.save(anchor)
    plan = make_region_color_plan(
        state,
        segment,
        page_source_hash="source-hash",
        page_index=0,
        references=[anchor],
        page_seed=3,
    )
    store = ColorStateStore(tmp_path / "job")
    store.write("color_book_state.json", state)
    store.write_page("plans", 0, plan)

    assert store.path("color_book_state.json").is_file()
    assert store.path("plans/page_00000.json").is_file()
    assert store.read("color_book_state.json")["state_hash"] == state.state_hash
    assert plan.seed == make_region_color_plan(
        state,
        segment,
        page_source_hash="source-hash",
        page_index=0,
        references=[anchor],
        page_seed=3,
    ).seed
    assert dependency_status(
        source_snapshot_hash="source-hash",
        current_source_snapshot_hash="source-hash",
        dependencies=(("renderer", "book-color-plan-v1"),),
        current={"renderer": "book-color-plan-v1"},
    ) == "current"
    assert dependency_status(
        source_snapshot_hash="changed",
        current_source_snapshot_hash="source-hash",
        dependencies=(),
        current={},
    ) == "stale"
    assert state.appearances[0].appearance_id == "main_female_01.default"
    assert state.scenes[0].scene_id == "scene.default"
    assert classify_reference_pools([anchor]) == (("unassigned", (str(anchor.resolve()),)),)
