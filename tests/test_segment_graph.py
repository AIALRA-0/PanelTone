from dataclasses import replace

import numpy as np
from PIL import Image, ImageDraw

from manga_repaint.segment_graph import VirtualClosure, build_segment_graph


def _source():
    image = Image.new("RGB", (96, 72), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 44, 64), outline="black", width=3)
    draw.rectangle((52, 8, 88, 64), outline="black", width=3)
    return image


def test_segment_graph_uses_source_owned_atomic_regions() -> None:
    source = _source()
    result = build_segment_graph(
        source, np.zeros((source.height, source.width), dtype=bool), page_index=7
    )
    assert result.graph.regions
    assert result.labels.shape == (source.height, source.width)
    assert all(item.region_id.startswith("p00007.r") for item in result.graph.regions)
    result.validate(source)


def test_unreviewed_virtual_closure_blocks_publication() -> None:
    source = _source()
    result = build_segment_graph(
        source, np.zeros((source.height, source.width), dtype=bool), page_index=0
    )
    closure = VirtualClosure("gap-1", (10, 10), (20, 20), 0.8)
    graph = replace(result.graph, closures=(closure,), unknown_ratio=0.0)
    assert graph.publishable() == (False, ("unreviewed_virtual_closures",))
