import numpy as np
from PIL import Image, ImageDraw

from manga_repaint.flat_planner import (
    preview_plan,
    propose_material_plan,
    propose_material_plan_with_object_masks,
    source_regions,
)
from manga_repaint.material_render import render_material_layers
from manga_repaint.sam2_regions import Sam2Region


def _fixture():
    source = Image.new("RGB", (96, 64), "white")
    draw = ImageDraw.Draw(source)
    draw.rectangle((8, 8, 44, 56), outline="black", width=3)
    draw.rectangle((52, 8, 88, 56), outline="black", width=3)
    for y in range(14, 52, 4):
        draw.point((20, y), fill=(130, 130, 130))
        draw.point((72, y), fill=(130, 130, 130))
    candidate = Image.new("RGB", source.size, "white")
    colours = np.asarray(candidate).copy()
    colours[11:54, 11:42] = (235, 165, 145)
    colours[11:54, 55:86] = (60, 100, 190)
    # Deliberate candidate stain must be voted away within the first region
    colours[20:25, 20:25] = (20, 220, 240)
    return source, Image.fromarray(colours), np.zeros((64, 96), dtype=bool)


def test_source_regions_ignore_candidate_geometry() -> None:
    source, _candidate, protected = _fixture()
    first, regions = source_regions(source, protected)
    shifted = Image.new("RGB", source.size, "magenta")
    second, repeated = source_regions(source, protected)
    assert np.array_equal(first, second)
    assert regions == repeated
    assert shifted.size == source.size


def test_flat_proposal_never_accepts_model_output_implicitly() -> None:
    source, candidate, protected = _fixture()
    proposal = propose_material_plan(
        source,
        candidate,
        protected,
        palette_revision="fixture-v1",
        evidence="fixture-candidate",
    )
    assert proposal.proposals
    assert all(slot.state == "proposed" for slot in proposal.plan.slots)
    assert all(region.state == "proposed" for region in proposal.plan.regions)


def test_material_cel_v4_exports_clean_inspectable_layers() -> None:
    source, candidate, protected = _fixture()
    proposal = propose_material_plan(
        source,
        candidate,
        protected,
        palette_revision="fixture-v1",
        evidence="fixture-candidate",
    )
    layers = render_material_layers(source, preview_plan(proposal))
    assert layers.final.size == source.size
    assert layers.flats.size == source.size
    assert set(np.unique(layers.shade_index)).issubset({0, 1, 2, 3})
    # The cyan candidate stain cannot survive as a local patch in a flat layer
    flat = np.asarray(layers.flats)
    assert not np.all(flat[22, 22] == (20, 220, 240))


def test_object_masks_share_colour_without_becoming_final_geometry() -> None:
    source, candidate, protected = _fixture()
    object_mask = np.zeros_like(protected)
    object_mask[5:59, 5:91] = True
    proposal = propose_material_plan_with_object_masks(
        source,
        candidate,
        protected,
        (Sam2Region(object_mask, 0.95, (20, 20)),),
        palette_revision="test-object-v1",
        evidence="test-sam2",
    )
    assert proposal.proposals
    assert any("object_mask:0" in item.reasons for item in proposal.proposals)
    assert all(item.state == "proposed" for item in proposal.proposals)
    rendered = render_material_layers(source, preview_plan(proposal)).final
    assert rendered.size == source.size
