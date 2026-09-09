from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from manga_repaint.color import image_sha256
from manga_repaint.material_render import (
    MaterialPlan,
    MaterialRegion,
    MaterialSlot,
    build_sparse_color_hint,
    evaluate_material_render,
    labels_from_masks,
    load_material_plan,
    propose_region_colour,
    protection_mask,
    render_material_flats,
    save_material_plan,
)


def fixture():
    # Independent masks drawn for the test, not recovered from renderer output
    yy, xx = np.indices((96, 128))
    gray = np.where((xx % 5 == 0) & (yy % 5 == 0), 75, 230).astype(np.uint8)
    gray[:, 62:66] = 0
    gray[0:5] = 0
    rgb = np.repeat(gray[..., None], 3, axis=2)
    source = Image.fromarray(rgb)
    labels = np.where(xx < 64, 1, 2).astype(np.uint16)
    protected = gray == 0
    slots = (
        MaterialSlot("lead.skin", "skin", (234, 181, 151), "accepted", "fixture", "lead"),
        MaterialSlot("lead.shirt", "white", (241, 241, 241), "accepted", "fixture"),
    )
    regions = (
        MaterialRegion(1, "lead.skin", "accepted", "independent test mask"),
        MaterialRegion(2, "lead.shirt", "accepted", "independent test mask"),
    )
    return source, MaterialPlan(image_sha256(source), labels, protected, slots, regions, "book-1")


def test_cel_render_preserves_structural_ink_but_descreens_ordinary_materials():
    source, plan = fixture()
    result = render_material_flats(source, plan)
    report = evaluate_material_render(source, result, plan)
    assert report["passed"], report
    assert report["protected_pixel_diff"] == 0
    rgb = np.asarray(result)
    # White fabric stays neutral, structural ink remains exact, and isolated
    # screentone dots select a coherent layer instead of becoming paint holes
    assert np.max(rgb[:, 66:]) == np.min(rgb[:, 66:], axis=2).max()
    assert np.array_equal(rgb[20, 63], np.asarray(source)[20, 63])
    assert np.max(np.abs(rgb[10, 10].astype(int) - rgb[11, 11].astype(int))) <= 1
    assert max(row["chromaticity_error_p95"] for row in report["regions"]) < 0.015


@pytest.mark.parametrize(
    "defect", ["purple_stain", "neutral_hole", "boundary_spill", "tone_hue", "shifted", "caustics"]
)
def test_independent_region_qa_rejects_stains_holes_bleed_and_screentone_hue(defect):
    source, reference = fixture()
    rgb = np.asarray(render_material_flats(source, reference)).copy()
    yy, xx = np.indices(rgb.shape[:2])
    if defect == "purple_stain":
        alpha = np.exp(-((xx - 32) ** 2 + (yy - 55) ** 2) / 350)[..., None]
        rgb = np.rint(rgb * (1 - alpha) + np.array([100, 70, 220]) * alpha).astype(np.uint8)
    elif defect == "neutral_hole":
        rgb[35:55, 25:45] = 220
    elif defect == "boundary_spill":
        rgb[15:80, 66:70] = (234, 181, 151)
    elif defect == "tone_hue":
        rgb[(xx < 60) & (np.asarray(source)[..., 0] < 100)] = (65, 20, 80)
    elif defect == "shifted":
        rgb = np.roll(rgb, 12, axis=1)
    else:
        alpha = ((np.sin(xx / 9) * np.sin(yy / 11) + 1) * 0.3)[..., None]
        rgb = np.rint(rgb * (1 - alpha) + np.array([80, 140, 210]) * alpha).astype(np.uint8)
    # Passing geometry/protected pixels alone must not hide the colour defect
    protected = protection_mask(source, reference.protected)
    rgb[protected] = np.asarray(source)[protected]
    report = evaluate_material_render(source, Image.fromarray(rgb), reference)
    assert not report["passed"]
    assert report["protected_pixel_diff"] == 0
    assert any("region_color_or_shading_mismatch" in reason for reason in report["reasons"])


def test_candidate_stains_never_enter_renderer_and_votes_never_self_accept():
    source, plan = fixture()
    clean = Image.new("RGB", source.size, (230, 180, 150))
    polluted = np.asarray(clean).copy()
    polluted[:, 0:32] = (20, 40, 200)
    first = propose_region_colour(clean, plan.labels == 1)
    second = propose_region_colour(Image.fromarray(polluted), plan.labels == 1)
    assert first["usable"] and first["state"] == "proposed"
    assert not second["usable"] and second["state"] == "proposed"
    # A proposal is not a plan mutation and the renderer cannot read a candidate
    before = render_material_flats(source, plan)
    assert image_sha256(before) == image_sha256(render_material_flats(source, plan))


def test_unknown_and_proposed_regions_are_not_silently_colored():
    source, plan = fixture()
    plan = replace(plan, regions=(replace(plan.regions[0], state="proposed"), plan.regions[1]))
    final = render_material_flats(source, plan)
    assert np.array_equal(np.asarray(final)[plan.labels == 1], np.asarray(source)[plan.labels == 1])
    report = evaluate_material_render(source, final, plan)
    assert not report["passed"] and report["unknown_pixels"] > 0


def test_conflicting_masks_become_unknown_not_last_writer_wins():
    a = np.zeros((32, 32), dtype=bool)
    b = a.copy()
    a[:, :20] = True
    b[:, 12:] = True
    labels, conflict = labels_from_masks({1: a, 2: b})
    assert np.all(labels[:, 12:20] == 0)
    assert np.all(conflict[:, 12:20])
    assert np.all(labels[:, :12] == 1) and np.all(labels[:, 20:] == 2)


def test_mask_hash_and_source_hash_block_stale_bundles(tmp_path):
    source, plan = fixture()
    manifest = save_material_plan(tmp_path / "review", source, plan)
    restored = load_material_plan(manifest, source)
    assert np.array_equal(restored.labels, plan.labels)
    assert image_sha256(render_material_flats(source, restored)) == image_sha256(
        render_material_flats(source, plan)
    )
    with pytest.raises(FileExistsError):
        save_material_plan(tmp_path / "review", source, plan)
    with pytest.raises(ValueError, match="stale"):
        load_material_plan(manifest, Image.new("RGB", source.size, "white"))
    Image.fromarray(np.zeros_like(plan.labels)).save(manifest.parent / "labels.png")
    with pytest.raises(ValueError, match="changed after review"):
        load_material_plan(manifest, source)


def test_validation_does_not_accept_fake_evidence_or_labels():
    source, plan = fixture()
    with pytest.raises(ValueError, match="review evidence"):
        replace(plan, slots=(replace(plan.slots[0], evidence=""), plan.slots[1])).validate(source)
    with pytest.raises(ValueError, match="exactly one"):
        replace(plan, regions=()).validate(source)
    with pytest.raises(ValueError, match="dimensions"):
        replace(plan, labels=plan.labels[:2]).validate(source)
    with pytest.raises(ValueError, match="integers"):
        replace(plan, labels=plan.labels.astype(float)).validate(source)


def test_shared_slot_controls_disjoint_regions_across_pages():
    source, plan = fixture()
    labels = plan.labels.copy()
    labels[40:60, 10:40] = 3
    second = replace(
        plan,
        labels=labels,
        regions=plan.regions + (MaterialRegion(3, "lead.skin", "accepted", "fixture"),),
    )
    assert np.array_equal(
        np.asarray(render_material_flats(source, plan)),
        np.asarray(render_material_flats(source, second)),
    )


def test_qa_checks_protection_and_dimensions_independently():
    source, plan = fixture()
    final = render_material_flats(source, plan)
    assert not evaluate_material_render(source, final.resize((64, 48)), plan)["passed"]
    changed = np.asarray(final).copy()
    changed[0, 0] = 255
    assert (
        evaluate_material_render(source, Image.fromarray(changed), plan)["protected_pixel_diff"]
        == 255
    )


def test_declared_pattern_retains_source_detail_without_model_texture():
    source, plan = fixture()
    plan = replace(plan, slots=(replace(plan.slots[0], material="patterned"), plan.slots[1]))
    final = render_material_flats(source, plan)
    report = evaluate_material_render(source, final, plan)
    assert report["passed"]
    # Pattern colour changes need separate labels/slots, not a hidden spatial layer
    assert report["regions"][0]["chroma_statistics"]["dispersion_p95"] < 0.01
    assert not np.array_equal(np.asarray(final)[10, 10], np.asarray(final)[11, 11])


def test_connected_ink_is_preserved_but_isolated_black_screentone_is_not():
    gray = np.full((80, 100), 255, dtype=np.uint8)
    gray[15:65, 20] = 0
    gray[15:65, 21] = 80  # antialiased fringe of a connected contour
    gray[30, 60] = 0  # isolated screentone sample
    source = Image.fromarray(np.repeat(gray[..., None], 3, axis=2))
    plan = MaterialPlan(
        image_sha256(source),
        np.ones(gray.shape, dtype=np.uint16),
        np.zeros(gray.shape, dtype=bool),
        (MaterialSlot("skin", "skin", (232, 176, 145), "accepted", "fixture"),),
        (MaterialRegion(1, "skin", "accepted", "fixture"),),
        "ink-vs-tone",
    )
    result = np.asarray(render_material_flats(source, plan))
    assert np.array_equal(result[40, 20], np.asarray(source)[40, 20])
    assert np.array_equal(result[40, 21], np.asarray(source)[40, 21])
    assert not np.array_equal(result[30, 60], np.asarray(source)[30, 60])
    assert np.max(np.abs(result[30, 60].astype(int) - result[30, 61].astype(int))) <= 1


def test_sparse_model_hints_do_not_encode_a_full_page_colour_mask():
    source, plan = fixture()
    hint = np.asarray(build_sparse_color_hint(source, plan, radius=3, max_points_per_region=3))
    alpha = hint[..., 3] > 0
    protected = protection_mask(source, plan.protected)
    assert alpha.any()
    assert float(alpha.mean()) < 0.03
    assert not np.any(alpha & protected)
    assert not np.any(hint[~alpha])
    skin = alpha & (plan.labels == 1)
    assert skin.any()
    assert np.all(hint[skin, :3] == np.asarray(plan.slots[0].rgb))
    assert np.array_equal(
        hint,
        np.asarray(build_sparse_color_hint(source, plan, radius=3, max_points_per_region=3)),
    )


def test_cli_isolated_render_does_not_construct_live_manager(tmp_path, monkeypatch):
    from manga_repaint import cli

    source, plan = fixture()
    source_path = tmp_path / "source.png"
    source.save(source_path)
    manifest = save_material_plan(tmp_path / "review", source, plan)

    def forbidden(*args):
        raise AssertionError("isolated renderer must not open or recover live data")

    monkeypatch.setattr(cli, "_manager", forbidden)
    assert (
        cli.main(["material-render", str(source_path), str(manifest), str(tmp_path / "out")]) == 0
    )
    assert (tmp_path / "out" / "qa.json").is_file()
    assert (tmp_path / "out" / "cobra-hint.png").is_file()


def test_transparent_source_uses_same_white_composite_as_source_hash():
    source, plan = fixture()
    rgba = np.array(source.convert("RGBA"))
    rgba[8:20, 8:20] = (0, 0, 0, 0)
    source = Image.fromarray(rgba)
    plan = replace(plan, source_hash=image_sha256(source))
    result = render_material_flats(source, plan)
    assert (
        np.max(np.abs(np.asarray(result)[10, 10].astype(int) - np.asarray(plan.slots[0].rgb))) <= 1
    )
    assert evaluate_material_render(source, result, plan)["passed"]


def test_shared_compositor_uses_material_plan_before_reading_candidate(tmp_path, monkeypatch):
    from manga_repaint.models import JobMode, JobSpec
    from manga_repaint.project import ProjectManager

    source, plan = fixture()
    # _compose_unit does not require manager construction/recovery for this route
    manager = object.__new__(ProjectManager)

    def forbidden(*args, **kwargs):
        raise AssertionError("candidate pixel path must not run for a reviewed material plan")

    monkeypatch.setattr(manager, "_render_color_candidate", forbidden)
    spec = JobSpec(source=tmp_path / "unused.png", workspace=tmp_path, mode=JobMode.COLORIZE)
    final, _ = manager._compose_unit(
        source, Image.new("RGB", (1, 1), "purple"), plan.protected, spec, material_plan=plan
    )
    assert image_sha256(final) == image_sha256(render_material_flats(source, plan))
    with pytest.raises(ValueError, match="runtime protection changed"):
        manager._compose_unit(
            source, source, np.ones_like(plan.protected), spec, material_plan=plan
        )
    unfinished = replace(plan, slots=(replace(plan.slots[0], state="proposed"), plan.slots[1]))
    with pytest.raises(ValueError, match="requires review"):
        manager._compose_unit(source, source, plan.protected, spec, material_plan=unfinished)
    spec.mode = JobMode.STYLE_FULL
    with pytest.raises(ValueError, match="only supported for COLORIZE"):
        manager._compose_unit(source, source, plan.protected, spec, material_plan=plan)


def test_model_mask_score_cannot_override_prompt_failures():
    from manga_repaint.material_proposals import MaskPrompt, evaluate_mask_proposal

    mask = np.ones((32, 32), dtype=bool)
    prompt = MaskPrompt((0, 0, 32, 32), ((5, 5),), ((20, 20),))
    report = evaluate_mask_proposal(mask, prompt)
    assert not report["prompt_consistent"]
    assert report["reasons"] == ["negative_points_included"]
    assert not report["auto_accept"] and report["semantic_status"] == "unverified"
    mask[19:22, 19:22] = False
    report = evaluate_mask_proposal(mask, prompt)
    assert report["prompt_consistent"] and not report["auto_accept"]
    assert report["enclosed_hole_pixels"] == 9
    with pytest.raises(ValueError, match="invalid"):
        MaskPrompt((32, 0, 32, 10), ((2, 2),)).validate(mask.shape)


def test_stain_measurement_is_independent_of_chosen_palette():
    from manga_repaint.material_render import region_chroma_statistics

    rgb = np.full((64, 64, 3), (80, 140, 200), dtype=np.uint8)
    region = np.ones(rgb.shape[:2], dtype=bool)
    stats = region_chroma_statistics(rgb, region)
    # A clean but different colour is not a stain; palette agreement is a
    # separate metric and must not be used to exaggerate benchmark improvement
    assert stats["dispersion_p95"] < 1e-5
    assert stats["low_frequency_residual_p95"] < 1e-5
    rgb[24:40, 24:40] = (200, 80, 150)
    stats = region_chroma_statistics(rgb, region)
    assert stats["dispersion_p95"] > 0.10
    assert stats["low_frequency_residual_p95"] > 0.04


def test_renderer_paints_distinct_base_shadow_and_highlight_layers():
    gray = np.full((96, 96), 255, dtype=np.uint8)
    gray[:, :32] = 115
    gray[32:64, :32] = 240  # authored highlight inside a broad shadow area
    source = Image.fromarray(np.repeat(gray[..., None], 3, axis=2))
    labels = np.ones(gray.shape, dtype=np.uint16)
    plan = MaterialPlan(
        image_sha256(source),
        labels,
        np.zeros(gray.shape, dtype=bool),
        (
            MaterialSlot(
                "skin",
                "skin",
                (232, 176, 145),
                "accepted",
                "fixture",
                shadow_rgb=(154, 80, 75),
                highlight_rgb=(250, 218, 193),
            ),
        ),
        (MaterialRegion(1, "skin", "accepted", "fixture"),),
        "layered-1",
    )
    result = np.asarray(render_material_flats(source, plan))
    base = result[70, 70].astype(int)
    shadow = result[10, 10].astype(int)
    highlight = result[45, 10].astype(int)
    # Base remains canonical paint while shade/highlight move toward their own
    # authored colours; a gray-times-RGB wash cannot satisfy both relationships
    assert np.linalg.norm(base - np.asarray((232, 176, 145))) <= 2
    assert np.linalg.norm(shadow - np.asarray((154, 80, 75))) < np.linalg.norm(shadow - base)
    assert highlight.mean() > shadow.mean()
    assert not np.allclose(shadow / 115, base / 255, atol=0.08)
