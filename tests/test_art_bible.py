from dataclasses import replace

import pytest

from manga_repaint.art_bible import (
    Appearance,
    ArtBibleStore,
    ArtPaletteSlot,
    BookArtBible,
    Character,
    Outfit,
    RegionFact,
    Scene,
    affected_pages,
    transition_state,
)


def _bible() -> BookArtBible:
    skin = ArtPaletteSlot(
        "lead.skin",
        "skin",
        (238, 177, 150),
        state="locked",
        confidence=1.0,
        evidence=("user:golden-page-81",),
        locked_by_user=True,
    )
    appearance = Appearance(
        "lead.default",
        "lead",
        ((0, 202),),
        (("skin", "lead.skin"),),
        state="accepted",
        evidence=("review:golden-set",),
    )
    return BookArtBible(
        book_id="book",
        revision=1,
        source_snapshot_hash="source-hash",
        characters=(
            Character(
                "lead",
                "主角",
                "accepted",
                ("review:golden-set",),
                (appearance,),
            ),
        ),
        palette_slots=(skin,),
        scenes=(Scene("room", "室内", ((0, 100),), state="proposed"),),
        regions=(
            RegionFact(
                8,
                "p0008-r1",
                1,
                "skin",
                "lead.skin",
                "lead",
                "lead.default",
                1.0,
                "locked",
                ("user:golden-page-9",),
            ),
        ),
        dependencies=(("mask_revision", "1"),),
    ).with_hash()


def test_art_bible_round_trip_and_publish_gate(tmp_path) -> None:
    bible = _bible()
    bible.validate()
    store = ArtBibleStore(tmp_path / "job")
    revision = store.save(bible)
    restored = store.load()
    assert revision.is_file()
    assert restored == bible
    assert restored.publishable({8}) == (True, ())


def test_art_bible_rejects_unreviewed_and_invalid_transitions() -> None:
    bible = _bible()
    proposed = replace(bible.regions[0], state="proposed")
    changed = replace(bible, regions=(proposed,), state_hash="").with_hash()
    assert changed.publishable({8}) == (False, ("unreviewed_regions",))
    with pytest.raises(ValueError, match="invalid state transition"):
        transition_state(bible.palette_slots[0], "proposed")


def test_art_bible_reports_only_pages_affected_by_a_palette_change() -> None:
    before = _bible()
    slot = replace(before.palette_slots[0], base_rgb=(230, 165, 145), revision=2)
    after = replace(before, palette_slots=(slot,), state_hash="").with_hash()
    assert affected_pages(before, after) == {8}


def test_art_bible_separates_character_appearance_and_outfit() -> None:
    before = _bible()
    outfit = Outfit(
        "lead.uniform",
        "lead",
        "制服",
        ((0, 20),),
        (("skin", "lead.skin"),),
        state="accepted",
        evidence=("review:outfit",),
    )
    appearance = replace(
        before.characters[0].appearances[0], outfit_ids=(outfit.outfit_id,)
    )
    character = replace(before.characters[0], appearances=(appearance,))
    updated = replace(
        before, characters=(character,), outfits=(outfit,), state_hash=""
    ).with_hash()
    updated.validate()
