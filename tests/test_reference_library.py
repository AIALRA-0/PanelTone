from pathlib import Path

from PIL import Image, ImageDraw

from manga_repaint.reference_library import ColorReferenceLibrary


def _page(path: Path, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (96, 128), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 18, 84, 110), fill=color)
    image.save(path)


def test_reference_library_retrieves_local_coloured_examples(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"
    first = jobs / "job-a" / "final" / "pages" / "page_00000.png"
    second = jobs / "job-b" / "final" / "pages" / "page_00000.png"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    _page(first, (220, 80, 80))
    _page(second, (80, 90, 220))

    library = ColorReferenceLibrary(jobs)
    matches = library.retrieve(first, limit=2, exclude=(first,))

    assert matches
    assert matches[0].path == second.resolve()
    assert (jobs.parent / "color-library" / "reference-index.json").is_file()


def test_reference_library_can_scope_retrieval_to_one_book(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"
    current = jobs / "book-a" / "final" / "pages" / "page_00000.png"
    same_book = jobs / "book-a" / "final" / "pages" / "page_00001.png"
    other_book = jobs / "book-b" / "final" / "pages" / "page_00000.png"
    for path in (current, same_book, other_book):
        path.parent.mkdir(parents=True, exist_ok=True)
    _page(current, (220, 80, 80))
    _page(same_book, (220, 80, 80))
    _page(other_book, (80, 90, 220))

    library = ColorReferenceLibrary(jobs)
    matches = library.retrieve(
        current,
        limit=4,
        exclude=(current,),
        scope_root=current.parent,
    )

    assert [item.path for item in matches] == [same_book.resolve()]


def test_reference_library_extracts_palette_anchors(tmp_path: Path) -> None:
    page = tmp_path / "page.png"
    _page(page, (220, 80, 80))
    anchors = ColorReferenceLibrary.palette_anchors([page])

    assert anchors
    assert any(
        0 <= hue < 180 and saturation > 40 and weight > 0
        for hue, saturation, weight in anchors
    )


def test_reference_library_balances_warm_references_with_colour_variety(
    tmp_path: Path,
) -> None:
    warm_a = tmp_path / "warm-a.png"
    warm_b = tmp_path / "warm-b.png"
    balanced = tmp_path / "balanced.png"
    _page(warm_a, (222, 144, 82))
    _page(warm_b, (196, 112, 64))
    image = Image.new("RGB", (96, 128), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 12, 46, 116), fill=(210, 115, 75))
    draw.rectangle((48, 12, 88, 62), fill=(66, 132, 190))
    draw.rectangle((48, 64, 88, 116), fill=(78, 158, 118))
    image.save(balanced)

    library = ColorReferenceLibrary(tmp_path / "jobs")
    selected = library.select_balanced([warm_a, warm_b, balanced], limit=2)

    assert balanced.resolve() in selected
    assert not {warm_a.resolve(), warm_b.resolve()}.issubset(selected)


def test_reference_library_keeps_explicit_reference_with_balanced_companion(
    tmp_path: Path,
) -> None:
    explicit = tmp_path / "explicit.png"
    warm = tmp_path / "warm.png"
    cool = tmp_path / "cool.png"
    _page(explicit, (220, 130, 72))
    _page(warm, (190, 105, 55))
    _page(cool, (60, 120, 205))

    library = ColorReferenceLibrary(tmp_path / "jobs")
    selected = library.select_balanced(
        [explicit, warm, cool],
        limit=2,
        required=[explicit],
    )

    assert selected == [explicit.resolve(), cool.resolve()]


def test_identity_hints_only_include_locked_records(tmp_path: Path) -> None:
    library = ColorReferenceLibrary(tmp_path / "jobs")
    hints = library.identity_hints(
        [
            {
                "identity_id": "a",
                "label": "A",
                "region": "skin",
                "color": "#e0a080",
                "locked": True,
            },
            {
                "identity_id": "b",
                "label": "B",
                "region": "hair",
                "color": "#101010",
                "locked": False,
            },
        ]
    )
    assert hints == ["A skin uses base color #e0a080"]
