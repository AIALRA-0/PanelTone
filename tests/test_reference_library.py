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
