import pytest
from PIL import Image

from manga_repaint.color import normalize_color_candidate_size


def test_cobra_working_canvas_is_resized_only_when_aspect_ratio_is_close() -> None:
    candidate = Image.new("RGB", (1008, 1440), "red")
    normalized = normalize_color_candidate_size(candidate, (1444, 2048))

    assert normalized.size == (1444, 2048)


def test_cobra_working_canvas_rejects_crop_or_distortion() -> None:
    candidate = Image.new("RGB", (500, 900), "red")

    with pytest.raises(ValueError, match="aspect ratio"):
        normalize_color_candidate_size(candidate, (1444, 2048))
