from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from manga_repaint.color import normalize_color_candidate_size
from manga_repaint.engines.base import EngineRequest
from manga_repaint.engines.cobra import CobraCandidateEngine, prepare_cobra_hint
from manga_repaint.models import JobMode


def test_cobra_working_canvas_is_resized_only_when_aspect_ratio_is_close() -> None:
    candidate = Image.new("RGB", (1008, 1440), "red")
    normalized = normalize_color_candidate_size(candidate, (1444, 2048))

    assert normalized.size == (1444, 2048)


def test_cobra_working_canvas_rejects_crop_or_distortion() -> None:
    candidate = Image.new("RGB", (500, 900), "red")

    with pytest.raises(ValueError, match="aspect ratio"):
        normalize_color_candidate_size(candidate, (1444, 2048))


def test_cobra_adapter_uploads_sparse_hint_as_a_separate_artifact(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.png"
    reference = tmp_path / "reference.png"
    hint = tmp_path / "hint.png"
    output = tmp_path / "output.png"
    Image.new("RGB", (32, 32), "white").save(source)
    Image.new("RGB", (32, 32), "red").save(reference)
    Image.new("RGBA", (32, 32), (0, 0, 0, 0)).save(hint)
    encoded = BytesIO()
    Image.new("RGB", (32, 32), "blue").save(encoded, format="PNG")
    captured = {}

    class Response:
        status_code = 200
        content = encoded.getvalue()
        headers = {"x-model-id": "JunhaoZhuang/Cobra"}

        @staticmethod
        def raise_for_status() -> None:
            return None

    class Client:
        def __init__(self, **kwargs):
            captured["client"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        @staticmethod
        def post(path, *, data, files):
            captured["path"] = path
            captured["data"] = data
            captured["fields"] = [field for field, _ in files]
            return Response()

    monkeypatch.setattr("manga_repaint.engines.http_service.httpx.Client", Client)
    engine = CobraCandidateEngine("cobra", "http://127.0.0.1:8783")
    result = engine.generate(
        EngineRequest(
            source_path=source,
            output_path=output,
            mode=JobMode.COLORIZE,
            seed=1,
            prompt="colour the line art",
            negative_prompt="",
            references=[reference],
            hint_path=hint,
        )
    )

    assert captured["path"] == "/generate"
    assert captured["fields"] == ["source", "references", "hint"]
    assert output.is_file() and result.engine_metadata["model"] == "JunhaoZhuang/Cobra"


def test_cobra_sparse_hint_keeps_unhinted_line_art_and_rejects_wrong_size() -> None:
    line = Image.new("RGB", (32, 32), "white")
    hint_array = np.zeros((32, 32, 4), dtype=np.uint8)
    hint_array[12:16, 18:22] = (220, 130, 90, 255)
    hint = Image.fromarray(hint_array, mode="RGBA")

    mask, colour = prepare_cobra_hint(
        line,
        hint,
        source_size=(32, 32),
        resolution=(32, 32),
    )

    mask_array = np.asarray(mask)
    colour_array = np.asarray(colour)
    assert np.all(mask_array[12:16, 18:22] == 255)
    assert np.all(colour_array[12:16, 18:22] == (220, 130, 90))
    assert np.all(colour_array[mask_array == 0] == 255)
    with pytest.raises(ValueError, match="尺寸"):
        prepare_cobra_hint(
            line,
            hint.resize((16, 16)),
            source_size=(32, 32),
            resolution=(32, 32),
        )
