from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from .http_service import HTTPImageEngine


def prepare_cobra_hint(
    extracted_line: Image.Image,
    hint: Image.Image,
    *,
    source_size: tuple[int, int],
    resolution: tuple[int, int],
) -> tuple[Image.Image, Image.Image]:
    """Convert a transparent sparse palette artifact to Cobra's hint inputs."""
    if hint.size != source_size:
        raise ValueError("Cobra 颜色提示图尺寸必须与源图一致")
    resized = hint.convert("RGBA").resize(resolution, Image.Resampling.NEAREST)
    rgba = np.asarray(resized)
    alpha = rgba[..., 3]
    hint_mask = Image.fromarray(alpha, mode="L")
    colour = np.asarray(extracted_line.convert("RGB")).copy()
    selected = alpha > 0
    colour[selected] = rgba[selected, :3]
    return hint_mask, Image.fromarray(colour, mode="RGB")


class CobraCandidateEngine(HTTPImageEngine):
    """PanelTone adapter for the isolated official Cobra HTTP candidate."""

    def healthcheck(self) -> dict[str, Any]:
        result = super().healthcheck()
        result.setdefault("model_id", "JunhaoZhuang/Cobra")
        result.setdefault("candidate", True)
        return result
