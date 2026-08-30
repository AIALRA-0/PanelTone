from __future__ import annotations

import colorsys
import random
from pathlib import Path

import numpy as np
from PIL import Image

from .base import EngineRequest, EngineResult


class PaletteEngine:
    """Deterministic non-AI engine used for pipeline verification and safe fallbacks."""

    name = "palette"

    def __init__(self, saturation: float = 0.52):
        self.saturation = saturation

    def _reference_colors(self, references: list[Path]) -> list[tuple[int, int, int]]:
        colors: list[tuple[int, int, int]] = []
        for path in references:
            with Image.open(path) as image:
                sample = image.convert("RGB").resize((48, 48))
                quantized = sample.quantize(colors=8).convert("RGB")
                counts = quantized.getcolors(maxcolors=48 * 48) or []
                colors.extend(color for _, color in sorted(counts, reverse=True)[:4])
        return colors

    def generate(self, request: EngineRequest) -> EngineResult:
        with Image.open(request.source_path) as image:
            gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        height, width = gray.shape
        rng = random.Random(request.seed + request.attempt - 1)
        reference_colors = self._reference_colors(request.references)
        if reference_colors:
            rgb_a, rgb_b = reference_colors[0], reference_colors[min(1, len(reference_colors) - 1)]
        else:
            hue_a = rng.random()
            hue_b = (hue_a + 0.22 + rng.random() * 0.25) % 1.0
            rgb_a = tuple(
                round(value * 255) for value in colorsys.hsv_to_rgb(hue_a, self.saturation, 1)
            )
            rgb_b = tuple(
                round(value * 255) for value in colorsys.hsv_to_rgb(hue_b, self.saturation, 1)
            )

        y_axis, x_axis = np.mgrid[0:height, 0:width]
        blend = (0.55 * x_axis / max(1, width - 1) + 0.45 * y_axis / max(1, height - 1))[..., None]
        color_a = np.asarray(rgb_a, dtype=np.float32)
        color_b = np.asarray(rgb_b, dtype=np.float32)
        chroma = color_a * (1 - blend) + color_b * blend
        value = 0.24 + gray[..., None] * 0.76
        output = np.clip(chroma * value, 0, 255).astype(np.uint8)
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(output, mode="RGB").save(request.output_path, format="PNG")
        return EngineResult(
            request.output_path,
            {"kind": "deterministic_test_engine", "seed": request.seed, "attempt": request.attempt},
        )

    def healthcheck(self) -> dict[str, object]:
        return {"ok": True, "engine": self.name, "requires_model": False}
