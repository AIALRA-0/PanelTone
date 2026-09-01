from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
from PIL import Image

from .masks import border_mask, bubble_mask, ink_detail_mask, text_region_mask

SEMANTIC_CLASSES = (
    "text",
    "bubbles",
    "borders",
    "ink",
    "eyes",
    "mouth",
    "skin",
    "hair",
    "clothing",
    "hands_feet_body",
    "face",
    "accessories",
    "objects",
    "background",
)

SEMANTIC_ORDER = (
    "text",
    "bubbles",
    "borders",
    "ink",
    "eyes",
    "mouth",
    "skin",
    "hair",
    "clothing",
    "hands_feet_body",
    "face",
    "accessories",
    "objects",
    "background",
)


@dataclass(slots=True)
class SemanticMaskResult:
    masks: dict[str, np.ndarray]
    confidence: np.ndarray
    uncertain: np.ndarray
    provider: str
    version: str

    @property
    def uncertain_ratio(self) -> float:
        return float(self.uncertain.mean()) if self.uncertain.size else 0.0


class SemanticMaskEngine(Protocol):
    provider: str
    version: str

    def segment(self, image: Image.Image) -> SemanticMaskResult:
        """Return conservative masks without changing the source image."""


class ConservativeSemanticMaskEngine:
    """Deterministic fallback used when an optional segmentation model is absent.

    It deliberately refuses to guess skin, hair, or clothing on grayscale input
    and protects only regions that can be identified from ink geometry. This
    prevents a bad semantic guess from painting exposed skin with clothing color.
    """

    provider = "deterministic-protection"
    version = "1"

    def __init__(self, confidence_threshold: float = 0.78) -> None:
        self.confidence_threshold = confidence_threshold

    def segment(self, image: Image.Image) -> SemanticMaskResult:
        rgb = image.convert("RGB")
        shape = (rgb.height, rgb.width)
        ink = ink_detail_mask(rgb, threshold=64)
        borders = border_mask(rgb)
        text = text_region_mask(rgb)
        bubbles = bubble_mask(rgb)
        protected = ink | borders | text | bubbles
        # There is no trustworthy skin/clothing signal in a monochrome source.
        # Keep those classes empty and mark non-protected pixels uncertain so a
        # caller can preserve luminance and request a human correction.
        confidence = np.zeros(shape, dtype=np.float32)
        confidence[protected] = 1.0
        uncertain = confidence < self.confidence_threshold
        masks = {name: np.zeros(shape, dtype=bool) for name in SEMANTIC_CLASSES}
        masks["text"] = text
        masks["bubbles"] = bubbles
        masks["borders"] = borders
        masks["ink"] = ink
        masks["background"] = self._background_mask(rgb, protected)
        return SemanticMaskResult(
            masks=masks,
            confidence=confidence,
            uncertain=uncertain,
            provider=self.provider,
            version=self.version,
        )

    @staticmethod
    def _background_mask(image: Image.Image, protected: np.ndarray) -> np.ndarray:
        gray = np.asarray(image.convert("L"))
        light = gray >= 238
        # Only label broad connected light regions, never narrow areas touching ink.
        candidates = (light & ~protected).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, 8)
        result = np.zeros_like(candidates, dtype=bool)
        threshold = max(100, int(image.width * image.height * 0.02))
        for label in range(1, count):
            if int(stats[label, cv2.CC_STAT_AREA]) >= threshold:
                result[labels == label] = True
        return result


def semantic_descriptor(
    engine: SemanticMaskEngine, result: SemanticMaskResult | None = None
) -> dict[str, object]:
    return {
        "status": "fallback" if engine.provider == "deterministic-protection" else "ready",
        "provider": engine.provider,
        "version": engine.version,
        "classes": list(SEMANTIC_CLASSES),
        "order": list(SEMANTIC_ORDER),
        "confidence_threshold": getattr(engine, "confidence_threshold", 0.78),
        "uncertain_ratio": result.uncertain_ratio if result else 0.0,
        "cached": result is not None,
    }


__all__ = [
    "SEMANTIC_CLASSES",
    "SEMANTIC_ORDER",
    "ConservativeSemanticMaskEngine",
    "SemanticMaskEngine",
    "SemanticMaskResult",
    "semantic_descriptor",
]
