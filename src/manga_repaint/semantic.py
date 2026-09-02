from __future__ import annotations

import base64
import io
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import httpx
import numpy as np
from PIL import Image

from .masks import border_mask, bubble_mask, ink_detail_mask, text_region_mask

logger = logging.getLogger("paneltone.semantic")

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
        _, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, 8)
        threshold = max(100, int(image.width * image.height * 0.02))
        valid = stats[:, cv2.CC_STAT_AREA] >= threshold
        valid[0] = False
        return valid[labels].astype(bool)


class KoharuSemanticMaskEngine:
    """Lazy local adapter for the four-class Koharu manga segmenter.

    The published model covers page frames, dialogue text, balloons and
    onomatopoeia.  Ink and border protection still comes from the deterministic
    masks so a layout model is never treated as an ink detector.
    """

    provider = "semantic-manga-v1-compatible/koharu-yolo26s"

    def __init__(
        self,
        model_dir: Path,
        *,
        version: str = "koharu-yolo26s",
        confidence_threshold: float = 0.78,
        inference_confidence: float = 0.20,
        image_size: int = 1280,
        inference_device: str | int | None = None,
    ) -> None:
        self.model_dir = model_dir.resolve()
        self.version = version
        self.confidence_threshold = confidence_threshold
        self.inference_confidence = inference_confidence
        self.image_size = image_size
        self.inference_device = (
            inference_device if inference_device is not None else self._default_device()
        )
        self._model = None
        self._names: dict[int, str] = {}
        self._load_error: str | None = None

    @staticmethod
    def _default_device() -> str | int:
        requested = os.getenv("PANELTONE_SEMANTIC_DEVICE", "auto").strip().casefold()
        if requested and requested != "auto":
            return int(requested) if requested.isdigit() else requested
        try:
            import torch

            return 0 if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _load(self):
        if self._model is not None:
            return self._model
        if self._load_error:
            raise RuntimeError(self._load_error)
        try:
            from safetensors.torch import load_file
            from ultralytics import YOLO

            config_path = self.model_dir / "config.json"
            architecture_path = self.model_dir / "yolo26s-seg.yaml"
            weights_path = self.model_dir / "model.safetensors"
            if not all(path.is_file() for path in (config_path, architecture_path, weights_path)):
                raise FileNotFoundError(
                    "语义模型目录缺少 config.json、yolo26s-seg.yaml 或 model.safetensors"
                )
            config = json.loads(config_path.read_text(encoding="utf-8"))
            names = config.get("names", {})
            if isinstance(names, list):
                self._names = {index: str(value) for index, value in enumerate(names)}
            elif isinstance(names, dict):
                self._names = {int(key): str(value) for key, value in names.items()}
            else:
                raise ValueError("语义模型类别配置无效")
            model = YOLO(str(architecture_path), task="segment")
            model.model.load_state_dict(load_file(str(weights_path)), strict=True)
            model.model.names = self._names
            self._model = model
            return model
        except Exception as exc:  # pragma: no cover - exercised by the optional runtime
            self._load_error = f"语义模型加载失败: {exc}"
            raise RuntimeError(self._load_error) from exc

    @staticmethod
    def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
        return cv2.resize(
            mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    def segment(self, image: Image.Image) -> SemanticMaskResult:
        model = self._load()
        source = image.convert("RGB")
        width, height = source.size
        fallback = ConservativeSemanticMaskEngine(self.confidence_threshold).segment(source)
        masks = {name: value.copy() for name, value in fallback.masks.items()}
        confidence = fallback.confidence.copy()
        prediction = model.predict(
            source=np.asarray(source),
            imgsz=self.image_size,
            conf=self.inference_confidence,
            verbose=False,
            device=self.inference_device,
        )[0]
        result_masks = getattr(prediction, "masks", None)
        boxes = getattr(prediction, "boxes", None)
        if result_masks is not None and boxes is not None and result_masks.data is not None:
            raw_masks = result_masks.data.detach().cpu().numpy()
            classes = boxes.cls.detach().cpu().numpy().astype(int)
            scores = boxes.conf.detach().cpu().numpy().astype(float)
            for raw_mask, class_id, score in zip(raw_masks, classes, scores, strict=False):
                region = self._resize_mask(raw_mask > 0.5, width, height)
                name = self._names.get(int(class_id), "")
                if name in {"dialogue_text", "onomatopoeia_text", "text"}:
                    target = "text"
                elif name in {"balloon", "bubble", "bubbles"}:
                    target = "bubbles"
                elif name in {"frame", "panel", "borders", "border"}:
                    # A frame prediction covers its panel interior.  Protect
                    # only the frame edge here; the deterministic detector is
                    # still the authority for actual border pixels.
                    region = cv2.morphologyEx(
                        region.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
                    ).astype(bool)
                    target = "borders"
                else:
                    continue
                masks[target] |= region
                confidence[region] = np.maximum(confidence[region], float(score))
        protected = masks["text"] | masks["bubbles"] | masks["borders"] | masks["ink"]
        confidence[protected] = np.maximum(confidence[protected], 1.0)
        uncertain = confidence < self.confidence_threshold
        return SemanticMaskResult(
            masks=masks,
            confidence=confidence,
            uncertain=uncertain,
            provider=self.provider,
            version=self.version,
        )


class RemoteSemanticMaskEngine:
    """Call the optional semantic service without importing its GPU stack."""

    def __init__(
        self,
        base_url: str,
        *,
        version: str = "koharu-yolo26s",
        confidence_threshold: float = 0.78,
        timeout_seconds: float = 180.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.provider = "semantic-manga-v1-compatible/koharu-yolo26s"
        self.version = version
        self.confidence_threshold = confidence_threshold
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _decode_mask(value: str, *, mode: str) -> np.ndarray:
        with Image.open(io.BytesIO(base64.b64decode(value))) as image:
            array = np.asarray(image.convert(mode))
        return array

    @staticmethod
    def _fit_array(array: np.ndarray, width: int, height: int) -> np.ndarray:
        if array.shape == (height, width):
            return array
        return cv2.resize(array, (width, height), interpolation=cv2.INTER_NEAREST)

    def segment(self, image: Image.Image) -> SemanticMaskResult:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        response = httpx.post(
            f"{self.base_url}/segment",
            files={"file": ("page.png", buffer.getvalue(), "image/png")},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        masks = {
            name: self._fit_array(
                self._decode_mask(encoded, mode="1").astype(bool), image.width, image.height
            )
            for name, encoded in dict(payload.get("masks") or {}).items()
        }
        for name in SEMANTIC_CLASSES:
            masks.setdefault(name, np.zeros((image.height, image.width), dtype=bool))
        confidence = self._fit_array(
            self._decode_mask(str(payload["confidence"]), mode="L"), image.width, image.height
        ).astype(np.float32)
        confidence /= 255.0
        uncertain = self._fit_array(
            self._decode_mask(str(payload["uncertain"]), mode="1"), image.width, image.height
        ).astype(bool)
        return SemanticMaskResult(
            masks=masks,
            confidence=confidence,
            uncertain=uncertain,
            provider=str(payload.get("provider") or self.provider),
            version=str(payload.get("version") or self.version),
        )


def configured_semantic_engine(model_root: Path) -> SemanticMaskEngine:
    """Select the local adapter only when its explicit service is configured."""
    base_url = os.getenv("PANELTONE_SEMANTIC_URL", "").strip()
    if base_url:
        return RemoteSemanticMaskEngine(base_url)
    model_dir = Path(
        os.getenv(
            "PANELTONE_SEMANTIC_MODEL_DIR",
            str(model_root / "semantic" / "koharu-yolo26s"),
        )
    )
    if all(
        (model_dir / filename).is_file()
        for filename in ("model.safetensors", "config.json", "yolo26s-seg.yaml")
    ):
        return KoharuSemanticMaskEngine(model_dir)
    return ConservativeSemanticMaskEngine()


def semantic_descriptor(
    engine: SemanticMaskEngine, result: SemanticMaskResult | None = None
) -> dict[str, object]:
    provider = result.provider if result is not None else engine.provider
    supported_classes = ["text", "bubbles", "borders", "ink"]
    if provider == "deterministic-protection":
        supported_classes.append("background")
    unsupported_classes = [
        name for name in SEMANTIC_CLASSES if name not in supported_classes
    ]
    return {
        "status": "fallback" if provider == "deterministic-protection" else "ready",
        "provider": provider,
        "version": result.version if result is not None else engine.version,
        "classes": list(SEMANTIC_CLASSES),
        "supported_classes": supported_classes,
        "unsupported_classes": unsupported_classes,
        "order": list(SEMANTIC_ORDER),
        "confidence_threshold": getattr(engine, "confidence_threshold", 0.78),
        "uncertain_ratio": result.uncertain_ratio if result else 0.0,
        "cached": result is not None,
    }


__all__ = [
    "SEMANTIC_CLASSES",
    "SEMANTIC_ORDER",
    "ConservativeSemanticMaskEngine",
    "KoharuSemanticMaskEngine",
    "RemoteSemanticMaskEngine",
    "SemanticMaskEngine",
    "SemanticMaskResult",
    "configured_semantic_engine",
    "semantic_descriptor",
]
