from __future__ import annotations

import base64
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from .semantic import SEMANTIC_CLASSES, KoharuSemanticMaskEngine

SUPPORTED_CLASSES = ("text", "bubbles", "borders", "ink")


def _encoded_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    # These masks stay on the loopback service boundary.  Avoid Pillow's
    # expensive optimizer for every full-page layer; low compression keeps
    # segmentation latency bounded without changing any mask pixels.
    image.save(buffer, format="PNG", optimize=False, compress_level=1)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


model_dir = Path(
    os.getenv("PANELTONE_SEMANTIC_MODEL_DIR", "models/semantic/koharu-yolo26s")
)
engine = KoharuSemanticMaskEngine(model_dir)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield


app = FastAPI(title="PanelTone Semantic Protection Service", version="0.2.0-alpha.3")
app.router.lifespan_context = lifespan


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "service": "semantic-manga-v1-compatible",
        "provider": engine.provider,
        "version": engine.version,
        "available": all(
            (engine.model_dir / filename).is_file()
            for filename in ("model.safetensors", "config.json", "yolo26s-seg.yaml")
        ),
        "classes": list(SEMANTIC_CLASSES),
        "supported_classes": list(SUPPORTED_CLASSES),
        "unsupported_classes": [
            name for name in SEMANTIC_CLASSES if name not in SUPPORTED_CLASSES
        ],
    }


@app.post("/segment")
def segment(file: Annotated[UploadFile, File(...)]) -> dict[str, object]:
    try:
        image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
        result = engine.segment(image)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="语义模型当前不可用") from exc
    return {
        "provider": result.provider,
        "version": result.version,
        "confidence_threshold": engine.confidence_threshold,
        "uncertain_ratio": result.uncertain_ratio,
        "masks": {
            name: _encoded_png(Image.fromarray(mask.astype("uint8") * 255, mode="L"))
            for name, mask in result.masks.items()
        },
        "confidence": _encoded_png(
            Image.fromarray((result.confidence.clip(0, 1) * 255).astype("uint8"), mode="L")
        ),
        "uncertain": _encoded_png(
            Image.fromarray(result.uncertain.astype("uint8") * 255, mode="L")
        ),
    }


__all__ = ["app"]
