from __future__ import annotations

import io
import os
import threading
import time
from typing import Annotated

from fastapi import Depends, FastAPI, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image


class Flux2Runtime:
    def __init__(self) -> None:
        self.model_id = os.getenv(
            "PANELTONE_MODEL_ID",
            os.getenv("MANGA_REPAINT_MODEL_ID", "black-forest-labs/FLUX.2-klein-4B"),
        )
        self.device = os.getenv(
            "PANELTONE_MODEL_DEVICE",
            os.getenv("MANGA_REPAINT_MODEL_DEVICE", "cuda"),
        )
        self.cpu_offload = (
            os.getenv(
                "PANELTONE_MODEL_CPU_OFFLOAD",
                os.getenv("MANGA_REPAINT_MODEL_CPU_OFFLOAD", "1"),
            )
            == "1"
        )
        self._pipeline = None
        self._load_lock = threading.Lock()
        self.inference_lock = threading.Lock()

    def pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        with self._load_lock:
            if self._pipeline is not None:
                return self._pipeline
            try:
                import torch
                from diffusers import Flux2KleinPipeline
            except ImportError as exc:
                raise RuntimeError(
                    "Model service dependencies are missing; run scripts/install_flux2_klein.ps1"
                ) from exc
            pipeline = Flux2KleinPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
            )
            if self.cpu_offload:
                pipeline.enable_model_cpu_offload()
            else:
                pipeline.to(self.device)
            pipeline.set_progress_bar_config(disable=True)
            self._pipeline = pipeline
            return pipeline

    def generate(self, source: Image.Image, references: list[Image.Image], prompt: str, seed: int):
        import torch

        pipeline = self.pipeline()
        width = max(256, min(1536, round(source.width / 16) * 16))
        height = max(256, min(1536, round(source.height / 16) * 16))
        images = [source, *references[:3]]
        with self.inference_lock, torch.inference_mode():
            return pipeline(
                image=images if len(images) > 1 else images[0],
                prompt=prompt
                or (
                    "Colorize or restyle this manga panel while preserving every character, "
                    "object, "
                    "pose, camera angle, panel layout, speech bubble, and text placement"
                ),
                height=height,
                width=width,
                guidance_scale=1.0,
                num_inference_steps=4,
                generator=torch.Generator(device=self.device).manual_seed(seed),
            ).images[0]


runtime = Flux2Runtime()
app = FastAPI(title="PanelTone Model Service", version="0.2.0-alpha.1")


def verify_key(authorization: Annotated[str | None, Header()] = None) -> None:
    expected = os.getenv(
        "PANELTONE_MODEL_API_KEY",
        os.getenv("MANGA_REPAINT_MODEL_API_KEY", ""),
    )
    if expected and authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Invalid model service key")


@app.get("/health", dependencies=[Depends(verify_key)])
def health() -> dict[str, object]:
    return {
        "service": "flux2-klein",
        "model_id": runtime.model_id,
        "loaded": runtime._pipeline is not None,
        "cpu_offload": runtime.cpu_offload,
    }


@app.post("/generate", dependencies=[Depends(verify_key)])
async def generate(
    source: UploadFile,
    references: list[UploadFile] | None = None,
    prompt: Annotated[str, Form()] = "",
    negative_prompt: Annotated[str, Form()] = "",
    seed: Annotated[int, Form()] = 0,
    mode: Annotated[str, Form()] = "style_locked",
    metadata_json: Annotated[str, Form()] = "{}",
) -> Response:
    del negative_prompt, mode, metadata_json
    source_image = Image.open(io.BytesIO(await source.read())).convert("RGB")
    reference_images = []
    for reference in references or []:
        reference_images.append(Image.open(io.BytesIO(await reference.read())).convert("RGB"))
    started = time.perf_counter()
    try:
        result = runtime.generate(source_image, reference_images, prompt, seed)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    buffer = io.BytesIO()
    result.save(buffer, format="PNG")
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    return Response(
        buffer.getvalue(),
        media_type="image/png",
        headers={"X-Model-Id": runtime.model_id, "X-Elapsed-Ms": str(elapsed_ms)},
    )
