from __future__ import annotations

import gc
import inspect
import io
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

from . import __version__

logger = logging.getLogger("paneltone.model")


class GenerationInterrupted(RuntimeError):
    """Raised when a user pause or cancel reaches the model step callback."""


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
        self._cancel_event = threading.Event()
        self._active_requests = 0
        self.last_activity = 0.0
        # Keep the model warm for a short idle window, then release CPU/GPU
        # weights automatically.  Set the value to 0 to disable the reaper.
        try:
            self.idle_release_seconds = max(
                0.0,
                float(os.getenv("PANELTONE_MODEL_IDLE_RELEASE_SECONDS", "60")),
            )
        except ValueError:
            self.idle_release_seconds = 60.0
        self.state = "idle"
        self.last_error = ""
        self._stop_event = threading.Event()
        self._reaper = threading.Thread(
            target=self._idle_reaper_loop,
            name="paneltone-model-idle-reaper",
            daemon=True,
        )
        self._reaper.start()

    def pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        with self._load_lock:
            if self._pipeline is not None:
                return self._pipeline
            self.state = "loading"
            self.last_error = ""
            logger.info("loading model=%s cpu_offload=%s", self.model_id, self.cpu_offload)
            try:
                import torch
                from diffusers import Flux2KleinPipeline
            except ImportError as exc:
                self.state = "failed"
                self.last_error = str(exc)
                logger.exception("model dependencies missing model=%s", self.model_id)
                raise RuntimeError(
                    "Model service dependencies are missing; run scripts/install_flux2_klein.ps1"
                ) from exc
            try:
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
                self.state = "ready"
                self.last_activity = time.time()
                logger.info("model ready model=%s", self.model_id)
                return pipeline
            except Exception as exc:
                self.state = "failed"
                self.last_error = str(exc)
                logger.exception("model load failed model=%s", self.model_id)
                raise

    def generate(
        self,
        source: Image.Image,
        references: list[Image.Image],
        prompt: str,
        seed: int,
        *,
        negative_prompt: str = "",
        mode: str = "style_locked",
        metadata: dict[str, object] | None = None,
    ):
        import torch

        width = max(256, min(1536, round(source.width / 16) * 16))
        height = max(256, min(1536, round(source.height / 16) * 16))
        images = [source, *references[:3]]
        with self.inference_lock, torch.inference_mode():
            pipeline = self.pipeline()
            if self._cancel_event.is_set():
                self._cancel_event.clear()
                raise GenerationInterrupted("模型请求在开始前已被中断")
            self.state = "generating"
            self._active_requests += 1
            self.last_activity = time.time()
            try:
                metadata = metadata or {}
                style_guidance = str(metadata.get("style_guidance") or "").strip()
                if style_guidance:
                    prompt = f"{prompt.rstrip('. ')}. {style_guidance}"
                mode_guidance = {
                    "colorize": (
                        "Keep the original drawing and only add believable colour and light."
                    ),
                    "style_locked": (
                        "Keep the exact drawing, layout and identities while changing "
                        "rendering style."
                    ),
                    "style_full": (
                        "Use a clearly different rendering style but keep content and "
                        "layout recognizable."
                    ),
                }.get(mode)
                if mode_guidance:
                    prompt = f"{prompt.rstrip('. ')}. {mode_guidance}"
                try:
                    guidance_scale = min(20.0, max(0.0, float(metadata.get("guidance_scale", 1.0))))
                except (TypeError, ValueError):
                    guidance_scale = 1.0
                try:
                    inference_steps = min(100, max(1, int(metadata.get("num_inference_steps", 4))))
                except (TypeError, ValueError):
                    inference_steps = 4
                call_kwargs = {
                    "image": images if len(images) > 1 else images[0],
                    "prompt": prompt
                    or (
                        "Colorize or restyle this manga panel while preserving every character, "
                        "object, "
                        "pose, camera angle, panel layout, speech bubble, and text placement"
                    ),
                    "height": height,
                    "width": width,
                    "guidance_scale": guidance_scale,
                    "num_inference_steps": inference_steps,
                    "generator": torch.Generator(device=self.device).manual_seed(seed),
                }
                # Diffusers has used two callback spellings across releases.  Add
                # the one supported by the installed pipeline so a pause can
                # stop between denoising steps instead of waiting for a timeout.
                try:
                    parameters = inspect.signature(pipeline.__call__).parameters
                except (TypeError, ValueError):
                    parameters = {}
                if "negative_prompt" in parameters and negative_prompt.strip():
                    call_kwargs["negative_prompt"] = negative_prompt.strip()
                elif negative_prompt.strip():
                    # Some FLUX pipelines do not expose classifier-free
                    # negative conditioning.  Keep the constraint visible in
                    # the prompt instead of silently discarding it.
                    call_kwargs["prompt"] = (
                        f"{call_kwargs['prompt']}. Avoid {negative_prompt.strip()}"
                    )
                for name in ("guidance_scale", "num_inference_steps"):
                    if name not in parameters:
                        call_kwargs.pop(name, None)
                if "callback_on_step_end" in parameters:
                    call_kwargs["callback_on_step_end"] = self._step_callback
                    if "callback_on_step_end_tensor_inputs" in parameters:
                        call_kwargs["callback_on_step_end_tensor_inputs"] = []
                elif "callback" in parameters:
                    call_kwargs["callback"] = self._legacy_callback
                    if "callback_steps" in parameters:
                        call_kwargs["callback_steps"] = 1
                return pipeline(**call_kwargs).images[0]
            except Exception as exc:
                self.last_error = str(exc)
                raise
            finally:
                self._active_requests = max(0, self._active_requests - 1)
                self.last_activity = time.time()
                self.state = "ready" if self._pipeline is not None else "failed"

    def _step_callback(self, _pipeline, _step_index, _timestep, callback_kwargs):
        if self._cancel_event.is_set():
            self._cancel_event.clear()
            raise GenerationInterrupted("模型请求已被用户中断")
        return callback_kwargs

    def _legacy_callback(self, *_args):
        if self._cancel_event.is_set():
            self._cancel_event.clear()
            raise GenerationInterrupted("模型请求已被用户中断")

    def request_interrupt(self) -> dict[str, object]:
        active = self.state == "generating" and self._active_requests > 0
        if active:
            self._cancel_event.set()
            logger.info("generation interrupt requested")
            return {"status": "interrupt_requested", "active": True}
        return {"status": "idle", "active": False}

    def release_if_idle(self, now: float | None = None) -> dict[str, object]:
        """Release a warm pipeline after the configured idle grace period."""
        if self.idle_release_seconds <= 0:
            return {"status": "disabled", "active": False, "state": self.state}
        if self.state != "ready" or self._pipeline is None or self._active_requests:
            return {"status": "not_idle", "active": self._active_requests > 0, "state": self.state}
        current = time.time() if now is None else now
        if not self.last_activity or current - self.last_activity < self.idle_release_seconds:
            return {"status": "not_due", "active": False, "state": self.state}
        return self.release()

    def _idle_reaper_loop(self) -> None:
        interval = min(5.0, max(1.0, self.idle_release_seconds / 4))
        while not self._stop_event.wait(interval):
            try:
                result = self.release_if_idle()
                if result.get("status") == "released":
                    logger.info(
                        "model pipeline auto-released after idle seconds=%s",
                        self.idle_release_seconds,
                    )
            except Exception:
                logger.exception("idle model release failed")

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._reaper.is_alive():
            self._reaper.join(timeout=1.0)

    def release(self) -> dict[str, object]:
        """Unload only when no request is using the pipeline."""
        if not self.inference_lock.acquire(blocking=False):
            return {"status": "busy", "active": True, "state": self.state}
        try:
            with self._load_lock:
                if self.state in {"loading", "generating"} or self._active_requests:
                    return {"status": "busy", "active": True, "state": self.state}
                pipeline = self._pipeline
                self._pipeline = None
                self._cancel_event.clear()
                self.state = "idle"
                self.last_activity = time.time()
        finally:
            self.inference_lock.release()
        if pipeline is not None:
            del pipeline
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except (ImportError, RuntimeError) as exc:
            logger.warning("cuda cache release incomplete: %s", exc)
        logger.info("model pipeline released")
        return {"status": "released", "active": False, "state": self.state}


runtime = Flux2Runtime()
app = FastAPI(title="PanelTone Model Service", version=__version__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    runtime.shutdown()


app.router.lifespan_context = lifespan


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
        "state": runtime.state,
        "last_error": runtime.last_error,
        "active_requests": runtime._active_requests,
        "last_activity": runtime.last_activity,
        "idle_release_seconds": runtime.idle_release_seconds,
        "supports_interrupt": True,
        "supports_release": True,
    }


@app.post("/interrupt", dependencies=[Depends(verify_key)])
def interrupt() -> dict[str, object]:
    return runtime.request_interrupt()


@app.post("/release", dependencies=[Depends(verify_key)])
@app.post("/unload", dependencies=[Depends(verify_key)])
def release() -> dict[str, object]:
    result = runtime.release()
    if result.get("status") == "busy":
        raise HTTPException(status_code=409, detail="模型当前正在使用，完成或中断后才能释放显存")
    return result


@app.post("/generate", dependencies=[Depends(verify_key)])
def generate(
    source: UploadFile,
    references: list[UploadFile] | None = None,
    prompt: Annotated[str, Form()] = "",
    negative_prompt: Annotated[str, Form()] = "",
    seed: Annotated[int, Form()] = 0,
    mode: Annotated[str, Form()] = "style_locked",
    metadata_json: Annotated[str, Form()] = "{}",
) -> Response:
    try:
        metadata = json.loads(metadata_json or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="模型参数 JSON 无效") from exc
    if not isinstance(metadata, dict):
        raise HTTPException(status_code=400, detail="模型参数必须是对象")
    source_image = Image.open(io.BytesIO(source.file.read())).convert("RGB")
    reference_images = []
    for reference in references or []:
        reference_images.append(Image.open(io.BytesIO(reference.file.read())).convert("RGB"))
    started = time.perf_counter()
    logger.info(
        "generation started model=%s size=%sx%s references=%s seed=%s",
        runtime.model_id,
        source_image.width,
        source_image.height,
        len(reference_images),
        seed,
    )
    try:
        result = runtime.generate(
            source_image,
            reference_images,
            prompt,
            seed,
            negative_prompt=negative_prompt,
            mode=mode,
            metadata=metadata,
        )
    except GenerationInterrupted as exc:
        logger.info("generation interrupted model=%s", runtime.model_id)
        raise HTTPException(status_code=499, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("generation failed model=%s", runtime.model_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    buffer = io.BytesIO()
    result.save(buffer, format="PNG")
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    logger.info("generation finished model=%s elapsed_ms=%s", runtime.model_id, elapsed_ms)
    return Response(
        buffer.getvalue(),
        media_type="image/png",
        headers={"X-Model-Id": runtime.model_id, "X-Elapsed-Ms": str(elapsed_ms)},
    )
