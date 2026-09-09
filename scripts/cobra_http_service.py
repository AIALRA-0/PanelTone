from __future__ import annotations

import gc
import json
import logging
import os
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from manga_repaint.color import normalize_color_candidate_size
from manga_repaint.engines.cobra import prepare_cobra_hint

logger = logging.getLogger("paneltone.cobra")
# FastAPI's dependency declarations intentionally call File/Form at definition
# time; this is the framework's documented multipart signature.
# ruff: noqa: B008
app = FastAPI(title="PanelTone Cobra Candidate", version="0.2.0-alpha.3")
lock = threading.RLock()
state: dict[str, Any] = {
    "state": "idle",
    "loaded": False,
    "active_requests": 0,
    "last_request_id": None,
    "last_error": None,
    "backend": "official-cobra",
}
runtime: Any = None


def _repo() -> Path:
    value = os.getenv("PANELTONE_COBRA_REPO", "").strip()
    if not value:
        raise FileNotFoundError("PANELTONE_COBRA_REPO 未配置")
    path = Path(value).resolve()
    if not (path / "app.py").is_file():
        raise FileNotFoundError(f"Cobra app.py 不存在: {path}")
    return path


def _load_runtime() -> Any:
    global runtime
    if runtime is not None:
        return runtime
    repo = _repo()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import importlib

    runtime = importlib.import_module("app")
    return runtime


def _health_payload() -> dict[str, Any]:
    with lock:
        return {**state, "service": "cobra-candidate", "repository": "JunhaoZhuang/Cobra"}


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        repo = _repo()
        available = repo.is_dir()
        payload = _health_payload()
        payload["available"] = available
        payload["configured"] = True
        return payload
    except FileNotFoundError as exc:
        payload = _health_payload()
        payload.update({"available": False, "configured": False, "error": str(exc)})
        return payload


@app.post("/interrupt")
def interrupt() -> dict[str, Any]:
    with lock:
        active = int(state["active_requests"])
        return {
            "status": "interrupt_requested" if active else "idle",
            "request_received": True,
            "active": bool(active),
            "active_requests": active,
            "cancelled": False,
            "cleared": not bool(active),
        }


@app.post("/release")
def release() -> dict[str, Any]:
    global runtime
    with lock:
        if state["active_requests"]:
            raise HTTPException(status_code=409, detail="Cobra 候选服务仍有活动请求")
        runtime = None
        state.update({"loaded": False, "state": "idle"})
        gc.collect()
        return {"status": "released", **_health_payload()}


@app.post("/generate")
def generate(
    source: UploadFile = File(...),
    references: list[UploadFile] = File(default=[]),
    hint: UploadFile | None = File(default=None),
    prompt: str = Form(""),
    negative_prompt: str = Form(""),
    seed: str = Form("0"),
    mode: str = Form("colorize"),
    metadata_json: str = Form("{}"),
) -> FileResponse:
    request_id = uuid.uuid4().hex
    temp_root = (
        Path(os.getenv("PANELTONE_COBRA_TMP", tempfile.gettempdir()))
        / f"paneltone-cobra-{request_id}"
    )
    temp_root.mkdir(parents=True, exist_ok=True)
    with lock:
        if state["active_requests"]:
            raise HTTPException(status_code=409, detail="Cobra 候选服务一次只处理一个请求")
        state.update(
            {
                "state": "generating",
                "active_requests": 1,
                "last_request_id": request_id,
                "last_error": None,
            }
        )
    started = time.monotonic()
    try:
        source_path = temp_root / (Path(source.filename or "source.png").name or "source.png")
        source_path.write_bytes(source.file.read())
        reference_paths: list[Path] = []
        # Cobra is designed to use a broad same-book reference set. The
        # PanelTone adapter uploads bounded, downscaled proxies. Sixty-four
        # same-book references are the measured quality/performance ceiling on
        # this host; larger sets caused an unbounded local inference wait.
        for index, item in enumerate(references[:64]):
            target = (
                temp_root / f"reference-{index:03d}-{Path(item.filename or 'reference.png').name}"
            )
            target.write_bytes(item.file.read())
            reference_paths.append(target)
        if not reference_paths:
            raise HTTPException(status_code=400, detail="Cobra 候选需要至少一张颜色参考图")
        try:
            metadata = json.loads(metadata_json or "{}")
        except ValueError:
            metadata = {}
        cobra = _load_runtime()
        with cobra.Image.open(source_path) as image:
            source_image = image.convert("RGB")
        extracted = cobra.extract_sketch_line_image(
            source_image, str(metadata.get("cobra_style", "line + shadow"))
        )
        extracted_line, _, hint_mask, query_origin, extracted_original, resolution = extracted
        if hint is not None:
            hint_image = cobra.Image.open(hint.file).convert("RGBA")
            try:
                hint_mask, extracted_hint_color = prepare_cobra_hint(
                    extracted_line,
                    hint_image,
                    source_size=source_image.size,
                    resolution=resolution,
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=str(exc),
                ) from exc
        else:
            extracted_hint_color = extracted_line

        class _Ref:
            def __init__(self, name: str) -> None:
                self.name = name

        result = cobra.colorize_image(
            extracted_line,
            [_Ref(str(path)) for path in reference_paths],
            resolution,
            int(seed or 0),
            int(metadata.get("cobra_steps", 10)),
            min(int(metadata.get("cobra_top_k", 6)), len(reference_paths)),
            hint_mask,
            extracted_hint_color,
            query_origin,
            extracted_original,
        )
        output = normalize_color_candidate_size(result[0].convert("RGB"), source_image.size)
        output_path = temp_root / "result.png"
        output.save(output_path, format="PNG")
        elapsed_ms = round((time.monotonic() - started) * 1000, 1)
        with lock:
            state.update({"state": "idle", "loaded": True, "active_requests": 0})
        return FileResponse(
            output_path,
            media_type="image/png",
            headers={
                "x-request-id": request_id,
                "x-model-id": "JunhaoZhuang/Cobra",
                "x-elapsed-ms": str(elapsed_ms),
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Cobra candidate request failed id=%s", request_id)
        with lock:
            state.update({"state": "failed", "active_requests": 0, "last_error": str(exc)})
        raise HTTPException(status_code=500, detail=f"Cobra 候选推理失败: {exc}") from exc
    finally:
        with lock:
            if state["active_requests"]:
                state["active_requests"] = 0
                if state["state"] == "generating":
                    state["state"] = "idle"
