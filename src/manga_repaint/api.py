from __future__ import annotations

import io
import json
import logging
import re
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

import httpx
import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from . import __version__
from .catalog import Catalog
from .config import Settings
from .engines import EngineRegistry
from .models import DetailMode, JobMode, JobSpec, JobStatus, ProtectionMode
from .observability import RawLogStore, SystemTelemetry, redact_sensitive_text
from .presets import presets_payload
from .project import DisplayAssetPending, ProjectManager
from .runtime import EventHub, JobQueue
from .schemas import (
    FolderNode,
    GpuMetrics,
    ImportOperation,
    JobProgress,
    ModelDescriptor,
    PresetCatalog,
    RawLogEntry,
    SourceDescriptor,
)
from .semantic import SEMANTIC_CLASSES, semantic_descriptor

logger = logging.getLogger("paneltone.api")

ALLOWED_UPLOADS = {
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".tif": "image",
    ".tiff": "image",
    ".bmp": "image",
    ".pdf": "book",
    ".zip": "book",
    ".cbz": "book",
    ".rar": "book",
    ".cbr": "book",
}

DOWNLOAD_OUTPUT_NAMES = {
    "book.cbz",
    "book.pdf",
    "book-images.zip",
    "book-jpeg.zip",
    "book-webp.zip",
}


def _iter_file_range(path: Path, start: int, end: int, chunk_size: int = 1024 * 1024):
    """Yield one bounded file range without buffering the complete archive."""
    with path.open("rb") as stream:
        stream.seek(start)
        remaining = end - start
        while remaining > 0:
            chunk = stream.read(min(chunk_size, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


class JobOptions(BaseModel):
    mode: JobMode = JobMode.COLORIZE
    engine: str = "palette"
    protection: ProtectionMode = ProtectionMode.STRICT
    detail_mode: DetailMode = DetailMode.STRICT
    output_format: Literal["cbz", "pdf", "images", "jpeg", "webp"] = "cbz"
    panel_mode: Literal["page", "detect"] = "page"
    seed: int = 0
    prompt: str = ""
    negative_prompt: str = ""
    color_preset: str = "natural"
    style_preset: str = "original_ink"
    preserve_text: bool = True
    preserve_ink: bool = True
    ink_gamma: float = Field(default=0.42, ge=0.05, le=2.0)
    chroma_strength: float = Field(default=1.15, ge=0.0, le=2.5)
    style_reference_ids: list[str] = Field(default_factory=list)
    max_retries: int = Field(default=2, ge=0, le=10)
    adult_fictional_content: bool = False


class CreateJobRequest(JobOptions):
    source: str | None = None
    source_id: str | None = None
    display_name: str = "未命名漫画"


class BatchJobRequest(JobOptions):
    source_ids: list[str] = Field(min_length=1)
    image_order: list[str] = Field(default_factory=list)
    image_book_name: str = "图片合集"


class RenameJobRequest(BaseModel):
    display_name: str = Field(min_length=1, max_length=120)


class ReorderJobsRequest(BaseModel):
    job_ids: list[str]


class FolderRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    parent_id: str | None = None


class FolderPatchRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    parent_id: str | None = None


class LibraryMoveRequest(BaseModel):
    folder_id: str | None = None
    before_job_id: str | None = None


class LibraryReorderRequest(BaseModel):
    folder_id: str | None = None
    job_ids: list[str] = Field(default_factory=list)
    # Folder ordering uses the same endpoint as job ordering so a drag/drop
    # client only needs one persistence contract. ``parent_id`` names the
    # folder whose direct children are being reordered; ``folder_ids`` must
    # contain every active sibling exactly once.
    parent_id: str | None = None
    folder_ids: list[str] = Field(default_factory=list)


class DeleteJobRequest(BaseModel):
    confirmation: Literal["永久删除"]


def _option_payload() -> dict[str, list[dict[str, str]]]:
    return {
        "modes": [
            {
                "id": "colorize",
                "name": "原作上色",
                "description": "只添加颜色，原始构图和线稿不会改变",
                "best_for": "黑白漫画批量上色",
                "changes": "颜色、环境光和材质",
                "tradeoff": "画风变化较小",
            },
            {
                "id": "style_locked",
                "name": "锁定内容换画风",
                "description": "锁定构图、文字和主要线条，再改变渲染风格",
                "best_for": "内容一致性优先的风格转换",
                "changes": "颜色、阴影、材质和局部笔触",
                "tradeoff": "完整画风变化受线稿保护限制",
            },
            {
                "id": "style_full",
                "name": "完整画风重绘",
                "description": "可能改变线条、表情、构图和细节",
                "best_for": "画风变化优先的页面",
                "changes": "线条、五官画法、阴影和材质",
                "tradeoff": "人物细节可能发生变化",
            },
        ],
        "details": [
            {
                "id": "strict",
                "name": "严格保真",
                "description": "使用原图亮度和纯黑墨线，细节最稳定",
                "best_for": "文字、网点与线条保护",
                "changes": "主要改变色彩",
                "tradeoff": "颜色相对柔和",
            },
            {
                "id": "balanced",
                "name": "均衡",
                "description": "保留主要墨线，同时允许更丰富的明暗",
                "best_for": "大多数彩色漫画页面",
                "changes": "色彩与部分明暗",
                "tradeoff": "细线可能略有变化",
            },
            {
                "id": "generative",
                "name": "生成优先",
                "description": "采用模型完整结果，只回贴受保护区域",
                "best_for": "完整风格变化",
                "changes": "画面中的大部分像素",
                "tradeoff": "内容一致性最低",
            },
        ],
        "panels": [
            {
                "id": "page",
                "name": "整页处理",
                "description": "整页一次生成，跨格色调更统一",
                "best_for": "普通页面和快速处理",
                "changes": "不改变页结构",
                "tradeoff": "复杂页面的小人物细节较弱",
            },
            {
                "id": "detect",
                "name": "自动分镜",
                "description": "识别分镜后逐格生成并重新组装",
                "best_for": "复杂页面和人物细节",
                "changes": "每个分镜独立生成",
                "tradeoff": "速度较慢，跨格颜色需要参考约束",
            },
        ],
        "outputs": [
            {
                "id": "cbz",
                "name": "CBZ 漫画包",
                "description": "保留 PNG 页面并打包，适合漫画阅读器",
                "best_for": "日常阅读和无损母版",
                "changes": "只改变容器格式",
                "tradeoff": "文件通常比 PDF 大",
            },
            {
                "id": "pdf",
                "name": "PDF",
                "description": "把页面重新组装为通用 PDF",
                "best_for": "分享和打印",
                "changes": "页面进入 PDF 容器",
                "tradeoff": "阅读器对超长图片支持不同",
            },
            {
                "id": "images",
                "name": "PNG 图片包",
                "description": "保留逐页 PNG 并打包下载，不改变页面像素",
                "best_for": "继续编辑和质量检查",
                "changes": "只改变下载容器格式",
                "tradeoff": "下载后需要解压文件夹",
            },
            {
                "id": "jpeg",
                "name": "JPEG 图片包",
                "description": "以高质量 JPEG 打包逐页图片，兼容性好",
                "best_for": "通用图片查看和分享",
                "changes": "页面改为高质量有损压缩",
                "tradeoff": "细线和文字边缘可能有轻微压缩损失",
            },
            {
                "id": "webp",
                "name": "WEBP 图片包",
                "description": "以无损 WEBP 打包逐页图片，体积更小",
                "best_for": "现代浏览器和本地归档",
                "changes": "页面编码为无损 WEBP",
                "tradeoff": "部分旧阅读器不支持 WEBP",
            },
        ],
    }


def create_app(
    settings: Settings | None = None,
    registry: EngineRegistry | None = None,
    engine_config: Path = Path("configs/engines.json"),
) -> FastAPI:
    settings = settings or Settings.from_env()
    registry = registry or EngineRegistry.from_json(engine_config, settings.comfyui_url)
    catalog = Catalog(settings.data_root)
    manager = ProjectManager(settings, registry)
    model_catalog = json.loads(Path(__file__).with_name("model_catalog.json").read_text("utf-8"))

    def safe_message(value: object) -> str | None:
        if value is None:
            return None
        known_roots = (
            settings.data_root.resolve(),
            settings.model_root.resolve(),
            Path.cwd().resolve(),
        )
        return redact_sensitive_text(str(value), known_roots)

    def normalize_relative_path(value: str | None, fallback_name: str) -> str:
        """Keep browser folder paths as metadata while rejecting traversal."""
        candidate = (value or fallback_name).replace("\\", "/").strip()
        if not candidate:
            candidate = fallback_name
        path = PurePosixPath(candidate)
        if (
            path.is_absolute()
            # Reject both absolute Windows drives (``C:/foo``) and
            # drive-relative forms (``C:foo``).  The latter is not absolute
            # according to ``PurePosixPath`` but still escapes the virtual
            # folder namespace when interpreted by a Windows client.
            or re.match(r"^[A-Za-z]:", candidate)
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("相对路径无效")
        normalized = "/".join(path.parts)
        if len(normalized) > 512:
            raise ValueError("相对路径过长")
        return normalized

    def public_job(job: dict[str, Any]) -> dict[str, Any]:
        spec = dict(job.get("spec", {}))
        source_value = spec.pop("source", None)
        spec.pop("workspace", None)
        references = spec.pop("style_references", [])
        spec.pop("metadata", None)
        progress = dict(job.get("progress", {}))
        source_bytes = int(
            progress.get("bytes_total")
            or progress.get("total_upload_bytes")
            or 0
        )
        if source_bytes <= 0 and source_value:
            source_path = Path(source_value)
            try:
                stat_key = str(source_path.resolve())
                mtime = source_path.stat().st_mtime_ns
                cached = source_size_cache.get(stat_key)
                if cached and cached[0] == mtime:
                    source_bytes = cached[1]
                elif source_path.is_file():
                    source_bytes = source_path.stat().st_size
                    source_size_cache[stat_key] = (mtime, source_bytes)
                elif source_path.is_dir():
                    source_bytes = sum(
                        path.stat().st_size for path in source_path.rglob("*") if path.is_file()
                    )
                    source_size_cache[stat_key] = (mtime, source_bytes)
            except OSError:
                source_bytes = 0
        spec["style_reference_count"] = len(references)
        job["spec"] = spec
        progress["uploaded_bytes"] = source_bytes
        progress["total_upload_bytes"] = source_bytes
        # Result repair is a CPU-only maintenance pass and must not look like
        # a completed job is generating again. Surface its own progress while
        # the operation is running, then fall back to the manifest snapshot
        # once the operation reaches a terminal state.
        latest_operation = catalog.latest_operation_for_job(str(job["id"]))
        if (
            latest_operation
            and latest_operation.get("status") == "running"
            and latest_operation.get("stage") == "repairing"
        ):
            operation_progress = dict(latest_operation.get("progress") or {})
            progress["stage"] = "repairing"
            progress["stage_percent"] = float(
                operation_progress.get("stage_percent") or 0.0
            )
            progress["percent"] = progress["stage_percent"]
            progress["seconds_per_megapixel"] = None
            progress["eta_seconds"] = None
            progress["latest_message"] = (
                operation_progress.get("latest_message") or "正在重组已生成页面"
            )
        elif (
            latest_operation
            and latest_operation.get("status") == "completed"
            and latest_operation.get("stage") == "ready"
            and int((latest_operation.get("progress") or {}).get("repaired_pages") or 0) > 0
            and str(job.get("status")) not in {JobStatus.RUNNING.value, JobStatus.QUEUED.value}
        ):
            # Historical CPU repair timestamps are not inference samples. Do
            # not expose their misleading speed or ETA after an app restart.
            progress["seconds_per_megapixel"] = None
            progress["eta_seconds"] = None
            progress["latest_message"] = "已有页面结果，速度将在新的模型处理后重新估算"
        job["progress"] = progress
        job["error"] = safe_message(job.get("error"))
        return job

    def public_jobs(include_archived: bool = False) -> list[dict[str, Any]]:
        jobs_by_id = {item["id"]: item for item in manager.list_jobs()}
        catalog_jobs = catalog.jobs(include_archived=include_archived)
        known = {item["id"] for item in catalog_jobs}
        for job_id, job in jobs_by_id.items():
            if job_id not in known:
                name = job.get("spec", {}).get("display_name") or f"漫画 {job_id[:8]}"
                catalog.upsert_job(job_id, name, job["status"])
                catalog_jobs.append(catalog.job(job_id) or {})
        results: list[dict[str, Any]] = []
        for catalog_job in catalog_jobs:
            job = jobs_by_id.get(catalog_job.get("id"))
            if job is None:
                continue
            job["display_name"] = catalog_job["display_name"]
            job["queue_position"] = catalog_job["queue_position"]
            job["archived_at"] = catalog_job["archived_at"]
            job["folder_id"] = catalog_job.get("folder_id")
            job["library_order"] = catalog_job.get("library_order", 0)
            if catalog_job["archived_at"]:
                job["status"] = JobStatus.ARCHIVED.value
            results.append(public_job(job))
        return results

    def derived_asset_url(
        path: Path, base: str, asset_revision: str | None = None
    ) -> str:
        """Add a file revision so a repaired page is fetched immediately."""
        if asset_revision:
            version = asset_revision
        else:
            try:
                version = path.stat().st_mtime_ns
            except OSError:
                version = 0
        return f"{base}?v={version}"

    def sanitize_event_payload(payload: dict[str, Any]) -> dict[str, Any]:
        def visit(value: Any) -> Any:
            if isinstance(value, dict):
                return {str(key): visit(item) for key, item in value.items()}
            if isinstance(value, list):
                return [visit(item) for item in value]
            if isinstance(value, tuple):
                return [visit(item) for item in value]
            if isinstance(value, str):
                return safe_message(value)
            return value

        return visit(payload)

    hub = EventHub(catalog, public_jobs, sanitize=sanitize_event_payload)
    log_store = RawLogStore(
        settings.data_root.parent / "logs",
        redacted_roots=(settings.data_root, settings.model_root, Path.cwd()),
    )
    operation_by_job: dict[str, str] = {}
    operation_lock = threading.RLock()
    auto_start_operations: dict[str, bool] = {}
    scheduled_operations: set[str] = set()
    source_size_cache: dict[str, tuple[int, int]] = {}
    telemetry: SystemTelemetry | None = None

    def publish(kind: str, payload: dict[str, Any], job_id: str | None = None) -> None:
        try:
            # SystemTelemetry already persists GPU samples at a bounded rate.
            # Publishing the SSE event must not write the same sample twice.
            if kind != "gpu_metrics":
                log_store.write(
                    component="model" if kind.startswith("model") else "task",
                    event=kind,
                    message=str(payload.get("message") or kind),
                    error_code=(
                        str(payload["error_code"])
                        if payload.get("error_code") is not None
                        else None
                    ),
                    job_id=job_id,
                    page_index=(
                        int(payload["page_index"])
                        if payload.get("page_index") is not None
                        else None
                    ),
                    unit_index=(
                        int(payload["unit_index"])
                        if payload.get("unit_index") is not None
                        else None
                    ),
                    metrics={
                        key: value
                        for key, value in payload.items()
                        if key not in {"message", "page_index", "unit_index"}
                    },
                    kind="raw",
                )
        except Exception:
            logger.exception("unable to write raw log event=%s", kind)
        if telemetry is not None and kind in {"job_status", "model_progress"}:
            status = str(payload.get("status") or payload.get("state") or "")
            telemetry.set_busy(
                status in {JobStatus.RUNNING.value, "generating", "loading", "downloading"}
            )
        if job_id and kind in {"job_status", "job_queued"} and catalog.job(job_id):
            values: dict[str, Any] = {}
            if payload.get("status"):
                values["status"] = payload["status"]
                if payload["status"] != JobStatus.QUEUED.value:
                    values["queue_position"] = None
            if "queue_position" in payload:
                values["queue_position"] = payload["queue_position"]
            if values:
                catalog.update_job(job_id, **values)
        if job_id and kind in {"ingest_progress", "job_ready", "job_status", "job_error"}:
            with operation_lock:
                operation_id = operation_by_job.get(job_id)
            if operation_id:
                try:
                    if kind == "ingest_progress":
                        catalog.update_operation(
                            operation_id,
                            status="running",
                            stage=str(payload.get("stage") or "ingesting"),
                            progress=payload,
                        )
                    elif kind == "job_ready":
                        # For auto-start imports, ``job_ready`` only means the
                        # ingest phase is done.  Keep the operation running
                        # until the job has actually been handed to the GPU
                        # queue; otherwise a fast worker can make the client
                        # observe a completed operation while the catalog is
                        # still in the transient ``ready`` state.
                        with operation_lock:
                            requested_auto_start = auto_start_operations.get(operation_id, True)
                        catalog.update_operation(
                            operation_id,
                            status="running" if requested_auto_start else "completed",
                            stage="ready",
                            progress=payload,
                        )
                    elif kind == "job_status":
                        status = str(payload.get("status") or "")
                        if status in {
                            JobStatus.CREATED.value,
                            JobStatus.INGESTING.value,
                            JobStatus.READY.value,
                        }:
                            catalog.update_operation(
                                operation_id,
                                status="running",
                                stage=status,
                                progress=payload,
                            )
                        elif status in {
                            JobStatus.PAUSED.value,
                            JobStatus.CANCELLED.value,
                        }:
                            catalog.update_operation(
                                operation_id,
                                status=status,
                                stage=status,
                                progress=payload,
                            )
                        elif status == JobStatus.FAILED.value:
                            catalog.update_operation(
                                operation_id,
                                status="failed",
                                stage="failed",
                                error=str(
                                    payload.get("error")
                                    or payload.get("message")
                                    or "任务失败"
                                ),
                                progress=payload,
                            )
                    elif kind == "job_error":
                        catalog.update_operation(
                            operation_id,
                            status="failed",
                            stage="failed",
                            error=str(payload.get("message") or "建书失败"),
                            progress=payload,
                        )
                except KeyError:
                    pass
        hub.publish(kind, payload, job_id)

    manager.set_event_callback(publish)
    scheduler = JobQueue(manager.process, publish)

    def restore_queued_jobs() -> None:
        """Rebuild the in-memory GPU queue after a local app restart.

        The manifest remains the source of truth, while ``JobQueue`` is
        intentionally process-local.  Restore only jobs that were explicitly
        queued, leaving paused, running-recovered, archived and waiting-model
        jobs untouched.
        """
        for job in manager.list_jobs():
            if job["status"] != JobStatus.QUEUED.value:
                continue
            job_id = str(job["id"])
            catalog_job = catalog.job(job_id)
            if catalog_job and catalog_job.get("archived_at"):
                continue
            catalog.update_job(job_id, status=JobStatus.QUEUED.value, queue_position=None)
            position = scheduler.enqueue(job_id)
            logger.info("job=%s restored to GPU queue at position=%s", job_id, position)

    restore_queued_jobs()
    ingest_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="paneltone-ingest")
    telemetry = SystemTelemetry(settings, log_store, event_callback=publish)
    model_download_lock = threading.RLock()
    model_downloads: set[str] = set()
    model_health_signatures: dict[str, tuple[object, object, object]] = {}
    recovery_stop = threading.Event()

    def recover_waiting_jobs() -> None:
        while not recovery_stop.wait(5):
            try:
                engine_health = registry.health()
                for engine_id, state in engine_health.items():
                    signature = (
                        state.get("state"),
                        state.get("loaded"),
                        state.get("active_requests"),
                    )
                    previous = model_health_signatures.get(engine_id)
                    model_health_signatures[engine_id] = signature
                    if previous is None or previous == signature:
                        continue
                    auto_released = (
                        previous[1] is True
                        and state.get("loaded") is False
                        and state.get("state") == "idle"
                    )
                    publish(
                        "model_progress",
                        {
                            "model_id": state.get("model_id") or engine_id,
                            "status": "auto_released" if auto_released else state.get("state"),
                            "state": state.get("state"),
                            "loaded": state.get("loaded"),
                            "active_requests": state.get("active_requests"),
                            "message": (
                                "模型闲置后已自动释放显存"
                                if auto_released
                                else "模型服务状态已更新"
                            ),
                        },
                    )
                for job in manager.list_jobs():
                    if job["status"] != JobStatus.WAITING_MODEL.value:
                        continue
                    catalog_job = catalog.job(job["id"])
                    if catalog_job and catalog_job.get("archived_at"):
                        continue
                    engine_id = str(job.get("spec", {}).get("engine", ""))
                    if not engine_health.get(engine_id, {}).get("ok", False):
                        continue
                    manager.queue(job["id"])
                    position = scheduler.enqueue(job["id"])
                    hub.publish(
                        "model_reconnected",
                        {"status": JobStatus.QUEUED.value, "queue_position": position},
                        job["id"],
                    )
                    logger.info("job=%s requeued after engine=%s recovered", job["id"], engine_id)
            except Exception:
                logger.exception("waiting-model recovery cycle failed")
                continue

    recovery_thread = threading.Thread(
        target=recover_waiting_jobs, name="paneltone-model-recovery", daemon=True
    )
    recovery_thread.start()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        recovery_stop.set()
        if recovery_thread.is_alive():
            recovery_thread.join(timeout=2.0)
        telemetry.stop()
        ingest_executor.shutdown(wait=False, cancel_futures=False)
        scheduler.stop()

    app = FastAPI(title="PanelTone", version=__version__, lifespan=lifespan)

    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        started = datetime.now(UTC)
        try:
            response = await call_next(request)
        except Exception as exc:
            log_store.write(
                level="ERROR",
                component="api",
                event="request_error",
                message=str(exc),
                error_code=type(exc).__name__,
                metrics={"method": request.method, "path": request.url.path},
            )
            raise
        elapsed_ms = round((datetime.now(UTC) - started).total_seconds() * 1000, 1)
        if not (
            request.method == "GET"
            and request.url.path.startswith("/api/assets/")
            and response.status_code < 400
        ):
            log_store.write(
                component="api",
                event="request",
                message=f"{request.method} {request.url.path}",
                metrics={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "elapsed_ms": elapsed_ms,
                },
            )
        if request.url.path.startswith("/api/") and request.method == "GET":
            response.headers.setdefault("Cache-Control", "no-store, max-age=0")
            response.headers.setdefault("Pragma", "no-cache")
        return response

    app.state.manager = manager
    app.state.catalog = catalog
    app.state.events = hub
    app.state.scheduler = scheduler
    app.state.log_store = log_store
    app.state.telemetry = telemetry

    static_root = Path(__file__).with_name("web") / "static"
    app.mount("/static", StaticFiles(directory=static_root), name="static")

    @app.get("/")
    def home() -> FileResponse:
        return FileResponse(static_root / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {"ok": True, "engines": registry.health(), "queued_jobs": len(scheduler.positions())}

    def event_message(item: dict[str, Any]) -> str:
        payload = item["payload"]
        kind = item["kind"]
        page = payload.get("page_index")
        page_text = f"第 {int(page) + 1} 页" if isinstance(page, int) else "页面"
        status_messages = {
            "created": "正在建立任务",
            "ingesting": "正在展开并检查页面",
            "ready": "任务已经准备好",
            "queued": "任务已进入处理队列",
            "running": "模型已开始处理",
            "paused": "任务已暂停",
            "waiting_model": "模型服务未连接，任务已安全等待",
            "needs_attention": "部分页面需要检查",
            "completed": "整本漫画已经处理完成",
            "failed": "任务处理失败",
            "cancelled": "任务已取消",
            "archived": "任务已移到回收站",
        }
        if kind == "job_status":
            return status_messages.get(str(payload.get("status")), "任务状态已更新")
        if kind == "job_queued":
            return "任务已进入处理队列"
        if kind == "unit_started":
            return f"{page_text}开始生成"
        if kind == "unit_finished":
            return f"{page_text}已经生成并通过细节检查"
        if kind == "page_ready":
            return f"{page_text}输出已可预览"
        if kind == "page_preview_ready":
            return f"{page_text}已有临时预览，等待细节检查"
        if kind == "job_progress":
            return "处理进度已更新"
        if kind == "ingest_started":
            return str(payload.get("message") or "已提交，正在展开漫画来源")
        if kind == "ingest_progress":
            return str(payload.get("message") or "正在建立页面")
        if kind == "job_error":
            return f"{page_text}处理失败，请查看任务状态后重试"
        if kind == "gpu_metrics":
            return "GPU 和系统指标已更新"
        if kind == "model_progress":
            model_states = {
                "downloading": "正在下载模型",
                "downloaded": "模型下载完成",
                "failed": "模型下载失败",
                "auto_released": "模型闲置后已自动释放显存",
            }
            return model_states.get(str(payload.get("status")), "模型状态已更新")
        if kind == "model_reconnected":
            return "模型服务已恢复，任务已自动继续"
        if kind == "upload_progress":
            return "来源文件上传完成" if payload.get("completed") else "正在上传来源文件"
        return "后台活动已更新"

    @app.get("/api/logs")
    def activity_logs(
        job_id: str | None = None,
        limit: int = 80,
        kind: Literal["activity", "raw", "gpu"] = "activity",
        after: int | None = None,
        level: str | None = None,
        component: str | None = None,
    ) -> list[dict[str, Any]]:
        if kind in {"raw", "gpu"}:
            return [
                RawLogEntry.model_validate(item).model_dump(exclude_none=True)
                for item in log_store.read(
                    kind=kind,
                    job_id=None if kind == "gpu" else job_id,
                    level=level,
                    component=component,
                    limit=limit,
                    after=after,
                )
            ]
        events = catalog.recent_events(job_id=job_id, limit=limit)
        return [
            {
                "id": item["id"],
                "job_id": item["job_id"],
                "kind": item["kind"],
                "created_at": item["created_at"],
                "message": event_message(item),
            }
            for item in events
            if item["kind"] != "snapshot"
        ]

    @app.get("/api/logs/download")
    def download_logs(
        kind: Literal["raw", "gpu"] = "raw", job_id: str | None = None
    ) -> Response:
        content = log_store.export(kind=kind, job_id=job_id)
        filename = f"paneltone-{kind}-logs.jsonl"
        if job_id:
            filename = f"paneltone-{kind}-{job_id[:8]}-logs.jsonl"
        return Response(
            content=content,
            media_type="application/x-ndjson",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-PanelTone-Log-Warning": "可能包含本机路径，请确认后再分享",
            },
        )

    @app.get("/api/gpu", response_model=GpuMetrics)
    def gpu_metrics() -> dict[str, Any]:
        return telemetry.current()

    @app.get("/api/presets")
    def presets() -> PresetCatalog:
        result: dict[str, Any] = presets_payload()
        result.update(_option_payload())
        return PresetCatalog.model_validate(result)

    @app.post("/api/import", response_model=None, status_code=201)
    def import_local_file(
        file: Annotated[UploadFile, File()],
        client_upload_id: Annotated[str | None, Form()] = None,
        upload_id_form: Annotated[str | None, Form(alias="upload_id")] = None,
        relative_path: Annotated[str | None, Form()] = None,
        upload_id_header: Annotated[str | None, Header(alias="X-Upload-ID")] = None,
        upload_offset: Annotated[int | None, Header(alias="X-Upload-Offset")] = None,
        upload_total: Annotated[int | None, Header(alias="X-Upload-Total")] = None,
    ) -> Response | SourceDescriptor:
        supplied_ids = [
            value.strip()
            for value in (upload_id_header, upload_id_form, client_upload_id)
            if value and value.strip()
        ]
        if supplied_ids and len(set(supplied_ids)) != 1:
            raise HTTPException(status_code=400, detail="上传编号不一致")
        supplied_upload_id = supplied_ids[0] if supplied_ids else ""
        if supplied_upload_id and not re.fullmatch(r"[A-Za-z0-9_-]{8,96}", supplied_upload_id):
            raise HTTPException(status_code=400, detail="上传编号格式无效")
        upload_id = supplied_upload_id or uuid.uuid4().hex
        original_name = Path(file.filename or "upload.bin").name
        suffix = Path(original_name).suffix.casefold()
        if suffix not in ALLOWED_UPLOADS:
            raise HTTPException(status_code=400, detail="不支持这个文件格式")
        try:
            normalized_relative_path = normalize_relative_path(relative_path, original_name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        incoming = settings.data_root / "imports" / "incoming"
        incoming.mkdir(parents=True, exist_ok=True)
        if upload_offset is not None and upload_offset < 0:
            raise HTTPException(status_code=400, detail="上传偏移不能为负数")
        if upload_total is not None and upload_total < 0:
            raise HTTPException(status_code=400, detail="上传总字节数不能为负数")
        existing_upload = catalog.upload(upload_id)
        if existing_upload:
            previous_name = str(existing_upload.get("original_name") or "")
            previous_relative = existing_upload.get("relative_path")
            if previous_name and previous_name != original_name:
                raise HTTPException(status_code=409, detail="同一上传编号不能更换文件名")
            if previous_relative and previous_relative != normalized_relative_path:
                raise HTTPException(status_code=409, detail="同一上传编号不能更换相对路径")
        if existing_upload and existing_upload.get("status") == "completed":
            source_id = existing_upload.get("source_id")
            if source_id:
                source = catalog.source(str(source_id))
                return SourceDescriptor(
                    source_id=source["id"],
                    name=source["original_name"],
                    kind=source["kind"],
                    size=int(source["size"]),
                    sha256=source["sha256"],
                    duplicate=False,
                    upload_id=upload_id,
                    relative_path=source.get("relative_path") or normalized_relative_path,
                    uploaded_bytes=int(source["size"]),
                    total_bytes=int(source["size"]),
                    complete=True,
                    resume_url=f"/api/import/{upload_id}",
                    import_batch_id=source.get("import_batch_id"),
                )
        target = (
            Path(str(existing_upload["temp_path"]))
            if existing_upload and existing_upload.get("temp_path")
            else incoming / f"{upload_id}.part"
        )
        current_size = target.stat().st_size if target.is_file() else 0
        requested_offset = current_size if upload_offset is None else max(0, upload_offset)
        if requested_offset != current_size:
            return JSONResponse(
                status_code=409,
                content={
                    "detail": "上传偏移不匹配",
                    "upload_id": upload_id,
                    "uploaded_bytes": current_size,
                    "expected_offset": current_size,
                },
            )
        declared_total = upload_total is not None
        expected_size = upload_total if declared_total else int(file.size or 0)
        max_size = settings.max_upload_mib * 1024 * 1024
        if expected_size < 0 or expected_size > max_size:
            raise HTTPException(status_code=413, detail="文件超过本机任务允许的大小")
        catalog.save_upload(
            upload_id,
            original_name=original_name,
            relative_path=normalized_relative_path,
            temp_path=target,
            total_bytes=expected_size or None,
            uploaded_bytes=current_size,
        )
        size = current_size
        last_reported = 0
        try:
            with target.open("ab") as destination:
                while chunk := file.file.read(1024 * 1024):
                    if size + len(chunk) > max_size:
                        raise HTTPException(status_code=413, detail="文件超过本机任务允许的大小")
                    if declared_total and size + len(chunk) > expected_size:
                        raise HTTPException(status_code=400, detail="上传数据超过声明的总字节数")
                    written = destination.write(chunk)
                    if written != len(chunk):
                        raise OSError("上传写入不完整")
                    size += written
                    if size - last_reported >= 8 * 1024 * 1024:
                        hub.publish(
                            "upload_progress",
                            {
                                "upload_id": upload_id,
                                "name": original_name,
                                "uploaded_bytes": size,
                                "total_bytes": expected_size,
                            },
                        )
                        last_reported = size
            catalog.save_upload(
                upload_id,
                original_name=original_name,
                relative_path=normalized_relative_path,
                temp_path=target,
                total_bytes=expected_size or None,
                uploaded_bytes=size,
            )
        except Exception:
            # Keep the partial file and its offset so the browser can retry
            # the same upload without sending already accepted bytes again.
            try:
                size = target.stat().st_size
            except OSError:
                size = current_size
            catalog.save_upload(
                upload_id,
                original_name=original_name,
                relative_path=normalized_relative_path,
                temp_path=target,
                total_bytes=expected_size or None,
                uploaded_bytes=size,
            )
            raise
        if declared_total and size > expected_size:
            raise HTTPException(status_code=400, detail="上传数据超过声明的总字节数")
        complete = (
            size >= expected_size
            if declared_total
            else expected_size <= 0 or size >= expected_size
        )
        if not complete:
            return JSONResponse(
                status_code=202,
                content={
                    "upload_id": upload_id,
                    "uploaded_bytes": size,
                    "total_bytes": expected_size,
                    "complete": False,
                    "resume_url": f"/api/import/{upload_id}",
                },
            )
        # The resumable writer uses a ``.part`` suffix while the upload is
        # incomplete.  Move the completed bytes to a file carrying the
        # original extension before handing them to the catalog; otherwise a
        # directory import would later see a ``.part`` file and reject the
        # whole batch as containing no supported images.
        complete_target = target
        if target.suffix.casefold() != suffix:
            complete_target = target.with_suffix(suffix)
            target.replace(complete_target)
        source = catalog.add_source(
            original_name,
            complete_target,
            ALLOWED_UPLOADS[suffix],
            relative_path=normalized_relative_path,
            import_batch_id=upload_id,
        )
        catalog.save_upload(
            upload_id,
            original_name=original_name,
            relative_path=normalized_relative_path,
            temp_path=complete_target,
            total_bytes=expected_size or size,
            uploaded_bytes=size,
            status="completed",
            source_id=str(source["id"]),
        )
        hub.publish(
            "upload_progress",
            {
                "upload_id": upload_id,
                "name": original_name,
                "uploaded_bytes": size,
                "total_bytes": expected_size or size,
                "completed": True,
            },
        )
        return SourceDescriptor(
            source_id=source["id"],
            name=source["original_name"],
            kind=source["kind"],
            size=source["size"],
            sha256=source["sha256"],
            duplicate=source["duplicate"],
            upload_id=upload_id,
            relative_path=source.get("relative_path") or normalized_relative_path,
            uploaded_bytes=size,
            total_bytes=expected_size or size,
            complete=True,
            resume_url=f"/api/import/{upload_id}",
            import_batch_id=source.get("import_batch_id"),
        )

    @app.get("/api/import/{upload_id}")
    def import_status(upload_id: str) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9_-]{8,96}", upload_id):
            raise HTTPException(status_code=404, detail="找不到这个上传任务")
        upload = catalog.upload(upload_id)
        if upload is None:
            raise HTTPException(status_code=404, detail="找不到这个上传任务")
        source = None
        if upload.get("source_id"):
            try:
                source = catalog.source(str(upload["source_id"]))
            except KeyError:
                # A stale upload record must not turn a resumable status poll
                # into a 500 after a user-side cleanup. Keep the cursor so the
                # client can decide whether to retry the upload.
                source = None
        return {
            "upload_id": upload_id,
            "name": upload["original_name"],
            "relative_path": upload.get("relative_path"),
            "uploaded_bytes": int(upload.get("uploaded_bytes") or 0),
            "total_bytes": upload.get("total_bytes"),
            "status": upload["status"],
            "source_id": upload.get("source_id"),
            "sha256": source.get("sha256") if source else None,
            "duplicate": False,
            "import_batch_id": source.get("import_batch_id") if source else upload_id,
            "complete": upload["status"] == "completed" and source is not None,
            "resume_url": f"/api/import/{upload_id}",
        }

    def build_spec(options: JobOptions, source: Path, display_name: str) -> JobSpec:
        references = [Path(catalog.source(item)["path"]) for item in options.style_reference_ids]
        return JobSpec(
            source=source,
            workspace=settings.data_root,
            mode=options.mode,
            engine=options.engine,
            protection=options.protection,
            detail_mode=options.detail_mode,
            output_format=options.output_format,
            panel_mode=options.panel_mode,
            seed=options.seed,
            prompt=options.prompt,
            negative_prompt=options.negative_prompt,
            color_preset=options.color_preset,
            style_preset=options.style_preset,
            preserve_text=options.preserve_text,
            preserve_ink=options.preserve_ink,
            ink_gamma=options.ink_gamma,
            chroma_strength=options.chroma_strength,
            style_references=references,
            max_retries=options.max_retries,
            adult_fictional_content=options.adult_fictional_content,
            display_name=display_name,
        )

    def create_one(options: JobOptions, source: Path, display_name: str) -> str:
        job_id = manager.create(build_spec(options, source, display_name))
        catalog.upsert_job(job_id, display_name, JobStatus.READY.value)
        hub.publish("job_status", {"status": JobStatus.READY.value}, job_id)
        return job_id

    def source_size(source: Path) -> int:
        if source.is_file():
            return source.stat().st_size
        if source.is_dir():
            return sum(path.stat().st_size for path in source.rglob("*") if path.is_file())
        return 0

    def schedule_async_ingest(
        job_id: str, operation_id: str, *, auto_start: bool = True
    ) -> None:
        # ``/start`` is intentionally idempotent.  Multiple clicks while the
        # first request is still being accepted must not create two ingest
        # workers that race the manifest or enqueue the same job twice after
        # the first worker has already popped it from the GPU queue.
        with operation_lock:
            if operation_id in scheduled_operations:
                return
            scheduled_operations.add(operation_id)

        def run() -> None:
            try:
                catalog.update_operation(operation_id, status="running", stage="ingesting")
                manager.ingest(job_id)
                summary = manager.status(job_id)
                with operation_lock:
                    requested_auto_start = auto_start_operations.get(operation_id, auto_start)
                summary_status = str(summary["status"])
                if summary_status == JobStatus.READY.value and requested_auto_start:
                    # Keep the operation non-terminal until enqueue has
                    # completed. Otherwise a poller can observe ``completed``
                    # while the job is still briefly in ``ready``.
                    operation_status = "running"
                elif summary_status == JobStatus.FAILED.value:
                    operation_status = "failed"
                elif summary_status == JobStatus.CANCELLED.value:
                    operation_status = "cancelled"
                elif summary_status == JobStatus.PAUSED.value:
                    operation_status = "paused"
                elif summary_status == JobStatus.READY.value:
                    operation_status = "completed"
                else:
                    operation_status = "running"
                catalog.update_operation(
                    operation_id,
                    status=operation_status,
                    stage=summary_status,
                    progress=summary["progress"],
                )
                if not requested_auto_start:
                    return
                if summary_status != JobStatus.READY.value:
                    return
                manager.queue(job_id)
                # ``JobQueue.enqueue`` emits the authoritative queue event.
                # Do not write a second queued status after enqueue: a very
                # small job can already finish on the worker and a late
                # catalog write would regress ``completed`` back to ``queued``.
                position = scheduler.enqueue(job_id)
                queued_summary = manager.status(job_id)
                queued_job_status = str(queued_summary.get("status") or "queued")
                queued_progress = dict(queued_summary.get("progress") or {})
                # The import operation itself is complete, but its reported
                # stage must follow the job snapshot captured after enqueue.
                # This prevents a fast job from being displayed as queued
                # after its worker has already produced the final result.
                if queued_job_status in {
                    JobStatus.RUNNING.value,
                    JobStatus.QUEUED.value,
                }:
                    queued_stage = "queued"
                    queued_progress.update(
                        {
                            "stage": "queued",
                            "stage_percent": 100.0,
                            "queue_position": position,
                            "latest_message": "已进入处理队列",
                        }
                    )
                else:
                    queued_stage = queued_job_status
                catalog.update_operation(
                    operation_id,
                    status="completed",
                    stage=queued_stage,
                    progress=queued_progress,
                )
            except Exception as exc:
                try:
                    catalog.update_operation(
                        operation_id,
                        status="failed",
                        stage="failed",
                        error=str(exc),
                    )
                except KeyError:
                    logger.exception("async ingest operation disappeared job=%s", job_id)
                logger.exception("async ingest failed job=%s", job_id)
            finally:
                with operation_lock:
                    scheduled_operations.discard(operation_id)

        ingest_executor.submit(run)

    def create_async_one(
        options: JobOptions, source: Path, display_name: str, *, auto_start: bool = True
    ) -> tuple[str, str]:
        job_id = manager.create_shell(build_spec(options, source, display_name))
        operation_id = uuid.uuid4().hex
        catalog.upsert_job(job_id, display_name, JobStatus.INGESTING.value)
        catalog.create_operation(operation_id, job_id, stage="accepted")
        with operation_lock:
            operation_by_job[job_id] = operation_id
            auto_start_operations[operation_id] = auto_start
        schedule_async_ingest(job_id, operation_id, auto_start=auto_start)
        return job_id, operation_id

    # Reattach orphaned ingest operations after an application restart. The
    # manifest is the source of truth, so an already indexed page is reused.
    for recovering_job_id in manager.ingesting_jobs():
        existing_operation = catalog.latest_operation_for_job(recovering_job_id)
        recovering_operation_id = (
            existing_operation["id"] if existing_operation else uuid.uuid4().hex
        )
        if existing_operation is None:
            catalog.create_operation(recovering_operation_id, recovering_job_id, stage="recovered")
        with operation_lock:
            operation_by_job[recovering_job_id] = recovering_operation_id
            auto_start_operations[recovering_operation_id] = True
        catalog.upsert_job(
            recovering_job_id,
            manager.status(recovering_job_id)["spec"].get("display_name", "未命名漫画"),
            JobStatus.INGESTING.value,
        )
        schedule_async_ingest(recovering_job_id, recovering_operation_id)

    @app.get("/api/jobs")
    def list_jobs(include_archived: bool = False) -> list[dict[str, object]]:
        return public_jobs(include_archived)

    @app.get("/api/library/tree")
    def library_tree(
        include_archived: bool = False,
        q: str = Query(default="", max_length=120),
    ) -> dict[str, Any]:
        """Return a nested virtual library without changing queue order."""
        all_jobs = public_jobs(include_archived)
        jobs_by_folder: dict[str | None, list[dict[str, Any]]] = {}
        query = q.strip().casefold()
        for job in all_jobs:
            folder_id = job.get("folder_id")
            if query and query not in str(job.get("display_name", "")).casefold():
                continue
            jobs_by_folder.setdefault(folder_id, []).append(job)
        folder_rows = catalog.folders(include_archived=include_archived)
        nodes: dict[str, FolderNode] = {}
        for row in folder_rows:
            nodes[str(row["id"])] = FolderNode(
                id=str(row["id"]),
                parent_id=row.get("parent_id"),
                name=str(row["name"]),
                sort_order=int(row.get("sort_order") or 0),
                archived_at=row.get("archived_at"),
                job_count=len(jobs_by_folder.get(str(row["id"]), [])),
                jobs=jobs_by_folder.get(str(row["id"]), []),
            )
        roots: list[FolderNode] = []
        for node in nodes.values():
            parent = nodes.get(str(node.parent_id)) if node.parent_id else None
            if parent is None:
                roots.append(node)
            else:
                parent.children.append(node)
        def sort_nodes(items: list[FolderNode]) -> None:
            items.sort(key=lambda item: (item.sort_order, item.name.casefold()))
            for item in items:
                sort_nodes(item.children)
                item.jobs.sort(key=lambda job: int(job.get("library_order") or 0))
        sort_nodes(roots)
        root_jobs = jobs_by_folder.get(None, [])
        root_jobs.sort(
            key=lambda job: (
                int(job.get("library_order") or 0),
                str(job.get("display_name") or "").casefold(),
            )
        )
        if query:
            def keep(node: FolderNode) -> bool:
                own = query in node.name.casefold() or bool(node.jobs)
                node.children[:] = [child for child in node.children if keep(child)]
                return own or bool(node.children)
            roots = [node for node in roots if keep(node)]
        return {
            "folders": [node.model_dump() for node in roots],
            "root_jobs": root_jobs,
        }

    @app.get("/api/library/search")
    def library_search(q: str = Query(min_length=1, max_length=120)) -> dict[str, Any]:
        return library_tree(include_archived=False, q=q)

    @app.post("/api/folders", status_code=201)
    def create_folder(request: FolderRequest) -> dict[str, Any]:
        try:
            folder = catalog.create_folder(request.name, request.parent_id)
            hub.publish("folder_changed", {"folder_id": folder["id"], "action": "created"})
            return folder
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.patch("/api/folders/{folder_id}")
    def patch_folder(folder_id: str, request: FolderPatchRequest) -> dict[str, Any]:
        try:
            folder = catalog.update_folder(
                folder_id,
                name=request.name,
                parent_id=request.parent_id if "parent_id" in request.model_fields_set else ...,
            )
            hub.publish("folder_changed", {"folder_id": folder_id, "action": "updated"})
            return folder
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/folders/{folder_id}/archive")
    def archive_folder(folder_id: str) -> dict[str, Any]:
        try:
            folder = catalog.archive_folder(folder_id)
            hub.publish("folder_changed", {"folder_id": folder_id, "action": "archived"})
            return folder
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/folders/{folder_id}/restore")
    def restore_folder(folder_id: str) -> dict[str, Any]:
        try:
            folder = catalog.restore_folder(folder_id)
            hub.publish("folder_changed", {"folder_id": folder_id, "action": "restored"})
            return folder
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.delete("/api/folders/{folder_id}")
    def delete_folder(folder_id: str, request: DeleteJobRequest) -> dict[str, str]:
        try:
            catalog.delete_folder(folder_id, confirmation=request.confirmation)
            hub.publish("folder_changed", {"folder_id": folder_id, "action": "deleted"})
            return {"folder_id": folder_id, "status": "deleted"}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/library/jobs/{job_id}/move")
    def move_library_job(job_id: str, request: LibraryMoveRequest) -> dict[str, Any]:
        try:
            job = catalog.move_job_library(job_id, request.folder_id, request.before_job_id)
            hub.publish(
                "library_changed",
                {
                    "job_id": job_id,
                    "folder_id": request.folder_id,
                    "before_job_id": request.before_job_id,
                },
                job_id,
            )
            return job
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/library/reorder")
    def reorder_library(request: LibraryReorderRequest) -> dict[str, str]:
        try:
            if request.folder_ids:
                # For folder drag/drop, ``parent_id`` is explicit. Keep
                # accepting ``folder_id`` as the parent alias for clients
                # built against the first directory-tree prototype.
                parent_id = (
                    request.parent_id
                    if "parent_id" in request.model_fields_set
                    else request.folder_id
                )
                catalog.reorder_folders(parent_id, request.folder_ids)
            elif request.job_ids:
                catalog.reorder_library(request.folder_id, request.job_ids)
            else:
                raise ValueError("目录或任务排序列表不能为空")
            hub.publish(
                "library_changed",
                {
                    "folder_id": request.folder_id,
                    "parent_id": request.parent_id,
                    "job_ids": request.job_ids,
                    "folder_ids": request.folder_ids,
                },
            )
            return {"status": "reordered"}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/jobs", response_model=None)
    def create_job(
        request: CreateJobRequest,
        async_ingest: bool = Query(False, alias="async"),
        auto_start: bool = True,
    ) -> Response | dict[str, str]:
        try:
            if request.source_id:
                source = Path(catalog.source(request.source_id)["path"])
            elif request.source:
                source = Path(request.source)
            else:
                raise ValueError("缺少漫画来源")
            # Every import receives a task shell immediately.  The async query
            # remains accepted for old clients, but no longer changes the
            # response contract or blocks the request on book construction.
            job_id, operation_id = create_async_one(
                request, source, request.display_name, auto_start=auto_start
            )
            return JSONResponse(
                status_code=202,
                content={
                    "operation_id": operation_id,
                    "job_id": job_id,
                    "display_name": request.display_name,
                    "status": JobStatus.INGESTING.value,
                    "auto_start": auto_start,
                    "progress_url": f"/api/operations/{operation_id}",
                },
            )
        except (ValueError, KeyError, PermissionError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/jobs/batch", status_code=202, response_model=None)
    def create_batch(
        request: BatchJobRequest,
        async_ingest: bool = Query(False, alias="async"),
        auto_start: bool = True,
    ) -> Response | dict[str, list[dict[str, Any]]]:
        try:
            sources = [catalog.source(source_id) for source_id in request.source_ids]
            image_sources = [item for item in sources if item["kind"] == "image"]
            book_sources = [item for item in sources if item["kind"] == "book"]
            created: list[dict[str, Any]] = []
            operation_ids: list[str] = []
            # Batch imports use the same non-blocking task-shell contract as
            # single imports, including small files, so the UI can return to
            # the processing page without guessing how long ingest will take.
            if image_sources:
                natural_sources = sorted(
                    image_sources,
                    key=lambda item: [
                        int(part) if part.isdigit() else part.casefold()
                        for part in re.split(r"(\d+)", item["original_name"])
                    ],
                )
                order = request.image_order or [item["id"] for item in natural_sources]
                if len(order) != len(image_sources) or set(order) != {
                    item["id"] for item in image_sources
                }:
                    raise ValueError("图片顺序必须包含全部已选择图片")
                sources_by_id = {str(item["id"]): item for item in image_sources}
                grouped: dict[str, list[str]] = {}
                for source_id in order:
                    source = sources_by_id[str(source_id)]
                    relative = str(source.get("relative_path") or source["original_name"])
                    parent = PurePosixPath(relative).parent
                    folder_key = "" if str(parent) in {"", "."} else str(parent)
                    grouped.setdefault(folder_key, []).append(str(source_id))
                for folder_key, group_order in sorted(grouped.items(), key=lambda item: item[0]):
                    folder_parts = [part for part in folder_key.split("/") if part]
                    folder_id = catalog.ensure_folder_path(folder_parts)
                    display_name = (
                        folder_parts[-1] if folder_parts else request.image_book_name
                    )
                    group = catalog.group_images(group_order, display_name)
                    job_id, operation_id = create_async_one(
                        request, group, display_name, auto_start=auto_start
                    )
                    if folder_id is not None:
                        catalog.update_job(job_id, folder_id=folder_id)
                    operation_ids.append(operation_id)
                    created.append(
                        {
                            "job_id": job_id,
                            "display_name": display_name,
                            "folder_id": folder_id,
                        }
                    )
            for source in book_sources:
                name = Path(source["original_name"]).stem
                job_id, operation_id = create_async_one(
                    request, Path(source["path"]), name, auto_start=auto_start
                )
                operation_ids.append(operation_id)
                created.append({"job_id": job_id, "display_name": name, "folder_id": None})
            return JSONResponse(
                status_code=202,
                content={
                    "operation_id": operation_ids[0] if operation_ids else None,
                    "operation_ids": operation_ids,
                    "jobs": [
                        {
                            **item,
                            "status": JobStatus.INGESTING.value,
                            "operation_id": operation_id,
                            "progress_url": f"/api/operations/{operation_id}",
                        }
                        for item, operation_id in zip(created, operation_ids, strict=False)
                    ],
                    "status": JobStatus.INGESTING.value,
                    "auto_start": auto_start,
                },
            )
        except (ValueError, KeyError, PermissionError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/operations/{operation_id}", response_model=ImportOperation)
    def operation_status(operation_id: str) -> dict[str, Any]:
        operation = catalog.operation(operation_id)
        if operation is None:
            raise HTTPException(status_code=404, detail="找不到这个导入操作")
        progress = dict(operation.get("progress") or {})
        try:
            job_progress = manager.status(operation["job_id"])["progress"]
        except (KeyError, ValueError):
            job_progress = {}
        return {
            "operation_id": operation_id,
            "job_id": operation["job_id"],
            "status": operation["status"],
            "stage": operation["stage"],
            "stage_percent": float(
                progress.get("stage_percent", job_progress.get("stage_percent", 0.0)) or 0.0
            ),
            "discovered_pages": int(
                progress.get("discovered_pages", job_progress.get("total_pages", 0)) or 0
            ),
            "total_pages": job_progress.get("total_pages") or progress.get("total_pages"),
            "bytes_processed": int(
                progress.get("bytes_processed", job_progress.get("bytes_processed", 0)) or 0
            ),
            "bytes_total": int(
                progress.get("bytes_total", job_progress.get("bytes_total", 0)) or 0
            ),
            "current_file": progress.get("current_file") or job_progress.get("current_file"),
            "latest_message": progress.get("message")
            or progress.get("latest_message")
            or job_progress.get("latest_message"),
            "error": safe_message(operation.get("error")),
        }

    @app.get("/api/events")
    def events(
        request: Request, last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None
    ) -> StreamingResponse:
        raw_cursor = last_event_id or request.query_params.get("after", "0") or "0"
        try:
            cursor = max(0, int(raw_cursor))
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="无效的事件编号") from None
        return StreamingResponse(
            hub.stream(cursor),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/jobs/{job_id}/start", status_code=202)
    def start_job(job_id: str) -> dict[str, object]:
        try:
            summary = manager.status(job_id)
            if summary["status"] in {JobStatus.CREATED.value, JobStatus.INGESTING.value}:
                operation = catalog.latest_operation_for_job(job_id)
                if operation is None:
                    # Older task manifests may predate the operation table.
                    # Attach a new resumable operation instead of returning a
                    # successful no-op that leaves a ``created`` task stuck.
                    operation_id = uuid.uuid4().hex
                    catalog.create_operation(operation_id, job_id, stage="recovered")
                    operation = catalog.operation(operation_id)
                if operation is not None:
                    with operation_lock:
                        operation_by_job[job_id] = operation["id"]
                        auto_start_operations[operation["id"]] = True
                    if (
                        summary["status"] == JobStatus.CREATED.value
                        and operation.get("status") != "running"
                    ):
                        schedule_async_ingest(job_id, operation["id"], auto_start=True)
                    elif operation.get("status") in {"accepted", "failed", "cancelled", "paused"}:
                        catalog.update_operation(
                            operation["id"], status="accepted", stage="recovered", error=None
                        )
                        schedule_async_ingest(job_id, operation["id"], auto_start=True)
                return {
                    "job_id": job_id,
                    "status": summary["status"],
                    "auto_start": True,
                    "message": "建书完成后自动开始处理",
                }
            if summary["status"] in {
                JobStatus.RUNNING.value,
                JobStatus.QUEUED.value,
                JobStatus.COMPLETED.value,
            }:
                raise ValueError("这个任务当前不能再次开始")
            if (catalog.job(job_id) or {}).get("archived_at"):
                raise ValueError("请先从回收站恢复任务")
            manager.queue(job_id)
            position = scheduler.enqueue(job_id)
            return {"job_id": job_id, "status": JobStatus.QUEUED.value, "queue_position": position}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/jobs/{job_id}/pause", status_code=202)
    def pause_job(job_id: str) -> dict[str, Any]:
        try:
            summary = manager.status(job_id)
            if summary["status"] not in {
                JobStatus.QUEUED.value,
                JobStatus.RUNNING.value,
                JobStatus.WAITING_MODEL.value,
            }:
                raise ValueError("当前任务没有可暂停的运行请求")
            if summary["status"] == JobStatus.QUEUED.value:
                scheduler.remove(job_id)
            control = manager.pause(job_id)
            latest = manager.status(job_id)
            status = (
                JobStatus.PAUSED.value
                if latest["status"] == JobStatus.PAUSED.value
                else "pause_requested"
            )
            return {
                "job_id": job_id,
                "status": status,
                "control_state": control,
                "message": control.get("message") or "暂停中",
                "poll_url": f"/api/jobs/{job_id}/progress",
            }
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/jobs/{job_id}/cancel", status_code=202)
    def cancel_job(job_id: str) -> dict[str, Any]:
        try:
            summary = manager.status(job_id)
            if summary["status"] not in {
                JobStatus.QUEUED.value,
                JobStatus.RUNNING.value,
                JobStatus.WAITING_MODEL.value,
            }:
                raise ValueError("当前任务没有可取消的运行请求")
            if summary["status"] == JobStatus.QUEUED.value:
                scheduler.remove(job_id)
            control = manager.cancel(job_id)
            latest = manager.status(job_id)
            status = (
                JobStatus.CANCELLED.value
                if latest["status"] == JobStatus.CANCELLED.value
                else "cancel_requested"
            )
            return {
                "job_id": job_id,
                "status": status,
                "control_state": control,
                "message": control.get("message") or "取消中",
                "poll_url": f"/api/jobs/{job_id}/progress",
            }
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.patch("/api/jobs/{job_id}")
    def rename_job(job_id: str, request: RenameJobRequest) -> dict[str, str]:
        try:
            manager.rename(job_id, request.display_name)
            catalog.update_job(job_id, display_name=request.display_name)
            hub.publish("job_status", {"display_name": request.display_name}, job_id)
            return {"job_id": job_id, "display_name": request.display_name}
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/jobs/{job_id}/duplicate", status_code=201)
    def duplicate_job(job_id: str) -> dict[str, str]:
        try:
            new_id = manager.duplicate(job_id)
            summary = manager.status(new_id)
            name = summary["spec"].get("display_name", "漫画副本")
            catalog.upsert_job(new_id, name, JobStatus.READY.value)
            return {"job_id": new_id, "status": JobStatus.READY.value}
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/jobs/{job_id}/archive")
    def archive_job(job_id: str) -> dict[str, str]:
        try:
            current = manager.status(job_id)["status"]
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="找不到这个任务") from exc
        if current in {
            JobStatus.CREATED.value,
            JobStatus.INGESTING.value,
            JobStatus.RUNNING.value,
            JobStatus.QUEUED.value,
        }:
            raise HTTPException(status_code=409, detail="任务仍在准备或运行，完成后才能归档")
        try:
            catalog.update_job(job_id, archived_at=datetime.now(UTC).isoformat())
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="找不到这个任务") from exc
        hub.publish("job_status", {"status": JobStatus.ARCHIVED.value}, job_id)
        return {"job_id": job_id, "status": JobStatus.ARCHIVED.value}

    @app.post("/api/jobs/{job_id}/restore")
    def restore_job(job_id: str) -> dict[str, str]:
        try:
            summary = manager.status(job_id)
            catalog.update_job(job_id, archived_at=None, status=summary["status"])
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="找不到这个任务") from exc
        hub.publish("job_status", {"status": summary["status"]}, job_id)
        return {"job_id": job_id, "status": summary["status"]}

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str, request: DeleteJobRequest) -> dict[str, str]:
        del request
        catalog_job = catalog.job(job_id)
        if catalog_job is None:
            raise HTTPException(status_code=404, detail="找不到这个任务")
        if not catalog_job["archived_at"]:
            raise HTTPException(status_code=409, detail="只有回收站中的任务才能永久删除")
        try:
            job_dir = manager._job_dir(job_id).resolve()
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="找不到这个任务") from exc
        data_root = settings.data_root.resolve()
        if job_dir.parent != data_root or not (job_dir / "manifest.sqlite").is_file():
            raise HTTPException(status_code=409, detail="任务目录校验失败，未执行删除")
        shutil.rmtree(job_dir)
        catalog.delete_job(job_id)
        hub.publish("job_status", {"status": "deleted", "job_id": job_id})
        return {"job_id": job_id, "status": "deleted"}

    @app.post("/api/jobs/{job_id}/pages/{page_index}/retry", status_code=202)
    def retry_page(job_id: str, page_index: int) -> dict[str, object]:
        try:
            current = manager.status(job_id)["status"]
            if current in {
                JobStatus.CREATED.value,
                JobStatus.INGESTING.value,
                JobStatus.RUNNING.value,
                JobStatus.QUEUED.value,
            }:
                raise ValueError("任务仍在准备或运行，暂时不能重试页面")
            if (catalog.job(job_id) or {}).get("archived_at"):
                raise ValueError("请先从回收站恢复任务")
            reset_units = manager.retry_page(job_id, page_index)
            manager.queue(job_id)
            position = scheduler.enqueue(job_id)
            return {
                "job_id": job_id,
                "page_index": page_index,
                "reset_units": reset_units,
                "queue_position": position,
            }
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/jobs/{job_id}/repair-results", status_code=202)
    def repair_results(job_id: str) -> dict[str, str]:
        """Recompose cached model outputs without starting another GPU run."""
        try:
            summary = manager.status(job_id)
            if summary["status"] in {
                JobStatus.RUNNING.value,
                JobStatus.QUEUED.value,
                JobStatus.INGESTING.value,
            }:
                raise HTTPException(status_code=409, detail="任务仍在运行，完成或暂停后再修复结果")
            operation_id = uuid.uuid4().hex
            catalog.create_operation(operation_id, job_id, stage="repairing")
            catalog.update_operation(
                operation_id,
                status="running",
                stage="repairing",
                progress={
                    "stage": "repairing",
                    "stage_percent": 0.0,
                    "current_page": None,
                    "completed_pages": 0,
                    "total_pages": int(summary["progress"].get("total_pages") or 0),
                    "latest_message": "正在准备重组已生成页面",
                },
            )

            def run() -> None:
                try:
                    def emit_page_ready(page_index: int) -> None:
                        page = next(
                            item
                            for item in manager._manifest(job_id).pages(job_id)
                            if int(item["page_index"]) == page_index
                        )
                        if not page["output_path"]:
                            return
                        page_id = int(page["id"])
                        source_path = Path(page["source_path"])
                        output_path = Path(page["output_path"])
                        thumbnail_path = (
                            manager._job_dir(job_id)
                            / "final"
                            / "thumbnails"
                            / f"page_{page_index:05d}.jpg"
                        )
                        revision = manager._manifest(job_id).page_asset_revision(page_id)
                        if not revision:
                            candidates = [
                                candidate
                                for candidate in (source_path, output_path, thumbnail_path)
                                if candidate.is_file()
                            ]
                            if candidates:
                                revision = str(
                                    max(candidate.stat().st_mtime_ns for candidate in candidates)
                                )
                                manager._manifest(job_id).set_page_asset_revision(
                                    page_id, revision
                                )
                        hub.publish(
                            "page_ready",
                            {
                                "page_index": page_index,
                                "asset_revision": revision,
                                "status": "qa_passed",
                                "source_url": derived_asset_url(
                                    source_path,
                                    f"/api/jobs/{job_id}/pages/{page_index}/source",
                                    revision,
                                ),
                                "thumbnail_url": derived_asset_url(
                                    thumbnail_path,
                                    f"/api/jobs/{job_id}/pages/{page_index}/thumbnail",
                                    revision,
                                ),
                                "final_url": derived_asset_url(
                                    output_path,
                                    f"/api/jobs/{job_id}/pages/{page_index}/final",
                                    revision,
                                ),
                            },
                            job_id,
                        )

                    def report_page(page_index: int, completed: int, total: int) -> None:
                        catalog.update_operation(
                            operation_id,
                            status="running",
                            stage="repairing",
                            progress={
                                "stage": "repairing",
                                "stage_percent": round(
                                    completed / total * 100 if total else 100.0, 1
                                ),
                                "current_page": page_index,
                                "completed_pages": completed,
                                "total_pages": total,
                                "latest_message": f"已重组第 {page_index + 1} 页",
                            },
                        )

                    repaired = manager.repair_completed_colorization(
                        job_id, progress_callback=report_page
                    )
                    catalog.update_operation(
                        operation_id,
                        status="completed",
                        stage="ready",
                        progress={
                            "stage": "ready",
                            "stage_percent": 100.0,
                            "completed_pages": int(summary["progress"].get("total_pages") or 0),
                            "total_pages": int(summary["progress"].get("total_pages") or 0),
                            "repaired_pages": repaired,
                            "latest_message": "已完成结果重组",
                        },
                    )
                    hub.publish(
                        "results_repaired",
                        {"operation_id": operation_id, "repaired_pages": repaired},
                        job_id,
                    )
                except Exception as exc:
                    catalog.update_operation(
                        operation_id, status="failed", stage="failed", error=str(exc)
                    )
                    logger.exception("result repair failed job=%s", job_id)

            ingest_executor.submit(run)
            return {"job_id": job_id, "operation_id": operation_id, "status": "repairing"}
        except HTTPException:
            raise
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/jobs/reorder")
    def reorder_jobs(request: ReorderJobsRequest) -> dict[str, list[str]]:
        try:
            scheduler.reorder(request.job_ids)
            return {"job_ids": request.job_ids}
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> dict[str, object]:
        try:
            result = manager.status(job_id)
            catalog_job = catalog.job(job_id)
            if catalog_job:
                result["display_name"] = catalog_job["display_name"]
                result["queue_position"] = catalog_job["queue_position"]
            return public_job(result)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/jobs/{job_id}/progress")
    def job_progress(job_id: str) -> JobProgress:
        try:
            result = public_job(manager.status(job_id))
            progress = dict(result["progress"])
            progress["queue_position"] = (catalog.job(job_id) or {}).get("queue_position")
            return JobProgress.model_validate(progress)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/jobs/{job_id}/pages")
    def job_pages(job_id: str) -> JSONResponse:
        try:
            manifest = manager._manifest(job_id)
            pages = manifest.pages(job_id)
            units_by_page = manifest.page_units_for_job(job_id)
            semantic_by_page = manifest.semantic_masks_for_job(job_id)
            result: list[dict[str, Any]] = []

            for page in pages:
                units = units_by_page.get(int(page["page_index"]), [])
                source_path = Path(page["source_path"])
                thumbnail_path = (
                    manager._job_dir(job_id)
                    / "final"
                    / "thumbnails"
                    / f"page_{page['page_index']:05d}.jpg"
                )
                preview_path = (
                    manager._job_dir(job_id)
                    / "preview"
                    / "pages"
                    / f"page_{page['page_index']:05d}.png"
                )
                preview_thumbnail_path = (
                    manager._job_dir(job_id)
                    / "preview"
                    / "thumbnails"
                    / f"page_{page['page_index']:05d}.jpg"
                )
                source_display_path = (
                    manager._job_dir(job_id)
                    / "display"
                    / "source"
                    / f"page_{page['page_index']:05d}.webp"
                )
                final_display_path = (
                    manager._job_dir(job_id)
                    / "display"
                    / "final"
                    / f"page_{page['page_index']:05d}.webp"
                )
                output_path = Path(page["output_path"] or "")
                has_final = bool(page["output_path"] and output_path.is_file())
                revision = page.get("asset_revision")
                if not revision:
                    candidates = [
                        candidate
                        for candidate in (
                            source_path,
                            output_path,
                            preview_path,
                            thumbnail_path,
                            preview_thumbnail_path,
                        )
                        if candidate.is_file()
                    ]
                    revision = (
                        str(max(candidate.stat().st_mtime_ns for candidate in candidates))
                        if candidates
                        else None
                    )
                semantic_row = semantic_by_page.get(int(page["page_index"]))
                semantic_descriptor_payload = (
                    dict(semantic_row.get("descriptor") or {}) if semantic_row else None
                )
                source_base = f"/api/jobs/{job_id}/pages/{page['page_index']}/source"
                thumbnail_base = (
                    f"/api/jobs/{job_id}/pages/{page['page_index']}/thumbnail"
                )
                final_base = f"/api/jobs/{job_id}/pages/{page['page_index']}/final"
                preview_base = f"/api/jobs/{job_id}/pages/{page['page_index']}/preview"
                preview_thumbnail_base = (
                    f"/api/jobs/{job_id}/pages/{page['page_index']}/preview-thumbnail"
                )
                result.append(
                    {
                        "page_index": int(page["page_index"]),
                        "status": page["status"],
                        "completed_units": sum(unit["status"] == "qa_passed" for unit in units),
                        "total_units": len(units),
                        "error": next((unit["error"] for unit in units if unit["error"]), None),
                        "asset_revision": revision,
                        "source_url": derived_asset_url(source_path, source_base, revision),
                        "source_display_url": (
                            f"/api/assets/jobs/{job_id}/pages/"
                            f"{page['page_index']}/source.webp?v={revision or 0}"
                        )
                        if source_display_path.is_file()
                        else None,
                        "thumbnail_url": (
                            derived_asset_url(thumbnail_path, thumbnail_base, revision)
                            if has_final and thumbnail_path.is_file()
                            else derived_asset_url(
                                preview_thumbnail_path, preview_thumbnail_base, revision
                            )
                            if preview_thumbnail_path.is_file()
                            else None
                        ),
                        "final_url": derived_asset_url(output_path, final_base, revision)
                        if has_final
                        else None,
                        "final_display_url": (
                            f"/api/assets/jobs/{job_id}/pages/"
                            f"{page['page_index']}/final.webp?v={revision or 0}"
                            if has_final and final_display_path.is_file()
                            else None
                        ),
                        "preview_url": derived_asset_url(preview_path, preview_base, revision)
                        if preview_path.is_file()
                        else None,
                        "preview_only": bool(not has_final and preview_path.is_file()),
                        "mask_status": (
                            str(semantic_descriptor_payload.get("status") or "cached")
                            if semantic_descriptor_payload
                            else "pending"
                        ),
                        "semantic_mask": semantic_descriptor_payload,
                    }
                )
            return JSONResponse(
                content=result,
                headers={"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"},
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/jobs/{job_id}/pages/{page_index}/masks")
    def semantic_masks(job_id: str, page_index: int) -> dict[str, Any]:
        try:
            result = manager.semantic_page(job_id, page_index)
            descriptor = semantic_descriptor(manager.semantic_engine, result)
            base = f"/api/jobs/{job_id}/pages/{page_index}/semantic"
            return {
                "descriptor": descriptor,
                "protection_url": f"/api/jobs/{job_id}/pages/{page_index}/mask",
                "confidence_url": f"{base}/confidence",
                "uncertain_url": f"{base}/uncertain",
                "layers": {
                    name: f"{base}/{name}" for name in SEMANTIC_CLASSES
                },
            }
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="页面语义遮罩尚不可用") from exc

    @app.get("/api/jobs/{job_id}/pages/{page_index}/semantic/{layer}")
    def semantic_layer(job_id: str, page_index: int, layer: str) -> Response:
        allowed_layers = {*SEMANTIC_CLASSES, "confidence", "uncertain"}
        if layer not in allowed_layers:
            raise HTTPException(status_code=404, detail="未知语义层")
        try:
            result = manager.semantic_page(job_id, page_index)
            if layer == "confidence":
                array = np.clip(result.confidence * 255, 0, 255).astype("uint8")
            elif layer == "uncertain":
                array = result.uncertain.astype("uint8") * 255
            else:
                array = result.masks[layer].astype("uint8") * 255
            buffer = io.BytesIO()
            Image.fromarray(array, mode="L").save(buffer, format="PNG")
            return Response(buffer.getvalue(), media_type="image/png")
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="页面语义遮罩尚不可用") from exc

    @app.post("/api/jobs/{job_id}/pages/{page_index}/mask-corrections", status_code=202)
    def save_mask_corrections(
        job_id: str, page_index: int, request: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            manager.semantic_page(job_id, page_index)
            corrections = request.get("corrections", [])
            if not isinstance(corrections, list):
                raise ValueError("遮罩修正必须是数组")
            target = (
                manager._job_dir(job_id)
                / "masks"
                / "corrections"
                / f"page_{page_index:05d}.json"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(
                    {"page_index": page_index, "corrections": corrections},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            page = manager._manifest(job_id).page_by_index(job_id, page_index)
            manager._manifest(job_id).save_mask_correction(int(page["id"]), corrections)
            hub.publish(
                "mask_correction_saved",
                {"page_index": page_index, "count": len(corrections)},
                job_id,
            )
            return {"job_id": job_id, "page_index": page_index, "count": len(corrections)}
        except (KeyError, ValueError, StopIteration) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/jobs/{job_id}/identities")
    def identities(job_id: str) -> list[dict[str, Any]]:
        try:
            return manager._manifest(job_id).identities(job_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="找不到这个任务") from exc

    @app.put("/api/jobs/{job_id}/identities/{identity_id}")
    def save_identity(
        job_id: str, identity_id: str, request: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            label = str(request.get("label") or identity_id)[:120]
            region = str(request.get("region") or "character")[:80]
            manager._manifest(job_id).save_identity(
                job_id,
                identity_id=identity_id,
                label=label,
                region=region,
                color=str(request["color"]) if request.get("color") else None,
                shadow_color=(
                    str(request["shadow_color"]) if request.get("shadow_color") else None
                ),
                confidence=float(request.get("confidence") or 0.0),
                locked=bool(request.get("locked", False)),
            )
            hub.publish(
                "identity_updated",
                {"identity_id": identity_id, "label": label, "region": region},
                job_id,
            )
            return next(
                item
                for item in manager._manifest(job_id).identities(job_id)
                if item["identity_id"] == identity_id and item["region"] == region
            )
        except (KeyError, ValueError, StopIteration) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/jobs/{job_id}/pages/{page_index}/{variant}", response_model=None)
    def page_image(
        job_id: str,
        page_index: int,
        variant: Literal[
            "source",
            "final",
            "mask",
            "thumbnail",
            "preview",
            "preview-thumbnail",
        ],
    ) -> FileResponse | Response:
        try:
            manifest = manager._manifest(job_id)
            page = manifest.page_by_index(job_id, page_index)
            if variant == "thumbnail":
                path = (
                    manager._job_dir(job_id) / "final" / "thumbnails" / f"page_{page_index:05d}.jpg"
                )
                if not path.is_file():
                    path = (
                        manager._job_dir(job_id)
                        / "preview"
                        / "thumbnails"
                        / f"page_{page_index:05d}.jpg"
                    )
            elif variant == "preview-thumbnail":
                path = (
                    manager._job_dir(job_id)
                    / "preview"
                    / "thumbnails"
                    / f"page_{page_index:05d}.jpg"
                )
            elif variant == "preview":
                path = (
                    manager._job_dir(job_id)
                    / "preview"
                    / "pages"
                    / f"page_{page_index:05d}.png"
                )
                if not path.is_file():
                    recovered = manager._assemble_page_preview(job_id, manifest, int(page["id"]))
                    if recovered is not None:
                        path = recovered
            elif variant == "mask":
                units = manifest.page_units(int(page["id"]))
                if not units:
                    raise FileNotFoundError("mask")
                if len(units) == 1:
                    path = Path(units[0]["mask_path"])
                else:
                    with Image.open(page["source_path"]) as source_image:
                        canvas = Image.new("L", source_image.size, 0)
                    for unit in units:
                        with Image.open(unit["mask_path"]) as unit_mask:
                            canvas.paste(
                                unit_mask.convert("L"),
                                (int(unit["x"]), int(unit["y"])),
                            )
                    buffer = io.BytesIO()
                    canvas.save(buffer, format="PNG")
                    return Response(buffer.getvalue(), media_type="image/png")
            else:
                path = Path(
                    page["source_path"] if variant == "source" else page["output_path"] or ""
                )
            if not path.is_file():
                raise FileNotFoundError(path)
            media_type = (
                "image/jpeg"
                if variant in {"thumbnail", "preview-thumbnail"}
                else "image/png"
            )
            return FileResponse(
                path,
                media_type=media_type,
                headers={
                    "Cache-Control": "no-store, max-age=0",
                    "Pragma": "no-cache",
                },
            )
        except (KeyError, StopIteration, FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="页面图像尚不可用") from exc

    @app.get(
        "/api/assets/jobs/{job_id}/pages/{page_index}/{variant}.webp",
        response_model=None,
    )
    def page_display_image(
        job_id: str,
        page_index: int,
        variant: Literal["source", "final"],
        v: str | None = None,
    ) -> FileResponse | Response:
        try:
            path = manager.display_asset(job_id, page_index, variant)
            cache_control = (
                "private, max-age=31536000, immutable"
                if v
                else "no-store, max-age=0"
            )
            return FileResponse(
                path,
                media_type="image/webp",
                headers={"Cache-Control": cache_control},
            )
        except DisplayAssetPending:
            manager.schedule_display_asset(job_id, page_index, variant)
            return JSONResponse(
                {
                    "status": "preparing",
                    "page_index": page_index,
                    "variant": variant,
                    "message": "页面显示资源正在准备，请稍后重试",
                },
                status_code=202,
                headers={"Cache-Control": "no-store, max-age=0"},
            )
        except (KeyError, StopIteration, FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="页面显示图尚不可用") from exc

    def _download_output(job_id: str) -> Path:
        try:
            output_dir = manager._job_dir(job_id) / "output"
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="找不到这个任务") from exc
        outputs = sorted(
            (
                path
                for path in output_dir.glob("book*")
                if path.is_file() and path.name in DOWNLOAD_OUTPUT_NAMES
            ),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        if not outputs:
            raise HTTPException(status_code=404, detail="整本成品尚不可用")
        return outputs[0]

    def _download_media_type(path: Path) -> str:
        return {
            ".cbz": "application/vnd.comicbook+zip",
            ".pdf": "application/pdf",
            ".zip": "application/zip",
        }.get(path.suffix.casefold(), "application/octet-stream")

    def _download_headers(path: Path) -> dict[str, str]:
        size = path.stat().st_size
        return {
            "Accept-Ranges": "bytes",
            "Content-Length": str(size),
            "Content-Disposition": f'attachment; filename="{path.name}"',
            "Cache-Control": "no-store, max-age=0",
            "Pragma": "no-cache",
        }

    @app.get("/api/jobs/{job_id}/download-info")
    def download_info(job_id: str) -> dict[str, Any]:
        path = _download_output(job_id)
        stat = path.stat()
        return {
            "ready": True,
            "file_name": path.name,
            "media_type": _download_media_type(path),
            "size_bytes": stat.st_size,
            "revision": f"{stat.st_mtime_ns}-{stat.st_size}",
            "download_url": f"/api/jobs/{job_id}/download",
        }

    @app.head("/api/jobs/{job_id}/download")
    def download_head(job_id: str) -> Response:
        path = _download_output(job_id)
        return Response(
            status_code=200,
            media_type=_download_media_type(path),
            headers=_download_headers(path),
        )

    @app.get("/api/jobs/{job_id}/download", response_model=None)
    def download(job_id: str, request: Request) -> StreamingResponse | Response:
        path = _download_output(job_id)
        size = path.stat().st_size
        headers = _download_headers(path)
        range_header = request.headers.get("range")
        if not range_header:
            return StreamingResponse(
                _iter_file_range(path, 0, size),
                status_code=200,
                media_type=_download_media_type(path),
                headers=headers,
            )
        if not range_header.startswith("bytes=") or "," in range_header:
            return Response(
                status_code=416,
                headers={**headers, "Content-Range": f"bytes */{size}"},
            )
        value = range_header[6:].strip()
        try:
            start_text, end_text = value.split("-", 1)
            if not start_text:
                suffix_length = int(end_text)
                if suffix_length <= 0:
                    raise ValueError
                start = max(0, size - suffix_length)
                end = size - 1
            else:
                start = int(start_text)
                end = int(end_text) if end_text else size - 1
                if start < 0 or end < start:
                    raise ValueError
                end = min(end, size - 1)
            if start >= size or end < start:
                raise ValueError
        except (TypeError, ValueError):
            return Response(
                status_code=416,
                headers={**headers, "Content-Range": f"bytes */{size}"},
            )
        ranged_headers = {
            **headers,
            "Content-Length": str(end - start + 1),
            "Content-Range": f"bytes {start}-{end}/{size}",
        }
        return StreamingResponse(
            _iter_file_range(path, start, end + 1),
            status_code=206,
            media_type=_download_media_type(path),
            headers=ranged_headers,
        )

    @app.get("/api/models")
    def models() -> list[ModelDescriptor]:
        health_data = registry.health()
        connected_repositories = {
            str(state.get("model_id"))
            for name, state in health_data.items()
            if name != "palette" and state.get("ok") and state.get("model_id")
        }
        result: list[dict[str, Any]] = []
        for descriptor in model_catalog["models"]:
            marker = settings.model_root / "installed" / f"{descriptor['id']}.json"
            # The compatibility adapter was previously recorded under the
            # provisional ``semantic-manga-v1`` filename. Read that marker
            # during the transition, while exposing the truthful compatible
            # model id in the catalog and download actions.
            legacy_marker = (
                settings.model_root / "installed" / "semantic-manga-v1.json"
                if descriptor["id"] == "semantic-manga-v1-compatible"
                else None
            )
            item = dict(descriptor)
            service_state = next(
                (
                    state
                    for name, state in health_data.items()
                    if name != "palette"
                    and state.get("model_id") == descriptor["repository"]
                    and state.get("ok")
                ),
                None,
            )
            connected = descriptor["repository"] in connected_repositories
            if descriptor["id"] == "semantic-manga-v1-compatible":
                semantic_url = getattr(manager.semantic_engine, "base_url", "")
                if semantic_url:
                    try:
                        semantic_response = httpx.get(f"{semantic_url}/health", timeout=2)
                        semantic_payload = semantic_response.json()
                        connected = bool(
                            semantic_response.is_success and semantic_payload.get("available")
                        )
                    except (httpx.HTTPError, ValueError):
                        connected = False
            item["installed"] = marker.is_file() or bool(legacy_marker and legacy_marker.is_file())
            item["connected"] = connected
            item["supports_interrupt"] = bool(
                service_state and service_state.get("supports_interrupt", False)
            )
            item["supports_release"] = bool(
                service_state and service_state.get("supports_release", False)
            )
            item["status"] = (
                "ready" if connected else "installed" if item["installed"] else "not_ready"
            )
            with model_download_lock:
                if not connected and descriptor["id"] in model_downloads:
                    item["status"] = "downloading"
            result.append(item)
        return [ModelDescriptor.model_validate(item) for item in result]

    def download_model_task(model_id: str) -> None:
        hub.publish("model_progress", {"model_id": model_id, "status": "downloading"})
        try:
            from huggingface_hub import snapshot_download

            descriptor = next(item for item in model_catalog["models"] if item["id"] == model_id)
            local_dir = None
            if model_id == "semantic-manga-v1-compatible":
                local_dir = settings.model_root / "semantic" / "koharu-yolo26s"
                local_dir.mkdir(parents=True, exist_ok=True)
            snapshot_path = snapshot_download(
                repo_id=descriptor["repository"],
                revision=descriptor["revision"],
                cache_dir=settings.model_root / "hub",
                allow_patterns=descriptor.get("allow_patterns"),
                local_dir=local_dir,
            )
            marker = settings.model_root / "installed" / f"{model_id}.json"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(
                json.dumps(
                    {
                        "id": model_id,
                        "repository": descriptor["repository"],
                        "revision": descriptor["revision"],
                        "snapshot_path": snapshot_path,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            hub.publish("model_progress", {"model_id": model_id, "status": "downloaded"})
        except Exception as exc:
            hub.publish(
                "model_progress", {"model_id": model_id, "status": "failed", "message": str(exc)}
            )
        finally:
            with model_download_lock:
                model_downloads.discard(model_id)

    @app.post("/api/models/{model_id}/download", status_code=202)
    @app.post("/api/models/{model_id}/retry", status_code=202)
    def download_model(model_id: str) -> dict[str, str]:
        descriptor = next(
            (item for item in model_catalog["models"] if item["id"] == model_id), None
        )
        if descriptor is None:
            raise HTTPException(status_code=404, detail="未知模型")
        if descriptor.get("downloadable", True) is False:
            raise HTTPException(
                status_code=409,
                detail=str(
                    descriptor.get("unavailable_reason")
                    or "这个模型插槽暂未提供可下载权重"
                ),
            )
        with model_download_lock:
            if model_id in model_downloads:
                return {"model_id": model_id, "status": "already_downloading"}
            model_downloads.add(model_id)
        threading.Thread(target=download_model_task, args=(model_id,), daemon=True).start()
        return {"model_id": model_id, "status": "downloading"}

    @app.post("/api/models/{model_id}/release", status_code=202)
    def release_model(model_id: str) -> dict[str, object]:
        descriptor = next(
            (item for item in model_catalog["models"] if item["id"] == model_id), None
        )
        if descriptor is None:
            raise HTTPException(status_code=404, detail="未知模型")
        health_data = registry.health()
        engine_name = next(
            (
                name
                for name, state in health_data.items()
                if name != "palette"
                and state.get("model_id") == descriptor.get("repository")
            ),
            None,
        )
        if engine_name is None:
            raise HTTPException(status_code=409, detail="本地模型服务未连接")
        engine_state = health_data.get(engine_name) or {}
        if not engine_state.get("supports_release", False):
            raise HTTPException(
                status_code=409,
                detail="当前模型服务尚未启用显存释放控制，请在受保护任务终态后维护重启",
            )
        try:
            result = registry.release(engine_name)
        except Exception as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        hub.publish(
            "model_progress",
            {"model_id": model_id, "engine": engine_name, **result},
        )
        return {"model_id": model_id, "engine": engine_name, **result}

    return app


class _LazyApp:
    """Defer the default ASGI application until it is actually served.

    Importing :mod:`manga_repaint.api` is also used by the CLI and test suite
    and must not start a second queue worker or run job recovery as a side
    effect. Direct ASGI servers can still target ``manga_repaint.api:app``.
    """

    def __init__(self) -> None:
        self._app: FastAPI | None = None
        self._lock = threading.Lock()

    def _resolve(self) -> FastAPI:
        with self._lock:
            if self._app is None:
                self._app = create_app()
            return self._app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        await self._resolve()(scope, receive, send)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resolve(), name)


app = _LazyApp()
