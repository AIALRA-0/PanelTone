from __future__ import annotations

import json
import re
import shutil
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .catalog import Catalog
from .config import Settings
from .engines import EngineRegistry
from .models import DetailMode, JobMode, JobSpec, JobStatus, ProtectionMode
from .presets import presets_payload
from .project import ProjectManager
from .runtime import EventHub, JobQueue
from .schemas import JobProgress, ModelDescriptor, PresetCatalog, SourceDescriptor

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


class JobOptions(BaseModel):
    mode: JobMode = JobMode.COLORIZE
    engine: str = "palette"
    protection: ProtectionMode = ProtectionMode.STRICT
    detail_mode: DetailMode = DetailMode.STRICT
    output_format: Literal["cbz", "pdf", "images"] = "cbz"
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


class DeleteJobRequest(BaseModel):
    confirmation: Literal["永久删除"]


def _option_payload() -> dict[str, list[dict[str, str]]]:
    return {
        "modes": [
            {
                "id": "colorize",
                "name": "原作上色",
                "description": "保留原作构图与线稿，只生成颜色和光影",
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
                "description": "允许模型重画线条，只保护文字与分镜",
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
                "name": "PNG 图片",
                "description": "保留逐页 PNG，不生成压缩包",
                "best_for": "继续编辑和质量检查",
                "changes": "不额外压缩页面",
                "tradeoff": "需要自行管理文件夹",
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

    def public_job(job: dict[str, Any]) -> dict[str, Any]:
        spec = dict(job.get("spec", {}))
        source_value = spec.pop("source", None)
        spec.pop("workspace", None)
        references = spec.pop("style_references", [])
        spec.pop("metadata", None)
        source_bytes = 0
        if source_value:
            source_path = Path(source_value)
            if source_path.is_file():
                source_bytes = source_path.stat().st_size
            elif source_path.is_dir():
                source_bytes = sum(
                    path.stat().st_size for path in source_path.rglob("*") if path.is_file()
                )
        spec["style_reference_count"] = len(references)
        job["spec"] = spec
        progress = dict(job.get("progress", {}))
        progress["uploaded_bytes"] = source_bytes
        progress["total_upload_bytes"] = source_bytes
        job["progress"] = progress
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
            if catalog_job["archived_at"]:
                job["status"] = JobStatus.ARCHIVED.value
            results.append(public_job(job))
        return results

    hub = EventHub(catalog, public_jobs)

    def publish(kind: str, payload: dict[str, Any], job_id: str | None = None) -> None:
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
        hub.publish(kind, payload, job_id)

    manager.set_event_callback(publish)
    scheduler = JobQueue(manager.process, publish)
    app = FastAPI(title="PanelTone", version="0.2.0-alpha.1")
    app.state.manager = manager
    app.state.catalog = catalog
    app.state.events = hub
    app.state.scheduler = scheduler

    static_root = Path(__file__).with_name("web") / "static"
    app.mount("/static", StaticFiles(directory=static_root), name="static")

    @app.get("/")
    def home() -> FileResponse:
        return FileResponse(static_root / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {"ok": True, "engines": registry.health(), "queued_jobs": len(scheduler.positions())}

    @app.get("/api/presets")
    def presets() -> PresetCatalog:
        result: dict[str, Any] = presets_payload()
        result.update(_option_payload())
        return PresetCatalog.model_validate(result)

    @app.post("/api/import", status_code=201)
    def import_local_file(file: Annotated[UploadFile, File()]) -> SourceDescriptor:
        upload_id = uuid.uuid4().hex
        original_name = Path(file.filename or "upload.bin").name
        suffix = Path(original_name).suffix.casefold()
        if suffix not in ALLOWED_UPLOADS:
            raise HTTPException(status_code=400, detail="不支持这个文件格式")
        incoming = settings.data_root / "imports" / "incoming"
        incoming.mkdir(parents=True, exist_ok=True)
        target = incoming / f"{uuid.uuid4().hex}{suffix}"
        size = 0
        last_reported = 0
        expected_size = int(file.size or 0)
        max_size = settings.max_upload_mib * 1024 * 1024
        try:
            with target.open("wb") as destination:
                while chunk := file.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > max_size:
                        raise HTTPException(status_code=413, detail="文件超过本机任务允许的大小")
                    destination.write(chunk)
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
            source = catalog.add_source(original_name, target, ALLOWED_UPLOADS[suffix])
        except Exception:
            target.unlink(missing_ok=True)
            raise
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
        )

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

    @app.get("/api/jobs")
    def list_jobs(include_archived: bool = False) -> list[dict[str, object]]:
        return public_jobs(include_archived)

    @app.post("/api/jobs", status_code=201)
    def create_job(request: CreateJobRequest) -> dict[str, str]:
        try:
            if request.source_id:
                source = Path(catalog.source(request.source_id)["path"])
            elif request.source:
                source = Path(request.source)
            else:
                raise ValueError("缺少漫画来源")
            job_id = create_one(request, source, request.display_name)
            return {"job_id": job_id, "status": JobStatus.READY.value}
        except (ValueError, KeyError, PermissionError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/jobs/batch", status_code=201)
    def create_batch(request: BatchJobRequest) -> dict[str, list[dict[str, str]]]:
        try:
            sources = [catalog.source(source_id) for source_id in request.source_ids]
            image_sources = [item for item in sources if item["kind"] == "image"]
            book_sources = [item for item in sources if item["kind"] == "book"]
            created: list[dict[str, str]] = []
            if image_sources:
                natural_sources = sorted(
                    image_sources,
                    key=lambda item: [
                        int(part) if part.isdigit() else part.casefold()
                        for part in re.split(r"(\d+)", item["original_name"])
                    ],
                )
                order = request.image_order or [item["id"] for item in natural_sources]
                if set(order) != {item["id"] for item in image_sources}:
                    raise ValueError("图片顺序必须包含全部已选择图片")
                group = catalog.group_images(order, request.image_book_name)
                job_id = create_one(request, group, request.image_book_name)
                created.append({"job_id": job_id, "display_name": request.image_book_name})
            for source in book_sources:
                name = Path(source["original_name"]).stem
                job_id = create_one(request, Path(source["path"]), name)
                created.append({"job_id": job_id, "display_name": name})
            return {"jobs": created}
        except (ValueError, KeyError, PermissionError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/events")
    def events(
        request: Request, last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None
    ) -> StreamingResponse:
        cursor = int(last_event_id or request.query_params.get("after", "0") or "0")
        return StreamingResponse(
            hub.stream(cursor),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/jobs/{job_id}/start", status_code=202)
    def start_job(job_id: str) -> dict[str, object]:
        try:
            summary = manager.status(job_id)
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
            catalog.update_job(job_id, status=JobStatus.QUEUED.value, queue_position=position)
            return {"job_id": job_id, "status": JobStatus.QUEUED.value, "queue_position": position}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/jobs/{job_id}/pause", status_code=202)
    def pause_job(job_id: str) -> dict[str, str]:
        try:
            summary = manager.status(job_id)
            if summary["status"] == JobStatus.QUEUED.value and scheduler.remove(job_id):
                manager.set_status(job_id, JobStatus.PAUSED)
                return {"job_id": job_id, "status": JobStatus.PAUSED.value}
            manager.pause(job_id)
            return {"job_id": job_id, "status": "pause_requested"}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/jobs/{job_id}/cancel", status_code=202)
    def cancel_job(job_id: str) -> dict[str, str]:
        try:
            summary = manager.status(job_id)
            if summary["status"] == JobStatus.QUEUED.value and scheduler.remove(job_id):
                manager.set_status(job_id, JobStatus.CANCELLED)
                return {"job_id": job_id, "status": JobStatus.CANCELLED.value}
            manager.cancel(job_id)
            return {"job_id": job_id, "status": "cancel_requested"}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

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
        current = manager.status(job_id)["status"]
        if current in {JobStatus.RUNNING.value, JobStatus.QUEUED.value}:
            raise HTTPException(status_code=409, detail="运行中的任务不能归档")
        catalog.update_job(job_id, archived_at=datetime.now(UTC).isoformat())
        hub.publish("job_status", {"status": JobStatus.ARCHIVED.value}, job_id)
        return {"job_id": job_id, "status": JobStatus.ARCHIVED.value}

    @app.post("/api/jobs/{job_id}/restore")
    def restore_job(job_id: str) -> dict[str, str]:
        summary = manager.status(job_id)
        catalog.update_job(job_id, archived_at=None, status=summary["status"])
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
        job_dir = manager._job_dir(job_id).resolve()
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
            reset_units = manager.retry_page(job_id, page_index)
            manager.queue(job_id)
            position = scheduler.enqueue(job_id)
            catalog.update_job(
                job_id,
                status=JobStatus.QUEUED.value,
                queue_position=position,
            )
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
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/jobs/{job_id}/progress")
    def job_progress(job_id: str) -> JobProgress:
        try:
            result = public_job(manager.status(job_id))
            progress = dict(result["progress"])
            progress["queue_position"] = (catalog.job(job_id) or {}).get("queue_position")
            return JobProgress.model_validate(progress)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/jobs/{job_id}/pages")
    def job_pages(job_id: str) -> list[dict[str, Any]]:
        try:
            return [
                {
                    "page_index": int(page["page_index"]),
                    "status": page["status"],
                    "source_url": f"/api/jobs/{job_id}/pages/{page['page_index']}/source",
                    "thumbnail_url": (
                        f"/api/jobs/{job_id}/pages/{page['page_index']}/thumbnail"
                        if page["output_path"]
                        else None
                    ),
                    "final_url": f"/api/jobs/{job_id}/pages/{page['page_index']}/final"
                    if page["output_path"]
                    else None,
                }
                for page in manager._manifest(job_id).pages(job_id)
            ]
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/jobs/{job_id}/pages/{page_index}/{variant}")
    def page_image(
        job_id: str,
        page_index: int,
        variant: Literal["source", "final", "mask", "thumbnail"],
    ) -> FileResponse:
        try:
            manifest = manager._manifest(job_id)
            page = next(
                item for item in manifest.pages(job_id) if int(item["page_index"]) == page_index
            )
            if variant == "thumbnail":
                path = (
                    manager._job_dir(job_id) / "final" / "thumbnails" / f"page_{page_index:05d}.jpg"
                )
            elif variant == "mask":
                units = manifest.page_units(int(page["id"]))
                path = Path(units[0]["mask_path"]) if len(units) == 1 else Path("")
            else:
                path = Path(
                    page["source_path"] if variant == "source" else page["output_path"] or ""
                )
            if not path.is_file():
                raise FileNotFoundError(path)
            media_type = "image/jpeg" if variant == "thumbnail" else "image/png"
            return FileResponse(path, media_type=media_type)
        except (KeyError, StopIteration, FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="页面图像尚不可用") from exc

    @app.get("/api/jobs/{job_id}/download")
    def download(job_id: str) -> FileResponse:
        outputs = list((manager._job_dir(job_id) / "output").glob("book.*"))
        if not outputs:
            raise HTTPException(status_code=404, detail="整本成品尚不可用")
        media_type = {".cbz": "application/vnd.comicbook+zip", ".pdf": "application/pdf"}.get(
            outputs[0].suffix.casefold(), "application/octet-stream"
        )
        return FileResponse(outputs[0], filename=outputs[0].name, media_type=media_type)

    @app.get("/api/models")
    def models() -> list[ModelDescriptor]:
        health_data = registry.health()
        connected = any(state.get("ok") for name, state in health_data.items() if name != "palette")
        result: list[dict[str, Any]] = []
        for descriptor in model_catalog["models"]:
            marker = settings.model_root / "installed" / f"{descriptor['id']}.json"
            item = dict(descriptor)
            item["installed"] = marker.is_file()
            item["connected"] = connected
            item["status"] = (
                "ready" if connected else "installed" if marker.is_file() else "not_ready"
            )
            result.append(item)
        return [ModelDescriptor.model_validate(item) for item in result]

    def download_model_task(model_id: str) -> None:
        hub.publish("model_progress", {"model_id": model_id, "status": "downloading"})
        try:
            from huggingface_hub import snapshot_download

            descriptor = next(item for item in model_catalog["models"] if item["id"] == model_id)
            snapshot_path = snapshot_download(
                repo_id=descriptor["repository"],
                revision=descriptor["revision"],
                cache_dir=settings.model_root,
                allow_patterns=descriptor.get("allow_patterns"),
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

    @app.post("/api/models/{model_id}/download", status_code=202)
    @app.post("/api/models/{model_id}/retry", status_code=202)
    def download_model(model_id: str) -> dict[str, str]:
        if model_id not in {item["id"] for item in model_catalog["models"]}:
            raise HTTPException(status_code=404, detail="未知模型")
        threading.Thread(target=download_model_task, args=(model_id,), daemon=True).start()
        return {"model_id": model_id, "status": "downloading"}

    return app


app = create_app()
