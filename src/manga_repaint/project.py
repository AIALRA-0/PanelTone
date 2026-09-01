from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .color import composite_protected, composite_strict_colorization, preserve_ink_overlay
from .config import Settings, ensure_allowed_path
from .engines import EngineInterrupted, EngineRegistry, EngineRequest
from .export import export_book
from .hashing import stable_hash
from .ingest import ingest_book, page_metadata
from .manifest import Manifest
from .masks import (
    apply_mask_corrections,
    deterministic_protection_mask,
    ink_detail_mask,
    save_mask,
)
from .models import DetailMode, JobMode, JobSpec, JobStatus, ProtectionMode
from .panels import extract_panels
from .presets import build_prompt, get_color_preset, get_style_preset
from .qa import evaluate
from .semantic import (
    ConservativeSemanticMaskEngine,
    SemanticMaskEngine,
    SemanticMaskResult,
    semantic_descriptor,
)

logger = logging.getLogger("paneltone.job")


class _IngestCancelled(Exception):
    """Internal checkpoint signal used to stop a large import safely."""


class ProjectManager:
    def __init__(
        self,
        settings: Settings,
        registry: EngineRegistry,
        event_callback: Callable[[str, dict[str, Any], str | None], None] | None = None,
        semantic_engine: SemanticMaskEngine | None = None,
    ):
        self.settings = settings
        self.registry = registry
        self.settings.data_root.mkdir(parents=True, exist_ok=True)
        self._controls: dict[str, threading.Event] = {}
        self._cancel_controls: dict[str, threading.Event] = {}
        self._control_states: dict[str, dict[str, Any]] = {}
        self._control_lock = threading.RLock()
        self._control_watchers: set[str] = set()
        self._ingest_locks: dict[str, threading.Lock] = {}
        self._process_lock = threading.Lock()
        self._active_job_lock = threading.Lock()
        self._active_job_id: str | None = None
        # Preview assembly can be requested both by the GPU worker and by a
        # browser refresh.  Serialize the tiny disk write so a refresh cannot
        # observe a partially-written preview or race the temporary file.
        self._preview_lock = threading.RLock()
        self.semantic_engine = semantic_engine or ConservativeSemanticMaskEngine()
        self._semantic_cache: dict[tuple[str, int], SemanticMaskResult] = {}
        # Semantic masks are persisted on disk and can be large for a full
        # book. Keep only a tiny hot cache so a 300-page run cannot retain one
        # full-resolution mask set per page in the worker process.
        self._semantic_cache_limit = 2
        self._event_callback = event_callback
        self.recover_interrupted_jobs()

    def set_event_callback(
        self, callback: Callable[[str, dict[str, Any], str | None], None]
    ) -> None:
        self._event_callback = callback

    def _emit(self, kind: str, payload: dict[str, Any], job_id: str | None = None) -> None:
        if self._event_callback:
            self._event_callback(kind, payload, job_id)

    def _set_control_state(
        self,
        job_id: str,
        action: str,
        *,
        requested_at: str | None = None,
        deadline_at: str | None = None,
        active_request: bool = False,
        message: str | None = None,
    ) -> dict[str, Any]:
        state = {
            "action": action,
            "requested_at": requested_at,
            "deadline_at": deadline_at,
            "active_request": active_request,
            "message": message,
        }
        with self._control_lock:
            self._control_states[job_id] = state
        self._emit("job_control", state, job_id)
        return state

    def control_state(self, job_id: str) -> dict[str, Any] | None:
        with self._control_lock:
            state = self._control_states.get(job_id)
            return dict(state) if state else None

    def _watch_control(self, job_id: str, action: str, timeout: float = 15.0) -> None:
        """Wait for the active request to acknowledge a pause or cancel."""
        with self._control_lock:
            if job_id in self._control_watchers:
                return
            self._control_watchers.add(job_id)
        try:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                with self._active_job_lock:
                    active = self._active_job_id == job_id
                if not active:
                    return
                time.sleep(0.1)
            with self._active_job_lock:
                still_active = self._active_job_id == job_id
            if still_active:
                # Ask the engine again at the bounded deadline.  The worker
                # remains the sole authority for resetting its running unit.
                self._interrupt_active_engine(job_id)
                self._set_control_state(
                    job_id,
                    f"{action}_timeout",
                    active_request=True,
                    message="模型请求未在 15 秒内确认中断，正在等待安全回收",
                )
                logger.warning("job=%s %s request exceeded %.0f seconds", job_id, action, timeout)
        finally:
            with self._control_lock:
                self._control_watchers.discard(job_id)

    def recover_interrupted_jobs(self) -> list[str]:
        recovered: list[str] = []
        stale_statuses = {
            JobStatus.PAUSED.value,
            JobStatus.QUEUED.value,
            JobStatus.WAITING_MODEL.value,
            JobStatus.CANCELLED.value,
            JobStatus.NEEDS_ATTENTION.value,
        }
        for directory in self.settings.data_root.iterdir():
            manifest_path = directory / "manifest.sqlite"
            if not directory.is_dir() or not manifest_path.is_file():
                continue
            try:
                manifest = Manifest(manifest_path)
                if manifest.recover_interrupted(
                    directory.name,
                    stale_after_seconds=self.settings.recovery_stale_seconds,
                ):
                    recovered.append(directory.name)
                summary = manifest.summary(directory.name)
                if summary["status"] in stale_statuses:
                    reset = manifest.reset_running_units(
                        directory.name, "启动时清理非运行任务的遗留单元"
                    )
                    if reset:
                        logger.info(
                            "job=%s reset stale running units=%s during startup recovery",
                            directory.name,
                            reset,
                        )
            except Exception:
                continue
        return recovered

    def _job_dir(self, job_id: str) -> Path:
        if not job_id or any(character not in "0123456789abcdef-" for character in job_id):
            raise ValueError("Invalid job id")
        return self.settings.data_root / job_id

    def _manifest(self, job_id: str) -> Manifest:
        return Manifest(self._job_dir(job_id) / "manifest.sqlite")

    def _allowed_source_roots(self) -> list[Path]:
        if not self.settings.allowed_roots:
            return []
        roots = [root.resolve() for root in self.settings.allowed_roots]
        data_root = self.settings.data_root.resolve()
        if data_root not in roots:
            roots.append(data_root)
        return roots

    def _normalize_spec(self, spec: JobSpec) -> tuple[JobSpec, Path, list[Path]]:
        # Uploaded sources and their derived group directories live under the
        # application's private data root. They remain trusted even when the
        # user configured a narrower allow-list for local-path imports.
        allowed_roots = self._allowed_source_roots()
        source = ensure_allowed_path(spec.source, allowed_roots)
        references = [
            ensure_allowed_path(path, allowed_roots) for path in spec.style_references
        ]
        spec.source = source
        spec.style_references = references
        self.registry.get(spec.engine)
        get_color_preset(spec.color_preset)
        get_style_preset(spec.style_preset)
        if not 0.05 <= spec.ink_gamma <= 2.0:
            raise ValueError("Ink gamma must be between 0.05 and 2.0")
        if not 0.0 <= spec.chroma_strength <= 2.5:
            raise ValueError("Chroma strength must be between 0.0 and 2.5")
        return spec, source, references

    def create_shell(self, spec: JobSpec) -> str:
        spec, source, references = self._normalize_spec(spec)

        job_id = str(uuid.uuid4())
        job_dir = self._job_dir(job_id)
        for name in (
            "source",
            "pages",
            "panels",
            "masks",
            "generated",
            "final",
            "output",
            "references",
        ):
            (job_dir / name).mkdir(parents=True, exist_ok=True)
        spec.workspace = job_dir
        manifest = self._manifest(job_id)
        manifest.create_job(job_id, spec)
        manifest.set_job_status(job_id, JobStatus.INGESTING)
        manifest.set_ingest_progress(
            job_id,
            {
                "stage": "ingesting",
                "stage_percent": 0.0,
                "bytes_processed": 0,
                "bytes_total": source.stat().st_size if source.is_file() else 0,
                "discovered_pages": 0,
                "total_pages": 0,
                "latest_message": "正在读取漫画来源",
            },
        )
        (job_dir / "job.json").write_text(
            json.dumps(spec.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._ingest_locks.setdefault(job_id, threading.Lock())
        self._emit(
            "job_status",
            {"status": JobStatus.INGESTING.value, "stage": "ingesting"},
            job_id,
        )
        self._emit(
            "ingest_started",
            {"stage": "ingesting", "message": "正在读取漫画来源"},
            job_id,
        )
        return job_id

    def _set_ingest_progress(
        self,
        job_id: str,
        *,
        stage: str,
        current: int = 0,
        total: int = 0,
        message: str = "",
        current_page: int | None = None,
        current_file: str | None = None,
        bytes_processed: int | None = None,
        bytes_total: int | None = None,
    ) -> None:
        manifest = self._manifest(job_id)
        previous = manifest.ingest_progress(job_id)
        total = max(0, int(total))
        current = max(0, int(current))
        progress = {
            **previous,
            "stage": stage,
            "stage_percent": round(min(100.0, current / total * 100) if total else 0.0, 1),
            "current": current,
            "total": total,
            "latest_message": message or previous.get("latest_message"),
        }
        if current_page is not None:
            progress["current_page"] = current_page
        if current_file is not None:
            progress["current_file"] = current_file
        if bytes_processed is not None:
            progress["bytes_processed"] = max(0, int(bytes_processed))
        if bytes_total is not None:
            progress["bytes_total"] = max(0, int(bytes_total))
        if stage in {"indexing", "masking"}:
            progress["discovered_pages"] = max(
                int(progress.get("discovered_pages") or 0),
                current_page + 1 if current_page is not None else 0,
            )
            if total:
                progress["total_pages"] = total
        manifest.set_ingest_progress(job_id, progress)
        payload = {
            "stage": stage,
            "stage_percent": progress["stage_percent"],
            "current": current,
            "total": total,
            "message": progress["latest_message"],
            "current_page": current_page,
            "current_file": current_file,
            "bytes_processed": progress.get("bytes_processed", 0),
            "bytes_total": progress.get("bytes_total", 0),
        }
        self._emit("ingest_progress", payload, job_id)
        self._emit("job_progress", self.status(job_id)["progress"], job_id)

    def ingest(self, job_id: str) -> str:
        lock = self._ingest_locks.setdefault(job_id, threading.Lock())
        with lock:
            manifest = self._manifest(job_id)
            summary = manifest.summary(job_id)
            if summary["status"] in {
                JobStatus.READY.value,
                JobStatus.QUEUED.value,
                JobStatus.RUNNING.value,
                JobStatus.COMPLETED.value,
            }:
                return job_id
            if summary["status"] not in {
                JobStatus.CREATED.value,
                JobStatus.INGESTING.value,
            }:
                raise ValueError("这个任务当前不能继续建书")
            spec = self._load_spec(job_id)
            allowed_roots = self._allowed_source_roots()
            source = ensure_allowed_path(spec.source, allowed_roots)
            job_dir = self._job_dir(job_id)
            cancel_control = self._cancel_controls.setdefault(job_id, threading.Event())
            try:
                if source.is_file():
                    source_size = source.stat().st_size
                elif source.is_dir():
                    source_size = sum(
                        path.stat().st_size
                        for path in source.rglob("*")
                        if path.is_file()
                    )
                else:
                    source_size = 0

                checkpoint = manifest.ingest_checkpoint(job_id) or {}
                existing_rows = manifest.pages(job_id)
                checkpoint_total = int(checkpoint.get("total") or 0)
                existing_page_paths = [
                    Path(row["source_path"])
                    for row in sorted(existing_rows, key=lambda item: int(item["page_index"]))
                ]
                # If a previous process reached page indexing, the normalized
                # page files and their manifest rows are already a durable
                # checkpoint. Reuse them instead of expanding a large archive
                # again. A partial checkpoint deliberately falls back to the
                # deterministic importer, which will fill the missing rows.
                reuse_indexed_pages = bool(
                    checkpoint.get("stage") in {"indexing", "metadata", "masking", "units", "ready"}
                    and checkpoint_total > 0
                    and len(existing_rows) == checkpoint_total
                    and existing_page_paths
                    and all(path.is_file() for path in existing_page_paths)
                )

                def report(stage: str, current: int, total: int, message: str) -> None:
                    if cancel_control.is_set():
                        raise _IngestCancelled
                    mapped_stage = {
                        "extracting": "expanding_archive",
                        "copying": "writing_pages",
                        "normalizing": "writing_pages",
                        "rendering": "writing_pages",
                    }.get(stage, stage)
                    self._set_ingest_progress(
                        job_id,
                        stage=mapped_stage,
                        current=current,
                        total=total,
                        message=message,
                        current_file=message,
                        bytes_processed=min(source_size, int(source_size * current / total))
                        if source_size and total
                        else None,
                        bytes_total=source_size or None,
                    )

                if reuse_indexed_pages:
                    pages = existing_page_paths
                    self._set_ingest_progress(
                        job_id,
                        stage="validating_members",
                        current=len(pages),
                        total=len(pages),
                        message="已复用建书检查点，正在校验页面成员",
                        bytes_processed=source_size,
                        bytes_total=source_size,
                    )
                else:
                    self._set_ingest_progress(
                        job_id,
                        stage="reading_source",
                        current=0,
                        total=0,
                        message="正在读取漫画来源",
                        bytes_processed=0,
                        bytes_total=source_size,
                    )
                    pages = ingest_book(
                        source,
                        job_dir / "pages",
                        max_archive_members=self.settings.max_archive_members,
                        max_archive_ratio=self.settings.max_archive_ratio,
                        max_archive_uncompressed_bytes=(
                            self.settings.max_archive_uncompressed_mib * 1024 * 1024
                        ),
                        progress=report,
                    )
                if cancel_control.is_set():
                    raise _IngestCancelled
                self._set_ingest_progress(
                    job_id,
                    stage="validating_members",
                    current=0,
                    total=len(pages),
                    message=f"已展开 {len(pages)} 页，正在校验页面成员",
                    bytes_processed=source_size,
                    bytes_total=source_size,
                )
                # Persist an explicit indexing checkpoint before touching any
                # page rows. If the process stops while a large archive is
                # being indexed, startup can distinguish a complete page list
                # from a partial one and reuse only durable rows.
                self._set_ingest_progress(
                    job_id,
                    stage="indexing",
                    current=0,
                    total=len(pages),
                    message="正在建立页面索引",
                    bytes_processed=source_size,
                    bytes_total=source_size,
                )
                copied_references: list[Path] = []
                references = [
                    ensure_allowed_path(path, allowed_roots)
                    for path in spec.style_references
                ]
                for index, reference in enumerate(references):
                    target = (
                        job_dir
                        / "references"
                        / f"reference_{index:04d}{reference.suffix.casefold()}"
                    )
                    shutil.copy2(reference, target)
                    copied_references.append(target)
                spec.style_references = copied_references
                (job_dir / "job.json").write_text(
                    json.dumps(spec.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
                )

                for page_index, page_path in enumerate(pages):
                    if cancel_control.is_set():
                        raise _IngestCancelled
                    self._set_ingest_progress(
                        job_id,
                        stage="metadata",
                        current=page_index,
                        total=len(pages),
                        current_page=page_index,
                        current_file=page_path.name,
                        message=f"正在生成第 {page_index + 1} 页元数据",
                        bytes_processed=source_size,
                        bytes_total=source_size,
                    )
                    checksum, width, height = page_metadata(page_path)
                    page_id = manifest.add_page(
                        job_id, page_index, page_path, checksum, width, height
                    )
                    existing_units = manifest.page_units(page_id)
                    indexed = bool(existing_units) and all(
                        Path(str(unit["source_path"])).is_file()
                        and Path(str(unit["mask_path"])).is_file()
                        for unit in existing_units
                    )
                    if indexed:
                        if manifest.semantic_mask(page_id) is None:
                            self._set_ingest_progress(
                                job_id,
                                stage="masking",
                                current=page_index,
                                total=len(pages),
                                current_page=page_index,
                                current_file=page_path.name,
                                message=f"正在补齐第 {page_index + 1} 页语义遮罩",
                                bytes_processed=source_size,
                                bytes_total=source_size,
                            )
                            self.semantic_page(job_id, page_index)
                        self._set_ingest_progress(
                            job_id,
                            stage="units",
                            current=page_index + 1,
                            total=len(pages),
                            current_page=page_index,
                            current_file=page_path.name,
                            message=f"已复用第 {page_index + 1} 页处理单元",
                            bytes_processed=source_size,
                            bytes_total=source_size,
                        )
                        continue
                    panel_dir = job_dir / "panels" / f"page_{page_index:05d}"
                    mask_dir = job_dir / "masks" / f"page_{page_index:05d}"
                    units = extract_panels(
                        page_path,
                        panel_dir,
                        mode=spec.panel_mode,
                        min_area_ratio=self.settings.panel_min_area_ratio,
                        padding=self.settings.panel_padding,
                    )
                    for unit_index, (box, panel_path) in enumerate(units):
                        with Image.open(panel_path) as panel:
                            rgb = panel.convert("RGB")
                            mask = deterministic_protection_mask(rgb, spec.preserve_text)
                        mask_path = mask_dir / f"panel_{unit_index:04d}.png"
                        mask_path.parent.mkdir(parents=True, exist_ok=True)
                        save_mask(mask, str(mask_path))
                        params_hash = stable_hash(
                            {
                                "source": checksum,
                                "engine": spec.engine,
                                "mode": spec.mode.value,
                                "protection": spec.protection.value,
                                "detail_mode": spec.detail_mode.value,
                                "seed": spec.seed + page_index * 1000 + unit_index,
                                "prompt": spec.prompt,
                                "negative_prompt": spec.negative_prompt,
                                "color_preset": spec.color_preset,
                                "style_preset": spec.style_preset,
                                "preserve_text": spec.preserve_text,
                                "preserve_ink": spec.preserve_ink,
                                "ink_gamma": spec.ink_gamma,
                                "chroma_strength": spec.chroma_strength,
                                "references": [str(path) for path in copied_references],
                            }
                        )
                        manifest.add_unit(
                            page_id,
                            unit_index,
                            box,
                            spec.engine,
                            params_hash,
                            panel_path,
                            mask_path,
                        )
                    # Cache the conservative semantic result while the page is
                    # already in memory. Optional learned engines can replace
                    # this provider later without changing the manifest shape.
                    try:
                        self.semantic_page(job_id, page_index)
                        self._set_ingest_progress(
                            job_id,
                            stage="masking",
                            current=page_index + 1,
                            total=len(pages),
                            current_page=page_index,
                            message=f"已生成第 {page_index + 1} 页保护与语义遮罩",
                            bytes_processed=source_size,
                            bytes_total=source_size,
                        )
                    except Exception as exc:
                        logger.warning(
                            "job=%s page=%s semantic mask fallback: %s",
                            job_id,
                            page_index + 1,
                            exc,
                        )
                        self._emit(
                            "mask_fallback",
                            {
                                "page_index": page_index,
                                "error_code": type(exc).__name__,
                                "message": "语义模型不可用，已使用基础保护遮罩",
                            },
                            job_id,
                        )
                    self._set_ingest_progress(
                        job_id,
                        stage="units",
                        current=page_index + 1,
                        total=len(pages),
                        current_page=page_index,
                        message=f"已建立第 {page_index + 1} 页处理单元",
                        bytes_processed=source_size,
                        bytes_total=source_size,
                    )
                manifest.set_ingest_progress(
                    job_id,
                    {
                        **manifest.ingest_progress(job_id),
                        "stage": "ready",
                        "stage_percent": 100.0,
                        "discovered_pages": len(pages),
                        "total_pages": len(pages),
                        "latest_message": "建书完成，等待进入处理队列",
                    },
                )
                # A pause requested while a large source was being expanded
                # is honored at the next safe checkpoint instead of being
                # lost when the GPU worker starts.
                if self._controls.setdefault(job_id, threading.Event()).is_set():
                    manifest.set_job_status(job_id, JobStatus.PAUSED)
                    self._emit("job_status", {"status": JobStatus.PAUSED.value}, job_id)
                    return job_id
                manifest.set_job_status(job_id, JobStatus.READY)
                self._emit("job_status", {"status": JobStatus.READY.value}, job_id)
                self._emit(
                    "job_ready",
                    {"status": JobStatus.READY.value, "total_pages": len(pages)},
                    job_id,
                )
                return job_id
            except _IngestCancelled:
                manifest.set_job_status(job_id, JobStatus.CANCELLED)
                self._emit("job_status", {"status": JobStatus.CANCELLED.value}, job_id)
                return job_id
            except Exception as exc:
                manifest.set_job_status(job_id, JobStatus.FAILED, str(exc))
                logger.exception("job=%s ingest failed", job_id)
                self._emit(
                    "job_error",
                    {
                        "message": str(exc),
                        "stage": "ingesting",
                        "error_code": type(exc).__name__,
                    },
                    job_id,
                )
                raise

    def create(self, spec: JobSpec) -> str:
        job_id = self.create_shell(spec)
        self.ingest(job_id)
        return job_id

    def _load_spec(self, job_id: str) -> JobSpec:
        data = json.loads((self._job_dir(job_id) / "job.json").read_text(encoding="utf-8"))
        return JobSpec(
            source=Path(data["source"]),
            workspace=Path(data["workspace"]),
            mode=JobMode(data["mode"]),
            engine=data["engine"],
            protection=ProtectionMode(data["protection"]),
            detail_mode=DetailMode(data.get("detail_mode", "strict")),
            output_format=data["output_format"],
            panel_mode=data["panel_mode"],
            seed=int(data["seed"]),
            prompt=data["prompt"],
            negative_prompt=data["negative_prompt"],
            color_preset=data.get("color_preset", "natural"),
            style_preset=data.get("style_preset", "original_ink"),
            preserve_text=bool(data.get("preserve_text", True)),
            preserve_ink=bool(data.get("preserve_ink", True)),
            ink_gamma=float(data.get("ink_gamma", 0.42)),
            chroma_strength=float(data.get("chroma_strength", 1.15)),
            style_references=[Path(item) for item in data["style_references"]],
            max_retries=int(data["max_retries"]),
            adult_fictional_content=bool(data["adult_fictional_content"]),
            metadata=data.get("metadata", {}),
            display_name=data.get("display_name", "未命名漫画"),
        )

    def _identity_guidance(self, job_id: str) -> str:
        """Turn locked identity records into a short, deterministic prompt hint."""
        records = self._manifest(job_id).identities(job_id)
        locked = [item for item in records if item.get("locked") and item.get("color")]
        if not locked:
            return ""
        hints = []
        for item in locked[:40]:
            label = str(item.get("label") or item.get("identity_id") or "object")
            region = str(item.get("region") or "region")
            color = str(item["color"])
            shadow = str(item.get("shadow_color") or "")
            suffix = f" with shadow {shadow}" if shadow else ""
            hints.append(f"{label} {region} uses base color {color}{suffix}")
        return "Locked color records: " + "; ".join(hints)

    def _corrected_unit_mask(self, job_id: str, unit: dict[str, Any]) -> np.ndarray:
        with Image.open(unit["mask_path"]) as mask_image:
            mask = np.asarray(mask_image.convert("L")) > 0
        page_id = int(unit["page_id"])
        corrections = self._manifest(job_id).mask_corrections(page_id)
        return apply_mask_corrections(
            mask,
            corrections,
            offset=(int(unit["x"]), int(unit["y"])),
        )

    def _compose_unit(
        self,
        source: Image.Image,
        generated: Image.Image,
        mask: np.ndarray,
        spec: JobSpec,
    ) -> tuple[Image.Image, np.ndarray]:
        """Compose one generated unit and return its QA protection mask.

        Keeping this decision in one place is important for repair and retry:
        a CPU-only repair must produce the same pixels and QA boundary as a
        normal GPU run, including balanced and generative detail modes.
        """
        source_rgb = source.convert("RGB")
        generated_rgb = generated.convert("RGB")
        if (
            spec.mode == JobMode.STYLE_FULL
            or not spec.preserve_ink
            or spec.detail_mode == DetailMode.GENERATIVE
        ):
            final = composite_protected(source_rgb, generated_rgb, mask)
        elif spec.detail_mode == DetailMode.BALANCED:
            expanded_mask = cv2.dilate(
                mask.astype(np.uint8), np.ones((5, 5), np.uint8)
            ).astype(bool)
            final = preserve_ink_overlay(
                source_rgb,
                generated_rgb,
                protected_mask=expanded_mask,
                gamma=spec.ink_gamma,
            )
            if spec.preserve_ink:
                final = composite_protected(
                    source_rgb,
                    final,
                    ink_detail_mask(source_rgb, threshold=64),
                )
        else:
            final = composite_strict_colorization(
                source_rgb,
                generated_rgb,
                mask,
                chroma_strength=spec.chroma_strength,
            )
        qa_mask = mask
        if spec.preserve_ink and spec.mode != JobMode.STYLE_FULL:
            qa_mask = np.logical_or(mask, ink_detail_mask(source_rgb, threshold=64))
        return final, qa_mask

    def queue(self, job_id: str) -> None:
        # Starting or resuming a job explicitly clears controls left by a
        # previous pause/cancel request. This is the only place that clears
        # them, so a pause arriving at the queue boundary remains effective.
        self._controls.setdefault(job_id, threading.Event()).clear()
        self._cancel_controls.setdefault(job_id, threading.Event()).clear()
        self._set_control_state(job_id, "none", active_request=False, message=None)
        self._manifest(job_id).set_queued(job_id)
        self._emit("job_queued", {"status": JobStatus.QUEUED.value}, job_id)

    def process(self, job_id: str) -> Path:
        # The public API creates a shell and ingests it in the background, but
        # callers that use ProjectManager directly (CLI/tests/recovery tools)
        # still get a safe, synchronous checkpointed ingest before processing.
        current_status = self.status(job_id)["status"]
        if current_status in {JobStatus.CREATED.value, JobStatus.INGESTING.value}:
            self.ingest(job_id)
        with self._process_lock:
            with self._active_job_lock:
                self._active_job_id = job_id
            heartbeat_stop = threading.Event()
            heartbeat = threading.Thread(
                target=self._heartbeat_loop,
                args=(job_id, heartbeat_stop),
                name=f"paneltone-heartbeat-{job_id[:8]}",
                daemon=True,
            )
            heartbeat.start()
            try:
                return self._process_locked(job_id)
            finally:
                heartbeat_stop.set()
                heartbeat.join(timeout=1.0)
                with self._active_job_lock:
                    if self._active_job_id == job_id:
                        self._active_job_id = None
                try:
                    final_status = self._manifest(job_id).summary(job_id)["status"]
                except (KeyError, OSError, ValueError):
                    final_status = None
                if final_status in {
                    JobStatus.PAUSED.value,
                    JobStatus.CANCELLED.value,
                    JobStatus.COMPLETED.value,
                    JobStatus.NEEDS_ATTENTION.value,
                    JobStatus.FAILED.value,
                    JobStatus.WAITING_MODEL.value,
                }:
                    self._set_control_state(
                        job_id,
                        "none",
                        active_request=False,
                        message=(
                            "任务已暂停"
                            if final_status == JobStatus.PAUSED.value
                            else "任务已取消"
                            if final_status == JobStatus.CANCELLED.value
                            else None
                        ),
                    )

    def _interrupt_active_engine(self, job_id: str) -> bool:
        """Interrupt this job's in-flight request, if it owns the worker."""
        with self._active_job_lock:
            if self._active_job_id != job_id:
                return False
        try:
            engine_name = str(self._load_spec(job_id).engine)
            engine = self.registry.get(engine_name)
            interrupt = getattr(engine, "interrupt", None)
            if callable(interrupt):
                result = interrupt()
                logger.info("job=%s engine interrupt requested result=%s", job_id, result)
                return bool(result.get("active", True)) if isinstance(result, dict) else True
        except Exception as exc:
            logger.warning("job=%s engine interrupt request failed: %s", job_id, exc)
        return False

    def _release_engine_if_idle(self, job_id: str) -> None:
        """Release a model pipeline after a pause/cancel when it is safe.

        The model service remains authoritative about whether a request is
        still active.  A busy response is therefore expected when another
        job owns the single GPU worker and is deliberately not treated as an
        error that should stop that job.
        """
        try:
            engine_name = str(self._load_spec(job_id).engine)
            engine = self.registry.get(engine_name)
            release = getattr(engine, "release", None)
            if not callable(release):
                return
            result = release()
            logger.info("job=%s engine idle release result=%s", job_id, result)
            self._emit(
                "model_progress",
                {
                    "status": str(result.get("status", "released"))
                    if isinstance(result, dict)
                    else "released",
                    "state": str(result.get("state", "idle"))
                    if isinstance(result, dict)
                    else "idle",
                },
                job_id,
            )
        except Exception as exc:
            # A pre-maintenance model service may not expose /release yet.
            # Keep the task control operation successful and leave a clear
            # diagnostic for the next maintenance window.
            logger.warning("job=%s engine idle release unavailable: %s", job_id, exc)

    def _heartbeat_loop(self, job_id: str, stop: threading.Event) -> None:
        """Keep active work distinguishable from a crashed worker on restart."""
        while not stop.wait(15.0):
            try:
                self._manifest(job_id).touch_job(job_id)
            except (OSError, KeyError, ValueError):
                return

    def _process_locked(self, job_id: str) -> Path:
        manifest = self._manifest(job_id)
        spec = self._load_spec(job_id)
        engine = self.registry.get(spec.engine)
        engine_health = self.registry.health().get(spec.engine, {})
        if not engine_health.get("ok", False):
            message = str(
                engine_health.get("detail")
                or engine_health.get("error")
                or "模型服务未连接"
            )
            manifest.set_job_status(
                job_id, JobStatus.WAITING_MODEL, message
            )
            logger.warning("job=%s waiting for engine=%s: %s", job_id, spec.engine, message)
            self._emit(
                "job_status",
                {"status": JobStatus.WAITING_MODEL.value, "message": message},
                job_id,
            )
            return self._job_dir(job_id) / "final"
        control = self._controls.setdefault(job_id, threading.Event())
        cancel_control = self._cancel_controls.setdefault(job_id, threading.Event())
        # A pause request can race with the queue worker taking the item out
        # of the pending list. Never clear that request here, otherwise a job
        # that was just paused could acquire the GPU and run anyway.
        if cancel_control.is_set():
            manifest.set_job_status(job_id, JobStatus.CANCELLED)
            self._emit("job_status", {"status": JobStatus.CANCELLED.value}, job_id)
            return self._job_dir(job_id) / "final"
        if control.is_set():
            manifest.set_job_status(job_id, JobStatus.PAUSED)
            self._emit("job_status", {"status": JobStatus.PAUSED.value}, job_id)
            return self._job_dir(job_id) / "final"
        manifest.set_job_status(job_id, JobStatus.RUNNING)
        logger.info(
            "job=%s started engine=%s units=%s",
            job_id,
            spec.engine,
            len(manifest.pending_units(job_id)),
        )
        self._emit("job_status", {"status": JobStatus.RUNNING.value}, job_id)
        failed_units: list[int] = []
        try:
            for unit in manifest.pending_units(job_id):
                if cancel_control.is_set():
                    manifest.set_job_status(job_id, JobStatus.CANCELLED)
                    self._emit("job_status", {"status": JobStatus.CANCELLED.value}, job_id)
                    return self._job_dir(job_id) / "final"
                if control.is_set():
                    manifest.set_job_status(job_id, JobStatus.PAUSED)
                    self._emit("job_status", {"status": JobStatus.PAUSED.value}, job_id)
                    return self._job_dir(job_id) / "final"
                unit_id = int(unit["id"])
                attempts_used = int(unit["attempt"])
                attempts_this_run = 0
                passed = False
                while attempts_this_run < spec.max_retries + 1:
                    manifest.mark_unit_running(unit_id)
                    manifest.set_page_status(int(unit["page_id"]), "running")
                    self._emit(
                        "unit_started",
                        {
                            "page_index": int(unit["page_index"]),
                            "unit_index": int(unit["unit_index"]),
                            "attempt": attempts_used + 1,
                        },
                        job_id,
                    )
                    attempts_used += 1
                    attempts_this_run += 1
                    logger.info(
                        "job=%s page=%s unit=%s attempt=%s started",
                        job_id,
                        int(unit["page_index"]) + 1,
                        int(unit["unit_index"]) + 1,
                        attempts_used,
                    )
                    source_path = Path(unit["source_path"])
                    generated_path = (
                        self._job_dir(job_id)
                        / "generated"
                        / (
                            f"page_{unit['page_index']:05d}_panel_"
                            f"{unit['unit_index']:04d}_a{attempts_used}.png"
                        )
                    )
                    final_path = (
                        self._job_dir(job_id)
                        / "final"
                        / "panels"
                        / f"page_{unit['page_index']:05d}_panel_{unit['unit_index']:04d}.png"
                    )
                    final_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        prompt = build_prompt(
                            spec.mode,
                            spec.color_preset,
                            spec.style_preset,
                            ". ".join(
                                item
                                for item in (spec.prompt, self._identity_guidance(job_id))
                                if item
                            ),
                        )
                        request = EngineRequest(
                            source_path=source_path,
                            output_path=generated_path,
                            mode=spec.mode,
                            seed=spec.seed
                            + int(unit["page_index"]) * 1000
                            + int(unit["unit_index"]),
                            prompt=prompt,
                            negative_prompt=(
                                spec.negative_prompt.strip()
                                or "color bleeding across skin and clothing boundaries, "
                                "painted speech bubbles, distorted text, changed ink lines, "
                                "extra fingers, missing fingers, merged body parts, inconsistent "
                                "hair or eye colors"
                            ),
                            references=spec.style_references,
                            attempt=attempts_used,
                            metadata=spec.metadata,
                        )
                        cached_generated = Path(unit["generated_path"] or "")
                        can_reuse_generated = (
                            attempts_this_run == 1
                            and attempts_used >= 1
                            and cached_generated.is_file()
                        )
                        if can_reuse_generated:
                            generated_path = cached_generated
                        else:
                            self._emit(
                                "model_progress",
                                {
                                    "status": "generating",
                                    "state": "generating",
                                    "page_index": int(unit["page_index"]),
                                    "unit_index": int(unit["unit_index"]),
                                },
                                job_id,
                            )
                            try:
                                engine.generate(request)
                            except EngineInterrupted as exc:
                                interrupted_status = (
                                    JobStatus.CANCELLED
                                    if cancel_control.is_set()
                                    else JobStatus.PAUSED
                                )
                                manifest.reset_running_unit(unit_id, str(exc))
                                manifest.set_job_status(job_id, interrupted_status)
                                logger.info(
                                    "job=%s page=%s unit=%s interrupted status=%s",
                                    job_id,
                                    int(unit["page_index"]) + 1,
                                    int(unit["unit_index"]) + 1,
                                    interrupted_status.value,
                                )
                                self._emit(
                                    "job_status",
                                    {
                                        "status": interrupted_status.value,
                                        "error_code": "model_interrupted",
                                        "message": "模型请求已中断，未计入失败",
                                    },
                                    job_id,
                                )
                                self._release_engine_if_idle(job_id)
                                return self._job_dir(job_id) / "final"
                            self._emit(
                                "model_progress",
                                {
                                    "status": "ready",
                                    "state": "ready",
                                    "page_index": int(unit["page_index"]),
                                    "unit_index": int(unit["unit_index"]),
                                },
                                job_id,
                            )
                        with (
                            Image.open(source_path) as source_image,
                            Image.open(generated_path) as generated_image,
                        ):
                            mask = self._corrected_unit_mask(job_id, unit)
                            final, qa_mask = self._compose_unit(
                                source_image, generated_image, mask, spec
                            )
                            source_rgb = source_image.convert("RGB")
                            final.save(final_path, format="PNG")
                            qa = evaluate(
                                source_rgb,
                                final,
                                qa_mask,
                                line_f1_min=(
                                    self.settings.qa_line_f1_min
                                    if spec.mode != JobMode.STYLE_FULL
                                    else 0.0
                                ),
                                luminance_mae_max=(
                                    self.settings.qa_luminance_mae_max
                                    if spec.detail_mode == DetailMode.STRICT
                                    else 255.0
                                ),
                                pure_black_preservation_min=(
                                    0.0 if spec.mode == JobMode.STYLE_FULL else 0.999
                                ),
                            )
                        manifest.finish_unit(unit_id, generated_path, final_path, qa)
                        if qa.passed:
                            passed = True
                            logger.info(
                                "job=%s page=%s unit=%s passed line_f1=%.4f",
                                job_id,
                                int(unit["page_index"]) + 1,
                                int(unit["unit_index"]) + 1,
                                qa.line_edge_f1,
                            )
                            self._emit(
                                "unit_finished",
                                {
                                    "page_index": int(unit["page_index"]),
                                    "unit_index": int(unit["unit_index"]),
                                    "passed": True,
                                },
                                job_id,
                            )
                            ready_path = self._assemble_page_if_ready(
                                job_id, manifest, int(unit["page_id"])
                            )
                            if ready_path is not None:
                                ready_page_index = int(unit["page_index"])
                                ready_units = manifest.page_units(int(unit["page_id"]))
                                asset_revision = manifest.page_asset_revision(
                                    int(unit["page_id"])
                                ) or str(time.time_ns())
                                asset_query = f"?v={asset_revision}"
                                self._emit(
                                    "page_ready",
                                    {
                                        "page_index": ready_page_index,
                                        "asset_revision": asset_revision,
                                        "status": "qa_passed",
                                        "completed_units": len(ready_units),
                                        "total_units": len(ready_units),
                                        "source_url": (
                                            f"/api/jobs/{job_id}/pages/"
                                            f"{ready_page_index}/source{asset_query}"
                                        ),
                                        "thumbnail_url": (
                                            f"/api/jobs/{job_id}/pages/"
                                            f"{ready_page_index}/thumbnail{asset_query}"
                                        ),
                                        "final_url": (
                                            f"/api/jobs/{job_id}/pages/"
                                            f"{ready_page_index}/final{asset_query}"
                                        ),
                                    },
                                    job_id,
                                )
                            self._emit("job_progress", self.status(job_id)["progress"], job_id)
                            break
                    except Exception as exc:
                        health_after_error = self.registry.health().get(spec.engine, {})
                        if spec.engine != "palette" and not health_after_error.get("ok", False):
                            message = str(
                                health_after_error.get("detail")
                                or health_after_error.get("error")
                                or str(exc)
                            )
                            manifest.defer_unit(unit_id, message)
                            manifest.set_job_status(job_id, JobStatus.WAITING_MODEL, message)
                            logger.warning(
                                "job=%s page=%s unit=%s deferred because engine=%s is "
                                "unavailable: %s",
                                job_id,
                                int(unit["page_index"]) + 1,
                                int(unit["unit_index"]) + 1,
                                spec.engine,
                                message,
                            )
                            self._emit(
                                "job_status",
                                {"status": JobStatus.WAITING_MODEL.value, "message": message},
                                job_id,
                            )
                            return self._job_dir(job_id) / "final"
                        manifest.fail_unit(unit_id, str(exc))
                        logger.warning(
                            "job=%s page=%s unit=%s attempt=%s failed: %s",
                            job_id,
                            int(unit["page_index"]) + 1,
                            int(unit["unit_index"]) + 1,
                            attempts_used,
                            exc,
                        )
                        if attempts_this_run >= spec.max_retries + 1:
                            break
                if not passed:
                    failed_units.append(unit_id)
                    manifest.set_page_status(int(unit["page_id"]), "failed")
                    self._emit(
                        "job_error",
                        {
                            "page_index": int(unit["page_index"]),
                            "unit_index": int(unit["unit_index"]),
                            "message": f"处理单元在 {attempts_used} 次尝试后仍未通过检查",
                            "error_code": "unit_failed_after_retries",
                        },
                        job_id,
                    )

                # Keep a viewable page even when QA rejects one of its units.
                # The preview is never used for export, but it lets the user
                # inspect already-generated color instead of seeing the
                # untouched black-and-white source until the whole book ends.
                page_record = next(
                    item
                    for item in manifest.pages(job_id)
                    if int(item["id"]) == int(unit["page_id"])
                )
                final_available = bool(
                    page_record["output_path"]
                    and Path(page_record["output_path"]).is_file()
                )
                preview_path = (
                    None
                    if final_available
                    else self._assemble_page_preview(
                        job_id, manifest, int(unit["page_id"])
                    )
                )
                if preview_path is not None:
                    page_units = manifest.page_units(int(unit["page_id"]))
                    page_index = int(unit["page_index"])
                    asset_revision = manifest.page_asset_revision(
                        int(unit["page_id"])
                    ) or str(time.time_ns())
                    asset_query = f"?v={asset_revision}"
                    page_has_failure = any(
                        item["status"] in {"failed", "qa_failed"}
                        for item in page_units
                    )
                    preview_thumbnail = (
                        self._job_dir(job_id)
                        / "preview"
                        / "thumbnails"
                        / f"page_{page_index:05d}.jpg"
                    )
                    self._emit(
                        "page_preview_ready",
                        {
                            "page_index": page_index,
                            "asset_revision": asset_revision,
                            "status": "needs_attention" if page_has_failure else "processing",
                            "completed_units": sum(
                                item["status"] == "qa_passed" for item in page_units
                            ),
                            "total_units": len(page_units),
                            "source_url": (
                                f"/api/jobs/{job_id}/pages/{page_index}/source{asset_query}"
                            ),
                            "preview_url": (
                                f"/api/jobs/{job_id}/pages/{page_index}/preview{asset_query}"
                            ),
                            "thumbnail_url": (
                                f"/api/jobs/{job_id}/pages/{page_index}/preview-thumbnail{asset_query}"
                            )
                            if preview_thumbnail.is_file()
                            else None,
                        },
                        job_id,
                    )

            if cancel_control.is_set():
                manifest.set_job_status(job_id, JobStatus.CANCELLED)
                self._emit("job_status", {"status": JobStatus.CANCELLED.value}, job_id)
                return self._job_dir(job_id) / "final"
            if control.is_set():
                manifest.set_job_status(job_id, JobStatus.PAUSED)
                self._emit("job_status", {"status": JobStatus.PAUSED.value}, job_id)
                return self._job_dir(job_id) / "final"

            if failed_units:
                manifest.set_job_status(job_id, JobStatus.NEEDS_ATTENTION)
                logger.warning("job=%s needs attention failed_units=%s", job_id, failed_units)
                self._emit(
                    "job_status",
                    {"status": JobStatus.NEEDS_ATTENTION.value, "failed_units": failed_units},
                    job_id,
                )
                return self._job_dir(job_id) / "final"

            final_pages = self._assemble_pages(job_id, manifest)
            export_path = export_book(
                final_pages, self._job_dir(job_id) / "output", spec.output_format
            )
            manifest.add_event(job_id, "book_exported", {"path": str(export_path)})
            manifest.set_job_status(job_id, JobStatus.COMPLETED)
            logger.info("job=%s completed output=%s", job_id, export_path.name)
            self._emit("job_status", {"status": JobStatus.COMPLETED.value}, job_id)
            return export_path
        except Exception as exc:
            manifest.set_job_status(job_id, JobStatus.FAILED, str(exc))
            logger.exception("job=%s failed unexpectedly", job_id)
            self._emit(
                "job_error",
                {"message": str(exc), "error_code": type(exc).__name__},
                job_id,
            )
            raise

    def _assemble_page_if_ready(
        self, job_id: str, manifest: Manifest, page_id: int, *, force: bool = False
    ) -> Path | None:
        page = next(item for item in manifest.pages(job_id) if int(item["id"]) == page_id)
        if not force and page["output_path"] and Path(page["output_path"]).is_file():
            return Path(page["output_path"])
        units = manifest.page_units(page_id)
        if not units or any(unit["status"] != "qa_passed" for unit in units):
            return None
        with Image.open(page["source_path"]) as source:
            canvas = source.convert("RGB")
        for unit in units:
            with Image.open(unit["final_path"]) as result:
                canvas.paste(result.convert("RGB"), (unit["x"], unit["y"]))
        output = self._job_dir(job_id) / "final" / "pages" / f"page_{page['page_index']:05d}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output, format="PNG")
        thumbnail = (
            self._job_dir(job_id) / "final" / "thumbnails" / f"page_{page['page_index']:05d}.jpg"
        )
        thumbnail.parent.mkdir(parents=True, exist_ok=True)
        preview = canvas.copy()
        preview.thumbnail((240, 320), Image.Resampling.LANCZOS)
        preview.save(thumbnail, format="JPEG", quality=82, optimize=True)
        manifest.finish_page(page_id, output, str(time.time_ns()))
        return output

    def _assemble_page_preview(
        self, job_id: str, manifest: Manifest, page_id: int
    ) -> Path | None:
        """Compose all available unit outputs for an inspectable live preview.

        A page preview may contain a mixture of generated panels and original
        source pixels.  It is deliberately stored outside ``final/pages`` so
        an incomplete or QA-failed page can never be exported as a finished
        book page.
        """
        with self._preview_lock:
            page = next(item for item in manifest.pages(job_id) if int(item["id"]) == page_id)
            units = manifest.page_units(page_id)
            available = [
                unit
                for unit in units
                if unit["final_path"] and Path(unit["final_path"]).is_file()
            ]
            if not available:
                return None
            with Image.open(page["source_path"]) as source:
                canvas = source.convert("RGB")
            for unit in available:
                with Image.open(unit["final_path"]) as result:
                    canvas.paste(result.convert("RGB"), (int(unit["x"]), int(unit["y"])))
            page_index = int(page["page_index"])
            output = (
                self._job_dir(job_id)
                / "preview"
                / "pages"
                / f"page_{page_index:05d}.png"
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(".tmp.png")
            canvas.save(temporary, format="PNG")
            temporary.replace(output)
            thumbnail = (
                self._job_dir(job_id)
                / "preview"
                / "thumbnails"
                / f"page_{page_index:05d}.jpg"
            )
            thumbnail.parent.mkdir(parents=True, exist_ok=True)
            preview = canvas.copy()
            preview.thumbnail((240, 320), Image.Resampling.LANCZOS)
            preview.save(thumbnail, format="JPEG", quality=82, optimize=True)
            # The revision is stored in the manifest before the event is
            # emitted.  The page endpoint and all three derived URLs can then
            # use one cache-busting token even when the page is incomplete.
            manifest.set_page_asset_revision(page_id, str(time.time_ns()))
            return output

    def repair_completed_colorization(
        self,
        job_id: str,
        progress_callback: Callable[[int, int, int], None] | None = None,
    ) -> int:
        """Rebuild completed color pages after a deterministic compositor fix.

        Generated model images are kept intact, so this operation never invokes
        the GPU and can safely repair a paused or completed job in place.
        """
        manifest = self._manifest(job_id)
        spec = self._load_spec(job_id)
        repaired = 0
        pages = manifest.pages(job_id)
        total_pages = len(pages)
        for page_number, page in enumerate(pages, start=1):
            for unit in manifest.page_units(int(page["id"])):
                generated_path = Path(unit["generated_path"] or "")
                source_path = Path(unit["source_path"])
                final_path = Path(unit["final_path"] or "")
                if (
                    not generated_path.is_file()
                    or not source_path.is_file()
                    or not final_path.is_file()
                ):
                    continue
                with (
                    Image.open(source_path) as source_image,
                    Image.open(generated_path) as generated_image,
                ):
                    mask = self._corrected_unit_mask(job_id, unit)
                    final, qa_mask = self._compose_unit(
                        source_image, generated_image, mask, spec
                    )
                    source_rgb = source_image.convert("RGB")
                    final_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = final_path.with_suffix(".repair.png")
                    final.save(temporary, format="PNG")
                    temporary.replace(final_path)
                    qa = evaluate(
                        source_rgb,
                        final,
                        qa_mask,
                        line_f1_min=self.settings.qa_line_f1_min,
                        luminance_mae_max=(
                            self.settings.qa_luminance_mae_max
                            if spec.detail_mode == DetailMode.STRICT
                            else 255.0
                        ),
                        pure_black_preservation_min=(
                            0.0 if spec.mode == JobMode.STYLE_FULL else 0.999
                        ),
                    )
                    if qa.passed:
                        # A repaired unit may have been marked failed by an
                        # older compositor. Re-accept it only after the new
                        # deterministic pixels pass the normal QA gate.
                        manifest.accept_repaired_unit(
                            int(unit["id"]), generated_path, final_path, qa
                        )
                    else:
                        # Preserve an existing QA decision when recomposition
                        # cannot improve it. The asset itself is still useful
                        # for inspection and does not erase the failure cause.
                        manifest.replace_unit_output(
                            int(unit["id"]), generated_path, final_path
                        )
                    repaired += 1
            ready_path = self._assemble_page_if_ready(
                job_id, manifest, int(page["id"]), force=True
            )
            if ready_path is None or not ready_path.is_file():
                preview_path = self._assemble_page_preview(job_id, manifest, int(page["id"]))
                if preview_path is not None:
                    page_id = int(page["id"])
                    revision = manifest.page_asset_revision(page_id)
                    asset_query = f"?v={revision}" if revision else ""
                    preview_thumbnail = (
                        self._job_dir(job_id)
                        / "preview"
                        / "thumbnails"
                        / f"page_{int(page['page_index']):05d}.jpg"
                    )
                    self._emit(
                        "page_preview_ready",
                        {
                            "page_index": int(page["page_index"]),
                            "asset_revision": revision,
                            "status": "needs_attention",
                            "completed_units": sum(
                                item["status"] == "qa_passed"
                                for item in manifest.page_units(int(page["id"]))
                            ),
                            "total_units": len(manifest.page_units(int(page["id"]))),
                            "source_url": (
                                f"/api/jobs/{job_id}/pages/{int(page['page_index'])}/source"
                                f"{asset_query}"
                            ),
                            "preview_url": (
                                f"/api/jobs/{job_id}/pages/{int(page['page_index'])}/preview"
                                f"{asset_query}"
                            ),
                            "thumbnail_url": (
                                f"/api/jobs/{job_id}/pages/{int(page['page_index'])}/preview-thumbnail"
                                f"{asset_query}"
                            )
                            if preview_thumbnail.is_file()
                            else None,
                        },
                        job_id,
                    )
            if progress_callback is not None:
                progress_callback(int(page["page_index"]), page_number, total_pages)
        return repaired

    def _assemble_pages(self, job_id: str, manifest: Manifest) -> list[Path]:
        output_pages: list[Path] = []
        for page in manifest.pages(job_id):
            page_id = int(page["id"])
            with Image.open(page["source_path"]) as source:
                canvas = source.convert("RGB")
            with manifest.connect() as connection:
                units = connection.execute(
                    "SELECT * FROM units WHERE page_id=? ORDER BY unit_index", (page_id,)
                ).fetchall()
            for unit in units:
                if unit["status"] != "qa_passed" or not unit["final_path"]:
                    raise RuntimeError(f"Page {page['page_index']} has unfinished units")
                with Image.open(unit["final_path"]) as result:
                    canvas.paste(result.convert("RGB"), (unit["x"], unit["y"]))
            output = (
                self._job_dir(job_id) / "final" / "pages" / f"page_{page['page_index']:05d}.png"
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(output, format="PNG")
            manifest.finish_page(page_id, output, str(time.time_ns()))
            output_pages.append(output)
        return output_pages

    def pause(self, job_id: str) -> dict[str, Any]:
        summary = self.status(job_id)
        if summary["status"] in {
            JobStatus.PAUSED.value,
            JobStatus.COMPLETED.value,
            JobStatus.CANCELLED.value,
            JobStatus.NEEDS_ATTENTION.value,
            JobStatus.FAILED.value,
        }:
            return self.control_state(job_id) or {
                "action": "none",
                "active_request": False,
                "message": "当前任务没有可暂停的运行请求",
            }
        self._controls.setdefault(job_id, threading.Event()).set()
        with self._active_job_lock:
            active_for_job = self._active_job_id == job_id
        requested_at = datetime.now(UTC).isoformat()
        deadline_at = datetime.fromtimestamp(
            datetime.now(UTC).timestamp() + 15, UTC
        ).isoformat()
        state = self._set_control_state(
            job_id,
            "pause_requested",
            requested_at=requested_at,
            deadline_at=deadline_at,
            active_request=active_for_job,
            message="暂停中，正在中断当前模型请求" if active_for_job else "暂停中",
        )
        interrupted = self._interrupt_active_engine(job_id)
        if not active_for_job:
            reset = self._manifest(job_id).reset_running_units(
                job_id, "任务暂停时清理遗留运行单元"
            )
            if reset:
                logger.info("job=%s reset stale running units=%s on pause", job_id, reset)
            self._manifest(job_id).set_job_status(job_id, JobStatus.PAUSED)
            self._emit("job_status", {"status": JobStatus.PAUSED.value}, job_id)
        if not interrupted:
            self._release_engine_if_idle(job_id)
        if active_for_job:
            threading.Thread(
                target=self._watch_control,
                args=(job_id, "pause"),
                name=f"paneltone-pause-watch-{job_id[:8]}",
                daemon=True,
            ).start()
        return state

    def cancel(self, job_id: str) -> dict[str, Any]:
        summary = self.status(job_id)
        if summary["status"] in {
            JobStatus.CANCELLED.value,
            JobStatus.COMPLETED.value,
            JobStatus.FAILED.value,
        }:
            return self.control_state(job_id) or {
                "action": "none",
                "active_request": False,
                "message": "当前任务没有可取消的运行请求",
            }
        self._cancel_controls.setdefault(job_id, threading.Event()).set()
        with self._active_job_lock:
            active_for_job = self._active_job_id == job_id
        requested_at = datetime.now(UTC).isoformat()
        deadline_at = datetime.fromtimestamp(
            datetime.now(UTC).timestamp() + 15, UTC
        ).isoformat()
        state = self._set_control_state(
            job_id,
            "cancel_requested",
            requested_at=requested_at,
            deadline_at=deadline_at,
            active_request=active_for_job,
            message="取消中，正在中断当前模型请求" if active_for_job else "取消中",
        )
        interrupted = self._interrupt_active_engine(job_id)
        if not active_for_job:
            reset = self._manifest(job_id).reset_running_units(
                job_id, "任务取消时清理遗留运行单元"
            )
            if reset:
                logger.info("job=%s reset stale running units=%s on cancel", job_id, reset)
            self._manifest(job_id).set_job_status(job_id, JobStatus.CANCELLED)
            self._emit("job_status", {"status": JobStatus.CANCELLED.value}, job_id)
        if not interrupted:
            self._release_engine_if_idle(job_id)
        if active_for_job:
            threading.Thread(
                target=self._watch_control,
                args=(job_id, "cancel"),
                name=f"paneltone-cancel-watch-{job_id[:8]}",
                daemon=True,
            ).start()
        return state

    def set_status(self, job_id: str, status: JobStatus) -> None:
        self._manifest(job_id).set_job_status(job_id, status)
        self._emit("job_status", {"status": status.value}, job_id)

    def retry_page(self, job_id: str, page_index: int) -> int:
        page = next(
            (
                item
                for item in self._manifest(job_id).pages(job_id)
                if int(item["page_index"]) == page_index
            ),
            None,
        )
        if page is None:
            raise KeyError(f"Unknown page: {page_index}")
        if page["output_path"]:
            Path(page["output_path"]).unlink(missing_ok=True)
        count = self._manifest(job_id).retry_page(job_id, page_index)
        self._emit(
            "job_status",
            {"status": JobStatus.READY.value, "retry_page": page_index},
            job_id,
        )
        return count

    def rename(self, job_id: str, display_name: str) -> None:
        display_name = display_name.strip()
        if not display_name or len(display_name) > 120:
            raise ValueError("书名长度必须在 1 到 120 个字符之间")
        manifest = self._manifest(job_id)
        manifest.rename_job(job_id, display_name)
        job_json = self._job_dir(job_id) / "job.json"
        data = json.loads(job_json.read_text(encoding="utf-8"))
        data["display_name"] = display_name
        job_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def duplicate(self, job_id: str) -> str:
        spec = self._load_spec(job_id)
        spec.workspace = self.settings.data_root
        spec.display_name = f"{spec.display_name} 副本"
        return self.create(spec)

    def status(self, job_id: str) -> dict[str, Any]:
        manifest = self._manifest(job_id)
        result = manifest.summary(job_id)
        result["progress"] = manifest.progress(job_id)
        control = self.control_state(job_id)
        if control is not None:
            result["progress"]["control_state"] = control
        return result

    def semantic_page(self, job_id: str, page_index: int) -> SemanticMaskResult:
        key = (job_id, page_index)
        if key in self._semantic_cache:
            return self._semantic_cache[key]
        page = next(
            (
                item
                for item in self._manifest(job_id).pages(job_id)
                if int(item["page_index"]) == page_index
            ),
            None,
        )
        if page is None:
            raise KeyError(f"Unknown page: {page_index}")
        with Image.open(page["source_path"]) as source:
            result = self.semantic_engine.segment(source)
        semantic_dir = self._job_dir(job_id) / "masks" / "semantic" / f"page_{page_index:05d}"
        semantic_dir.mkdir(parents=True, exist_ok=True)
        for name, mask in result.masks.items():
            save_mask(mask, str(semantic_dir / f"{name}.png"))
        confidence_path = semantic_dir / "confidence.png"
        uncertain_path = semantic_dir / "uncertain.png"
        Image.fromarray(np.clip(result.confidence * 255, 0, 255).astype(np.uint8), mode="L").save(
            confidence_path, format="PNG"
        )
        save_mask(result.uncertain, str(uncertain_path))
        self._manifest(job_id).save_semantic_mask(
            int(page["id"]),
            provider=result.provider,
            version=result.version,
            descriptor=semantic_descriptor(self.semantic_engine, result),
            confidence_path=confidence_path,
            uncertain_path=uncertain_path,
        )
        self._semantic_cache[key] = result
        while len(self._semantic_cache) > self._semantic_cache_limit:
            self._semantic_cache.pop(next(iter(self._semantic_cache)))
        return result

    def list_jobs(self) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        for directory in sorted(self.settings.data_root.iterdir(), reverse=True):
            if not directory.is_dir() or not (directory / "manifest.sqlite").is_file():
                continue
            try:
                jobs.append(self.status(directory.name))
            except Exception:
                continue
        return jobs

    def ingesting_jobs(self) -> list[str]:
        return [
            job["id"]
            for job in self.list_jobs()
            if job.get("status") in {JobStatus.CREATED.value, JobStatus.INGESTING.value}
        ]
