from __future__ import annotations

import io
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

from .color import (
    apply_render_profile,
    apply_vibrance_grade,
    classify_source_page,
    composite_geometry_locked_colorization,
    composite_protected,
    composite_reference_locked_colorization,
    composite_strict_colorization,
    geometry_barrier_mask,
    image_sha256,
    is_already_colorized,
    validated_colorization_protection,
)
from .color_state import (
    RENDERER_VERSION,
    ColorBookState,
    ColorStateStore,
    RegionColorPlan,
    RenderEvidence,
    artifact_digest,
    build_color_book_state,
    build_identity_graph,
    build_segment_graph,
    make_region_color_plan,
)
from .config import Settings, ensure_allowed_path
from .engines import EngineInterrupted, EngineRegistry, EngineRequest
from .export import export_book
from .hashing import stable_hash
from .ingest import ingest_book, page_metadata
from .manifest import Manifest
from .masks import (
    apply_mask_corrections,
    deterministic_protection_mask,
    save_mask,
)
from .material_render import (
    MaterialPlan,
    evaluate_material_render,
    render_material_flats,
)
from .material_render import (
    protection_mask as material_protection_mask,
)
from .models import DetailMode, JobMode, JobSpec, JobStatus, ProtectionMode
from .panels import extract_panels
from .presets import build_prompt, get_color_preset, get_style_preset, render_profile
from .qa import evaluate
from .reading_assets import ReadingAssetCache
from .reference_library import ColorReferenceLibrary
from .semantic import (
    ConservativeSemanticMaskEngine,
    SemanticMaskEngine,
    SemanticMaskResult,
    configured_semantic_engine,
    semantic_descriptor,
)

logger = logging.getLogger("paneltone.job")


def _balloons_with_detected_text(
    balloon_mask: np.ndarray,
    text_mask: np.ndarray,
    source_gray: np.ndarray,
) -> np.ndarray:
    """Keep semantic balloon components that contain detected dialogue text.

    Manga segmentation can mistake bright body parts or props for a balloon.
    Restoring an entire false-positive component makes that region pure white
    in an otherwise coloured page. Dialogue balloons are retained as filled
    protection only when their component contains (or immediately surrounds)
    semantic text; source geometry remains protected independently.
    """
    if balloon_mask.shape != text_mask.shape or balloon_mask.shape != source_gray.shape:
        raise ValueError("balloon, text and source arrays must have identical dimensions")
    count, labels = cv2.connectedComponents(balloon_mask.astype(np.uint8), connectivity=8)
    nearby_text = cv2.dilate(
        text_mask.astype(np.uint8), np.ones((7, 7), np.uint8)
    ).astype(bool)
    accepted = np.zeros_like(balloon_mask, dtype=bool)
    for component in range(1, count):
        region = labels == component
        dark_ratio = float(np.mean(source_gray[region] <= 100))
        text_ratio = float(np.mean(nearby_text[region]))
        # Filled balloon proposals should contain actual dark glyphs. Bright
        # body parts and props that both detectors weakly mistake for dialogue
        # have virtually no source-dark content and are rejected here.
        if dark_ratio >= 0.02 and text_ratio >= 0.03:
            accepted |= region
    return accepted


class _IngestCancelled(Exception):
    """Internal checkpoint signal used to stop a large import safely."""


class DisplayAssetPending(FileNotFoundError):
    """A completed page is waiting for its prebuilt display asset."""


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
        self.reading_cache = ReadingAssetCache(
            settings.reading_cache_root or settings.data_root / ".reading-cache"
        )
        self._display_backfill_lock = threading.Lock()
        self._display_backfill_pending: set[tuple[str, int, str]] = set()
        self.semantic_engine = semantic_engine or configured_semantic_engine(settings.model_root)
        self._semantic_fallback = ConservativeSemanticMaskEngine()
        self._semantic_cache: dict[tuple[str, int], SemanticMaskResult] = {}
        # Semantic masks are persisted on disk and can be large for a full
        # book. Keep only a tiny hot cache so a 300-page run cannot retain one
        # full-resolution mask set per page in the worker process.
        self._semantic_cache_limit = 2
        self.reference_library = ColorReferenceLibrary(settings.data_root)
        self._palette_cache: dict[
            tuple[tuple[str, int, int], ...], list[tuple[float, float, float]]
        ] = {}
        # One immutable book context is reused by normal processing, retries,
        # and CPU repair.  The cache is process-local; the JSON sidecars are
        # the durable source for later runs and audit tooling.
        self._color_context_cache: dict[str, dict[str, Any]] = {}
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
        try:
            return Manifest(self._job_dir(job_id) / "manifest.sqlite", initialize=False)
        except FileNotFoundError as exc:
            raise KeyError("Unknown job") from exc

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
        references = [ensure_allowed_path(path, allowed_roots) for path in spec.style_references]
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
        manifest = Manifest(job_dir / "manifest.sqlite")
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
                JobStatus.RUNNING.value,
                JobStatus.COMPLETED.value,
            }:
                return job_id
            if summary["status"] == JobStatus.QUEUED.value and manifest.pages(job_id):
                return job_id
            if summary["status"] not in {
                JobStatus.CREATED.value,
                JobStatus.INGESTING.value,
                JobStatus.QUEUED.value,
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
                        path.stat().st_size for path in source.rglob("*") if path.is_file()
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
                    ensure_allowed_path(path, allowed_roots) for path in spec.style_references
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
        hints = self.reference_library.identity_hints(records)
        if not hints:
            return ""
        return "Locked color records: " + "; ".join(hints)

    def _book_color_anchor_paths(self, job_id: str, spec: JobSpec) -> list[Path]:
        """Return deliberately curated book anchors for the color state.

        Automatic retrieval remains available to the candidate engine, but it
        is never promoted to a locked identity palette merely because a page
        happened to be visually similar.  Only explicit references and the
        reviewed ``references/curated`` directory enter the durable state.
        """
        curated_root = self._job_dir(job_id) / "references" / "curated"
        curated = sorted(
            path
            for path in curated_root.glob("*")
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        paths: list[Path] = []
        for path in (*spec.style_references, *curated):
            resolved = Path(path).resolve()
            if resolved.is_file() and resolved not in paths:
                paths.append(resolved)
        return paths[:64]

    def _prepare_color_context(
        self, job_id: str, manifest: Manifest, spec: JobSpec
    ) -> dict[str, Any]:
        """Build or reuse the durable book-level color planning context.

        The context is intentionally conservative: identity observations are
        empty until a detector or curator provides evidence, while existing
        manifest identity records can still contribute explicit palette slots.
        This makes uncertainty visible instead of turning a weak visual match
        into a false character lock.
        """
        pages = manifest.pages(job_id)
        source_snapshot: dict[int, str] = {}
        for page in pages:
            source_path = Path(page["source_path"])
            with Image.open(source_path) as image:
                source_snapshot[int(page["page_index"])] = image_sha256(image)
        snapshot_hash = stable_hash(source_snapshot)
        cached = self._color_context_cache.get(job_id)
        if cached is not None and cached["source_snapshot_hash"] == snapshot_hash:
            return cached

        store = ColorStateStore(self._job_dir(job_id))
        segment_graphs: dict[int, Any] = {}
        for page in pages:
            page_index = int(page["page_index"])
            with Image.open(page["source_path"]) as image:
                segment = build_segment_graph(
                    image.convert("RGB"), job_id=job_id, page_index=page_index
                )
            segment_graphs[page_index] = segment
            store.write_page("segments", page_index, segment)
        segment_graph_hash = stable_hash(
            {str(index): graph.analysis_hash for index, graph in segment_graphs.items()}
        )
        identity_graph = build_identity_graph(job_id)
        anchor_paths = self._book_color_anchor_paths(job_id, spec)
        observed_anchors = self._reference_palette(anchor_paths)
        state = build_color_book_state(
            job_id,
            source_snapshot,
            identity_graph,
            segment_graph_hash,
            identities=manifest.identities(job_id),
            observed_palette_anchors=observed_anchors,
        )
        store.write(
            "source_snapshot.json",
            {"job_id": job_id, "snapshot_hash": snapshot_hash, "pages": source_snapshot},
        )
        store.write("identity_graph.json", identity_graph)
        store.write("color_book_state.json", state)
        context = {
            "source_snapshot_hash": snapshot_hash,
            "source_snapshot": source_snapshot,
            "segment_graphs": segment_graphs,
            "segment_graph_hash": segment_graph_hash,
            "state": state,
            "store": store,
        }
        self._color_context_cache[job_id] = context
        return context

    def _page_color_plan(
        self,
        context: dict[str, Any],
        *,
        source_path: Path,
        page_index: int,
        references: list[Path],
        page_seed: int,
    ) -> RegionColorPlan:
        """Create the page plan consumed by both inference and CPU repair."""
        with Image.open(source_path) as image:
            source_hash = image_sha256(image)
        segment = context["segment_graphs"][page_index]
        plan = make_region_color_plan(
            context["state"],
            segment,
            page_source_hash=source_hash,
            page_index=page_index,
            references=references,
            page_seed=page_seed,
        )
        context["store"].write_page("plans", page_index, plan)
        return plan

    def _combined_palette_anchors(
        self,
        state: ColorBookState, references: list[Path]
    ) -> list[tuple[float, float, float]]:
        """Combine durable book vocabulary with page-local evidence."""
        anchors = list(state.observed_palette_anchors)
        # Automatic page references are local suggestions, while durable book
        # anchors remain first. Keep a stable ordering and avoid duplicates;
        # the compositor still treats these as soft hints, never as masks.
        for anchor in self._reference_palette(references):
            if anchor not in anchors:
                anchors.append(anchor)
        return anchors

    def _write_render_evidence(
        self,
        context: dict[str, Any] | None,
        *,
        job_id: str,
        page_index: int,
        unit_index: int,
        attempt: int,
        plan: RegionColorPlan | None,
        spec: JobSpec,
        source_hash: str,
        generated_hash: str | None,
        final_hash: str | None,
        references: list[Path],
        qa: Any,
        render_seed: int | None = None,
        engine_metadata: dict[str, Any] | None = None,
    ) -> None:
        if context is None:
            return
        plan_hash = plan.plan_hash if plan is not None else "bypass"
        reference_hashes: list[str] = []
        for path in references:
            try:
                with Image.open(path) as reference_image:
                    reference_hashes.append(image_sha256(reference_image))
            except (OSError, ValueError):
                continue
        evidence_metadata = [
            ("mode", spec.mode.value),
            ("color_preset", spec.color_preset),
            ("style_preset", spec.style_preset),
        ]
        if engine_metadata:
            evidence_metadata.extend(
                (str(key), str(value)) for key, value in sorted(engine_metadata.items())
            )
        evidence = RenderEvidence(
            version=1,
            job_id=job_id,
            page_index=page_index,
            unit_index=unit_index,
            plan_hash=plan_hash,
            model_id=str(spec.engine),
            model_revision=(
                str(engine_metadata.get("model_revision"))
                if engine_metadata and engine_metadata.get("model_revision") is not None
                else None
            ),
            renderer_version=RENDERER_VERSION,
            source_hash=source_hash,
            generated_hash=generated_hash,
            final_hash=final_hash,
            reference_hashes=tuple(reference_hashes),
            seed=(
                render_seed
                if render_seed is not None
                else plan.seed
                if plan is not None
                else spec.seed
            ),
            metadata=tuple(evidence_metadata),
            qa_hash=artifact_digest(qa.to_json_dict()) if qa is not None else None,
        )
        context["store"].write(
            f"evidence/page_{page_index:05d}_unit_{unit_index:04d}_a{attempt}.json",
            evidence,
        )

    def _reference_paths(self, job_id: str, source_path: Path, spec: JobSpec) -> list[Path]:
        """Combine explicit references with local automatic retrieval.

        Cobra benefits from several same-work visual examples. Existing
        engines keep their explicit references only; the candidate engine gets
        local retrieval as an additive input and never sends the index itself.
        """
        explicit = [path.resolve() for path in spec.style_references if path.is_file()]
        if spec.engine != "cobra-candidate":
            return explicit
        # A reviewed book-level anchor set is stronger than automatically
        # retrieved final pages.  Automatic retrieval is useful for discovering
        # candidates, but a previous page can itself contain dropped colour and
        # would otherwise teach Cobra the same defect on every later page.
        curated_root = self._job_dir(job_id) / "references" / "curated"
        curated = sorted(
            path
            for path in curated_root.glob("*")
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        if curated:
            paths: list[Path] = []
            for path in (*explicit, *curated):
                if path.is_file() and path not in paths:
                    paths.append(path)
            return paths[:12]
        # Reference colours must be scoped to the current book. The previous
        # global index could select a page from another manga, which made a
        # character's palette drift even when the candidate itself was stable.
        scope_root = self._job_dir(job_id) / "final" / "pages"
        matches = self.reference_library.retrieve(
            source_path,
            limit=12,
            exclude=(source_path, *explicit),
            scope_root=scope_root,
            use_payload_cache=True,
            min_colour_coverage=0.12,
        )
        if not matches:
            matches = self.reference_library.retrieve(
                source_path,
                limit=6,
                exclude=(source_path, *explicit),
                scope_root=scope_root,
                use_payload_cache=True,
            )
        # A brand-new book may have no completed colour page yet. Keep the
        # candidate usable in that case, but make the fallback small and
        # explicit; once the first page is committed, same-book references are
        # always preferred.
        if not matches and not explicit:
            matches = self.reference_library.retrieve(
                source_path,
                limit=12,
                exclude=(source_path,),
                use_payload_cache=True,
                min_colour_coverage=0.20,
            )
        paths: list[Path] = []
        for path in (*explicit, *(match.path for match in matches)):
            if path.is_file() and path not in paths:
                paths.append(path)
        return paths[:64]

    def _reference_palette(
        self, paths: list[Path] | None
    ) -> list[tuple[float, float, float]]:
        if not paths:
            return []
        signature: list[tuple[str, int, int]] = []
        for path in paths:
            try:
                stat = path.stat()
            except OSError:
                continue
            signature.append((str(path), stat.st_mtime_ns, stat.st_size))
        key = tuple(signature)
        if key not in self._palette_cache:
            self._palette_cache[key] = self.reference_library.palette_anchors(paths)
        return self._palette_cache[key]

    @staticmethod
    def _model_tier(spec: JobSpec, source_classification: Any) -> str:
        """Return an auditable tier label for each unit request."""
        if source_classification is not None and source_classification.source_passthrough:
            return "bypass"
        if spec.engine == "cobra-candidate":
            return "cobra-candidate"
        if spec.engine == "palette":
            return "deterministic"
        return "stable-production"

    def _corrected_unit_masks(
        self, job_id: str, unit: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        # Recompute deterministic protection so a compositor repair is not
        # forced to reuse an older, overly broad bubble/text mask.
        spec = self._load_spec(job_id)
        with Image.open(unit["source_path"]) as source_image:
            source_rgb = source_image.convert("RGB")
            source_gray = np.asarray(source_rgb.convert("L"))
            mask = deterministic_protection_mask(
                source_rgb, preserve_text=spec.preserve_text
            )
        page_id = int(unit["page_id"])
        corrections = self._manifest(job_id).mask_corrections(page_id)
        corrected = apply_mask_corrections(
            mask,
            corrections,
            offset=(int(unit["x"]), int(unit["y"])),
        )
        try:
            semantic = self.semantic_page(job_id, int(unit["page_index"]))
            x, y = int(unit["x"]), int(unit["y"])
            height, width = corrected.shape
            if (
                x < 0
                or y < 0
                or width <= 0
                or height <= 0
                or y + height > semantic.confidence.shape[0]
                or x + width > semantic.confidence.shape[1]
            ):
                raise ValueError("semantic crop falls outside the page")
            protected = np.zeros_like(corrected)
            text_page_mask = semantic.masks.get("text")
            text_crop = (
                text_page_mask[y : y + height, x : x + width]
                if text_page_mask is not None
                else np.zeros_like(corrected)
            )
            if text_crop.shape != corrected.shape:
                raise ValueError("semantic text crop shape mismatch")
            for name in ("text", "bubbles", "borders", "ink"):
                page_mask = semantic.masks.get(name)
                if page_mask is not None:
                    crop = page_mask[y : y + height, x : x + width]
                    if crop.shape != corrected.shape:
                        raise ValueError(f"semantic {name} crop shape mismatch")
                    if name == "text":
                        # Preserve glyph strokes, not every bright region in a
                        # coarse text proposal.  Balloon interiors are handled
                        # component-wise below.
                        crop = np.logical_and(crop, source_gray <= 160)
                    elif name == "bubbles":
                        crop = _balloons_with_detected_text(
                            crop, text_crop, source_gray
                        )
                    elif name == "ink":
                        # The semantic fallback's broad ink proposal can include
                        # halftoned skin or clothing.  Only genuinely dark source
                        # pixels are core ink that must be restored exactly;
                        # otherwise whole body regions stay white after colour
                        # transfer even though the candidate contains valid skin.
                        crop = np.logical_and(crop, source_gray <= 64)
                    protected |= crop
            uncertain = semantic.uncertain[y : y + height, x : x + width]
            if uncertain.shape != corrected.shape:
                raise ValueError("semantic uncertainty crop shape mismatch")
            corrected |= protected
        except (KeyError, OSError, ValueError, RuntimeError) as exc:
            logger.warning(
                "semantic unit mask unavailable job=%s page=%s unit=%s: %s",
                job_id,
                int(unit["page_index"]) + 1,
                int(unit["unit_index"]) + 1,
                exc,
            )
            uncertain = np.zeros_like(corrected)
        return corrected, uncertain

    def _compose_unit(
        self,
        source: Image.Image,
        generated: Image.Image,
        mask: np.ndarray,
        spec: JobSpec,
        uncertain_mask: np.ndarray | None = None,
        reference_paths: list[Path] | None = None,
        palette_anchors: list[tuple[float, float, float]] | None = None,
        allow_legacy_candidate: bool = False,
        material_plan: MaterialPlan | None = None,
    ) -> tuple[Image.Image, np.ndarray]:
        """Compose one generated unit and return its QA protection mask.

        Keeping this decision in one place is important for repair and retry:
        a CPU-only repair must produce the same pixels and QA boundary as a
        normal GPU run, including balanced and generative detail modes.
        """
        if material_plan is not None:
            # An explicit reviewed plan is the only new-renderer entry point
            # Legacy/unresolved book sidecars are NOT silently promoted here
            if spec.mode != JobMode.COLORIZE:
                raise ValueError("material rendering is only supported for COLORIZE")
            material_plan.validate(source)
            if mask.shape != material_plan.protected.shape:
                raise ValueError("runtime protection does not match reviewed material plan")
            reviewed_protection = material_protection_mask(source, material_plan.protected)
            if np.any(mask.astype(bool) & ~reviewed_protection):
                # A newly broad semantic mask must not erase reviewed colour
                # and then hide the resulting gray islands as "protected"
                raise ValueError("runtime protection changed; material plan requires review")
            final = render_material_flats(source, material_plan)
            report = evaluate_material_render(source, final, material_plan)
            if not report["passed"]:
                raise ValueError("material plan requires review: " + ", ".join(report["reasons"]))
            return final, reviewed_protection

        source_rgb = source.convert("RGB")
        generated_rgb = self._render_color_candidate(
            source,
            generated,
            spec,
            allow_legacy_aspect=allow_legacy_candidate,
        )
        if spec.engine == "cobra-candidate":
            # Cobra's reference-guided render is deliberately conservative.
            # Enrich its existing material colours after the selected style
            # profile, while keeping hue, value, geometry and neutral paper
            # unchanged. Pastel/noir profiles remain restrained because their
            # earlier profile grade supplies less chroma to this bounded lift.
            generated_rgb = apply_vibrance_grade(generated_rgb)
        effective_chroma = spec.chroma_strength * float(
            render_profile(spec.color_preset, spec.style_preset)["chroma_multiplier"]
        )
        compositor = None
        if spec.mode != JobMode.STYLE_FULL:
            if is_already_colorized(source_rgb):
                # Do not let a colour cover/credits page turn into a different
                # scene when the generation service invents structure.
                final = source_rgb
            else:
                # Cobra is a reference-guided line-art colourizer: its material,
                # shadow and highlight rendering is the actual product, not a
                # palette mask. Restore reviewed source ink over that render and
                # let geometry QA reject shifted or invented structure. Generic
                # generators remain colour-hint providers because their spatial
                # output is not reliably aligned to manga line art.
                if spec.engine == "cobra-candidate":
                    compositor = composite_strict_colorization
                elif spec.mode == JobMode.COLORIZE and spec.engine != "palette":
                    compositor = composite_reference_locked_colorization
                else:
                    compositor = composite_geometry_locked_colorization
                if compositor is composite_geometry_locked_colorization:
                    mask = validated_colorization_protection(source_rgb, generated_rgb, mask)
                final = compositor(
                    source_rgb,
                    generated_rgb,
                    mask,
                    chroma_strength=effective_chroma,
                    ink_core_threshold=64,
                    **(
                        {"ink_edge_threshold": 128}
                        if compositor is composite_strict_colorization
                        else {}
                    ),
                    **(
                        {
                            "palette_anchors": palette_anchors
                            if palette_anchors is not None
                            else self._reference_palette(reference_paths),
                            "palette_strength": 0.28,
                        }
                        if compositor is composite_reference_locked_colorization
                        else {}
                    ),
                )
        else:
            final = composite_protected(source_rgb, generated_rgb, mask)

        # For all geometry-locked modes, uncertainty is diagnostic only.
        # Returning those areas to source luminance caused large gray islands
        # in otherwise valid generated colour. Explicit STYLE_FULL remains the
        # only mode allowed to import generated structure into the result.
        if spec.mode == JobMode.STYLE_FULL and uncertain_mask is not None and uncertain_mask.any():
            final = composite_protected(source_rgb, final, uncertain_mask)
        qa_mask = mask
        if spec.mode != JobMode.STYLE_FULL:
            # The legacy barrier compositor restores its one-pixel geometry
            # guard, while the reference compositor keeps that guard only as a
            # diagnostic boundary to avoid gray islands on small panels.
            if compositor in {
                composite_reference_locked_colorization,
                composite_strict_colorization,
            }:
                # The reference compositor intentionally keeps the barrier as
                # a diagnostic boundary, not a broad source-pixel restore. A
                # broad restore recreates the gray islands this route removes.
                qa_mask = mask
            else:
                qa_mask = geometry_barrier_mask(source_rgb, mask, ink_core_threshold=64)
        return final, qa_mask

    @staticmethod
    def _render_generated(generated: Image.Image, spec: JobSpec) -> Image.Image:
        """Return the exact style-graded image used as the colour base."""
        profile = render_profile(spec.color_preset, spec.style_preset)
        return apply_render_profile(
            generated.convert("RGB"),
            saturation=float(profile["saturation"]),
            contrast=float(profile["contrast"]),
            hue_shift=float(profile["hue_shift"]),
        )

    @staticmethod
    def _render_color_candidate(
        source: Image.Image,
        generated: Image.Image,
        spec: JobSpec,
        *,
        allow_legacy_aspect: bool = False,
    ) -> Image.Image:
        """Align a model colour candidate without accepting new warped output.

        Small stride-rounding differences are expected.  A material aspect
        mismatch is rejected for every new generation because inverse resizing
        creates colour ghosts.  Transactional repair may explicitly accept old
        malformed candidates so the source-guided compositor can salvage their
        low-frequency colour without deleting historical evidence.
        """
        rendered = ProjectManager._render_generated(generated, spec)
        if rendered.size == source.size:
            return rendered
        source_aspect = source.width / source.height
        candidate_aspect = rendered.width / rendered.height
        aspect_error = abs(candidate_aspect / source_aspect - 1.0)
        if aspect_error > 0.025 and not allow_legacy_aspect:
            raise ValueError(
                "model colour candidate aspect ratio differs from source "
                f"({rendered.width}x{rendered.height} vs {source.width}x{source.height})"
            )
        if aspect_error > 0.025:
            logger.warning(
                "repairing legacy warped colour candidate from %sx%s to %sx%s",
                rendered.width,
                rendered.height,
                source.width,
                source.height,
            )
        logger.info(
            "resizing model colour candidate from %sx%s to source canvas %sx%s",
            rendered.width,
            rendered.height,
            source.width,
            source.height,
        )
        return rendered.resize(source.size, Image.Resampling.LANCZOS)

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
        needs_ingest = current_status in {
            JobStatus.CREATED.value,
            JobStatus.INGESTING.value,
        }
        if current_status == JobStatus.QUEUED.value:
            needs_ingest = not self._manifest(job_id).pages(job_id)
        if needs_ingest:
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
        if not manifest.pages(job_id):
            raise RuntimeError("建书完成后没有可处理的页面")
        color_context = (
            self._prepare_color_context(job_id, manifest, spec)
            if spec.mode == JobMode.COLORIZE
            else None
        )
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
                source_path = Path(unit["source_path"])
                page_source_path = Path(unit.get("page_source_path") or source_path)
                source_classification = None
                if spec.mode == JobMode.COLORIZE:
                    with Image.open(source_path) as source_image:
                        source_classification = classify_source_page(source_image)
                reference_paths = (
                    self._reference_paths(job_id, page_source_path, spec)
                    if spec.mode == JobMode.COLORIZE
                    else []
                )
                color_plan = (
                    self._page_color_plan(
                        color_context,
                        source_path=page_source_path,
                        page_index=int(unit["page_index"]),
                        references=reference_paths,
                        page_seed=spec.seed,
                    )
                    if color_context is not None
                    else None
                )
                palette_anchors = (
                    self._combined_palette_anchors(color_context["state"], reference_paths)
                    if color_context is not None
                    else None
                )
                if not source_classification or not source_classification.source_passthrough:
                    engine_health = self.registry.health().get(spec.engine, {})
                    if not engine_health.get("ok", False):
                        message = str(
                            engine_health.get("detail")
                            or engine_health.get("error")
                            or "模型服务未连接"
                        )
                        manifest.defer_unit(unit_id, message)
                        manifest.set_job_status(job_id, JobStatus.WAITING_MODEL, message)
                        logger.warning(
                            "job=%s waiting for engine=%s before page=%s unit=%s: %s",
                            job_id,
                            spec.engine,
                            int(unit["page_index"]) + 1,
                            int(unit["unit_index"]) + 1,
                            message,
                        )
                        self._emit(
                            "job_status",
                            {"status": JobStatus.WAITING_MODEL.value, "message": message},
                            job_id,
                        )
                        return self._job_dir(job_id) / "final"
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
                    engine_metadata: dict[str, Any] = {}
                    try:
                        if source_classification and source_classification.source_passthrough:
                            with Image.open(source_path) as source_image:
                                source_rgb = source_image.convert("RGB")
                                source_rgb.save(final_path, format="PNG")
                            qa = evaluate(
                                source_rgb,
                                source_rgb,
                                np.zeros((source_rgb.height, source_rgb.width), dtype=bool),
                                line_f1_min=0.0,
                                luminance_mae_max=255.0,
                                pure_black_preservation_min=0.0,
                                source_class=source_classification.source_class,
                                source_passthrough=True,
                                bypass_reason=source_classification.bypass_reason,
                                source_sha256=image_sha256(source_rgb),
                                final_sha256=image_sha256(source_rgb),
                            )
                            self._write_render_evidence(
                                color_context,
                                job_id=job_id,
                                page_index=int(unit["page_index"]),
                                unit_index=int(unit["unit_index"]),
                                attempt=attempts_used,
                                plan=color_plan,
                                spec=spec,
                                source_hash=image_sha256(source_rgb),
                                generated_hash=None,
                                final_hash=image_sha256(source_rgb),
                                references=reference_paths,
                                qa=qa,
                            )
                            manifest.finish_bypassed_unit(unit_id, final_path, qa)
                            passed = qa.passed
                            self._emit(
                                "unit_finished",
                                {
                                    "page_index": int(unit["page_index"]),
                                    "unit_index": int(unit["unit_index"]),
                                    "passed": passed,
                                    "source_class": source_classification.source_class,
                                    "source_passthrough": True,
                                },
                                job_id,
                            )
                            if passed:
                                self._assemble_page_if_ready(job_id, manifest, int(unit["page_id"]))
                                self._emit("job_progress", self.status(job_id)["progress"], job_id)
                            break
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
                        render_settings = render_profile(spec.color_preset, spec.style_preset)
                        request = EngineRequest(
                            source_path=source_path,
                            output_path=generated_path,
                            mode=spec.mode,
                            # Engines add ``attempt - 1`` exactly once when
                            # materializing the request.  Keeping this seed
                            # as the page/unit base prevents retry paths from
                            # accidentally applying the attempt offset twice.
                            seed=(
                                color_plan.seed + int(unit["unit_index"])
                                if color_plan is not None
                                else spec.seed
                                + int(unit["page_index"]) * 1000
                                + int(unit["unit_index"])
                            ),
                            prompt=prompt,
                            negative_prompt=(
                                spec.negative_prompt.strip()
                                or "color bleeding across skin and clothing boundaries, "
                                "painted speech bubbles, distorted text, changed ink lines, "
                                "extra fingers, missing fingers, merged body parts, inconsistent "
                                "hair or eye colors"
                            ),
                            references=reference_paths,
                            attempt=attempts_used,
                            metadata={
                                **spec.metadata,
                                "style_preset": spec.style_preset,
                                "color_preset": spec.color_preset,
                                "detail_mode": spec.detail_mode.value,
                                "model_tier": self._model_tier(spec, source_classification),
                                **render_settings,
                                # FLUX.2 Klein's colourization baseline is
                                # deliberately fixed at the official 4-step,
                                # guidance=1.0 setting.  A style profile may
                                # still shape prompt/chroma, but it must not
                                # turn colourize into an unbounded redraw.
                                "guidance_scale": (
                                    1.0
                                    if spec.mode == JobMode.COLORIZE
                                    else render_settings["guidance_scale"]
                                ),
                                "num_inference_steps": (
                                    4
                                    if spec.mode == JobMode.COLORIZE
                                    else render_settings["num_inference_steps"]
                                ),
                                "cobra_top_k": 6,
                                "cobra_steps": 10,
                                **(
                                    {
                                        "color_plan_hash": color_plan.plan_hash,
                                        "palette_state_hash": color_plan.palette_state_hash,
                                        "identity_graph_hash": color_plan.identity_graph_hash,
                                        "segment_graph_hash": color_plan.segment_graph_hash,
                                        "renderer_version": RENDERER_VERSION,
                                        "color_plan_seed": color_plan.seed,
                                    }
                                    if color_plan is not None
                                    else {}
                                ),
                            },
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
                                engine_result = engine.generate(request)
                                engine_metadata = dict(engine_result.engine_metadata or {})
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
                            mask, uncertain_mask = self._corrected_unit_masks(job_id, unit)
                            if spec.engine == "cobra-candidate":
                                final, qa_mask = self._compose_unit(
                                    source_image,
                                    generated_image,
                                    mask,
                                    spec,
                                    uncertain_mask,
                                    reference_paths,
                                    palette_anchors,
                                )
                            else:
                                final, qa_mask = self._compose_unit(
                                    source_image,
                                    generated_image,
                                    mask,
                                    spec,
                                    uncertain_mask,
                                    reference_paths,
                                    palette_anchors,
                                )
                            source_rgb = source_image.convert("RGB")
                            final.save(final_path, format="PNG")
                            qa_generated = (
                                source_rgb
                                if spec.mode == JobMode.COLORIZE
                                and is_already_colorized(source_rgb)
                                else self._render_color_candidate(
                                    source_image, generated_image, spec
                                )
                            )
                            qa = evaluate(
                                source_rgb,
                                final,
                                qa_mask,
                                generated=qa_generated,
                                line_f1_min=(
                                    self.settings.qa_line_f1_min
                                    if spec.mode != JobMode.STYLE_FULL
                                    else 0.0
                                ),
                                luminance_mae_max=(
                                    self.settings.qa_luminance_mae_max
                                    if spec.detail_mode == DetailMode.STRICT
                                    and spec.mode != JobMode.COLORIZE
                                    else 255.0
                                ),
                                pure_black_preservation_min=(
                                    0.0 if spec.mode == JobMode.STYLE_FULL else 0.999
                                ),
                                source_class=(
                                    source_classification.source_class
                                    if source_classification
                                    else "line_art"
                                ),
                                source_sha256=image_sha256(source_rgb),
                                final_sha256=image_sha256(final),
                                geometry_locked=spec.mode != JobMode.STYLE_FULL,
                                color_retention_min=0.75,
                                chroma_edge_alignment_min=(
                                    0.0 if spec.engine == "palette" else 0.995
                                ),
                            )
                            self._write_render_evidence(
                                color_context,
                                job_id=job_id,
                                page_index=int(unit["page_index"]),
                                unit_index=int(unit["unit_index"]),
                                attempt=attempts_used,
                                plan=color_plan,
                                spec=spec,
                                source_hash=image_sha256(source_rgb),
                                generated_hash=image_sha256(generated_image),
                                final_hash=image_sha256(final),
                                references=reference_paths,
                                qa=qa,
                                render_seed=request.seed,
                                engine_metadata=engine_metadata,
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
                    page_record["output_path"] and Path(page_record["output_path"]).is_file()
                )
                preview_path = (
                    None
                    if final_available
                    else self._assemble_page_preview(job_id, manifest, int(unit["page_id"]))
                )
                if preview_path is not None:
                    page_units = manifest.page_units(int(unit["page_id"]))
                    page_index = int(unit["page_index"])
                    asset_revision = manifest.page_asset_revision(int(unit["page_id"])) or str(
                        time.time_ns()
                    )
                    asset_query = f"?v={asset_revision}"
                    page_has_failure = any(
                        item["status"] in {"failed", "qa_failed"} for item in page_units
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
        valid_units = self._paste_units_safely(canvas, units, job_id=job_id)
        if valid_units != len(units):
            logger.error(
                "page assembly rejected job=%s page=%s valid_units=%s total_units=%s",
                job_id,
                page["page_index"],
                valid_units,
                len(units),
            )
            return None
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
        self.prebuild_display_assets(job_id, int(page["page_index"]))
        return output

    def _assemble_page_preview(self, job_id: str, manifest: Manifest, page_id: int) -> Path | None:
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
                dict(unit)
                for unit in units
                if unit["final_path"] and Path(unit["final_path"]).is_file()
            ]
            if not available:
                return None
            with Image.open(page["source_path"]) as source:
                canvas = source.convert("RGB")
            self._paste_units_safely(canvas, available, job_id=job_id)
            page_index = int(page["page_index"])
            output = self._job_dir(job_id) / "preview" / "pages" / f"page_{page_index:05d}.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(".tmp.png")
            canvas.save(temporary, format="PNG")
            temporary.replace(output)
            thumbnail = (
                self._job_dir(job_id) / "preview" / "thumbnails" / f"page_{page_index:05d}.jpg"
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

    def display_asset(self, job_id: str, page_index: int, variant: str) -> Path:
        """Create a browser-sized WebP without altering the archival PNG."""
        if variant not in {"source", "final"}:
            raise ValueError("Unknown display asset variant")
        manifest = self._manifest(job_id)
        page = manifest.page_by_index(job_id, page_index)
        source_path = Path(
            page["source_path"] if variant == "source" else page["output_path"] or ""
        )
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        output = self._job_dir(job_id) / "display" / variant / f"page_{page_index:05d}.webp"
        # Existing files must not wait behind a whole-book backfill/export lock.
        if output.is_file() and output.stat().st_mtime_ns >= source_path.stat().st_mtime_ns:
            return output
        with self._preview_lock:
            if output.is_file() and output.stat().st_mtime_ns >= source_path.stat().st_mtime_ns:
                return output
            if page["status"] == "qa_passed":
                manifest.add_event(
                    job_id,
                    "display_asset_missing",
                    {"page_index": page_index, "variant": variant},
                )
                raise DisplayAssetPending("display asset is being prepared")
            return self._write_display_asset(source_path, output)

    def quality_candidate_asset(self, job_id: str, page_index: int) -> Path:
        """Return a prebuilt, isolated review candidate without touching live output."""
        self._manifest(job_id).page_by_index(job_id, page_index)
        path = (
            self._job_dir(job_id)
            / "quality-candidates"
            / "cobra"
            / "display"
            / f"page_{page_index:05d}.webp"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def reading_asset(self, job_id: str, page_index: int, variant: str, size: str) -> Path:
        if variant not in {"source", "final"}:
            raise ValueError("Unknown reading variant")
        page = self._manifest(job_id).page_by_index(job_id, page_index)
        source = Path(page["source_path"] if variant == "source" else page["output_path"] or "")
        if not source.is_file():
            raise FileNotFoundError(source)
        output = self.reading_cache.request(source, size)
        if output is None:
            raise DisplayAssetPending("reading preview is being prepared")
        return output

    def _write_display_asset(self, source_path: Path, output: Path) -> Path:
        """Encode one display asset atomically with bounded CPU work."""
        with Image.open(source_path) as image:
            display = image.convert("RGB")
            display.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
            encoded = b""
            for quality in (88, 85, 82):
                buffer = io.BytesIO()
                display.save(buffer, format="WEBP", quality=quality, method=4)
                encoded = buffer.getvalue()
                if len(encoded) <= 700 * 1024:
                    break
            while len(encoded) > 900 * 1024 and max(display.size) > 960:
                next_long_edge = max(960, int(max(display.size) * 0.88))
                resized = image.convert("RGB")
                resized.thumbnail((next_long_edge, next_long_edge), Image.Resampling.LANCZOS)
                display = resized
                buffer = io.BytesIO()
                display.save(buffer, format="WEBP", quality=82, method=4)
                encoded = buffer.getvalue()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp.webp")
        temporary.write_bytes(encoded)
        temporary.replace(output)
        return output

    @staticmethod
    def _link_or_copy(source: Path, destination: Path) -> Path:
        """Reuse an immutable same-disk artifact without encoding it again."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            destination.hardlink_to(source)
        except OSError:
            shutil.copy2(source, destination)
        return destination

    @staticmethod
    def _replace_with_windows_retry(source: Path, destination: Path) -> None:
        """Atomically move a path, tolerating short-lived Windows file locks.

        Antivirus and indexing services can briefly retain a handle to a newly
        written export archive.  Windows then rejects the directory rename even
        though the destination has already been moved into the rollback area.
        Retrying only ``PermissionError`` keeps the transaction atomic while
        still surfacing real path, space and filesystem errors immediately.
        """
        for attempt in range(20):
            try:
                source.replace(destination)
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.2 * (attempt + 1))

    @staticmethod
    def _path_entry_exists(path: Path) -> bool:
        """Return whether a directory entry exists, including dangling junctions."""
        try:
            path.lstat()
        except FileNotFoundError:
            return False
        return True

    def _publish_staged_output_files(
        self, staging_output: Path, live_output: Path
    ) -> None:
        """Publish completed archives without renaming their containing directory.

        A completed download can be held briefly by Windows Defender or an
        indexer.  Renaming the whole staging directory then fails even though
        moving the individual closed archive is permitted.  The live directory
        stays absent until the old output has been moved to the rollback area;
        files are moved atomically one by one into a newly created directory.
        """
        if self._path_entry_exists(live_output):
            raise FileExistsError(f"Live output already exists: {live_output}")
        artifacts = sorted(staging_output.iterdir())
        if not artifacts or any(not artifact.is_file() for artifact in artifacts):
            raise RuntimeError("Staged output must contain only completed files")
        live_output.mkdir(parents=False, exist_ok=False)
        moved: list[Path] = []
        try:
            for artifact in artifacts:
                destination = live_output / artifact.name
                self._replace_with_windows_retry(artifact, destination)
                moved.append(destination)
        except Exception:
            for artifact in reversed(moved):
                self._replace_with_windows_retry(
                    artifact, staging_output / artifact.name
                )
            live_output.rmdir()
            raise
        staging_output.rmdir()

    def _withdraw_published_output_files(
        self, live_output: Path, failed_output: Path
    ) -> None:
        """Move a newly published output back out before restoring its backup."""
        failed_output.mkdir(parents=True, exist_ok=False)
        for artifact in sorted(live_output.iterdir()):
            if not artifact.is_file():
                raise RuntimeError("Published output unexpectedly contains a directory")
            self._replace_with_windows_retry(artifact, failed_output / artifact.name)
        live_output.rmdir()

    def prebuild_display_assets(
        self,
        job_id: str,
        page_index: int,
        variants: tuple[str, ...] = ("source", "final"),
    ) -> None:
        """Materialize browser assets before a completed page is exposed."""
        manifest = self._manifest(job_id)
        page = manifest.page_by_index(job_id, page_index)
        with self._preview_lock:
            for variant in variants:
                if variant not in {"source", "final"}:
                    raise ValueError("Unknown display asset variant")
                source_path = Path(
                    page["source_path"] if variant == "source" else page["output_path"] or ""
                )
                if not source_path.is_file():
                    continue
                for size in ("preview", "reader"):
                    self.reading_cache.request(source_path, size)
                output = self._job_dir(job_id) / "display" / variant / f"page_{page_index:05d}.webp"
                if output.is_file() and output.stat().st_mtime_ns >= source_path.stat().st_mtime_ns:
                    continue
                self._write_display_asset(source_path, output)

    def prebuild_display_assets_for_job(
        self,
        job_id: str,
        variants: tuple[str, ...] = ("source", "final"),
    ) -> int:
        """Build all available display assets without touching manifest rows.

        This is the explicit maintenance/backfill entry point used before a
        completed book is exposed through the tunnel.  It never encodes in a
        request thread and skips a final asset until its page has an output.
        """
        if any(variant not in {"source", "final"} for variant in variants):
            raise ValueError("Unknown display asset variant")
        manifest = self._manifest(job_id)
        pages = manifest.pages(job_id)
        written = 0
        with self._preview_lock:
            for page in pages:
                page_index = int(page["page_index"])
                for variant in variants:
                    source_value = (
                        page["source_path"] if variant == "source" else page["output_path"] or ""
                    )
                    source_path = Path(source_value)
                    if not source_path.is_file():
                        continue
                    output = (
                        self._job_dir(job_id) / "display" / variant / f"page_{page_index:05d}.webp"
                    )
                    if (
                        output.is_file()
                        and output.stat().st_mtime_ns >= source_path.stat().st_mtime_ns
                    ):
                        continue
                    self._write_display_asset(source_path, output)
                    written += 1
        return written

    def schedule_display_asset(self, job_id: str, page_index: int, variant: str) -> None:
        """Queue one missing completed-page asset for low-priority backfill."""
        if variant not in {"source", "final"}:
            raise ValueError("Unknown display asset variant")
        key = (job_id, page_index, variant)
        with self._display_backfill_lock:
            if key in self._display_backfill_pending:
                return
            self._display_backfill_pending.add(key)

        def build() -> None:
            try:
                self.prebuild_display_assets(job_id, page_index, (variant,))
            except (OSError, KeyError, ValueError) as exc:
                logger.warning(
                    "display asset backfill failed job=%s page=%s variant=%s: %s",
                    job_id,
                    page_index,
                    variant,
                    exc,
                )
            finally:
                with self._display_backfill_lock:
                    self._display_backfill_pending.discard(key)

        threading.Thread(
            target=build,
            name=f"paneltone-display-{job_id[:8]}-{page_index}-{variant}",
            daemon=True,
        ).start()

    def _paste_units_safely(
        self, canvas: Image.Image, units: list[dict[str, Any]], *, job_id: str
    ) -> int:
        """Paste panel results without clipping, resizing, or double painting.

        PIL silently clips a paste that runs outside the page.  That behavior
        can turn one bad crop into a cross-panel colour block, so every unit is
        checked first.  Overlapping padded crops keep the first writer's pixels
        and leave the source image visible for any invalid unit.
        """
        page_width, page_height = canvas.size
        occupied = np.zeros((page_height, page_width), dtype=bool)
        valid = 0
        for unit in units:
            try:
                x, y = int(unit["x"]), int(unit["y"])
                width, height = int(unit["width"]), int(unit["height"])
                final_path = Path(unit["final_path"] or "")
                if (
                    width <= 0
                    or height <= 0
                    or x < 0
                    or y < 0
                    or x + width > page_width
                    or y + height > page_height
                    or not final_path.is_file()
                ):
                    raise ValueError("panel bounds or result path invalid")
                with Image.open(final_path) as result:
                    result_rgb = result.convert("RGB")
                    if result_rgb.size != (width, height):
                        raise ValueError(f"result size {result_rgb.size} != crop {(width, height)}")
                    target = occupied[y : y + height, x : x + width]
                    write_mask = ~target
                    if not write_mask.all():
                        logger.warning(
                            "overlapping panel crop kept first result job=%s page=%s unit=%s",
                            job_id,
                            unit.get("page_index"),
                            unit.get("unit_index"),
                        )
                    if write_mask.any():
                        canvas.paste(
                            result_rgb,
                            (x, y),
                            Image.fromarray((write_mask.astype(np.uint8) * 255), mode="L"),
                        )
                        target[write_mask] = True
                valid += 1
            except (OSError, TypeError, ValueError, KeyError) as exc:
                logger.error(
                    "invalid panel result left as source job=%s page=%s unit=%s: %s",
                    job_id,
                    unit.get("page_index"),
                    unit.get("unit_index"),
                    exc,
                )
        return valid

    def repair_completed_colorization(
        self,
        job_id: str,
        progress_callback: Callable[[int, int, int], None] | None = None,
    ) -> int:
        """Stage, verify and atomically publish a deterministic CPU repair.

        Generated model images are kept intact. No live panel, page, manifest
        row or export is changed until every staged unit and assembled page has
        passed validation.
        """
        manifest = self._manifest(job_id)
        spec = self._load_spec(job_id)
        pages = manifest.pages(job_id)
        total_pages = len(pages)
        if not pages:
            return 0
        color_context = (
            self._prepare_color_context(job_id, manifest, spec)
            if spec.mode == JobMode.COLORIZE
            else None
        )
        job_dir = self._job_dir(job_id)
        live_final = job_dir / "final"
        live_output = job_dir / "output"
        live_display = job_dir / "display"

        token = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
        staging_root = job_dir / f".repair-staging-{token}"
        staging_final = staging_root / "final"
        staging_output = staging_root / "output"
        staging_display = staging_root / "display"
        backup_root = job_dir / "repair-backups" / token
        staged_units: list[dict[str, Any]] = []
        staged_pages: list[dict[str, Any]] = []
        current_page_index: int | None = None
        current_unit_index: int | None = None
        committed = False
        try:
            for page_number, page in enumerate(pages, start=1):
                page_index = int(page["page_index"])
                current_page_index = page_index
                page_units: list[dict[str, Any]] = []
                for stored_unit in manifest.page_units(int(page["id"])):
                    unit = {**stored_unit, "page_index": page_index}
                    current_unit_index = int(unit["unit_index"])
                    generated_path = (
                        Path(unit["generated_path"]) if unit["generated_path"] else None
                    )
                    source_path = Path(unit["source_path"])
                    if not source_path.is_file():
                        raise FileNotFoundError(
                            f"Missing repair input for page {page_index + 1}, "
                            f"unit {int(unit['unit_index']) + 1}"
                        )
                    reference_paths = (
                        self._reference_paths(job_id, Path(page["source_path"]), spec)
                        if spec.mode == JobMode.COLORIZE
                        else []
                    )
                    color_plan = (
                        self._page_color_plan(
                            color_context,
                            source_path=Path(page["source_path"]),
                            page_index=page_index,
                            references=reference_paths,
                            page_seed=spec.seed,
                        )
                        if color_context is not None
                        else None
                    )
                    palette_anchors = (
                        self._combined_palette_anchors(
                            color_context["state"], reference_paths
                        )
                        if color_context is not None
                        else None
                    )
                    final_name = f"page_{page_index:05d}_panel_{int(unit['unit_index']):04d}.png"
                    staged_path = staging_final / "panels" / final_name
                    staged_path.parent.mkdir(parents=True, exist_ok=True)
                    with Image.open(source_path) as source_image:
                        source_rgb = source_image.convert("RGB")
                        classification = (
                            classify_source_page(source_image)
                            if spec.mode == JobMode.COLORIZE
                            else None
                        )
                        if classification and classification.source_passthrough:
                            final = source_rgb.copy()
                            qa = evaluate(
                                source_rgb,
                                final,
                                np.zeros((source_rgb.height, source_rgb.width), dtype=bool),
                                line_f1_min=0.0,
                                luminance_mae_max=255.0,
                                pure_black_preservation_min=0.0,
                                source_class=classification.source_class,
                                source_passthrough=True,
                                bypass_reason=classification.bypass_reason,
                                source_sha256=image_sha256(source_rgb),
                                final_sha256=image_sha256(final),
                            )
                            self._write_render_evidence(
                                color_context,
                                job_id=job_id,
                                page_index=page_index,
                                unit_index=int(unit["unit_index"]),
                                attempt=int(unit["attempt"]),
                                plan=color_plan,
                                spec=spec,
                                source_hash=image_sha256(source_rgb),
                                generated_hash=None,
                                final_hash=image_sha256(final),
                                references=reference_paths,
                                qa=qa,
                            )
                        else:
                            if generated_path is None or not generated_path.is_file():
                                raise FileNotFoundError(
                                    f"Missing generated repair input for page {page_index + 1}, "
                                    f"unit {int(unit['unit_index']) + 1}"
                                )
                            with Image.open(generated_path) as generated_image:
                                mask, uncertain_mask = self._corrected_unit_masks(job_id, unit)
                                source_aspect = source_image.width / source_image.height
                                generated_aspect = generated_image.width / generated_image.height
                                legacy_warped_candidate = (
                                    abs(generated_aspect / source_aspect - 1.0) > 0.025
                                )
                                if spec.engine == "cobra-candidate":
                                    final, qa_mask = self._compose_unit(
                                        source_image,
                                        generated_image,
                                        mask,
                                        spec,
                                        uncertain_mask,
                                        reference_paths,
                                        palette_anchors,
                                        True,
                                    )
                                else:
                                    final, qa_mask = self._compose_unit(
                                        source_image,
                                        generated_image,
                                        mask,
                                        spec,
                                        uncertain_mask,
                                        reference_paths,
                                        palette_anchors,
                                        True,
                                    )
                                qa = evaluate(
                                    source_rgb,
                                    final,
                                    qa_mask,
                                    generated=self._render_color_candidate(
                                        source_image,
                                        generated_image,
                                        spec,
                                        allow_legacy_aspect=True,
                                    ),
                                    line_f1_min=(
                                        self.settings.qa_line_f1_min
                                        if spec.mode != JobMode.STYLE_FULL
                                        else 0.0
                                    ),
                                    luminance_mae_max=(
                                        self.settings.qa_luminance_mae_max
                                        if spec.detail_mode == DetailMode.STRICT
                                        and spec.mode != JobMode.COLORIZE
                                        else 255.0
                                    ),
                                    pure_black_preservation_min=(
                                        0.0 if spec.mode == JobMode.STYLE_FULL else 0.999
                                    ),
                                    source_class=(
                                        classification.source_class
                                        if classification
                                        else "line_art"
                                    ),
                                    source_sha256=image_sha256(source_rgb),
                                    final_sha256=image_sha256(final),
                                    geometry_locked=spec.mode != JobMode.STYLE_FULL,
                                    color_retention_min=(
                                        0.70 if legacy_warped_candidate else 0.75
                                    ),
                                    color_dropout_tiles_max=(
                                        4 if legacy_warped_candidate else 0
                                    ),
                                    neutral_island_ratio_max=(
                                        0.30 if legacy_warped_candidate else 0.08
                                    ),
                                    largest_neutral_island_ratio_max=(
                                        0.30 if legacy_warped_candidate else 0.03
                                    ),
                                    chroma_edge_alignment_min=(
                                        0.0 if spec.engine == "palette" else 0.995
                                    ),
                                )
                                self._write_render_evidence(
                                    color_context,
                                    job_id=job_id,
                                    page_index=page_index,
                                    unit_index=int(unit["unit_index"]),
                                    attempt=int(unit["attempt"]),
                                    plan=color_plan,
                                    spec=spec,
                                    source_hash=image_sha256(source_rgb),
                                    generated_hash=image_sha256(generated_image),
                                    final_hash=image_sha256(final),
                                    references=reference_paths,
                                    qa=qa,
                                    render_seed=(
                                        color_plan.seed + int(unit["unit_index"])
                                        if color_plan is not None
                                        else spec.seed
                                    ),
                                )
                        if not qa.passed:
                            raise RuntimeError(
                                f"Repair QA failed for page {page_index + 1}: "
                                f"{', '.join(qa.reasons)} "
                                "(chroma_edge_alignment="
                                f"{qa.chroma_edge_alignment:.6f}, "
                                f"color_retention={qa.color_retention_ratio:.6f}, "
                                f"color_dropout_tiles={qa.color_dropout_tiles}, "
                                f"neutral_island={qa.neutral_island_ratio:.6f}, "
                                "largest_neutral_island="
                                f"{qa.largest_neutral_island_ratio:.6f})"
                            )
                        final.save(staged_path, format="PNG")
                    live_path = live_final / "panels" / final_name
                    staged_unit = {
                        "id": int(unit["id"]),
                        "generated_path": generated_path,
                        "final_path": live_path,
                        "qa": qa,
                    }
                    staged_units.append(staged_unit)
                    page_units.append({**unit, "final_path": str(staged_path)})

                with Image.open(page["source_path"]) as source:
                    canvas = source.convert("RGB")
                if self._paste_units_safely(canvas, page_units, job_id=job_id) != len(page_units):
                    raise RuntimeError(f"Staged page assembly failed for page {page_index + 1}")
                staged_page_path = staging_final / "pages" / f"page_{page_index:05d}.png"
                staged_page_path.parent.mkdir(parents=True, exist_ok=True)
                single_full_page_unit = (
                    len(page_units) == 1
                    and int(page_units[0]["x"]) == 0
                    and int(page_units[0]["y"]) == 0
                    and int(page_units[0]["width"]) == canvas.width
                    and int(page_units[0]["height"]) == canvas.height
                )
                if single_full_page_unit:
                    # Page-mode jobs store the same full-resolution pixels as a
                    # panel and as an assembled page. Link the verified staged
                    # panel instead of PNG-compressing and storing it twice.
                    self._link_or_copy(Path(page_units[0]["final_path"]), staged_page_path)
                else:
                    canvas.save(staged_page_path, format="PNG")
                thumbnail = staging_final / "thumbnails" / f"page_{page_index:05d}.jpg"
                thumbnail.parent.mkdir(parents=True, exist_ok=True)
                preview = canvas.copy()
                preview.thumbnail((240, 320), Image.Resampling.LANCZOS)
                preview.save(thumbnail, format="JPEG", quality=82, optimize=True)
                source_display = staging_display / "source" / f"page_{page_index:05d}.webp"
                live_source_display = live_display / "source" / f"page_{page_index:05d}.webp"
                if live_source_display.is_file():
                    # Repair never changes source pixels. Reuse the already
                    # verified browser asset instead of encoding all sources a
                    # second time during every repair attempt.
                    self._link_or_copy(live_source_display, source_display)
                else:
                    self._write_display_asset(Path(page["source_path"]), source_display)
                self._write_display_asset(
                    staged_page_path,
                    staging_display / "final" / f"page_{page_index:05d}.webp",
                )
                self.reading_cache.prepare_publication(
                    staged_page_path, live_final / "pages" / f"page_{page_index:05d}.png"
                )
                staged_pages.append(
                    {
                        "id": int(page["id"]),
                        "output_path": live_final / "pages" / f"page_{page_index:05d}.png",
                        "asset_revision": str(time.time_ns()),
                        "staged_path": staged_page_path,
                    }
                )
                if progress_callback is not None:
                    progress_callback(page_index, page_number, total_pages)

            export_book(
                [Path(page["staged_path"]) for page in staged_pages],
                staging_output,
                spec.output_format,
            )
            backup_root.mkdir(parents=True, exist_ok=False)
            manifest.backup_to(backup_root / "manifest.sqlite")
            old_final_moved = False
            old_output_moved = False
            old_display_moved = False
            new_final_published = False
            new_output_published = False
            new_display_published = False
            try:
                if live_final.exists():
                    self._replace_with_windows_retry(live_final, backup_root / "final")
                    old_final_moved = True
                self._replace_with_windows_retry(staging_final, live_final)
                new_final_published = True
                if self._path_entry_exists(live_output):
                    self._replace_with_windows_retry(live_output, backup_root / "output")
                    old_output_moved = True
                self._publish_staged_output_files(staging_output, live_output)
                new_output_published = True
                if live_display.exists():
                    self._replace_with_windows_retry(live_display, backup_root / "display")
                    old_display_moved = True
                self._replace_with_windows_retry(staging_display, live_display)
                new_display_published = True
                manifest.apply_repaired_outputs(
                    job_id, staged_units, staged_pages, backup_path=backup_root
                )
            except Exception:
                if new_final_published and live_final.exists():
                    self._replace_with_windows_retry(
                        live_final, staging_root / "failed-final"
                    )
                if old_final_moved and (backup_root / "final").exists():
                    self._replace_with_windows_retry(backup_root / "final", live_final)
                if new_output_published and self._path_entry_exists(live_output):
                    self._withdraw_published_output_files(
                        live_output, staging_root / "failed-output"
                    )
                if old_output_moved and self._path_entry_exists(backup_root / "output"):
                    self._replace_with_windows_retry(backup_root / "output", live_output)
                if new_display_published and live_display.exists():
                    self._replace_with_windows_retry(
                        live_display, staging_root / "failed-display"
                    )
                if old_display_moved and (backup_root / "display").exists():
                    self._replace_with_windows_retry(backup_root / "display", live_display)
                raise
            committed = True
            return len(staged_units)
        except Exception as exc:
            # A failed repair must leave a durable, small forensic record even
            # when the large staging tree is removed.  The report intentionally
            # contains page/unit indexes and counts only; source and credential
            # paths do not belong in the recovery record.
            report_dir = job_dir / "repair-reports"
            report_path = report_dir / f"{token}.json"
            report = {
                "job_id": job_id,
                "token": token,
                "status": "rolled_back",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "failed_page_index": current_page_index,
                "failed_unit_index": current_unit_index,
                "staged_page_count": len(staged_pages),
                "staged_unit_count": len(staged_units),
                "backup_created": backup_root.exists(),
                "live_results_preserved": not committed,
            }
            try:
                report_dir.mkdir(parents=True, exist_ok=True)
                report_path.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except OSError:
                logger.exception("unable to persist repair failure report job=%s", job_id)
            logger.error("repair rolled back job=%s report=%s", job_id, report_path)
            raise
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)
            if not committed and backup_root.exists() and not any(backup_root.iterdir()):
                backup_root.rmdir()

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
            if any(unit["status"] != "qa_passed" or not unit["final_path"] for unit in units):
                raise RuntimeError(f"Page {page['page_index']} has unfinished units")
            valid_units = self._paste_units_safely(
                canvas, [dict(unit) for unit in units], job_id=job_id
            )
            if valid_units != len(units):
                raise RuntimeError(f"Page {page['page_index']} has invalid panel results")
            output = (
                self._job_dir(job_id) / "final" / "pages" / f"page_{page['page_index']:05d}.png"
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(output, format="PNG")
            manifest.finish_page(page_id, output, str(time.time_ns()))
            self.prebuild_display_assets(job_id, int(page["page_index"]))
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
        deadline_at = datetime.fromtimestamp(datetime.now(UTC).timestamp() + 15, UTC).isoformat()
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
            reset = self._manifest(job_id).reset_running_units(job_id, "任务暂停时清理遗留运行单元")
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
        deadline_at = datetime.fromtimestamp(datetime.now(UTC).timestamp() + 15, UTC).isoformat()
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
            reset = self._manifest(job_id).reset_running_units(job_id, "任务取消时清理遗留运行单元")
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
            try:
                result = self.semantic_engine.segment(source)
                engine = self.semantic_engine
            except Exception as exc:
                logger.warning(
                    "semantic provider unavailable job=%s page=%s; "
                    "using deterministic fallback: %s",
                    job_id,
                    page_index + 1,
                    exc,
                )
                result = self._semantic_fallback.segment(source)
                engine = self._semantic_fallback
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
            descriptor=semantic_descriptor(engine, result),
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
