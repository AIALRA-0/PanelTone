from __future__ import annotations

import json
import shutil
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .color import composite_protected, preserve_ink_overlay, preserve_luminance_lab
from .config import Settings, ensure_allowed_path
from .engines import EngineRegistry, EngineRequest
from .export import export_book
from .hashing import stable_hash
from .ingest import ingest_book, page_metadata
from .manifest import Manifest
from .masks import deterministic_protection_mask, pure_black_ink_mask, save_mask
from .models import DetailMode, JobMode, JobSpec, JobStatus, ProtectionMode
from .panels import extract_panels
from .presets import build_prompt, get_color_preset, get_style_preset
from .qa import evaluate


class ProjectManager:
    def __init__(
        self,
        settings: Settings,
        registry: EngineRegistry,
        event_callback: Callable[[str, dict[str, Any], str | None], None] | None = None,
    ):
        self.settings = settings
        self.registry = registry
        self.settings.data_root.mkdir(parents=True, exist_ok=True)
        self._controls: dict[str, threading.Event] = {}
        self._cancel_controls: dict[str, threading.Event] = {}
        self._process_lock = threading.Lock()
        self._event_callback = event_callback
        self.recover_interrupted_jobs()

    def set_event_callback(
        self, callback: Callable[[str, dict[str, Any], str | None], None]
    ) -> None:
        self._event_callback = callback

    def _emit(self, kind: str, payload: dict[str, Any], job_id: str | None = None) -> None:
        if self._event_callback:
            self._event_callback(kind, payload, job_id)

    def recover_interrupted_jobs(self) -> list[str]:
        recovered: list[str] = []
        for directory in self.settings.data_root.iterdir():
            manifest_path = directory / "manifest.sqlite"
            if not directory.is_dir() or not manifest_path.is_file():
                continue
            try:
                if Manifest(manifest_path).recover_interrupted(directory.name):
                    recovered.append(directory.name)
            except Exception:
                continue
        return recovered

    def _job_dir(self, job_id: str) -> Path:
        if not job_id or any(character not in "0123456789abcdef-" for character in job_id):
            raise ValueError("Invalid job id")
        return self.settings.data_root / job_id

    def _manifest(self, job_id: str) -> Manifest:
        return Manifest(self._job_dir(job_id) / "manifest.sqlite")

    def create(self, spec: JobSpec) -> str:
        source = ensure_allowed_path(spec.source, self.settings.allowed_roots)
        references = [
            ensure_allowed_path(path, self.settings.allowed_roots) for path in spec.style_references
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
        try:
            pages = ingest_book(
                source,
                job_dir / "pages",
                max_archive_members=self.settings.max_archive_members,
                max_archive_ratio=self.settings.max_archive_ratio,
            )
            copied_references: list[Path] = []
            for index, reference in enumerate(references):
                target = (
                    job_dir / "references" / f"reference_{index:04d}{reference.suffix.casefold()}"
                )
                shutil.copy2(reference, target)
                copied_references.append(target)
            spec.style_references = copied_references
            (job_dir / "job.json").write_text(
                json.dumps(spec.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )

            for page_index, page_path in enumerate(pages):
                checksum, width, height = page_metadata(page_path)
                page_id = manifest.add_page(job_id, page_index, page_path, checksum, width, height)
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
            manifest.set_job_status(job_id, JobStatus.READY)
            self._emit("job_status", {"status": JobStatus.READY.value}, job_id)
            return job_id
        except Exception as exc:
            manifest.set_job_status(job_id, JobStatus.FAILED, str(exc))
            raise

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

    def queue(self, job_id: str) -> None:
        self._manifest(job_id).set_queued(job_id)
        self._emit("job_queued", {"status": JobStatus.QUEUED.value}, job_id)

    def process(self, job_id: str) -> Path:
        with self._process_lock:
            return self._process_locked(job_id)

    def _process_locked(self, job_id: str) -> Path:
        manifest = self._manifest(job_id)
        spec = self._load_spec(job_id)
        engine = self.registry.get(spec.engine)
        engine_health = self.registry.health().get(spec.engine, {})
        if not engine_health.get("ok", False):
            manifest.set_job_status(
                job_id, JobStatus.WAITING_MODEL, str(engine_health.get("detail", ""))
            )
            self._emit(
                "job_status",
                {"status": JobStatus.WAITING_MODEL.value, "message": engine_health.get("detail")},
                job_id,
            )
            return self._job_dir(job_id) / "final"
        control = self._controls.setdefault(job_id, threading.Event())
        cancel_control = self._cancel_controls.setdefault(job_id, threading.Event())
        control.clear()
        cancel_control.clear()
        manifest.set_job_status(job_id, JobStatus.RUNNING)
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
                            spec.prompt,
                        )
                        request = EngineRequest(
                            source_path=source_path,
                            output_path=generated_path,
                            mode=spec.mode,
                            seed=spec.seed
                            + int(unit["page_index"]) * 1000
                            + int(unit["unit_index"]),
                            prompt=prompt,
                            negative_prompt=spec.negative_prompt,
                            references=spec.style_references,
                            attempt=attempts_used,
                            metadata=spec.metadata,
                        )
                        engine.generate(request)
                        with (
                            Image.open(source_path) as source_image,
                            Image.open(generated_path) as generated_image,
                            Image.open(unit["mask_path"]) as mask_image,
                        ):
                            source_rgb = source_image.convert("RGB")
                            generated_rgb = generated_image.convert("RGB")
                            mask = np.asarray(mask_image.convert("L")) > 0
                            if (
                                spec.mode == JobMode.STYLE_FULL
                                or not spec.preserve_ink
                                or spec.detail_mode == DetailMode.GENERATIVE
                            ):
                                final = composite_protected(source_rgb, generated_rgb, mask)
                            elif spec.detail_mode == DetailMode.BALANCED:
                                final = preserve_ink_overlay(
                                    source_rgb,
                                    generated_rgb,
                                    protected_mask=mask,
                                    gamma=spec.ink_gamma,
                                )
                            else:
                                generated_rgb = preserve_luminance_lab(
                                    source_rgb,
                                    generated_rgb,
                                    chroma_strength=spec.chroma_strength,
                                )
                                strict_mask = np.logical_or(mask, pure_black_ink_mask(source_rgb))
                                final = composite_protected(source_rgb, generated_rgb, strict_mask)
                            final.save(final_path, format="PNG")
                            qa = evaluate(
                                source_rgb,
                                final,
                                mask,
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
                                self._emit(
                                    "page_ready",
                                    {
                                        "page_index": int(unit["page_index"]),
                                        "url": (
                                            f"/api/jobs/{job_id}/pages/"
                                            f"{int(unit['page_index'])}/final"
                                        ),
                                    },
                                    job_id,
                                )
                            self._emit("job_progress", self.status(job_id)["progress"], job_id)
                            break
                    except Exception as exc:
                        manifest.fail_unit(unit_id, str(exc))
                        if attempts_this_run >= spec.max_retries + 1:
                            break
                if not passed:
                    failed_units.append(unit_id)
                    self._emit(
                        "job_error",
                        {
                            "page_index": int(unit["page_index"]),
                            "unit_index": int(unit["unit_index"]),
                            "message": f"处理单元在 {attempts_used} 次尝试后仍未通过检查",
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
            self._emit("job_status", {"status": JobStatus.COMPLETED.value}, job_id)
            return export_path
        except Exception as exc:
            manifest.set_job_status(job_id, JobStatus.FAILED, str(exc))
            self._emit("job_error", {"message": str(exc)}, job_id)
            raise

    def _assemble_page_if_ready(self, job_id: str, manifest: Manifest, page_id: int) -> Path | None:
        page = next(item for item in manifest.pages(job_id) if int(item["id"]) == page_id)
        if page["output_path"] and Path(page["output_path"]).is_file():
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
        manifest.finish_page(page_id, output)
        thumbnail = (
            self._job_dir(job_id) / "final" / "thumbnails" / f"page_{page['page_index']:05d}.jpg"
        )
        thumbnail.parent.mkdir(parents=True, exist_ok=True)
        preview = canvas.copy()
        preview.thumbnail((240, 320), Image.Resampling.LANCZOS)
        preview.save(thumbnail, format="JPEG", quality=82, optimize=True)
        return output

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
            manifest.finish_page(page_id, output)
            output_pages.append(output)
        return output_pages

    def pause(self, job_id: str) -> None:
        self._controls.setdefault(job_id, threading.Event()).set()

    def cancel(self, job_id: str) -> None:
        self._cancel_controls.setdefault(job_id, threading.Event()).set()

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
