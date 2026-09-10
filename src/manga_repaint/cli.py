from __future__ import annotations

import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .api import create_app
from .config import Settings
from .engines import EngineRegistry
from .models import DetailMode, JobMode, JobSpec, ProtectionMode
from .project import ProjectManager


def _manager(args: argparse.Namespace) -> ProjectManager:
    settings = Settings.from_json(Path(args.settings)) if args.settings else Settings.from_env()
    registry = EngineRegistry.from_json(Path(args.engines), settings.comfyui_url)
    return ProjectManager(settings, registry)


def _print(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _configure_logging(settings: Settings) -> None:
    """Persist local worker logs next to the user's data without exposing book paths in the UI."""
    logger = logging.getLogger("paneltone")
    logger.setLevel(logging.INFO)
    log_dir = settings.data_root.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "paneltone.log"
    if not any(
        isinstance(handler, RotatingFileHandler)
        and Path(getattr(handler, "baseFilename", "")).resolve() == log_path.resolve()
        for handler in logger.handlers
    ):
        handler = RotatingFileHandler(
            log_path,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        logger.addHandler(handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paneltone")
    parser.add_argument(
        "--settings", help="JSON settings file; environment variables are used by default"
    )
    parser.add_argument("--engines", default="configs/engines.json", help="Engine registry JSON")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="Ingest a book and create a resumable job")
    create.add_argument("source")
    create.add_argument("--mode", choices=[item.value for item in JobMode], default="colorize")
    create.add_argument("--engine", default="palette")
    create.add_argument(
        "--protection", choices=[item.value for item in ProtectionMode], default="strict"
    )
    create.add_argument(
        "--detail-mode", choices=[item.value for item in DetailMode], default="strict"
    )
    create.add_argument(
        "--output-format", choices=["cbz", "pdf", "images", "jpeg", "webp"], default="cbz"
    )
    create.add_argument("--panel-mode", choices=["page", "detect"], default="page")
    create.add_argument("--seed", type=int, default=0)
    create.add_argument("--prompt", default="")
    create.add_argument("--negative-prompt", default="")
    create.add_argument("--color-preset", default="natural")
    create.add_argument("--style-preset", default="original_ink")
    create.add_argument("--no-preserve-text", action="store_true")
    create.add_argument("--no-preserve-ink", action="store_true")
    create.add_argument("--ink-gamma", type=float, default=0.42)
    create.add_argument("--chroma-strength", type=float, default=1.15)
    create.add_argument("--reference", action="append", default=[])
    create.add_argument("--max-retries", type=int, default=2)
    create.add_argument("--adult-fictional-content", action="store_true")

    run = subparsers.add_parser("run", help="Run or resume a job")
    run.add_argument("job_id")

    status = subparsers.add_parser("status", help="Show a job summary")
    status.add_argument("job_id")

    subparsers.add_parser("list", help="List local jobs")
    subparsers.add_parser("health", help="Check configured engines")

    material = subparsers.add_parser(
        "material-render", help="Render a reviewed material bundle into a NEW isolated directory"
    )
    material.add_argument("source")
    material.add_argument(
        "plan", help="Reviewed plan.json with source-sized label/protection masks"
    )
    material.add_argument("output", help="New directory; existing paths are never overwritten")

    proposal = subparsers.add_parser(
        "flat-proposal",
        help="Create a source-topology flat-colour proposal in a NEW isolated directory",
    )
    proposal.add_argument("source")
    proposal.add_argument(
        "candidate", help="Colour candidate used only for per-region colour votes"
    )
    proposal.add_argument("output", help="New review directory; live results are never changed")
    proposal.add_argument("--palette-revision", default="isolated-proposal-v1")

    sam_proposal = subparsers.add_parser(
        "sam2-flat-proposal",
        help="Create an object-consistent SAM2 flat proposal in a NEW review directory",
    )
    sam_proposal.add_argument("source")
    sam_proposal.add_argument("candidate")
    sam_proposal.add_argument("output")
    sam_proposal.add_argument("--model-root", required=True)
    sam_proposal.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    sam_proposal.add_argument("--grid-x", type=int, default=10)
    sam_proposal.add_argument("--grid-y", type=int, default=14)
    sam_proposal.add_argument(
        "--material-labels", help="Reviewed labels-only JSON; VLM coordinates are rejected"
    )
    sam_proposal.add_argument("--palette-revision", default="sam2-isolated-proposal-v1")

    cel_candidate = subparsers.add_parser(
        "cel-candidate",
        help="Build a source-geometry cel candidate in a NEW isolated review directory",
    )
    cel_candidate.add_argument("source")
    cel_candidate.add_argument("candidate", help="Model image used only as an albedo proposal")
    cel_candidate.add_argument(
        "output", help="New review directory; live results are never changed"
    )
    cel_candidate.add_argument(
        "--refinement",
        action="append",
        default=[],
        metavar="LEFT,TOP,RIGHT,BOTTOM=IMAGE",
        help="Optional high-resolution panel colour proposal; may be repeated",
    )
    cel_candidate.add_argument("--chroma-strength", type=float, default=1.0)

    serve = subparsers.add_parser("serve", help="Start the local review application")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "material-render":
        # Standalone review must not construct a live manager or touch recovery
        from PIL import Image

        from .material_render import (
            build_sparse_color_hint,
            evaluate_material_render,
            load_material_plan,
            render_material_flats,
        )

        with Image.open(args.source) as image:
            source = image.copy()
        plan = load_material_plan(Path(args.plan), source)
        final = render_material_flats(source, plan)
        report = evaluate_material_render(source, final, plan)
        hint = build_sparse_color_hint(source, plan)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)
        final.save(output / "preview.png")
        hint.save(output / "cobra-hint.png")
        (output / "qa.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        _print(report)
        return 0 if report["passed"] else 2

    if args.command == "flat-proposal":
        # This is an isolated review artifact. Candidate geometry, texture and
        # light never enter the rendered pixels, and the saved plan remains in
        # proposed state until an explicit review operation accepts it
        from dataclasses import asdict

        import numpy as np
        from PIL import Image

        from .flat_planner import preview_plan, propose_material_plan
        from .material_render import render_material_layers, save_material_plan
        from .semantic import ConservativeSemanticMaskEngine

        with Image.open(args.source) as image:
            source = image.convert("RGB")
        with Image.open(args.candidate) as image:
            candidate = image.convert("RGB")
        semantic = ConservativeSemanticMaskEngine().segment(source)
        protected = np.logical_or.reduce(
            [semantic.masks[name] for name in ("text", "bubbles", "borders", "ink")]
        )
        result = propose_material_plan(
            source,
            candidate,
            protected,
            palette_revision=args.palette_revision,
            evidence=f"candidate:{Path(args.candidate).name}",
        )
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)
        save_material_plan(output / "plan", source, result.plan)
        layers = render_material_layers(source, preview_plan(result))
        layers.flats.save(output / "flats.png")
        layers.shadows.save(output / "shadows.png")
        layers.final.save(output / "cel.png")
        Image.fromarray(layers.shade_index * 85).save(output / "shade-index.png")
        Image.fromarray(layers.screentone.astype(np.uint8) * 255).save(
            output / "screentone.png"
        )
        report = {
            "status": "review_required",
            "publishable": False,
            "candidate_pixels_used_in_final": False,
            "proposal_count": len(result.proposals),
            "unknown_ratio": result.unknown_ratio,
            "conflict_ratio": result.conflict_ratio,
            "proposals": [asdict(item) for item in result.proposals],
        }
        (output / "proposal.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _print(report)
        return 0

    if args.command == "sam2-flat-proposal":
        # This command is intentionally run by the separate model environment
        # and emits review artifacts only.  Nothing is written into live data
        from dataclasses import asdict

        import numpy as np
        from PIL import Image

        from .flat_planner import preview_plan, propose_material_plan_with_object_masks
        from .material_render import render_material_layers, save_material_plan
        from .sam2_regions import Sam2AutomaticMasker, render_region_overlay
        from .semantic import ConservativeSemanticMaskEngine
        from .vlm_proposals import parse_material_labels, reviewed_materials

        with Image.open(args.source) as image:
            source = image.convert("RGB")
        with Image.open(args.candidate) as image:
            candidate = image.convert("RGB")
        semantic = ConservativeSemanticMaskEngine().segment(source)
        protected = np.logical_or.reduce(
            [semantic.masks[name] for name in ("text", "bubbles", "borders", "ink")]
        )
        masker = Sam2AutomaticMasker(Path(args.model_root), device=args.device)
        masks = masker.propose(
            source, exclude=protected, grid_x=args.grid_x, grid_y=args.grid_y
        )
        material_labels = None
        if args.material_labels:
            payload = Path(args.material_labels).read_text(encoding="utf-8")
            proposals = parse_material_labels(payload, expected_ids=set(range(len(masks))))
            material_labels = reviewed_materials(proposals)
        result = propose_material_plan_with_object_masks(
            source,
            candidate,
            protected,
            masks,
            palette_revision=args.palette_revision,
            evidence=f"sam2+candidate:{Path(args.candidate).name}",
            material_labels=material_labels,
        )
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)
        render_region_overlay(source, masks).save(output / "sam2-regions.png")
        save_material_plan(output / "plan", source, result.plan)
        layers = render_material_layers(source, preview_plan(result))
        layers.flats.save(output / "flats.png")
        layers.shadows.save(output / "shadows.png")
        layers.final.save(output / "cel.png")
        Image.fromarray(layers.shade_index * 85).save(output / "shade-index.png")
        Image.fromarray(layers.screentone.astype(np.uint8) * 255).save(
            output / "screentone.png"
        )
        report = {
            "status": "review_required",
            "publishable": False,
            "candidate_pixels_used_in_final": False,
            "sam2_mask_count": len(masks),
            "proposal_count": len(result.proposals),
            "unknown_ratio": result.unknown_ratio,
            "conflict_ratio": result.conflict_ratio,
            "proposals": [asdict(item) for item in result.proposals],
        }
        (output / "proposal.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _print(report)
        return 0

    if args.command == "cel-candidate":
        # This produces review evidence only. The candidate supplies albedo;
        # source geometry, tone, text and ink remain authoritative
        import hashlib

        import numpy as np
        from PIL import Image

        from .color import composite_cel_locked_colorization, merge_candidate_albedo
        from .qa import evaluate
        from .semantic import ConservativeSemanticMaskEngine

        with Image.open(args.source) as image:
            source = image.convert("RGB")
        with Image.open(args.candidate) as image:
            candidate = image.convert("RGB")
        refinements = []
        for value in args.refinement:
            try:
                coordinates, image_path = value.split("=", 1)
                box = tuple(int(item) for item in coordinates.split(","))
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    "refinement must use LEFT,TOP,RIGHT,BOTTOM=IMAGE"
                ) from exc
            if len(box) != 4:
                raise ValueError("refinement must contain four coordinates")
            with Image.open(image_path) as image:
                refinements.append((box, image.convert("RGB")))
        if refinements:
            candidate = merge_candidate_albedo(candidate, refinements)
        if candidate.size != source.size:
            raise ValueError("candidate dimensions must match source exactly")
        semantic = ConservativeSemanticMaskEngine().segment(source)
        protected = np.logical_or.reduce(
            [semantic.masks[name] for name in ("text", "bubbles", "borders", "ink")]
        )
        final = composite_cel_locked_colorization(
            source,
            candidate,
            protected,
            chroma_strength=args.chroma_strength,
        )
        qa = evaluate(
            source,
            final,
            protected,
            generated=candidate,
            luminance_mae_max=255.0,
            geometry_locked=True,
            color_retention_min=0.75,
            chroma_edge_alignment_min=0.995,
        )
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)
        candidate.save(output / "albedo-proposal.png")
        final.save(output / "cel-locked.png")
        final.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        final.save(output / "display.webp", format="WEBP", quality=88, method=4)
        report = {
            "status": "review_required",
            "publishable": False,
            "candidate_pixels_used_in_final": False,
            "source_geometry_owner": True,
            "refinement_count": len(refinements),
            "qa_passed": qa.passed,
            "qa": qa.to_json_dict(),
            "source_sha256": hashlib.sha256(Path(args.source).read_bytes()).hexdigest(),
            "candidate_sha256": hashlib.sha256(
                (output / "albedo-proposal.png").read_bytes()
            ).hexdigest(),
            "render_sha256": hashlib.sha256(
                (output / "cel-locked.png").read_bytes()
            ).hexdigest(),
        }
        (output / "review.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _print(report)
        return 0

    # The web server creates its own application manager.  Do not construct a
    # throw-away manager first: its startup recovery pass could observe a live
    # job and mark it paused before the actual server is ready.
    if args.command == "serve":
        import uvicorn

        settings = Settings.from_json(Path(args.settings)) if args.settings else Settings.from_env()
        registry = EngineRegistry.from_json(Path(args.engines), settings.comfyui_url)
        _configure_logging(settings)
        uvicorn.run(
            create_app(settings, registry),
            host=args.host,
            port=args.port,
        )
        return 0

    manager = _manager(args)
    if args.command == "create":
        spec = JobSpec(
            source=Path(args.source),
            workspace=manager.settings.data_root,
            mode=JobMode(args.mode),
            engine=args.engine,
            protection=ProtectionMode(args.protection),
            detail_mode=DetailMode(args.detail_mode),
            output_format=args.output_format,
            panel_mode=args.panel_mode,
            seed=args.seed,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            color_preset=args.color_preset,
            style_preset=args.style_preset,
            preserve_text=not args.no_preserve_text,
            preserve_ink=not args.no_preserve_ink,
            ink_gamma=args.ink_gamma,
            chroma_strength=args.chroma_strength,
            style_references=[Path(item) for item in args.reference],
            max_retries=args.max_retries,
            adult_fictional_content=args.adult_fictional_content,
        )
        job_id = manager.create(spec)
        _print({"job_id": job_id, "status": "ready"})
        return 0
    if args.command == "run":
        output = manager.process(args.job_id)
        _print({"job_id": args.job_id, "status": "completed", "output": output})
        return 0
    if args.command == "status":
        _print(manager.status(args.job_id))
        return 0
    if args.command == "list":
        _print(manager.list_jobs())
        return 0
    if args.command == "health":
        _print(manager.registry.health())
        return 0
    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
