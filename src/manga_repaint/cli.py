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
    create.add_argument("--output-format", choices=["cbz", "pdf", "images"], default="cbz")
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

    serve = subparsers.add_parser("serve", help="Start the local review application")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

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
