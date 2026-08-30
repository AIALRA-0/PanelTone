from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Settings:
    data_root: Path = Path("jobs")
    model_root: Path = Path("models")
    comfyui_url: str = "http://127.0.0.1:8188"
    max_upload_mib: int = 4096
    qa_line_f1_min: float = 0.98
    qa_luminance_mae_max: float = 2.0
    panel_min_area_ratio: float = 0.02
    panel_padding: int = 32
    max_archive_members: int = 5000
    max_archive_ratio: int = 200
    allowed_roots: list[Path] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> Settings:
        local_data = Path(os.getenv("LOCALAPPDATA", Path.home() / ".local" / "share"))
        default_root = local_data / "PanelTone" / "jobs"
        default_models = local_data / "PanelTone" / "models"
        raw_roots = os.getenv(
            "PANELTONE_ALLOWED_ROOTS",
            os.getenv("MANGA_REPAINT_ALLOWED_ROOTS", ""),
        )
        roots = [Path(item).resolve() for item in raw_roots.split(os.pathsep) if item]
        return cls(
            data_root=Path(
                os.getenv(
                    "PANELTONE_DATA_ROOT",
                    os.getenv("MANGA_REPAINT_DATA_ROOT", str(default_root)),
                )
            ).resolve(),
            model_root=Path(os.getenv("PANELTONE_MODEL_ROOT", str(default_models))).resolve(),
            comfyui_url=os.getenv(
                "PANELTONE_COMFYUI_URL",
                os.getenv("MANGA_REPAINT_COMFYUI_URL", "http://127.0.0.1:8188"),
            ),
            max_upload_mib=int(
                os.getenv(
                    "PANELTONE_MAX_UPLOAD_MIB",
                    os.getenv("MANGA_REPAINT_MAX_UPLOAD_MIB", "4096"),
                )
            ),
            qa_line_f1_min=float(
                os.getenv(
                    "PANELTONE_QA_LINE_F1_MIN",
                    os.getenv("MANGA_REPAINT_QA_LINE_F1_MIN", "0.98"),
                )
            ),
            qa_luminance_mae_max=float(
                os.getenv(
                    "PANELTONE_QA_LUMINANCE_MAE_MAX",
                    os.getenv("MANGA_REPAINT_QA_LUMINANCE_MAE_MAX", "2.0"),
                )
            ),
            panel_min_area_ratio=float(
                os.getenv(
                    "PANELTONE_PANEL_MIN_AREA_RATIO",
                    os.getenv("MANGA_REPAINT_PANEL_MIN_AREA_RATIO", "0.02"),
                )
            ),
            panel_padding=int(
                os.getenv(
                    "PANELTONE_PANEL_PADDING",
                    os.getenv("MANGA_REPAINT_PANEL_PADDING", "32"),
                )
            ),
            max_archive_members=int(os.getenv("PANELTONE_MAX_ARCHIVE_MEMBERS", "5000")),
            max_archive_ratio=int(os.getenv("PANELTONE_MAX_ARCHIVE_RATIO", "200")),
            allowed_roots=roots,
        )

    @classmethod
    def from_json(cls, path: Path) -> Settings:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if "data_root" in data:
            data["data_root"] = Path(data["data_root"]).resolve()
        if "model_root" in data:
            data["model_root"] = Path(data["model_root"]).resolve()
        if "allowed_roots" in data:
            data["allowed_roots"] = [Path(item).resolve() for item in data["allowed_roots"]]
        return cls(**data)


def ensure_allowed_path(path: Path, allowed_roots: list[Path]) -> Path:
    resolved = path.resolve(strict=True)
    if not allowed_roots:
        return resolved
    if not any(resolved == root or root in resolved.parents for root in allowed_roots):
        raise PermissionError(f"Path is outside configured allowed roots: {resolved}")
    return resolved
