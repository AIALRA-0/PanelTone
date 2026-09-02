from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class JobMode(StrEnum):
    COLORIZE = "colorize"
    STYLE_LOCKED = "style_locked"
    STYLE_FULL = "style_full"


class ProtectionMode(StrEnum):
    LUMINANCE = "luminance"
    LINE_ART = "line_art"
    STRICT = "strict"


class DetailMode(StrEnum):
    STRICT = "strict"
    BALANCED = "balanced"
    GENERATIVE = "generative"


class JobStatus(StrEnum):
    CREATED = "created"
    INGESTING = "ingesting"
    READY = "ready"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_MODEL = "waiting_model"
    NEEDS_ATTENTION = "needs_attention"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ARCHIVED = "archived"


class UnitStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    QA_PASSED = "qa_passed"
    QA_FAILED = "qa_failed"
    FAILED = "failed"


@dataclass(slots=True)
class JobSpec:
    source: Path
    workspace: Path
    mode: JobMode = JobMode.COLORIZE
    engine: str = "palette"
    protection: ProtectionMode = ProtectionMode.STRICT
    detail_mode: DetailMode = DetailMode.STRICT
    output_format: str = "cbz"
    panel_mode: str = "page"
    seed: int = 0
    prompt: str = ""
    negative_prompt: str = ""
    color_preset: str = "natural"
    style_preset: str = "original_ink"
    preserve_text: bool = True
    preserve_ink: bool = True
    ink_gamma: float = 0.42
    chroma_strength: float = 1.15
    style_references: list[Path] = field(default_factory=list)
    max_retries: int = 2
    adult_fictional_content: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    display_name: str = "未命名漫画"

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source"] = str(self.source)
        payload["workspace"] = str(self.workspace)
        payload["mode"] = self.mode.value
        payload["protection"] = self.protection.value
        payload["style_references"] = [str(path) for path in self.style_references]
        return payload


@dataclass(slots=True)
class PanelBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def to_tuple(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.right, self.bottom


@dataclass(slots=True)
class QAResult:
    passed: bool
    protected_pixel_diff: int
    line_edge_f1: float
    luminance_mae: float
    dimension_match: bool
    reasons: list[str] = field(default_factory=list)
    pure_black_preservation_rate: float = 1.0
    protected_area_ratio: float = 0.0
    generated_color_coverage: float = 0.0
    result_color_coverage: float = 0.0
    color_retention_ratio: float = 1.0
    color_dropout_tiles: int = 0

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)
