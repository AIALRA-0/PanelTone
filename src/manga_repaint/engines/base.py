from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from manga_repaint.models import JobMode


@dataclass(slots=True)
class EngineRequest:
    source_path: Path
    output_path: Path
    mode: JobMode
    seed: int
    prompt: str
    negative_prompt: str
    references: list[Path] = field(default_factory=list)
    attempt: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EngineResult:
    output_path: Path
    engine_metadata: dict[str, Any] = field(default_factory=dict)


class Engine(Protocol):
    name: str

    def generate(self, request: EngineRequest) -> EngineResult: ...

    def healthcheck(self) -> dict[str, Any]: ...
