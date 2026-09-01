from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from PIL import Image

from .base import EngineRequest, EngineResult


class ExternalCommandEngine:
    def __init__(
        self,
        name: str,
        command: list[str],
        cwd: Path | None = None,
        timeout_seconds: int = 1800,
        env: dict[str, str] | None = None,
    ):
        if not command:
            raise ValueError("External engine command cannot be empty")
        self.name = name
        self.command = command
        self.cwd = cwd
        self.timeout_seconds = timeout_seconds
        self.env = env

    def _render(self, value: str, request: EngineRequest) -> str:
        replacements = {
            "{{SOURCE}}": str(request.source_path.resolve()),
            "{{OUTPUT}}": str(request.output_path.resolve()),
            "{{PROMPT}}": request.prompt,
            "{{NEGATIVE_PROMPT}}": request.negative_prompt,
            "{{SEED}}": str(request.seed + request.attempt - 1),
            "{{MODE}}": request.mode.value,
            "{{REFERENCES}}": ";".join(str(path.resolve()) for path in request.references),
        }
        for token, replacement in replacements.items():
            value = value.replace(token, replacement)
        return value

    def generate(self, request: EngineRequest) -> EngineResult:
        command = [self._render(item, request) for item in self.command]
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            command,
            cwd=self.cwd,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr[-4000:].strip()
            raise RuntimeError(
                f"External engine {self.name} failed ({completed.returncode}): {stderr}"
            )
        if not request.output_path.is_file():
            raise RuntimeError(f"External engine {self.name} did not create {request.output_path}")
        with Image.open(request.output_path) as image:
            image.verify()
        return EngineResult(
            request.output_path,
            {
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-1000:],
                "stderr_tail": completed.stderr[-1000:],
            },
        )

    def healthcheck(self) -> dict[str, Any]:
        executable = Path(self.command[0])
        available = executable.is_file() if executable.is_absolute() else True
        return {"ok": available, "engine": self.name, "command": self.command[0]}
