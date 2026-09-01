from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import Engine
from .comfyui import ComfyUIEngine
from .external import ExternalCommandEngine
from .http_service import HTTPImageEngine
from .palette import PaletteEngine


class EngineRegistry:
    def __init__(self, engines: dict[str, Engine] | None = None):
        self._engines: dict[str, Engine] = {"palette": PaletteEngine()}
        if engines:
            self._engines.update(engines)

    def register(self, engine: Engine) -> None:
        self._engines[engine.name] = engine

    def get(self, name: str) -> Engine:
        if name not in self._engines:
            choices = ", ".join(sorted(self._engines))
            raise KeyError(f"Unknown engine '{name}'. Configured engines: {choices}")
        return self._engines[name]

    def health(self) -> dict[str, dict[str, Any]]:
        return {name: engine.healthcheck() for name, engine in self._engines.items()}

    def release(self, name: str) -> dict[str, Any]:
        engine = self.get(name)
        release = getattr(engine, "release", None)
        if not callable(release):
            raise ValueError(f"引擎 {name} 不支持释放模型显存")
        result = release()
        return result if isinstance(result, dict) else {"status": "released"}

    @classmethod
    def from_json(cls, path: Path, default_comfy_url: str) -> EngineRegistry:
        registry = cls()
        if not path.is_file():
            return registry
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data.get("engines", []):
            kind = item["kind"]
            if kind == "comfyui":
                registry.register(
                    ComfyUIEngine(
                        name=item["name"],
                        base_url=item.get("base_url", default_comfy_url),
                        workflow_path=(path.parent / item["workflow"]).resolve(),
                        timeout_seconds=int(item.get("timeout_seconds", 1800)),
                        output_node=item.get("output_node"),
                    )
                )
            elif kind == "external":
                registry.register(
                    ExternalCommandEngine(
                        name=item["name"],
                        command=list(item["command"]),
                        cwd=(path.parent / item["cwd"]).resolve() if item.get("cwd") else None,
                        timeout_seconds=int(item.get("timeout_seconds", 1800)),
                    )
                )
            elif kind == "http_image":
                registry.register(
                    HTTPImageEngine(
                        name=item["name"],
                        base_url=item["base_url"],
                        timeout_seconds=int(item.get("timeout_seconds", 1800)),
                        api_key=item.get("api_key"),
                    )
                )
            else:
                raise ValueError(f"Unsupported engine kind: {kind}")
        return registry
