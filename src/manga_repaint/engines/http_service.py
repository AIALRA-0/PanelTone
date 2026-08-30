from __future__ import annotations

import json
from typing import Any

import httpx
from PIL import Image

from .base import EngineRequest, EngineResult


class HTTPImageEngine:
    def __init__(
        self,
        name: str,
        base_url: str,
        timeout_seconds: int = 1800,
        api_key: str | None = None,
    ):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def generate(self, request: EngineRequest) -> EngineResult:
        files: list[tuple[str, tuple[str, Any, str]]] = []
        handles = []
        try:
            source_handle = request.source_path.open("rb")
            handles.append(source_handle)
            files.append(("source", (request.source_path.name, source_handle, "image/png")))
            for reference in request.references:
                handle = reference.open("rb")
                handles.append(handle)
                files.append(("references", (reference.name, handle, "image/png")))
            payload = {
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "seed": str(request.seed + request.attempt - 1),
                "mode": request.mode.value,
                "metadata_json": json.dumps(request.metadata, ensure_ascii=False),
            }
            with httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                headers=self._headers(),
            ) as client:
                response = client.post("/generate", data=payload, files=files)
                response.raise_for_status()
                request.output_path.parent.mkdir(parents=True, exist_ok=True)
                request.output_path.write_bytes(response.content)
        finally:
            for handle in handles:
                handle.close()
        with Image.open(request.output_path) as image:
            image.convert("RGB").save(request.output_path, format="PNG")
        return EngineResult(
            request.output_path,
            {
                "service": self.base_url,
                "model": response.headers.get("x-model-id", "unknown"),
                "elapsed_ms": response.headers.get("x-elapsed-ms"),
            },
        )

    def healthcheck(self) -> dict[str, Any]:
        try:
            with httpx.Client(base_url=self.base_url, timeout=5, headers=self._headers()) as client:
                response = client.get("/health")
                response.raise_for_status()
                return {"ok": True, "engine": self.name, **response.json()}
        except Exception as exc:
            return {"ok": False, "engine": self.name, "error": str(exc)}
