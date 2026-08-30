from __future__ import annotations

import copy
import json
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from .base import EngineRequest, EngineResult


class ComfyUIEngine:
    def __init__(
        self,
        name: str,
        base_url: str,
        workflow_path: Path,
        timeout_seconds: int = 1800,
        poll_interval: float = 0.75,
        output_node: str | None = None,
    ):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.workflow_path = workflow_path
        self.timeout_seconds = timeout_seconds
        self.poll_interval = poll_interval
        self.output_node = output_node

    def _upload(self, client: httpx.Client, path: Path) -> str:
        with path.open("rb") as handle:
            response = client.post(
                "/upload/image",
                files={"image": (path.name, handle, "image/png")},
                data={"overwrite": "true", "type": "input"},
            )
        response.raise_for_status()
        payload = response.json()
        return str(payload.get("name") or path.name)

    def _replace_tokens(self, value: Any, replacements: dict[str, Any]) -> Any:
        if isinstance(value, dict):
            return {key: self._replace_tokens(item, replacements) for key, item in value.items()}
        if isinstance(value, list):
            return [self._replace_tokens(item, replacements) for item in value]
        if isinstance(value, str):
            if value in replacements:
                return replacements[value]
            for token, replacement in replacements.items():
                value = value.replace(token, str(replacement))
        return value

    def _find_output(self, history: dict[str, Any]) -> dict[str, str]:
        outputs = history.get("outputs", {})
        nodes = [self.output_node] if self.output_node else list(outputs)
        for node_id in nodes:
            if node_id is None or node_id not in outputs:
                continue
            for image in outputs[node_id].get("images", []):
                if image.get("filename"):
                    return image
        raise RuntimeError("ComfyUI workflow completed without an image output")

    def generate(self, request: EngineRequest) -> EngineResult:
        workflow = json.loads(self.workflow_path.read_text(encoding="utf-8"))
        client_id = str(uuid.uuid4())
        with httpx.Client(base_url=self.base_url, timeout=60) as client:
            input_name = self._upload(client, request.source_path)
            reference_names = [self._upload(client, path) for path in request.references]
            replacements: dict[str, Any] = {
                "{{INPUT_IMAGE}}": input_name,
                "{{REFERENCE_IMAGE}}": reference_names[0] if reference_names else input_name,
                "{{REFERENCE_IMAGES}}": reference_names,
                "{{PROMPT}}": request.prompt,
                "{{NEGATIVE_PROMPT}}": request.negative_prompt,
                "{{SEED}}": request.seed + request.attempt - 1,
            }
            prompt = self._replace_tokens(copy.deepcopy(workflow), replacements)
            response = client.post("/prompt", json={"prompt": prompt, "client_id": client_id})
            response.raise_for_status()
            submission = response.json()
            if "prompt_id" not in submission:
                raise RuntimeError(f"ComfyUI rejected workflow: {submission}")
            prompt_id = str(submission["prompt_id"])

            deadline = time.monotonic() + self.timeout_seconds
            history: dict[str, Any] | None = None
            while time.monotonic() < deadline:
                response = client.get(f"/history/{prompt_id}")
                response.raise_for_status()
                payload = response.json()
                if prompt_id in payload:
                    history = payload[prompt_id]
                    break
                time.sleep(self.poll_interval)
            if history is None:
                try:
                    client.post("/interrupt")
                finally:
                    raise TimeoutError(f"ComfyUI workflow timed out after {self.timeout_seconds}s")

            status = history.get("status", {})
            if status.get("status_str") == "error" or status.get("completed") is False:
                raise RuntimeError(f"ComfyUI workflow failed: {status}")
            image_info = self._find_output(history)
            response = client.get(
                "/view",
                params={
                    "filename": image_info["filename"],
                    "subfolder": image_info.get("subfolder", ""),
                    "type": image_info.get("type", "output"),
                },
            )
            response.raise_for_status()
            request.output_path.parent.mkdir(parents=True, exist_ok=True)
            request.output_path.write_bytes(response.content)
            with Image.open(request.output_path) as image:
                image.convert("RGB").save(request.output_path, format="PNG")
            return EngineResult(
                request.output_path,
                {"prompt_id": prompt_id, "image": image_info, "workflow": str(self.workflow_path)},
            )

    def healthcheck(self) -> dict[str, Any]:
        if not self.workflow_path.is_file():
            return {
                "ok": False,
                "engine": self.name,
                "error": f"workflow_not_found:{self.workflow_path}",
            }
        try:
            with httpx.Client(base_url=self.base_url, timeout=5) as client:
                response = client.get("/system_stats")
                response.raise_for_status()
                return {"ok": True, "engine": self.name, "system_stats": response.json()}
        except Exception as exc:
            return {"ok": False, "engine": self.name, "error": str(exc)}
