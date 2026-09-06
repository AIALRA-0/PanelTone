from __future__ import annotations

from typing import Any

from .http_service import HTTPImageEngine


class CobraCandidateEngine(HTTPImageEngine):
    """PanelTone adapter for the isolated official Cobra HTTP candidate."""

    def healthcheck(self) -> dict[str, Any]:
        result = super().healthcheck()
        result.setdefault("model_id", "JunhaoZhuang/Cobra")
        result.setdefault("candidate", True)
        return result
