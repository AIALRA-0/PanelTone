from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SourceDescriptor(BaseModel):
    source_id: str
    name: str
    kind: str
    size: int
    sha256: str
    duplicate: bool


class JobProgress(BaseModel):
    stage: str
    completed_units: int
    total_units: int
    failed_units: int
    completed_pages: int
    total_pages: int
    current_page: int | None
    percent: float
    seconds_per_megapixel: float | None
    eta_seconds: int | None
    elapsed_seconds: int
    ready_page_indices: list[int]
    uploaded_bytes: int = 0
    total_upload_bytes: int = 0
    queue_position: int | None = None


class JobEvent(BaseModel):
    id: int | None = None
    kind: str
    job_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class PresetDescriptor(BaseModel):
    id: str
    name: str
    description: str
    best_for: str | None = None
    changes: str | None = None
    tradeoff: str | None = None
    speed: str | None = None
    memory: str | None = None


class PresetCatalog(BaseModel):
    colors: list[PresetDescriptor]
    styles: list[PresetDescriptor]
    modes: list[PresetDescriptor]
    details: list[PresetDescriptor]
    panels: list[PresetDescriptor]
    outputs: list[PresetDescriptor]


class ModelDescriptor(BaseModel):
    id: str
    name: str
    repository: str
    revision: str
    allow_patterns: list[str]
    license: str
    license_url: str
    memory: str
    download_size: str
    storage: str
    purpose: str
    installed: bool
    connected: bool
    status: str
