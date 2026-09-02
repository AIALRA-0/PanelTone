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
    upload_id: str | None = None
    relative_path: str | None = None
    uploaded_bytes: int | None = None
    total_bytes: int | None = None
    complete: bool = True
    resume_url: str | None = None
    import_batch_id: str | None = None


class PageProgress(BaseModel):
    page_index: int
    status: str
    completed_units: int = 0
    total_units: int = 0
    error: str | None = None


class JobProgress(BaseModel):
    stage: str
    stage_percent: float = 0.0
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
    bytes_processed: int = 0
    bytes_total: int = 0
    current_file: str | None = None
    current_unit: int | None = None
    latest_message: str | None = None
    page_states: list[PageProgress] = Field(default_factory=list)
    control_state: JobControlState | None = None


class JobControlState(BaseModel):
    action: str = "none"
    requested_at: str | None = None
    deadline_at: str | None = None
    active_request: bool = False
    message: str | None = None


class FolderNode(BaseModel):
    id: str
    parent_id: str | None = None
    name: str
    sort_order: int = 0
    archived_at: str | None = None
    job_count: int = 0
    children: list[FolderNode] = Field(default_factory=list)
    jobs: list[dict[str, Any]] = Field(default_factory=list)


class LibraryNode(BaseModel):
    kind: str
    id: str
    name: str
    parent_id: str | None = None
    sort_order: int = 0
    archived_at: str | None = None
    job_count: int = 0
    folder_id: str | None = None
    job: dict[str, Any] | None = None


class JobEvent(BaseModel):
    id: int | None = None
    kind: str
    job_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class ImportOperation(BaseModel):
    operation_id: str
    job_id: str
    status: str
    stage: str
    stage_percent: float = 0.0
    discovered_pages: int = 0
    total_pages: int | None = None
    bytes_processed: int = 0
    bytes_total: int = 0
    current_file: str | None = None
    latest_message: str | None = None
    error: str | None = None


class RawLogEntry(BaseModel):
    id: int | None = None
    timestamp: str
    level: str
    component: str
    job_id: str | None = None
    page_index: int | None = None
    unit_index: int | None = None
    event: str
    message: str
    error_code: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class PageAsset(BaseModel):
    page_index: int
    status: str
    source_url: str
    source_display_url: str | None = None
    preview_url: str | None = None
    final_url: str | None = None
    final_display_url: str | None = None
    thumbnail_url: str | None = None
    asset_revision: str | None = None
    preview_only: bool = False
    error: str | None = None
    mask_status: str = "pending"
    semantic_mask: dict[str, Any] | None = None


class GpuMetrics(BaseModel):
    timestamp: str
    available: bool
    reason: str | None = None
    gpu_index: int | None = None
    name: str | None = None
    utilization_percent: float | None = None
    memory_used_mib: int | None = None
    memory_total_mib: int | None = None
    temperature_c: float | None = None
    power_w: float | None = None
    driver_version: str | None = None
    process_count: int | None = None
    cpu_percent: float | None = None
    memory_percent: float | None = None
    disk_free_gib: float | None = None


class SemanticMaskDescriptor(BaseModel):
    status: str
    provider: str
    version: str | None = None
    classes: list[str] = Field(default_factory=list)
    confidence_threshold: float = 0.78
    uncertain_ratio: float = 0.0
    cached: bool = False


class IdentityRecord(BaseModel):
    identity_id: str
    label: str
    region: str
    color: str | None = None
    shadow_color: str | None = None
    confidence: float = 0.0
    locked: bool = False


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
    downloadable: bool = True
    unavailable_reason: str | None = None
    supports_interrupt: bool = False
    supports_release: bool = False


JobProgress.model_rebuild()
FolderNode.model_rebuild()
