from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from manga_repaint.config import Settings
from manga_repaint.observability import RawLogStore, SystemTelemetry


def test_raw_log_sequence_survives_store_restart(tmp_path: Path) -> None:
    first = RawLogStore(tmp_path)
    first_entry = first.write(component="task", event="one", message="first")
    second = RawLogStore(tmp_path)
    second_entry = second.write(component="task", event="two", message="second")
    assert second_entry["id"] > first_entry["id"]
    records = second.read(kind="raw")
    assert [record["id"] for record in records[-2:]] == [first_entry["id"], second_entry["id"]]


def test_raw_log_filters_are_scoped_by_component_and_level(tmp_path: Path) -> None:
    store = RawLogStore(tmp_path)
    store.write(component="api", event="request", message="ok")
    store.write(component="gpu", event="metrics", message="warn", level="WARNING", kind="gpu")
    assert len(store.read(kind="raw", component="api")) == 1
    assert len(store.read(kind="gpu", level="WARNING", component="gpu")) == 1


def test_tail_queries_use_bounded_memory_after_startup(tmp_path: Path, monkeypatch) -> None:
    store = RawLogStore(tmp_path)
    for index in range(20):
        store.write(component="api", event="request", message=str(index))

    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("tail query rescanned the file")
        ),
    )

    assert [item["message"] for item in store.read(limit=3)] == ["17", "18", "19"]


def test_raw_log_redacts_paths_and_secrets(tmp_path: Path) -> None:
    store = RawLogStore(tmp_path, redacted_roots=[tmp_path])
    entry = store.write(
        component="api",
        event="request_error",
        message=f"path={tmp_path / 'private.txt'} authorization: Bearer-secret",
    )
    encoded = json.dumps(entry, ensure_ascii=False)
    assert str(tmp_path) not in encoded
    assert "<本地数据目录>" in encoded
    assert "<已隐藏>" in encoded


def test_raw_log_redacts_posix_paths_without_hiding_api_routes(tmp_path: Path) -> None:
    store = RawLogStore(tmp_path)
    redacted = store.redact("failed at /tmp/private/book/page.png; route=/api/jobs/123")

    assert "/tmp/private/book/page.png" not in redacted
    assert "<本地路径>" in redacted
    assert "/api/jobs/123" in redacted


def test_missing_nvml_library_falls_back_without_raising(
    tmp_path: Path, monkeypatch
) -> None:
    class NvmlUnavailable(Exception):
        pass

    store = RawLogStore(tmp_path / "logs")
    telemetry = SystemTelemetry(Settings(data_root=tmp_path / "jobs"), store)
    telemetry.stop()
    monkeypatch.setitem(
        sys.modules,
        "pynvml",
        SimpleNamespace(nvmlInit=lambda: (_ for _ in ()).throw(NvmlUnavailable("missing"))),
    )
    monkeypatch.setattr(
        "manga_repaint.observability.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )

    gpu, reason = telemetry._sample_gpu()

    assert gpu is None
    assert reason and "missing" in reason


def test_telemetry_reports_unavailable_gpu_without_fake_numbers(
    tmp_path: Path, monkeypatch
) -> None:
    store = RawLogStore(tmp_path / "logs")
    telemetry = SystemTelemetry(Settings(data_root=tmp_path / "jobs"), store)
    try:
        monkeypatch.setattr(telemetry, "_sample_gpu", lambda: (None, "测试环境没有 GPU"))
        payload = telemetry.sample()
        assert payload["available"] is False
        assert payload["reason"] == "测试环境没有 GPU"
        assert payload.get("memory_used_mib") is None
    finally:
        telemetry.stop()


def test_telemetry_persists_at_lower_rate_than_live_samples(
    tmp_path: Path, monkeypatch
) -> None:
    store = RawLogStore(tmp_path / "logs")
    telemetry = SystemTelemetry(Settings(data_root=tmp_path / "jobs"), store)
    telemetry.stop()
    monkeypatch.setattr(telemetry, "_sample_gpu", lambda: (None, "测试环境没有 GPU"))
    moments = iter((100.0, 100.0, 105.0, 111.0, 111.0))
    monkeypatch.setattr("manga_repaint.observability.time.monotonic", lambda: next(moments))
    telemetry._last_persisted = 0.0
    before = len(store.read(kind="gpu"))

    telemetry.sample()
    telemetry.sample()
    telemetry.sample()

    assert len(store.read(kind="gpu")) == before + 2
