from __future__ import annotations

import contextlib
import csv
import json
import logging
import re
import shutil
import subprocess
import threading
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import Settings

logger = logging.getLogger("paneltone.observability")

WINDOWS_LOCAL_PATH = re.compile(
    r"(?i)\b[A-Z]:\\(?:[^\\/:*?\"<>|\r\n]+\\)*[^\\/:*?\"<>|\r\n]*"
)
POSIX_LOCAL_PATH = re.compile(
    r"(?<![:/A-Za-z0-9_.-])/(?:tmp|home|var|Users|mnt|opt|srv|etc|run|root|"
    r"private|usr|workspace|app|data)(?:/[^\s\"'<>|,;:]+)+"
)
SECRET_ASSIGNMENT = re.compile(
    r"(?i)(authorization|access[_-]?token|api[_-]?key|secret|password)"
    r"(\s*[:=]\s*)([^\s,;]+)"
)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def redact_sensitive_text(value: str, redacted_roots: Iterable[Path] = ()) -> str:
    """Hide local filesystem paths and credential-like assignments."""
    result = value
    for root in redacted_roots:
        resolved = root.resolve()
        for representation in {str(resolved), resolved.as_posix()}:
            result = result.replace(representation, "<本地数据目录>")
    result = WINDOWS_LOCAL_PATH.sub("<本地路径>", result)
    result = POSIX_LOCAL_PATH.sub("<本地路径>", result)
    return SECRET_ASSIGNMENT.sub(r"\1\2<已隐藏>", result)


class RawLogStore:
    """Small, local-only JSONL store for troubleshooting and reproducibility."""

    def __init__(
        self,
        root: Path,
        *,
        redacted_roots: Iterable[Path] = (),
        max_bytes: int = 20 * 1024 * 1024,
        backups: int = 3,
    ) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.redacted_roots = tuple(path.resolve() for path in redacted_roots)
        self.max_bytes = max(1024 * 1024, max_bytes)
        self.backups = max(1, backups)
        self._lock = threading.RLock()
        # Event IDs are sent to the browser as cursors.  Reusing ``1`` after
        # an application restart makes Last-Event-ID ambiguous and can cause
        # logs to disappear or be shown twice.  Recover the largest ID once at
        # startup so IDs remain monotonic across normal restarts and rotations.
        self._sequence = self._read_last_sequence()

    def _read_last_sequence(self) -> int:
        last = 0
        for path in self.root.glob("paneltone-*.jsonl*"):
            try:
                with path.open("rb") as stream:
                    for line in stream:
                        try:
                            value = json.loads(line.decode("utf-8"))
                            last = max(last, int(value.get("id", 0)))
                        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
                            continue
            except OSError:
                continue
        return last

    def redact(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self.redact(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.redact(item) for item in value]
        if isinstance(value, tuple):
            return [self.redact(item) for item in value]
        if not isinstance(value, str):
            return value
        return redact_sensitive_text(value, self.redacted_roots)

    def _path_for(self, component: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", component).strip("-").lower() or "runtime"
        return self.root / f"paneltone-{safe}.jsonl"

    def _rotate(self, path: Path, next_size: int) -> None:
        if not path.is_file() or path.stat().st_size + next_size <= self.max_bytes:
            return
        for index in range(self.backups, 0, -1):
            old = path.with_name(f"{path.name}.{index}")
            if index == self.backups:
                old.unlink(missing_ok=True)
            else:
                old.replace(path.with_name(f"{path.name}.{index + 1}")) if old.exists() else None
        path.replace(path.with_name(f"{path.name}.1"))

    def write(
        self,
        *,
        level: str = "INFO",
        component: str,
        event: str,
        message: str,
        error_code: str | None = None,
        job_id: str | None = None,
        page_index: int | None = None,
        unit_index: int | None = None,
        metrics: dict[str, Any] | None = None,
        kind: str = "raw",
    ) -> dict[str, Any]:
        with self._lock:
            self._sequence += 1
            entry: dict[str, Any] = {
                "id": self._sequence,
                "timestamp": _timestamp(),
                "level": level.upper(),
                "component": component,
                "job_id": job_id,
                "page_index": page_index,
                "unit_index": unit_index,
                "event": event,
                "message": message,
                "error_code": error_code,
                "metrics": metrics or {},
                "kind": kind,
            }
            entry = self.redact(entry)
            encoded = (json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
            path = self._path_for(component)
            self._rotate(path, len(encoded))
            with path.open("ab") as stream:
                stream.write(encoded)
            return entry

    def _iter_paths(self) -> list[Path]:
        return sorted(
            self.root.glob("paneltone-*.jsonl*"),
            key=lambda path: path.stat().st_mtime_ns if path.exists() else 0,
        )

    def read(
        self,
        *,
        kind: str = "raw",
        job_id: str | None = None,
        level: str | None = None,
        component: str | None = None,
        limit: int = 200,
        after: int | None = None,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for path in self._iter_paths():
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if kind == "gpu" and item.get("kind") != "gpu" and item.get("component") != "gpu":
                    continue
                if kind == "raw" and item.get("kind") == "gpu":
                    continue
                if job_id and item.get("job_id") != job_id:
                    continue
                if level and str(item.get("level", "")).upper() != level.upper():
                    continue
                if component and item.get("component") != component:
                    continue
                if after is not None and int(item.get("id", 0)) <= after:
                    continue
                records.append(item)
        records.sort(key=lambda item: int(item.get("id", 0)))
        return records[-max(1, min(limit, 20_000)) :]

    def export(self, *, kind: str = "raw", job_id: str | None = None) -> bytes:
        records = self.read(kind=kind, job_id=job_id, limit=20_000)
        return b"".join(
            (json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            for item in records
        )


def _float_or_none(value: str) -> float | None:
    try:
        return float(value.strip().replace("%", ""))
    except (TypeError, ValueError):
        return None


def _int_or_none(value: str) -> int | None:
    try:
        return int(float(value.strip()))
    except (TypeError, ValueError):
        return None


class SystemTelemetry:
    """Best-effort local telemetry with explicit unavailable states."""

    def __init__(
        self,
        settings: Settings,
        store: RawLogStore,
        event_callback: Callable[[str, dict[str, Any], str | None], None] | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.event_callback = event_callback
        self._stop = threading.Event()
        self._busy = threading.Event()
        self._lock = threading.RLock()
        self._latest: dict[str, Any] = {
            "timestamp": _timestamp(),
            "available": False,
            "reason": "尚未采集本机指标",
        }
        self._thread = threading.Thread(
            target=self._run,
            name="paneltone-system-telemetry",
            daemon=True,
        )
        self._thread.start()

    def set_busy(self, busy: bool) -> None:
        if busy:
            self._busy.set()
        else:
            self._busy.clear()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=max(0.0, timeout))

    def current(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._latest)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.sample()
            except Exception:
                logger.exception("telemetry sample failed")
            interval = (
                self.settings.telemetry_gpu_interval_seconds
                if self._busy.is_set()
                else self.settings.telemetry_idle_interval_seconds
            )
            self._stop.wait(max(0.25, interval))

    def _sample_gpu(self) -> tuple[dict[str, Any] | None, str | None]:
        # NVML avoids spawning a process every second and exposes the same
        # counters on Windows and Linux.  It is optional so a clean install
        # still works with the nvidia-smi fallback below.
        try:
            import pynvml

            pynvml.nvmlInit()
            try:
                count = int(pynvml.nvmlDeviceGetCount())
                if count <= 0:
                    return None, "NVML 没有检测到 GPU"
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode(errors="replace")
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                except Exception:
                    temperature = None
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                except Exception:
                    power = None
                try:
                    driver = pynvml.nvmlSystemGetDriverVersion()
                    if isinstance(driver, bytes):
                        driver = driver.decode(errors="replace")
                except Exception:
                    driver = None
                process_count: int | None = None
                process_functions = (
                    "nvmlDeviceGetComputeRunningProcesses",
                    "nvmlDeviceGetGraphicsRunningProcesses",
                )
                process_ids: set[int] = set()
                process_api_available = False
                for function_name in process_functions:
                    function = getattr(pynvml, function_name, None)
                    if not callable(function):
                        continue
                    process_api_available = True
                    try:
                        process_ids.update(
                            int(process.pid)
                            for process in (function(handle) or [])
                            if getattr(process, "pid", None) is not None
                        )
                    except Exception:
                        # NVML may deny process enumeration while still
                        # allowing the normal utilization counters.
                        continue
                if process_api_available:
                    process_count = len(process_ids)
                return {
                    "gpu_index": 0,
                    "name": str(name) if name else None,
                    "utilization_percent": float(utilization.gpu),
                    "memory_used_mib": int(memory.used / 2**20),
                    "memory_total_mib": int(memory.total / 2**20),
                    "temperature_c": float(temperature) if temperature is not None else None,
                    "power_w": float(power) if power is not None else None,
                    "driver_version": str(driver) if driver else None,
                    "process_count": process_count,
                }, None
            finally:
                with contextlib.suppress(Exception):
                    pynvml.nvmlShutdown()
        except Exception as exc:
            # NVML bindings use their own exception hierarchy. A machine
            # without the shared library is an expected optional-probe miss,
            # so fall back to nvidia-smi instead of surfacing a thread error.
            nvml_reason = str(exc)

        query = (
            "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,"
            "power.draw,driver_version"
        )
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-gpu={query}",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except (FileNotFoundError, OSError, subprocess.SubprocessError) as exc:
            reason = nvml_reason if "nvml_reason" in locals() else str(exc)
            return None, f"无法调用 NVML 或 nvidia-smi: {reason}"
        if completed.returncode != 0:
            return None, (completed.stderr.strip() or "nvidia-smi 返回失败")
        rows = list(csv.reader(line for line in completed.stdout.splitlines() if line.strip()))
        if not rows or len(rows[0]) < 8:
            return None, "nvidia-smi 没有返回 GPU 指标"
        row = [item.strip() for item in rows[0]]
        process_count = self._sample_process_count_smi()
        return {
            "gpu_index": _int_or_none(row[0]),
            "name": row[1] or None,
            "utilization_percent": _float_or_none(row[2]),
            "memory_used_mib": _int_or_none(row[3]),
            "memory_total_mib": _int_or_none(row[4]),
            "temperature_c": _float_or_none(row[5]),
            "power_w": _float_or_none(row[6]),
            "driver_version": row[7] or None,
            "process_count": process_count,
        }, None

    @staticmethod
    def _sample_process_count_smi() -> int | None:
        """Return the number of GPU compute/graphics processes when exposed.

        ``nvidia-smi --query-gpu`` does not include a process count. Keep this
        second query best-effort so a permissions-restricted driver still
        reports all other GPU fields instead of inventing a zero.
        """
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            if result.returncode != 0:
                return None
            counts: set[int] = set()
            unavailable = False
            for line in result.stdout.splitlines():
                value = line.strip().split(",", 1)[0].strip()
                if not value:
                    continue
                try:
                    counts.add(int(value))
                except ValueError:
                    # WDDM and permission-restricted drivers commonly return
                    # ``N/A``.  That is an unknown count, not evidence of
                    # zero processes.
                    unavailable = True
            if unavailable and not counts:
                return None
            return len(counts)
        except (FileNotFoundError, OSError, subprocess.SubprocessError):
            return None

    def _sample_system(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        try:
            import psutil

            result["cpu_percent"] = float(psutil.cpu_percent(interval=None))
            result["memory_percent"] = float(psutil.virtual_memory().percent)
        except ImportError:
            result["system_reason"] = "未安装 psutil，CPU 和内存指标不可用"
        except Exception as exc:
            result["system_reason"] = f"系统指标不可用: {exc}"
        try:
            result["disk_free_gib"] = round(
                shutil.disk_usage(self.settings.data_root).free / 2**30, 2
            )
        except OSError as exc:
            result["disk_reason"] = f"磁盘指标不可用: {exc}"
        return result

    def sample(self) -> dict[str, Any]:
        gpu, reason = self._sample_gpu()
        payload: dict[str, Any] = {
            "timestamp": _timestamp(),
            "available": gpu is not None,
            "reason": reason,
            **(gpu or {}),
            **self._sample_system(),
        }
        payload = self.store.redact(payload)
        with self._lock:
            self._latest = payload
        self.store.write(
            component="gpu",
            event="metrics",
            message="GPU 和系统指标已采集" if gpu else "GPU 指标不可用，已记录原因",
            metrics=payload,
            kind="gpu",
            level="INFO" if gpu else "WARNING",
        )
        if self.event_callback:
            self.event_callback("gpu_metrics", payload, None)
        return payload


__all__ = ["RawLogStore", "SystemTelemetry", "redact_sensitive_text"]
