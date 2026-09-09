"""Small, immutable reading derivatives; never encode in an HTTP request.

The cache is disposable, separate from archival results, and revision-addressed.
One bounded daemon queue prevents rapid scrolling from spawning encoder threads.
"""
from __future__ import annotations

import hashlib
import io
import logging
import queue
import shutil
import threading
from pathlib import Path

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)
PROFILES = {"reader": (1280, 300 * 1024, (80, 76, 72)),
            "preview": (480, 48 * 1024, (74, 68, 62))}


class ReadingAssetCache:
    def __init__(self, root: Path):
        self.root = root
        self._lock = threading.Lock()
        self._encode_lock = threading.Lock()
        self._pending: set[Path] = set()
        self._queue: queue.Queue[tuple[Path, Path, str]] = queue.Queue(maxsize=1024)
        self._worker: threading.Thread | None = None

    def path(self, source: Path, profile: str) -> Path:
        if profile not in PROFILES:
            raise ValueError("Unknown reading size")
        stat = source.stat()
        return self._version_path(source, profile, stat.st_mtime_ns, stat.st_size)

    def _version_path(self, source: Path, profile: str, mtime: int, size: int) -> Path:
        identity = f"reader-v1:{source.resolve()}:{mtime}:{size}:{profile}"
        digest = hashlib.sha256(identity.encode()).hexdigest()
        return self.root / digest[:2] / f"{digest}.webp"

    def prepare_publication(self, staged: Path, live: Path) -> None:
        """Prepare both tiers under their future live identity before the swap.

        The transaction preserves file mtime and size on same-volume rename.
        Unused derivatives after rollback are harmless; no live file is changed.
        """
        for profile in PROFILES:
            encoded = self.build(staged, profile)
            stat = staged.stat()
            target = self._version_path(live, profile, stat.st_mtime_ns, stat.st_size)
            with self._encode_lock:
                if target.is_file():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(".tmp")
                shutil.copyfile(encoded, temporary)
                temporary.replace(target)

    def build(self, source: Path, profile: str) -> Path:
        output = self.path(source, profile)
        if output.is_file():
            return output
        # Maintenance backfills and the queue share this lock, HTTP readers do not.
        with self._encode_lock:
            if output.is_file():
                return output
            long_edge, budget, qualities = PROFILES[profile]
            with Image.open(source) as original:
                original.load()
                display = ImageOps.exif_transpose(original).convert("RGB")
                display.thumbnail((long_edge, long_edge), Image.Resampling.LANCZOS)
                while True:
                    for quality in qualities:
                        buffer = io.BytesIO()
                        display.save(buffer, format="WEBP", quality=quality, method=4)
                        payload = buffer.getvalue()
                        if len(payload) <= budget:
                            break
                    if len(payload) <= budget:
                        break
                    display.thumbnail((max(1, int(display.width * .85)),
                                       max(1, int(display.height * .85))),
                                      Image.Resampling.LANCZOS)
            # Never label bytes generated from a changed source with an old revision.
            if self.path(source, profile) != output:
                raise OSError("Source changed while building reading preview")
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(".tmp")
            temporary.write_bytes(payload)
            temporary.replace(output)
        return output

    def request(self, source: Path, profile: str) -> Path | None:
        output = self.path(source, profile)
        if output.is_file():
            return output
        with self._lock:
            if output not in self._pending:
                try:
                    self._queue.put_nowait((source, output, profile))
                except queue.Full:
                    return None
                self._pending.add(output)
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._run, daemon=True,
                                                name="paneltone-reading-cache")
                self._worker.start()
        return None

    def _run(self) -> None:
        while True:
            source, output, profile = self._queue.get()
            try:
                self.build(source, profile)
            except (OSError, ValueError):
                logger.exception("Reading preview preparation failed")
            finally:
                with self._lock:
                    self._pending.discard(output)
                self._queue.task_done()
