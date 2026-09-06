from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True, slots=True)
class ReferenceMatch:
    path: Path
    score: float
    source: str = "automatic"


class ColorReferenceLibrary:
    """Small local-only index for automatic manga colour reference retrieval.

    The index stores only paths and compact image features beside the user's
    data root. It never copies a page into the repository or sends it to a
    remote service. Existing identity records remain the source of truth for
    manual corrections; this index only chooses useful coloured examples.
    """

    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root.resolve()
        self.storage_root = self.data_root.parent / "color-library"
        self.index_path = self.storage_root / "reference-index.json"

    @staticmethod
    def _feature(path: Path) -> list[float]:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR))
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]).flatten()
        hist = hist / max(float(hist.sum()), 1.0)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        thumb = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA).flatten()
        aspect = np.asarray([math.log(max(rgb.shape[1], 1) / max(rgb.shape[0], 1))])
        return np.concatenate((hist.astype(np.float32), thumb, aspect)).round(6).tolist()

    def _candidate_paths(self, extra: Iterable[Path] = ()) -> list[Path]:
        paths: set[Path] = set()
        # The live root keeps each job below ``jobs/<job_id>``; include the
        # extra direct layout as well for small isolated fixtures.
        patterns = ("jobs/*/final/pages/page_*.png", "*/final/pages/page_*.png")
        for pattern in patterns:
            for path in self.data_root.glob(pattern):
                if path.is_file():
                    paths.add(path.resolve())
        for path in extra:
            candidate = Path(path).resolve()
            if candidate.is_file():
                paths.add(candidate)
        return sorted(paths)

    def refresh(self, extra: Iterable[Path] = ()) -> int:
        candidates = self._candidate_paths(extra)
        old: dict[str, dict[str, object]] = {}
        if self.index_path.is_file():
            try:
                payload = json.loads(self.index_path.read_text(encoding="utf-8"))
                old = {str(item["path"]): item for item in payload.get("items", [])}
            except (OSError, ValueError, KeyError, TypeError):
                old = {}
        items: list[dict[str, object]] = []
        for path in candidates:
            try:
                stat = path.stat()
                key = str(path)
                cached = old.get(key)
                if (
                    cached
                    and cached.get("mtime_ns") == stat.st_mtime_ns
                    and cached.get("size") == stat.st_size
                ):
                    items.append(cached)
                    continue
                items.append(
                    {
                        "path": key,
                        "mtime_ns": stat.st_mtime_ns,
                        "size": stat.st_size,
                        "feature": self._feature(path),
                    }
                )
            except (OSError, ValueError):
                continue
        self.storage_root.mkdir(parents=True, exist_ok=True)
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"schema_version": 1, "items": items}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self.index_path)
        return len(items)

    def retrieve(
        self,
        query: Path,
        *,
        limit: int = 6,
        exclude: Iterable[Path] = (),
    ) -> list[ReferenceMatch]:
        if limit <= 0:
            return []
        self.refresh(extra=(query,))
        try:
            query_feature = np.asarray(self._feature(query), dtype=np.float32)
        except (OSError, ValueError):
            return []
        excluded = {str(Path(path).resolve()) for path in exclude}
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
            items = payload.get("items", [])
        except (OSError, ValueError, TypeError):
            return []
        matches: list[ReferenceMatch] = []
        for item in items:
            try:
                path = Path(str(item["path"])).resolve()
                if not path.is_file() or str(path) in excluded:
                    continue
                feature = np.asarray(item["feature"], dtype=np.float32)
                distance = float(np.linalg.norm(query_feature - feature))
                matches.append(ReferenceMatch(path, 1.0 / (1.0 + distance)))
            except (OSError, TypeError, ValueError, KeyError):
                continue
        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[:limit]

    def identity_hints(self, records: Iterable[dict[str, object]]) -> list[str]:
        hints: list[str] = []
        for record in records:
            if not record.get("locked") or not record.get("color"):
                continue
            label = str(record.get("label") or record.get("identity_id") or "object")
            region = str(record.get("region") or "region")
            color = str(record["color"])
            shadow = str(record.get("shadow_color") or "")
            hints.append(
                f"{label} {region} uses base color {color}"
                + (f" with shadow {shadow}" if shadow else "")
            )
        return hints[:40]


__all__ = ["ColorReferenceLibrary", "ReferenceMatch"]
