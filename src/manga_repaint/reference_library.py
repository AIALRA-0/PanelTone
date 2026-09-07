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

    @staticmethod
    def _colour_coverage(path: Path) -> float:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR))
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        return float(((hsv[..., 1] >= 24) & (hsv[..., 2] >= 24)).mean())

    def _candidate_paths(
        self,
        extra: Iterable[Path] = (),
        *,
        scope_root: Path | None = None,
    ) -> list[Path]:
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
        if scope_root is None:
            return sorted(paths)
        root = scope_root.resolve()
        return sorted(path for path in paths if self._within(path, root))

    @staticmethod
    def _within(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def refresh(
        self,
        extra: Iterable[Path] = (),
        *,
        scope_root: Path | None = None,
    ) -> int:
        candidates = self._candidate_paths(extra, scope_root=scope_root)
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
                    and "colour_coverage" in cached
                ):
                    items.append(cached)
                    continue
                items.append(
                    {
                        "path": key,
                        "mtime_ns": stat.st_mtime_ns,
                        "size": stat.st_size,
                        "feature": self._feature(path),
                        "colour_coverage": self._colour_coverage(path),
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
        scope_root: Path | None = None,
        use_payload_cache: bool = False,
        min_colour_coverage: float = 0.0,
    ) -> list[ReferenceMatch]:
        if limit <= 0:
            return []
        # Keep one complete on-disk index; apply the book scope while reading
        # matches so a scoped lookup cannot erase references for other jobs.
        self.refresh(extra=(query,))
        try:
            query_feature = np.asarray(self._feature(query), dtype=np.float32)
        except (OSError, ValueError):
            return []
        excluded = {str(Path(path).resolve()) for path in exclude}
        resolved_scope = scope_root.resolve() if scope_root is not None else None
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
                if resolved_scope is not None and not self._within(path, resolved_scope):
                    continue
                if float(item.get("colour_coverage", 0.0)) < min_colour_coverage:
                    continue
                feature = np.asarray(item["feature"], dtype=np.float32)
                distance = float(np.linalg.norm(query_feature - feature))
                match_path = self.payload_path(path) if use_payload_cache else path
                matches.append(ReferenceMatch(match_path, 1.0 / (1.0 + distance)))
            except (OSError, TypeError, ValueError, KeyError):
                continue
        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[:limit]

    def payload_path(self, path: Path, *, max_edge: int = 1024) -> Path:
        """Return a small cached reference image for model upload.

        Cobra resizes references internally. Uploading a full-resolution PNG for
        every page wastes time and memory, while a bounded RGB proxy preserves
        the palette information the candidate actually consumes. The cache is
        outside the repository and is invalidated by source mtime and size.
        """
        source = Path(path).resolve()
        stat = source.stat()
        key = f"{source}|{stat.st_mtime_ns}|{stat.st_size}|{max_edge}"
        import hashlib

        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        cache_dir = self.storage_root / "reference-payloads"
        target = cache_dir / f"{digest}.webp"
        if target.is_file():
            return target
        with Image.open(source) as image:
            rgb = image.convert("RGB")
            scale = min(1.0, max_edge / max(rgb.width, rgb.height))
            if scale < 1.0:
                rgb = rgb.resize(
                    (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))),
                    Image.Resampling.LANCZOS,
                )
            cache_dir.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".tmp.webp")
            rgb.save(temporary, format="WEBP", quality=88, method=4)
            temporary.replace(target)
        return target

    @staticmethod
    def palette_anchors(
        paths: Iterable[Path], *, max_anchors: int = 12
    ) -> list[tuple[float, float, float]]:
        """Extract stable hue/saturation anchors from coloured references.

        The anchors are advisory compositor input. They do not classify a
        pixel as skin, hair, or clothing; they only provide the book's observed
        colour vocabulary so a candidate cannot drift freely between pages.
        """
        histogram = np.zeros(36, dtype=np.float64)
        samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for path in paths:
            try:
                with Image.open(path) as image:
                    rgb = np.asarray(
                        image.convert("RGB").resize((192, 192), Image.Resampling.BILINEAR)
                    )
                hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
            except (OSError, ValueError):
                continue
            sat = hsv[..., 1]
            value = hsv[..., 2]
            mask = (sat >= 28) & (value >= 18) & (value <= 250)
            if not mask.any():
                continue
            hue = hsv[..., 0][mask]
            weights = (sat[mask] / 255.0).astype(np.float64)
            bins = np.clip((hue / 5.0).astype(np.int32), 0, 35)
            histogram += np.bincount(bins, weights=weights, minlength=36)
            samples.append((hue, sat[mask], weights))
        if not samples or histogram.max() <= 0:
            return []
        # Circular smoothing makes neighbouring red bins one palette family.
        smooth = sum(np.roll(histogram, offset) for offset in (-2, -1, 0, 1, 2)) / 5.0
        order = np.argsort(smooth)[::-1]
        chosen: list[int] = []
        for index in order:
            index = int(index)
            if all(min((index - other) % 36, (other - index) % 36) >= 3 for other in chosen):
                chosen.append(index)
            if len(chosen) >= max_anchors:
                break
        total = float(smooth[chosen].sum()) if chosen else 1.0
        anchors: list[tuple[float, float, float]] = []
        for index in chosen:
            center = index * 5.0 + 2.5
            values: list[tuple[float, float, float]] = []
            for hue, sat, weights in samples:
                distance = np.abs(hue - center)
                distance = np.minimum(distance, 180.0 - distance)
                keep = distance <= 10.0
                values.extend(zip(hue[keep], sat[keep], weights[keep], strict=False))
            if not values:
                continue
            array = np.asarray(values, dtype=np.float64)
            angle = array[:, 0] * np.pi / 90.0
            weight = array[:, 2]
            mean_angle = np.arctan2(
                np.sum(np.sin(angle) * weight), np.sum(np.cos(angle) * weight)
            )
            mean_hue = float((mean_angle * 90.0 / np.pi) % 180.0)
            mean_sat = float(np.average(array[:, 1], weights=weight))
            anchors.append((mean_hue, mean_sat, float(smooth[index] / total)))
        return anchors

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
