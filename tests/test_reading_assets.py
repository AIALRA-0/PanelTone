from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
from PIL import Image

from manga_repaint.reading_assets import PROFILES, ReadingAssetCache


def test_reading_profiles_are_bounded_immutable_and_do_not_modify_original(tmp_path: Path):
    rng = np.random.default_rng(18)
    source = tmp_path / "source.png"
    Image.fromarray(rng.integers(0, 256, (1900, 1300, 3), dtype=np.uint8)).save(source)
    original = source.read_bytes()
    cache = ReadingAssetCache(tmp_path / "cache")
    for profile, (long_edge, budget, _) in PROFILES.items():
        output = cache.build(source, profile)
        with Image.open(output) as image:
            assert max(image.size) <= long_edge
        assert output.stat().st_size <= budget
        stamp = output.stat().st_mtime_ns
        assert cache.request(source, profile) == output
        assert output.stat().st_mtime_ns == stamp
    assert source.read_bytes() == original
    first = cache.path(source, "reader")
    Image.new("RGB", (120, 160), "red").save(source)
    assert cache.path(source, "reader") != first
    assert first.is_file()


def test_missing_asset_is_deduplicated_and_hot_reads_do_not_wait(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.png"
    Image.new("RGB", (120, 160), "red").save(source)
    cache = ReadingAssetCache(tmp_path / "cache")
    hot = cache.build(source, "preview")
    started, release = threading.Event(), threading.Event()
    original = cache.build
    calls = []

    def blocked(source, profile):
        calls.append(profile)
        started.set()
        assert release.wait(5)
        return original(source, profile)

    monkeypatch.setattr(cache, "build", blocked)
    assert cache.request(source, "reader") is None
    assert started.wait(2)
    try:
        for _ in range(30):
            assert cache.request(source, "reader") is None
            assert cache.request(source, "preview") == hot
        assert calls == ["reader"]
    finally:
        release.set()
        cache._queue.join()
    assert cache.request(source, "reader").is_file()


def test_staged_preview_is_ready_immediately_after_atomic_rename(tmp_path: Path):
    staged = tmp_path / "staged.png"
    live = tmp_path / "live.png"
    Image.new("RGB", (300, 400), "orange").save(staged)
    cache = ReadingAssetCache(tmp_path / "cache")
    cache.prepare_publication(staged, live)
    assert not live.exists()
    staged.replace(live)
    assert cache.request(live, "reader").is_file()
    assert cache.request(live, "preview").is_file()
