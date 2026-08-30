from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from PIL import Image

from .hashing import sha256_file

SUPPORTED_IMAGES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


def _natural_key(path: Path) -> list[str | int]:
    return [
        int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", path.name)
    ]


def _safe_member_path(root: Path, member_name: str) -> Path:
    candidate = (root / member_name).resolve()
    if root.resolve() not in candidate.parents and candidate != root.resolve():
        raise ValueError(f"Archive member escapes extraction root: {member_name}")
    return candidate


def _copy_normalized(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.convert("RGB").save(destination, format="PNG", compress_level=6)


def _ingest_directory(source: Path, pages_dir: Path) -> list[Path]:
    images = sorted(
        (path for path in source.rglob("*") if path.suffix.casefold() in SUPPORTED_IMAGES),
        key=_natural_key,
    )
    if not images:
        raise ValueError(f"No supported images found in {source}")
    outputs: list[Path] = []
    for index, image_path in enumerate(images):
        output = pages_dir / f"page_{index:05d}.png"
        _copy_normalized(image_path, output)
        outputs.append(output)
    return outputs


def _ingest_cbz(
    source: Path,
    pages_dir: Path,
    max_members: int = 5000,
    max_ratio: int = 200,
) -> list[Path]:
    with tempfile.TemporaryDirectory(prefix="manga-repaint-cbz-") as temp_name:
        temp_root = Path(temp_name)
        with zipfile.ZipFile(source) as archive:
            all_members = [item for item in archive.infolist() if not item.is_dir()]
            if len(all_members) > max_members:
                raise ValueError(f"Archive has more than {max_members} files")
            nested = {".zip", ".cbz", ".rar", ".cbr", ".7z"}
            if any(Path(item.filename).suffix.casefold() in nested for item in all_members):
                raise ValueError("Nested archives are not supported")
            for item in all_members:
                compressed = max(1, item.compress_size)
                if item.file_size / compressed > max_ratio:
                    raise ValueError(
                        f"Archive member exceeds compression ratio limit: {item.filename}"
                    )
            members = [
                item
                for item in all_members
                if not item.is_dir() and Path(item.filename).suffix.casefold() in SUPPORTED_IMAGES
            ]
            if not members:
                raise ValueError(f"No supported images found in {source}")
            extracted: list[Path] = []
            for member in members:
                target = _safe_member_path(temp_root, member.filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted.append(target)
        return _ingest_file_list(sorted(extracted, key=_natural_key), pages_dir)


def _ingest_cbr(source: Path, pages_dir: Path, max_members: int = 5000) -> list[Path]:
    known_windows_path = Path("C:/Program Files/7-Zip/7z.exe")
    seven_zip = (
        shutil.which("7z")
        or shutil.which("7zz")
        or (str(known_windows_path) if known_windows_path.is_file() else None)
    )
    if seven_zip is None:
        raise RuntimeError("CBR input requires 7-Zip with 7z.exe available on PATH")
    with tempfile.TemporaryDirectory(prefix="manga-repaint-cbr-") as temp_name:
        temp_root = Path(temp_name).resolve()
        completed = subprocess.run(
            [seven_zip, "x", str(source.resolve()), f"-o{temp_root}", "-y"],
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"7-Zip could not extract CBR: {completed.stderr.strip()}")
        for path in temp_root.rglob("*"):
            if path.is_symlink() or temp_root not in path.resolve().parents:
                raise ValueError(f"Unsafe CBR member detected: {path}")
            if path.is_file() and path.suffix.casefold() in {".zip", ".cbz", ".rar", ".cbr", ".7z"}:
                raise ValueError("Nested archives are not supported")
        images = sorted(
            (path for path in temp_root.rglob("*") if path.suffix.casefold() in SUPPORTED_IMAGES),
            key=_natural_key,
        )
        if len(images) > max_members:
            raise ValueError(f"Archive has more than {max_members} images")
        return _ingest_file_list(images, pages_dir)


def _ingest_pdf(source: Path, pages_dir: Path, render_dpi: int = 300) -> list[Path]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PDF input requires PyMuPDF") from exc

    outputs: list[Path] = []
    with fitz.open(source) as document:
        if document.page_count == 0:
            raise ValueError(f"PDF has no pages: {source}")
        for index, page in enumerate(document):
            output = pages_dir / f"page_{index:05d}.png"
            images = page.get_images(full=True)
            used_original = False
            if len(images) == 1:
                extracted = document.extract_image(images[0][0])
                raw_path = pages_dir / f".page_{index:05d}.{extracted['ext']}"
                raw_path.write_bytes(extracted["image"])
                try:
                    with Image.open(raw_path) as image:
                        coverage = (image.width * image.height) / max(
                            1, page.rect.width * page.rect.height
                        )
                        if coverage >= 4:
                            _copy_normalized(raw_path, output)
                            used_original = True
                finally:
                    raw_path.unlink(missing_ok=True)
            if not used_original:
                scale = render_dpi / 72
                pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                pixmap.save(str(output))
            outputs.append(output)
    return outputs


def _ingest_file_list(images: list[Path], pages_dir: Path) -> list[Path]:
    if not images:
        raise ValueError("Archive contains no supported images")
    outputs: list[Path] = []
    for index, image_path in enumerate(images):
        output = pages_dir / f"page_{index:05d}.png"
        _copy_normalized(image_path, output)
        outputs.append(output)
    return outputs


def ingest_book(
    source: Path,
    pages_dir: Path,
    max_archive_members: int = 5000,
    max_archive_ratio: int = 200,
) -> list[Path]:
    source = source.resolve(strict=True)
    pages_dir.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        return _ingest_directory(source, pages_dir)
    suffix = source.suffix.casefold()
    if suffix == ".cbz" or suffix == ".zip":
        return _ingest_cbz(source, pages_dir, max_archive_members, max_archive_ratio)
    if suffix == ".cbr" or suffix == ".rar":
        return _ingest_cbr(source, pages_dir, max_archive_members)
    if suffix == ".pdf":
        return _ingest_pdf(source, pages_dir)
    if suffix in SUPPORTED_IMAGES:
        return _ingest_file_list([source], pages_dir)
    raise ValueError(f"Unsupported source type: {source.suffix}")


def page_metadata(path: Path) -> tuple[str, int, int]:
    with Image.open(path) as image:
        width, height = image.size
    return sha256_file(path), width, height
