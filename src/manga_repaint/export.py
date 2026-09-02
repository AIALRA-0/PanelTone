from __future__ import annotations

import io
import zipfile
from pathlib import Path

from PIL import Image


def export_cbz(pages: list[Path], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for index, page in enumerate(pages):
            archive.write(page, arcname=f"page_{index:05d}.png")
    return output


def export_image_archive(
    pages: list[Path], output: Path, *, image_format: str, extension: str
) -> Path:
    """Package consistently encoded page images for non-CBZ readers."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for index, page in enumerate(pages):
            buffer = io.BytesIO()
            with Image.open(page) as image:
                converted = image.convert("RGB")
                save_options = {"format": image_format}
                if image_format == "JPEG":
                    save_options.update(quality=95, optimize=True, progressive=True)
                else:
                    save_options["lossless"] = True
                converted.save(buffer, **save_options)
            archive.writestr(f"page_{index:05d}.{extension}", buffer.getvalue())
    return output


def export_pdf(pages: list[Path], output: Path, dpi: int = 300) -> Path:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PDF export requires PyMuPDF") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    try:
        for page_path in pages:
            pixmap = fitz.Pixmap(str(page_path))
            width_points = pixmap.width * 72 / dpi
            height_points = pixmap.height * 72 / dpi
            page = document.new_page(width=width_points, height=height_points)
            page.insert_image(page.rect, filename=str(page_path))
        document.save(output, garbage=4, deflate=True)
    finally:
        document.close()
    return output


def export_book(pages: list[Path], output_dir: Path, output_format: str) -> Path:
    if not pages:
        raise ValueError("Cannot export a book without completed pages")
    output_format = output_format.casefold()
    if output_format == "cbz":
        return export_cbz(pages, output_dir / "book.cbz")
    if output_format == "pdf":
        return export_pdf(pages, output_dir / "book.pdf")
    if output_format == "images":
        return export_image_archive(
            pages, output_dir / "book-images.zip", image_format="PNG", extension="png"
        )
    if output_format == "jpeg":
        return export_image_archive(
            pages, output_dir / "book-jpeg.zip", image_format="JPEG", extension="jpg"
        )
    if output_format == "webp":
        return export_image_archive(
            pages, output_dir / "book-webp.zip", image_format="WEBP", extension="webp"
        )
    raise ValueError(f"Unsupported output format: {output_format}")
