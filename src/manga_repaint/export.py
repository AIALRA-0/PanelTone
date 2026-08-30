from __future__ import annotations

import zipfile
from pathlib import Path


def export_cbz(pages: list[Path], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for index, page in enumerate(pages):
            archive.write(page, arcname=f"page_{index:05d}.png")
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
        return output_dir
    raise ValueError(f"Unsupported output format: {output_format}")
