"""
Konversi PDF menjadi gambar per-halaman.

Output: `<nama_pdf>_page<N>.jpg` di direktori yang sama (atau `output_dir` kustom).
Resolusi optimal (~200 DPI) menghasilkan gambar terbaca untuk OCR/VLM tanpa
ukuran file berlebihan.
"""
from __future__ import annotations

from pathlib import Path

import pymupdf

# 200 DPI adalah titik seimbang: cukup tajam untuk teks/OCR, wajar ukurannya.
DEFAULT_DPI = 200
# Format gambar output (JPEG cocok untuk dokumen; PNG lebih berat tapi lossless).
DEFAULT_IMAGE_EXT = ".jpg"


def pdf_to_images(
    pdf_path: str | Path,
    output_dir: str | Path | None = None,
    dpi: int = DEFAULT_DPI,
    image_ext: str = DEFAULT_IMAGE_EXT,
) -> list[Path]:
    """
    Ubah seluruh halaman PDF menjadi gambar, satu file per halaman.

    Nama file: `<nama_pdf>_page<N>.<ext>` (N mulai dari 1).
    Mengembalikan daftar path gambar yang dihasilkan (berurutan).

    Args:
        pdf_path: path file PDF.
        output_dir: direktori output (default: direktori PDF berada).
        dpi: resolusi dalam DPI (default 200).
        image_ext: ekstensi gambar (default ".jpg").

    Raises:
        FileNotFoundError: jika PDF tidak ada.
        ValueError: jika bukan PDF valid / tidak ada halaman.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF tidak ditemukan: {pdf_path}")

    output_dir = Path(output_dir) if output_dir else pdf_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # pymupdf memakai matriks zoom; 1 unit ≈ 72 DPI, jadi zoom = dpi / 72.
    zoom = dpi / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)

    stem = pdf_path.stem
    image_ext = image_ext if image_ext.startswith(".") else f".{image_ext}"

    outputs: list[Path] = []
    with pymupdf.open(pdf_path) as doc:
        if doc.page_count == 0:
            raise ValueError(f"PDF tidak memiliki halaman: {pdf_path}")

        for i, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            out_path = output_dir / f"{stem}_page{i}{image_ext}"
            pix.save(str(out_path))
            outputs.append(out_path)

    return outputs
