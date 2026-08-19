"""
Konversi PDF menjadi gambar per-halaman & pemrosesan dokumen PDF multi-halaman.

Menyediakan:
  - `pdf_to_images`: Konversi PDF menjadi gambar beresolusi optimal (~200 DPI).
  - `process_multipage_pdf`: Mengekstrak seluruh halaman PDF, menjaga kontinuitas header,
    dan menyatukannya menjadi `ExtractedDocument` Markdown yang siap di-chunking.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pymupdf

from .multi_page import stitch_pages_to_markdown
from .prompts import normalize_specs
from .schemas import DocumentPage, ExtractedDocument

if TYPE_CHECKING:
    from .graph import DocumentExtractionPipeline

# 200 DPI adalah titik seimbang: cukup tajam untuk teks/OCR, wajar ukurannya.
DEFAULT_DPI: int = 400
DEFAULT_IMAGE_EXT: str = ".jpg"


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
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF tidak ditemukan: {pdf_path}")

    output_dir = Path(output_dir) if output_dir else pdf_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

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


def process_multipage_pdf(
    pdf_path: str | Path,
    pipeline: DocumentExtractionPipeline,
    output_dir: str | Path | None = None,
    dpi: int = DEFAULT_DPI,
    forced_specs: list[str] | str | None = None,
    forced_doc_type: str | None = None,
) -> ExtractedDocument:
    """
    Proses seluruh halaman PDF dan gabungkan hasil ekstraksi menjadi teks Markdown utuh siap chunking.
    Mendukung multi-spesifikasi komposit per-halaman.
    """
    pdf_path = Path(pdf_path)
    page_images = pdf_to_images(pdf_path, output_dir=output_dir, dpi=dpi)

    pages: list[DocumentPage] = []
    pages_md: list[str] = []
    all_page_specs: list[list[str]] = []
    previous_context: str | None = None

    active_forced = forced_specs or forced_doc_type

    for idx, img_path in enumerate(page_images, start=1):
        # Jalankan pipeline per halaman dengan membawa konteks halaman sebelumnya
        res = pipeline.run(
            str(img_path),
            forced_specs=active_forced,
            previous_page_context=previous_context,
        )

        page_md: str = res["markdown_content"]
        ocr_txt: str | None = res.get("ocr_text")
        detected_specs: list[str] = res.get("specs") or ["plain"]

        pages_md.append(page_md)
        all_page_specs.append(detected_specs)
        previous_context = page_md[-400:] if len(page_md) > 400 else page_md

        pages.append(
            DocumentPage(
                page_number=idx,
                specs=detected_specs,
                markdown_content=page_md,
                ocr_text=ocr_txt,
                image_path=str(img_path),
            )
        )

    # Kumpulkan seluruh spesifikasi unik dokumen
    if active_forced:
        overall_specs = normalize_specs(active_forced)
    else:
        flat_specs: list[str] = []
        for s_list in all_page_specs:
            for s in s_list:
                if s not in flat_specs:
                    flat_specs.append(s)
        overall_specs = flat_specs or ["plain"]

    # Gabungkan halaman dengan kontinuitas heading
    stitched_markdown = stitch_pages_to_markdown(pages_md)

    return ExtractedDocument(
        file_path=str(pdf_path),
        specs=overall_specs,
        total_pages=len(page_images),
        markdown_content=stitched_markdown,
        pages=pages,
        metadata={
            "source_type": "pdf",
            "dpi": dpi,
            "filename": pdf_path.name,
        },
    )
