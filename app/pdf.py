"""
Konversi PDF menjadi gambar per-halaman & pemrosesan dokumen PDF multi-halaman.

Menyediakan:
  - `pdf_to_images`: Konversi PDF menjadi gambar beresolusi optimal (~200 DPI).
  - `process_multipage_pdf`: Mengekstrak seluruh halaman PDF, menjaga kontinuitas header,
    dan menyatukannya menjadi `ExtractedDocument` Markdown yang siap di-chunking.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pymupdf

from .multi_page import stitch_pages_to_markdown
from .prompts import normalize_specs
from .schemas import DocumentPage, ExtractedDocument

if TYPE_CHECKING:
    from .graph import DocumentExtractionPipeline

# 200 DPI adalah titik seimbang: cukup tajam untuk teks/OCR, wajar ukurannya.
DEFAULT_DPI: int = 200
DEFAULT_IMAGE_EXT: str = ".jpg"

# Ukuran batch halaman per pemrosesan (dipertahankan 10 agar konsisten dengan
# strategi bertahap "proses per 10 slide/pages" dan meminimalkan beban memori/disk).
PDF_PAGE_BATCH: int = 10


def pdf_page_count(pdf_path: str | Path) -> int:
    """
    Hitung jumlah halaman file PDF menggunakan PyMuPDF.
    """
    path_obj = Path(pdf_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"PDF tidak ditemukan: {path_obj}")
    with pymupdf.open(path_obj) as doc:
        return doc.page_count


def pdf_to_images(
    pdf_path: str | Path,
    output_dir: str | Path | None = None,
    dpi: int = DEFAULT_DPI,
    image_ext: str = DEFAULT_IMAGE_EXT,
    max_pages: int | None = None,
    pages: list[int] | None = None,
) -> list[Path]:
    """
    Ubah halaman PDF menjadi gambar, satu file per halaman.

    Args:
        pdf_path: Path file PDF.
        output_dir: Direktori penyimpanan gambar.
        dpi: Resolusi gambar (default 200 DPI).
        image_ext: Format ekstensi gambar (.jpg / .png).
        max_pages: Batas maksimal halaman yang dirender dari awal.
        pages: Daftar nomor halaman spesifik (0-indexed) yang ingin dirender.

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
        total_in_doc = doc.page_count
        if total_in_doc == 0:
            raise ValueError(f"PDF tidak memiliki halaman: {pdf_path}")

        # Tentukan halaman mana saja yang akan diproses
        if pages is not None:
            target_indices = [idx for idx in pages if 0 <= idx < total_in_doc]
        else:
            limit = min(max_pages, total_in_doc) if (max_pages and max_pages > 0) else total_in_doc
            target_indices = list(range(limit))

        for idx in target_indices:
            page = doc[idx]
            page_num = idx + 1
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            out_path = output_dir / f"{stem}_page{page_num}{image_ext}"
            pix.save(str(out_path))
            outputs.append(out_path)

    return outputs


def extract_pdf_with_pymupdf4llm(
    pdf_path: str | Path,
    page_chunks: bool = True,
    write_images: bool = False,
    image_path: str | Path | None = None,
    pages: list[int] | None = None,
) -> list[dict[str, Any]] | str:
    """
    Ekstrak dokumen PDF digital langsung menjadi Markdown menggunakan pymupdf4llm.
    Mendukung deteksi tabel GFM, urutan baca multi-kolom, dan chunking per halaman.

    Args:
        pdf_path: Path file PDF.
        page_chunks: Jika True, mengembalikan list chunk per-halaman lengkap dengan metadata.
        write_images: Jika True, simpan gambar/grafik yang diekstrak dari PDF ke image_path.
        image_path: Direktori penyimpanan gambar jika write_images=True.
        pages: Daftar indeks halaman (0-indexed) yang ingin diekstrak.

    Returns:
        List dictionary per-halaman jika page_chunks=True, atau string Markdown utuh jika False.
    """
    import pymupdf4llm

    path_obj = Path(pdf_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"PDF tidak ditemukan: {path_obj}")

    img_dir_str = str(Path(image_path).resolve()) if image_path else None
    if write_images and img_dir_str:
        Path(img_dir_str).mkdir(parents=True, exist_ok=True)

    result = pymupdf4llm.to_markdown(
        str(path_obj),
        pages=pages,
        page_chunks=page_chunks,
        write_images=write_images,
        image_path=img_dir_str,
    )

    if page_chunks and isinstance(result, list):
        formatted_pages: list[dict[str, Any]] = []
        for item in result:
            meta = item.get("metadata", {})
            page_num = meta.get("page", 1)
            formatted_pages.append(
                {
                    "page_number": page_num,
                    "text": item.get("text", ""),
                    "metadata": meta,
                    "tables": item.get("tables", []),
                    "images": item.get("images", []),
                }
            )
        return formatted_pages

    return result


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
    total_pages = pdf_page_count(pdf_path)

    pages: list[DocumentPage] = []
    pages_md: list[str] = []
    all_page_specs: list[list[str]] = []
    previous_context: str | None = None

    active_forced = forced_specs or forced_doc_type

    # Proses BERTAHAP per batch 10 halaman: render batch -> ekstrak batch -> lanjut.
    # Konsisten dengan strategi "proses per 10 pages" dan meminimalkan beban memori/disk.
    for b_start in range(0, total_pages, PDF_PAGE_BATCH):
        b_end = min(b_start + PDF_PAGE_BATCH, total_pages)
        page_images = pdf_to_images(
            pdf_path,
            output_dir=output_dir,
            dpi=dpi,
            pages=list(range(b_start, b_end)),
        )

        for idx, img_path in enumerate(page_images, start=b_start + 1):
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
        total_pages=total_pages,
        markdown_content=stitched_markdown,
        pages=pages,
        metadata={
            "source_type": "pdf",
            "dpi": dpi,
            "filename": pdf_path.name,
        },
    )
