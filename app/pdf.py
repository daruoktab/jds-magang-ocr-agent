"""
Konversi PDF menjadi gambar per-halaman & pemrosesan dokumen PDF multi-halaman.

Menyediakan:
  - `pdf_to_images`: Konversi PDF menjadi gambar beresolusi optimal (~200 DPI).
  - `process_multipage_pdf`: Menjalankan pipeline agentik pada seluruh halaman PDF
    dan mengagregasikannya menjadi satu objek `MultiPageExtractionResult` yang utuh.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pymupdf

from .schemas import (
    DocumentExtraction,
    MultiPageExtractionResult,
    ValidationSummary,
    VisionRAGResult,
)

if TYPE_CHECKING:
    from .graph import VisionRAGPipeline

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
    pipeline: VisionRAGPipeline,
    output_dir: str | Path | None = None,
    dpi: int = DEFAULT_DPI,
    query: str | None = None,
) -> MultiPageExtractionResult:
    """
    Proses seluruh halaman PDF dan gabungkan hasil ekstraksi menjadi satu struktur terpadu.
    """
    pdf_path = Path(pdf_path)
    page_images = pdf_to_images(pdf_path, output_dir=output_dir, dpi=dpi)

    page_results: list[VisionRAGResult] = []
    doc_types: list[str] = []
    consolidated_data: dict[str, Any] = {}
    all_issues: list[str] = []
    total_reflection_attempts = 0

    # Kumpulkan daftar list per-kategori untuk digabung
    aggregated_lists: dict[str, list[Any]] = {}

    for idx, img_path in enumerate(page_images, 1):
        state = pipeline.run(str(img_path), query=query)
        final_dict = state["final_result"]
        res = VisionRAGResult(
            doc_type=final_dict["doc_type"],
            extraction=DocumentExtraction(**final_dict["extraction"]),
            ocr_text=final_dict.get("ocr_text"),
            context=final_dict.get("context", []),
            validation=ValidationSummary(**final_dict.get("validation", {})),
        )
        page_results.append(res)
        doc_types.append(res.doc_type)

        if not res.validation.is_valid:
            for issue in res.validation.issues:
                all_issues.append(f"Halaman {idx}: {issue}")
        total_reflection_attempts += res.validation.reflection_attempts

        # Agregasi data key-value & array
        page_data = res.extraction.data or {}
        for k, v in page_data.items():
            if isinstance(v, list):
                if k not in aggregated_lists:
                    aggregated_lists[k] = []
                aggregated_lists[k].extend(v)
            else:
                # Timpa nilai skalar jika sebelumnya belum ada atau diisi nilai baru
                if k not in consolidated_data or v:
                    consolidated_data[k] = v

    # Masukkan seluruh list yang telah diagregasi ke consolidated_data
    for k, v in aggregated_lists.items():
        consolidated_data[k] = v

    # Tentukan doc_type dominan
    primary_doc_type = max(set(doc_types), key=doc_types.count) if doc_types else "generic"

    combined_validation = ValidationSummary(
        is_valid=len(all_issues) == 0,
        score=max(0.0, 1.0 - (0.2 * len(all_issues))),
        issues=all_issues,
        reflection_attempts=total_reflection_attempts,
    )

    return MultiPageExtractionResult(
        filename=pdf_path.name,
        total_pages=len(page_images),
        doc_type=primary_doc_type,
        consolidated_data=consolidated_data,
        pages=page_results,
        validation=combined_validation,
    )
