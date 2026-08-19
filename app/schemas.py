"""
Pydantic Schemas untuk Ekstraksi Dokumen Vision OCR -> Markdown Siap Chunking.
Mendukung multi-spesifikasi / karakteristik komposit pada satu dokumen.
"""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class OCRResult(BaseModel):
    """Hasil ekstraksi OCR teks mentah."""
    text: str = Field(..., description="Teks mentah yang berhasil diekstrak model OCR")


class ClassificationResult(BaseModel):
    """Hasil klasifikasi satu atau lebih karakteristik layout dokumen."""
    specs: list[str] = Field(
        default_factory=lambda: ["plain"],
        description="Daftar karakteristik yang terdeteksi: plain, markdown_hierarchy, bilingual_journal, presentation_slides",
    )
    confidence: float = Field(default=1.0, description="Tingkat keyakinan klasifikasi")

    @property
    def primary_spec(self) -> str:
        """Karakteristik utama dokumen."""
        return self.specs[0] if self.specs else "plain"


class DocumentSection(BaseModel):
    """Bagian dokumen berbasis heading."""
    heading: str = Field(..., description="Judul heading (mis. '## Pendahuluan')")
    level: int = Field(default=2, description="Level heading (1, 2, 3, dst.)")
    content: str = Field(default="", description="Konten isi dalam heading ini")


class DocumentPage(BaseModel):
    """Hasil ekstraksi per-halaman dokumen."""
    page_number: int = Field(..., description="Nomor urut halaman (mulai 1)")
    specs: list[str] = Field(
        default_factory=lambda: ["plain"],
        description="Daftar karakteristik layout pada halaman ini",
    )
    markdown_content: str = Field(..., description="Teks Markdown yang diekstrak dari halaman ini")
    ocr_text: str | None = Field(default=None, description="Teks mentah hasil OCR tambahan")
    image_path: str | None = Field(default=None, description="Path gambar halaman bila ada")


class ExtractedDocument(BaseModel):
    """
    Hasil ekstraksi lengkap seluruh dokumen dalam format Markdown utuh siap chunking.
    """
    file_path: str = Field(..., description="Path file input dokumen (PDF/PPTX/Image)")
    specs: list[str] = Field(
        default_factory=lambda: ["plain"],
        description="Daftar karakteristik layout dokumen yang terdeteksi",
    )
    total_pages: int = Field(default=1, description="Jumlah total halaman / slide")
    markdown_content: str = Field(..., description="Teks Markdown utuh dari awal sampai akhir, siap di-chunking")
    pages: list[DocumentPage] = Field(default_factory=list, description="Detail ekstraksi per-halaman")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata tambahan dokumen")

    @property
    def doc_type(self) -> str:
        """String gabungan spesifikasi (kompatibilitas)."""
        return ", ".join(self.specs) if self.specs else "plain"


class ChunkItem(BaseModel):
    """Satu potongan chunk hasil text splitting."""
    chunk_index: int = Field(..., description="Indeks urutan chunk")
    char_count: int = Field(..., description="Jumlah karakter dalam chunk")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata header/halaman dari chunk")
    content: str = Field(..., description="Isi teks chunk")


class ChunkingPreview(BaseModel):
    """Hasil simulasi chunking pada dokumen."""
    total_chunks: int = Field(..., description="Jumlah total potongan chunk")
    chunks: list[ChunkItem] = Field(default_factory=list, description="Daftar potongan chunk")
