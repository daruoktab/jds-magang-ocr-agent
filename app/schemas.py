"""
Schema Pydantic untuk hasil ekstraksi dan klasifikasi dokumen.

Prinsip: `DocumentExtraction` tetap generik (model bebas menentukan struktur),
sehingga satu pipeline dapat menangani banyak jenis dokumen tanpa schema kaku.
Untuk integrasi yang butuh schema ketat, teruskan model Pydantic sendiri ke
`VisionExtractor` - alur tidak berubah.
"""
from typing import Any

from pydantic import BaseModel, Field


class DocumentClassification(BaseModel):
    """Hasil klasifikasi jenis dokumen."""

    doc_type: str = Field(
        description="Jenis dokumen (mis. receipt, table, form, business_card, ...)"
    )


class DocumentExtraction(BaseModel):
    """
    Hasil ekstraksi generik.

    - doc_type : jenis dokumen, diputuskan VLM.
    - data     : struktur bebas hasil ekstraksi (dict bersarang, list, kombinasi).
    """

    doc_type: str = Field(description="Jenis dokumen yang diputuskan model")
    data: dict[str, Any] = Field(
        description="Struktur bebas hasil ekstraksi, ditentukan oleh model"
    )


class OCRResult(BaseModel):
    """
    Hasil OCR terstruktur dari model `ocr-lighton` (VLM kecil tuned).

    Model OCR mengembalikan teks biasa (bukan JSON), jadi text mentah
    disimpan di `text`. Field lain opsional untuk normalisasi lanjutan.
    """

    text: str = Field(description="Teks mentah hasil OCR (persis keluaran model)")
    doc_type: str = Field(default="ocr", description="Jenis dokumen (default 'ocr')")


class RetrievedChunk(BaseModel):
    """Potongan konteks relevan hasil retrieval."""

    content: str = Field(description="Isi potongan dokumen yang relevan")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata dokumen")


class ValidationSummary(BaseModel):
    """Ringkasan hasil validasi konsistensi & audit agentic reflection."""

    is_valid: bool = Field(default=True, description="Apakah ekstraksi lolos validasi konsistensi")
    score: float = Field(default=1.0, description="Skor validasi kualitas (0.0 - 1.0)")
    issues: list[str] = Field(default_factory=list, description="Daftar kejanggalan/isu yang terdeteksi")
    reflection_attempts: int = Field(default=0, description="Berapa kali VLM melakukan self-reflection retry")


class VisionRAGResult(BaseModel):
    """Hasil akhir vision RAG: klasifikasi + ekstraksi + konteks retrieval + validasi."""

    doc_type: str = Field(description="Jenis dokumen yang terdeteksi")
    extraction: DocumentExtraction = Field(description="Hasil ekstraksi terstruktur")
    ocr_text: str | None = Field(default=None, description="Teks mentah hasil auxiliary OCR")
    context: list[RetrievedChunk] = Field(
        default_factory=list, description="Konteks relevan hasil retrieval"
    )
    validation: ValidationSummary = Field(
        default_factory=ValidationSummary, description="Hasil validasi dan audit trail refleksi"
    )


class MultiPageExtractionResult(BaseModel):
    """Hasil ekstraksi gabungan untuk dokumen PDF multi-halaman."""

    filename: str = Field(description="Nama file dokumen asli")
    total_pages: int = Field(description="Total halaman yang diproses")
    doc_type: str = Field(description="Jenis dokumen utama terdeteksi")
    consolidated_data: dict[str, Any] = Field(
        description="Hasil gabungan data terstruktur seluruh halaman"
    )
    pages: list[VisionRAGResult] = Field(
        default_factory=list, description="Hasil ekstraksi per-halaman secara rinci"
    )
    validation: ValidationSummary = Field(
        default_factory=ValidationSummary, description="Hasil validasi gabungan"
    )

