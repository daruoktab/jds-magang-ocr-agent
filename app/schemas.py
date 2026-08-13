"""
Schema Pydantic untuk hasil ekstraksi dan klasifikasi dokumen.

Prinsip: `DocumentExtraction` tetap generik (model bebas menentukan struktur),
sehingga satu pipeline dapat menangani banyak jenis dokumen tanpa schema kaku.
Untuk integrasi yang butuh schema ketat, teruskan model Pydantic sendiri ke
`VisionExtractor` - alur tidak berubah.
"""
from typing import Any, Dict, List

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
    data: Dict[str, Any] = Field(
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
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata dokumen")


class VisionRAGResult(BaseModel):
    """Hasil akhir vision RAG: klasifikasi + ekstraksi + konteks retrieval."""

    doc_type: str = Field(description="Jenis dokumen yang terdeteksi")
    extraction: DocumentExtraction = Field(description="Hasil ekstraksi terstruktur")
    context: List[RetrievedChunk] = Field(
        default_factory=list, description="Konteks relevan hasil retrieval"
    )
