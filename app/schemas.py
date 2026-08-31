"""
Pydantic Schemas untuk Ekstraksi Dokumen Vision OCR -> Markdown Siap Chunking & Tabular Database.
Mendukung multi-spesifikasi / karakteristik komposit pada satu dokumen, pemisahan data tabular transaksional ke SQLite,
serta ekstraksi diagram visual ke sintaks Mermaid.js secara selektif.
"""

from __future__ import annotations

from typing import Any, Literal

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
    markdown_content: str = Field(
        ..., description="Teks Markdown yang diekstrak dari halaman ini"
    )
    ocr_text: str | None = Field(
        default=None, description="Teks mentah hasil OCR tambahan"
    )
    image_path: str | None = Field(
        default=None, description="Path gambar halaman bila ada"
    )


# ==============================================================================
# Tabular & SQLite Ingestion / Verification Schemas
# ==============================================================================

TableTypeLiteral = Literal[
    "transactional_log",
    "financial_statement",
    "inventory_ledger",
    "narrative_matrix",
    "form_key_value",
    "generic_table",
]

RecommendedStorageLiteral = Literal["sqlite_database", "vector_rag"]


class TableClassificationResult(BaseModel):
    """Hasil klasifikasi tabel: membedakan tabel transaksional (DB) vs tabel naratif (Vector RAG)."""

    table_id: str = Field(default="table_1", description="Identifier tabel")
    is_transactional: bool = Field(
        default=False,
        description="True jika tabel berupa data log/transaksi numerik yang memerlukan agregasi SQL (SUM, AVG, filter)",
    )
    table_type: TableTypeLiteral = Field(
        default="generic_table",
        description="Tipe semantik tabel (transactional_log, financial_statement, inventory_ledger, narrative_matrix, form_key_value, generic_table)",
    )
    recommended_storage: RecommendedStorageLiteral = Field(
        default="vector_rag",
        description="Rekomendasi storage: 'sqlite_database' untuk transaksional, 'vector_rag' untuk naratif",
    )
    confidence: float = Field(
        default=1.0, description="Tingkat keyakinan klasifikasi (0.0 - 1.0)"
    )
    reasoning: str = Field(
        default="", description="Alasan klasifikasi dan karakteristik yang ditemukan"
    )
    numeric_density: float = Field(
        default=0.0, description="Rasio kolom/sel bernilai numerik"
    )
    date_density: float = Field(
        default=0.0, description="Rasio kolom/sel bertipe tanggal"
    )
    total_rows: int = Field(default=0, description="Estimasi total baris data")
    total_columns: int = Field(default=0, description="Jumlah kolom terdeteksi")
    columns_detected: list[str] = Field(
        default_factory=list, description="Daftar nama kolom header"
    )


class TableColumnSchema(BaseModel):
    """Skema definisi satu kolom dalam tabel SQLite."""

    name: str = Field(
        ..., description="Nama kolom yang disanitasi untuk identifier SQL aman"
    )
    original_name: str = Field(
        ..., description="Nama kolom asli pada dokumen/tabel sumber"
    )
    sql_type: Literal["TEXT", "INTEGER", "REAL", "NUMERIC", "DATE", "DATETIME"] = Field(
        default="TEXT", description="Tipe data SQL yang sesuai"
    )
    is_nullable: bool = Field(
        default=True, description="Apakah kolom boleh bernilai NULL"
    )
    description: str | None = Field(default=None, description="Deskripsi makna kolom")
    sample_values: list[Any] = Field(
        default_factory=list, description="Contoh nilai data untuk verifikasi"
    )


class TableSchema(BaseModel):
    """Skema lengkap tabel terstruktur untuk database SQLite."""

    table_name: str = Field(..., description="Nama tabel pada SQLite database")
    source_file: str | None = Field(
        default=None, description="Path dokumen sumber asal tabel"
    )
    columns: list[TableColumnSchema] = Field(
        default_factory=list, description="Daftar skema kolom"
    )
    primary_key: list[str] | None = Field(
        default=None, description="Kolom primary key jika ada (mis. id, no_ref)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata dokumen, header form, atau periode transaksi",
    )


class VerificationCheck(BaseModel):
    """Hasil satu item pemeriksaan validitas data tabular."""

    check_name: str = Field(
        ...,
        description="Nama pemeriksaan (mis. row_count_check, numeric_integrity_check)",
    )
    passed: bool = Field(..., description="Apakah pemeriksaan lolos (True/False)")
    details: str = Field(
        ..., description="Penjelasan detail hasil pemeriksaan atau temuan anomali"
    )
    metric_value: Any | None = Field(default=None, description="Nilai metrik terukur")


class TableVerificationReport(BaseModel):
    """Laporan verifikasi ganda (double-verification) integritas data tabel SQLite."""

    table_name: str = Field(..., description="Nama tabel yang diverifikasi")
    database_path: str = Field(..., description="Path database SQLite yang diuji")
    is_valid: bool = Field(..., description="Apakah seluruh kriteria verifikasi lolos")
    confidence_score: float = Field(
        default=1.0, description="Skor kepercayaan validitas data (0.0 - 1.0)"
    )
    verification_status: Literal["verified", "needs_revision", "rejected"] = Field(
        default="verified", description="Status verifikasi akhir"
    )
    checks: list[VerificationCheck] = Field(
        default_factory=list, description="Rincian seluruh pemeriksaan yang dijalankan"
    )
    summary: str = Field(default="", description="Ringkasan evaluasi verifikasi")
    verified_row_count: int = Field(
        default=0, description="Jumlah baris yang diverifikasi dalam SQLite"
    )
    test_queries: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Daftar query SQL uji coba (mis. SELECT COUNT(*), SUM(...)) dan hasilnya",
    )
    llm_reflection: str | None = Field(
        default=None,
        description="Catatan refleksi/penalaran LLM atas kualitas dan akurasi ekstraksi",
    )


class TableIngestionResult(BaseModel):
    """Hasil proses ekstraksi dan ingesti tabel ke SQLite."""

    status: Literal["success", "warning", "error"] = Field(
        default="success", description="Status hasil ingesti"
    )
    table_name: str = Field(..., description="Nama tabel di SQLite")
    database_path: str = Field(
        ..., description="Path file database SQLite tempat data disimpan"
    )
    total_rows_ingested: int = Field(
        default=0, description="Jumlah baris yang berhasil di-insert"
    )
    columns: list[str] = Field(
        default_factory=list, description="Daftar kolom yang berhasil dibuat"
    )
    verification_report: TableVerificationReport | None = Field(
        default=None, description="Laporan verifikasi integritas data"
    )
    sample_data: list[dict[str, Any]] = Field(
        default_factory=list, description="Contoh 3-5 baris data teratas"
    )
    message: str = Field(default="", description="Pesan status atau informasi tambahan")


class TabularQueryResult(BaseModel):
    """Hasil eksekusi query SQL pada database SQLite dokumen."""

    query: str = Field(..., description="Query SQL yang dijalankan")
    status: Literal["success", "error"] = Field(
        default="success", description="Status eksekusi query"
    )
    row_count: int = Field(default=0, description="Jumlah baris hasil query")
    columns: list[str] = Field(
        default_factory=list, description="Daftar kolom hasil query"
    )
    rows: list[dict[str, Any]] = Field(
        default_factory=list, description="Baris data hasil query"
    )
    execution_time_ms: float = Field(
        default=0.0, description="Waktu eksekusi dalam milidetik"
    )
    error_message: str | None = Field(
        default=None, description="Pesan error jika query gagal"
    )


class PageTabularEvent(BaseModel):
    """Event pemrosesan mandiri Sub-Agent SQL per-halaman/slide."""

    page_number: int = Field(..., description="Nomor halaman/slide yang diproses")
    tables_detected: int = Field(
        default=0, description="Jumlah tabel yang ditemukan pada halaman ini"
    )
    existing_tables_inspected: list[str] = Field(
        default_factory=list,
        description="Daftar tabel eksisting di SQLite yang diperiksa sebelum ingesti",
    )
    actions_taken: list[str] = Field(
        default_factory=list,
        description="Daftar tindakan yang dieksekusi (cek skema, append baris, buat tabel baru)",
    )
    queries_executed: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Query SQL mandiri yang dijalankan oleh sub-agent untuk validasi state tabel",
    )
    rows_ingested_total: int = Field(
        default=0, description="Total baris data yang di-ingest dari halaman ini"
    )
    status: Literal["no_tables", "created_new_table", "appended_existing_table", "error"] = Field(
        default="no_tables", description="Status hasil pemrosesan halaman"
    )


class DualTrackGuardrailReport(BaseModel):
    """Laporan Guardrail & Audit Komparatif Jalur Ganda (Markdown Track vs SQLite Tabular Track)."""

    source_file: str = Field(..., description="File sumber dokumen")
    database_path: str = Field(..., description="Path database SQLite yang diaudit")
    total_pages_processed: int = Field(default=1, description="Total halaman yang diproses")
    total_markdown_tables: int = Field(
        default=0, description="Total tabel yang terdeteksi di teks Markdown"
    )
    total_sqlite_tables: int = Field(
        default=0, description="Total tabel yang tersimpan di SQLite database"
    )
    total_markdown_rows: int = Field(
        default=0, description="Total baris data dari seluruh tabel di Markdown"
    )
    total_sqlite_rows: int = Field(
        default=0, description="Total baris data yang berhasil tercatat di SQLite"
    )
    guardrail_status: Literal["PASSED", "WARNING", "FAILED"] = Field(
        default="PASSED",
        description="Status verifikasi akhir pengawasan agent utama",
    )
    table_comparisons: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Rincian komparasi tiap tabel (nama tabel, baris MD, baris SQLite, status)",
    )
    discrepancies: list[str] = Field(
        default_factory=list,
        description="Daftar anomali atau perbedaan antara jalur Markdown dan SQLite",
    )
    supervisor_notes: str = Field(
        default="",
        description="Catatan pengawasan dan evaluasi kualitas dari Master Supervisor Agent",
    )


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
    markdown_content: str = Field(
        ..., description="Teks Markdown utuh dari awal sampai akhir, siap di-chunking"
    )
    pages: list[DocumentPage] = Field(
        default_factory=list, description="Detail ekstraksi per-halaman"
    )
    tabular_events: list[PageTabularEvent] = Field(
        default_factory=list,
        description="Log eksekusi pemahaman Sub-Agent Tabular per-halaman",
    )
    guardrail_report: DualTrackGuardrailReport | None = Field(
        default=None,
        description="Laporan audit guardrail komparatif Markdown vs SQLite",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadata tambahan dokumen"
    )

    @property
    def doc_type(self) -> str:
        """String gabungan spesifikasi (kompatibilitas)."""
        return ", ".join(self.specs) if self.specs else "plain"


class ChunkItem(BaseModel):
    """Satu potongan chunk hasil text splitting."""

    chunk_index: int = Field(..., description="Indeks urutan chunk")
    char_count: int = Field(..., description="Jumlah karakter dalam chunk")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadata header/halaman dari chunk"
    )
    content: str = Field(..., description="Isi teks chunk")


class ChunkingPreview(BaseModel):
    """Hasil simulasi chunking pada dokumen."""

    total_chunks: int = Field(..., description="Jumlah total potongan chunk")
    chunks: list[ChunkItem] = Field(
        default_factory=list, description="Daftar potongan chunk"
    )


# ==============================================================================
# Diagram & Mermaid Specialist Schemas
# ==============================================================================

DiagramTypeLiteral = Literal[
    "flowchart",
    "sequence_diagram",
    "class_diagram",
    "state_diagram",
    "er_diagram",
    "mindmap",
    "gantt_chart",
    "block_architecture",
    "git_graph",
    "unsuitable_statistical_chart",
    "unsuitable_map_or_spatial",
    "unsuitable_photo_or_illustration",
    "unsuitable_complex_schematic",
    "non_diagram",
    "generic_diagram",
]

DiagramFormatRecommendation = Literal[
    "mermaid",
    "markdown_table",
    "text_description",
    "none",
]


class DiagramConvertibilityResult(BaseModel):
    """Hasil evaluasi kelayakan diagram untuk diubah menjadi kode Mermaid."""

    is_convertible: bool = Field(
        default=False,
        description="True jika diagram memiliki topologi diskrit/relasi yang cocok untuk sintaks Mermaid",
    )
    diagram_type: DiagramTypeLiteral = Field(
        default="generic_diagram",
        description="Tipe semantik diagram yang terdeteksi",
    )
    recommended_format: DiagramFormatRecommendation = Field(
        default="text_description",
        description="Format output yang direkomendasikan ('mermaid', 'markdown_table', 'text_description', 'none')",
    )
    mermaid_type: str | None = Field(
        default=None,
        description="Jenis diagram Mermaid yang disarankan (mis. 'flowchart TD', 'sequenceDiagram', 'erDiagram', 'stateDiagram-v2', 'classDiagram', 'mindmap', 'gantt')",
    )
    confidence: float = Field(
        default=1.0,
        description="Tingkat keyakinan evaluasi (0.0 - 1.0)",
    )
    reasoning: str = Field(
        default="",
        description="Penjelasan detail mengapa diagram cocok atau tidak cocok dikonversi ke Mermaid",
    )
    nodes_or_entities: list[str] = Field(
        default_factory=list,
        description="Daftar node/entitas utama yang terdeteksi dalam diagram",
    )


class DiagramExtractionResult(BaseModel):
    """Hasil ekstraksi diagram visual menjadi kode Mermaid atau deskripsi terstruktur."""

    status: Literal["success", "unsuitable", "error"] = Field(
        default="success",
        description="Status hasil ekstraksi",
    )
    is_mermaid: bool = Field(
        default=False,
        description="True jika berhasil menghasilkan kode Mermaid valid",
    )
    diagram_type: str = Field(
        default="diagram",
        description="Tipe diagram yang diekstrak",
    )
    mermaid_code: str | None = Field(
        default=None,
        description="Kode Mermaid lengkap (dalam blok ```mermaid ... ``` atau raw)",
    )
    text_summary: str = Field(
        default="",
        description="Deskripsi naratif/ringkasan terstruktur dari diagram",
    )
    reasoning: str = Field(
        default="",
        description="Penalaran pemilihan format dan konversi diagram",
    )
    convertibility: DiagramConvertibilityResult | None = Field(
        default=None,
        description="Hasil evaluasi kelayakan konversi diagram",
    )
