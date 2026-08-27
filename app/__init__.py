"""
Vision OCR & Document Text Extractor (Ready for Chunking & Tabular Database).

Modul:
  - config.py        : Pengaturan lingkungan & model via .env
  - preprocess.py    : Preprocessing gambar (auto-rotate EXIF, contrast enhancement)
  - ocr.py           : Model OCR tuned untuk referensi teks resolusi tinggi
  - prompts.py       : Prompt spesialisasi 4 spesifikasi tata letak dokumen
  - extractor.py     : Ekstraktor VLM multimodal -> Markdown
  - agents.py        : Registry agent untuk spesifikasi dokumen
  - ppt.py           : Ekstraktor presentasi PowerPoint (.pptx / .ppt)
  - pdf.py           : Konversi & ekstraksi PDF multi-halaman
  - multi_page.py    : Penyambung halaman (header continuity & chunking simulation)
  - graph.py         : Pipeline LangGraph orkestrasi ekstraksi dokumen
  - deep_agent.py    : Harness Deep Agents untuk ekstraksi dokumen & database tabular
  - tabular_db.py    : Deteksi tabel transaksional, SQLite ingestion, double-verification, & SQL querying
  - schemas.py       : Schema data (Pydantic models untuk dokumen, chunking, tabel, dan verifikasi)
"""

from .agents import AGENT_REGISTRY, DocumentExtractionAgent, get_agent
from .batch import batch_extract_documents, scan_document_directories
from .config import Settings, get_settings
from .deep_agent import build_deep_agent, run_deep_reasoning_agent
from .extractor import VisionExtractor
from .graph import (
    DocumentExtractionPipeline,
    DocumentExtractionState,
    VisionRAGPipeline,
)
from .mcp_server import server as mcp_server
from .multi_page import (
    format_page_delimiter,
    preview_markdown_chunks,
    split_markdown_by_pages,
    stitch_pages_to_markdown,
)
from .ocr import OCRExtractor, build_ocr_extractor
from .pdf import (
    extract_pdf_with_pymupdf4llm,
    pdf_page_count,
    pdf_to_images,
    process_multipage_pdf,
)
from .ppt import (
    convert_presentation_to_pdf,
    count_presentation_slides,
    pptx_to_structured_text,
    process_presentation,
    process_presentation_vision,
    render_presentation_slides_to_images,
)
from .preprocess import preprocess_image
from .schemas import (
    ChunkingPreview,
    ChunkItem,
    ClassificationResult,
    DocumentPage,
    DocumentSection,
    ExtractedDocument,
    OCRResult,
    TableClassificationResult,
    TableColumnSchema,
    TableIngestionResult,
    TableSchema,
    TableVerificationReport,
    TabularQueryResult,
    VerificationCheck,
)
from .tabular_db import (
    TabularDatabaseManager,
    TabularVerifier,
    classify_table_heuristic,
    extract_and_ingest_tables_from_markdown,
    infer_table_schema,
    parse_markdown_tables,
    query_sqlite,
)

__all__ = [
    "AGENT_REGISTRY",
    "ChunkItem",
    "ChunkingPreview",
    "ClassificationResult",
    "DocumentExtractionAgent",
    "DocumentExtractionPipeline",
    "DocumentExtractionState",
    "DocumentPage",
    "DocumentSection",
    "ExtractedDocument",
    "OCRExtractor",
    "OCRResult",
    "Settings",
    "TableClassificationResult",
    "TableColumnSchema",
    "TableIngestionResult",
    "TableSchema",
    "TableVerificationReport",
    "TabularDatabaseManager",
    "TabularQueryResult",
    "TabularVerifier",
    "VerificationCheck",
    "VisionExtractor",
    "VisionRAGPipeline",
    "batch_extract_documents",
    "build_deep_agent",
    "build_ocr_extractor",
    "classify_table_heuristic",
    "convert_presentation_to_pdf",
    "count_presentation_slides",
    "extract_and_ingest_tables_from_markdown",
    "extract_pdf_with_pymupdf4llm",
    "format_page_delimiter",
    "get_agent",
    "get_settings",
    "infer_table_schema",
    "mcp_server",
    "parse_markdown_tables",
    "pdf_page_count",
    "pdf_to_images",
    "pptx_to_structured_text",
    "preprocess_image",
    "preview_markdown_chunks",
    "process_multipage_pdf",
    "process_presentation",
    "process_presentation_vision",
    "query_sqlite",
    "render_presentation_slides_to_images",
    "run_deep_reasoning_agent",
    "scan_document_directories",
    "split_markdown_by_pages",
    "stitch_pages_to_markdown",
]
