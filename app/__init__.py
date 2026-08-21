"""
Vision OCR & Document Text Extractor (Ready for Chunking).

Modul:
  - config.py        : Pengaturan lingkungan & model via .env
  - preprocess.py    : Preprocessing gambar (auto-rotate EXIF, contrast enhancement)
  - ocr.py           : Model OCR tuned untuk referensi teks resolusi tinggi
  - prompts.py       : Prompt spesialisasi 4 spesifikasi tata letak dokumen
  - extractor.py     : Ekstraktor VLM multimodal -> Markdown
  - agents.py        : Registry agent untuk 4 spesifikasi dokumen
  - ppt.py           : Ekstraktor presentasi PowerPoint (.pptx / .ppt)
  - pdf.py           : Konversi & ekstraksi PDF multi-halaman
  - multi_page.py    : Penyambung halaman (header continuity & chunking simulation)
  - graph.py         : Pipeline LangGraph orkestrasi ekstraksi dokumen
  - deep_agent.py    : Harness Deep Agents untuk ekstraksi dokumen
  - schemas.py       : Schema data (ExtractedDocument, DocumentPage, ChunkingPreview)
"""

from .agents import AGENT_REGISTRY, DocumentExtractionAgent, get_agent
from .batch import batch_extract_documents, scan_document_directories
from .config import Settings, get_settings
from .deep_agent import build_deep_agent
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
    count_presentation_slides,
    pptx_to_structured_text,
    process_presentation,
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
    "VisionExtractor",
    "VisionRAGPipeline",
    "batch_extract_documents",
    "build_deep_agent",
    "build_ocr_extractor",
    "count_presentation_slides",
    "extract_pdf_with_pymupdf4llm",
    "format_page_delimiter",
    "get_agent",
    "get_settings",
    "mcp_server",
    "pdf_page_count",
    "pdf_to_images",
    "pptx_to_structured_text",
    "preprocess_image",
    "preview_markdown_chunks",
    "process_multipage_pdf",
    "process_presentation",
    "render_presentation_slides_to_images",
    "scan_document_directories",
    "split_markdown_by_pages",
    "stitch_pages_to_markdown",
]
