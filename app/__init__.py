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
from .config import Settings, get_settings
from .deep_agent import build_deep_agent
from .extractor import VisionExtractor
from .graph import DocumentExtractionPipeline, DocumentExtractionState, VisionRAGPipeline
from .mcp_server import server as mcp_server
from .multi_page import preview_markdown_chunks, stitch_pages_to_markdown
from .ocr import OCRExtractor, build_ocr_extractor
from .pdf import pdf_to_images, process_multipage_pdf
from .ppt import pptx_to_structured_text, process_presentation
from .preprocess import preprocess_image
from .schemas import (
    ChunkItem,
    ChunkingPreview,
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
    "build_deep_agent",
    "build_ocr_extractor",
    "get_agent",
    "get_settings",
    "pdf_to_images",
    "pptx_to_structured_text",
    "preprocess_image",
    "preview_markdown_chunks",
    "process_multipage_pdf",
    "process_presentation",
    "stitch_pages_to_markdown",
]
