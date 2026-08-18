"""
Vision RAG Agent Multimodal & Agentic Document Extraction.

Arsitektur:
  - config.py        : Settings (VLM, OCR, embedding, reranker) via .env
  - preprocess.py    : Image preprocessing (auto-rotate EXIF, contrast enhancement)
  - embedding.py     : VisionEmbedder & adapter LangChain Embeddings
  - reranker.py      : Qwen3VLReranker multimodal scoring & document reranking
  - llm.py           : builder ChatOpenAI (VLM normal + OCR) ke endpoint OpenAI-compatible
  - schemas.py       : schema Pydantic (klasifikasi, ekstraksi, validasi, multi-page)
  - validation.py    : validasi konsistensi matematika & kelengkapan data
  - prompts.py       : prompt general + per jenis dokumen + fusion & reflection
  - extractor.py     : pipeline gambar -> VLM -> structured output (Pydantic)
  - ocr.py           : pipeline gambar -> OCR tuned -> teks terstruktur (OCRResult)
  - agents.py        : registry agent ekstraksi per jenis dokumen
  - vector_store.py  : indeks RAG (Two-Stage Retrieval + persistensi lokal)
  - graph.py         : orkestrasi LangGraph Agentik (VLM+OCR Fusion + Self-Reflection)
  - pdf.py           : konversi PDF -> gambar & pemrosesan multi-halaman
  - deep_agent.py    : harness Deep Agents (create_deep_agent + subagents)
"""
from .agents import AGENT_REGISTRY, ExtractionAgent, get_agent
from .config import Settings, get_settings
from .deep_agent import build_deep_agent
from .embedding import (
    LlamaServerEmbeddings,
    LlamaVLEmbeddings,
    VisionEmbedder,
    build_embeddings,
)
from .extractor import VisionExtractor
from .graph import VisionRAGPipeline, VisionRAGState
from .ocr import OCRExtractor, build_ocr_extractor
from .pdf import pdf_to_images, process_multipage_pdf
from .preprocess import preprocess_image
from .report import generate_report
from .reranker import Qwen3VLReranker, build_reranker
from .schemas import (
    DocumentClassification,
    DocumentExtraction,
    MultiPageExtractionResult,
    OCRResult,
    ValidationSummary,
    VisionRAGResult,
)
from .validation import ValidationResult, validate_extraction
from .vector_store import VisionIndex

__all__ = [
    "AGENT_REGISTRY",
    "DocumentClassification",
    "DocumentExtraction",
    "ExtractionAgent",
    "LlamaServerEmbeddings",
    "LlamaVLEmbeddings",
    "MultiPageExtractionResult",
    "OCRExtractor",
    "OCRResult",
    "Qwen3VLReranker",
    "Settings",
    "ValidationResult",
    "ValidationSummary",
    "VisionEmbedder",
    "VisionExtractor",
    "VisionIndex",
    "VisionRAGPipeline",
    "VisionRAGResult",
    "VisionRAGState",
    "build_deep_agent",
    "build_embeddings",
    "build_ocr_extractor",
    "build_reranker",
    "generate_report",
    "get_agent",
    "get_settings",
    "pdf_to_images",
    "preprocess_image",
    "process_multipage_pdf",
    "validate_extraction",
]
