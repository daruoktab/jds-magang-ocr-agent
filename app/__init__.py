"""
Vision RAG agent.

Arsitektur:
  - config.py        : Settings (3 kategori model: VLM, OCR, embedding) via .env
  - embedding.py     : VisionEmbedder (subprocess llama-vl-embedding) + adapter LangChain
  - llm.py           : builder ChatOpenAI (VLM normal + OCR) ke endpoint OpenAI-compatible
  - schemas.py       : schema Pydantic (klasifikasi, ekstraksi, OCR, hasil RAG)
  - prompts.py       : prompt general + per jenis dokumen
  - extractor.py     : pipeline gambar -> VLM -> structured output (Pydantic)
  - ocr.py           : pipeline gambar -> OCR tuned -> teks terstruktur (OCRResult)
  - agents.py        : registry agent ekstraksi per jenis dokumen
  - vector_store.py  : indeks RAG (InMemoryVectorStore + vision embedding)
  - graph.py         : orkestrasi LangGraph (classify -> extract -> retrieve -> result)
  - deep_agent.py    : harness Deep Agents (create_deep_agent + subagents)
"""
from .agents import AGENT_REGISTRY, ExtractionAgent, get_agent
from .config import Settings, get_settings
from .deep_agent import build_deep_agent
from .embedding import LlamaServerEmbeddings, LlamaVLEmbeddings, VisionEmbedder, build_embeddings
from .extractor import VisionExtractor
from .graph import VisionRAGPipeline
from .ocr import OCRExtractor, build_ocr_extractor
from .pdf import pdf_to_images
from .report import generate_report
from .schemas import DocumentClassification, DocumentExtraction, OCRResult
from .vector_store import VisionIndex

__all__ = [
    "AGENT_REGISTRY",
    "ExtractionAgent",
    "get_agent",
    "Settings",
    "get_settings",
    "build_deep_agent",
    "build_embeddings",
    "build_ocr_extractor",
    "LlamaVLEmbeddings",
    "LlamaServerEmbeddings",
    "VisionEmbedder",
    "VisionExtractor",
    "OCRExtractor",
    "pdf_to_images",
    "generate_report",
    "VisionRAGPipeline",
    "DocumentClassification",
    "DocumentExtraction",
    "OCRResult",
    "VisionIndex",
]
