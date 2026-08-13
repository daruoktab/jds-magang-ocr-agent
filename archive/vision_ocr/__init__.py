"""
vision_ocr - Ekstraksi dokumen multi-agent berbasis VLM + Pydantic.

Alur: agent utama (DocumentDispatcher) mengklasifikasikan jenis dokumen
dari gambar, lalu memilih agent ekstraksi yang sesuai (tiap agent punya
prompt tuning untuk jenis dokumennya). Hasil divalidasi Pydantic.

Komponen modular:
  - llm.py       : abstraksi backend LLM (Ollama, OpenAI-compatible)
  - schemas.py   : schema Pydantic hasil ekstraksi
  - prompts.py   : prompt general + system prompt
  - agents.py    : registry agent ekstraksi per jenis dokumen
  - dispatcher.py: agent utama (klasifikasi + routing)
  - extractor.py : pipeline gambar -> VLM -> JSON -> Pydantic
"""

from .agents import AGENT_REGISTRY, ExtractionAgent, get_agent
from .dispatcher import DocumentDispatcher
from .embedder import VisionEmbedder
from .extractor import VisionExtractor, create_extractor
from .schemas import DocumentExtraction

__all__ = [
    "AGENT_REGISTRY",
    "ExtractionAgent",
    "get_agent",
    "DocumentDispatcher",
    "VisionEmbedder",
    "VisionExtractor",
    "create_extractor",
    "DocumentExtraction",
]
