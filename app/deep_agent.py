"""
Harness Deep Agents untuk vision RAG.

Membungkus kemampuan ekstraksi (VLM normal), OCR, dan retrieval sebagai tools
LangChain, lalu membungkusnya menjadi deep agent (`create_deep_agent`) dengan
subagent spesialis yang bisa dipanggil lewat `task`.
"""
from __future__ import annotations

from deepagents import SubAgent, create_deep_agent
from langchain_core.tools import tool

from .config import Settings, get_settings
from .embedding import build_embeddings
from .extractor import VisionExtractor
from .llm import build_vlm
from .ocr import build_ocr_extractor
from .vector_store import VisionIndex


def build_deep_agent(settings: Settings | None = None):
    """Bangun deep agent vision RAG (mengembalikan compiled agent)."""
    settings = settings or get_settings()
    vlm = build_vlm(settings)
    extractor = VisionExtractor(vlm)
    ocr = build_ocr_extractor(settings)

    # Indeks dibuat malas: hanya butuh binary llama-vl-embedding saat dipakai.
    # Kalau embedding dimatikan / binary belum ada -> None (tool menanganinya).
    _index_cache: dict[str, VisionIndex | None] = {}

    def _index() -> VisionIndex | None:
        if "index" not in _index_cache:
            if not settings.embedding_enabled:
                _index_cache["index"] = None
            else:
                try:
                    _index_cache["index"] = VisionIndex(build_embeddings(settings))
                except Exception as e:  # noqa: BLE001 - binary belum dibuild, dll.
                    print(f"[warn] Embedding tidak tersedia: {e}")
                    _index_cache["index"] = None
        return _index_cache["index"]

    @tool
    def extract_document(image_path: str) -> dict:
        """Ekstrak dokumen dari gambar menjadi JSON terstruktur (doc_type + data)."""
        return extractor.extract(image_path).model_dump()

    @tool
    def ocr_document(image_path: str) -> dict:
        """OCR dokumen menjadi teks terstruktur (menggunakan model OCR tuned)."""
        return ocr.extract(image_path).model_dump()

    @tool
    def search_index(query: str) -> str:
        """Ambil konteks relevan dari indeks vision embedding."""
        idx = _index()
        if idx is None:
            return "(embedding tidak tersedia - retrieval dilewati)"
        docs = idx.search(query, k=4)
        if not docs:
            return "(tidak ada hasil)"
        return "\n\n".join(d.page_content for d in docs)

    document_agent: SubAgent = {
        "name": "document-extractor",
        "description": "Ekstrak JSON terstruktur dari gambar dokumen (receipt, invoice, table, form, dll).",
        "system_prompt": (
            "Kamu spesialis ekstraksi dokumen. Panggil tool extract_document "
            "dengan path gambar untuk mendapatkan JSON terstruktur."
        ),
        "tools": [extract_document],
    }

    ocr_agent: SubAgent = {
        "name": "ocr",
        "description": "OCR dokumen menjadi teks terstruktur (model OCR tuned).",
        "system_prompt": (
            "Kamu spesialis OCR. Panggil tool ocr_document dengan path gambar "
            "untuk mendapatkan seluruh teks dalam gambar."
        ),
        "tools": [ocr_document],
    }

    retrieval_agent: SubAgent = {
        "name": "retriever",
        "description": "Cari konteks relevan dari indeks vision embedding.",
        "system_prompt": "Kamu spesialis retrieval. Panggil tool search_index untuk mencari konteks.",
        "tools": [search_index],
    }

    return create_deep_agent(
        name="vision-rag-agent",
        model=vlm,
        tools=[extract_document, ocr_document, search_index],
        system_prompt=(
            "Kamu adalah vision RAG agent. Terima gambar dokumen, ekstrak "
            "informasi (JSON terstruktur) atau lakukan OCR, dan gunakan retrieval "
            "bila konteks tambahan diperlukan untuk menjawab."
        ),
        subagents=[document_agent, ocr_agent, retrieval_agent],
    )
