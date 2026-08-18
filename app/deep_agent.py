"""
Harness Deep Agents untuk vision RAG multimodal.

Membungkus kemampuan:
  - Ekstraksi VLM + OCR Fusion
  - OCR teks terstruktur
  - Validasi konsistensi (matematika, format, kelengkapan)
  - Ingest & Two-Stage Multimodal Retrieval (Embedding + Reranker)
menjadi tools LangChain dan subagents spesialis di dalam `create_deep_agent`.
"""
from __future__ import annotations

import json

from deepagents import SubAgent, create_deep_agent
from langchain_core.tools import tool

from .agents import get_agent
from .config import Settings, get_settings
from .embedding import build_embeddings
from .llm import build_vlm
from .ocr import build_ocr_extractor
from .preprocess import preprocess_image
from .reranker import build_reranker
from .validation import validate_extraction
from .vector_store import VisionIndex


def build_deep_agent(settings: Settings | None = None):
    """Bangun deep agent vision RAG (mengembalikan compiled agent)."""
    settings = settings or get_settings()
    vlm = build_vlm(settings)
    ocr = build_ocr_extractor(settings)

    # Inisialisasi indeks & reranker secara lazy
    _index_cache: dict[str, VisionIndex | None] = {}

    def _index() -> VisionIndex | None:
        if "index" not in _index_cache:
            if not settings.embedding_enabled:
                _index_cache["index"] = None
            else:
                try:
                    embeddings = build_embeddings(settings)
                    reranker = build_reranker(settings)
                    _index_cache["index"] = VisionIndex(embeddings=embeddings, reranker=reranker)
                except Exception as e:  # noqa: BLE001
                    print(f"[warn] Embedding tidak tersedia: {e}")
                    _index_cache["index"] = None
        return _index_cache["index"]

    @tool
    def extract_document(image_path: str, doc_type: str = "generic", ocr_text: str | None = None, critique: str | None = None) -> dict:
        """Ekstrak dokumen dari gambar menjadi JSON terstruktur, dengan opsi bantuan teks OCR dan catatan perbaikan."""
        proc = preprocess_image(image_path)
        agent = get_agent(doc_type)
        res = agent.run(
            proc.processed_path,
            llm=vlm,
            ocr_text=ocr_text,
            critique=critique,
        )
        return res.model_dump()

    @tool
    def ocr_document(image_path: str) -> dict:
        """OCR dokumen menjadi teks terstruktur (menggunakan model OCR tuned)."""
        proc = preprocess_image(image_path)
        return ocr.extract(proc.processed_path).model_dump()

    @tool
    def validate_data(doc_type: str, data_json_str: str) -> dict:
        """Validasi konsistensi matematika dan kelengkapan data hasil ekstraksi."""
        try:
            data = json.loads(data_json_str) if isinstance(data_json_str, str) else data_json_str
            res = validate_extraction(doc_type, data)
            return {
                "is_valid": res.is_valid,
                "score": res.score,
                "issues": res.issues,
                "critique": res.format_critique(),
            }
        except Exception as e:  # noqa: BLE001
            return {"is_valid": False, "score": 0.0, "issues": [str(e)], "critique": str(e)}

    @tool
    def index_document(content_or_json: str, metadata_json_str: str = "{}") -> str:
        """Tambahkan teks / data katalog baru ke dalam indeks pencarian RAG."""
        idx = _index()
        if idx is None:
            return "(embedding tidak aktif - data tidak diindeks)"
        try:
            meta = json.loads(metadata_json_str) if isinstance(metadata_json_str, str) else {}
            idx.add_texts([content_or_json], metadatas=[meta])
            return "Sukses mengindeks dokumen ke ruang vektor."
        except Exception as e:  # noqa: BLE001
            return f"Gagal mengindeks: {e}"

    @tool
    def search_index(query: str) -> str:
        """Ambil konteks relevan dari indeks vision embedding & reranker."""
        idx = _index()
        if idx is None:
            return "(embedding tidak tersedia - retrieval dilewati)"
        docs = idx.search(query, k=4, rerank=settings.reranker_enabled)
        if not docs:
            return "(tidak ada hasil)"
        return "\n\n".join(d.page_content for d in docs)

    document_agent: SubAgent = {
        "name": "document-extractor",
        "description": "Ekstrak JSON terstruktur dari gambar dokumen (receipt, invoice, table, form, dll).",
        "system_prompt": (
            "Kamu spesialis ekstraksi dokumen. Panggil tool extract_document "
            "dengan path gambar dan teks OCR untuk mendapatkan JSON terstruktur."
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

    validator_agent: SubAgent = {
        "name": "validator",
        "description": "Validasi konsistensi matematika dan logika data dokumen.",
        "system_prompt": (
            "Kamu spesialis validasi. Panggil tool validate_data untuk memeriksa apakah "
            "total hitungan, subtotal, dan field penting sudah konsisten."
        ),
        "tools": [validate_data],
    }

    retrieval_agent: SubAgent = {
        "name": "retriever",
        "description": "Cari dan kelola konteks relevan dari indeks vision embedding & reranker.",
        "system_prompt": "Kamu spesialis retrieval. Panggil tool search_index atau index_document.",
        "tools": [search_index, index_document],
    }

    all_tools = [extract_document, ocr_document, validate_data, search_index, index_document]

    return create_deep_agent(
        name="vision-rag-agent",
        model=vlm,
        tools=all_tools,
        system_prompt=(
            "Kamu adalah Vision RAG Deep Agent tingkat lanjut. Alur kerjamu:\n"
            "1. Lakukan OCR jika teks kecil/pudar perlu dibaca presisi.\n"
            "2. Ekstrak dokumen menjadi JSON terstruktur (manfaatkan teks OCR bila ada).\n"
            "3. Validasi hasil ekstraksi dengan validator; jika ada kesalahan hitung, lakukan perbaikan.\n"
            "4. Gunakan retrieval bila konteks eksternal diperlukan."
        ),
        subagents=[document_agent, ocr_agent, validator_agent, retrieval_agent],
    )
