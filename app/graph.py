"""
Orkestrasi vision RAG dengan LangGraph.

Alur (StateGraph linear):
    classify -> extract -> retrieve -> build_result

- classify     : VLM mengklasifikasikan jenis dokumen (structured output).
- extract      : agent ekstraksi (VLM normal) mengubah gambar -> Pydantic.
- retrieve     : query di-embed dan dicari di indeks vektor.
- build_result : gabungkan hasil ekstraksi + konteks retrieval -> VisionRAGResult.

OCR dipisahkan sebagai kemampuan mandiri (`OCRExtractor`), tidak digabungkan
ke dalam graph di atas, karena keluarannya teks terstruktur (bukan JSON bebas).
"""
from __future__ import annotations

import operator
import warnings
from functools import cached_property
from typing import Annotated, Any, List, Optional, TypedDict, cast

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from .agents import AGENT_REGISTRY, get_agent
from .config import Settings, get_settings
from .embedding import build_embeddings
from .llm import build_vlm, image_data_uri
from .prompts import CLASSIFY_PROMPT, CLASSIFY_SYSTEM
from .schemas import (
    DocumentClassification,
    DocumentExtraction,
    RetrievedChunk,
    VisionRAGResult,
)
from .vector_store import VisionIndex


class VisionRAGState(TypedDict, total=False):
    image_path: str
    query: str
    doc_type: str
    extraction: dict
    retrieved_docs: Annotated[List[Document], operator.add]
    final_result: dict


class VisionRAGPipeline:
    """Pipeline LangGraph untuk vision RAG (VLM normal + embedding)."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.vlm = build_vlm(self.settings)
        self._classifier = self.vlm.with_structured_output(DocumentClassification)
        self.graph = self._build_graph()

    @cached_property
    def index(self) -> VisionIndex:
        # Malas: hanya butuh binary llama-vl-embedding saat retrieval dipakai.
        return VisionIndex(build_embeddings(self.settings))

    # --- LangGraph -------------------------------------------------------
    def _build_graph(self):
        builder = StateGraph(cast(Any, VisionRAGState))
        builder.add_node("classify", self._classify)
        builder.add_node("extract", self._extract)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("build_result", self._build_result)
        builder.add_edge(START, "classify")
        builder.add_edge("classify", "extract")
        builder.add_edge("extract", "retrieve")
        builder.add_edge("retrieve", "build_result")
        builder.add_edge("build_result", END)
        return builder.compile()

    # --- API publik ------------------------------------------------------
    def classify(self, image_path: str) -> str:
        """Klasifikasikan jenis dokumen saja (tanpa ekstraksi/retrieval)."""
        types = "\n".join(
            f"- {a.name}: {a.description}" for a in AGENT_REGISTRY.values()
        )
        prompt = CLASSIFY_PROMPT.format(types=types)
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data_uri(image_path)}},
        ]
        messages = [SystemMessage(content=CLASSIFY_SYSTEM), HumanMessage(content=cast(Any, content))]
        result = cast(DocumentClassification, self._classifier.invoke(messages))
        return result.doc_type

    def run(self, image_path: str, query: Optional[str] = None) -> dict:
        """Jalankan pipeline penuh, kembalikan state akhir (dict)."""
        return self.graph.invoke({"image_path": image_path, "query": query or ""})

    # --- node ------------------------------------------------------------
    def _classify(self, state: VisionRAGState) -> dict:
        return {"doc_type": self.classify(state["image_path"])}

    def _extract(self, state: VisionRAGState) -> dict:
        doc_type = state.get("doc_type") or "generic"
        result = get_agent(doc_type).run(state["image_path"], self.vlm)
        return {"extraction": result.model_dump()}

    def _retrieve(self, state: VisionRAGState) -> dict:
        # Embedding bisa di-off via EMBEDDING_ENABLED=false (VLM/OCR saja).
        if not self.settings.embedding_enabled:
            return {"retrieved_docs": []}
        query = state.get("query") or state.get("doc_type") or ""
        if not query:
            return {"retrieved_docs": []}
        # Auto-degrade: kalau binary/embedding tidak tersedia, lewati retrieval.
        try:
            return {"retrieved_docs": self.index.search(query, k=4)}
        except Exception as e:  # noqa: BLE001 - binary belum dibuild, dll.
            warnings.warn(f"Embedding tidak tersedia, node retrieve dilewati: {e}")
            return {"retrieved_docs": []}

    def _build_result(self, state: VisionRAGState) -> dict:
        extraction = DocumentExtraction(**state["extraction"])
        context = [
            RetrievedChunk(content=d.page_content, metadata=d.metadata)
            for d in state.get("retrieved_docs", [])
        ]
        result = VisionRAGResult(
            doc_type=state.get("doc_type", "generic"),
            extraction=extraction,
            context=context,
        )
        return {"final_result": result.model_dump()}
