"""
Orkestrasi Vision RAG Agentik dengan LangGraph & Self-Correction Feedback Loop.

Alur StateGraph Agentik:
    START -> preprocess -> ocr -> classify -> extract -> validate
                                                ^           |
                                                |           v
                                             reflect <-- [is_valid=False & retry < max]
                                                            |
                                                   [is_valid=True / max_retry]
                                                            |
                                                            v
                                                        retrieve -> build_result -> END

Fitur:
  - Multimodal Vision + OCR Fusion (teks OCR membantu VLM untuk teks kecil/pudar).
  - Preprocessing otomatis (EXIF auto-rotate & contrast enhancement).
  - Agentic Reflection & Validation (cek konsistensi matematika, tanggal, dan format data).
  - Two-Stage Retrieval (Embeddings + Qwen3-VL-Reranker jika aktif).
"""
from __future__ import annotations

import operator
import warnings
from functools import cached_property
from typing import Annotated, Any, Literal, TypedDict, cast

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from .agents import AGENT_REGISTRY, get_agent
from .config import Settings, get_settings
from .embedding import build_embeddings
from .llm import build_vlm, image_data_uri
from .ocr import build_ocr_extractor
from .preprocess import preprocess_image
from .prompts import CLASSIFY_PROMPT, CLASSIFY_SYSTEM
from .reranker import build_reranker
from .schemas import (
    DocumentClassification,
    DocumentExtraction,
    RetrievedChunk,
    ValidationSummary,
    VisionRAGResult,
)
from .validation import validate_extraction
from .vector_store import VisionIndex


class VisionRAGState(TypedDict, total=False):
    image_path: str
    preprocessed_path: str
    query: str
    doc_type: str
    ocr_text: str
    extraction: dict[str, Any]
    validation: dict[str, Any]
    critique: str
    retry_count: int
    max_retries: int
    retrieved_docs: Annotated[list[Document], operator.add]
    final_result: dict[str, Any]


class VisionRAGPipeline:
    """Pipeline LangGraph Agentik untuk Vision RAG (VLM + OCR Fusion + Self-Correction)."""

    def __init__(self, settings: Settings | None = None, max_retries: int = 2) -> None:
        self.settings = settings or get_settings()
        self.max_retries = max_retries
        self.vlm = build_vlm(self.settings)
        self.ocr = build_ocr_extractor(self.settings)
        self._classifier = self.vlm.with_structured_output(DocumentClassification)
        self.graph = self._build_graph()

    @cached_property
    def index(self) -> VisionIndex:
        """Inisialisasi index vektor & reranker secara lazy."""
        embeddings = build_embeddings(self.settings)
        reranker = build_reranker(self.settings)
        return VisionIndex(embeddings=embeddings, reranker=reranker)

    # --- LangGraph Builder -----------------------------------------------
    def _build_graph(self):
        builder = StateGraph(cast(Any, VisionRAGState))

        # Daftarkan node-node
        builder.add_node("preprocess", self._node_preprocess)
        builder.add_node("ocr", self._node_ocr)
        builder.add_node("classify", self._node_classify)
        builder.add_node("extract", self._node_extract)
        builder.add_node("validate", self._node_validate)
        builder.add_node("reflect", self._node_reflect)
        builder.add_node("retrieve", self._node_retrieve)
        builder.add_node("build_result", self._node_build_result)

        # Edges utama
        builder.add_edge(START, "preprocess")
        builder.add_edge("preprocess", "ocr")
        builder.add_edge("ocr", "classify")
        builder.add_edge("classify", "extract")
        builder.add_edge("extract", "validate")

        # Conditional Edge: Self-Correction Loop
        builder.add_conditional_edges(
            "validate",
            self._route_after_validation,
            ["reflect", "retrieve"],
        )
        builder.add_edge("reflect", "extract")

        # Jalur akhir
        builder.add_edge("retrieve", "build_result")
        builder.add_edge("build_result", END)

        return builder.compile()

    # --- Router & Conditional Logic --------------------------------------
    def _route_after_validation(self, state: VisionRAGState) -> Literal["reflect", "retrieve"]:
        validation = state.get("validation", {})
        is_valid = validation.get("is_valid", True)
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", self.max_retries)

        if not is_valid and retry_count < max_retries:
            return "reflect"
        return "retrieve"

    # --- Public API ------------------------------------------------------
    def classify(self, image_path: str) -> str:
        """Klasifikasikan jenis dokumen dari gambar."""
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

    def run(
        self,
        image_path: str,
        query: str | None = None,
        max_retries: int | None = None,
    ) -> dict[str, Any]:
        """Jalankan pipeline agentik penuh, kembalikan state akhir."""
        init_state: VisionRAGState = {
            "image_path": image_path,
            "query": query or "",
            "retry_count": 0,
            "max_retries": max_retries if max_retries is not None else self.max_retries,
            "retrieved_docs": [],
        }
        return self.graph.invoke(init_state)

    # --- Node Implementations --------------------------------------------
    def _node_preprocess(self, state: VisionRAGState) -> dict[str, Any]:
        image_path = state["image_path"]
        try:
            proc = preprocess_image(image_path)
            return {"preprocessed_path": proc.processed_path}
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"Gagal melakukan preprocessing ({e}), menggunakan gambar asli.")
            return {"preprocessed_path": image_path}

    def _node_ocr(self, state: VisionRAGState) -> dict[str, Any]:
        img_path = state.get("preprocessed_path") or state["image_path"]
        try:
            ocr_res = self.ocr.extract(img_path)
            return {"ocr_text": ocr_res.text}
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"Panggilan OCR gagal/dilewati ({e}).")
            return {"ocr_text": ""}

    def _node_classify(self, state: VisionRAGState) -> dict[str, Any]:
        img_path = state.get("preprocessed_path") or state["image_path"]
        doc_type = self.classify(img_path)
        return {"doc_type": doc_type}

    def _node_extract(self, state: VisionRAGState) -> dict[str, Any]:
        img_path = state.get("preprocessed_path") or state["image_path"]
        doc_type = state.get("doc_type") or "generic"
        ocr_text = state.get("ocr_text") or None
        critique = state.get("critique") or None

        agent = get_agent(doc_type)
        result = agent.run(
            image_path=img_path,
            llm=self.vlm,
            ocr_text=ocr_text,
            critique=critique,
        )
        return {"extraction": result.model_dump()}

    def _node_validate(self, state: VisionRAGState) -> dict[str, Any]:
        doc_type = state.get("doc_type") or "generic"
        raw_extraction = state.get("extraction", {})
        data = raw_extraction.get("data", {})

        val_res = validate_extraction(doc_type=doc_type, data=data)
        val_summary = {
            "is_valid": val_res.is_valid,
            "score": val_res.score,
            "issues": val_res.issues,
            "critique_text": val_res.format_critique(),
            "reflection_attempts": state.get("retry_count", 0),
        }
        return {"validation": val_summary}

    def _node_reflect(self, state: VisionRAGState) -> dict[str, Any]:
        current_retry = state.get("retry_count", 0) + 1
        val_summary = state.get("validation", {})
        critique_text = val_summary.get("critique_text", "")

        return {
            "retry_count": current_retry,
            "critique": critique_text,
        }

    def _node_retrieve(self, state: VisionRAGState) -> dict[str, Any]:
        if not self.settings.embedding_enabled:
            return {"retrieved_docs": []}

        query = state.get("query") or state.get("doc_type") or ""
        if not query:
            return {"retrieved_docs": []}

        try:
            docs = self.index.search(query, k=4, rerank=self.settings.reranker_enabled)
            return {"retrieved_docs": docs}
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"Retrieval gagal ({e}), node dilewati.")
            return {"retrieved_docs": []}

    def _node_build_result(self, state: VisionRAGState) -> dict[str, Any]:
        extraction = DocumentExtraction(**state["extraction"])
        val_data = state.get("validation", {})
        val_summary = ValidationSummary(
            is_valid=val_data.get("is_valid", True),
            score=val_data.get("score", 1.0),
            issues=val_data.get("issues", []),
            reflection_attempts=state.get("retry_count", 0),
        )

        context = [
            RetrievedChunk(content=d.page_content, metadata=d.metadata)
            for d in state.get("retrieved_docs", [])
        ]

        result = VisionRAGResult(
            doc_type=state.get("doc_type", "generic"),
            extraction=extraction,
            ocr_text=state.get("ocr_text") or None,
            context=context,
            validation=val_summary,
        )
        return {"final_result": result.model_dump()}
