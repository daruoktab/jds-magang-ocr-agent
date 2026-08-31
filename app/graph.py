"""
Orkestrasi Pipeline Ekstraksi Dokumen VLM -> Markdown Siap Chunking dengan LangGraph.
Mendukung multi-spesifikasi komposit layout dokumen dengan logging transparan.

Alur StateGraph:
    START -> preprocess -> classify -> extract_markdown -> END
"""

from __future__ import annotations

import logging
import time
from typing import Any, TypedDict, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .agents import get_agent
from .config import Settings, get_settings
from .extractor import VisionExtractor
from .llm import build_vlm
from .preprocess import preprocess_image
from .prompts import normalize_specs

logger = logging.getLogger("app.graph")


class DocumentExtractionState(TypedDict, total=False):
    image_path: str
    preprocessed_path: str
    forced_specs: list[str] | str | None
    forced_doc_type: str | None
    previous_page_context: str | None
    specs: list[str]
    doc_type: str
    markdown_content: str


class DocumentExtractionPipeline:
    """Pipeline LangGraph untuk mengekstrak dokumen gambar/scan ke Markdown siap chunking."""

    def __init__(
        self,
        settings: Settings | None = None,
        vlm: BaseChatModel | Any | None = None,
    ) -> None:
        self.settings: Settings = settings or get_settings()
        self.vlm = vlm or build_vlm(self.settings)
        self.extractor = VisionExtractor(self.vlm)
        self.graph: CompiledStateGraph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:
        builder = StateGraph(cast(Any, DocumentExtractionState))

        # Node pipeline
        builder.add_node("preprocess", self._node_preprocess)
        builder.add_node("classify", self._node_classify)
        builder.add_node("extract_markdown", self._node_extract_markdown)

        # Edges
        builder.add_edge(START, "preprocess")
        builder.add_edge("preprocess", "classify")
        builder.add_edge("classify", "extract_markdown")
        builder.add_edge("extract_markdown", END)

        return builder.compile()

    def run(
        self,
        image_path: str,
        *,
        forced_specs: list[str] | str | None = None,
        forced_doc_type: str | None = None,
        previous_page_context: str | None = None,
    ) -> dict[str, Any]:
        """
        Jalankan pipeline ekstraksi lengkap pada satu gambar halaman dokumen.

        Returns:
            Dict berisi:
                - preprocessed_path: path gambar yang telah di-preprocess
                - specs: list string spesifikasi yang terdeteksi
                - doc_type: string spesifikasi gabungan terurut (kompatibilitas)
                - markdown_content: string teks Markdown hasil ekstraksi
        """
        initial_state: DocumentExtractionState = {
            "image_path": image_path,
            "forced_specs": forced_specs,
            "forced_doc_type": forced_doc_type,
            "previous_page_context": previous_page_context,
        }

        logger.info("[Pipeline] Memulai ekstraksi: %s", image_path)
        final_state = cast(dict[str, Any], self.graph.invoke(initial_state))

        return {
            "preprocessed_path": final_state.get("preprocessed_path", image_path),
            "specs": final_state.get("specs", ["plain"]),
            "doc_type": final_state.get("doc_type", "plain"),
            "markdown_content": final_state.get("markdown_content", ""),
        }

    # =========================================================================
    # Node Implementations
    # =========================================================================

    def _node_preprocess(
        self, state: DocumentExtractionState
    ) -> DocumentExtractionState:
        """Tahap 1: Preprocessing gambar."""
        t0 = time.perf_counter()
        image_path = state["image_path"]

        result = preprocess_image(image_path)
        elapsed = (time.perf_counter() - t0) * 1000

        logger.info(
            "[Pipeline:Preprocess] %s -> %s (modified: %s, %dx%d, %.1fms)",
            image_path,
            result.processed_path,
            result.is_modified,
            result.dimensions[0],
            result.dimensions[1],
            elapsed,
        )

        return {
            **state,
            "preprocessed_path": result.processed_path,
        }

    def _node_classify(
        self, state: DocumentExtractionState
    ) -> DocumentExtractionState:
        """Tahap 2: Klasifikasi multi-trait karakteristik dokumen."""
        forced_specs = state.get("forced_specs")
        forced_doc_type = state.get("forced_doc_type")

        # Jika dipaksa manual oleh user, gunakan langsung
        if forced_specs:
            norm_specs = normalize_specs(forced_specs)
            logger.info(
                "[Pipeline:Classify] Menggunakan forced_specs: %s", norm_specs
            )
            return {
                **state,
                "specs": norm_specs,
                "doc_type": ",".join(norm_specs),
            }

        if forced_doc_type:
            norm_specs = normalize_specs(forced_doc_type)
            logger.info(
                "[Pipeline:Classify] Menggunakan forced_doc_type: %s", norm_specs
            )
            return {
                **state,
                "specs": norm_specs,
                "doc_type": ",".join(norm_specs),
            }

        # Klasifikasi otomatis via VLM
        t0 = time.perf_counter()
        img = state.get("preprocessed_path") or state["image_path"]
        detected_specs = self.extractor.classify(img)
        elapsed = (time.perf_counter() - t0) * 1000

        logger.info(
            "[Pipeline:Classify] Karakteristik terdeteksi: %s (%.1fms)",
            detected_specs,
            elapsed,
        )

        return {
            **state,
            "specs": detected_specs,
            "doc_type": ",".join(detected_specs),
        }

    def _node_extract_markdown(
        self, state: DocumentExtractionState
    ) -> DocumentExtractionState:
        """Tahap 3: Ekstraksi teks Markdown menggunakan Specialized Composite Agent."""
        t0 = time.perf_counter()
        img = state.get("preprocessed_path") or state["image_path"]
        specs = state.get("specs", ["plain"])
        prev_context = state.get("previous_page_context")

        # Dapatkan agen komposit yang sesuai dengan seluruh trait spesifikasi
        agent = get_agent(specs)
        md_text = agent.run(
            img,
            llm=self.vlm,
            previous_page_context=prev_context,
        )

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "[Pipeline:ExtractMarkdown] Selesai (%d karakter, %.1fms)",
            len(md_text),
            elapsed,
        )

        return {
            **state,
            "markdown_content": md_text,
        }


VisionRAGPipeline = DocumentExtractionPipeline

__all__ = [
    "DocumentExtractionPipeline",
    "DocumentExtractionState",
    "VisionRAGPipeline",
]
