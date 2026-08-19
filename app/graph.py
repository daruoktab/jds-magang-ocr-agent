"""
Orkestrasi Pipeline Ekstraksi Dokumen Vision OCR -> Markdown Siap Chunking dengan LangGraph.

Alur StateGraph:
    START -> preprocess -> ocr -> classify -> extract_markdown -> END

Fitur:
  - Preprocessing otomatis (EXIF auto-rotate & contrast enhancement).
  - OCR Auxilary Text (ocr-lighton) untuk grounding teks resolusi tinggi.
  - Klasifikasi tata letak (plain, markdown_hierarchy, bilingual_journal, presentation_slides).
  - Ekstraksi teks Markdown bersih yang siap langsung dipotong oleh text chunkers.
"""
from __future__ import annotations

import warnings
from typing import Any, TypedDict, cast

from langgraph.graph import END, START, StateGraph

from .agents import get_agent
from .config import Settings, get_settings
from .extractor import VisionExtractor
from .llm import build_vlm
from .ocr import build_ocr_extractor
from .preprocess import preprocess_image


class DocumentExtractionState(TypedDict, total=False):
    image_path: str
    preprocessed_path: str
    forced_doc_type: str | None
    previous_page_context: str | None
    doc_type: str
    ocr_text: str
    markdown_content: str


class DocumentExtractionPipeline:
    """Pipeline LangGraph untuk mengekstrak dokumen gambar/scan ke Markdown siap chunking."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.vlm = build_vlm(self.settings)
        self.ocr = build_ocr_extractor(self.settings)
        self.extractor = VisionExtractor(self.vlm)
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(cast(Any, DocumentExtractionState))

        # Node pipeline
        builder.add_node("preprocess", self._node_preprocess)
        builder.add_node("ocr", self._node_ocr)
        builder.add_node("classify", self._node_classify)
        builder.add_node("extract_markdown", self._node_extract_markdown)

        # Edges
        builder.add_edge(START, "preprocess")
        builder.add_edge("preprocess", "ocr")
        builder.add_edge("ocr", "classify")
        builder.add_edge("classify", "extract_markdown")
        builder.add_edge("extract_markdown", END)

        return builder.compile()

    def run(
        self,
        image_path: str,
        *,
        forced_doc_type: str | None = None,
        previous_page_context: str | None = None,
    ) -> dict[str, Any]:
        """Jalankan pipeline ekstraksi pada satu gambar dokumen."""
        init_state: DocumentExtractionState = {
            "image_path": image_path,
            "forced_doc_type": forced_doc_type,
            "previous_page_context": previous_page_context,
        }
        return self.graph.invoke(init_state)

    def _node_preprocess(self, state: DocumentExtractionState) -> dict[str, Any]:
        image_path = state["image_path"]
        try:
            proc = preprocess_image(image_path)
            return {"preprocessed_path": proc.processed_path}
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"Gagal melakukan preprocessing ({e}), menggunakan gambar asli.")
            return {"preprocessed_path": image_path}

    def _node_ocr(self, state: DocumentExtractionState) -> dict[str, Any]:
        img_path = state.get("preprocessed_path") or state["image_path"]
        try:
            ocr_res = self.ocr.extract(img_path)
            return {"ocr_text": ocr_res.text}
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"Panggilan OCR gagal/dilewati ({e}).")
            return {"ocr_text": ""}

    def _node_classify(self, state: DocumentExtractionState) -> dict[str, Any]:
        if state.get("forced_doc_type"):
            return {"doc_type": state["forced_doc_type"]}

        img_path = state.get("preprocessed_path") or state["image_path"]
        try:
            doc_type = self.extractor.classify(img_path)
            return {"doc_type": doc_type}
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"Klasifikasi otomatis gagal ({e}), fallback ke 'plain'.")
            return {"doc_type": "plain"}

    def _node_extract_markdown(self, state: DocumentExtractionState) -> dict[str, Any]:
        img_path = state.get("preprocessed_path") or state["image_path"]
        doc_type = state.get("doc_type") or "plain"
        ocr_text = state.get("ocr_text") or None
        previous_context = state.get("previous_page_context") or None

        agent = get_agent(doc_type)
        md_text = agent.run(
            image_path=img_path,
            llm=self.vlm,
            ocr_text=ocr_text,
            previous_page_context=previous_context,
        )

        return {"markdown_content": md_text}


# Alias untuk kompatibilitas ke belakang
VisionRAGPipeline = DocumentExtractionPipeline
