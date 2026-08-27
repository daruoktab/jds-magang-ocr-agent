"""
Orkestrasi Pipeline Ekstraksi Dokumen Vision OCR -> Markdown Siap Chunking dengan LangGraph.
Mendukung multi-spesifikasi komposit layout dokumen dengan logging transparan.

Alur StateGraph:
    START -> preprocess -> ocr -> classify -> extract_markdown -> END
"""

from __future__ import annotations

import logging
import time
from typing import Any, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .agents import get_agent
from .config import Settings, get_settings
from .extractor import VisionExtractor
from .llm import build_vlm
from .ocr import build_ocr_extractor
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
    ocr_text: str
    markdown_content: str


class DocumentExtractionPipeline:
    """Pipeline LangGraph untuk mengekstrak dokumen gambar/scan ke Markdown siap chunking."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings: Settings = settings or get_settings()
        self.vlm = build_vlm(self.settings)
        self.ocr = build_ocr_extractor(self.settings)
        self.extractor = VisionExtractor(self.vlm)
        self.graph: CompiledStateGraph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:
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
        forced_specs: list[str] | str | None = None,
        forced_doc_type: str | None = None,
        previous_page_context: str | None = None,
    ) -> dict[str, Any]:
        """Jalankan pipeline ekstraksi komposit pada satu gambar dokumen dengan pelacakan waktu & log terperinci."""
        start_t = time.perf_counter()
        logger.info(
            "================================================================================"
        )
        logger.info(
            "[Workflow] Memulai pipeline ekstraksi untuk file: '%s'", image_path
        )
        if forced_specs or forced_doc_type:
            logger.info(
                "[Workflow] Override spesifikasi layout: %s",
                forced_specs or forced_doc_type,
            )

        init_state: DocumentExtractionState = {
            "image_path": image_path,
            "forced_specs": forced_specs or forced_doc_type,
            "forced_doc_type": forced_doc_type,
            "previous_page_context": previous_page_context,
        }
        try:
            result = cast(dict[str, Any], self.graph.invoke(init_state))
            elapsed = time.perf_counter() - start_t
            md_len = len(result.get("markdown_content", ""))
            specs_used = result.get("specs", [])
            logger.info(
                "[Workflow Selesai] Total waktu: %.2fs | Specs: %s | Markdown: %d karakter",
                elapsed,
                specs_used,
                md_len,
            )
            logger.info(
                "================================================================================"
            )
            return result
        except Exception:
            elapsed = time.perf_counter() - start_t
            logger.exception(
                "[Workflow Gagal] Terjadi error pada pipeline setelah %.2fs untuk file '%s'",
                elapsed,
                image_path,
            )
            raise

    def _node_preprocess(self, state: DocumentExtractionState) -> dict[str, Any]:
        image_path = state["image_path"]
        t0 = time.perf_counter()
        logger.info("[Node 1/4: Preprocess] Menyiapkan gambar dokumen...")
        try:
            proc = preprocess_image(image_path)
            dt = time.perf_counter() - t0
            logger.info(
                "[Node 1/4: Preprocess] Selesai (%.2fs) | Path: %s | Modifikasi: %s | Dimensi: %s",
                dt,
                proc.processed_path,
                proc.is_modified,
                proc.dimensions,
            )
            return {"preprocessed_path": proc.processed_path}
        except Exception as e:
            dt = time.perf_counter() - t0
            logger.warning(
                "[Node 1/4: Preprocess] Gagal dalam %.2fs (%s). Menggunakan gambar asli: '%s'",
                dt,
                e,
                image_path,
                exc_info=True,
            )
            return {"preprocessed_path": image_path}

    def _node_ocr(self, state: DocumentExtractionState) -> dict[str, Any]:
        img_path = state.get("preprocessed_path") or state["image_path"]
        t0 = time.perf_counter()
        logger.info(
            "[Node 2/4: OCR] Mengekstrak referensi teks mentah via model OCR..."
        )
        try:
            ocr_res = self.ocr.extract(img_path)
            dt = time.perf_counter() - t0
            text_len = len(ocr_res.text)
            sample = ocr_res.text[:60].replace("\n", " ").strip()
            preview = f" ('{sample}...')" if text_len > 60 else f" ('{sample}')"
            logger.info(
                "[Node 2/4: OCR] Selesai (%.2fs) | Teks OCR: %d karakter%s",
                dt,
                text_len,
                preview if text_len > 0 else "",
            )
            return {"ocr_text": ocr_res.text}
        except Exception as e:
            dt = time.perf_counter() - t0
            logger.warning(
                "[Node 2/4: OCR] Panggilan OCR gagal/dilewati dalam %.2fs: %s",
                dt,
                e,
                exc_info=True,
            )
            return {"ocr_text": ""}

    def _node_classify(self, state: DocumentExtractionState) -> dict[str, Any]:
        forced = state.get("forced_specs") or state.get("forced_doc_type")
        if forced:
            specs = normalize_specs(forced)
            logger.info(
                "[Node 3/4: Classify] Spesifikasi layout dipaksa (forced): %s", specs
            )
            return {"specs": specs, "doc_type": specs[0]}

        img_path = state.get("preprocessed_path") or state["image_path"]
        t0 = time.perf_counter()
        logger.info(
            "[Node 3/4: Classify] Mengidentifikasi karakteristik layout dokumen via VLM..."
        )
        try:
            specs = self.extractor.classify(img_path)
            dt = time.perf_counter() - t0
            logger.info(
                "[Node 3/4: Classify] Selesai (%.2fs) | Terdeteksi: %s", dt, specs
            )
            return {"specs": specs, "doc_type": specs[0] if specs else "plain"}
        except Exception as e:
            dt = time.perf_counter() - t0
            logger.warning(
                "[Node 3/4: Classify] Klasifikasi otomatis gagal dalam %.2fs (%s). Fallback ke ['plain']",
                dt,
                e,
                exc_info=True,
            )
            return {"specs": ["plain"], "doc_type": "plain"}

    def _node_extract_markdown(self, state: DocumentExtractionState) -> dict[str, Any]:
        img_path = state.get("preprocessed_path") or state["image_path"]
        specs = state.get("specs") or ["plain"]
        ocr_text = state.get("ocr_text") or None
        previous_context = state.get("previous_page_context") or None

        t0 = time.perf_counter()
        logger.info(
            "[Node 4/4: Extract] Menjalankan ekstraksi Markdown dengan spesifikasi: %s...",
            specs,
        )
        agent = get_agent(specs)
        try:
            md_text = agent.run(
                image_path=img_path,
                llm=self.vlm,
                ocr_text=ocr_text,
                previous_page_context=previous_context,
            )
            dt = time.perf_counter() - t0
            logger.info(
                "[Node 4/4: Extract] Selesai (%.2fs) | Panjang Markdown: %d karakter | %d baris",
                dt,
                len(md_text),
                len(md_text.splitlines()),
            )
            return {"markdown_content": md_text}
        except Exception:
            dt = time.perf_counter() - t0
            logger.exception(
                "[Node 4/4: Extract] Gagal dalam %.2fs saat ekstraksi Markdown",
                dt,
            )
            raise


# Alias untuk kompatibilitas ke belakang
VisionRAGPipeline = DocumentExtractionPipeline
