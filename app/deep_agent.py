"""
Harness Deep Agents untuk ekstraksi dokumen multi-modal -> Markdown Siap Chunking.
"""
from __future__ import annotations

import json
from deepagents import SubAgent, create_deep_agent
from langchain_core.tools import tool

from .agents import get_agent
from .config import Settings, get_settings
from .llm import build_vlm
from .multi_page import preview_markdown_chunks
from .ocr import build_ocr_extractor
from .ppt import process_presentation
from .preprocess import preprocess_image


def build_deep_agent(settings: Settings | None = None):
    """Bangun deep agent untuk ekstraksi dokumen ke Markdown siap chunking."""
    settings = settings or get_settings()
    vlm = build_vlm(settings)
    ocr = build_ocr_extractor(settings)

    @tool
    def extract_to_markdown(
        image_path: str,
        doc_type: str = "plain",
        ocr_text: str | None = None,
    ) -> str:
        """Ekstrak dokumen dari gambar menjadi teks Markdown bersih sesuai spesifikasi layout (plain, markdown_hierarchy, bilingual_journal, presentation_slides)."""
        proc = preprocess_image(image_path)
        agent = get_agent(doc_type)
        return agent.run(
            proc.processed_path,
            llm=vlm,
            ocr_text=ocr_text,
        )

    @tool
    def ocr_document(image_path: str) -> str:
        """OCR dokumen untuk mendapatkan teks mentah beresolusi tinggi."""
        proc = preprocess_image(image_path)
        return ocr.extract(proc.processed_path).text

    @tool
    def extract_presentation_pptx(pptx_path: str) -> str:
        """Ekstrak file presentasi PowerPoint (.pptx) menjadi Markdown terstruktur per slide."""
        return process_presentation(pptx_path)

    @tool
    def preview_chunks(markdown_text: str, chunk_size: int = 1000) -> str:
        """Simulasikan pemecahan dokumen Markdown dengan splitter berbasis header & recursive splitter."""
        chunks = preview_markdown_chunks(markdown_text, chunk_size=chunk_size)
        return json.dumps(chunks, indent=2, ensure_ascii=False)

    doc_agent: SubAgent = {
        "name": "markdown-extractor",
        "description": "Ekstrak teks Markdown terstruktur dari gambar dokumen.",
        "system_prompt": "Kamu spesialis ekstraksi Markdown dari gambar. Panggil tool extract_to_markdown.",
        "tools": [extract_to_markdown],
    }

    ocr_subagent: SubAgent = {
        "name": "ocr",
        "description": "OCR dokumen menjadi teks mentah.",
        "system_prompt": "Kamu spesialis OCR. Panggil tool ocr_document.",
        "tools": [ocr_document],
    }

    ppt_subagent: SubAgent = {
        "name": "presentation-extractor",
        "description": "Ekstrak presentasi PPTX menjadi Markdown per-slide.",
        "system_prompt": "Kamu spesialis presentasi. Panggil tool extract_presentation_pptx.",
        "tools": [extract_presentation_pptx],
    }

    chunker_subagent: SubAgent = {
        "name": "chunker",
        "description": "Simulasi pemotongan teks Markdown siap chunking.",
        "system_prompt": "Kamu spesialis chunking. Panggil tool preview_chunks.",
        "tools": [preview_chunks],
    }

    all_tools = [extract_to_markdown, ocr_document, extract_presentation_pptx, preview_chunks]

    return create_deep_agent(
        name="document-markdown-extractor-agent",
        model=vlm,
        tools=all_tools,
        system_prompt=(
            "Kamu adalah AI asisten ekstraksi dokumen multi-modal. "
            "Tugasmu membantu pengguna mengubah file dokumen (PDF, PPTX, Scan, Gambar) "
            "menjadi format Markdown bersih yang siap langsung di-chunking oleh pipeline RAG."
        ),
        subagents=[doc_agent, ocr_subagent, ppt_subagent, chunker_subagent],
    )
