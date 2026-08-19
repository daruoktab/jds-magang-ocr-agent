"""
Registry agent spesialis ekstraksi dokumen ke Markdown siap chunking.
"""
from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel

from .extractor import VisionExtractor
from .prompts import (
    SYSTEM_DOCUMENT_EXTRACTOR,
    _PROMPT_BILINGUAL_JOURNAL,
    _PROMPT_MARKDOWN_HIERARCHY,
    _PROMPT_PLAIN_DOCUMENT,
    _PROMPT_PRESENTATION_SLIDES,
)


@dataclass
class DocumentExtractionAgent:
    """Agent ekstraksi untuk satu spesifikasi layout dokumen."""

    name: str
    description: str
    doc_type: str
    system_prompt: str = SYSTEM_DOCUMENT_EXTRACTOR

    def build(self, llm: BaseChatModel) -> VisionExtractor:
        return VisionExtractor(
            llm=llm,
            system_prompt=self.system_prompt,
        )

    def run(
        self,
        image_path: str,
        llm: BaseChatModel,
        *,
        ocr_text: str | None = None,
        previous_page_context: str | None = None,
    ) -> str:
        extractor = self.build(llm)
        return extractor.extract_markdown(
            image_path=image_path,
            doc_type=self.doc_type,
            ocr_text=ocr_text,
            previous_page_context=previous_page_context,
        )


AGENT_REGISTRY: dict[str, DocumentExtractionAgent] = {
    "plain": DocumentExtractionAgent(
        name="plain",
        description="Dokumen standar / biasa (surat, formulir, memo, teks umum)",
        doc_type="plain",
    ),
    "markdown_hierarchy": DocumentExtractionAgent(
        name="markdown_hierarchy",
        description="Dokumen hierarki Markdown (#, ##, ### yang runtut dan tidak putus)",
        doc_type="markdown_hierarchy",
    ),
    "bilingual_journal": DocumentExtractionAgent(
        name="bilingual_journal",
        description="Jurnal ilmiah / dokumen 2-kolom & 2-bahasa (column-aware reading order)",
        doc_type="bilingual_journal",
    ),
    "presentation_slides": DocumentExtractionAgent(
        name="presentation_slides",
        description="Dokumen slide presentasi PPT / PDF (bullet points, diagram & visual)",
        doc_type="presentation_slides",
    ),
}


def get_agent(name: str) -> DocumentExtractionAgent:
    """Ambil agent berdasarkan nama spesifikasi. Default -> 'plain'."""
    key = name.lower()
    if key in AGENT_REGISTRY:
        return AGENT_REGISTRY[key]
    if "journal" in key or "bilingual" in key or "2col" in key:
        return AGENT_REGISTRY["bilingual_journal"]
    if "hierarchy" in key or "markdown" in key:
        return AGENT_REGISTRY["markdown_hierarchy"]
    if "slide" in key or "ppt" in key or "presentation" in key:
        return AGENT_REGISTRY["presentation_slides"]
    return AGENT_REGISTRY["plain"]
