"""
Registry agent spesialis ekstraksi dokumen ke Markdown siap chunking.
Mendukung multi-spesifikasi komposit layout dokumen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.language_models.chat_models import BaseChatModel

from .extractor import VisionExtractor
from .prompts import SYSTEM_DOCUMENT_EXTRACTOR, normalize_specs


@dataclass
class DocumentExtractionAgent:
    """Agent ekstraksi untuk satu atau kombinasi spesifikasi layout dokumen."""

    name: str
    description: str
    specs: list[str] = field(default_factory=lambda: ["plain"])
    system_prompt: str = SYSTEM_DOCUMENT_EXTRACTOR

    @property
    def doc_type(self) -> str:
        """Alias spesifikasi utama untuk kompatibilitas."""
        return self.specs[0] if self.specs else "plain"

    def build(self, llm: BaseChatModel) -> VisionExtractor:
        """Bangun instance VisionExtractor yang dikonfigurasi dengan system prompt agent."""
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
        """Jalankan ekstraksi Markdown komposit pada gambar input."""
        extractor = self.build(llm)
        return extractor.extract_markdown(
            image_path=image_path,
            specs=self.specs,
            ocr_text=ocr_text,
            previous_page_context=previous_page_context,
        )


AGENT_REGISTRY: dict[str, DocumentExtractionAgent] = {
    "plain": DocumentExtractionAgent(
        name="plain",
        description="Dokumen standar / biasa (surat, formulir, memo, teks umum)",
        specs=["plain"],
    ),
    "markdown_hierarchy": DocumentExtractionAgent(
        name="markdown_hierarchy",
        description="Dokumen hierarki Markdown (#, ##, ### yang runtut dan tidak putus)",
        specs=["markdown_hierarchy"],
    ),
    "bilingual_journal": DocumentExtractionAgent(
        name="bilingual_journal",
        description="Jurnal ilmiah / dokumen 2-kolom & 2-bahasa (column-aware reading order)",
        specs=["bilingual_journal"],
    ),
    "presentation_slides": DocumentExtractionAgent(
        name="presentation_slides",
        description="Dokumen slide presentasi PPT / PDF (bullet points, diagram & visual)",
        specs=["presentation_slides"],
    ),
}


def get_agent(specs: list[str] | str | None = None) -> DocumentExtractionAgent:
    """
    Ambil atau bangun agent berdasarkan satu atau kombinasi nama spesifikasi.
    Mendukung format 'journal,hierarchy' atau ['bilingual_journal', 'markdown_hierarchy'].
    """
    normalized = normalize_specs(specs)

    # Jika spesifikasi tunggal dan terdaftar
    if len(normalized) == 1 and normalized[0] in AGENT_REGISTRY:
        return AGENT_REGISTRY[normalized[0]]

    # Jika kombinasi multi-spesifikasi
    combo_name = "+".join(normalized)
    desc = " + ".join(
        AGENT_REGISTRY[s].description if s in AGENT_REGISTRY else s for s in normalized
    )
    return DocumentExtractionAgent(
        name=combo_name,
        description=f"Komposit: {desc}",
        specs=normalized,
    )
