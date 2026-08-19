"""
Pipeline ekstraksi: gambar dokumen -> VLM (+ OCR Fusion) -> Markdown Bersih Siap Chunking.
Mendukung multi-spesifikasi karakteristik tata letak dokumen secara komposit.
"""
from __future__ import annotations

import json
import re
from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .llm import image_data_uri
from .prompts import (
    CLASSIFY_PROMPT,
    CLASSIFY_SYSTEM,
    SYSTEM_DOCUMENT_EXTRACTOR,
    build_extraction_prompt,
    normalize_specs,
)


class VisionExtractor:
    """Ekstraktor dokumen multimodal: Mengubah gambar dokumen menjadi Markdown siap chunking."""

    def __init__(
        self,
        llm: BaseChatModel,
        system_prompt: str = SYSTEM_DOCUMENT_EXTRACTOR,
    ) -> None:
        self.llm: BaseChatModel = llm
        self.system_prompt: str = system_prompt

    def classify(self, image_path: str) -> list[str]:
        """
        Klasifikasikan satu atau lebih karakteristik layout dokumen yang ada pada gambar:
        ['plain'], ['bilingual_journal', 'markdown_hierarchy'], dsb.
        """
        content: list[dict[str, Any]] = [
            {"type": "text", "text": CLASSIFY_PROMPT},
            {"type": "image_url", "image_url": {"url": image_data_uri(image_path)}},
        ]
        messages = [
            SystemMessage(content=CLASSIFY_SYSTEM),
            HumanMessage(content=cast(Any, content)),
        ]

        resp = self.llm.invoke(messages)
        text_resp = str(resp.content).strip()

        # Parse JSON output jika ada
        match = re.search(r"\{.*?\}", text_resp, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                raw_specs = data.get("specs") or [data.get("doc_type")]
                return normalize_specs(raw_specs)
            except Exception:
                pass

        # Fallback multi-matching via regex
        text_lower = text_resp.lower()
        detected: list[str] = []
        if "journal" in text_lower or "bilingual" in text_lower or "2col" in text_lower:
            detected.append("bilingual_journal")
        if "hierarchy" in text_lower or "markdown" in text_lower or "heading" in text_lower:
            detected.append("markdown_hierarchy")
        if "slide" in text_lower or "presentation" in text_lower or "ppt" in text_lower:
            detected.append("presentation_slides")

        return detected if detected else ["plain"]

    def extract_markdown(
        self,
        image_path: str,
        *,
        specs: list[str] | str | None = None,
        ocr_text: str | None = None,
        previous_page_context: str | None = None,
    ) -> str:
        """
        Ekstrak gambar dokumen menjadi teks Markdown bersih sesuai spesifikasi layout komposit.

        Args:
            image_path: Path ke file gambar dokumen.
            specs: Satu atau daftar karakteristik dokumen ('plain', 'markdown_hierarchy', 'bilingual_journal', 'presentation_slides').
            ocr_text: Teks mentah OCR tambahan untuk grounding / fusion.
            previous_page_context: Konteks halaman sebelumnya untuk menjaga kontinuitas header.

        Returns:
            Teks Markdown terstruktur siap chunking.
        """
        user_prompt = build_extraction_prompt(
            specs=specs,
            ocr_text=ocr_text,
            previous_page_context=previous_page_context,
        )

        content: list[dict[str, Any]] = [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": image_data_uri(image_path)}},
        ]

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=cast(Any, content)),
        ]

        response = self.llm.invoke(messages)
        md_text = str(response.content).strip()

        # Bersihkan pembungkus markdown block ```markdown ... ``` jika VLM membungkusnya
        if md_text.startswith("```markdown") and md_text.endswith("```"):
            md_text = md_text[len("```markdown"):-3].strip()
        elif md_text.startswith("```md") and md_text.endswith("```"):
            md_text = md_text[len("```md"):-3].strip()
        elif md_text.startswith("```") and md_text.endswith("```"):
            md_text = md_text[3:-3].strip()

        return md_text
