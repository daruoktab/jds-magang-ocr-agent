"""
Ekstraksi OCR via model `ocr-lighton` (VLM kecil yang di-tuning untuk OCR).

Format request = chat biasa dengan `content` berupa ARRAY `[text, image_url]`
(bukan string), persis seperti endpoint `curl` yang terdokumentasi. Outputnya
TEKS biasa di `choices[0].message.content` (bukan structured JSON), sehingga
TIDAK memakai `with_structured_output`.
"""
from __future__ import annotations

from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from .llm import image_data_uri
from .schemas import OCRResult

# Prompt ringkas untuk OCR (model kecil, cukup "extract all text").
OCR_DEFAULT_PROMPT = "Extract all text from this image."


class OCRExtractor:
    """OCR dokumen -> teks terstruktur (`OCRResult`)."""

    def __init__(
        self,
        llm: BaseChatModel,
        prompt: str = OCR_DEFAULT_PROMPT,
    ) -> None:
        self.llm = llm
        self.prompt = prompt

    def extract(self, image_path: str) -> OCRResult:
        content: list[dict] = [
            {"type": "text", "text": self.prompt},
            {"type": "image_url", "image_url": {"url": image_data_uri(image_path)}},
        ]

        # Catatan: `content` HARUS array (text + image_url), bukan string.
        message = HumanMessage(content=content)
        resp = self.llm.invoke([message])

        # Keluaran model OCR = teks biasa.
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        return OCRResult(text=text.strip())


def build_ocr_extractor(settings=None) -> OCRExtractor:
    from .config import get_settings
    from .llm import build_ocr

    settings = settings or get_settings()
    return OCRExtractor(build_ocr(settings))
