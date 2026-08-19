"""
Ekstraksi OCR via model `ocr-lighton` (VLM kecil yang di-tuning khusus untuk OCR).

Format request = chat biasa dengan `content` berupa list `[{"type": "text", ...}, {"type": "image_url", ...}]`.
Output teks biasa di `choices[0].message.content`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from .llm import image_data_uri
from .schemas import OCRResult

if TYPE_CHECKING:
    from .config import Settings

OCR_DEFAULT_PROMPT: str = "Extract all text from this image."


class OCRExtractor:
    """OCR dokumen -> teks mentah (`OCRResult`)."""

    def __init__(
        self,
        llm: BaseChatModel,
        prompt: str = OCR_DEFAULT_PROMPT,
    ) -> None:
        self.llm = llm
        self.prompt = prompt

    def extract(self, image_path: str) -> OCRResult:
        """Ekstrak seluruh teks mentah dari file gambar menggunakan model OCR."""
        content: list[dict[str, Any]] = [
            {"type": "text", "text": self.prompt},
            {"type": "image_url", "image_url": {"url": image_data_uri(image_path)}},
        ]

        message = HumanMessage(content=cast(Any, content))
        resp = self.llm.invoke([message])

        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        return OCRResult(text=text.strip())


def build_ocr_extractor(settings: Settings | None = None) -> OCRExtractor:
    """Bangun instance OCRExtractor berdasarkan konfigurasi Settings."""
    from .config import get_settings
    from .llm import build_ocr

    resolved_settings = settings or get_settings()
    return OCRExtractor(build_ocr(resolved_settings))
