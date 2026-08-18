"""
Pipeline ekstraksi: gambar -> VLM -> structured output (Pydantic).

`VisionExtractor` memakai `ChatOpenAI.with_structured_output()` agar respons
model langsung divalidasi menjadi instance Pydantic - pola idiomatik LangChain
(sekaligus menggantikan parsing JSON manual pada implementasi lama).
"""
from __future__ import annotations

from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from .llm import image_data_uri
from .prompts import GENERAL_USER_PROMPT, SYSTEM_EXTRACTOR, build_extraction_user_prompt
from .schemas import DocumentExtraction


class VisionExtractor:
    """Ekstraktor generik: satu model + satu schema + satu prompt (dengan opsi OCR fusion & critique)."""

    def __init__(
        self,
        llm: BaseChatModel,
        schema: type[BaseModel] = DocumentExtraction,
        user_prompt: str = GENERAL_USER_PROMPT,
        system_prompt: str = SYSTEM_EXTRACTOR,
    ) -> None:
        self.llm = llm
        self.schema = schema
        self.user_prompt = user_prompt
        self.system_prompt = system_prompt
        # Model lokal harus mendukung tool/function calling atau JSON schema
        # agar structured output bekerja (qwen3.5 mendukung tool calling).
        self._structured = llm.with_structured_output(schema)

    def extract(
        self,
        image_path: str,
        *,
        ocr_text: str | None = None,
        critique: str | None = None,
    ) -> BaseModel:
        """
        Ekstrak informasi dari gambar menjadi instance Pydantic.

        Args:
            image_path: Path ke file gambar dokumen.
            ocr_text: Teks OCR opsional untuk membantu referensi VLM (fusion).
            critique: Catatan validasi / kejanggalan sebelumnya untuk refleksi mandiri.

        Raises:
            pydantic.ValidationError: jika output model gagal divalidasi.
        """
        effective_user_prompt = build_extraction_user_prompt(
            self.user_prompt, ocr_text=ocr_text, critique=critique
        )

        content: list[dict] = [
            {"type": "text", "text": effective_user_prompt},
            {"type": "image_url", "image_url": {"url": image_data_uri(image_path)}},
        ]

        messages: list = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        messages.append(HumanMessage(content=cast(Any, content)))

        return cast(BaseModel, self._structured.invoke(messages))

