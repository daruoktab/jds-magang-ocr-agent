"""
Pipeline ekstraksi: gambar -> VLM -> structured output (Pydantic).

`VisionExtractor` memakai `ChatOpenAI.with_structured_output()` agar respons
model langsung divalidasi menjadi instance Pydantic - pola idiomatik LangChain
(sekaligus menggantikan parsing JSON manual pada implementasi lama).
"""
from __future__ import annotations

from typing import Optional, Type

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from .llm import image_data_uri
from .prompts import GENERAL_USER_PROMPT, SYSTEM_EXTRACTOR
from .schemas import DocumentExtraction


class VisionExtractor:
    """Ekstraktor generik: satu model + satu schema + satu prompt."""

    def __init__(
        self,
        llm: BaseChatModel,
        schema: Type[BaseModel] = DocumentExtraction,
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

    def extract(self, image_path: str) -> BaseModel:
        """
        Ekstrak informasi dari gambar menjadi instance Pydantic.

        Raises:
            pydantic.ValidationError: jika output model gagal divalidasi.
        """
        content: list[dict] = [
            {"type": "text", "text": self.user_prompt},
            {"type": "image_url", "image_url": {"url": image_data_uri(image_path)}},
        ]

        messages: list = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        messages.append(HumanMessage(content=content))

        return self._structured.invoke(messages)
