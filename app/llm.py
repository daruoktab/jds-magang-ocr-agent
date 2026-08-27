"""
Pembangun model chat (`ChatOpenAI`) untuk endpoint OpenAI-compatible.

Menyediakan:
  - build_chat_model(base_url, model, api_key, ...) : builder generik
  - build_vlm(settings)  : VLM normal (ekstraksi bebas + agent, structured output)
  - build_ocr(settings)  : OCR (VLM kecil, output teks terstruktur)

Endpoint dapat berupa LM Studio lokal, `llama-server`, atau server remote -
cukup ubah `.env`.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI

from .config import Settings

logger = logging.getLogger("app.llm")


class LoggingCallbackHandler(BaseCallbackHandler):
    """Callback handler transparan untuk mencatat setiap request & response LLM/VLM."""

    def __init__(self, model_name: str, base_url: str) -> None:
        self.model_name = model_name
        self.base_url = base_url
        self._start_time: float = 0.0

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> None:
        self._start_time = time.perf_counter()
        prompt_len = sum(len(p) for p in prompts)
        logger.info(
            "--> [LLM Request] Mengirim ke %s | Model: %s | Prompt: %d karakter (%d batch)",
            self.base_url,
            self.model_name,
            prompt_len,
            len(prompts),
        )

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        elapsed = time.perf_counter() - self._start_time
        gen_count = len(response.generations)
        total_tokens = ""
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            total_tokens = f" | Tokens: {usage.get('total_tokens', 'N/A')}"

        text_preview = ""
        if response.generations and response.generations[0]:
            first_gen = response.generations[0][0].text
            preview = first_gen[:80].replace("\n", " ").strip()
            text_preview = (
                f" | Preview: '{preview}...'"
                if len(first_gen) > 80
                else f" | Res: '{preview}'"
            )

        logger.info(
            "<-- [LLM Response] Selesai dalam %.2fs | Generasi: %d%s%s",
            elapsed,
            gen_count,
            total_tokens,
            text_preview,
        )

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        elapsed = time.perf_counter() - self._start_time if self._start_time else 0.0
        logger.error(
            "<xx [LLM Error] Gagal setelah %.2fs pada %s (Model: %s): %s",
            elapsed,
            self.base_url,
            self.model_name,
            error,
        )


def _read_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _guess_mime(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
    }.get(ext, "image/png")


def image_data_uri(image_path: str) -> str:
    """Baca gambar menjadi data URI untuk content block `image_url`."""
    return f"data:{_guess_mime(image_path)};base64,{_read_image_base64(image_path)}"


def build_chat_model(
    base_url: str,
    model: str,
    api_key: str,
    temperature: float,
    timeout: float,
    enable_thinking: bool | None = None,
    **kwargs,
) -> ChatOpenAI:
    """ChatOpenAI generik ke endpoint OpenAI-compatible dengan logging transparan.

    Args:
        enable_thinking: bila tidak None, kirim
            `chat_template_kwargs: {enable_thinking: <value>}` di body request
            (dipakai model qwen family agar tidak membuang token thinking).
            None = jangan kirim parameter ini sama sekali.
    """
    params = dict(kwargs)
    if enable_thinking is not None:
        extra_body = dict(params.pop("extra_body", None) or {})
        extra_body["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}
        params["extra_body"] = extra_body

    callbacks = list(params.pop("callbacks", None) or [])
    callbacks.append(LoggingCallbackHandler(model_name=model, base_url=base_url))

    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        timeout=timeout,
        callbacks=callbacks,
        **params,
    )


def build_vlm(settings: Settings) -> ChatOpenAI:
    """VLM normal (ekstraksi + agent), dengan enable_thinking dari config."""
    return build_chat_model(
        base_url=settings.vlm_base_url,
        model=settings.vlm_model,
        api_key=settings.vlm_api_key,
        temperature=settings.vlm_temperature,
        timeout=settings.vlm_timeout,
        enable_thinking=settings.vlm_enable_thinking,
    )


def build_ocr(settings: Settings) -> ChatOpenAI:
    """OCR (VLM kecil, output teks terstruktur)."""
    return build_chat_model(
        base_url=settings.ocr_base_url,
        model=settings.ocr_model,
        api_key=settings.ocr_api_key,
        temperature=settings.ocr_temperature,
        timeout=settings.ocr_timeout,
        max_tokens=settings.ocr_max_tokens,
    )
