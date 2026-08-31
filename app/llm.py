"""
Pembangun model chat (`ChatOpenAI`) untuk endpoint OpenAI-compatible.

Menyediakan:
  - build_chat_model(base_url, model, api_key, ...) : builder generik
  - build_vlm(settings)  : VLM normal (ekstraksi bebas + agent, structured output)
  - get_vlm(settings)    : Helper singleton / factory untuk VLM
  - encode_image, encode_image_to_base64, image_data_uri : utility encoding citra

Endpoint dapat berupa LM Studio lokal, `llama-server`, atau server remote -
cukup ubah `.env`.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI

from .config import Settings, get_settings

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
        logger.info(
            "--> [LLM Request] Model: %s | URL: %s | Prompts: %d item",
            self.model_name,
            self.base_url,
            len(prompts),
        )
        for i, p in enumerate(prompts):
            preview = p[:120].replace("\n", " ")
            logger.debug("    Prompt #%d: %s...", i + 1, preview)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        elapsed = time.perf_counter() - self._start_time
        gen_count = sum(len(g) for g in response.generations)
        logger.info(
            "<-- [LLM Response] Model: %s | Waktu: %.2fs | Generasi: %d item",
            self.model_name,
            elapsed,
            gen_count,
        )


def encode_image(image_path: str | Path) -> str:
    """Baca file gambar dan encode ke base64 string."""
    with open(str(image_path), "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_image_to_base64(image_path: str | Path) -> tuple[str, str]:
    """
    Encode file gambar ke base64 dan tentukan tipe mime.
    Returns:
        (b64_string, mime_type) contoh: ("abc...", "image/png")
    """
    path_str = str(image_path)
    ext = os.path.splitext(path_str)[1].lower().lstrip(".")
    mime_sub = "jpeg" if ext in ("jpg", "jpeg") else ext
    mime = f"image/{mime_sub}" if mime_sub else "image/png"
    b64 = encode_image(path_str)
    return b64, mime


def image_data_uri(image_path: str | Path) -> str:
    """Format file gambar menjadi data URI (data:image/...;base64,...)."""
    b64, mime = encode_image_to_base64(image_path)
    return f"data:{mime};base64,{b64}"


def build_chat_model(
    *,
    base_url: str,
    model: str,
    api_key: str,
    temperature: float = 0.1,
    timeout: float = 300,
    enable_thinking: bool | None = None,
    callbacks: list[Any] | None = None,
    **extra_kwargs: Any,
) -> ChatOpenAI:
    """
    Buat instance `ChatOpenAI` generik untuk endpoint OpenAI-compatible.
    """
    params: dict[str, Any] = dict(extra_kwargs)

    # Dukungan model reasoning (mis. Qwen 2.5 / DeepSeek-R1 / Qwen3)
    if enable_thinking is not None:
        params["extra_body"] = {"enable_thinking": enable_thinking}

    return ChatOpenAI(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
        timeout=timeout,
        callbacks=callbacks,
        **params,
    )


def build_vlm(settings: Settings | None = None) -> ChatOpenAI:
    """VLM normal (ekstraksi + agent), dengan enable_thinking dari config."""
    resolved = settings or get_settings()
    return build_chat_model(
        base_url=resolved.vlm_base_url,
        model=resolved.vlm_model,
        api_key=resolved.vlm_api_key,
        temperature=resolved.vlm_temperature,
        timeout=resolved.vlm_timeout,
        enable_thinking=resolved.vlm_enable_thinking,
    )


def get_vlm(settings: Settings | None = None) -> ChatOpenAI:
    """Alias/Helper untuk mendapatkan instance VLM."""
    return build_vlm(settings)
