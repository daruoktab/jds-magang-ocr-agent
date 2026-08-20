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
import os

from langchain_openai import ChatOpenAI

from .config import Settings


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
    """ChatOpenAI generik ke endpoint OpenAI-compatible.

    Args:
        enable_thinking: bila tidak None, kirim
            `chat_template_kwargs: {enable_thinking: <value>}` di body request
            (dipakai model qwen family agar tidak membuang token thinking).
            None = jangan kirim parameter ini sama sekali.

    Catatan: parameter non-OpenAI (seperti chat_template_kwargs) harus lewat
    `extra_body` - bukan `model_kwargs` (yang di-merge ke top-level dan bisa
    memicu error parse di client OpenAI).
    """
    params = dict(kwargs)
    if enable_thinking is not None:
        extra_body = dict(params.pop("extra_body", None) or {})
        extra_body["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}
        params["extra_body"] = extra_body
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        timeout=timeout,
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
