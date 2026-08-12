"""
Abstraksi backend LLM (modular).

Pola: Strategy + Factory.
  - LLMBackend     : protocol (ABC) yang harus dipenuhi semua backend
  - OllamaBackend  : Ollama native API (/api/chat) dengan dukungan gambar base64 + JSON mode
  - OpenAICompatBackend : endpoint OpenAI-compatible (OpenAI, DeepSeek, vLLM, dst.)
  - create_backend : factory untuk memilih backend dari konfigurasi

Untuk menambah backend baru: buat class yang implement LLMBackend, lalu
daftarkan di BACKEND_REGISTRY.
"""
import base64
import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import requests


class LLMBackend(ABC):
    """Kontrak minimal semua backend LLM/VLM."""

    @abstractmethod
    def generate(
        self,
        *,
        prompt: str,
        image_path: Optional[str] = None,
        system: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Kirim prompt (+ gambar opsional), kembalikan teks respons.

        Args:
            prompt: instruksi utama untuk model
            image_path: path ke file gambar (di-encode base64 oleh backend)
            system: system prompt (opsional)
            json_mode: minta model mengembalikan JSON murni
            temperature: suhu sampling
            max_tokens: batas token output
        """
        raise NotImplementedError


def _read_image_base64(image_path: str) -> str:
    """Baca file gambar dan encode base64 (tanpa prefix data URI)."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class OllamaBackend(LLMBackend):
    """Backend Ollama menggunakan native /api/chat (paling andal untuk gambar)."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: float = 300,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        *,
        prompt: str,
        image_path: Optional[str] = None,
        system: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> str:
        messages: list[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})

        user_msg: Dict[str, Any] = {"role": "user", "content": prompt}
        if image_path:
            user_msg["images"] = [_read_image_base64(image_path)]
        messages.append(user_msg)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            # Model reasoning (qwen3.5, deepseek-r1, dll.) bisa menghabiskan
            # seluruh token output untuk thinking sehingga content kosong.
            # think=false memaksa jawaban langsung di content.
            "think": False,
            "options": {
                "temperature": temperature,
            },
        }
        if json_mode:
            # Ollama native: format="json" memaksa output JSON yang valid.
            payload["format"] = "json"
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


class OpenAICompatBackend(LLMBackend):
    """Backend OpenAI-compatible (OpenAI, DeepSeek, vLLM, atau Ollama /v1)."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 300,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.timeout = timeout

    def generate(
        self,
        *,
        prompt: str,
        image_path: Optional[str] = None,
        system: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> str:
        messages: list[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})

        if image_path:
            mime = _guess_mime(image_path)
            data_uri = f"data:{mime};base64,{_read_image_base64(image_path)}"
            user_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]
        else:
            user_content = prompt
        messages.append({"role": "user", "content": user_content})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if max_tokens:
            payload["max_tokens"] = max_tokens

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


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


BACKEND_REGISTRY = {
    "ollama": OllamaBackend,
    "openai_compat": OpenAICompatBackend,
}


def create_backend(
    backend: str = "ollama",
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 300,
) -> LLMBackend:
    """Factory: buat backend dari nama backend + config."""
    cls = BACKEND_REGISTRY.get(backend)
    if cls is None:
        raise ValueError(
            f"Backend tidak dikenal: {backend!r}. Tersedia: {sorted(BACKEND_REGISTRY)}"
        )
    kwargs: Dict[str, Any] = {"timeout": timeout}
    if cls is OllamaBackend:
        kwargs.update(model=model, base_url=base_url or "http://localhost:11434")
    elif cls is OpenAICompatBackend:
        kwargs.update(model=model, base_url=base_url, api_key=api_key)
    return cls(**kwargs)
