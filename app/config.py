"""
Konfigurasi terpusat untuk Document Vision OCR & Extraction (Ready for Chunking).

Endpoint model dapat dikonfigurasi melalui environment variable atau file `.env` di root.

Kategori model yang digunakan:
  1. VLM NORMAL : Ekstraksi dokumen & layout reasoning multimodal ke Markdown
  2. OCR        : VLM kecil yang di-tuning khusus untuk grounding teks beresolusi tinggi
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Loader `.env` minimal tanpa dependensi eksternal."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        # Buang komentar inline: "#" yang didahului spasi (mis. "sk-abc # isi di sini")
        value = re.split(r"\s+#", value, maxsplit=1)[0]
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(_PROJECT_ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    """Ambil string dari environment variable."""
    return os.environ.get(name, default)


def _env_or(name: str, fallback_name: str, default: str = "") -> str:
    """Ambil `name`; jika kosong, ambil `fallback_name`; else `default`."""
    val = os.environ.get(name)
    if val:
        return val
    return os.environ.get(fallback_name, default)


def _float_env(name: str, default: str) -> float:
    """Parse float dari environment variable."""
    return float(_env(name, default))


def _int_env(name: str, default: str) -> int:
    """Parse integer dari environment variable."""
    return int(_env(name, default))


def _bool_env(name: str, default: str) -> bool:
    """Parse boolean dari environment variable ('1'/'true'/'yes'/'on' -> True)."""
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    """Pengaturan konfigurasi LLM, VLM, dan OCR."""

    # --- Global Fallback ---
    llm_base_url: str = field(
        default_factory=lambda: _env("LLM_BASE_URL", "http://localhost:1234/v1")
    )
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", "lm-studio"))

    # --- 1. VLM Normal (Ekstraksi Markdown + Layout Classifier) ---
    vlm_model: str = field(default_factory=lambda: _env("VLM_MODEL", "qwen-35b-vision"))
    vlm_base_url: str = field(
        default_factory=lambda: _env_or(
            "VLM_BASE_URL", "LLM_BASE_URL", "http://localhost:1234/v1"
        )
    )
    vlm_api_key: str = field(
        default_factory=lambda: _env_or("VLM_API_KEY", "LLM_API_KEY", "lm-studio")
    )
    vlm_temperature: float = field(
        default_factory=lambda: _float_env("VLM_TEMPERATURE", "0.1")
    )
    vlm_timeout: float = field(default_factory=lambda: _float_env("VLM_TIMEOUT", "300"))
    vlm_enable_thinking: bool = field(
        default_factory=lambda: _bool_env("VLM_ENABLE_THINKING", "false")
    )

    # --- 2. OCR (Grounding Teks Resolusi Tinggi) ---
    ocr_model: str = field(default_factory=lambda: _env("OCR_MODEL", "ocr-lighton"))
    ocr_base_url: str = field(
        default_factory=lambda: _env_or(
            "OCR_BASE_URL", "LLM_BASE_URL", "http://localhost:1234/v1"
        )
    )
    ocr_api_key: str = field(
        default_factory=lambda: _env_or("OCR_API_KEY", "LLM_API_KEY", "lm-studio")
    )
    ocr_temperature: float = field(
        default_factory=lambda: _float_env("OCR_TEMPERATURE", "0.0")
    )
    ocr_timeout: float = field(default_factory=lambda: _float_env("OCR_TIMEOUT", "300"))
    ocr_max_tokens: int = field(
        default_factory=lambda: _int_env("OCR_MAX_TOKENS", "500")
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    """Kembalikan singleton Settings (diinisialisasi secara lazy saat pertama kali diakses)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
