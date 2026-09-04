"""
Konfigurasi terpusat untuk Document Vision VLM & Extraction (Ready for Chunking).

Menggunakan Pydantic-like dataclass `Settings` yang membaca environment variables
dengan fallback yang aman untuk local inference (LM Studio / Ollama / llama-server).

Peran Model:
  1. VLM (Vision LLM): Model multimodal utama untuk interpretasi visual dan ekstraksi Markdown
  2. Logging Config  : Konfigurasi level logging
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DPI: int = 200
PDF_PAGE_BATCH: int = 10
SUPPORTED_IMAGE_EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg", ".webp"}


def _env(name: str, default: str) -> str:
    """Ambil env var atau kembalikan default jika kosong."""
    val = os.environ.get(name, "").strip()
    return val if val else default


def _env_or(primary: str, fallback: str, default: str) -> str:
    """Ambil `primary` env var; jika kosong coba `fallback`; jika kosong pakai `default`."""
    val = os.environ.get(primary, "").strip()
    if val:
        return val
    val = os.environ.get(fallback, "").strip()
    return val if val else default


def _float_env(name: str, default: str) -> float:
    """Parse float dari environment variable dengan fallback."""
    raw = os.environ.get(name, default).strip()
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _int_env(name: str, default: str) -> int:
    """Parse int dari environment variable dengan fallback."""
    raw = os.environ.get(name, default).strip()
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _bool_env(name: str, default: str) -> bool:
    """Parse boolean dari environment variable ('1'/'true'/'yes'/'on' -> True)."""
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    """Pengaturan konfigurasi LLM, VLM, dan logging."""

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

    # --- 2. Logging Configuration ---
    log_level: str = field(
        default_factory=lambda: _env("LOG_LEVEL", "INFO").strip().upper() or "INFO"
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    """Singleton getter untuk Settings."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def setup_logging(
    level: str | None = None,
    log_file: str | Path | None = None,
    auto_log_stem: str | None = None,
    auto_log_dir: str | Path | None = None,
) -> None:
    """Inisialisasi logging terformat dengan timestamp ke konsol dan opsional ke file.

    Jika `log_file` tidak diberikan tetapi `auto_log_stem` ada, log otomatis
    ditulis real-time ke `{auto_log_dir}/{auto_log_stem}_latest.log` atau
    `output/{auto_log_stem}/logs/{auto_log_stem}_latest.log`.
    """
    effective_level = (level or get_settings().log_level).upper()
    log_format = "%(asctime)s | %(levelname)-7s | [%(name)s] %(message)s"
    date_format = "%H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        p = Path(log_file).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(p), mode="a", encoding="utf-8"))
    elif auto_log_stem:
        if auto_log_dir:
            log_dir = Path(auto_log_dir).resolve()
        else:
            log_dir = (Path("output") / auto_log_stem / "logs").resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        p = log_dir / f"{auto_log_stem}_latest.log"
        handlers.append(logging.FileHandler(str(p), mode="w", encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, effective_level, logging.INFO),
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True,
    )

