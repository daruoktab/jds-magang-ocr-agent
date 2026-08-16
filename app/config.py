"""
Konfigurasi terpusat untuk vision RAG agent.

Endpoint model diganti lewat environment variable atau file `.env` di root.

Ada TIGA kategori model, masing-masing independen:
  1. VLM NORMAL   : ekstraksi dokumen bebas + agent (reasoning / structured output)
  2. OCR          : VLM kecil yang di-tuning untuk OCR (output teks terstruktur)
  3. EMBEDDING    : multimodal (teks + gambar) untuk retrieval

Pola env var (setiap kategori punya var sendiri, fallback ke nilai global):
  # Global (fallback untuk semua kategori)
  LLM_BASE_URL        default http://localhost:1234/v1
  LLM_API_KEY         default lm-studio

  # 1. VLM normal
  VLM_MODEL           default qwen-35b
  VLM_BASE_URL        (fallback LLM_BASE_URL)
  VLM_API_KEY         (fallback LLM_API_KEY)
  VLM_TEMPERATURE     default 0.1
  VLM_TIMEOUT         default 300

  # 2. OCR
  OCR_MODEL           default ocr-lighton
  OCR_BASE_URL        (fallback LLM_BASE_URL)
  OCR_API_KEY         (fallback LLM_API_KEY)
  OCR_TEMPERATURE     default 0.0
  OCR_TIMEOUT         default 300
  OCR_MAX_TOKENS      default 500

  # 3. Embedding
  EMBEDDING_MODE      subprocess | http | transformers  (default subprocess)
  EMBEDDING_MODEL     default Qwen3-VL-Embedding-2B-f16.gguf
  EMBEDDING_MMPROJ    path mmproj (wajib utk gambar)
  EMBEDDING_BASE_URL  default http://localhost:8080/v1  (hanya mode http)
  EMBEDDING_API_KEY   default ""                          (hanya mode http)
  LLAMA_VL_EMBEDDING_BIN  path binary (kosong = cari di PATH)
  EMBEDDING_HF_MODEL  path/nama model safetensors (hanya mode transformers)
  EMBEDDING_DEVICE    auto | cuda | cpu (hanya mode transformers)
  EMBEDDING_POOLING / EMBEDDING_NORMALIZE / EMBEDDING_CONTEXT / EMBEDDING_NGL
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Loader `.env` minimal (tidak menimpa env yang sudah ada).

    Komentar inline ("# ..." setelah nilai) ikut dibuang agar aman kalau
    key ditempel di baris yang sama dengan komentar.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        # Buang komentar inline: "#" yang didahului spasi (mis. "sk-abc # isi di sini").
        value = re.split(r"\s+#", value, maxsplit=1)[0]
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(_PROJECT_ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _env_or(name: str, fallback_name: str, default: str = "") -> str:
    """Ambil `name`; jika kosong, ambil `fallback_name`; else `default`."""
    val = os.environ.get(name)
    if val:
        return val
    return os.environ.get(fallback_name, default)


def _float_env(name: str, default: str) -> float:
    return float(_env(name, default))


def _int_env(name: str, default: str) -> int:
    return int(_env(name, default))


def _bool_env(name: str, default: str) -> bool:
    """Parse env boolean: '1'/'true'/'yes'/'on' -> True, lainnya False."""
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    # --- Global (fallback bersama) ---
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "http://localhost:1234/v1"))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", "lm-studio"))

    # --- 1. VLM normal (ekstraksi bebas + agent) ---
    vlm_model: str = field(default_factory=lambda: _env("VLM_MODEL", "qwen-35b-vision"))
    vlm_base_url: str = field(default_factory=lambda: _env_or("VLM_BASE_URL", "LLM_BASE_URL", "http://localhost:1234/v1"))
    vlm_api_key: str = field(default_factory=lambda: _env_or("VLM_API_KEY", "LLM_API_KEY", "lm-studio"))
    vlm_temperature: float = field(default_factory=lambda: _float_env("VLM_TEMPERATURE", "0.1"))
    vlm_timeout: float = field(default_factory=lambda: _float_env("VLM_TIMEOUT", "300"))
    # qwen-35b-vision: kirim chat_template_kwargs enable_thinking=false agar
    # tidak membuang token untuk thinking. OCR TIDAK perlu ini (lihat catatan).
    vlm_enable_thinking: bool = field(default_factory=lambda: _bool_env("VLM_ENABLE_THINKING", "false"))

    # --- 2. OCR (VLM kecil, tuned, output teks terstruktur) ---
    ocr_model: str = field(default_factory=lambda: _env("OCR_MODEL", "ocr-lighton"))
    ocr_base_url: str = field(default_factory=lambda: _env_or("OCR_BASE_URL", "LLM_BASE_URL", "http://localhost:1234/v1"))
    ocr_api_key: str = field(default_factory=lambda: _env_or("OCR_API_KEY", "LLM_API_KEY", "lm-studio"))
    ocr_temperature: float = field(default_factory=lambda: _float_env("OCR_TEMPERATURE", "0.0"))
    ocr_timeout: float = field(default_factory=lambda: _float_env("OCR_TIMEOUT", "300"))
    ocr_max_tokens: int = field(default_factory=lambda: _int_env("OCR_MAX_TOKENS", "500"))

    # --- 3. Embedding (multimodal) ---
    embedding_enabled: bool = field(default_factory=lambda: _bool_env("EMBEDDING_ENABLED", "true"))
    embedding_mode: str = field(default_factory=lambda: _env("EMBEDDING_MODE", "subprocess"))
    embedding_base_url: str = field(default_factory=lambda: _env("EMBEDDING_BASE_URL", "http://localhost:8080/v1"))
    embedding_api_key: str = field(default_factory=lambda: _env("EMBEDDING_API_KEY", ""))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "Qwen3-VL-Embedding-2B-f16.gguf"))
    embedding_mmproj: str | None = field(default_factory=lambda: os.environ.get("EMBEDDING_MMPROJ") or None)
    embedder_binary: str | None = field(default_factory=lambda: os.environ.get("LLAMA_VL_EMBEDDING_BIN") or None)
    embedding_hf_model: str | None = field(default_factory=lambda: os.environ.get("EMBEDDING_HF_MODEL") or None)
    embedding_device: str = field(default_factory=lambda: _env("EMBEDDING_DEVICE", "auto"))
    embedding_pooling: str = field(default_factory=lambda: _env("EMBEDDING_POOLING", "last"))
    embedding_normalize: int = field(default_factory=lambda: _int_env("EMBEDDING_NORMALIZE", "2"))
    embedding_context: int = field(default_factory=lambda: _int_env("EMBEDDING_CONTEXT", "4096"))
    embedding_ngl: str = field(default_factory=lambda: _env("EMBEDDING_NGL", "auto"))


_settings: Settings | None = None


def get_settings() -> Settings:
    """Kembalikan singleton Settings (malas, agar env bisa di-set sebelum dipakai)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
