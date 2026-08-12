"""Konfigurasi default modul vision_ocr.

Semua nilai bisa di-override via parameter saat instantiate (lihat cli.py).
"""

# Backend default: Ollama lokal (endpoint native /api/chat).
DEFAULT_BACKEND = "ollama"
DEFAULT_MODEL = "qwen3.5:4b"          # VLM (vision + tools + thinking)
DEFAULT_BASE_URL = "http://localhost:11434"

# Alternatif OpenAI-compatible (mis. untuk OpenAI / DeepSeek / vLLM / Ollama /v1)
# DEFAULT_BACKEND = "openai_compat"
# DEFAULT_MODEL = "gpt-4o"
# DEFAULT_BASE_URL = "https://api.openai.com/v1"
# DEFAULT_API_KEY = "sk-..."

# Timeout request LLM (detik) - loading model pertama kali bisa lambat.
DEFAULT_TIMEOUT = 300
