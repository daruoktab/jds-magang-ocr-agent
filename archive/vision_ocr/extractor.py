"""
Pipeline ekstraksi: gambar -> VLM -> JSON -> validasi Pydantic.

VisionExtractor menggabungkan satu backend LLM + satu schema Pydantic +
satu template prompt menjadi unit yang siap pakai. Semua bagian modular:
backend, schema, dan prompt bisa diganti sesuai kebutuhan.
"""
import json
import re
from typing import Optional, Type, Union

from pydantic import BaseModel, ValidationError

from .llm import LLMBackend, create_backend
from .prompts import GENERAL_USER_PROMPT, SYSTEM_EXTRACTOR
from .schemas import DocumentExtraction

# Model pydantic yang bisa dihasilkan extractor
AnySchema = Type[BaseModel]


class VisionExtractor:
    """Ekstraktor generik: satu backend + satu schema + satu prompt."""

    def __init__(
        self,
        backend: LLMBackend,
        schema: AnySchema = DocumentExtraction,
        user_prompt: str = GENERAL_USER_PROMPT,
        system_prompt: str = SYSTEM_EXTRACTOR,
        json_mode: bool = True,
    ) -> None:
        self.backend = backend
        self.schema = schema
        self.user_prompt = user_prompt
        self.system_prompt = system_prompt
        self.json_mode = json_mode

    def extract(self, image_path: str) -> BaseModel:
        """
        Ekstrak informasi dari gambar.

        Raises:
            ValueError: jika respons model tidak bisa di-parse/divalidasi
                menjadi schema yang dituju (setelah mencoba fallback).
        """
        raw = self.backend.generate(
            prompt=self.user_prompt,
            image_path=image_path,
            system=self.system_prompt,
            json_mode=self.json_mode,
        )
        return self._parse(raw)

    # --- parsing robust -------------------------------------------------
    def _parse(self, raw: str) -> BaseModel:
        candidates = self._extract_json_candidates(raw)
        last_error: Optional[Exception] = None
        for candidate in candidates:
            try:
                return self.schema.model_validate_json(candidate)
            except (ValidationError, json.JSONDecodeError) as e:
                last_error = e
            # Model kecil sering mengabaikan kerangka {doc_type, data} dan
            # mengeluarkan struktur flat. Normalisasi jika schema default.
            normalized = self._try_normalize(candidate)
            if normalized is not None:
                try:
                    return self.schema.model_validate(normalized)
                except (ValidationError, json.JSONDecodeError) as e:
                    last_error = e
        raise ValueError(
            f"Gagal memvalidasi respons model menjadi {self.schema.__name__}. "
            f"Error terakhir: {last_error}\n"
            f"Respons model (awal): {raw[:500]}"
        )

    def _try_normalize(self, candidate: str) -> Optional[dict]:
        """Ubah struktur flat (doc_type + field) menjadi {doc_type, data}."""
        if self.schema is not DocumentExtraction:
            return None
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict):
            return None
        if isinstance(obj.get("data"), dict):
            return obj
        if "doc_type" in obj:
            return {
                "doc_type": obj["doc_type"],
                "data": {k: v for k, v in obj.items() if k != "doc_type"},
            }
        return {"doc_type": "generic", "data": obj}

    @staticmethod
    def _extract_json_candidates(raw: str) -> list[str]:
        """Kumpulkan kandidat JSON: raw utuh, blok ```json ... ```, objek {..}."""
        candidates: list[str] = []

        stripped = raw.strip()
        if stripped:
            candidates.append(stripped)

        # Blok markdown ```json ... ```
        for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", stripped):
            candidates.append(match.group(1).strip())

        # Objek JSON (greedy, mencakup nested object)
        for match in re.finditer(r"\{[\s\S]*\}", stripped):
            candidates.append(match.group(0))

        # Deduplikasi, pertahankan urutan
        seen: set[str] = set()
        unique: list[str] = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        return unique


def create_extractor(
    backend: Optional[LLMBackend] = None,
    backend_name: str = "ollama",
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 300,
    schema: AnySchema = DocumentExtraction,
    user_prompt: str = GENERAL_USER_PROMPT,
) -> VisionExtractor:
    """
    Factory utama: rakit extractor dari backend + schema + prompt.

    Default: backend Ollama + schema DocumentExtraction + prompt general
    (model bebas menentukan struktur hasil). Semua bisa di-override.

    Contoh:
        ex = create_extractor(model="Orsta-7B")
        data = ex.extract("struk.png")   # -> DocumentExtraction
    """
    if backend is None:
        backend = create_backend(
            backend=backend_name,
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )

    return VisionExtractor(
        backend=backend,
        schema=schema,
        user_prompt=user_prompt,
    )
