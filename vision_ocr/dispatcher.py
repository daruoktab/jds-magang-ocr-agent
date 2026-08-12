"""
Agent utama (dispatcher): klasifikasi jenis dokumen -> routing ke agent ekstraksi.

Alur:
1. Gambar dikirim ke VLM untuk KLASIFIKASI JENIS DOKUMEN (memilih dari
   daftar agent yang tersedia di AGENT_REGISTRY).
2. Berdasarkan hasil klasifikasi, agent yang sesuai dipilih dari registry.
3. Agent terpilih dieksekusi (menggunakan modelnya sendiri jika di-set,
   selain itu model default). Hasil divalidasi Pydantic.
"""
import json
import re
from typing import Optional, Tuple

from pydantic import BaseModel

from .agents import AGENT_REGISTRY, ExtractionAgent, get_agent
from .llm import LLMBackend, create_backend
from .prompts import SYSTEM_EXTRACTOR

CLASSIFY_SYSTEM = (
    "Kamu adalah pengklasifikasi jenis dokumen yang teliti. Kamu melihat "
    "sebuah gambar dokumen, memutuskan jenisnya, dan mengembalikan satu "
    "objek JSON berisi nama jenis tersebut. Tidak ada teks lain."
)

CLASSIFY_PROMPT = """Analisis gambar dokumen berikut dan klasifikasikan jenisnya.

Pilih SATU jenis dari daftar berikut - gunakan persis nama yang tertera:
{types}

Output HANYA JSON:
{{"doc_type": "<nama jenis>"}}
"""


class DocumentDispatcher:
    """Agent utama: klasifikasi jenis + routing ke agent ekstraksi."""

    def __init__(
        self,
        backend: Optional[LLMBackend] = None,
        backend_name: str = "ollama",
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 300,
    ) -> None:
        if backend is None:
            backend = create_backend(
                backend=backend_name,
                model=model,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
            )
        self.backend = backend
        self.backend_name = backend_name
        self.default_model = model or getattr(backend, "model", None)
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout

    # --- API publik ----------------------------------------------------
    def dispatch(self, image_path: str) -> Tuple[str, BaseModel]:
        """
        Klasifikasikan gambar, pilih agent, lalu ekstrak.

        Returns:
            (nama_jenis, hasil_ekstraksi) - hasil berupa model Pydantic.
        """
        doc_type = self._classify(image_path)
        agent = get_agent(doc_type)
        result = agent.run(image_path, self._backend_for(agent))
        return agent.name, result

    def classify(self, image_path: str) -> str:
        """Hanya klasifikasi jenis dokumen (tanpa ekstraksi)."""
        return self._classify(image_path)

    # --- internal ------------------------------------------------------
    def _classify(self, image_path: str) -> str:
        types_list = "\n".join(
            f"- {a.name}: {a.description}" for a in AGENT_REGISTRY.values()
        )
        prompt = CLASSIFY_PROMPT.format(types=types_list)

        raw = self.backend.generate(
            prompt=prompt,
            image_path=image_path,
            system=CLASSIFY_SYSTEM,
            json_mode=True,
        )

        doc_type = self._parse_doc_type(raw)
        return doc_type if doc_type in AGENT_REGISTRY else "generic"

    @staticmethod
    def _parse_doc_type(raw: str) -> str:
        """Ambil nama jenis dari respons model (JSON atau teks bebas)."""
        try:
            return str(json.loads(raw).get("doc_type", "")).strip().lower()
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fallback: cari nama jenis yang muncul di teks respons
        lowered = raw.lower()
        for name in AGENT_REGISTRY:
            if re.search(rf"\b{re.escape(name)}\b", lowered):
                return name
        return "generic"

    def _backend_for(self, agent: ExtractionAgent) -> LLMBackend:
        """Backend untuk agent: model khusus agent jika di-set, selain itu default."""
        if agent.model and self.default_model and agent.model != self.default_model:
            return create_backend(
                backend=self.backend_name,
                model=agent.model,
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self.backend
