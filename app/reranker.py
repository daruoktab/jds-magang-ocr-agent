"""
Reranker multimodal (Qwen3-VL-Reranker) berbasis transformers.

Implementasi mengikuti kode resmi QwenLM/Qwen3-VL-Embedding
(`src/models/qwen3_vl_reranker.py`):
  - Template: system "judge yes/no" + user `<Instruct>/<Query>/<Document>`
  - Skor    : binary head `Linear(D, 1)` dengan bobot `(lm_head["yes"] - lm_head["no"])`
              diterapkan pada `last_hidden_state[:, -1]` -> sigmoid.

Mode backend:
  - `transformers` (default): safetensors via transformers (GPU/CPU). Sudah teruji.
  - llama.cpp (subprocess)   : BELUM tersedia - biarkan placeholder untuk pengembangan
    berikutnya (ubah `_score_pair_impl` saat binary siap).

Penggunaan dua-stage (rekomendasi resmi Qwen):
  embedding (recall murah) -> reranker (refine presisi lintas-modalitas).
"""
from __future__ import annotations

import os
from typing import Any, cast

import torch

from .config import Settings, get_settings

RERANKER_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query and the Instruct "
    "provided. Note that the answer can only be \"yes\" or \"no\"."
)
MIN_PIXELS = 4 * 32 * 32        # 4096
MAX_PIXELS = 1800 * 32 * 32     # 1,843,200


class Qwen3VLReranker:
    """Reranker multimodal Qwen3-VL-Reranker (transformers, GPU/CPU)."""

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "auto",
        max_length: int = 8192,
        instruction: str = "Given a search query, retrieve relevant candidates that answer the query.",
        dtype: Any = torch.bfloat16,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.instruction = instruction
        self._device = (
            "cuda"
            if device == "auto" and torch.cuda.is_available()
            else ("cpu" if device == "auto" else device)
        )
        self._lm: Any = None      # Qwen3VLForConditionalGeneration
        self._model: Any = None   # .model bagian dalam (return last_hidden_state)
        self._processor: Any = None
        self._score_linear: Any = None
        self._dtype = dtype

    # --- Lazy loading: muat model saat pertama kali dipakai -------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            AutoTokenizer,
        )

        model_path: str = self.model_name_or_path
        lm = cast(Any, AutoModelForImageTextToText).from_pretrained(
            model_path, dtype=self._dtype
        ).to(self._device).eval()
        processor = AutoProcessor.from_pretrained(self.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        assert tokenizer is not None

        # Binary classification head resmi: Linear(D,1) bobot (yes - no) dari lm_head
        vocab = tokenizer.get_vocab()
        w_yes = lm.lm_head.weight.data[vocab["yes"]]
        w_no = lm.lm_head.weight.data[vocab["no"]]
        D = w_yes.size(0)
        linear = torch.nn.Linear(D, 1, bias=False).to(self._device)
        with torch.no_grad():
            linear.weight[0] = w_yes - w_no

        self._lm = lm
        self._model = lm.model
        self._processor = processor
        self._score_linear = linear.to(lm.dtype)
        print(f"[reranker] model siap di {self._device}")

    # --- API publik ------------------------------------------------------
    def score(self, query: str, document_images: list[Any]) -> float:
        """Skor relevansi (query, dokumen-gambar) -> [0,1]."""
        return self.score_many(query, [document_images])[0]

    def score_many(self, query: str, documents: list[list[Any]]) -> list[float]:
        """Skor untuk banyak dokumen (tiap dokumen = list gambar halaman)."""
        self._ensure_loaded()
        scores = []
        for pages in documents:
            scores.append(self._score_pair(query, pages))
        return scores

    def process(self, payload: dict[str, Any]) -> list[float]:
        """API kompatibel pola resmi: {instruction, query:{text?,image?}, documents:[{image?}...]}."""
        instruction = payload.get("instruction") or self.instruction
        query = payload.get("query", {})
        q_text = query.get("text", "")
        scores = []
        for doc in payload.get("documents", []):
            images = doc.get("image", [])
            images = [images] if isinstance(images, str) else images
            scores.append(self._score_pair(q_text, images, instruction=instruction))
        return scores

    # --- inti scoring ----------------------------------------------------
    def _score_pair(
        self, query: str, pages: list[Any], instruction: str | None = None
    ) -> float:
        from PIL import Image  # noqa: F401

        content: list[dict[str, Any]] = [
            {"type": "text", "text": f"<Instruct>: {instruction or self.instruction}"},
            {"type": "text", "text": f"<Query>: {query}"},
            {"type": "text", "text": "\n<Document>: "},
        ]
        content += [
            {"type": "image", "image": p, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS}
            for p in pages
        ]
        conv = [{"role": "system", "content": [{"type": "text", "text": RERANKER_SYSTEM}]},
                {"role": "user", "content": content}]

        text = self._processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
        inputs = self._processor(text=[text], images=pages or None, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            hidden = self._model(**inputs).last_hidden_state[:, -1]
        return float(torch.sigmoid(self._score_linear(hidden)).item())


def build_reranker(settings: Settings | None = None) -> Qwen3VLReranker | None:
    """Bangun reranker dari settings. None jika dimatikan / model belum di-set."""
    settings = settings or get_settings()
    if not settings.reranker_enabled:
        return None
    if not settings.reranker_model:
        print("[warn] RERANKER_ENABLED=true tapi RERANKER_MODEL kosong - reranker di-skip")
        return None
    if not os.path.exists(settings.reranker_model):
        print(f"[warn] Model reranker tidak ditemukan: {settings.reranker_model} - di-skip")
        return None
    return Qwen3VLReranker(
        model_name_or_path=settings.reranker_model,
        device=settings.reranker_device,
        max_length=settings.reranker_max_length,
        instruction=settings.reranker_instruction,
    )
