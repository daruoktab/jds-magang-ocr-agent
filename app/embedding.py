"""
Vision embedding berbasis Qwen3-VL-Embedding via binary `llama-vl-embedding`.

Dua mode embedding:
  - `subprocess` (default): wrapper subprocess ke binary `llama-vl-embedding`
    (fork llama.cpp, reproduksi embedding teks+gambar patched). Tanpa torch.
  - `http`: klien ke `llama-server` (endpoint OpenAI-compatible `/v1/embeddings`)
    untuk embedding teks. Berguna saat mengetes dari server remote, tetapi TIDAK
    mereproduksi embedding vision patched (yang ada di binary `llama-vl-embedding`).

`LlamaVLEmbeddings` / `LlamaServerEmbeddings` sama-sama mengimplementasikan
antarmuka `Embeddings` LangChain agar bisa dipasang ke vector store.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any, Dict, List, Optional

import numpy as np
import requests
import torch
import torch.nn.functional as F
from langchain_core.embeddings import Embeddings

from .config import Settings, get_settings


class VisionEmbedder:
    """Wrapper subprocess untuk binary `llama-vl-embedding` (fork llama.cpp)."""

    def __init__(
        self,
        binary: Optional[str] = None,
        model_path: Optional[str] = None,
        mmproj_path: Optional[str] = None,
        pooling: str = "last",
        embd_normalize: int = 2,
        context: int = 4096,
        ngl: Optional[Any] = "auto",
        timeout: float = 300,
    ) -> None:
        self.binary = binary or shutil.which("llama-vl-embedding")
        if not self.binary:
            raise FileNotFoundError(
                "Binary 'llama-vl-embedding' tidak ditemukan. Build dulu: "
                "cmake --build llama.cpp/build --target llama-vl-embedding "
                "(atau set LLAMA_VL_EMBEDDING_BIN)."
            )
        if not model_path:
            raise ValueError("model_path (GGUF) wajib diisi")
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.pooling = pooling
        self.embd_normalize = embd_normalize
        self.context = context
        self.ngl = ngl
        self.timeout = timeout

    # --- API publik -------------------------------------------------------
    def embed_text(self, texts: List[str]) -> np.ndarray:
        """Embed daftar teks -> (n, dim)."""
        return self._run([{"text": t} for t in texts])

    def embed_image(self, image_paths: List[str]) -> np.ndarray:
        """Embed daftar path gambar -> (n, dim). Wajib mmproj."""
        if not self.mmproj_path:
            raise ValueError("mmproj_path wajib diisi untuk embedding gambar")
        return self._run([{"image": p} for p in image_paths])

    def embed_mixed(self, items: List[Dict[str, str]]) -> np.ndarray:
        """
        Embed campuran; tiap item dict {text?, image?} (salah satu atau keduanya).

        Contoh:
            embed_mixed([{"text": "A dog on the beach", "image": "./0.jpeg"}])
        """
        return self._run(items)

    # --- internal ---------------------------------------------------------
    def _run(self, items: List[Dict[str, str]]) -> np.ndarray:
        if not items:
            return np.zeros((0, 0), dtype=np.float32)

        cmd = [
            self.binary,
            "-m",
            self.model_path,
            "--inputs",
            json.dumps(items, ensure_ascii=False),
            "--pooling",
            self.pooling,
            "--embd-normalize",
            str(self.embd_normalize),
            "--embd-output-format",
            "array",
            "-c",
            str(self.context),
        ]
        if self.mmproj_path:
            cmd += ["--mmproj", self.mmproj_path]
        if self.ngl is not None:
            cmd += ["-ngl", str(self.ngl)]

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
        if proc.returncode != 0:
            raise RuntimeError(
                f"llama-vl-embedding gagal (rc={proc.returncode}):\n"
                f"{proc.stderr[-2000:]}"
            )
        return self._parse_output(proc.stdout)

    @staticmethod
    def _parse_output(stdout: str) -> np.ndarray:
        text = stdout.strip()
        start = text.rfind("[")
        if start == -1:
            raise ValueError(f"Output tidak mengandung array JSON:\n{text[:500]}")
        end = text.rfind("]")
        payload = text[start : end + 1]

        try:
            data = json.loads(payload)
        except json.JSONDecodeError as e:
            raise ValueError(f"Gagal parse output embedding: {e}") from e

        if isinstance(data, dict):
            for key in ("embeddings", "vectors", "data"):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break

        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(f"Bentuk embedding tak terduga: {arr.shape}")
        return arr


class HFTransformersEmbeddings(Embeddings):
    """
    Embedding via transformers (safetensors lokal, GPU/CPU) - EMBEDDING_MODE=transformers.

    Dipakai untuk model Qwen3-VL-Embedding FP8/safetensors (mis. alexliap/
    Qwen3-VL-Embedding-2B-FP8-DYNAMIC) tanpa binary llama.cpp. Model dimuat malas
    (lazy) agar import ringan; dimuat sekali lalu di-cache.

    Catatan: embedding teks<->gambar model ini kurang sejajar tanpa vLLM
    (is_matryoshka). Untuk retrieval teks<->teks tetap valid.
    """

    DEFAULT_INSTRUCTION = "Represent the user's input."

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "auto",
        pooling: str = "last",
        dtype: Any = torch.bfloat16,
        normalize: bool = True,
        max_length: int = 8192,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.pooling = pooling
        self.dtype = dtype
        self.normalize = normalize
        self.max_length = max_length
        self._device = (
            "cuda"
            if device == "auto" and torch.cuda.is_available()
            else ("cpu" if device == "auto" else device)
        )
        self._model: Any = None
        self._processor: Any = None

    # --- malas: muat model saat pertama kali dipakai --------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoProcessor

        self._model = (
            AutoModel.from_pretrained(self.model_name_or_path, trust_remote_code=True, dtype=self.dtype)
            .to(self._device)
            .eval()
        )
        self._processor = AutoProcessor.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        print(f"[embedding] transformers model siap di {self._device}")

    # --- inti: embed daftar item ----------------------------------------
    def _embed(self, items: List[Dict[str, str]]) -> List[List[float]]:
        """Tiap item: {text?} dan/atau {image: path}."""
        self._ensure_loaded()
        from PIL import Image

        text_prompts: List[str] = []
        img_by_idx: Dict[int, Any] = {}
        for idx, item in enumerate(items):
            content: list = []
            if item.get("image"):
                content.append({"type": "image", "image": item["image"]})
            if item.get("text"):
                content.append({"type": "text", "text": item["text"]})
            conv = [
                {"role": "system", "content": [{"type": "text", "text": self.DEFAULT_INSTRUCTION}]},
                {"role": "user", "content": content},
            ]
            text_prompts.append(
                self._processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            )
            if item.get("image"):
                img_by_idx[idx] = Image.open(item["image"]).convert("RGB")

        results: List[List[float]] = [None] * len(items)  # type: ignore[list-item]

        # Batch item teks-only (tanpa gambar)
        txt_idx = [i for i in range(len(items)) if i not in img_by_idx]
        if txt_idx:
            outs = self._embed_batch([text_prompts[i] for i in txt_idx], images=None)
            for i, out in zip(txt_idx, outs):
                results[i] = out

        # Batch item dengan gambar
        if img_by_idx:
            idx = sorted(img_by_idx)
            outs = self._embed_batch(
                [text_prompts[i] for i in idx],
                images=[img_by_idx[i] for i in idx],
            )
            for i, out in zip(idx, outs):
                results[i] = out

        return results

    def _embed_batch(
        self, text_prompts: List[str], images: Optional[List[Any]]
    ) -> List[List[float]]:
        inputs = self._processor(
            text=text_prompts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            hidden = self._model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].float()

        if self.pooling == "mean":
            emb = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        else:  # "last"
            flipped = mask.flip(dims=[1])
            col = mask.shape[1] - flipped.argmax(dim=1) - 1
            row = torch.arange(hidden.shape[0], device=hidden.device)
            emb = hidden[row, col]

        if self.normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb.float().cpu().tolist()

    # --- antarmuka LangChain Embeddings --------------------------------
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed([{"text": t} for t in texts])

    def embed_query(self, text: str) -> List[float]:
        return self._embed([{"text": text}])[0]

    def embed_images(self, image_paths: List[str]) -> List[List[float]]:
        return self._embed([{"image": p} for p in image_paths])

    def embed_mixed(self, items: List[Dict[str, str]]) -> List[List[float]]:
        return self._embed(items)


class LlamaVLEmbeddings(Embeddings):
    """Adapter LangChain `Embeddings` di atas `VisionEmbedder` (subprocess)."""

    def __init__(self, embedder: VisionEmbedder) -> None:
        self._embedder = embedder

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embedder.embed_text(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._embedder.embed_text([text])[0].tolist()

    def embed_images(self, image_paths: List[str]) -> List[List[float]]:
        return self._embedder.embed_image(image_paths).tolist()

    def embed_mixed(self, items: List[Dict[str, str]]) -> List[List[float]]:
        return self._embedder.embed_mixed(items).tolist()


class LlamaServerEmbeddings(Embeddings):
    """
    Embedding via `llama-server` (endpoint OpenAI-compatible `/v1/embeddings`).

    Cocok untuk mengetes embedding teks dari server remote. Untuk embedding
    teks+gambar patched (Qwen3-VL), gunakan mode `subprocess`.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 300,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or ""
        self.timeout = timeout

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = requests.post(
            f"{self.base_url}{path}", json=payload, headers=headers, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        data = self._post("/embeddings", {"model": self.model, "input": texts})
        items = sorted(data["data"], key=lambda x: x.get("index", 0))
        return [item["embedding"] for item in items]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def embed_images(self, image_paths: List[str]) -> List[List[float]]:
        raise NotImplementedError(
            "Embedding gambar via HTTP tidak didukung endpoint standar "
            "llama-server. Gunakan EMBEDDING_MODE=subprocess."
        )

    def embed_mixed(self, items: List[Dict[str, str]]) -> List[List[float]]:
        raise NotImplementedError(
            "Embedding campuran via HTTP tidak didukung endpoint standar "
            "llama-server. Gunakan EMBEDDING_MODE=subprocess."
        )


def build_embeddings(settings: Optional[Settings] = None) -> Embeddings:
    """Bangun embedding sesuai `settings.embedding_mode` (subprocess | http | transformers)."""
    settings = settings or get_settings()

    if settings.embedding_mode == "transformers":
        if not settings.embedding_hf_model:
            raise ValueError(
                "EMBEDDING_MODE=transformers butuh EMBEDDING_HF_MODEL "
                "(path model atau HF id, mis. models/Qwen3-VL-Embedding-2B-FP8-DYNAMIC)"
            )
        return HFTransformersEmbeddings(
            model_name_or_path=settings.embedding_hf_model,
            device=settings.embedding_device,
            pooling=settings.embedding_pooling,
        )

    if settings.embedding_mode == "http":
        return LlamaServerEmbeddings(
            base_url=settings.embedding_base_url,
            model=settings.embedding_model,
            api_key=settings.embedding_api_key,
        )

    embedder = VisionEmbedder(
        binary=settings.embedder_binary,
        model_path=settings.embedding_model,
        mmproj_path=settings.embedding_mmproj,
        pooling=settings.embedding_pooling,
        embd_normalize=settings.embedding_normalize,
        context=settings.embedding_context,
        ngl=settings.embedding_ngl,
    )
    return LlamaVLEmbeddings(embedder)
