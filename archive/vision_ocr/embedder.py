"""
VisionEmbedder - embedding multimodal (teks + gambar) berbasis Qwen3-VL-Embedding
via binary `llama-vl-embedding` (fork qwen3-vl-embedding di project ini).

Pengganti Embedder bge-m3 pada retriever TableRAG: satu model embedding untuk
teks, tabel markdown, dan gambar - semuanya masuk ke ruang vektor yang sama.

Setup (lihat README fork qwen3-vl-embedding):
  1. cd qwen3-vl-embedding && git submodule update --init --recursive
  2. Download model Qwen3-VL-Embedding-2B/8B dari Hugging Face
     (folder kosong `Qwen3-VL-Embedding` akan terisi oleh submodule HF reference)
  3. Convert ke GGUF:  convert_hf_to_gguf.py (main + --mmproj)
  4. Build:            cmake -S llama.cpp -B llama.cpp/build
                       cmake --build llama.cpp/build --target llama-vl-embedding
  5. Binary hasil:     llama.cpp/build/bin/llama-vl-embedding

API batch: --inputs menerima JSON array, jadi beberapa teks/gambar bisa
di-embed dalam satu panggilan subprocess.
"""
import json
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Union

import numpy as np


class VisionEmbedder:
    """Wrapper subprocess untuk `llama-vl-embedding`."""

    def __init__(
        self,
        binary: Optional[str] = None,
        model_path: Optional[str] = None,
        mmproj_path: Optional[str] = None,
        pooling: str = "last",
        embd_normalize: int = 2,
        context: int = 4096,
        ngl: Optional[Union[int, str]] = "auto",
        timeout: float = 300,
    ) -> None:
        """
        Args:
            binary: path ke binary llama-vl-embedding. Default: cari di PATH.
            model_path: path ke GGUF utama (mis. Qwen3-VL-Embedding-2B-f16.gguf)
            mmproj_path: path ke GGUF mmproj (wajib untuk input gambar)
            pooling: mode pooling (default 'last' sesuai fork)
            embd_normalize: normalisasi L2 (2 = L2 normalize)
            context: ukuran konteks (-c)
            ngl: jumlah layer GPU ('auto' atau int; 0 = CPU)
            timeout: timeout subprocess (detik)
        """
        self.binary = binary or shutil.which("llama-vl-embedding")
        if not self.binary:
            raise FileNotFoundError(
                "Binary 'llama-vl-embedding' tidak ditemukan. Build dulu: "
                "cmake --build llama.cpp/build --target llama-vl-embedding"
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

    # --- API publik (dengan pola seperti Embedder.encode) ---------------
    def embed_text(self, texts: List[str]) -> np.ndarray:
        """Embed daftar teks -> (n, dim)."""
        items = [{"text": t} for t in texts]
        return self._run(items)

    def embed_image(self, image_paths: List[str]) -> np.ndarray:
        """Embed daftar path gambar -> (n, dim). Wajib mmproj."""
        if not self.mmproj_path:
            raise ValueError("mmproj_path wajib diisi untuk embedding gambar")
        items = [{"image": p} for p in image_paths]
        return self._run(items)

    def embed_mixed(self, items: List[Dict[str, str]]) -> np.ndarray:
        """
        Embed campuran; tiap item dict {text?, image?} (satu saja atau keduanya).

        Contoh:
            embed_mixed([{"text": "A dog on the beach", "image": "./0.jpeg"}])
        """
        return self._run(items)

    # --- internal -------------------------------------------------------
    def _run(self, items: List[Dict[str, str]]) -> np.ndarray:
        if not items:
            return np.zeros((0, 0), dtype=np.float32)

        binary_path = self.binary
        assert binary_path is not None

        cmd = [
            binary_path,
            "-m", self.model_path,
            "--inputs", json.dumps(items, ensure_ascii=False),
            "--pooling", self.pooling,
            "--embd-normalize", str(self.embd_normalize),
            "--embd-output-format", "array",
            "-c", str(self.context),
        ]
        if self.mmproj_path:
            cmd += ["--mmproj", self.mmproj_path]
        if self.ngl is not None:
            cmd += ["-ngl", str(self.ngl)]

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"llama-vl-embedding gagal (rc={proc.returncode}):\n"
                f"{proc.stderr[-2000:]}"
            )

        return self._parse_output(proc.stdout)

    @staticmethod
    def _parse_output(stdout: str) -> np.ndarray:
        """Parse output --embd-output-format array (JSON)."""
        text = stdout.strip()
        # Cari blok JSON terakhir (binary bisa mencetak log sebelum JSON).
        start = text.rfind("[")
        if start == -1:
            raise ValueError(f"Output tidak mengandung array JSON:\n{text[:500]}")
        end = text.rfind("]")
        payload = text[start : end + 1]

        try:
            data = json.loads(payload)
        except json.JSONDecodeError as e:
            raise ValueError(f"Gagal parse output embedding: {e}") from e

        # Format bisa berupa [[...], ...] atau {"embeddings": [[...], ...]}
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
