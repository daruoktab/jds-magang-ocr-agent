"""
Indeks RAG Multimodal: embed + simpan teks/metadata ke ruang vektor vision embedding,
dilengkapi dengan Two-Stage Retrieval (Embedding Search -> Qwen3-VL-Reranker)
dan opsi persistensi lokal.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from .reranker import Qwen3VLReranker


class VisionIndex:
    """
    Indeks vektor untuk dokumen teks & metadata multimodal dengan dukungan
    Two-Stage Retrieval (Embeddings + Reranker) dan persistensi lokal.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        reranker: Qwen3VLReranker | None = None,
        persist_path: str | Path | None = None,
    ) -> None:
        self._embeddings = embeddings
        self._reranker = reranker
        self._persist_path = Path(persist_path) if persist_path else None
        self._store = InMemoryVectorStore(embeddings)
        self._docs_cache: list[Document] = []

        if self._persist_path and self._persist_path.exists():
            self._load_from_disk()

    # --- Ingestion & Storage ------------------------------------------------
    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        auto_save: bool = True,
    ) -> list[str]:
        """Tambahkan daftar teks ke indeks vektor."""
        if not texts:
            return []
        metas = metadatas or [{} for _ in texts]
        docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metas)]
        self._docs_cache.extend(docs)
        ids = self._store.add_texts(texts, metadatas=metadatas)
        if auto_save and self._persist_path:
            self._save_to_disk()
        return ids

    def add_documents(
        self,
        documents: list[Document],
        auto_save: bool = True,
    ) -> list[str]:
        """Tambahkan daftar Document LangChain ke indeks."""
        if not documents:
            return []
        self._docs_cache.extend(documents)
        ids = self._store.add_documents(documents)
        if auto_save and self._persist_path:
            self._save_to_disk()
        return ids

    def ingest_knowledge(self, source: str | Path | list[dict[str, Any]]) -> int:
        """
        Ingest pengetahuan eksternal (daftar dict, file JSON, atau file teks).
        Tiap item: {"content": "...", "metadata": {...}}.
        """
        items_to_add: list[Document] = []
        if isinstance(source, (str, Path)):
            p = Path(source)
            if not p.exists():
                raise FileNotFoundError(f"Sumber data tidak ditemukan: {p}")
            if p.suffix.lower() == ".json":
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    for item in data:
                        content = item.get("content") or item.get("text") or json.dumps(item, ensure_ascii=False)
                        meta = {k: v for k, v in item.items() if k not in ("content", "text")}
                        items_to_add.append(Document(page_content=content, metadata=meta))
            else:
                text = p.read_text(encoding="utf-8")
                items_to_add.append(Document(page_content=text, metadata={"source": str(p)}))
        elif isinstance(source, list):
            for item in source:
                content = item.get("content") or item.get("text") or json.dumps(item, ensure_ascii=False)
                meta = {k: v for k, v in item.items() if k not in ("content", "text")}
                items_to_add.append(Document(page_content=content, metadata=meta))

        if items_to_add:
            self.add_documents(items_to_add)
        return len(items_to_add)

    # --- Two-Stage Retrieval ------------------------------------------------
    def search(
        self,
        query: str,
        k: int = 4,
        rerank: bool = True,
    ) -> list[Document]:
        """
        Cari dokumen relevan. Jika reranker aktif, gunakan Two-Stage Retrieval:
        1. Recall kandidat (k * 3 atau min 10) via Vector Embeddings.
        2. Rerank kandidat via Qwen3-VL-Reranker -> pilih top-k.
        """
        if not self._docs_cache:
            return []

        if not self._reranker or not rerank:
            return self._store.similarity_search(query, k=k)

        # Stage 1: Ambil kandidat lebih banyak dari vector store
        fetch_k = max(k * 3, 10)
        candidates = self._store.similarity_search(query, k=fetch_k)
        if not candidates:
            return []

        # Stage 2: Refine presisi dengan Qwen3-VL-Reranker
        try:
            return self._reranker.rerank_documents(query=query, documents=candidates, top_k=k)
        except Exception as e:
            print(f"[warn] Gagal melakukan reranking ({e}), fallback ke similarity search biasa.")
            return candidates[:k]

    def as_retriever(self, k: int = 4, rerank: bool = True):
        """Retriever wrapper kompatibel LangChain."""
        class _CustomRetriever:
            def __init__(self, idx: VisionIndex, k_: int, rerank_: bool):
                self.idx = idx
                self.k = k_
                self.rerank = rerank_

            def invoke(self, query: str) -> list[Document]:
                return self.idx.search(query, k=self.k, rerank=self.rerank)

        return _CustomRetriever(self, k, rerank)

    # --- Persistensi --------------------------------------------------------
    def _save_to_disk(self) -> None:
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {"content": d.page_content, "metadata": d.metadata}
            for d in self._docs_cache
        ]
        self._persist_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _load_from_disk(self) -> None:
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            payload = json.loads(self._persist_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                docs = [Document(page_content=item["content"], metadata=item.get("metadata", {})) for item in payload]
                self.add_documents(docs, auto_save=False)
                print(f"[vector_store] Berhasil memuat {len(docs)} dokumen dari {self._persist_path}")
        except Exception as e:
            print(f"[warn] Gagal memuat indeks dari {self._persist_path}: {e}")
