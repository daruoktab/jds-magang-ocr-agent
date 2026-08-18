"""
Indeks RAG: embed + simpan teks ke ruang vektor vision embedding.

Mendukung:
  1. Vector Store Persisten (ChromaDB) - data tersimpan ke disk dan dapat diakses antar sesi.
  2. In-Memory Vector Store (InMemoryVectorStore) - fallback cepat tanpa persistensi disk.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore

from .config import Settings, get_settings


class VisionIndex:
    """Indeks vektor multimodal untuk dokumen teks, tabel markdown, hasil ekstraksi, dll."""

    def __init__(
        self,
        embeddings: Embeddings,
        persist_directory: Optional[str] = None,
        collection_name: str = "jds_documents",
    ) -> None:
        self._embeddings = embeddings
        self._persist_directory = persist_directory
        self._collection_name = collection_name
        self._store = self._init_store()

    def _init_store(self) -> VectorStore:
        if self._persist_directory:
            try:
                from langchain_chroma import Chroma

                Path(self._persist_directory).mkdir(parents=True, exist_ok=True)
                return Chroma(
                    collection_name=self._collection_name,
                    embedding_function=self._embeddings,
                    persist_directory=self._persist_directory,
                )
            except Exception as e:
                warnings.warn(
                    f"Gagal menginisialisasi Chroma persisten di '{self._persist_directory}' ({e}). "
                    "Beralih menggunakan InMemoryVectorStore."
                )
        return InMemoryVectorStore(self._embeddings)

    @property
    def is_persistent(self) -> bool:
        """Apakah vector store ini menyimpan data secara persisten ke disk."""
        return self._persist_directory is not None and not isinstance(
            self._store, InMemoryVectorStore
        )

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Tambahkan teks ke dalam indeks vektor."""
        return self._store.add_texts(texts, metadatas=metadatas, ids=ids)

    def add_documents(
        self, documents: List[Document], ids: Optional[List[str]] = None
    ) -> List[str]:
        """Tambahkan dokumen LangChain Document ke dalam indeks vektor."""
        return self._store.add_documents(documents, ids=ids)

    def search(self, query: str, k: int = 4) -> List[Document]:
        """Cari dokumen yang paling relevan dengan query."""
        return self._store.similarity_search(query, k=k)

    def search_with_score(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        """Cari dokumen relevan beserta skor jarak/kesamaan."""
        if hasattr(self._store, "similarity_search_with_score"):
            return self._store.similarity_search_with_score(query, k=k)
        docs = self._store.similarity_search(query, k=k)
        return [(doc, 0.0) for doc in docs]

    def count(self) -> int:
        """Hitung jumlah item yang tersimpan di dalam vector store."""
        try:
            if hasattr(self._store, "_collection") and self._store._collection:
                return self._store._collection.count()
            if hasattr(self._store, "store"):
                return len(self._store.store)
        except Exception:
            pass
        return 0

    def clear(self) -> None:
        """Kosongkan seluruh data di dalam collection vector store."""
        if hasattr(self._store, "delete_collection"):
            try:
                self._store.delete_collection()
                # Re-inisialisasi store setelah dihapus
                self._store = self._init_store()
                return
            except Exception as e:
                warnings.warn(f"Gagal menghapus collection: {e}")
        self._store = InMemoryVectorStore(self._embeddings)

    def as_retriever(self, k: int = 4):
        """Kembalikan retriever LangChain standar."""
        return self._store.as_retriever(search_kwargs={"k": k})


def build_vision_index(
    settings: Optional[Settings] = None,
    embeddings: Optional[Embeddings] = None,
    allow_fallback: bool = True,
) -> VisionIndex:
    """
    Factory helper untuk membangun VisionIndex persisten dari Settings.

    Jika model embedding belum siap / binary belum dibuild, fallback otomatis ke
    DeterministicFakeEmbedding agar operasi vektor store tetap berjalan.
    """
    settings = settings or get_settings()
    if embeddings is None:
        from .embedding import build_embeddings

        try:
            embeddings = build_embeddings(settings)
        except Exception as e:
            if allow_fallback:
                from langchain_core.embeddings import DeterministicFakeEmbedding

                warnings.warn(
                    f"Model embedding tidak dapat dimuat ({e}). "
                    "Menggunakan embedding fallback (DeterministicFakeEmbedding) untuk operasi vektor."
                )
                embeddings = DeterministicFakeEmbedding(size=1024)
            else:
                raise

    persist_dir = settings.vector_store_dir if settings.vector_store_dir else None
    return VisionIndex(
        embeddings=embeddings,
        persist_directory=persist_dir,
        collection_name=settings.vector_store_collection,
    )
