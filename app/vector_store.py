"""
Indeks RAG: embed + simpan teks ke ruang vektor vision embedding.

Menggunakan `InMemoryVectorStore` dari LangChain dengan embedding
`LlamaVLEmbeddings` (Qwen3-VL-Embedding via llama.cpp). Untuk produksi,
ganti dengan vector store persisten (Chroma/FAISS/LanceDB) tanpa mengubah
antarmuka retrieval.
"""
from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from .embedding import LlamaVLEmbeddings


class VisionIndex:
    """Indeks vektor untuk dokumen teks (tabel markdown, hasil ekstraksi, dll.)."""

    def __init__(self, embeddings: LlamaVLEmbeddings) -> None:
        self._embeddings = embeddings
        self._store = InMemoryVectorStore(embeddings)

    def add_texts(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[str]:
        return self._store.add_texts(texts, metadatas=metadatas)

    def add_documents(self, documents: List[Document]) -> List[str]:
        return self._store.add_documents(documents)

    def search(self, query: str, k: int = 4) -> List[Document]:
        return self._store.similarity_search(query, k=k)

    def as_retriever(self, k: int = 4):
        """Kembalikan retriever LangChain standar."""
        return self._store.as_retriever(search_kwargs={"k": k})
