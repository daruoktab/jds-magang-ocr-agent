"""
Modul Ingestion Dokumen untuk Vision RAG Index.

Mendukung pembacaan, chunking, dan penyimpanan dokumen referensi ke dalam
VisionIndex (ChromaDB persisten atau InMemoryVectorStore).

Format dokumen yang didukung:
  - Teks / Markdown: `.txt`, `.md`, `.csv`, `.log`
  - PDF Dokumen    : `.pdf` (diekstrak per halaman via pymupdf)
  - Data JSON      : `.json` (hasil ekstraksi sebelumnya atau data terstruktur)
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pymupdf
from langchain_core.documents import Document

from .vector_store import VisionIndex

SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".log", ".json", ".pdf"}


def recursive_split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: Optional[List[str]] = None,
) -> List[str]:
    """
    Pemotong teks rekursif (chunking) tanpa dependensi eksternal tambahan.
    Memotong berdasarkan paragraf, baris baru, spasi, atau karakter.
    """
    if not text or not text.strip():
        return []

    if len(text) <= chunk_size:
        return [text.strip()]

    separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def _split(txt: str, seps: List[str]) -> List[str]:
        if len(txt) <= chunk_size or not seps:
            return [txt] if txt else []

        sep = seps[0]
        remaining_seps = seps[1:]

        if sep == "":
            # Potong per karakter dengan overlap
            chunks: List[str] = []
            step = max(1, chunk_size - chunk_overlap)
            for i in range(0, len(txt), step):
                c = txt[i : i + chunk_size].strip()
                if c:
                    chunks.append(c)
            return chunks

        splits = txt.split(sep)
        chunks: List[str] = []
        current_chunk = ""

        for part in splits:
            part_str = part + sep if sep != "" else part
            if len(part_str) > chunk_size and remaining_seps:
                # Bagian ini terlalu besar, potong dengan pemisah level berikutnya
                sub_chunks = _split(part, remaining_seps)
                for sc in sub_chunks:
                    if len(current_chunk) + len(sc) <= chunk_size:
                        current_chunk += (" " if current_chunk else "") + sc
                    else:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        current_chunk = sc
            elif len(current_chunk) + len(part_str) <= chunk_size:
                current_chunk += part_str
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = part_str

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    raw_chunks = _split(text, separators)
    # Filter chunk kosong & pastikan ukuran sesuai
    final_chunks: List[str] = []
    for c in raw_chunks:
        c_clean = c.strip()
        if c_clean:
            final_chunks.append(c_clean)

    return final_chunks


def load_file_documents(
    file_path: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """Baca satu berkas dan potong menjadi daftar LangChain Document."""
    file_path = Path(file_path).resolve()
    if not file_path.exists() or not file_path.is_file():
        return []

    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return []

    documents: List[Document] = []
    timestamp = datetime.now().isoformat(timespec="seconds")
    base_meta = {
        "source": str(file_path),
        "filename": file_path.name,
        "extension": ext,
        "ingested_at": timestamp,
    }

    if ext in {".txt", ".md", ".csv", ".log"}:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            chunks = recursive_split_text(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for idx, chunk in enumerate(chunks):
                meta = {**base_meta, "chunk_index": idx, "total_chunks": len(chunks)}
                documents.append(Document(page_content=chunk, metadata=meta))
        except Exception as e:
            print(f"[warn] Gagal membaca teks '{file_path.name}': {e}")

    elif ext == ".pdf":
        try:
            doc = pymupdf.open(str(file_path))
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text().strip()
                if not page_text:
                    continue
                chunks = recursive_split_text(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                for idx, chunk in enumerate(chunks):
                    meta = {
                        **base_meta,
                        "page": page_num + 1,
                        "total_pages": len(doc),
                        "chunk_index": idx,
                    }
                    documents.append(Document(page_content=chunk, metadata=meta))
            doc.close()
        except Exception as e:
            print(f"[warn] Gagal membaca PDF '{file_path.name}': {e}")

    elif ext == ".json":
        try:
            raw = file_path.read_text(encoding="utf-8", errors="ignore")
            data = json.loads(raw)
            # Format JSON menjadi string representasi terstruktur
            text_repr = json.dumps(data, indent=2, ensure_ascii=False)
            chunks = recursive_split_text(text_repr, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for idx, chunk in enumerate(chunks):
                meta = {**base_meta, "chunk_index": idx, "doc_type": data.get("doc_type", "json")}
                documents.append(Document(page_content=chunk, metadata=meta))
        except Exception as e:
            print(f"[warn] Gagal membaca JSON '{file_path.name}': {e}")

    return documents


def ingest_path(
    path: str | Path,
    index: VisionIndex,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> Dict[str, Any]:
    """
    Ingest satu file atau seluruh isi direktori ke dalam VisionIndex.

    Returns:
        Dict berisi statistik: total_files, total_chunks, files_processed.
    """
    target = Path(path).resolve()
    if not target.exists():
        raise FileNotFoundError(f"Path tidak ditemukan: {target}")

    files_to_process: List[Path] = []
    if target.is_file():
        files_to_process.append(target)
    else:
        for item in target.rglob("*"):
            if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
                files_to_process.append(item)

    total_chunks = 0
    processed_files: List[str] = []

    for f in sorted(files_to_process):
        docs = load_file_documents(f, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if docs:
            index.add_documents(docs)
            total_chunks += len(docs)
            processed_files.append(f.name)
            print(f"  [+] Ingested '{f.name}' -> {len(docs)} chunk")

    return {
        "total_files": len(processed_files),
        "total_chunks": total_chunks,
        "files_processed": processed_files,
        "is_persistent": index.is_persistent,
    }
