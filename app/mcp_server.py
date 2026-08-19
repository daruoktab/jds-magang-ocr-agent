"""
MCP (Model Context Protocol) Server untuk jds-magang-ocr-agent.

Mengekspos alat-alat ekstraksi Vision OCR, PowerPoint parser, multi-page PDF stitcher,
pemindaian direktori dataset, ekstraksi massal, dan simulasi chunking sebagai MCP Tools berstandar SDK v2.

Menjalankan server:
    python -m app.mcp_server
    python mcp_server.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from .batch import batch_extract_documents as run_batch_extract, scan_document_directories
from .config import get_settings
from .deep_agent import build_deep_agent
from .extractor import VisionExtractor
from .graph import DocumentExtractionPipeline
from .llm import build_vlm
from .multi_page import preview_markdown_chunks as sim_preview_chunks
from .ocr import build_ocr_extractor
from .pdf import process_multipage_pdf
from .ppt import process_presentation
from .preprocess import preprocess_image

# Inisialisasi Server MCP
server = MCPServer(
    name="jds-magang-ocr-agent",
    description="Vision OCR & Document Extractor MCP Server: PDF, PPTX, Scan -> Markdown Siap Chunking",
    version="0.1.0",
)


@server.tool(
    name="scan_document_folders",
    description=(
        "Pindai direktori (misal 'dataset', 'output', 'input', atau path khusus) dan sub-subfoldernya "
        "untuk mendeteksi keberadaan folder yang berisi file dokumen (PDF, PPTX, PPT, Gambar). "
        "Mengembalikan daftar folder yang tersedia, jumlah file per ekstensi, dan contoh nama file."
    ),
)
def scan_document_folders(
    root_dir: str = ".",
) -> str:
    """
    Pindai root_dir untuk menemukan folder-folder yang berisi file dokumen.
    """
    try:
        found_folders = scan_document_directories(root_dir)
        return json.dumps(
            {
                "root_scanned": str(Path(root_dir).resolve()),
                "total_folders_found": len(found_folders),
                "folders": found_folders,
            },
            indent=2,
            ensure_ascii=False,
        )
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat memindai folder: {e}"


@server.tool(
    name="batch_extract_documents",
    description=(
        "Ekstrak dokumen secara massal dari satu atau banyak folder yang dipilih oleh user. "
        "Mendukung pemilihan multi-folder (list atau koma), pembatasan kuota jumlah data (total limit atau limit per folder), "
        "pemilihan spesifikasi layout dokumen, dan penyimpanan hasil ke direktori output."
    ),
)
def batch_extract_documents(
    folders: str,
    limit: int | None = None,
    limit_per_folder: int | None = None,
    specs: str = "plain",
    output_dir: str = "output/extracted_md",
    preview_chunks: bool = False,
    chunk_size: int = 1000,
) -> str:
    """
    Ekstrak dokumen dari folder-folder terpilih dengan kuota data tertentu.

    Args:
        folders: Path folder atau daftar folder dipisah koma (mis. 'dataset/download/indonesian,dataset/download/english').
        limit: Batas total file yang diproses.
        limit_per_folder: Batas file per folder.
        specs: Spesifikasi layout ('plain', 'markdown_hierarchy', 'bilingual_journal', 'presentation_slides', atau komposit).
        output_dir: Folder tujuan penyimpanan hasil Markdown.
        preview_chunks: Sertakan simulasi chunking.
        chunk_size: Ukuran chunk karakter.
    """
    try:
        result = run_batch_extract(
            folders=folders,
            limit=limit,
            limit_per_folder=limit_per_folder,
            specs=specs,
            output_dir=output_dir,
            preview_chunks=preview_chunks,
            chunk_size=chunk_size,
        )
        return json.dumps(result, indent=2, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat batch ekstraksi dokumen: {e}"


@server.tool(
    name="extract_document_to_markdown",
    description=(
        "Ekstrak file dokumen tunggal (PDF, PPTX, JPG, PNG, WEBP) menjadi teks Markdown bersih yang siap langsung di-chunking. "
        "Mendukung spesifikasi tunggal maupun komposit: 'plain', 'markdown_hierarchy', 'bilingual_journal', "
        "'presentation_slides', atau kombinasi seperti 'journal,hierarchy'."
    ),
)
def extract_document_to_markdown(
    file_path: str,
    specs: str = "plain",
    dpi: int = 200,
) -> str:
    """
    Ekstrak file dokumen ke Markdown terstruktur.
    """
    path_obj = Path(file_path).resolve()
    if not path_obj.exists():
        return f"ERROR: File tidak ditemukan: {path_obj}"

    ext = path_obj.suffix.lower()
    settings = get_settings()

    try:
        if ext in (".pptx", ".ppt"):
            return process_presentation(path_obj)
        elif ext == ".pdf":
            pipeline = DocumentExtractionPipeline(settings)
            extracted = process_multipage_pdf(
                pdf_path=path_obj,
                pipeline=pipeline,
                dpi=dpi,
                forced_specs=specs,
            )
            return extracted.markdown_content
        else:
            pipeline = DocumentExtractionPipeline(settings)
            res = pipeline.run(str(path_obj), forced_specs=specs)
            return str(res["markdown_content"])
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat ekstraksi dokumen: {e}"


@server.tool(
    name="ocr_image",
    description="Ekstrak teks mentah literal beresolusi tinggi dari gambar menggunakan model OCR tuned (ocr-lighton).",
)
def ocr_image(image_path: str) -> str:
    """
    Lakukan OCR langsung pada file gambar untuk mendapatkan teks mentah.
    """
    path_obj = Path(image_path).resolve()
    if not path_obj.exists():
        return f"ERROR: File tidak ditemukan: {path_obj}"

    settings = get_settings()
    try:
        proc = preprocess_image(str(path_obj))
        ocr = build_ocr_extractor(settings)
        return ocr.extract(proc.processed_path).text
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat OCR: {e}"


@server.tool(
    name="classify_document_layout",
    description="Analisis dan klasifikasikan seluruh spesifikasi tata letak dokumen yang aktif (plain, markdown_hierarchy, bilingual_journal, presentation_slides).",
)
def classify_document_layout(image_path: str) -> str:
    """
    Klasifikasi layout gambar dokumen (multi-label).
    """
    path_obj = Path(image_path).resolve()
    if not path_obj.exists():
        return f"ERROR: File tidak ditemukan: {path_obj}"

    settings = get_settings()
    try:
        proc = preprocess_image(str(path_obj))
        vlm = build_vlm(settings)
        extractor = VisionExtractor(vlm)
        specs = extractor.classify(proc.processed_path)
        return json.dumps({"file": str(path_obj), "specs": specs}, indent=2, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat klasifikasi: {e}"


@server.tool(
    name="extract_presentation_pptx",
    description="Ekstrak file presentasi PowerPoint (.pptx / .ppt) menjadi Markdown terstruktur per slide dengan bullet points hierarkis, tabel GFM, dan speaker notes.",
)
def extract_presentation_pptx(pptx_path: str) -> str:
    """
    Ekstrak presentasi PowerPoint ke Markdown.
    """
    path_obj = Path(pptx_path).resolve()
    if not path_obj.exists():
        return f"ERROR: File tidak ditemukan: {path_obj}"

    try:
        return process_presentation(path_obj)
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat memproses presentasi: {e}"


@server.tool(
    name="preview_markdown_chunks",
    description="Simulasikan pemecahan dokumen Markdown dengan splitter berbasis header (#, ##, ###) dan recursive character text splitter.",
)
def preview_markdown_chunks(
    markdown_text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> str:
    """
    Simulasikan chunking pada teks Markdown.
    """
    try:
        chunks = sim_preview_chunks(
            markdown_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return json.dumps(
            {"total_chunks": len(chunks), "chunks": chunks},
            indent=2,
            ensure_ascii=False,
        )
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat simulasi chunking: {e}"


@server.tool(
    name="run_deep_reasoning_agent",
    description="Jalankan Master Deep Reasoning Agent dengan delegasi otonom ke 6 sub-agents untuk mengekstrak dan memproses dokumen.",
)
def run_deep_reasoning_agent(
    file_path: str,
    instruction: str = "",
) -> str:
    """
    Jalankan deep reasoning agent pada dokumen.
    """
    path_obj = Path(file_path).resolve()
    if not path_obj.exists():
        return f"ERROR: File tidak ditemukan: {path_obj}"

    settings = get_settings()
    try:
        agent = build_deep_agent(settings)
        prompt = (
            f"Ekstrak dan proses file dokumen berikut secara lengkap: '{path_obj}'. "
            f"Gunakan penalaran tata letak dan delegasikan ke sub-agent spesialis yang relevan. "
            f"{instruction}"
        )
        resp = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        messages = resp.get("messages", [])
        return messages[-1].content if messages else str(resp)
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat menjalankan deep reasoning agent: {e}"


def main() -> None:
    """Entry point untuk menjalankan MCP Server via stdio transport."""
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
