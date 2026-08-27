"""
MCP (Model Context Protocol) Server untuk jds-magang-ocr-agent.

Mengekspos alat-alat ekstraksi Vision OCR, PowerPoint parser, rendering slide ke gambar,
multi-page PDF stitcher, pemindaian direktori dataset, ekstraksi massal, simulasi chunking,
ingesti & double-verification tabel transaksional ke database SQLite,
serta evaluasi & ekstraksi selektif diagram ke sintaks Mermaid.js sebagai MCP Tools berstandar SDK v2.

Menjalankan server:
    python -m app.mcp_server
    python mcp_server.py
"""

from __future__ import annotations

import json
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from .batch import batch_extract_documents as run_batch_extract
from .batch import scan_document_directories
from .config import get_settings
from .deep_agent import build_deep_agent
from .diagram import (
    classify_diagram_convertibility as run_classify_diagram,
)
from .diagram import (
    extract_diagram_to_mermaid as run_extract_diagram,
)
from .extractor import VisionExtractor
from .graph import DocumentExtractionPipeline
from .llm import build_vlm
from .multi_page import preview_markdown_chunks as sim_preview_chunks
from .ocr import build_ocr_extractor
from .pdf import process_multipage_pdf
from .ppt import process_presentation_vision
from .preprocess import preprocess_image
from .tabular_db import (
    TabularDatabaseManager,
    extract_and_ingest_tables_from_markdown,
    query_sqlite,
)

# Inisialisasi Server MCP
server = MCPServer(
    name="jds-magang-ocr-agent",
    description="Vision OCR & Document Extractor MCP Server: PDF, PPTX, Scan -> Markdown Siap Chunking, SQLite Tabular Engine, & Mermaid Diagrams",
    version="0.1.0",
)


@server.tool(
    name="render_presentation_slides",
    description=(
        "Render seluruh halaman/slide file presentasi PowerPoint (.pptx / .ppt) menjadi file gambar PNG beresolusi tinggi "
        "agar dapat dilihat dan dianalisis secara visual langsung oleh AI Multimodal / VLM (renderer: LibreOffice headless)."
    ),
)
def render_presentation_slides(
    pptx_path: str,
    output_dir: str | None = None,
) -> str:
    """
    Render slide PPTX ke file gambar PNG per slide.
    """
    path_obj = Path(pptx_path).resolve()
    if not path_obj.exists():
        return f"ERROR: File presentasi tidak ditemukan: {path_obj}"

    out_dir = Path(output_dir or f"output/rendered_slides/{path_obj.stem}").resolve()
    if out_dir.exists():
        for old_f in out_dir.glob("slide_*_img_*.*"):
            try:
                old_f.unlink(missing_ok=True)
            except OSError:
                pass

    try:
        import importlib

        import app.ppt

        importlib.reload(app.ppt)
        images = app.ppt.render_presentation_slides_to_images(
            path_obj, output_dir=out_dir
        )
        return json.dumps(
            {
                "pptx_file": str(path_obj),
                "total_slides_rendered": len(images),
                "rendered_image_paths": [str(p) for p in images],
            },
            indent=2,
            ensure_ascii=False,
        )
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat merender slide presentasi: {e}"


@server.tool(
    name="scan_document_folders",
    description=(
        "Pindai direktori (misal 'dataset', 'output', 'input', atau path khusus) dan sub-subfoldernya "
        "untuk mendeteksi keberadaan folder yang berisi file dokumen (PDF, PPTX, PPT, Gambar). "
        "Mengembalikan daftar folder yang tersedia, jumlah file per ekstensi, dan contoh nama file."
    ),
)
def scan_document_folders(
    base_dir: str = "dataset",
    max_depth: int = 5,
) -> str:
    """
    Pindai struktur direktori dokumen.
    """
    try:
        results = scan_document_directories(
            root_dir=base_dir,
            max_depth=max_depth,
        )
        return json.dumps(results, indent=2, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat memindai direktori: {e}"


@server.tool(
    name="batch_extract_documents",
    description=(
        "Jalankan ekstraksi dokumen secara massal (batch) pada daftar folder atau direktori tertentu. "
        "Mendukung batas jumlah dokumen, spesifikasi tata letak, dan simulasi preview chunking."
    ),
)
def batch_extract_documents(
    target_folders: list[str] | str = "dataset",
    output_dir: str = "output/extracted_md",
    max_documents: int | None = None,
    limit_per_folder: int | None = None,
    forced_specs: str = "plain",
    preview_chunks: bool = False,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> str:
    """
    Ekstraksi dokumen secara massal.
    """
    try:
        summary = run_batch_extract(
            folders=target_folders,
            output_dir=output_dir,
            limit=max_documents,
            limit_per_folder=limit_per_folder,
            specs=forced_specs,
            preview_chunks=preview_chunks,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return json.dumps(summary, indent=2, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat ekstraksi batch: {e}"


@server.tool(
    name="extract_document_to_markdown",
    description=(
        "Ekstrak satu file dokumen (PDF, PPTX, Scan, Gambar) menjadi teks Markdown bersih siap chunking. "
        "Mendukung multi-spesifikasi layout komposit ('plain', 'markdown_hierarchy', 'bilingual_journal', 'presentation_slides')."
    ),
)
def extract_document_to_markdown(
    file_path: str,
    specs: str | None = None,
    dpi: int = 200,
) -> str:
    """
    Ekstrak file dokumen ke Markdown.
    """
    path_obj = Path(file_path).resolve()
    if not path_obj.exists():
        return f"ERROR: File tidak ditemukan: {path_obj}"

    ext = path_obj.suffix.lower()
    settings = get_settings()

    try:
        pipeline = DocumentExtractionPipeline(settings)

        if ext == ".pdf":
            extracted = process_multipage_pdf(
                pdf_path=path_obj,
                pipeline=pipeline,
                dpi=dpi,
                forced_specs=specs,
            )
            return extracted.markdown_content

        if ext in {".pptx", ".ppt"}:
            return process_presentation_vision(
                pptx_path=path_obj,
                pipeline=pipeline,
                forced_specs=specs or "presentation_slides",
            )

        if ext in {".png", ".jpg", ".jpeg", ".webp"}:
            result = pipeline.run(str(path_obj), forced_specs=specs)
            return result.get("markdown_content", "")

        return f"ERROR: Ekstensi file tidak didukung: {ext}"
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat ekstraksi dokumen: {e}"


@server.tool(
    name="ocr_image",
    description="Ekstrak teks mentah literal dari file gambar menggunakan model OCR resolusi tinggi tanpa interpretasi layout.",
)
def ocr_image(image_path: str) -> str:
    """
    Jalankan OCR teks mentah pada gambar.
    """
    path_obj = Path(image_path).resolve()
    if not path_obj.exists():
        return f"ERROR: File tidak ditemukan: {path_obj}"

    settings = get_settings()
    try:
        proc = preprocess_image(str(path_obj))
        ocr = build_ocr_extractor(settings)
        res = ocr.extract(proc.processed_path)
        return res.text
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat OCR: {e}"


@server.tool(
    name="classify_document_layout",
    description="Analisis karakteristik visual dokumen untuk mendeteksi spesifikasi layout yang relevan (plain, markdown_hierarchy, bilingual_journal, presentation_slides).",
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
        return json.dumps(
            {"file": str(path_obj), "specs": specs}, indent=2, ensure_ascii=False
        )
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat klasifikasi: {e}"


@server.tool(
    name="classify_diagram_convertibility",
    description=(
        "Evaluasi apakah diagram/visual pada dokumen cocok dikonversi menjadi kode diagram Mermaid.js "
        "(flowchart, sequence, ERD, state, class, mindmap, block architecture) atau tidak cocok "
        "(grafik statistik kontinu numerik, peta spasial, foto, skematik sirkuit mikro)."
    ),
)
def classify_diagram_convertibility(image_path: str) -> str:
    """
    Evaluasi kelayakan konversi diagram ke Mermaid.
    """
    path_obj = Path(image_path).resolve()
    if not path_obj.exists():
        return f"ERROR: File tidak ditemukan: {path_obj}"

    settings = get_settings()
    try:
        vlm = build_vlm(settings)
        res = run_classify_diagram(path_obj, llm=vlm)
        return json.dumps(res.model_dump(), indent=2, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat evaluasi diagram: {e}"


@server.tool(
    name="extract_diagram_to_mermaid",
    description=(
        "Ekstrak diagram visual menjadi kode Mermaid.js yang 100% valid dan terstruktur, "
        "atau kembalikan deskripsi struktural/tabel jika diagram tidak cocok untuk Mermaid."
    ),
)
def extract_diagram_to_mermaid(
    image_path: str,
    diagram_hint: str | None = None,
) -> str:
    """
    Ekstrak diagram ke sintaks Mermaid.js.
    """
    path_obj = Path(image_path).resolve()
    if not path_obj.exists():
        return f"ERROR: File tidak ditemukan: {path_obj}"

    settings = get_settings()
    try:
        vlm = build_vlm(settings)
        res = run_extract_diagram(path_obj, llm=vlm, forced_diagram_type=diagram_hint)
        return json.dumps(res.model_dump(), indent=2, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat ekstraksi diagram Mermaid: {e}"


@server.tool(
    name="extract_presentation_pptx",
    description="Render file presentasi PowerPoint (.pptx / .ppt) menjadi gambar per slide, kirim setiap gambar ke VLM, lalu gabungkan hasilnya menjadi satu Markdown terstruktur per slide.",
)
def extract_presentation_pptx(
    pptx_path: str, specs: str = "presentation_slides"
) -> str:
    """
    Ekstrak presentasi PowerPoint ke Markdown.
    """
    path_obj = Path(pptx_path).resolve()
    if not path_obj.exists():
        return f"ERROR: File tidak ditemukan: {path_obj}"

    try:
        settings = get_settings()
        pipeline = DocumentExtractionPipeline(settings)
        return process_presentation_vision(
            pptx_path=path_obj,
            pipeline=pipeline,
            forced_specs=specs,
        )
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


# --- Tabular SQLite MCP Tools ---


@server.tool(
    name="ingest_markdown_tables_to_sqlite",
    description=(
        "Pindai dokumen Markdown untuk menemukan tabel-tabel data, klasifikasikan mana yang bertipe transaksional "
        "(rekening koran, mutasi, log keuangan, faktur, ledger), ubah menjadi skema SQL bersih, simpan ke SQLite, "
        "dan lakukan double-verification otomatis (pemeriksaan baris, validasi tipe data, uji kalkulasi agregat SUM/AVG)."
    ),
)
def ingest_markdown_tables_to_sqlite(
    markdown_text: str,
    source_file: str = "",
    db_path: str | None = None,
) -> str:
    """
    Ingest tabel transaksional dari Markdown ke SQLite dengan double-verification.
    """
    try:
        settings = get_settings()
        vlm = build_vlm(settings)
        results = extract_and_ingest_tables_from_markdown(
            markdown_text=markdown_text,
            source_file=source_file,
            db_path=db_path,
            llm=vlm,
        )
        return json.dumps(
            {
                "status": "success",
                "tables_ingested_count": len(results),
                "results": [r.model_dump() for r in results],
            },
            indent=2,
            ensure_ascii=False,
        )
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat mengingest tabel ke SQLite: {e}"


@server.tool(
    name="query_tabular_database",
    description=(
        "Jalankan query SQL (misal 'SELECT SUM(debit_amount), COUNT(*) FROM ...') pada database SQLite dokumen "
        "untuk melakukan kalkulasi agregat berpresisi 100% yang tidak dapat dilakukan oleh Vector RAG biasa."
    ),
)
def query_tabular_database(
    sql_query: str,
    db_path: str | None = None,
) -> str:
    """
    Eksekusi query SQL pada database SQLite dokumen.
    """
    try:
        res = query_sqlite(sql_query, db_path=db_path)
        return json.dumps(res.model_dump(), indent=2, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat menjalankan query SQL: {e}"


@server.tool(
    name="inspect_tabular_database",
    description="Periksa daftar tabel, skema kolom, jumlah baris, dan contoh record pada database SQLite dokumen.",
)
def inspect_tabular_database(
    db_path: str | None = None,
) -> str:
    """
    Inspeksi skema dan status tabel pada database SQLite.
    """
    try:
        db_mgr = TabularDatabaseManager(db_path)
        info = db_mgr.inspect_database()
        return json.dumps(info, indent=2, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat menginspeksi database: {e}"


@server.tool(
    name="run_deep_reasoning_agent",
    description="Jalankan Master Deep Reasoning Agent dengan delegasi otonom ke 8 sub-agents untuk mengekstrak dan memproses dokumen, diagram visual Mermaid, serta data tabular SQLite.",
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
