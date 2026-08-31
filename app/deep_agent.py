"""
Harness Deep Reasoning Agents untuk Ekstraksi Dokumen Vision VLM -> Markdown Siap Chunking, Tabular SQLite Ingestion, & Diagram Mermaid.js.
Menggunakan arsitektur Master Orchestrator dengan 7 Sub-Agent terspesialisasi berbasis Vision Language Model murni.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deepagents import SubAgent, create_deep_agent
from langchain_core.tools import tool

from .agents import get_agent
from .config import Settings, get_settings
from .diagram import (
    classify_diagram_convertibility,
)
from .diagram import (
    extract_diagram_to_mermaid as run_extract_diagram,
)
from .extractor import VisionExtractor
from .llm import build_vlm
from .multi_page import preview_markdown_chunks
from .pdf import process_multipage_pdf
from .ppt import process_presentation_vision
from .preprocess import preprocess_image
from .tabular_db import (
    TabularDatabaseManager,
    TabularVerifier,
    classify_table_heuristic,
    extract_and_ingest_tables_from_markdown,
    parse_markdown_tables,
    query_sqlite,
)


def build_deep_agent(settings: Settings | None = None) -> Any:
    """
    Bangun Deep Reasoning Agent utama dengan armada 7 Sub-Agent spesialis (Pure VLM):
      1. `layout-classifier`          : Mengklasifikasikan multi-trait dokumen
      2. `markdown-extractor`         : Ekstraksi VLM multimodal ke Markdown
      3. `diagram-mermaid-specialist` : Evaluasi selektif & ekstraksi diagram ke sintaks Mermaid.js
      4. `presentation-specialist`    : Parsing file presentasi PowerPoint (.pptx / .ppt)
      5. `pdf-orchestrator`           : Orkestrasi multi-halaman PDF & heading continuity
      6. `chunking-simulator`         : Simulasi partisi teks Markdown siap RAG
      7. `tabular-db-specialist`      : Deteksi tabel transaksional, ingesti ke SQLite, double-verification, & eksekusi SQL
    """
    resolved_settings = settings or get_settings()
    vlm = build_vlm(resolved_settings)
    extractor = VisionExtractor(vlm)

    # --- Tool Definitions ---

    @tool
    def classify_layout(image_path: str) -> str:
        """Analisis gambar dokumen dan kembalikan daftar spesifikasi layout yang aktif (plain, markdown_hierarchy, bilingual_journal, presentation_slides)."""
        proc = preprocess_image(image_path)
        specs = extractor.classify(proc.processed_path)
        return json.dumps({"specs": specs}, ensure_ascii=False)

    @tool
    def extract_to_markdown(
        image_path: str,
        specs: str = "plain",
        previous_context: str | None = None,
    ) -> str:
        """Ekstrak gambar dokumen menjadi teks Markdown bersih sesuai satu atau kombinasi spesifikasi (mis. 'journal,hierarchy', 'presentation_slides')."""
        proc = preprocess_image(image_path)
        agent = get_agent(specs)
        return agent.run(
            proc.processed_path,
            llm=vlm,
            previous_page_context=previous_context,
        )

    @tool
    def classify_diagram_suitability(image_path: str) -> str:
        """Evaluasi kelayakan diagram visual pada dokumen: apakah cocok dikonversi menjadi kode Mermaid yang valid (flowchart, sequence, ERD, state, class, mindmap, block architecture) atau tidak cocok (grafik statistik kontinu, peta, foto, skematik sirkuit mikro)."""
        proc = preprocess_image(image_path)
        res = classify_diagram_convertibility(proc.processed_path, llm=vlm)
        return json.dumps(res.model_dump(), indent=2, ensure_ascii=False)

    @tool
    def extract_diagram_to_mermaid(
        image_path: str,
        diagram_hint: str | None = None,
    ) -> str:
        """Ekstrak diagram visual pada dokumen menjadi kode Mermaid.js yang valid dan terstruktur, atau berikan deskripsi terstruktur jika diagram tidak cocok untuk Mermaid."""
        proc = preprocess_image(image_path)
        res = run_extract_diagram(proc.processed_path, llm=vlm, forced_diagram_type=diagram_hint)
        return json.dumps(res.model_dump(), indent=2, ensure_ascii=False)

    @tool
    def extract_presentation_pptx(pptx_path: str) -> str:
        """Ekstrak dokumen presentasi PowerPoint (.pptx/.ppt) dengan merender tiap slide menjadi gambar kanvas visual lalu dianalisis oleh VLM."""
        res = process_presentation_vision(pptx_path, llm=vlm)
        return res if isinstance(res, str) else res.full_markdown

    @tool
    def extract_pdf_document(
        pdf_path: str,
        forced_specs: str | None = None,
    ) -> str:
        """Ekstrak dokumen PDF multi-halaman dengan heading continuity, ekstraksi tabel mandiri per-halaman ke SQLite, dan audit guardrail jalur ganda."""
        res = process_multipage_pdf(
            pdf_path,
            llm=vlm,
            forced_specs=forced_specs,
            auto_tabular_db=True,
        )
        return res.full_markdown

    @tool
    def preview_chunks(
        markdown_text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> str:
        """Simulasikan pemecahan teks Markdown hasil ekstraksi menjadi potongan-potongan chunk siap indeks RAG."""
        preview = preview_markdown_chunks(
            markdown_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return json.dumps(preview.model_dump(), indent=2, ensure_ascii=False)

    @tool
    def classify_table_storage(markdown_text: str) -> str:
        """Analisis tabel-tabel pada teks Markdown untuk membedakan mana yang bertipe transaksional/finansial (layak SQLite) vs tabel naratif (layak Vector RAG)."""
        tables = parse_markdown_tables(markdown_text)
        results = []
        for idx, t in enumerate(tables, start=1):
            cls_res = classify_table_heuristic(
                headers=t["headers"],
                rows=t["rows"],
                context=t["context"],
            )
            results.append({"table_index": idx, "classification": cls_res.model_dump()})
        return json.dumps(results, indent=2, ensure_ascii=False)

    @tool
    def ingest_tables_to_sqlite(
        markdown_text: str,
        db_path: str | None = None,
        table_name_prefix: str | None = None,
        force_all: bool = False,
    ) -> str:
        """Ekstrak tabel-tabel transaksional dari teks Markdown, buat skema otomatis, ingest ke database SQLite, dan lakukan verifikasi ganda."""
        results = extract_and_ingest_tables_from_markdown(
            markdown_text=markdown_text,
            db_path=db_path,
            table_name_prefix=table_name_prefix,
            force_all_tables=force_all,
            llm=vlm,
        )
        return json.dumps([r.model_dump() for r in results], indent=2, ensure_ascii=False)

    @tool
    def inspect_sqlite_tables(db_path: str | None = None) -> str:
        """Inspeksi database SQLite dokumen untuk melihat daftar seluruh tabel aktif, struktur skema kolom, jumlah baris, dan sampel data."""
        mgr = TabularDatabaseManager(db_path)
        info = mgr.inspect_database()
        return json.dumps(info, indent=2, ensure_ascii=False)

    @tool
    def query_sqlite_database(query: str, db_path: str | None = None) -> str:
        """Eksekusi query SELECT analitik SQL pada database SQLite dokumen secara aman."""
        res = query_sqlite(query, db_path=db_path)
        return json.dumps(res.model_dump(), indent=2, ensure_ascii=False)

    @tool
    def verify_table_data_integrity(
        table_name: str,
        expected_rows: int | None = None,
        db_path: str | Path | None = None,
    ) -> str:
        """Lakukan audit verifikasi ganda (double-verification) pada tabel SQLite (integritas baris, skema, agregasi SUM/AVG, dan kontinuitas saldo transaksi)."""
        mgr = TabularDatabaseManager(db_path)
        verifier = TabularVerifier(mgr, llm=vlm)
        report = verifier.verify_table(table_name, expected_row_count=expected_rows)
        return json.dumps(report.model_dump(), indent=2, ensure_ascii=False)

    tools = [
        classify_layout,
        extract_to_markdown,
        classify_diagram_suitability,
        extract_diagram_to_mermaid,
        extract_presentation_pptx,
        extract_pdf_document,
        preview_chunks,
        classify_table_storage,
        ingest_tables_to_sqlite,
        inspect_sqlite_tables,
        query_sqlite_database,
        verify_table_data_integrity,
    ]

    # --- Sub-Agent Definitions ---
    subagents = [
        SubAgent(
            name="layout-classifier",
            description="Sub-agent untuk mengidentifikasi dan mengklasifikasikan karakteristik layout dokumen.",
            system_prompt=(
                "Anda adalah Sub-Agent Spesialis Klasifikasi Dokumen. "
                "Tugas Anda: Analisis citra dan identifikasi seluruh karakteristik dokumen (plain, hierarchy, journal, slides). "
                "Gunakan tool 'classify_layout' untuk menentukan spesifikasi."
            ),
            tools=[classify_layout],
        ),
        SubAgent(
            name="markdown-extractor",
            description="Sub-agent untuk mengekstrak citra halaman dokumen menjadi teks Markdown bersih siap chunking.",
            system_prompt=(
                "Anda adalah Sub-Agent Spesialis Ekstraksi Markdown. "
                "Tugas Anda: Ubah citra dokumen menjadi teks Markdown bersih siap chunking RAG. "
                "Pertahankan hierarki heading, list, dan konteks antar-halaman."
            ),
            tools=[extract_to_markdown],
        ),
        SubAgent(
            name="diagram-mermaid-specialist",
            description="Sub-agent untuk mengevaluasi kelayakan diagram dan mengekstraknya menjadi kode Mermaid.js atau deskripsi.",
            system_prompt=(
                "Anda adalah Sub-Agent Spesialis Diagram & Visual Artifacts. "
                "Tugas Anda: Evaluasi selektif citra diagram (flowchart, sequence, ERD, mindmap, block architecture, class diagram) "
                "dan konversi ke blok kode Mermaid.js yang valid (```mermaid) menggunakan tool 'extract_diagram_to_mermaid'. "
                "Jika diagram tidak cocok untuk Mermaid (grafik statistik kontinu, peta spasial, foto, skematik mikro), berikan deskripsi terstruktur."
            ),
            tools=[classify_diagram_suitability, extract_diagram_to_mermaid],
        ),
        SubAgent(
            name="presentation-specialist",
            description="Sub-agent untuk memproses presentasi PowerPoint (.pptx/.ppt) secara visual per slide.",
            system_prompt=(
                "Anda adalah Sub-Agent Spesialis Presentasi PowerPoint (.pptx / .ppt). "
                "Tugas Anda: Render tiap slide menjadi gambar kanvas visual lalu ekstrak teks judul, poin peluru, diagram, dan tabel secara terstruktur."
            ),
            tools=[extract_presentation_pptx],
        ),
        SubAgent(
            name="pdf-orchestrator",
            description="Sub-agent untuk orkestrasi pemrosesan multi-halaman PDF dengan heading continuity.",
            system_prompt=(
                "Anda adalah Sub-Agent Spesialis Dokumen PDF Multi-Halaman. "
                "Tugas Anda: Proses PDF halaman demi halaman secara berurutan, jaga kesinambungan heading antar-halaman, dan gabungkan hasilnya."
            ),
            tools=[extract_pdf_document],
        ),
        SubAgent(
            name="chunking-simulator",
            description="Sub-agent untuk mensimulasikan partisi teks Markdown menjadi potongan chunk RAG.",
            system_prompt=(
                "Anda adalah Sub-Agent Evaluator Chunking RAG. "
                "Tugas Anda: Simulasikan pemecahan teks Markdown hasil ekstraksi menjadi chunk-chunk terstruktur dan laporkan statistiknya."
            ),
            tools=[preview_chunks],
        ),
        SubAgent(
            name="tabular-db-specialist",
            description="Sub-agent untuk deteksi tabel transaksional, ingesti ke SQLite, double-verification, dan eksekusi query SQL.",
            system_prompt=(
                "Anda adalah Sub-Agent Spesialis Tabular SQLite Database & Data Integrity Auditor. \n"
                "Tugas Anda:\n"
                "1. Analisis tabel pada Markdown dan pilah mana yang bertipe transaksional/finansial ('classify_table_storage').\n"
                "2. Ingest tabel transaksional ke database SQLite dokumen ('ingest_tables_to_sqlite').\n"
                "3. Lakukan audit verifikasi ganda integritas baris dan kalkulasi agregat ('verify_table_data_integrity').\n"
                "4. Lakukan inspeksi skema ('inspect_sqlite_tables') dan eksekusi query SQL bila diminta ('query_sqlite_database')."
            ),
            tools=[
                classify_table_storage,
                ingest_tables_to_sqlite,
                inspect_sqlite_tables,
                query_sqlite_database,
                verify_table_data_integrity,
            ],
        ),
    ]

    master_system_prompt = (
        "Anda adalah Master Orchestrator Deep Reasoning Agent untuk Sistem Ekstraksi Dokumen Vision VLM -> Markdown Siap Chunking, Tabular Database SQLite, & Diagram Mermaid.js.\n\n"
        "Anda mengorkestrasi 7 Sub-Agent spesialis:\n"
        "  - 'layout-classifier'         : Menentukan tipe dokumen & karakteristik komposit.\n"
        "  - 'markdown-extractor'        : Mengonversi halaman menjadi Markdown bersih.\n"
        "  - 'diagram-mermaid-specialist': Menangani diagram alur/relasi visual menjadi sintaks Mermaid.js.\n"
        "  - 'presentation-specialist'   : Menangani slide PPT/PPTX visual.\n"
        "  - 'pdf-orchestrator'          : Mengelola multi-halaman PDF dengan heading continuity.\n"
        "  - 'chunking-simulator'        : Mensimulasikan pemotongan chunk siap RAG.\n"
        "  - 'tabular-db-specialist'     : Memisahkan tabel transaksional ke SQLite dan melakukan double-verification.\n\n"
        "Instruksi Kerja:\n"
        "1. Identifikasi format dokumen masukan (PDF, PPTX, gambar tunggal).\n"
        "2. Delegasikan tugas ke sub-agent yang relevan.\n"
        "3. Jika ada diagram visual, evaluasi kelayakannya dan ekstrak kode Mermaid.js.\n"
        "4. Jika ada tabel transaksional, pisahkan ke basis data SQLite dan lakukan audit verifikasi ganda.\n"
        "5. Sajikan hasil ekstraksi akhir yang rapi, lengkap dengan laporan database SQLite dan blok kode Mermaid bila ada."
    )

    agent = create_deep_agent(
        model=vlm,
        tools=tools,
        subagents=subagents,
        system_prompt=master_system_prompt,
    )
    return agent


def run_deep_reasoning_agent(prompt: str, settings: Settings | None = None) -> str:
    """Eksekusi Deep Reasoning Agent dengan instruksi prompt pengguna."""
    agent = build_deep_agent(settings)
    res = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    return res.get("messages", [])[-1].content if res.get("messages") else ""
