"""
Harness Deep Reasoning Agents untuk Ekstraksi Dokumen VLM/OCR -> Markdown Siap Chunking, Tabular SQLite Ingestion, & Diagram Mermaid.js.
Menggunakan arsitektur Master Orchestrator dengan 8 Sub-Agent terspesialisasi.
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
from .ocr import build_ocr_extractor
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
    Bangun Deep Reasoning Agent utama dengan armada 8 Sub-Agent spesialis:
      1. `ocr-specialist`             : Membaca teks mentah literal (ocr-lighton)
      2. `layout-classifier`          : Mengklasifikasikan multi-trait dokumen
      3. `markdown-extractor`         : Ekstraksi VLM multimodal ke Markdown
      4. `diagram-mermaid-specialist` : Evaluasi selektif & ekstraksi diagram ke sintaks Mermaid.js
      5. `presentation-specialist`    : Parsing file presentasi PowerPoint (.pptx / .ppt)
      6. `pdf-orchestrator`           : Orkestrasi multi-halaman PDF & heading continuity
      7. `chunking-simulator`         : Simulasi partisi teks Markdown siap RAG
      8. `tabular-db-specialist`      : Deteksi tabel transaksional, ingesti ke SQLite, double-verification, & eksekusi SQL
    """
    resolved_settings = settings or get_settings()
    vlm = build_vlm(resolved_settings)
    ocr = build_ocr_extractor(resolved_settings)
    extractor = VisionExtractor(vlm)

    # --- Tool Definitions ---

    @tool
    def ocr_document(image_path: str) -> str:
        """OCR dokumen untuk mendapatkan teks mentah beresolusi tinggi tanpa halusinasi."""
        proc = preprocess_image(image_path)
        return ocr.extract(proc.processed_path).text

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
        res = run_extract_diagram(
            proc.processed_path, llm=vlm, forced_diagram_type=diagram_hint
        )
        return json.dumps(res.model_dump(), indent=2, ensure_ascii=False)

    @tool
    def extract_presentation_pptx(pptx_path: str) -> str:
        """Render setiap slide PowerPoint (.pptx / .ppt) menjadi gambar, kirim tiap gambar ke VLM, lalu gabungkan menjadi satu Markdown."""
        from .graph import DocumentExtractionPipeline

        pipeline = DocumentExtractionPipeline(resolved_settings)
        return process_presentation_vision(
            pptx_path=pptx_path,
            pipeline=pipeline,
            forced_specs="presentation_slides",
        )

    @tool
    def extract_pdf_document(
        pdf_path: str,
        specs: str | None = None,
        dpi: int = 200,
    ) -> str:
        """Ekstrak seluruh halaman dokumen PDF menjadi satu dokumen Markdown utuh dengan kontinuitas heading antar halaman."""
        from .graph import DocumentExtractionPipeline

        pipeline = DocumentExtractionPipeline(resolved_settings)
        extracted = process_multipage_pdf(
            pdf_path=pdf_path,
            pipeline=pipeline,
            dpi=dpi,
            forced_specs=specs,
        )
        return extracted.markdown_content

    @tool
    def preview_chunks(
        markdown_text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ) -> str:
        """Simulasikan pemecahan dokumen Markdown dengan splitter berbasis header (#, ##, ###) & recursive splitter."""
        chunks = preview_markdown_chunks(
            markdown_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return json.dumps(chunks, indent=2, ensure_ascii=False)

    # --- Tabular SQLite Tools ---

    @tool
    def classify_table_storage(markdown_text: str) -> str:
        """Deteksi dan klasifikasikan tabel-tabel di dalam dokumen: apakah transaksional (harus disimpan ke SQLite untuk kalkulasi SUM/AVG/Filter) atau tabel naratif (bisa di-chunking ke Vector RAG)."""
        tables = parse_markdown_tables(markdown_text)
        if not tables:
            return json.dumps(
                {"status": "no_tables_found", "tables": []},
                indent=2,
                ensure_ascii=False,
            )

        results = []
        for t in tables:
            cls_res = classify_table_heuristic(
                t["headers"], t["rows"], context=t.get("context", "")
            )
            results.append(cls_res.model_dump())

        return json.dumps(
            {"total_tables": len(tables), "classifications": results},
            indent=2,
            ensure_ascii=False,
        )

    @tool
    def ingest_table_to_sqlite(
        markdown_text: str,
        source_file: str = "",
        db_path: str | None = None,
        table_name: str | None = None,
    ) -> str:
        """Ekstrak tabel transaksional dari Markdown dokumen, ubah menjadi skema SQL bersih, simpan ke database SQLite, dan jalankan double-verification otomatis."""
        ingest_results = extract_and_ingest_tables_from_markdown(
            markdown_text=markdown_text,
            source_file=source_file,
            db_path=db_path,
            table_name_prefix=table_name,
            llm=vlm,
        )
        serialized = [r.model_dump() for r in ingest_results]
        return json.dumps(
            {"total_tables_ingested": len(ingest_results), "results": serialized},
            indent=2,
            ensure_ascii=False,
        )

    @tool
    def verify_sqlite_table(
        table_name: str,
        db_path: str | None = None,
        sample_markdown: str | None = None,
    ) -> str:
        """Lakukan verifikasi ganda (double-verification) terhadap tabel di SQLite: cek integritas baris, validitas tipe data numerik, tes kalkulasi agregat (SUM/AVG), kontinuitas saldo, dan catatan refleksi."""
        db_mgr = TabularDatabaseManager(db_path)
        verifier = TabularVerifier(db_mgr, llm=vlm)
        report = verifier.verify_table(
            table_name, source_markdown_sample=sample_markdown
        )
        return json.dumps(report.model_dump(), indent=2, ensure_ascii=False)

    @tool
    def query_tabular_database(
        sql_query: str,
        db_path: str | None = None,
    ) -> str:
        """Jalankan query SQL (SELECT, SUM, AVG, COUNT, date filter) pada database SQLite dokumen untuk mendapatkan hasil kalkulasi berpresisi 100%."""
        res = query_sqlite(sql_query, db_path=db_path)
        return json.dumps(res.model_dump(), indent=2, ensure_ascii=False)

    @tool
    def inspect_tabular_database(db_path: str | None = None) -> str:
        """Periksa daftar tabel, skema kolom, jumlah baris, dan contoh record pada database SQLite dokumen."""
        db_mgr = TabularDatabaseManager(db_path)
        info = db_mgr.inspect_database()
        return json.dumps(info, indent=2, ensure_ascii=False)

    # --- Sub-Agent Definitions ---

    ocr_subagent: SubAgent = {
        "name": "ocr-specialist",
        "description": "Spesialis OCR pembacaan teks mentah literal (angka, simbol, istilah teknis, tabel mentah).",
        "system_prompt": (
            "Kamu adalah spesialis OCR beresolusi tinggi. "
            "Panggil tool ocr_document untuk mendapatkan seluruh teks mentah dari gambar tanpa mengubah arti."
        ),
        "tools": [ocr_document],
    }

    classifier_subagent: SubAgent = {
        "name": "layout-classifier",
        "description": "Spesialis analisis dan klasifikasi tata letak dokumen (mendeteksi 2-kolom, hierarki bab, slide, dsb).",
        "system_prompt": (
            "Kamu adalah analis tata letak dokumen. "
            "Panggil tool classify_layout untuk mendeteksi seluruh karakteristik spesifikasi yang ada pada gambar dokumen."
        ),
        "tools": [classify_layout],
    }

    markdown_subagent: SubAgent = {
        "name": "markdown-extractor",
        "description": "Spesialis ekstraksi gambar menjadi Markdown bersih terstruktur dengan panduan spesifikasi komposit.",
        "system_prompt": (
            "Kamu adalah ahli ekstraksi Markdown dokumen. "
            "Panggil tool extract_to_markdown dengan path gambar dan spesifikasi tata letak yang sesuai."
        ),
        "tools": [extract_to_markdown],
    }

    diagram_subagent: SubAgent = {
        "name": "diagram-mermaid-specialist",
        "description": (
            "Spesialis evaluasi dan ekstraksi visual diagram: memutuskan secara selektif apakah diagram pada dokumen "
            "cocok dikonversi ke kode Mermaid.js (flowchart, sequence, ERD, state machine, class UML, mindmap, block architecture) "
            "atau menyajikannya dalam bentuk deskripsi struktural/tabel jika tidak cocok (grafik statistik, peta, foto)."
        ),
        "system_prompt": (
            "Kamu adalah spesialis visualisasi diagram & sintaks Mermaid.js.\n"
            "Tugasmu:\n"
            "1. Panggil 'classify_diagram_suitability' untuk mengevaluasi apakah diagram pada dokumen cocok diubah ke Mermaid.\n"
            "2. Untuk diagram yang cocok (flowchart, sequence, ERD, state diagram, class diagram, mindmap, block architecture): "
            "panggil 'extract_diagram_to_mermaid' untuk menghasilkan kode Mermaid yang valid secara sintaks.\n"
            "3. Untuk diagram yang tidak cocok (grafik scatter/line statistik numerik padat, peta geografis, foto, skematik sirkuit mikro, infografis seni bebas): "
            "jangan paksakan ke Mermaid, berikan representasi tabel Markdown atau deskripsi struktural yang tepat."
        ),
        "tools": [classify_diagram_suitability, extract_diagram_to_mermaid],
    }

    ppt_subagent: SubAgent = {
        "name": "presentation-specialist",
        "description": "Spesialis ekstraksi presentasi PowerPoint (.pptx / .ppt) menjadi Markdown terstruktur per-slide.",
        "system_prompt": (
            "Kamu adalah spesialis presentasi. "
            "Panggil tool extract_presentation_pptx untuk merender seluruh slide menjadi gambar, membaca tiap gambar dengan VLM, dan menggabungkan hasilnya."
        ),
        "tools": [extract_presentation_pptx],
    }

    pdf_subagent: SubAgent = {
        "name": "pdf-orchestrator",
        "description": "Spesialis pemrosesan dokumen multi-halaman PDF dengan kontinuitas heading antar-halaman.",
        "system_prompt": (
            "Kamu adalah orkestrator PDF multi-halaman. "
            "Panggil tool extract_pdf_document untuk memproses seluruh halaman PDF menjadi Markdown utuh."
        ),
        "tools": [extract_pdf_document],
    }

    chunker_subagent: SubAgent = {
        "name": "chunking-simulator",
        "description": "Spesialis evaluasi dan simulasi pemotongan teks Markdown siap chunking untuk RAG.",
        "system_prompt": (
            "Kamu adalah spesialis chunking. "
            "Panggil tool preview_chunks untuk memverifikasi kesiapan teks Markdown dipartisi menjadi chunks."
        ),
        "tools": [preview_chunks],
    }

    tabular_subagent: SubAgent = {
        "name": "tabular-db-specialist",
        "description": "Spesialis pemrosesan data tabular transaksional (rekening koran, log keuangan, ledger), ingesti ke SQLite, double-verification integritas data, dan eksekusi query SQL agregasi.",
        "system_prompt": (
            "Kamu adalah spesialis data tabular & database SQL.\n"
            "Tugasmu:\n"
            "1. Panggil 'classify_table_storage' untuk mendeteksi apakah tabel memerlukan penyimpanan database SQLite atau Vector RAG.\n"
            "2. Untuk tabel transaksional, panggil 'ingest_table_to_sqlite' untuk menyimpan data ke SQLite dan melakukan double-verification.\n"
            "3. Jika diperlukan pengujian atau query kalkulasi agregat (SUM, AVG, COUNT, filter tanggal), panggil 'query_tabular_database' atau 'verify_sqlite_table'."
        ),
        "tools": [
            classify_table_storage,
            ingest_table_to_sqlite,
            verify_sqlite_table,
            query_tabular_database,
            inspect_tabular_database,
        ],
    }

    all_tools = [
        ocr_document,
        classify_layout,
        extract_to_markdown,
        classify_diagram_suitability,
        extract_diagram_to_mermaid,
        extract_presentation_pptx,
        extract_pdf_document,
        preview_chunks,
        classify_table_storage,
        ingest_table_to_sqlite,
        verify_sqlite_table,
        query_tabular_database,
        inspect_tabular_database,
    ]

    all_subagents = [
        ocr_subagent,
        classifier_subagent,
        markdown_subagent,
        diagram_subagent,
        ppt_subagent,
        pdf_subagent,
        chunker_subagent,
        tabular_subagent,
    ]

    return create_deep_agent(
        name="document-vision-deep-reasoning-agent",
        model=vlm,
        tools=all_tools,
        system_prompt=(
            "Kamu adalah Master Deep Reasoning Agent untuk ekstraksi dokumen multi-modal, diagram visual, dan pemrosesan data tabular terstruktur.\n"
            "Tugasmu: Menganalisis file dokumen pengguna (PDF, PPTX, Scan, Gambar), menghasilkan teks Markdown bersih siap chunking, mengekstrak diagram visual ke kode Mermaid yang valid, dan memisahkan tabel data transaksional ke database SQLite.\n\n"
            "Strategi Eksekusi Otonom:\n"
            "1. Jika file berformat .pptx / .ppt: Delegasikan ke 'presentation-specialist'.\n"
            "2. Jika file berformat .pdf multi-halaman: Delegasikan ke 'pdf-orchestrator'.\n"
            "3. Jika file berupa gambar (PNG/JPG/WEBP/Scan):\n"
            "   a. Panggil 'ocr-specialist' untuk teks mentah presisi tinggi.\n"
            "   b. Panggil 'layout-classifier' untuk menentukan spesifikasi layout.\n"
            "   c. Panggil 'markdown-extractor' untuk menyusun teks Markdown utuh.\n"
            "4. Jika dokumen/gambar mengandung diagram visual, bagan alur, sequence, ERD, arsitektur blok, state machine, atau mindmap:\n"
            "   Delegasikan ke 'diagram-mermaid-specialist' untuk mengevaluasi kelayakan konversi ke kode Mermaid dan mengekstraknya secara terstruktur.\n"
            "5. Jika dokumen berisi tabel transaksional / log mutasi / rekening koran / laporan keuangan numerik yang memerlukan kalkulasi agregat (SUM, AVG, COUNT, date filter):\n"
            "   Delegasikan ke 'tabular-db-specialist' untuk menyimpan tabel ke SQLite dan memverifikasi integritasnya.\n"
            "6. Jika pengguna meminta simulasi chunking: Panggil 'chunking-simulator'.\n"
            "7. Kembalikan teks Markdown dokumen dan laporkan status diagram Mermaid serta tabel database bila ada."
        ),
        subagents=all_subagents,
    )


def run_deep_reasoning_agent(
    file_path: str | Path,
    *,
    forced_specs: str | None = None,
    preview_chunks: bool = False,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    ingest_tables_to_db: bool = True,
    extract_diagrams_to_mermaid: bool = True,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """
    Jalankan eksekusi dokumen secara otomatis menggunakan Deep Reasoning Agent & armada Sub-Agents.
    """
    resolved_settings = settings or get_settings()
    path_obj = Path(file_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"File dokumen tidak ditemukan: {path_obj}")

    agent = build_deep_agent(resolved_settings)

    prompt_parts = [
        f"Ekstrak dan proses file dokumen berikut secara lengkap: '{path_obj}'.",
        "Lakukan penalaran (reasoning) tata letak dan delegasikan ke sub-agent spesialis yang sesuai.",
    ]
    if forced_specs:
        prompt_parts.append(f"Gunakan spesifikasi layout: {forced_specs}.")
    if preview_chunks:
        prompt_parts.append(
            f"Lakukan simulasi preview chunking (chunk_size={chunk_size}, overlap={chunk_overlap})."
        )
    if ingest_tables_to_db:
        prompt_parts.append(
            "Jika dokumen mengandung tabel transaksional/keuangan/log mutasi, simpan tabel tersebut ke database SQLite dan lakukan verifikasi integritas data."
        )
    if extract_diagrams_to_mermaid:
        prompt_parts.append(
            "Jika dokumen mengandung diagram alur/proses/relasi yang cocok untuk Mermaid, ekstrak menjadi kode Mermaid yang valid."
        )
    prompt_parts.append("Pastikan output akhir terstruktur rapi.")

    user_prompt = " ".join(prompt_parts)
    resp = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})

    messages = resp.get("messages", [])
    final_text = messages[-1].content if messages else str(resp)

    return {
        "file_path": str(path_obj),
        "final_output": final_text,
        "messages": messages,
    }
