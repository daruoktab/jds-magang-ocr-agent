"""
CLI Document VLM Text Extractor (Ready for Chunking).

Secara default, mengeksekusi ekstraksi dokumen via Vision Language Model (VLM):
  - File PPTX / PPT   : Dirender otomatis menjadi gambar kanvas per slide dan dikirim ke VLM (default), atau via `--ppt-native` untuk parser cepat tanpa VLM.
  - File Gambar / PDF : Diekstrak via pipeline VLM / Deep Reasoning Agent.
  - Data Tabular / DB : Sub-Agent SQL aktif mandiri per-halaman/slide untuk memahami, menginspeksi, meng-ingest tabel ke SQLite (`output/databases/{nama_dokumen}.sqlite`), serta diakhiri Guardrail Cross-Verification oleh Agent Pusat.

Contoh Penggunaan:
    python main.py input/presentasi.pptx -o output/ppt01.md            # Ekstrak PPTX via gambar slide -> Model Vision (VLM)
    python main.py input/presentasi.pptx --ppt-native -o output/ppt01.md # Ekstrak PPTX native (cepat, tanpa VLM)
    python main.py dokumen.pdf -o output/dokumen.md                    # Ekstrak PDF multi-halaman via Dual-Track
    python main.py scan.jpg --debug                                    # Ekstrak gambar dengan log lengkap
    python main.py dokumen.pdf --force-all-tables                      # Ingest seluruh tabel ke SQLite
    python main.py --scan-folders dataset                              # Pindai folder-folder dokumen
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Pastikan output stream di Windows menggunakan encoding UTF-8
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if callable(_reconfigure):
        _reconfigure(encoding="utf-8", errors="replace")

from app.agents import AGENT_REGISTRY
from app.batch import (
    batch_extract_documents,
    find_document_files,
    scan_document_directories,
)
from app.config import get_settings, setup_logging
from app.deep_agent import build_deep_agent
from app.graph import DocumentExtractionPipeline
from app.multi_page import preview_markdown_chunks
from app.pdf import pdf_to_images, process_multipage_pdf
from app.ppt import process_presentation, process_presentation_vision
from app.tabular_db import cross_verify_dual_track, process_page_tabular_agent

logger = logging.getLogger("app.cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jds-magang",
        description="Ekstraksi dokumen berbasis VLM -> Markdown siap chunking RAG, Tabular SQLite Database, & Mermaid Diagrams.",
    )
    parser.add_argument(
        "document",
        nargs="?",
        default=None,
        help="Path ke file dokumen yang akan diproses (PDF, PPTX, PPT, atau Gambar).",
    )
    parser.add_argument(
        "-t",
        "--doc-type",
        dest="doc_type",
        default=None,
        help="Karakteristik dokumen (plain, markdown_hierarchy, bilingual_journal, presentation_slides, atau komposit misal 'journal,hierarchy').",
    )
    parser.add_argument(
        "-o",
        "--out",
        dest="out",
        default=None,
        help="Path file/folder output Markdown (default: stdout atau output/{nama_file}.md).",
    )
    parser.add_argument(
        "--db-path",
        dest="db_path",
        default=None,
        help="Path file database SQLite target untuk data tabular (default: output/databases/{nama_file}.sqlite).",
    )
    parser.add_argument(
        "--force-all-tables",
        action="store_true",
        help="Paksa seluruh tabel (termasuk tabel matriks/naratif) untuk di-ingest ke basis data SQLite.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Resolusi DPI untuk render PDF / slide PPTX ke gambar (default: 200).",
    )
    parser.add_argument(
        "--preview-chunks",
        action="store_true",
        help="Tampilkan pratinjau statistik pemecahan chunking (char count, token estimate, sample preview).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Ukuran target karakter per chunk (default: 1000).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Ukuran overlap karakter antar chunk (default: 150).",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Jalankan ekstraksi menggunakan Deep Reasoning Multi-Agent Harness (LangGraph StateGraph).",
    )
    parser.add_argument(
        "--ppt-native",
        action="store_true",
        help="Ekstrak slide PPTX/PPT secara native murni teks/tabel tanpa rendering gambar kanvas.",
    )
    parser.add_argument(
        "--vision",
        action="store_true",
        help="Paksa ekstraksi visual berbasis VLM untuk semua tipe dokumen.",
    )
    parser.add_argument(
        "--direct-graph",
        action="store_true",
        help="Jalankan via alur LangGraph StateGraph.",
    )
    parser.add_argument(
        "--classify-only",
        action="store_true",
        help="Hanya jalankan klasifikasi multi-karakteristik tanpa melakukan ekstraksi teks.",
    )
    parser.add_argument(
        "--pdf-split-only",
        action="store_true",
        help="Hanya render halaman PDF menjadi file gambar di folder output.",
    )
    parser.add_argument(
        "--scan-folders",
        metavar="FOLDER",
        help="Pindai direktori FOLDER secara rekursif untuk mendata seluruh file dokumen yang tersedia.",
    )
    parser.add_argument(
        "--batch",
        metavar="FOLDERS_OR_FILES",
        help="Ekstraksi massal dokumen dari daftar folder/file (dipisah koma).",
    )
    parser.add_argument(
        "--list-types",
        action="store_true",
        help="Tampilkan daftar spesifikasi/tipe dokumen yang didukung.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Aktifkan pesan log level DEBUG.",
    )
    parser.add_argument(
        "--log-file",
        dest="log_file",
        default=None,
        help="Path ke file untuk menyimpan log eksekusi secara real-time.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Konfigurasi level logging
    log_level = "DEBUG" if args.debug else "INFO"
    setup_logging(level=log_level, log_file=args.log_file)

    settings = get_settings()

    # Opsi informasional: Daftar tipe dokumen
    if args.list_types:
        print("Spesifikasi Karakteristik Dokumen:")
        for name, agent in AGENT_REGISTRY.items():
            print(f"  - {name:<22}: {agent.description}")
        return 0

    # Opsi utilitas: Pindai folder dokumen
    if args.scan_folders:
        folder_path = Path(args.scan_folders).resolve()
        if not folder_path.exists():
            print(f"ERROR: Folder tidak ditemukan: {folder_path}", file=sys.stderr)
            return 1
        catalog = scan_document_directories(folder_path)
        print(f"Hasil Pemindaian Direktori: {folder_path}")
        total = 0
        for entry in catalog:
            folder_name = entry["relative_path"]
            doc_count = entry["total_documents"]
            ext_counts = entry["extension_counts"]
            samples = entry["sample_files"]
            print(f"\n[{folder_name}] ({doc_count} file) - {ext_counts}:")
            for sample in samples:
                print(f"  - {sample}")
            total += doc_count
        print(f"\nTotal Dokumen Ditemukan: {total}")
        return 0

    # Opsi utilitas: Ekstraksi massal (batch)
    if args.batch:
        raw_items = [s.strip() for s in args.batch.split(",") if s.strip()]
        files_to_process: list[Path] = []
        for item in raw_items:
            p = Path(item).resolve()
            if p.is_dir():
                files_to_process.extend(find_document_files(p))
            elif p.is_file():
                files_to_process.append(p)

        if not files_to_process:
            print(
                f"ERROR: Tidak ditemukan file yang valid dari: {args.batch}",
                file=sys.stderr,
            )
            return 1

        out_dir = Path(args.out).resolve() if args.out else Path("output/batch").resolve()
        batch_extract_documents(
            files_to_process,
            output_dir=out_dir,
            forced_specs=args.doc_type,
            dpi=args.dpi,
            preview_chunks=args.preview_chunks,
            use_agent=args.agent,
            settings=settings,
        )
        return 0

    # Validasi input file tunggal
    if not args.document:
        parser.print_help()
        return 1

    input_path = Path(args.document).resolve()
    if not input_path.exists():
        print(f"ERROR: File tidak ditemukan: {input_path}", file=sys.stderr)
        return 1

    ext = input_path.suffix.lower()

    # Tentukan path target database SQLite
    if args.db_path:
        db_target_file: Path | None = Path(args.db_path).resolve()
    elif args.out:
        db_dir = Path(args.out).parent / "databases"
        db_target_file = db_dir / f"{input_path.stem}.sqlite"
    else:
        db_dir = Path("output/databases").resolve()
        db_target_file = db_dir / f"{input_path.stem}.sqlite"

    try:
        # 3. Mode Render PDF Halaman saja
        if args.pdf_split_only and ext == ".pdf":
            out_pages = pdf_to_images(input_path, dpi=args.dpi)
            for p in out_pages:
                print(str(p))
            return 0

        # 4. Mode Klasifikasi Saja
        if args.classify_only:
            pipeline = DocumentExtractionPipeline(settings)
            specs = pipeline.extractor.classify(str(input_path))
            print(json.dumps({"file": str(input_path), "specs": specs}, indent=2))
            return 0

        # 5. File Presentasi (PPTX/PPT)
        if ext in (".pptx", ".ppt") and not args.agent:
            if args.ppt_native and not args.vision and not args.direct_graph:
                markdown_content = process_presentation(input_path)
            else:
                logger.info(
                    "Mengekstrak presentasi via rendering gambar kanvas per slide -> Dual-Track VLM & Sub-Agent SQL..."
                )
                pipeline = DocumentExtractionPipeline(settings)
                markdown_content = process_presentation_vision(
                    pptx_path=input_path,
                    pipeline=pipeline,
                    forced_specs=args.doc_type or "presentation_slides",
                    db_path=db_target_file,
                    force_all_tables=args.force_all_tables,
                )

        # 6. File PDF Multi-Halaman
        elif ext == ".pdf" and not args.agent:
            logger.info("Mengekstrak PDF multi-halaman via Dual-Track Vision & Sub-Agent SQL...")
            pipeline = DocumentExtractionPipeline(settings)
            doc_result = process_multipage_pdf(
                pdf_path=input_path,
                pipeline=pipeline,
                forced_specs=args.doc_type,
                dpi=args.dpi,
                db_path=db_target_file,
                force_all_tables=args.force_all_tables,
            )
            markdown_content = doc_result.full_markdown

        # 7. File Gambar Tunggal
        elif ext in (".png", ".jpg", ".jpeg", ".webp") and not args.agent:
            logger.info("Mengekstrak gambar via Dual-Track Vision & Sub-Agent SQL...")
            pipeline = DocumentExtractionPipeline(settings)
            result = pipeline.run(str(input_path), forced_specs=args.doc_type)
            markdown_content = result["markdown_content"]

            # Sub-Agent SQL mandiri pada gambar tunggal
            if db_target_file:
                _tab_event, _tab_results = process_page_tabular_agent(
                    page_markdown=markdown_content,
                    page_number=1,
                    source_file=str(input_path),
                    db_path=db_target_file,
                    table_name_prefix=input_path.stem,
                    force_all_tables=args.force_all_tables,
                )
                cross_verify_dual_track(
                    stitched_markdown=markdown_content,
                    db_path=db_target_file,
                    source_file=str(input_path),
                    total_pages=1,
                )

        # 8. Deep Agent Mode
        elif args.agent:
            logger.info("Mengekstrak via Master Deep Reasoning Agent & 7 Sub-Agent Spesialis...")
            deep_agent = build_deep_agent(settings)
            res = deep_agent.invoke({
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Ekstrak dokumen berikut ke Markdown bersih, evaluasi diagram visual jika ada, "
                            f"dan ingest tabel transaksional ke database SQLite '{db_target_file}':\n{input_path}"
                        ),
                    }
                ]
            })
            markdown_content = (
                res.get("messages", [])[-1].content
                if res.get("messages")
                else ""
            )

        else:
            print(f"ERROR: Format file tidak didukung: {ext}", file=sys.stderr)
            return 1

        # Output Markdown hasil ekstraksi
        if args.out:
            out_file = Path(args.out).resolve()
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(markdown_content, encoding="utf-8")
            logger.info("Hasil Markdown berhasil disimpan ke: %s", out_file)
        else:
            print(markdown_content)

        # Simulasi Chunking jika diminta
        if args.preview_chunks:
            chunks = preview_markdown_chunks(
                markdown_content,
                source_file=input_path.name,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
            print("\n" + "=" * 60)
            print("SIMULASI PEMBAGIAN CHUNKING (SIAP RAG):")
            print(f"Total Karakter : {chunks.total_characters}")
            print(f"Total Chunks    : {chunks.total_chunks}")
            print(f"Target Size     : {chunks.chunk_size} char (overlap: {chunks.chunk_overlap})")
            print(f"Rata-rata Size  : {chunks.avg_chunk_size:.1f} char")
            print("=" * 60)
            for c in chunks.chunks[:3]:
                print(f"[Chunk #{c.chunk_id} | {c.char_count} chars | ~{c.token_estimate} tokens]")
                print(f"{c.preview}\n---")

        return 0

    except Exception:
        logger.exception("Terjadi kesalahan saat memproses dokumen.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
