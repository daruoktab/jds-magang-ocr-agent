"""
CLI Document Vision OCR & Text Extractor (Ready for Chunking).

Secara default, mengeksekusi ekstraksi dokumen menggunakan Deep Reasoning Agent
dengan armada 6 Sub-Agent spesialis:
  - `ocr-specialist`           : Grounding teks mentah presisi tinggi (ocr-lighton)
  - `layout-classifier`        : Analisis & klasifikasi multi-spesifikasi layout
  - `markdown-extractor`       : Ekstraksi VLM multimodal dengan composable prompts
  - `presentation-specialist`  : Parser file presentasi PowerPoint (.pptx / .ppt)
  - `pdf-orchestrator`         : Orkestrasi multi-halaman PDF & kontinuitas heading
  - `chunking-simulator`       : Simulasi partisi teks Markdown siap RAG

Contoh Penggunaan:
    python main.py dokumen.pdf                        # Ekstrak PDF otomatis via Deep Reasoning Agent
    python main.py presentasi.pptx                    # Ekstrak PPTX ke Markdown
    python main.py scan.jpg                           # Ekstrak gambar ke Markdown
    python main.py dokumen.pdf -o output.md           # Simpan output ke file Markdown
    python main.py dokumen.pdf --preview-chunks       # Lihat simulasi hasil chunking
    python main.py jurnal.pdf --type journal,hierarchy # Paksa spesifikasi komposit
    python main.py --scan-folders dataset             # Pindai folder-folder yang berisi dokumen
    python main.py --batch-folders dataset/indonesian,dataset/english --limit 10 # Ekstraksi massal
    python main.py --list-types                       # Lihat daftar spesifikasi layout
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.agents import AGENT_REGISTRY
from app.batch import batch_extract_documents, scan_document_directories
from app.config import get_settings
from app.deep_agent import build_deep_agent
from app.graph import DocumentExtractionPipeline
from app.multi_page import preview_markdown_chunks
from app.ocr import build_ocr_extractor
from app.pdf import pdf_to_images, process_multipage_pdf
from app.ppt import process_presentation
from app.prompts import normalize_specs
from app.schemas import ExtractedDocument


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vision-doc-extractor",
        description="Ekstraksi Dokumen Vision OCR -> Markdown Bersih Siap Chunking (Deep Reasoning Agent).",
    )
    p.add_argument("document", nargs="?", help="Path file dokumen (PDF, PPTX, PPT, atau Gambar)")
    p.add_argument("-o", "--out", default=None, help="Path file output .md untuk menyimpan hasil ekstraksi (atau direktori output batch)")
    p.add_argument(
        "-t",
        "--type",
        dest="doc_type",
        default=None,
        help="Paksa spesifikasi tata letak dokumen, bisa komposit dipisah koma (mis. 'journal,hierarchy', 'plain', 'presentation_slides')",
    )
    p.add_argument("--preview-chunks", action="store_true", help="Tampilkan simulasi pemecahan chunk")
    p.add_argument("--chunk-size", type=int, default=1000, help="Ukuran chunk karakter untuk preview (default 1000)")
    p.add_argument("--chunk-overlap", type=int, default=150, help="Overlap chunk karakter (default 150)")
    p.add_argument("--dpi", type=int, default=200, help="DPI render untuk PDF (default 200)")
    p.add_argument("--direct-graph", action="store_true", help="Gunakan eksekusi grafik deterministik langsung (bypass agent reasoning)")
    p.add_argument("--ocr-only", action="store_true", help="Hanya jalankan model OCR tanpa VLM")
    p.add_argument("--classify-only", action="store_true", help="Hanya klasifikasi karakteristik dokumen")
    p.add_argument("--pdf-split-only", action="store_true", help="Hanya render PDF menjadi gambar per-halaman")
    p.add_argument("--list-types", action="store_true", help="Daftar spesifikasi karakteristik dokumen yang didukung")
    
    # Batch & Folder Scanning Options
    p.add_argument("--scan-folders", nargs="?", const=".", help="Pindai direktori untuk mendeteksi folder-folder dokumen")
    p.add_argument("--batch-folders", dest="batch_folders", help="Ekstrak batch dokumen dari folder-folder yang dipisah koma (mis. 'dataset/indonesian,dataset/english')")
    p.add_argument("--limit", type=int, default=None, help="Batas total dokumen yang diproses pada batch")
    p.add_argument("--limit-per-folder", type=int, default=None, help="Batas dokumen per-folder pada batch")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_types:
        print("Spesifikasi Tata Letak & Kemampuan Ekstraksi Dokumen (Dapat Dikombinasikan):")
        for name, agent in AGENT_REGISTRY.items():
            print(f"- {name:22s} : {agent.description}")
        return 0

    settings = get_settings()

    # 1. Mode Pemindaian Folder Dokumen (--scan-folders)
    if args.scan_folders is not None:
        scan_root = args.scan_folders if args.scan_folders else "."
        try:
            folders = scan_document_directories(scan_root)
            print(f"Hasil Pemindaian Direktori Dokumen di '{Path(scan_root).resolve()}':")
            print(f"Ditemukan {len(folders)} folder berisi file dokumen:\n")
            for idx, f_info in enumerate(folders, 1):
                ext_str = ", ".join(f"{k}: {v}" for k, v in f_info["extension_counts"].items())
                print(f"[{idx}] {f_info['relative_path']}")
                print(f"    Path Penuh: {f_info['folder_path']}")
                print(f"    Total Dokumen: {f_info['total_documents']} ({ext_str})")
                print(f"    Contoh File: {', '.join(f_info['sample_files'][:3])}\n")
            return 0
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: {e}", file=sys.stderr)
            return 1

    # 2. Mode Batch Ekstraksi dari Folder Terpilih (--batch-folders)
    if args.batch_folders:
        out_target = args.out or "output/extracted_md"
        print(f"Menjalankan Batch Ekstraksi Dokumen dari: {args.batch_folders}")
        print(f"Kuota: total_limit={args.limit}, limit_per_folder={args.limit_per_folder}")
        print(f"Folder Output: {out_target}\n")
        
        batch_res = batch_extract_documents(
            folders=args.batch_folders,
            limit=args.limit,
            limit_per_folder=args.limit_per_folder,
            specs=args.doc_type or "plain",
            output_dir=out_target,
            preview_chunks=args.preview_chunks,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            settings=settings,
        )

        print(json.dumps(batch_res, indent=2, ensure_ascii=False))
        return 0

    if not args.document:
        print("ERROR: argumen file 'document' wajib diisi (atau gunakan --scan-folders / --batch-folders)", file=sys.stderr)
        return 1

    input_path = Path(args.document)
    if not input_path.exists():
        print(f"ERROR: File tidak ditemukan: {input_path}", file=sys.stderr)
        return 1

    ext = input_path.suffix.lower()

    try:
        # 3. Mode Render PDF Halaman saja
        if args.pdf_split_only and ext == ".pdf":
            out_pages = pdf_to_images(input_path, dpi=args.dpi)
            for p in out_pages:
                print(str(p))
            return 0

        # 4. Mode OCR Teks Mentah saja
        if args.ocr_only:
            ocr = build_ocr_extractor(settings)
            if ext == ".pdf":
                pages = pdf_to_images(input_path, dpi=args.dpi)
                full_ocr = []
                for idx, pg in enumerate(pages, 1):
                    full_ocr.append(f"--- Halaman {idx} ---\n{ocr.extract(str(pg)).text}")
                res_text = "\n\n".join(full_ocr)
            else:
                res_text = ocr.extract(str(input_path)).text

            print(res_text)
            return 0

        # 5. Mode Klasifikasi Saja
        if args.classify_only:
            pipeline = DocumentExtractionPipeline(settings)
            specs = pipeline.extractor.classify(str(input_path))
            print(json.dumps({"file": str(input_path), "specs": specs}, indent=2))
            return 0

        # 6. Mode Eksekusi Utama: Otomatis via Deep Reasoning Agent & Subagents
        if not args.direct_graph:
            deep_agent = build_deep_agent(settings)
            instruction_parts = [
                f"Tolong proses dan ekstrak file dokumen berikut secara lengkap: '{input_path.resolve()}'.",
                "Analisis tata letak dan delegasikan ke sub-agent spesialis yang sesuai.",
            ]
            if args.doc_type:
                instruction_parts.append(f"Spesifikasi tata letak dokumen yang dipaksakan: {args.doc_type}.")
            if args.preview_chunks:
                instruction_parts.append(f"Sertakan simulasi preview chunking (chunk_size={args.chunk_size}, overlap={args.chunk_overlap}).")
            instruction_parts.append("Pastikan hasil akhir berupa teks Markdown bersih siap chunking.")

            user_instruction = " ".join(instruction_parts)
            resp = deep_agent.invoke({
                "messages": [{"role": "user", "content": user_instruction}]
            })

            messages = resp.get("messages", [])
            markdown_content = messages[-1].content if messages else str(resp)

        # 7. Mode Fallback: Eksekusi LangGraph Deterministik Langsung (--direct-graph)
        else:
            if ext in (".pptx", ".ppt"):
                markdown_content = process_presentation(input_path)
            elif ext == ".pdf":
                pipeline = DocumentExtractionPipeline(settings)
                extracted_doc = process_multipage_pdf(
                    pdf_path=input_path,
                    pipeline=pipeline,
                    dpi=args.dpi,
                    forced_specs=args.doc_type,
                )
                markdown_content = extracted_doc.markdown_content
            else:
                pipeline = DocumentExtractionPipeline(settings)
                res = pipeline.run(str(input_path), forced_specs=args.doc_type)
                markdown_content = res["markdown_content"]

        # Output Teks Markdown
        print(markdown_content)

        # Simpan ke file jika diminta (-o / --out)
        if args.out:
            out_file = Path(args.out)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(markdown_content, encoding="utf-8")
            print(f"\n[OK] Dokumen Markdown berhasil disimpan ke: {out_file}", file=sys.stderr)

        # Preview Chunks jika diminta (pada mode direct-graph atau ekstra tampilan)
        if args.preview_chunks and args.direct_graph:
            chunks = preview_markdown_chunks(
                markdown_content,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
            print("\n" + "=" * 60, file=sys.stderr)
            print(f"--- PREVIEW CHUNKING ({len(chunks)} Potongan Chunk) ---", file=sys.stderr)
            print("=" * 60, file=sys.stderr)
            for ch in chunks:
                print(f"\n[Chunk #{ch['chunk_index']} | {ch['char_count']} chars | Meta: {ch['metadata']}]", file=sys.stderr)
                print(ch["content"], file=sys.stderr)
                print("-" * 40, file=sys.stderr)

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
