"""
CLI Document Vision OCR & Text Extractor (Ready for Chunking).

Ekstraksi dokumen (PDF, PPT/PPTX, Gambar) menjadi teks Markdown bersih
yang siap langsung di-chunking.

Contoh Penggunaan:
    python main.py dokumen.pdf                        # Ekstrak PDF ke Markdown
    python main.py presentasi.pptx                    # Ekstrak PPTX ke Markdown
    python main.py scan.jpg                           # Ekstrak gambar ke Markdown
    python main.py dokumen.pdf -o output.md           # Simpan output ke file Markdown
    python main.py dokumen.pdf --preview-chunks       # Lihat simulasi hasil chunking
    python main.py jurnal.pdf --type bilingual_journal # Paksa mode jurnal 2-kolom
    python main.py --list-types                       # Lihat daftar spesifikasi layout
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.agents import AGENT_REGISTRY
from app.config import get_settings
from app.graph import DocumentExtractionPipeline
from app.multi_page import preview_markdown_chunks
from app.ocr import build_ocr_extractor
from app.pdf import pdf_to_images, process_multipage_pdf
from app.ppt import process_presentation
from app.schemas import ExtractedDocument


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vision-doc-extractor",
        description="Ekstraksi Dokumen Vision OCR -> Markdown Bersih Siap Chunking.",
    )
    p.add_argument("document", nargs="?", help="Path file dokumen (PDF, PPTX, PPT, atau Gambar)")
    p.add_argument("-o", "--out", default=None, help="Path file output .md untuk menyimpan hasil ekstraksi")
    p.add_argument(
        "-t",
        "--type",
        dest="doc_type",
        choices=["plain", "markdown_hierarchy", "bilingual_journal", "presentation_slides"],
        default=None,
        help="Paksa spesifikasi tata letak dokumen (opsional, default auto-detect)",
    )
    p.add_argument("--preview-chunks", action="store_true", help="Tampilkan simulasi pemecahan chunk")
    p.add_argument("--chunk-size", type=int, default=1000, help="Ukuran chunk karakter untuk preview (default 1000)")
    p.add_argument("--chunk-overlap", type=int, default=150, help="Overlap chunk karakter (default 150)")
    p.add_argument("--dpi", type=int, default=200, help="DPI render untuk PDF (default 200)")
    p.add_argument("--ocr-only", action="store_true", help="Hanya jalankan model OCR tanpa VLM")
    p.add_argument("--classify-only", action="store_true", help="Hanya klasifikasi karakteristik dokumen")
    p.add_argument("--pdf-split-only", action="store_true", help="Hanya render PDF menjadi gambar per-halaman")
    p.add_argument("--list-types", action="store_true", help="Daftar spesifikasi karakteristik dokumen yang didukung")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_types:
        print("Spesifikasi Tata Letak & Kemampuan Ekstraksi Dokumen:")
        for name, agent in AGENT_REGISTRY.items():
            print(f"- {name:22s} : {agent.description}")
        return 0

    if not args.document:
        print("ERROR: argumen file 'document' wajib diisi (atau --list-types)", file=sys.stderr)
        return 1

    input_path = Path(args.document)
    if not input_path.exists():
        print(f"ERROR: File tidak ditemukan: {input_path}", file=sys.stderr)
        return 1

    ext = input_path.suffix.lower()
    settings = get_settings()

    try:
        # 1. Mode Render PDF Halaman saja
        if args.pdf_split_only and ext == ".pdf":
            out_pages = pdf_to_images(input_path, dpi=args.dpi)
            for p in out_pages:
                print(str(p))
            return 0

        # 2. Mode OCR Teks Mentah saja
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

        # 3. File PowerPoint Presentasi (.pptx / .ppt)
        if ext in (".pptx", ".ppt"):
            markdown_content = process_presentation(input_path)
            extracted_doc = ExtractedDocument(
                file_path=str(input_path),
                doc_type="presentation_slides",
                total_pages=1,
                markdown_content=markdown_content,
                metadata={"source_type": "presentation", "filename": input_path.name},
            )

        # 4. File PDF Multi-halaman
        elif ext == ".pdf":
            pipeline = DocumentExtractionPipeline(settings)
            extracted_doc = process_multipage_pdf(
                pdf_path=input_path,
                pipeline=pipeline,
                dpi=args.dpi,
                forced_doc_type=args.doc_type,
            )
            markdown_content = extracted_doc.markdown_content

        # 5. File Gambar Tunggal (PNG/JPG/WEBP)
        else:
            pipeline = DocumentExtractionPipeline(settings)
            if args.classify_only:
                doc_type = pipeline.extractor.classify(str(input_path))
                print(json.dumps({"file": str(input_path), "doc_type": doc_type}, indent=2))
                return 0

            res = pipeline.run(str(input_path), forced_doc_type=args.doc_type)
            markdown_content = res["markdown_content"]
            extracted_doc = ExtractedDocument(
                file_path=str(input_path),
                doc_type=res.get("doc_type", "plain"),
                total_pages=1,
                markdown_content=markdown_content,
                metadata={"source_type": "image", "filename": input_path.name},
            )

        # Output Markdown
        print(markdown_content)

        # Simpan ke file jika diminta (-o / --out)
        if args.out:
            out_file = Path(args.out)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(markdown_content, encoding="utf-8")
            print(f"\n[OK] Dokumen Markdown berhasil disimpan ke: {out_file}", file=sys.stderr)

        # Preview Chunks jika diminta
        if args.preview_chunks:
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
