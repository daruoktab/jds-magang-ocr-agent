"""
CLI vision RAG agent.

Contoh:
    python main.py gambar.png                    # pipeline penuh -> JSON terstruktur
    python main.py gambar.png --query "..."      # + retrieval context
    python main.py gambar.png --classify-only    # hanya klasifikasi jenis
    python main.py gambar.png --ocr              # OCR teks terstruktur (model OCR)
    python main.py dokumen.pdf --pdf             # konversi PDF -> gambar per halaman
    python main.py folder_gambar --report        # laporan markdown SEMUA gambar di folder
    python main.py gambar.png --report           # laporan markdown SATU gambar
    python main.py --list-agents                 # daftar jenis dokumen
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.agents import AGENT_REGISTRY
from app.config import get_settings
from app.graph import VisionRAGPipeline
from app.ocr import build_ocr_extractor
from app.pdf import pdf_to_images
from app.report import generate_report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vision-rag",
        description="Vision RAG agent: gambar -> JSON terstruktur + retrieval.",
    )
    p.add_argument("image", nargs="?", help="Path file gambar ATAU folder gambar")
    p.add_argument("--query", default=None, help="Query retrieval opsional")
    p.add_argument("--list-agents", action="store_true", help="Daftar jenis dokumen yang didukung")
    p.add_argument("--classify-only", action="store_true", help="Hanya klasifikasi jenis dokumen")
    p.add_argument("--ocr", action="store_true", help="OCR teks terstruktur (model OCR tuned)")
    p.add_argument("--pdf", action="store_true", help="Konversi PDF menjadi gambar per-halaman")
    p.add_argument("--dpi", type=int, default=200, help="DPI untuk konversi PDF (default 200)")
    p.add_argument("--out-dir", default="output", help="Direktori output (default: output/)")

    # Mode laporan markdown (file atau folder gambar)
    p.add_argument("--report", action="store_true",
                   help="Tulis laporan markdown (gambar dirender + hasil VLM/OCR)")
    p.add_argument("--skip-vlm", action="store_true", help="(report) lewati VLM")
    p.add_argument("--skip-ocr", action="store_true", help="(report) lewati OCR")
    p.add_argument("--vlm-interval", type=float, default=2.0,
                   help="(report) pacing VLM dalam detik (default 2.0)")
    p.add_argument("--ocr-interval", type=float, default=10.0,
                   help="(report) pacing OCR dalam detik (default 10.0)")
    p.add_argument("--max-retries", type=int, default=3,
                   help="(report) retry per request saat gagal/429")
    return p


def _output_json(text: str, out_dir: str | None, image: str, log_file=None) -> str:
    if out_dir:
        out_file = Path(out_dir) / f"{Path(image).stem}.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(text, encoding="utf-8")
        if log_file:
            print(f"\n-> {out_file}", file=log_file)
    return text


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_agents:
        for name, agent in AGENT_REGISTRY.items():
            marker = "  <- fallback" if name == "generic" else ""
            print(f"- {name:16s} {agent.description}{marker}")
        return 0

    if not args.image:
        print("ERROR: argumen 'image' wajib diisi (atau --list-agents)", file=sys.stderr)
        return 1

    try:
        settings = get_settings()

        if args.pdf:
            images = pdf_to_images(args.image, output_dir=args.out_dir, dpi=args.dpi)
            for img in images:
                print(img)
            return 0

        if args.report:
            out_file = Path(args.out_dir) / "report.md"
            generate_report(
                input_path=args.image,
                output_path=out_file,
                settings=settings,
                skip_vlm=args.skip_vlm,
                skip_ocr=args.skip_ocr,
                vlm_interval=args.vlm_interval,
                ocr_interval=args.ocr_interval,
                max_retries=args.max_retries,
            )
            return 0

        if args.ocr:
            out = _output_json(json.dumps(build_ocr_extractor(settings).extract(args.image).model_dump(), indent=2), args.out_dir, args.image, sys.stderr)
            print(out)
            return 0

        pipeline = VisionRAGPipeline(settings)

        if args.classify_only:
            out = _output_json(json.dumps({"doc_type": pipeline.classify(args.image)}, indent=2), args.out_dir, args.image, sys.stderr)
            print(out)
            return 0

        state = pipeline.run(args.image, query=args.query)
        result = json.dumps(state["final_result"], indent=2)
        if args.out_dir:
            out_file = Path(args.out_dir) / f"{Path(args.image).stem}.json"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(result, encoding="utf-8")
            print(result)
            print(f"\n-> {out_file}", file=sys.stderr)
        elif args.out_json:
            raise SystemExit(result)
        else:
            print(result)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
