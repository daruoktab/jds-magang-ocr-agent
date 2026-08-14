"""
CLI vision RAG agent.

Contoh:
    python main.py gambar.png                    # pipeline penuh -> JSON terstruktur
    python main.py gambar.png --query "..."      # + retrieval context
    python main.py gambar.png --classify-only    # hanya klasifikasi jenis
    python main.py gambar.png --ocr              # OCR teks terstruktur (model OCR)
    python main.py --list-agents                 # daftar jenis dokumen
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from app.agents import AGENT_REGISTRY
from app.config import get_settings
from app.graph import VisionRAGPipeline
from app.ocr import build_ocr_extractor
from app.pdf import pdf_to_images


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vision-rag",
        description="Vision RAG agent: gambar -> JSON terstruktur + retrieval.",
    )
    p.add_argument("image", nargs="?", help="Path ke file gambar (png/jpg/webp)")
    p.add_argument("--query", default=None, help="Query retrieval opsional")
    p.add_argument("--list-agents", action="store_true", help="Daftar jenis dokumen yang didukung")
    p.add_argument("--classify-only", action="store_true", help="Hanya klasifikasi jenis dokumen")
    p.add_argument("--ocr", action="store_true", help="OCR teks terstruktur (model OCR tuned)")
    p.add_argument("--pdf", action="store_true", help="Konversi PDF menjadi gambar per-halaman")
    p.add_argument("--dpi", type=int, default=200, help="DPI untuk konversi PDF (default 200)")
    p.add_argument("--out-dir", default=None, help="Direktori output (gambar PDF atau JSON hasil)")
    p.add_argument("--out-json", action="store_true", help="Simpan hasil JSON ke file alih-alih stdout")
    return p


def main(argv: Optional[list[str]] = None) -> int:
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

        if args.ocr:
            result = build_ocr_extractor(settings).extract(args.image)
            print(json.dumps(result.model_dump(), indent=2))
            return 0

        pipeline = VisionRAGPipeline(settings)

        if args.classify_only:
            print(json.dumps({"doc_type": pipeline.classify(args.image)}, indent=2))
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
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
