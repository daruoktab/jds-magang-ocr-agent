"""
CLI vision_ocr: ekstraksi dokumen multi-agent.

Default: agent utama (dispatcher) mengklasifikasikan jenis dokumen,
memilih agent ekstraksi yang sesuai, lalu mengekstrak.

Contoh:
    python -m vision_ocr.cli struk.png                 # auto: klasifikasi + agent terbaik
    python -m vision_ocr.cli gambar.png --agent table  # paksa agent jenis tertentu
    python -m vision_ocr.cli gambar.png --list-agents  # daftar jenis dokumen yang didukung
"""
import argparse
import sys
from typing import Optional

from . import config
from .agents import AGENT_REGISTRY
from .dispatcher import DocumentDispatcher


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vision_ocr",
        description="Ekstraksi dokumen multi-agent berbasis VLM + Pydantic.",
    )
    p.add_argument("image", nargs="?", help="Path ke file gambar (png/jpg/webp)")
    p.add_argument("--agent", default=None, help="Paksa agent jenis tertentu (skip klasifikasi)")
    p.add_argument("--list-agents", action="store_true", help="Tampilkan daftar jenis dokumen yang didukung")
    p.add_argument("--classify-only", action="store_true", help="Hanya klasifikasi jenis dokumen")
    p.add_argument("--backend", default=config.DEFAULT_BACKEND, help="Nama backend LLM")
    p.add_argument("--model", default=None, help=f"Nama model (default: {config.DEFAULT_MODEL})")
    p.add_argument("--base-url", default=None, help="Base URL API LLM")
    p.add_argument("--api-key", default=None, help="API key (untuk backend openai_compat)")
    p.add_argument("--timeout", type=float, default=config.DEFAULT_TIMEOUT, help="Timeout request (detik)")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_agents:
        for name, agent in AGENT_REGISTRY.items():
            marker = "  <- fallback" if name == "generic" else ""
            print(f"- {name:16s} {agent.description}{marker}")
        return 0

    if not args.image:
        print("ERROR: argumen 'image' wajib diisi (atau gunakan --list-agents)", file=sys.stderr)
        return 1

    model = args.model or config.DEFAULT_MODEL
    base_url = args.base_url or config.DEFAULT_BASE_URL

    try:
        dispatcher = DocumentDispatcher(
            backend_name=args.backend,
            model=model,
            base_url=base_url,
            api_key=args.api_key,
            timeout=args.timeout,
        )

        if args.classify_only:
            print(f"Klasifikasi {args.image} ...", file=sys.stderr)
            print(dispatcher.classify(args.image))
            return 0

        if args.agent:
            if args.agent not in AGENT_REGISTRY:
                print(
                    f"ERROR: agent '{args.agent}' tidak dikenal. Tersedia: {sorted(AGENT_REGISTRY)}",
                    file=sys.stderr,
                )
                return 1
            print(f"Memproses {args.image} dengan agent '{args.agent}' ...", file=sys.stderr)
            result = AGENT_REGISTRY[args.agent].run(args.image, dispatcher.backend)
            doc_type = args.agent
        else:
            print(f"Dispatcher: klasifikasi + ekstraksi {args.image} ...", file=sys.stderr)
            doc_type, result = dispatcher.dispatch(args.image)
            print(f"Jenis terdeteksi: '{doc_type}'", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
