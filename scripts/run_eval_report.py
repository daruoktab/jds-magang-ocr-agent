"""
Generate laporan evaluasi Markdown.

Menjalankan pipeline (VLM: classify + extract, dan OCR) pada N gambar acak dari
dataset FiftyOne `input/datatest`, lalu menulis laporan Markdown berisi:
  - gambar dokumen (dirender via path relatif)
  - ground truth kata-kata (label `words` dari dataset)
  - hasil VLM: doc_type + extraction JSON
  - hasil OCR : teks mentah (ocr-lighton)

Rate limit server:
  - OCR   : 6 req/menit  -> pacing default 10 detik antar request
  - VLM   : 40 req/menit -> pacing default 2 detik antar request

Contoh:
  python scripts/run_eval_report.py --samples 100
  python scripts/run_eval_report.py --samples 3 --fast   # tes cepat
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import fiftyone as fo

from app.agents import get_agent
from app.config import get_settings
from app.graph import VisionRAGPipeline
from app.ocr import build_ocr_extractor

DATASET_NAME = "Voxel51/form_understanding_in_noisy_scanned_documents_plus"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "eval_report.md"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_eval_report",
        description="Laporan evaluasi Markdown: gambar + ground truth + hasil VLM/OCR.",
    )
    p.add_argument("--samples", type=int, default=100, help="Jumlah gambar acak (default 100)")
    p.add_argument("--seed", type=int, default=42, help="Seed random agar reproducible")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path file Markdown output")
    p.add_argument("--ocr-interval", type=float, default=10.0, help="Pacing OCR (detik; min 6 req/menit)")
    p.add_argument("--vlm-interval", type=float, default=2.0, help="Pacing VLM (detik; min 40 req/menit)")
    p.add_argument("--max-retries", type=int, default=3, help="Retry per request saat gagal/429")
    p.add_argument("--skip-ocr", action="store_true", help="Lewati OCR")
    p.add_argument("--skip-vlm", action="store_true", help="Lewati VLM")
    p.add_argument("--fast", action="store_true", help="Percepat pacing (untuk tes, bisa kena 429)")
    return p.parse_args(argv)


class PacedCaller:
    """Beri jeda antar panggilan + retry sederhana."""

    def __init__(self, interval: float, max_retries: int) -> None:
        self.interval = interval
        self.max_retries = max_retries
        self._last = 0.0

    def __call__(self, fn: Callable[[], Any], label: str = "") -> Any:
        for attempt in range(1, self.max_retries + 1):
            elapsed = time.monotonic() - self._last
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self._last = time.monotonic()
            try:
                return fn()
            except Exception as e:
                wait = 5.0 * attempt
                print(f"    ! {label} gagal (percobaan {attempt}): "
                      f"{type(e).__name__}: {str(e)[:200]} -> tunggu {wait:.0f}s")
                time.sleep(wait)
        raise RuntimeError(f"{label} gagal setelah {self.max_retries} percobaan")


def ground_truth_words(sample) -> str:
    """Gabungkan teks dari label `words` (deteksi per kata) jadi satu string."""
    dets = getattr(sample, "words", None)
    if dets is None or not dets.detections:
        return ""
    texts = [d.text or "" for d in dets.detections if getattr(d, "text", None)]
    return " ".join(texts)


def render_section(
    idx: int,
    sample_path: Path,
    gt: str,
    vlm: Optional[Dict[str, Any]],
    ocr: Optional[str],
    errors: Dict[str, str],
    output_parent: Path,
) -> str:
    name = sample_path.name
    rel = os.path.relpath(str(sample_path), str(output_parent)).replace("\\", "/")
    lines: List[str] = []
    lines.append(f"## Sample {idx:03d} — {name}")
    lines.append("")
    lines.append(f"![{name}]({rel})")
    lines.append("")

    if gt:
        lines.append(f"- **Ground truth (kata-kata):** {gt[:400]}")
    else:
        lines.append("- **Ground truth:** *(tidak ada label)*")

    if vlm is not None:
        lines.append(f"- **VLM doc_type:** `{vlm.get('doc_type')}`")
        lines.append("- **VLM extraction:**")
        lines.append("```json")
        lines.append(json.dumps(vlm.get("extraction", {}), indent=2, ensure_ascii=False))
        lines.append("```")
    else:
        lines.append(f"- **VLM:** *{errors.get('vlm') or 'skip'}*")

    if ocr is not None:
        lines.append("- **OCR text:**")
        lines.append("```")
        lines.append(ocr)
        lines.append("```")
    else:
        lines.append(f"- **OCR:** *{errors.get('ocr') or 'skip'}*")

    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    settings = get_settings()

    if args.fast:
        args.ocr_interval = min(args.ocr_interval, 1.5)
        args.vlm_interval = min(args.vlm_interval, 0.5)

    # --- siapkan pipeline -------------------------------------------------
    pipeline = VisionRAGPipeline(settings)
    ocr_extractor = None if args.skip_ocr else build_ocr_extractor(settings)
    vlm_caller = PacedCaller(args.vlm_interval, args.max_retries)
    ocr_caller = PacedCaller(args.ocr_interval, args.max_retries)

    # --- pilih sampel acak dari dataset -----------------------------------
    dataset = fo.load_dataset(DATASET_NAME)
    n = min(args.samples, len(dataset))
    rng = random.Random(args.seed)
    selected = rng.sample(list(dataset), n)
    print(f"Dataset: {len(dataset)} sampel | memilih {n} acak (seed={args.seed})")

    # --- tulis header ------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("# Evaluasi Vision RAG\n\n")
        f.write(f"- Tanggal: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"- Dataset: {DATASET_NAME} ({len(dataset)} sampel)\n")
        f.write(f"- Sampel dievaluasi: {n} (seed={args.seed})\n")
        f.write(f"- VLM: {settings.vlm_model} @ {settings.vlm_base_url}\n")
        f.write(f"- OCR: {settings.ocr_model} @ {settings.ocr_base_url}\n")
        f.write("\n---\n\n")
        f.flush()

    # --- evaluasi tiap sampel (progressive write) --------------------------
    stats = {"vlm_ok": 0, "ocr_ok": 0, "vlm_fail": 0, "ocr_fail": 0}
    t0 = time.monotonic()

    for i, sample in enumerate(selected, start=1):
        image_path = Path(sample.filepath)
        print(f"[{i}/{n}] {image_path.name} ...", flush=True)

        vlm_result: Optional[Dict[str, Any]] = None
        ocr_text: Optional[str] = None
        errors: Dict[str, str] = {}

        if not args.skip_vlm:
            try:
                doc_type = vlm_caller(
                    lambda: pipeline.classify(str(image_path)), label="classify"
                )
                agent = get_agent(doc_type)
                extraction = vlm_caller(
                    lambda: agent.build(pipeline.vlm).extract(str(image_path)).model_dump(),
                    label="extract",
                )
                vlm_result = {"doc_type": doc_type, "extraction": extraction}
                stats["vlm_ok"] += 1
            except Exception as e:
                errors["vlm"] = f"{type(e).__name__}: {e}"
                stats["vlm_fail"] += 1

        if ocr_extractor is not None:
            try:
                ocr_text = ocr_caller(
                    lambda: ocr_extractor.extract(str(image_path)).text, label="ocr"
                )
                stats["ocr_ok"] += 1
            except Exception as e:
                errors["ocr"] = f"{type(e).__name__}: {e}"
                stats["ocr_fail"] += 1

        section = render_section(
            idx=i,
            sample_path=image_path,
            gt=ground_truth_words(sample),
            vlm=vlm_result,
            ocr=ocr_text,
            errors=errors,
            output_parent=args.output.parent,
        )
        with open(args.output, "a", encoding="utf-8") as f:
            f.write(section)
            f.flush()

        if i % 10 == 0:
            elapsed = time.monotonic() - t0
            print(f"  ... {i}/{n} selesai ({elapsed:.0f}s elapsed)")

    # --- ringkasan akhir ----------------------------------------------------
    elapsed = time.monotonic() - t0
    print("\n=== Ringkasan ===")
    print(f"Waktu total: {elapsed:.0f}s")
    print(f"VLM berhasil: {stats['vlm_ok']}, gagal: {stats['vlm_fail']}")
    print(f"OCR berhasil: {stats['ocr_ok']}, gagal: {stats['ocr_fail']}")
    print(f"Laporan: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
