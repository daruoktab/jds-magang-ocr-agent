"""
Generate laporan evaluasi Markdown dengan metrik kuantitatif (CER, WER, Recall).

Menjalankan pipeline agentik (VLM + OCR Fusion + Validasi) pada N gambar acak dari
dataset FiftyOne `input/datatest`, lalu menulis laporan Markdown berisi:
  - Tabel metrik kuantitatif ringkasan (CER, WER, JSON Recall, Waktu rata-rata)
  - Detail per-gambar: gambar dirender, ground truth kata-kata, hasil VLM, OCR, dan audit validasi.

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
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fiftyone as fo

from app.config import get_settings
from app.graph import VisionRAGPipeline
from app.ocr import build_ocr_extractor
from app.report import PacedCaller

DATASET_NAME = r"input/datatest/data"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "eval_report.md"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_eval_report",
        description="Laporan evaluasi kuantitatif Vision RAG: CER, WER, Recall, & validasi.",
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


# --- Metrik Evaluasi Kuantitatif --------------------------------------------
def _levenshtein_distance(seq1: list[Any] | str, seq2: list[Any] | str) -> int:
    """Hitung edit distance standar Levenshtein."""
    n, m = len(seq1), len(seq2)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return dp[m]


def compute_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate = edit_distance(ref_chars, hyp_chars) / max(len(ref), 1)."""
    ref_clean = reference.strip().lower()
    hyp_clean = hypothesis.strip().lower()
    if not ref_clean:
        return 0.0 if not hyp_clean else 1.0
    dist = _levenshtein_distance(ref_clean, hyp_clean)
    return min(1.0, dist / len(ref_clean))


def compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate = edit_distance(ref_words, hyp_words) / max(len(ref_words), 1)."""
    ref_words = reference.strip().lower().split()
    hyp_words = hypothesis.strip().lower().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    dist = _levenshtein_distance(ref_words, hyp_words)
    return min(1.0, dist / len(ref_words))


def compute_token_recall(reference: str, extracted_json: dict[str, Any] | None) -> float:
    """Hitung persentase kata unik di ground truth yang muncul dalam nilai JSON hasil ekstraksi."""
    if not reference or not extracted_json:
        return 0.0
    ref_tokens = set(reference.strip().lower().split())
    if not ref_tokens:
        return 1.0

    # Ambil seluruh representasi teks dari struktur JSON
    json_str = json.dumps(extracted_json, ensure_ascii=False).lower()
    found_count = sum(1 for token in ref_tokens if len(token) > 1 and token in json_str)
    return found_count / len(ref_tokens)


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
    vlm: dict[str, Any] | None,
    ocr: str | None,
    metrics: dict[str, float],
    validation: dict[str, Any] | None,
    errors: dict[str, str],
    output_parent: Path,
) -> str:
    name = sample_path.name
    rel = os.path.relpath(str(sample_path), str(output_parent)).replace("\\", "/")
    lines: list[str] = []
    lines.append(f"### Sample {idx:03d} — {name}")
    lines.append("")
    lines.append(f"![{name}]({rel})")
    lines.append("")

    if gt:
        lines.append(f"- **Ground truth (kata-kata):** {gt[:400]}")
    else:
        lines.append("- **Ground truth:** *(tidak ada label)*")

    # Tampilkan metrik per sampel
    lines.append(
        f"- **Metrik Sampel:** CER: `{metrics.get('cer', 0.0):.2%}` | "
        f"WER: `{metrics.get('wer', 0.0):.2%}` | "
        f"JSON Word Recall: `{metrics.get('recall', 0.0):.2%}`"
    )

    if validation:
        val_status = "Lolos" if validation.get("is_valid") else "Ada Isu"
        lines.append(f"- **Validasi Konsistensi:** `{val_status}` (Skor: `{validation.get('score', 1.0):.2f}`, Refleksi Retry: `{validation.get('reflection_attempts', 0)}`)")
        if validation.get("issues"):
            lines.append("  - *Isu:* " + "; ".join(validation["issues"]))

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


def main(argv: list[str] | None = None) -> int:
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

    # --- evaluasi tiap sampel ---------------------------------------------
    stats = {
        "vlm_ok": 0, "ocr_ok": 0, "vlm_fail": 0, "ocr_fail": 0,
        "cer_list": [], "wer_list": [], "recall_list": [], "val_scores": [],
    }
    sample_sections: list[str] = []
    t0 = time.monotonic()

    for i, sample in enumerate(selected, start=1):
        image_path = Path(sample.filepath)
        print(f"[{i}/{n}] {image_path.name} ...", flush=True)

        vlm_result: dict[str, Any] | None = None
        ocr_text: str | None = None
        validation_data: dict[str, Any] | None = None
        errors: dict[str, str] = {}
        gt_text = ground_truth_words(sample)

        # 1. OCR call
        if ocr_extractor is not None:
            try:
                img_str = str(image_path)
                ocr_text = ocr_caller(
                    lambda p=img_str: ocr_extractor.extract(p).text, label="ocr"
                )
                stats["ocr_ok"] += 1
            except Exception as e:  # noqa: BLE001
                errors["ocr"] = f"{type(e).__name__}: {e}"
                stats["ocr_fail"] += 1

        # 2. VLM Fusion & Validation call
        if not args.skip_vlm:
            try:
                img_str = str(image_path)
                state = vlm_caller(
                    lambda p=img_str, ot=ocr_text: pipeline.run(p),
                    label="vlm_agent",
                )
                final_res = state["final_result"]
                vlm_result = {
                    "doc_type": final_res["doc_type"],
                    "extraction": final_res["extraction"],
                }
                validation_data = final_res.get("validation")
                if validation_data:
                    stats["val_scores"].append(validation_data.get("score", 1.0))
                stats["vlm_ok"] += 1
            except Exception as e:  # noqa: BLE001
                errors["vlm"] = f"{type(e).__name__}: {e}"
                stats["vlm_fail"] += 1

        # 3. Hitung Metrik Kuantitatif
        cer_val = compute_cer(gt_text, ocr_text or "") if gt_text and ocr_text else 0.0
        wer_val = compute_wer(gt_text, ocr_text or "") if gt_text and ocr_text else 0.0
        recall_val = compute_token_recall(gt_text, vlm_result.get("extraction") if vlm_result else None) if gt_text else 0.0

        if gt_text:
            if ocr_text:
                stats["cer_list"].append(cer_val)
                stats["wer_list"].append(wer_val)
            if vlm_result:
                stats["recall_list"].append(recall_val)

        metrics = {"cer": cer_val, "wer": wer_val, "recall": recall_val}
        section = render_section(
            idx=i,
            sample_path=image_path,
            gt=gt_text,
            vlm=vlm_result,
            ocr=ocr_text,
            metrics=metrics,
            validation=validation_data,
            errors=errors,
            output_parent=args.output.parent,
        )
        sample_sections.append(section)

        if i % 5 == 0:
            elapsed = time.monotonic() - t0
            print(f"  ... {i}/{n} selesai ({elapsed:.0f}s elapsed)")

    total_time = time.monotonic() - t0
    avg_cer = (sum(stats["cer_list"]) / len(stats["cer_list"])) if stats["cer_list"] else 0.0
    avg_wer = (sum(stats["wer_list"]) / len(stats["wer_list"])) if stats["wer_list"] else 0.0
    avg_recall = (sum(stats["recall_list"]) / len(stats["recall_list"])) if stats["recall_list"] else 0.0
    avg_val = (sum(stats["val_scores"]) / len(stats["val_scores"])) if stats["val_scores"] else 1.0

    # --- tulis laporan Markdown lengkap dengan tabel ringkasan ---
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("# Laporan Evaluasi Kuantitatif Vision RAG\n\n")
        f.write(f"- **Tanggal:** {datetime.now(UTC).isoformat(timespec='seconds')}\n")
        f.write(f"- **Dataset:** `{DATASET_NAME}` ({len(dataset)} sampel)\n")
        f.write(f"- **Sampel Dievaluasi:** {n} (seed={args.seed})\n")
        f.write(f"- **VLM Model:** `{settings.vlm_model}` @ `{settings.vlm_base_url}`\n")
        f.write(f"- **OCR Model:** `{settings.ocr_model}` @ `{settings.ocr_base_url}`\n")
        f.write(f"- **Total Waktu:** {total_time:.1f} detik ({total_time / max(n, 1):.1f} s/sampel)\n\n")

        f.write("## 📊 Ringkasan Metrik Kuantitatif\n\n")
        f.write("| Metrik | Nilai Rata-rata | Keterangan |\n")
        f.write("|---|---|---|\n")
        f.write(f"| **OCR CER (Character Error Rate)** | **{avg_cer:.2%}** | Rata-rata kesalahan per karakter (lebih kecil lebih baik) |\n")
        f.write(f"| **OCR WER (Word Error Rate)** | **{avg_wer:.2%}** | Rata-rata kesalahan per kata |\n")
        f.write(f"| **JSON Field Word Recall** | **{avg_recall:.2%}** | Persentase kata GT yang tercakup dalam JSON VLM |\n")
        f.write(f"| **Validation Consistency Score** | **{avg_val:.2f} / 1.00** | Skor validasi matematika & konsistensi data |\n")
        f.write(f"| **Tingkat Keberhasilan VLM** | {stats['vlm_ok']} / {n} ({stats['vlm_ok']/max(n,1):.1%}) | Permintaan berhasil tanpa error |\n")
        f.write(f"| **Tingkat Keberhasilan OCR** | {stats['ocr_ok']} / {n} ({stats['ocr_ok']/max(n,1):.1%}) | Permintaan berhasil tanpa error |\n\n")
        f.write("---\n\n")
        f.write("## 📑 Rincian Sampel\n\n")

        for sec in sample_sections:
            f.write(sec)

    print("\n=== Ringkasan Evaluasi Kuantitatif ===")
    print(f"Waktu total: {total_time:.1f}s")
    print(f"Rata-rata CER: {avg_cer:.2%} | WER: {avg_wer:.2%}")
    print(f"Rata-rata JSON Recall: {avg_recall:.2%}")
    print(f"Rata-rata Skor Validasi: {avg_val:.2f}/1.00")
    print(f"VLM sukses: {stats['vlm_ok']}, gagal: {stats['vlm_fail']}")
    print(f"OCR sukses: {stats['ocr_ok']}, gagal: {stats['ocr_fail']}")
    print(f"Laporan lengkap: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
