"""
Generator laporan evaluasi Markdown (tanpa ground truth).

Memproses satu file gambar atau seluruh gambar dalam satu folder, menjalankan
VLM (classify + extract) dan OCR per gambar, lalu menulis laporan Markdown:
  - gambar dirender (path relatif agar muncul di preview)
  - hasil VLM: doc_type + extraction JSON
  - hasil OCR : teks mentah

Rate limit server dihormati lewat `PacedCaller` (OCR 6 req/menit, VLM 40 req/menit).
Embedding tidak disentuh di mode ini (cukup VLM + OCR).
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .agents import get_agent
from .config import Settings
from .graph import VisionRAGPipeline
from .ocr import build_ocr_extractor

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")


class PacedCaller:
    """Beri jeda antar panggilan + retry sederhana (rate limit server)."""

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
            except Exception as e:  # noqa: BLE001 - retry semua error
                wait = 5.0 * attempt
                print(
                    f"    ! {label} gagal (percobaan {attempt}): "
                    f"{type(e).__name__}: {str(e)[:200]} -> tunggu {wait:.0f}s"
                )
                time.sleep(wait)
        raise RuntimeError(f"{label} gagal setelah {self.max_retries} percobaan")


def discover_images(path: str | Path) -> List[Path]:
    """Kumpulkan gambar dari satu file atau seluruh isi folder (terurut)."""
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(
            f for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        )
    raise FileNotFoundError(f"Path tidak ditemukan: {p}")


def render_section(
    idx: int,
    sample_path: Path,
    vlm: Optional[Dict[str, Any]],
    ocr: Optional[str],
    errors: Dict[str, str],
    output_parent: Path,
) -> str:
    """Satu blok markdown per gambar: gambar + hasil VLM + hasil OCR."""
    name = sample_path.name
    rel = os.path.relpath(str(sample_path), str(output_parent)).replace("\\", "/")
    lines: List[str] = []
    lines.append(f"## Sample {idx:03d} — {name}")
    lines.append("")
    lines.append(f"![{name}]({rel})")
    lines.append("")

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


def generate_report(
    input_path: str | Path,
    output_path: str | Path,
    settings: Settings,
    *,
    skip_vlm: bool = False,
    skip_ocr: bool = False,
    vlm_interval: float = 2.0,
    ocr_interval: float = 10.0,
    max_retries: int = 3,
) -> Dict[str, int]:
    """Jalankan pipeline pada semua gambar lalu tulis laporan Markdown."""
    images = discover_images(input_path)
    if not images:
        raise ValueError(f"Tidak ada gambar ditemukan di: {input_path}")

    pipeline = VisionRAGPipeline(settings)
    ocr_extractor = None if skip_ocr else build_ocr_extractor(settings)
    vlm_caller = PacedCaller(vlm_interval, max_retries)
    ocr_caller = PacedCaller(ocr_interval, max_retries)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {"vlm_ok": 0, "ocr_ok": 0, "vlm_fail": 0, "ocr_fail": 0}

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Laporan Evaluasi Vision RAG\n\n")
        f.write(f"- Tanggal: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"- Input: {input_path} ({len(images)} gambar)\n")
        f.write(f"- VLM: {settings.vlm_model} @ {settings.vlm_base_url}\n")
        f.write(f"- OCR: {settings.ocr_model} @ {settings.ocr_base_url}\n")
        f.write("\n---\n\n")
        f.flush()

        for i, image_path in enumerate(images, start=1):
            print(f"[{i}/{len(images)}] {image_path.name} ...", flush=True)
            vlm_result: Optional[Dict[str, Any]] = None
            ocr_text: Optional[str] = None
            errors: Dict[str, str] = {}

            if not skip_vlm:
                try:
                    doc_type = vlm_caller(
                        lambda: pipeline.classify(str(image_path)), label="classify"
                    )
                    agent = get_agent(doc_type)
                    extraction = vlm_caller(
                        lambda: agent.build(pipeline.vlm)
                        .extract(str(image_path))
                        .model_dump(),
                        label="extract",
                    )
                    vlm_result = {"doc_type": doc_type, "extraction": extraction}
                    stats["vlm_ok"] += 1
                except Exception as e:  # noqa: BLE001
                    errors["vlm"] = f"{type(e).__name__}: {e}"
                    stats["vlm_fail"] += 1

            if ocr_extractor is not None:
                try:
                    ocr_text = ocr_caller(
                        lambda: ocr_extractor.extract(str(image_path)).text, label="ocr"
                    )
                    stats["ocr_ok"] += 1
                except Exception as e:  # noqa: BLE001
                    errors["ocr"] = f"{type(e).__name__}: {e}"
                    stats["ocr_fail"] += 1

            f.write(
                render_section(i, image_path, vlm_result, ocr_text, errors, output_path.parent)
            )
            f.flush()

    print(f"\nLaporan: {output_path}")
    print(
        f"VLM berhasil {stats['vlm_ok']}, gagal {stats['vlm_fail']} | "
        f"OCR berhasil {stats['ocr_ok']}, gagal {stats['ocr_fail']}"
    )
    return stats
