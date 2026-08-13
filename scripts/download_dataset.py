"""
Download dataset FiftyOne dari Hugging Face Hub ke `input/datatest`.

Contoh:
    python scripts/download_dataset.py

Catatan:
  - Lokasi media diatur via env `FIFTYONE_DEFAULT_DATASET_DIR`
    (setara `fo.config.default_dataset_dir`) - bukan `dataset_dir`.
  - Dataset dibuat persistent agar bisa di-load dari proses lain.
  - Set HF_TOKEN (https://huggingface.co/settings/tokens) bila kena 429.
  - HF_HUB_DISABLE_XET=1 untuk menghindari rate limit backend xet.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

# Sebelum import fiftyone: arahkan dataset dir + nonaktifkan xet & warning symlink.
DATASET_DIR = Path(__file__).resolve().parent.parent / "input" / "datatest"
os.environ.setdefault("FIFTYONE_DEFAULT_DATASET_DIR", str(DATASET_DIR))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub

REPO = "Voxel51/form_understanding_in_noisy_scanned_documents_plus"
MAX_ATTEMPTS = 3
RETRY_DELAY_S = 15


def main() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    fo.config.default_dataset_dir = str(DATASET_DIR)

    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            print(f"[percobaan {attempt}/{MAX_ATTEMPTS}] Memuat {REPO} ...")
            dataset = load_from_hub(REPO, persistent=True)
            print(f"Dataset '{dataset.name}' dimuat: {len(dataset)} sampel")
            print(f"Lokasi media: {fo.config.default_dataset_dir}")
            print(f"Nama dataset (untuk load ulang): '{dataset.name}'")
            return
        except Exception as e:  # noqa: BLE001 - retry semua error jaringan
            last_error = e
            print(f"[percobaan {attempt}] gagal: {type(e).__name__}: {e}")
            if attempt < MAX_ATTEMPTS:
                print(f"Menunggu {RETRY_DELAY_S}s sebelum retry ...")
                time.sleep(RETRY_DELAY_S)

    raise SystemExit(f"Gagal setelah {MAX_ATTEMPTS} percobaan. Terakhir: {last_error}")


if __name__ == "__main__":
    main()
