"""
Download a small sample (~15 MB) of PNG images from the FiftyOne
`form_understanding_in_noisy_scanned_documents_plus` dataset via Hugging Face Hub.

Each PNG is ~100-200 KB, so we need ~100 files for ~15 MB.

Saved to: input/datatest/

Usage:
    python scripts/download_dataset.py
"""
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

HF_REPO = "Voxel51/form_understanding_in_noisy_scanned_documents_plus"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "input" / "datatest"
TARGET_BYTES = 15 * 1024 * 1024  # ~15 MB
MAX_FILES = 150  # safety cap (each ~100-200 KB)

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    all_files = list(api.list_repo_files(HF_REPO, repo_type="dataset"))
    png_files = sorted(f for f in all_files if f.endswith(".png"))
    print(f"Found {len(png_files)} PNG files on HF Hub.")

    downloaded = 0
    total_bytes = 0

    for rel_path in png_files:
        if total_bytes >= TARGET_BYTES or downloaded >= MAX_FILES:
            break

        # Build dest path matching original structure
        rel = Path(rel_path).name
        dest = OUTPUT_DIR / rel
        if dest.exists():
            total_bytes += dest.stat().st_size
            downloaded += 1
            continue

        try:
            local_path = hf_hub_download(
                HF_REPO,
                filename=rel_path,
                repo_type="dataset",
                local_dir=OUTPUT_DIR,
            )
            actual = os.path.getsize(local_path)
            total_bytes += actual
            downloaded += 1
            print(f"  [{total_bytes / 1024 / 1024:.1f} MB] {rel} ({actual / 1024:.1f} KB)")
        except Exception as e:
            print(f"  SKIP {rel}: {e}")

    print(f"\nDone: {downloaded} files, {total_bytes / 1024 / 1024:.1f} MB saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
