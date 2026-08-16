"""
Download a small sample (~15 MB) of govdocs1 PDFs from Hugging Face Hub.

Saved to: input/govdocs1/

Usage:
    python scripts/download_govdocs1.py
"""
from __future__ import annotations

import os
from pathlib import Path

import requests
from huggingface_hub import HfApi, hf_hub_download

HF_REPO = "BEE-spoke-data/govdocs1-pdf-source"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "input" / "govdocs1"
TARGET_BYTES = 15 * 1024 * 1024  # ~15 MB

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    all_files = list(api.list_repo_files(HF_REPO, repo_type="dataset"))

    # Prefer sample/ directory (smaller PDFs, curated subset)
    sample_pdfs = sorted(f for f in all_files if f.startswith("sample/") and f.endswith(".pdf"))
    # Fallback to data/ if sample is too small
    data_pdfs = sorted(f for f in all_files if f.startswith("data/") and f.endswith(".pdf"))

    candidates = sample_pdfs if sample_pdfs else data_pdfs
    print(f"Found {len(candidates)} candidate PDFs in sample/ directory.")

    downloaded = 0
    total_bytes = 0

    for rel_path in candidates:
        if total_bytes >= TARGET_BYTES:
            break

        dest = OUTPUT_DIR / Path(rel_path).name
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
            size = os.path.getsize(local_path)
            total_bytes += size
            downloaded += 1
            print(f"  [{total_bytes / 1024 / 1024:.1f} MB] {rel_path}")
        except Exception as e:
            print(f"  SKIP {rel_path}: {e}")

    print(f"\nDone: {downloaded} files, {total_bytes / 1024 / 1024:.1f} MB saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
