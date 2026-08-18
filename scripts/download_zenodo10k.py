"""
Download a small sample (~15 MB) of Zenodo10K PPTX files from Hugging Face Hub.

Strategies:
  1. Skip known huge files (>20 MB) by name.
  2. Pick files spread across license subdirs for diversity.
  3. Stop at ~15 MB or 15 files, whichever comes first.

Saved to: input/zenodo10k/

Usage:
    python scripts/download_zenodo10k.py
"""
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

HF_REPO = "Forceless/Zenodo10K"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "input" / "zenodo10k"
TARGET_BYTES = 15 * 1024 * 1024  # ~15 MB
MAX_FILES = 15  # safety cap

# Known huge files to skip (from earlier size scan)
SKIP_NAMES = {
    "1A_S6d_0900_Ainsworth.pptx",
    "SRRon_GGBN.pptx",
    "escience_2021_conferentie_2.pptx",
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    all_files = list(api.list_repo_files(HF_REPO, repo_type="dataset"))
    pptx_files = sorted(f for f in all_files if f.endswith(".pptx"))
    print(f"Found {len(pptx_files)} PPTX files total.")

    # Spread across license dirs: take ~1 per 700 files (gives ~15 across dataset)
    step = max(1, len(pptx_files) // 15)
    candidates = [
        f for f in pptx_files[::step]
        if Path(f).name not in SKIP_NAMES
    ]
    print(f"Selected {len(candidates)} candidate files (spread across dataset).")

    downloaded = 0
    total_bytes = 0

    for rel_path in candidates:
        if total_bytes >= TARGET_BYTES or downloaded >= MAX_FILES:
            break

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
            # Skip if unexpectedly huge
            if actual > 20 * 1024 * 1024:
                os.remove(local_path)
                print(f"  SKIP {rel} ({actual / 1024 / 1024:.1f} MB - too large)")
                continue
            total_bytes += actual
            downloaded += 1
            print(f"  [{total_bytes / 1024 / 1024:.1f} MB] {rel} ({actual / 1024 / 1024:.2f} MB)")
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP {rel}: {e}")

    print(f"\nDone: {downloaded} files, {total_bytes / 1024 / 1024:.1f} MB saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
