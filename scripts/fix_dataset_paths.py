"""
Perbaiki filepath dataset FiftyOne agar menunjuk ke `input/datatest`.

Memakai ini karena media sudah dipindahkan manual ke input/datatest, sementara
dataset di DB masih menunjuk ke lokasi lama (~/fiftyone).
"""
from __future__ import annotations

from pathlib import Path

import fiftyone as fo

DATASET_DIR = Path(__file__).resolve().parent.parent / "input" / "datatest"
DATASET_NAME = "Voxel51/form_understanding_in_noisy_scanned_documents_plus"

# Prefix lama -> baru (Windows absolute path).
OLD_PREFIX = r"C:\Users\HYPE AMD\fiftyone"
NEW_PREFIX = str(DATASET_DIR)


def main() -> None:
    ds = fo.load_dataset(DATASET_NAME)
    changed = 0
    missing = 0
    with fo.ProgressBar() as pb:
        for sample in pb(ds):
            fp = sample.filepath
            if fp.startswith(OLD_PREFIX):
                sample.filepath = fp.replace(OLD_PREFIX, NEW_PREFIX, 1)
                sample.save()
                changed += 1
            if not Path(sample.filepath).exists():
                missing += 1

    print(f"Dataset: {DATASET_NAME} ({len(ds)} sampel)")
    print(f"Filepath diganti: {changed}")
    print(f"Filepath tidak ada di lokasi baru: {missing}")
    if changed:
        ds.save()
        print("Tersimpan.")


if __name__ == "__main__":
    main()
