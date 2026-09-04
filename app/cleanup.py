"""
Modul Utilitas Pembersihan & Migrasi Direktori Output.

Memastikan seluruh artefak ekstraksi (Markdown, database SQLite, render halaman/slide,
dan log) terenkapsulasi secara rapi di dalam sub-folder masing-masing dokumen:
`output/<nama_dokumen>/`.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def migrate_legacy_output(output_dir: str | Path = "output") -> None:
    """
    Migrasikan artefak legacy yang tercecer di root output/ ke sub-folder dokumen masing-masing.

    Contoh transformasi:
    - `output/{doc}.md` -> `output/{doc}/{doc}.md`
    - `output/databases/{doc}.sqlite` -> `output/{doc}/databases/{doc}.sqlite`
    - `output/pdf_pages/{doc}/*` -> `output/{doc}/pages/*`
    - `output/pptx_slides/{doc}/*` -> `output/{doc}/slides/*`
    - `output/logs/{doc}_*.log` -> `output/{doc}/logs/{doc}_*.log`
    """
    base = Path(output_dir).resolve()
    if not base.exists():
        return

    # 1. Kumpulkan seluruh nama stem dokumen dari file .md yang tercecer di root
    legacy_md_files = [
        f for f in base.glob("*.md") if f.is_file() and not f.name.startswith(".")
    ]

    # 2. Kumpulkan stem dari folder pdf_pages & pptx_slides jika ada
    known_stems: set[str] = {f.stem for f in legacy_md_files}
    legacy_pdf_pages_dir = base / "pdf_pages"
    if legacy_pdf_pages_dir.exists():
        for sub in legacy_pdf_pages_dir.iterdir():
            if sub.is_dir():
                known_stems.add(sub.name)

    legacy_pptx_slides_dir = base / "pptx_slides"
    if legacy_pptx_slides_dir.exists():
        for sub in legacy_pptx_slides_dir.iterdir():
            if sub.is_dir():
                known_stems.add(sub.name)

    legacy_db_dir = base / "databases"
    if legacy_db_dir.exists():
        for db in legacy_db_dir.glob("*.sqlite"):
            known_stems.add(db.stem)

    for stem in known_stems:
        target_doc_dir = base / stem
        target_doc_dir.mkdir(parents=True, exist_ok=True)

        # Migrasi file Markdown
        loose_md = base / f"{stem}.md"
        if loose_md.exists() and loose_md.is_file():
            dest_md = target_doc_dir / f"{stem}.md"
            if not dest_md.exists():
                shutil.move(str(loose_md), str(dest_md))
                logger.info("[Migration] Pindahkan %s -> %s", loose_md.name, dest_md)
            else:
                loose_md.unlink()

        # Migrasi file Database SQLite
        if legacy_db_dir.exists():
            loose_db = legacy_db_dir / f"{stem}.sqlite"
            if loose_db.exists():
                dest_db_dir = target_doc_dir / "databases"
                dest_db_dir.mkdir(parents=True, exist_ok=True)
                dest_db = dest_db_dir / f"{stem}.sqlite"
                if not dest_db.exists():
                    shutil.move(str(loose_db), str(dest_db))
                    logger.info("[Migration] Pindahkan %s -> %s", loose_db.name, dest_db)
                else:
                    loose_db.unlink()

        # Pastikan tabel SQLite memiliki ekspor CSV
        doc_db = target_doc_dir / "databases" / f"{stem}.sqlite"
        if doc_db.exists():
            dest_csv_dir = target_doc_dir / "csv"
            if not dest_csv_dir.exists() or not any(dest_csv_dir.glob("*.csv")):
                try:
                    from .tabular_db import TabularDatabaseManager

                    TabularDatabaseManager(doc_db).export_to_csv(output_dir=dest_csv_dir)
                except Exception as e_csv:  # noqa: BLE001
                    logger.warning("[Migration] Gagal ekspor CSV untuk %s: %s", stem, e_csv)


        # Migrasi halaman PDF
        if legacy_pdf_pages_dir.exists():
            loose_pages = legacy_pdf_pages_dir / stem
            if loose_pages.exists() and loose_pages.is_dir():
                dest_pages_dir = target_doc_dir / "pages"
                dest_pages_dir.mkdir(parents=True, exist_ok=True)
                for img_file in loose_pages.glob("*.*"):
                    dest_img = dest_pages_dir / img_file.name
                    if not dest_img.exists():
                        shutil.move(str(img_file), str(dest_img))
                shutil.rmtree(str(loose_pages), ignore_errors=True)
                logger.info("[Migration] Pindahkan folder pdf_pages/%s -> pages/", stem)

        # Migrasi slide PPTX
        if legacy_pptx_slides_dir.exists():
            loose_slides = legacy_pptx_slides_dir / stem
            if loose_slides.exists() and loose_slides.is_dir():
                dest_slides_dir = target_doc_dir / "slides"
                dest_slides_dir.mkdir(parents=True, exist_ok=True)
                for img_file in loose_slides.glob("*.*"):
                    dest_img = dest_slides_dir / img_file.name
                    if not dest_img.exists():
                        shutil.move(str(img_file), str(dest_img))
                shutil.rmtree(str(loose_slides), ignore_errors=True)
                logger.info("[Migration] Pindahkan folder pptx_slides/%s -> slides/", stem)

        # Migrasi file log
        legacy_logs_dir = base / "logs"
        if legacy_logs_dir.exists():
            dest_logs_dir = target_doc_dir / "logs"
            for log_file in legacy_logs_dir.glob(f"{stem}*.*"):
                dest_logs_dir.mkdir(parents=True, exist_ok=True)
                dest_log = dest_logs_dir / log_file.name
                if not dest_log.exists():
                    shutil.move(str(log_file), str(dest_log))
                else:
                    log_file.unlink()

    # Bersihkan direktori legacy kosong
    for d in (legacy_pdf_pages_dir, legacy_pptx_slides_dir, legacy_db_dir, base / "logs"):
        if d.exists() and not any(d.iterdir()):
            shutil.rmtree(str(d), ignore_errors=True)
            logger.info("[Migration] Hapus direktori legacy kosong: %s", d.name)


def clean_document_output(doc_stem: str, output_dir: str | Path = "output") -> None:
    """Hapus seluruh isi folder output dokumen tertentu."""
    target_doc_dir = Path(output_dir).resolve() / doc_stem
    if target_doc_dir.exists() and target_doc_dir.is_dir():
        shutil.rmtree(str(target_doc_dir), ignore_errors=True)
        logger.info("[Cleanup] Folder output dokumen dibersihkan: %s", target_doc_dir)


def clean_all_outputs(output_dir: str | Path = "output") -> None:
    """Hapus seluruh sub-folder dokumen di dalam direktori output kecuali file tersembunyi (.gitkeep)."""
    base = Path(output_dir).resolve()
    if not base.exists():
        return

    for item in base.iterdir():
        if item.name.startswith("."):
            continue
        if item.is_dir():
            shutil.rmtree(str(item), ignore_errors=True)
            logger.info("[Cleanup] Direktori dihapus: %s", item.name)
        elif item.is_file():
            item.unlink()
            logger.info("[Cleanup] File dihapus: %s", item.name)
