"""
Modul pemrosesan dokumen presentasi PowerPoint (.pptx / .ppt).

Alur presentasi dijaga sederhana dan eksplisit:
  - PPT/PPTX dirender menjadi gambar per slide
  - Setiap gambar slide dikirim langsung ke VLM Qwen 35B Vision (Jalur 1)
  - Sub-Agent SQL mandiri per slide mengecek tabel, skema, dan meng-ingest ke SQLite (Jalur 2)
  - Di akhir, Agent Pusat menjalankan Dual-Track Guardrail Cross-Verification.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

from .llm import image_data_uri
from .multi_page import stitch_pages_to_markdown
from .tabular_db import cross_verify_dual_track, process_page_tabular_agent

logger = logging.getLogger("app.ppt")

# Ukuran batch untuk pemrosesan presentasi/per dokumen massal.
SLIDE_BATCH_SIZE: int = 10

PPT_SLIDE_SYSTEM_PROMPT: str = (
    "Kamu adalah AI ekstraktor presentasi yang membaca gambar slide secara visual dan "
    "mengubahnya menjadi Markdown bersih. "
    "Jangan memakai OCR eksternal atau menambahkan teks pengantar. "
    "Output harus hanya isi slide dalam Markdown."
)


def _build_slide_prompt(
    slide_number: int,
    total_slides: int,
    previous_slide_context: str | None = None,
) -> str:
    """Susun instruksi khusus slide presentasi tanpa konteks OCR."""
    blocks: list[str] = [
        (
            f"Ekstrak slide {slide_number} dari {total_slides} berikut ini menjadi Markdown yang rapi "
            f"dan setia pada isi gambar."
        ),
        "Aturan output:",
        "- Pertahankan teks, angka, istilah teknis, label grafik, dan urutan visual semirip mungkin dengan slide.",
        "- Gunakan heading Markdown, bullet bertingkat, tabel Markdown, dan blockquote bila ada visual/diagram yang perlu dijelaskan.",
        "- Jangan menambahkan interpretasi di luar yang terlihat jelas pada slide.",
        "- Jangan menulis penjelasan pembuka atau penutup.",
        "- Jika ada teks yang tidak terbaca, tandai seperlunya secara singkat dan jangan mengarang.",
    ]

    if previous_slide_context and previous_slide_context.strip():
        blocks.append(
            "Konteks slide sebelumnya (untuk kontinuitas jika slide ini merupakan lanjutan):\n"
            f"```markdown\n{previous_slide_context.strip()[-500:]}\n```"
        )

    blocks.append("Outputkan HANYA Markdown final dari slide ini.")
    return "\n".join(blocks)


def _extract_slide_markdown(
    llm: BaseChatModel,
    image_path: str,
    slide_number: int,
    total_slides: int,
    previous_slide_context: str | None = None,
) -> str:
    """Kirim satu gambar slide langsung ke VLM dan ambil Markdown-nya."""
    prompt = _build_slide_prompt(
        slide_number=slide_number,
        total_slides=total_slides,
        previous_slide_context=previous_slide_context,
    )
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data_uri(image_path)}},
        ]
    )
    response = llm.invoke([SystemMessage(content=PPT_SLIDE_SYSTEM_PROMPT), message])
    md_text = str(response.content).strip()

    if md_text.startswith("```markdown") and md_text.endswith("```"):
        md_text = md_text[len("```markdown") : -3].strip()
    elif md_text.startswith("```md") and md_text.endswith("```"):
        md_text = md_text[len("```md") : -3].strip()
    elif md_text.startswith("```") and md_text.endswith("```"):
        md_text = md_text[3:-3].strip()

    return md_text


def _find_libreoffice() -> str:
    """Cari executable LibreOffice/soffice yang tersedia di sistem."""
    configured = os.environ.get("LIBREOFFICE_BIN", "").strip()
    if configured:
        configured_path = Path(configured).expanduser()
        if configured_path.exists():
            return str(configured_path)
        resolved = shutil.which(configured)
        if resolved:
            return resolved
        raise FileNotFoundError(
            f"Executable LibreOffice dari LIBREOFFICE_BIN tidak ditemukan: {configured}"
        )

    for executable in ("libreoffice", "soffice"):
        resolved = shutil.which(executable)
        if resolved:
            return resolved

    # Fallback paths standar untuk Windows, Linux, dan macOS
    candidate_paths = [
        Path(r"C:\Program Files\LibreOffice\program\soffice.exe"),
        Path(r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"),
        Path("/usr/bin/libreoffice"),
        Path("/usr/bin/soffice"),
        Path("/usr/local/bin/libreoffice"),
        Path("/usr/local/bin/soffice"),
        Path("/Applications/LibreOffice.app/Contents/MacOS/soffice"),
    ]
    for p in candidate_paths:
        if p.exists():
            return str(p)

    raise FileNotFoundError(
        "LibreOffice tidak ditemukan. Instal LibreOffice atau set "
        "environment variable LIBREOFFICE_BIN ke path executable soffice."
    )


def convert_presentation_to_pdf(
    presentation_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Konversi PPT/PPTX menjadi PDF menggunakan LibreOffice headless."""
    source = Path(presentation_path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"File presentasi tidak ditemukan: {source}")
    if source.suffix.lower() not in {".ppt", ".pptx"}:
        raise ValueError(f"Format presentasi tidak didukung: {source.suffix}")

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    executable = _find_libreoffice()
    with tempfile.TemporaryDirectory(prefix="libreoffice_profile_") as profile_name:
        profile_dir = Path(profile_name).resolve()
        command = [
            executable,
            "--headless",
            f"-env:UserInstallation={profile_dir.as_uri()}",
            "--convert-to",
            "pdf",
            "--outdir",
            str(destination),
            str(source),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Konversi LibreOffice gagal (code {result.returncode}): {result.stderr.strip() or result.stdout.strip()}"
            )

    pdf_candidate = destination / f"{source.stem}.pdf"
    if not pdf_candidate.exists():
        matches = sorted(destination.glob(f"{source.stem}.pdf"))
        if not matches:
            raise FileNotFoundError(
                f"Hasil PDF konversi LibreOffice tidak ditemukan di {destination}"
            )
        return matches[0]

    return pdf_candidate


def render_presentation_slides_to_images(
    presentation_path: str | Path,
    output_dir: str | Path | None = None,
    slides: list[int] | None = None,
    dpi: int = 200,
    image_ext: str = ".jpg",
) -> list[Path]:
    """Render kanvas slide PowerPoint menjadi gambar per slide via LibreOffice -> PDF -> PyMuPDF."""
    import pymupdf

    source = Path(presentation_path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"File presentasi tidak ditemukan: {source}")

    out_dir = (
        Path(output_dir).resolve()
        if output_dir
        else (source.parent / f"{source.stem}_slides").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    ext = image_ext if image_ext.startswith(".") else f".{image_ext}"

    with tempfile.TemporaryDirectory(prefix="libreoffice_pdf_") as temp_pdf_dir:
        pdf_path = convert_presentation_to_pdf(source, temp_pdf_dir)
        doc = pymupdf.open(pdf_path)
        total = len(doc)
        if total == 0:
            doc.close()
            return []

        target_indices = (
            list(range(total))
            if slides is None
            else [i for i in slides if 0 <= i < total]
        )

        zoom = dpi / 72.0
        matrix = pymupdf.Matrix(zoom, zoom)
        generated_paths: list[Path] = []
        total_targets = len(target_indices)

        for offset, idx in enumerate(target_indices, start=1):
            page = doc[idx]
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            slide_num = idx + 1
            img_path = out_dir / f"slide_{slide_num}{ext}"
            pix.save(str(img_path))
            generated_paths.append(img_path)
            if offset % SLIDE_BATCH_SIZE == 0 or offset == total_targets:
                logger.info(
                    "[Render PPT] Batch selesai: %d/%d slide dirender",
                    offset,
                    total_targets,
                )

        doc.close()
        return generated_paths


def count_presentation_slides(
    presentation_path: str | Path,
) -> int:
    """Hitung jumlah slide presentasi menggunakan LibreOffice -> PDF."""
    import pymupdf

    source = Path(presentation_path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"File presentasi tidak ditemukan: {source}")

    with tempfile.TemporaryDirectory(prefix="ppt_count_") as temp_pdf_dir:
        pdf_path = convert_presentation_to_pdf(source, temp_pdf_dir)
        doc = pymupdf.open(pdf_path)
        total = len(doc)
        doc.close()
        return total


def _extract_shape_text(shape: Any) -> list[str]:
    """Ekstrak teks dari shape dengan menjaga struktur paragraf & bullet level."""
    lines: list[str] = []
    if not shape.has_text_frame:
        return lines

    for paragraph in shape.text_frame.paragraphs:
        raw_text = paragraph.text.strip()
        if not raw_text:
            continue

        level = getattr(paragraph, "level", 0)
        indent = "  " * level
        bullet_marker = "- " if level > 0 else ""
        lines.append(f"{indent}{bullet_marker}{raw_text}")

    return lines


def _table_to_markdown(table_or_shape: Any) -> str:
    """Konversi shape tabel PPTX atau objek Table ke Markdown Table (GFM)."""
    table = table_or_shape.table if hasattr(table_or_shape, "table") else table_or_shape
    rows: list[list[str]] = []
    for row in getattr(table, "rows", []):
        cell_texts = [cell.text.replace("\n", " ").strip() for cell in row.cells]
        rows.append(cell_texts)

    if not rows:
        return ""

    header = rows[0]
    separator = ["---"] * len(header)
    md_lines: list[str] = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in rows[1:]:
        padded = row + [""] * (len(header) - len(row))
        md_lines.append("| " + " | ".join(padded[: len(header)]) + " |")

    return "\n".join(md_lines)


def pptx_to_structured_text(pptx_path: str | Path) -> list[dict[str, Any]]:
    """
    Ekstrak presentasi PPTX menjadi list struktur per slide via parser native python-pptx.
    """
    path_obj = Path(pptx_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"File presentasi tidak ditemukan: {path_obj}")

    start_time = time.time()
    prs = Presentation(str(path_obj))
    total_slides = len(prs.slides)
    logger.info(
        "[Workflow] Membaca presentasi: '%s' | Total slide: %d",
        path_obj.name,
        total_slides,
    )

    slides_data: list[dict[str, Any]] = []

    for idx, slide in enumerate(prs.slides, start=1):
        slide_title: str = ""
        body_lines: list[str] = []
        notes_text: str = ""
        image_count: int = 0
        table_count: int = 0

        title_shape = slide.shapes.title
        if title_shape and title_shape.has_text_frame:
            slide_title = title_shape.text.strip()

        for shape in slide.shapes:
            if shape == title_shape:
                continue

            if shape.has_table:
                table_count += 1
                tbl_md = _table_to_markdown(shape.table)
                if tbl_md:
                    body_lines.append(f"\n{tbl_md}\n")

            elif shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                image_count += 1

            elif shape.has_text_frame:
                extracted_lines = _extract_shape_text(shape)
                if extracted_lines:
                    body_lines.extend(extracted_lines)

        if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
            notes_text = slide.notes_slide.notes_text_frame.text.strip()

        title_display = slide_title or f"Slide {idx}"
        md_content_lines: list[str] = [f"## Slide {idx}: {title_display}"]
        if body_lines:
            md_content_lines.extend(body_lines)
        if notes_text:
            md_content_lines.append(f"\n> **Speaker Notes:** {notes_text}")

        slide_markdown = "\n".join(md_content_lines)

        slides_data.append(
            {
                "slide_number": idx,
                "title": title_display,
                "markdown": slide_markdown,
                "notes": notes_text,
                "image_count": image_count,
            }
        )

    duration = time.time() - start_time
    logger.info(
        "[Workflow] Selesai parsing %d slide dalam %.2fs", total_slides, duration
    )
    return slides_data


def process_presentation(pptx_path: str | Path) -> str:
    """
    Ekstrak seluruh file PPTX menjadi satu dokumen Markdown terpadu via native python-pptx.
    """
    slides = pptx_to_structured_text(pptx_path)
    file_stem = Path(pptx_path).stem.replace("_", " ").title()

    doc_lines: list[str] = [f"# {file_stem}\n"]
    for s in slides:
        doc_lines.append(s["markdown"])
        doc_lines.append("\n---\n")

    return "\n".join(doc_lines).strip()


def process_presentation_vision(
    pptx_path: str | Path,
    pipeline: Any,
    output_dir: str | Path | None = None,
    forced_specs: list[str] | str | None = "presentation_slides",
    db_path: str | Path | None = None,
    auto_tabular_db: bool = True,
    force_all_tables: bool = False,
) -> str:
    """
    Render slide PPT/PPTX menjadi gambar kanvas per slide dan jalankan arsitektur Dual-Track:
      - Jalur 1: Mengirim slide langsung ke VLM.
      - Jalur 2: Sub-Agent SQL mandiri per slide untuk memproses dan meng-ingest tabel SQLite.
      - Tahap Akhir: Guardrail Cross-Verification oleh Agent Pusat.
    """
    path_obj = Path(pptx_path).resolve()

    logger.info("[Vision PPT] Memulai rendering slide menjadi gambar PNG kanvas...")
    slide_images = render_presentation_slides_to_images(
        presentation_path=path_obj,
        output_dir=output_dir,
    )
    total_images = len(slide_images)
    if total_images == 0:
        raise RuntimeError(f"Tidak ada slide yang berhasil dirender dari: {path_obj}")

    logger.info(
        "[Vision PPT] Selesai render %d slide gambar. Memproses jalur ganda (VLM & Tabular Sub-Agent)...",
        total_images,
    )

    slide_markdowns: list[str] = []
    previous_context: str | None = None
    file_stem = path_obj.stem.replace("_", " ").title()
    llm: BaseChatModel | None = getattr(pipeline, "vlm", None)
    if llm is None:
        raise AttributeError(
            "pipeline harus menyediakan atribut 'vlm' untuk ekstraksi PPT Vision"
        )

    # Tentukan path SQLite jika aktif
    if db_path:
        resolved_db_path: Path | None = Path(db_path).resolve()
    elif output_dir:
        resolved_db_path = Path(output_dir).resolve() / "databases" / f"{path_obj.stem}.sqlite"
    else:
        resolved_db_path = Path("output/databases").resolve() / f"{path_obj.stem}.sqlite"

    if resolved_db_path and auto_tabular_db:
        resolved_db_path.parent.mkdir(parents=True, exist_ok=True)

    for idx, img_path in enumerate(slide_images, start=1):
        logger.info(
            "[Vision PPT] [Slide %d/%d] Mengirim gambar '%s' langsung ke VLM...",
            idx,
            total_images,
            img_path.name,
        )
        page_md = _extract_slide_markdown(
            llm=llm,
            image_path=str(img_path),
            slide_number=idx,
            total_slides=total_images,
            previous_slide_context=previous_context,
        )
        slide_markdowns.append(page_md)
        previous_context = page_md[-400:] if len(page_md) > 400 else page_md

        # Jalur 2: Sub-Agent SQL Tabular Engine per slide
        if auto_tabular_db and resolved_db_path:
            tab_event, _ = process_page_tabular_agent(
                page_markdown=page_md,
                page_number=idx,
                source_file=str(path_obj.resolve()),
                db_path=resolved_db_path,
                table_name_prefix=path_obj.stem,
                append_if_matching=True,
                force_all_tables=force_all_tables,
            )
            if tab_event.tables_detected > 0:
                logger.info(
                    "[Sub-Agent SQL Slide %d] Terdeteksi %d tabel | Status: %s | Baris: %d",
                    idx,
                    tab_event.tables_detected,
                    tab_event.status,
                    tab_event.rows_ingested_total,
                )

    stitched = stitch_pages_to_markdown(
        slide_markdowns,
        document_title=file_stem,
        include_page_markers=True,
        is_slide=True,
    )

    # Supervisor Guardrail Audit
    if auto_tabular_db and resolved_db_path:
        cross_verify_dual_track(
            stitched_markdown=stitched,
            db_path=resolved_db_path,
            source_file=str(path_obj.resolve()),
            total_pages=total_images,
        )

    return stitched
