"""
Modul Pemrosesan Dokumen Presentasi PowerPoint (.pptx / .ppt).

Menyediakan fungsi untuk:
  - Render slide presentasi menjadi gambar (DPI tinggi) menggunakan LibreOffice/soffice CLI atau pptx2pdf + PyMuPDF.
  - Membaca teks asli, shape, diagram, dan tabel per slide via python-pptx (opsional).
  - Ekstraksi visual slide menggunakan Vision LLM langsung dari kanvas gambar render.
  - Menjalankan sub-agent SQL tabular mandiri per-slide dan audit guardrail jalur ganda di tahap akhir.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from .config import DEFAULT_DPI
from .llm import encode_image_to_base64
from .multi_page import stitch_pages_to_markdown
from .prompts import get_vision_system_prompt
from .tabular_db import cross_verify_dual_track, process_page_tabular_agent

logger = logging.getLogger(__name__)


def _find_libreoffice_binary() -> str | None:
    """Temukan binary LibreOffice / soffice di sistem Windows atau Linux."""
    # 1. Cek di PATH
    for name in ("soffice", "libreoffice", "soffice.exe", "libreoffice.exe"):
        found = shutil.which(name)
        if found:
            return found

    # 2. Lokasi standar Windows
    if platform.system() == "Windows":
        candidates = [
            Path(r"C:\Program Files\LibreOffice\program\soffice.exe"),
            Path(r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"),
            Path(os.environ.get("PROGRAMFILES", "C:\\Program Files"))
            / "LibreOffice"
            / "program"
            / "soffice.exe",
            Path(os.environ.get("LOCALAPPDATA", ""))
            / "Programs"
            / "LibreOffice"
            / "program"
            / "soffice.exe",
        ]
        for c in candidates:
            if c.exists():
                return str(c)

    # 3. Lokasi standar Linux / macOS
    for loc in ("/usr/bin/soffice", "/usr/bin/libreoffice", "/Applications/LibreOffice.app/Contents/MacOS/soffice"):
        if Path(loc).exists():
            return loc

    return None


def count_presentation_slides(presentation_path: str | Path) -> int:
    """Hitung jumlah total slide dalam file PowerPoint."""
    path_obj = Path(presentation_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"File presentasi tidak ditemukan: {path_obj}")

    # Cara 1: python-pptx (sangat cepat, tidak perlu render)
    try:
        from pptx import Presentation

        prs = Presentation(str(path_obj))
        return len(prs.slides)
    except Exception:  # noqa: BLE001, S110
        pass

    # Cara 2: Konversi ke PDF via LibreOffice lalu hitung halamannya
    from .pdf import pdf_page_count

    pdf_out = _convert_presentation_to_pdf(path_obj)
    if pdf_out and pdf_out.exists():
        return pdf_page_count(pdf_out)

    return 1


def _convert_presentation_to_pdf(
    presentation_path: Path,
    output_dir: Path | None = None,
) -> Path | None:
    """Konversi file .pptx / .ppt menjadi .pdf menggunakan LibreOffice headless."""
    soffice = _find_libreoffice_binary()
    if not soffice:
        logger.warning(
            "LibreOffice (soffice) tidak ditemukan di sistem. "
            "Rendering visual slide presentasi mungkin terbatas."
        )
        return None

    target_dir = output_dir or Path(tempfile.mkdtemp(prefix="pptx_pdf_"))
    target_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        soffice,
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        str(target_dir),
        str(presentation_path),
    ]

    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if res.returncode != 0:
            logger.error("LibreOffice konversi PDF gagal: %s", res.stderr)
            return None

        pdf_candidate = target_dir / f"{presentation_path.stem}.pdf"
        if pdf_candidate.exists():
            return pdf_candidate
    except Exception as exc:  # noqa: BLE001
        logger.error("Gagal menjalankan perintah LibreOffice: %s", exc)

    return None


def _render_slides_native_pptx(
    presentation_path: Path,
    output_dir: Path,
    slides: list[int] | None = None,
    dpi: int = 200,
    image_ext: str = "jpg",
) -> list[Path]:
    """Render slide presentasi menggunakan konversi PDF + PyMuPDF."""
    from .pdf import pdf_to_images

    # Buat sub-folder sementara untuk PDF perantara
    temp_pdf_dir = Path(tempfile.mkdtemp(prefix="pptx_render_"))
    pdf_file = _convert_presentation_to_pdf(presentation_path, output_dir=temp_pdf_dir)

    if not pdf_file or not pdf_file.exists():
        raise RuntimeError(
            f"Gagal mengonversi presentasi '{presentation_path.name}' ke PDF untuk rendering visual."
        )

    images = pdf_to_images(
        pdf_path=pdf_file,
        output_dir=output_dir,
        dpi=dpi,
        image_ext=image_ext,
        pages=slides,
    )

    # Ubah penamaan file agar sesuai konvensi slide: slide_0001.jpg
    renamed: list[Path] = []
    for img in images:
        # Contoh: page_0001.jpg -> slide_0001.jpg
        stem = img.stem
        if stem.startswith("page_"):
            slide_num = stem.replace("page_", "")
            new_name = img.parent / f"slide_{slide_num}.{image_ext}"
            img.rename(new_name)
            renamed.append(new_name)
        else:
            renamed.append(img)

    return renamed


def render_presentation_slides_to_images(
    presentation_path: str | Path,
    output_dir: str | Path | None = None,
    slides: list[int] | None = None,
    dpi: int = 200,
    image_ext: str = "jpg",
) -> list[Path]:
    """
    Render seluruh slide PowerPoint menjadi file gambar beresolusi tinggi (DPI tinggi).

    Args:
        presentation_path: Path ke file presentasi .pptx / .ppt.
        output_dir: Folder output untuk menyimpan gambar slide.
        slides: Daftar indeks slide 0-indexed yang akan dirender (None = semua slide).
        dpi: Resolusi rendering (default 200 DPI).
        image_ext: Format file gambar ("png" atau "jpg").

    Returns:
        Daftar Path file gambar hasil render (berurutan).
    """
    path_obj = Path(presentation_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"File presentasi tidak ditemukan: {path_obj}")

    if output_dir is None:
        out_path = Path("output/pptx_slides") / path_obj.stem
    else:
        out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    return _render_slides_native_pptx(
        presentation_path=path_obj,
        output_dir=out_path,
        slides=slides,
        dpi=dpi,
        image_ext=image_ext,
    )


def pptx_to_structured_text(presentation_path: str | Path) -> list[dict[str, Any]]:
    """
    Ekstrak teks, tabel, dan catatan pembicara (speaker notes) dari file PowerPoint secara native.
    """
    path_obj = Path(presentation_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"File presentasi tidak ditemukan: {path_obj}")

    from pptx import Presentation

    prs = Presentation(str(path_obj))
    slides_data: list[dict[str, Any]] = []

    for idx, slide in enumerate(prs.slides, start=1):
        slide_title = ""
        paragraphs: list[str] = []
        tables_extracted: list[list[list[str]]] = []

        # Ekstrak elemen bentuk (shapes)
        for shape in slide.shapes:
            if shape.has_text_frame:
                text_frame = shape.text_frame
                text = text_frame.text.strip()
                if not text:
                    continue

                if shape == slide.shapes.title:
                    slide_title = text
                else:
                    for p in text_frame.paragraphs:
                        p_text = p.text.strip()
                        if p_text:
                            level = p.level
                            prefix = "  " * level + "- " if level > 0 else "- "
                            paragraphs.append(f"{prefix}{p_text}")

            elif shape.has_table:
                tbl = shape.table
                table_rows: list[list[str]] = []
                for row in tbl.rows:
                    row_cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                    table_rows.append(row_cells)
                tables_extracted.append(table_rows)

        # Speaker notes jika ada
        notes_text = ""
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
            notes_text = slide.notes_slide.notes_text_frame.text.strip()

        # Susun Markdown per slide
        md_lines: list[str] = []
        md_lines.append(f"<!-- SLIDE: {idx} -->")
        if slide_title:
            md_lines.append(f"# {slide_title}\n")
        elif paragraphs:
            md_lines.append(f"# Slide {idx}\n")

        if paragraphs:
            md_lines.extend(paragraphs)
            md_lines.append("")

        # Format tabel jika ada
        for tbl in tables_extracted:
            if not tbl:
                continue
            header = tbl[0]
            md_lines.append("| " + " | ".join(header) + " |")
            md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
            for r in tbl[1:]:
                padded = r + [""] * (len(header) - len(r))
                md_lines.append("| " + " | ".join(padded[: len(header)]) + " |")
            md_lines.append("")

        if notes_text:
            md_lines.append(f"> **Speaker Notes:** {notes_text}\n")

        slides_data.append(
            {
                "slide_number": idx,
                "title": slide_title,
                "paragraphs": paragraphs,
                "tables_count": len(tables_extracted),
                "has_notes": bool(notes_text),
                "markdown": "\n".join(md_lines).strip(),
            }
        )

    return slides_data


def _extract_slide_markdown(
    llm: BaseChatModel,
    image_path: str | Path,
    slide_number: int,
    total_slides: int,
    previous_slide_context: str | None = None,
) -> str:
    """Ekstrak konten satu gambar slide PowerPoint menggunakan Vision LLM."""
    base64_img, mime = encode_image_to_base64(image_path)
    img_data_url = f"data:{mime};base64,{base64_img}"

    system_prompt = get_vision_system_prompt(["presentation_slides"])

    context_prompt = ""
    if previous_slide_context:
        context_prompt = (
            f"\n\n[Konteks Slide Sebelumnya #{slide_number - 1}]:\n"
            f"'''\n{previous_slide_context[-300:]}\n'''\n"
            "Gunakan konteks ini untuk menjaga kesinambungan poin bahasan jika slide ini merupakan kelanjutan topik."
        )

    user_instruction = (
        f"Ekstrak Slide Presentasi #{slide_number} dari total {total_slides} slide.{context_prompt}\n\n"
        "Aturan Khusus Slide Presentasi:\n"
        f"1. Awali hasil dengan penanda `<!-- SLIDE: {slide_number} -->`.\n"
        "2. Judul slide jadikan `# Judul Slide`.\n"
        "3. Poin-poin peluru jadikan `- Poin` dengan indentasi yang tepat jika bertingkat.\n"
        "4. Jika terdapat tabel, tulis sebagai tabel Markdown standar.\n"
        "5. Jika terdapat diagram alur/hierarki visual sederhana, buatkan ```mermaid jika memungkinkan atau deskripsikan secara runtut.\n"
        "6. Jangan berikan teks pembuka atau penutup basa-basi, langsung hasilkan Markdown."
    )

    msg = HumanMessage(
        content=[
            {"type": "text", "text": f"{system_prompt}\n\n{user_instruction}"},
            {"type": "image_url", "image_url": {"url": img_data_url}},
        ]
    )

    response = llm.invoke([msg])
    content = response.content if hasattr(response, "content") else str(response)
    if isinstance(content, list):
        text_parts = [p.get("text", "") for p in content if isinstance(p, dict)]
        content = "".join(text_parts)

    return str(content).strip()


def pptx_to_markdown_native(pptx_path: str | Path) -> str:
    """Ekstraksi teks slide PPTX secara instan murni berbasis python-pptx."""
    slides = pptx_to_structured_text(pptx_path)
    file_stem = Path(pptx_path).stem.replace("_", " ").title()

    doc_lines: list[str] = [f"# {file_stem}\n"]
    for s in slides:
        doc_lines.append(s["markdown"])
        doc_lines.append("\n---\n")

    return "\n".join(doc_lines).strip()


def process_presentation_vision(
    pptx_path: str | Path,
    pipeline: Any = None,
    *,
    llm: BaseChatModel | None = None,
    output_dir: str | Path | None = None,
    dpi: int = DEFAULT_DPI,
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
        dpi=dpi,
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

    vlm_llm: BaseChatModel | None = llm
    if vlm_llm is None and pipeline is not None:
        vlm_llm = getattr(pipeline, "vlm", pipeline)
    if vlm_llm is None:
        from .llm import get_vlm

        vlm_llm = get_vlm()

    # Tentukan path SQLite jika aktif
    if db_path:
        resolved_db_path: Path | None = Path(db_path).resolve()
    elif output_dir:
        resolved_db_path = Path(output_dir).resolve() / "databases" / f"{path_obj.stem}.sqlite"
    else:
        resolved_db_path = Path("output/databases").resolve() / f"{path_obj.stem}.sqlite"

    if resolved_db_path and auto_tabular_db:
        resolved_db_path.parent.mkdir(parents=True, exist_ok=True)

    if pipeline is None:
        from .graph import DocumentExtractionPipeline
        pipeline = DocumentExtractionPipeline(vlm=vlm_llm)
    elif hasattr(pipeline, "invoke") and not hasattr(pipeline, "run"):
        from .graph import DocumentExtractionPipeline
        pipeline = DocumentExtractionPipeline(vlm=pipeline)

    for idx, img_path in enumerate(slide_images, start=1):
        logger.info(
            "[Vision PPT] [Slide %d/%d] Memproses slide '%s' via DocumentExtractionPipeline...",
            idx,
            total_images,
            img_path.name,
        )
        if hasattr(pipeline, "run"):
            res = pipeline.run(
                str(img_path),
                forced_specs=forced_specs or "presentation_slides",
                previous_page_context=previous_context,
            )
            page_md = res.get("markdown_content", "")
        else:
            page_md = _extract_slide_markdown(
                llm=vlm_llm,
                image_path=str(img_path),
                slide_number=idx,
                total_slides=total_images,
                previous_slide_context=previous_context,
            )

        # Pastikan penanda slide ada
        if not page_md.startswith(f"<!-- SLIDE: {idx} -->") and not page_md.startswith(f"<!-- slide: {idx} -->"):
            page_md = f"<!-- SLIDE: {idx} -->\n" + page_md

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
            if tab_event.tagged_markdown:
                page_md = tab_event.tagged_markdown
                slide_markdowns[-1] = page_md

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


convert_presentation_to_pdf = _convert_presentation_to_pdf
process_presentation = pptx_to_markdown_native
