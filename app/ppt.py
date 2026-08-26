"""
Modul pemrosesan dokumen presentasi PowerPoint (.pptx / .ppt).

Alur presentasi dijaga sederhana dan eksplisit:
  - PPT/PPTX dirender menjadi gambar per slide
  - Setiap gambar slide dikirim langsung ke VLM Qwen 35B Vision
  - Hasilnya digabung menjadi Markdown per-slide yang siap dichunking
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

from .config import get_ppt_renderer
from .llm import image_data_uri
from .multi_page import stitch_pages_to_markdown

logger = logging.getLogger("app.ppt")

# Batas lisensi Spire.Presentation Free: hanya 10 slide pertama per objek
# presentasi yang dirender penuh; slide ke-11 dst. menjadi blank + watermark.
# Karena itu rendering selalu dilakukan bertahap dalam batch sebesar nilai ini.
SPIRE_FREE_SLIDE_LIMIT: int = 10

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


def render_presentation_slides_to_images_libreoffice(
    presentation_path: str | Path,
    output_dir: str | Path | None = None,
    slides: list[int] | None = None,
    dpi: int = 200,
) -> list[Path]:
    """Render slide presentasi menjadi gambar via LibreOffice -> PDF -> PyMuPDF."""
    import fitz

    source = Path(presentation_path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"File presentasi tidak ditemukan: {source}")

    out_dir = (
        Path(output_dir).resolve()
        if output_dir
        else (source.parent / f"{source.stem}_slides").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="libreoffice_pdf_") as temp_pdf_dir:
        pdf_path = convert_presentation_to_pdf(source, temp_pdf_dir)
        doc = fitz.open(pdf_path)
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
        matrix = fitz.Matrix(zoom, zoom)
        generated_paths: list[Path] = []

        for idx in target_indices:
            page = doc[idx]
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            slide_num = idx + 1
            img_path = out_dir / f"slide_{slide_num}.png"
            pix.save(str(img_path))
            generated_paths.append(img_path)

        doc.close()
        return generated_paths


def count_presentation_slides_libreoffice(
    presentation_path: str | Path,
) -> int:
    """Hitung jumlah slide menggunakan LibreOffice -> PDF."""
    import fitz

    source = Path(presentation_path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"File presentasi tidak ditemukan: {source}")

    with tempfile.TemporaryDirectory(prefix="ppt_count_") as temp_pdf_dir:
        pdf_path = convert_presentation_to_pdf(source, temp_pdf_dir)
        doc = fitz.open(pdf_path)
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
    table = (
        table_or_shape.table
        if hasattr(table_or_shape, "table")
        else table_or_shape
    )
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


def count_presentation_slides(
    pptx_path: str | Path,
    renderer: Literal["spire", "libreoffice"] | None = None,
) -> int:
    """
    Hitung jumlah slide presentasi menggunakan backend yang dipilih (default dari konfigurasi).
    """
    active_renderer = (renderer or get_ppt_renderer()).strip().lower()
    if active_renderer == "libreoffice":
        return count_presentation_slides_libreoffice(pptx_path)
    if active_renderer != "spire":
        raise ValueError(f"Renderer tidak dikenal: {active_renderer}")

    from spire.presentation import Presentation

    path_obj = Path(pptx_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"File presentasi tidak ditemukan: {path_obj}")

    prs = Presentation()
    try:
        prs.LoadFromFile(str(path_obj))
        return prs.Slides.Count
    finally:
        prs.Dispose()


def _split_pptx_to_temp(
    src_path: Path,
    keep_start: int,
    keep_end: int,
    tmp_dir: Path,
) -> Path:
    """
    Buat salinan PPTX yang hanya mempertahankan slide indeks [keep_start, keep_end)
    (0-based) dan menyimpannya ke tmp_dir. Menggunakan python-pptx (tanpa batas lisensi).

    Slide di luar rentang dihapus melalui manipulasi `sldIdLst` + `drop_rel`,
    lalu disimpan sebagai file PPTX bersih (maksimal 10 slide) untuk di-render Spire.
    """
    from pptx import Presentation

    prs = Presentation(str(src_path))
    total = len(prs.slides)
    sldIdLst = prs.slides._sldIdLst
    slides = list(sldIdLst)

    # Hapus slide di luar rentang (urutkan descending agar indeks tetap valid).
    to_remove = [i for i in range(total) if not (keep_start <= i < keep_end)]
    for i in sorted(to_remove, reverse=True):
        prs.part.drop_rel(slides[i].rId)
        sldIdLst.remove(slides[i])

    out_path = tmp_dir / f"batch_{keep_start}_{keep_end}.pptx"
    prs.save(str(out_path))
    return out_path


def render_presentation_slides_to_images(
    pptx_path: str | Path,
    output_dir: str | Path | None = None,
    slides: list[int] | None = None,
    batch_size: int = SPIRE_FREE_SLIDE_LIMIT,
    renderer: Literal["spire", "libreoffice"] | None = None,
) -> list[Path]:
    """
    Render kanvas slide PowerPoint menjadi file gambar PNG (satu gambar per slide kanvas).
    Menggunakan renderer yang aktif (default dari PPT_RENDERER / konfigurasi terpusat).

    Args:
        pptx_path: Path file presentasi (.pptx / .ppt).
        output_dir: Direktori penyimpanan gambar (default: <folder_pptx>/<stem>_slides).
        slides: Daftar indeks slide 0-based yang ingin dirender (None = seluruh slide).
        batch_size: Jumlah slide per batch render (default 10 = limit lisensi Free;
            jangan dinaikkan melebihi 10 pada versi Free).
        renderer: Backend rendering, ``"spire"`` atau ``"libreoffice"`` (None = mengikuti konfigurasi).

    Nama file: `slide_<N>.png` (N mulai dari 1, sesuai nomor slide asli).
    Mengembalikan daftar path gambar yang dihasilkan (berurutan sesuai indeks input).
    """
    active_renderer = (renderer or get_ppt_renderer()).strip().lower()

    if active_renderer == "libreoffice":
        return render_presentation_slides_to_images_libreoffice(
            presentation_path=pptx_path,
            output_dir=output_dir,
            slides=slides,
        )
    if active_renderer != "spire":
        raise ValueError(f"Renderer tidak dikenal: {active_renderer}")

    from spire.presentation import Presentation

    path_obj = Path(pptx_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"File presentasi tidak ditemukan: {path_obj}")

    out_dir = (
        Path(output_dir).resolve()
        if output_dir
        else (path_obj.parent / f"{path_obj.stem}_slides").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    is_pptx = path_obj.suffix.lower() == ".pptx"

    # Hitung total slide: python-pptx untuk .pptx, Spire untuk .ppt.
    if is_pptx:
        from pptx import Presentation as PptxPresentation

        probe = PptxPresentation(str(path_obj))
        total = len(probe.slides)
    else:
        src_probe = Presentation()
        src_probe.LoadFromFile(str(path_obj))
        total = src_probe.Slides.Count
        src_probe.Dispose()

    if total == 0:
        return []

    if slides is None:
        target_indices = list(range(total))
    else:
        target_indices = [i for i in slides if 0 <= i < total]
    if not target_indices:
        return []

    # Clamp: versi Free tidak bisa merender > 10 slide per objek presentasi.
    effective_batch = max(1, min(batch_size, SPIRE_FREE_SLIDE_LIMIT))

    generated_images: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="pptx_batch_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        for b_start in range(0, len(target_indices), effective_batch):
            chunk = target_indices[b_start : b_start + effective_batch]

            batch_prs = Presentation()
            try:
                if is_pptx:
                    # Potong PPTX menjadi file kecil (hanya slide di chunk), lalu render.
                    split_path = _split_pptx_to_temp(
                        path_obj, chunk[0], chunk[-1] + 1, tmp_dir
                    )
                    batch_prs.LoadFromFile(str(split_path))
                else:
                    # Fallback .ppt: AppendBySlide ke presentasi baru.
                    src = Presentation()
                    src.LoadFromFile(str(path_obj))
                    try:
                        batch_prs.Slides.RemoveAt(0)
                        try:
                            batch_prs.SlideSize.Size = src.SlideSize.Size
                        except Exception:  # noqa: BLE001, S110
                            pass
                        for idx in chunk:
                            batch_prs.Slides.AppendBySlide(src.Slides[idx])
                    finally:
                        src.Dispose()

                for pos, src_idx in enumerate(chunk):
                    image = batch_prs.Slides[pos].SaveAsImage()
                    out_img = (out_dir / f"slide_{src_idx + 1}.png").resolve()
                    image.Save(str(out_img))
                    if out_img.exists():
                        generated_images.append(out_img)
            finally:
                batch_prs.Dispose()

    return generated_images


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
    logger.info("[Workflow] Membaca presentasi: '%s' | Total slide: %d", path_obj.name, total_slides)

    slides_data: list[dict[str, Any]] = []

    for idx, slide in enumerate(prs.slides, start=1):
        slide_title: str = ""
        body_lines: list[str] = []
        notes_text: str = ""
        image_count: int = 0
        table_count: int = 0

        # 1. Ambil judul slide jika ada
        if slide.shapes.title and slide.shapes.title.text.strip():
            slide_title = slide.shapes.title.text.strip()

        # 2. Iterasi shape dalam slide
        for shape in slide.shapes:
            if shape == slide.shapes.title:
                continue

            # A. Gambar & Media
            if shape.shape_type in (MSO_SHAPE_TYPE.PICTURE, MSO_SHAPE_TYPE.MEDIA):
                image_count += 1
                name = getattr(shape, "name", f"Image_{image_count}")
                body_lines.append(f"*[Visual / Diagram: {name}]*")

            # B. Tabel
            elif shape.has_table:
                table_count += 1
                md_table = _table_to_markdown(shape.table)
                if md_table:
                    body_lines.append(md_table)

            # C. Teks & Bullet points
            elif shape.has_text_frame:
                tf = shape.text_frame
                for p in tf.paragraphs:
                    text = p.text.strip()
                    if not text:
                        continue
                    if not slide_title and not body_lines:
                        slide_title = text
                        continue

                    level: int = getattr(p, "level", 0)
                    indent: str = "  " * level
                    body_lines.append(f"{indent}- {text}")

        # 3. Ambil speaker notes
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
            nt = slide.notes_slide.notes_text_frame.text.strip()
            if nt:
                notes_text = nt

        # 4. Susun markdown per slide
        title_display = slide_title or "Tanpa Judul"
        md_content_lines = [f"## Slide {idx}: {title_display}\n"]
        if body_lines:
            md_content_lines.append("\n".join(body_lines))
        if notes_text:
            md_content_lines.append(f"\n> **Speaker Notes:** {notes_text}")

        slide_markdown = "\n".join(md_content_lines)

        logger.info(
            "[Slide %d/%d] Selesai: '%s' | Shapes/Items: %d | Tabel: %d | Gambar: %d",
            idx,
            total_slides,
            title_display[:30] + ("..." if len(title_display) > 30 else ""),
            len(body_lines),
            table_count,
            image_count,
        )

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
    logger.info("[Workflow] Selesai parsing %d slide dalam %.2fs", total_slides, duration)
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
    renderer: Literal["spire", "libreoffice"] | None = None,
    forced_specs: list[str] | str | None = "presentation_slides",
) -> str:
    """Render slide PPT/PPTX menjadi gambar, lalu kirim setiap gambar langsung ke VLM."""
    path_obj = Path(pptx_path).resolve()

    logger.info("[Vision PPT] Memulai rendering slide menjadi gambar PNG kanvas...")
    slide_images = render_presentation_slides_to_images(
        pptx_path=path_obj,
        output_dir=output_dir,
        renderer=renderer,
    )
    total_images = len(slide_images)
    if total_images == 0:
        raise RuntimeError(f"Tidak ada slide yang berhasil dirender dari: {path_obj}")

    logger.info(
        "[Vision PPT] Selesai render %d slide gambar. Mengirim setiap gambar ke Vision Model (VLM)...",
        total_images,
    )

    slide_markdowns: list[str] = []
    previous_context: str | None = None
    file_stem = path_obj.stem.replace("_", " ").title()
    llm: BaseChatModel = getattr(pipeline, "vlm", None)
    if llm is None:
        raise AttributeError("pipeline harus menyediakan atribut 'vlm' untuk ekstraksi PPT Vision")

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

    return stitch_pages_to_markdown(
        slide_markdowns,
        document_title=file_stem,
        include_page_markers=True,
        is_slide=True,
    )
