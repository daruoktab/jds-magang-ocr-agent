"""
Modul pemrosesan dokumen presentasi PowerPoint (.pptx / .ppt).

Mengekstrak slide presentasi menjadi teks Markdown terstruktur yang siap dichunking:
  - Menjaga judul slide (`## Slide N: [Judul]`)
  - Menjaga hierarki bullet points (poin-poin bertingkat)
  - Mengonversi tabel presentasi ke format Markdown Table (GFM)
  - Mengekstrak catatan pembicara (*speaker notes*)
  - Merender slide presentasi ke gambar resolusi tinggi untuk analisis VLM
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE


def _table_to_markdown(table: Any) -> str:
    """Ubah objek tabel python-pptx menjadi teks tabel Markdown (GFM)."""
    rows: list[list[str]] = []
    for row in table.rows:
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


def render_presentation_slides_to_images(
    pptx_path: str | Path,
    output_dir: str | Path | None = None,
) -> list[Path]:
    """
    Render seluruh kanvas slide PowerPoint menjadi file gambar PNG (satu gambar per slide kanvas).
    Menggunakan Spire.Presentation untuk Python (standalone, cross-platform, tanpa perlu Microsoft Office/LibreOffice).
    """
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

    prs = Presentation()
    try:
        prs.LoadFromFile(str(path_obj))
        generated_images: list[Path] = []
        for i in range(prs.Slides.Count):
            slide = prs.Slides[i]
            image = slide.SaveAsImage()
            out_img = (out_dir / f"slide_{i + 1}.png").resolve()
            image.Save(str(out_img))
            if out_img.exists():
                generated_images.append(out_img)
        return generated_images
    finally:
        prs.Dispose()


def pptx_to_structured_text(pptx_path: str | Path) -> list[dict[str, Any]]:
    """
    Ekstrak presentasi PPTX menjadi list struktur per slide.
    """
    path_obj = Path(pptx_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"File presentasi tidak ditemukan: {path_obj}")

    prs = Presentation(str(path_obj))
    slides_data: list[dict[str, Any]] = []

    for idx, slide in enumerate(prs.slides, start=1):
        slide_title: str = ""
        body_lines: list[str] = []
        notes_text: str = ""
        image_count: int = 0

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

        slides_data.append(
            {
                "slide_number": idx,
                "title": title_display,
                "markdown": slide_markdown,
                "notes": notes_text,
                "image_count": image_count,
            }
        )

    return slides_data


def process_presentation(pptx_path: str | Path) -> str:
    """
    Ekstrak seluruh file PPTX menjadi satu dokumen Markdown terpadu siap chunking.
    """
    slides = pptx_to_structured_text(pptx_path)
    file_stem = Path(pptx_path).stem.replace("_", " ").title()

    doc_lines: list[str] = [f"# {file_stem}\n"]
    for s in slides:
        doc_lines.append(s["markdown"])
        doc_lines.append("\n---\n")

    return "\n".join(doc_lines).strip()
