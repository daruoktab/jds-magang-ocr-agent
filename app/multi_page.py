"""
Modul penyambung halaman dokumen multi-halaman (Multi-Page Document Stitcher).

Menyediakan logika cerdas untuk:
  - Menyambungkan Markdown dari banyak halaman PDF / Scan / PPT menjadi satu dokumen utuh.
  - Menjaga kontinuitas hierarki heading Markdown (#, ##, ###) antar halaman.
  - Membersihkan header & footer berulang (mis. "Halaman 1 dari 10", running headers).
  - Menyambung paragraf yang terpotong di akhir halaman.
  - Simulasi chunking (siap dimasukkan ke MarkdownHeaderTextSplitter / RecursiveTextSplitter).
"""

from __future__ import annotations

import re
from typing import Any

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


def _clean_page_artifacts(markdown: str) -> str:
    """Bersihkan artefak header/footer halaman umum."""
    lines = markdown.splitlines()
    cleaned_lines = []

    # Pola running page number: "Page 1", "Halaman 2 dari 10", "- 3 -", "1 / 15"
    page_num_pat = re.compile(
        r"^(halaman|page|\-)?\s*\d+\s*(dari|of|\/)?\s*\d*\s*(\-)?$", re.IGNORECASE
    )

    for line in lines:
        stripped = line.strip()
        if page_num_pat.match(stripped):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


PAGE_DELIMITER_RE = re.compile(
    r"<!--\s*(PAGE|SLIDE)\s*:\s*(\d+)\s*-->|<!--\s*(Page|Slide)\s*(\d+)\s*-->",
    re.IGNORECASE,
)


def format_page_delimiter(page_number: int, is_slide: bool = False) -> str:
    """Format penanda batas halaman/slide standar sistem."""
    tag = "SLIDE" if is_slide else "PAGE"
    return f"<!-- {tag}: {page_number} -->"


def split_markdown_by_pages(markdown: str) -> list[dict[str, Any]]:
    """
    Pisahkan teks Markdown berdasarkan penanda halaman/slide standar sistem
    (mis. `<!-- PAGE: 1 -->` atau `<!-- SLIDE: 1 -->`).

    Returns:
        List dictionary berisi page_number, type ('page' atau 'slide'), dan konten teksnya.
    """
    matches = list(PAGE_DELIMITER_RE.finditer(markdown))
    if not matches:
        return [{"page_number": 1, "type": "page", "content": markdown.strip()}]

    pages: list[dict[str, Any]] = []
    for i, m in enumerate(matches):
        tag_type = (m.group(1) or m.group(3) or "page").lower()
        page_num_str = m.group(2) or m.group(4) or str(i + 1)
        page_num = int(page_num_str)

        start_pos = m.end()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        content = markdown[start_pos:end_pos].strip()

        pages.append(
            {
                "page_number": page_num,
                "type": tag_type,
                "content": content,
            }
        )
    return pages


def stitch_pages_to_markdown(
    pages_markdown: list[str],
    *,
    document_title: str | None = None,
    include_page_markers: bool = True,
    is_slide: bool = False,
) -> str:
    """
    Gabungkan daftar Markdown per-halaman menjadi satu dokumen utuh yang konsisten.

    Args:
        pages_markdown: List string markdown dari setiap halaman (berurutan).
        document_title: Judul dokumen (opsional, akan menjadi heading # utama).
        include_page_markers: Jika True, sisipkan komentar `<!-- PAGE: N -->` / `<!-- SLIDE: N -->`.
        is_slide: Jika True, gunakan penanda `<!-- SLIDE: N -->`.

    Returns:
        String Markdown utuh siap dichunking.
    """
    if not pages_markdown:
        return ""

    stitched_blocks: list[str] = []

    if document_title:
        stitched_blocks.append(f"# {document_title.strip()}\n")

    for idx, page_md in enumerate(pages_markdown, start=1):
        cleaned_md = _clean_page_artifacts(page_md).strip()
        if not cleaned_md:
            continue

        if include_page_markers:
            stitched_blocks.append(
                f"\n{format_page_delimiter(idx, is_slide=is_slide)}\n"
            )

        # Cek kontinuitas paragraf: jika blok sebelumnya tidak diakhiri titik/header
        # dan halaman baru dimulai dengan huruf kecil, sambungkan secara mulus jika tanpa marker.
        if stitched_blocks and not include_page_markers:
            last_block = stitched_blocks[-1].rstrip()
            if last_block and not last_block.endswith(
                (".", ":", "!", "?", "#", ">", "|", "```")
            ):
                first_line = (
                    cleaned_md.splitlines()[0].strip()
                    if cleaned_md.splitlines()
                    else ""
                )
                if first_line and not first_line.startswith(
                    ("#", "-", "*", ">", "|", "1.", "2.")
                ):
                    stitched_blocks[-1] = last_block + " " + cleaned_md
                    continue

        stitched_blocks.append(cleaned_md)

    # Gabungkan dengan spasi paragraf ganda
    full_text = "\n\n".join(stitched_blocks)

    # Normalisasi spasi kosong berlebih
    full_text = re.sub(r"\n{3,}", "\n\n", full_text).strip()
    return full_text


def preview_markdown_chunks(
    markdown_text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[dict[str, Any]]:
    """
    Simulasikan pemecahan dokumen Markdown dengan splitter berbasis header & recursive text splitter.

    Returns:
        List potongan chunk dengan metadata header dan isi kontennya.
    """
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    md_header_splits = markdown_splitter.split_text(markdown_text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    final_splits = text_splitter.split_documents(md_header_splits)

    chunks_data = []
    for i, doc in enumerate(final_splits, start=1):
        chunks_data.append(
            {
                "chunk_index": i,
                "char_count": len(doc.page_content),
                "metadata": doc.metadata,
                "content": doc.page_content,
            }
        )

    return chunks_data
