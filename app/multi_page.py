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


def split_markdown_by_pages(
    markdown: str,
    default_page_number: int = 1,
    default_type: str = "page",
) -> list[dict[str, Any]]:
    """
    Pisahkan teks Markdown berdasarkan penanda halaman/slide standar sistem
    (mis. `<!-- PAGE: 1 -->` atau `<!-- SLIDE: 1 -->`).

    Returns:
        List dictionary berisi page_number, type ('page' atau 'slide'), dan konten teksnya.
    """
    matches = list(PAGE_DELIMITER_RE.finditer(markdown))
    if not matches:
        return [
            {
                "page_number": default_page_number,
                "type": default_type,
                "content": markdown.strip(),
            }
        ]

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


def extract_preamble(markdown: str) -> str:
    """Ekstrak teks/judul sebelum penanda halaman pertama jika ada."""
    m = PAGE_DELIMITER_RE.search(markdown)
    if m:
        return markdown[: m.start()].strip()
    return ""


def merge_and_stitch_markdown_pages(
    existing_markdown: str | None,
    incoming_markdown: str,
    *,
    default_page_number: int = 1,
    is_slide: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Gabungkan potongan Markdown halaman/slide yang baru diproses dengan Markdown
    yang sudah ada sebelumnya (inkremental per batch 10 halaman).

    Args:
        existing_markdown: Konten Markdown yang sudah ada di disk (atau None/kosong jika baru).
        incoming_markdown: Konten batch Markdown baru dari agent.
        default_page_number: Nomor halaman/slide default jika incoming_markdown tidak memiliki penanda delimiter.
        is_slide: True jika dokumen adalah presentasi (menggunakan tag SLIDE).

    Returns:
        tuple berisi (merged_full_markdown, sorted_pages_metadata_list).
    """
    tag_default = "slide" if is_slide else "page"
    incoming_pages = split_markdown_by_pages(
        incoming_markdown,
        default_page_number=default_page_number,
        default_type=tag_default,
    )
    incoming_preamble = extract_preamble(incoming_markdown)

    pages_dict: dict[int, dict[str, Any]] = {}
    preamble = incoming_preamble

    if existing_markdown and existing_markdown.strip():
        existing_pages = split_markdown_by_pages(
            existing_markdown, default_type=tag_default
        )
        existing_preamble = extract_preamble(existing_markdown)
        if not preamble and existing_preamble:
            preamble = existing_preamble
        for p in existing_pages:
            pages_dict[p["page_number"]] = p

    # Timpa atau sisipkan halaman dari batch incoming
    for p in incoming_pages:
        pages_dict[p["page_number"]] = p

    sorted_page_numbers = sorted(pages_dict.keys())
    sorted_pages = [pages_dict[num] for num in sorted_page_numbers]

    blocks: list[str] = []
    if preamble:
        blocks.append(preamble)

    for p in sorted_pages:
        delimiter = format_page_delimiter(p["page_number"], is_slide=is_slide)
        raw_content = p["content"].strip()
        cleaned_content = re.sub(r"^(\s*---\s*\n)+", "", raw_content)
        cleaned_content = re.sub(r"(\n\s*---\s*)+$", "", cleaned_content).strip()
        blocks.append(f"{delimiter}\n{cleaned_content}")

    merged_full_markdown = "\n\n---\n\n".join(blocks).strip() + "\n"
    return merged_full_markdown, sorted_pages


def stitch_pages_to_markdown(
    pages_markdown: list[str] | list[Any],
    *,
    document_title: str | None = None,
    include_page_markers: bool = True,
    is_slide: bool = False,
    **kwargs: Any,
) -> str:
    """
    Gabungkan daftar Markdown per-halaman menjadi satu dokumen utuh yang konsisten.

    Args:
        pages_markdown: List string markdown (atau objek DocumentPage) dari setiap halaman (berurutan).
        document_title: Judul dokumen (opsional, akan menjadi heading # utama).
        include_page_markers: Jika True, sisipkan komentar `<!-- PAGE: N -->` / `<!-- SLIDE: N -->`.\n        is_slide: Jika True, gunakan penanda `<!-- SLIDE: N -->`.\n
    Returns:
        String Markdown utuh siap dichunking.
    """
    if not pages_markdown:
        return ""

    raw_pages: list[str] = []
    for item in pages_markdown:
        if isinstance(item, str):
            raw_pages.append(item)
        elif hasattr(item, "markdown_content"):
            raw_pages.append(item.markdown_content)
        elif isinstance(item, dict) and "content" in item:
            raw_pages.append(item["content"])
        else:
            raw_pages.append(str(item))

    stitched_blocks: list[str] = []

    if document_title:
        stitched_blocks.append(f"# {document_title.strip()}\n")

    for idx, page_md in enumerate(raw_pages, start=1):
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

    # Gabungkan dengan pemisah standar
    if include_page_markers:
        return "\n".join(stitched_blocks).strip() + "\n"
    return "\n\n".join(stitched_blocks).strip() + "\n"


def preview_markdown_chunks(
    markdown_content: str,
    *,
    source_file: str = "document",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> Any:
    """
    Simulasikan pemecahan dokumen Markdown menjadi chunk-chunk siap RAG
    menggunakan kombinasi MarkdownHeaderTextSplitter dan RecursiveCharacterTextSplitter.
    """
    from app.schemas import ChunkingPreview, ChunkItem

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    # Level 1: Split berdasarkan heading struktur
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    header_splits = markdown_splitter.split_text(markdown_content)

    # Level 2: Split rekursif berbasis karakter & overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    final_docs = text_splitter.split_documents(header_splits)

    items: list[ChunkItem] = []
    for idx, doc in enumerate(final_docs, start=1):
        content = doc.page_content.strip()
        preview = content[:120].replace("\n", " ")
        items.append(
            ChunkItem(
                chunk_id=idx,
                char_count=len(content),
                token_estimate=max(1, len(content) // 4),
                preview=preview,
                content=content,
            )
        )

    total_chars = sum(c.char_count for c in items)
    avg_size = total_chars / len(items) if items else 0.0

    return ChunkingPreview(
        source_file=source_file,
        total_characters=total_chars,
        total_chunks=len(items),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        avg_chunk_size=round(avg_size, 1),
        chunks=items,
    )
