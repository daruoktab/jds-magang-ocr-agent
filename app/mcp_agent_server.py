"""
MCP (Model Context Protocol) Server 'Agent Mode' untuk jds-magang-ocr-agent.

Server ini dirancang untuk agent model dengan kemampuan vision bawaan
(opencode, Claude Desktop, Cursor, dsb.) dan TIDAK memanggil endpoint model
apa pun: OCR, VLM, maupun LLM tidak diaktifkan. Agent-lah yang membaca gambar
dokumen secara visual, menulis Markdown-nya sendiri, lalu menyimpannya sebagai
gold data example untuk melatih agent/LLM lokal di kemudian hari.

Alur kerja yang direkomendasikan untuk agent:
  1. scan_document_folders / select_document_batch : temukan & pilih dokumen
  2. render_presentation_slides / convert_pdf_to_images (+ preprocess_image) :
     ubah dokumen menjadi gambar dan KIRIM gambar tersebut LANGSUNG ke model
     (image content block) agar model dapat melihat dan membaca dokumen secara visual
  3. Agent menulis Markdown sesuai spesifikasi layout dari gambar yang dilihat
  4. preview_markdown_chunks : validasi kesiapan chunking
  5. save_extraction_result : simpan Markdown + metadata sebagai gold data

Menjalankan server:
    python -m app.mcp_agent_server
    python mcp_agent_server.py
"""

from __future__ import annotations

import base64
import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.types import ContentBlock, ImageContent, TextContent

from .multi_page import preview_markdown_chunks as sim_preview_chunks
from .pdf import pdf_to_images
from .preprocess import preprocess_image

SUPPORTED_EXTENSIONS: set[str] = {
    ".pdf",
    ".pptx",
    ".ppt",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}

_IGNORED_DIR_NAMES: set[str] = {
    ".git",
    ".venv",
    "__pycache__",
    ".ruff_cache",
    ".vscode",
    ".agents",
}

# Batas ukuran gambar yang dikirim langsung ke model (10 MB).
_MAX_IMAGE_BYTES: int = 10 * 1024 * 1024

# Batas maksimal gambar yang dikirim dalam satu tool call.
DEFAULT_MAX_IMAGES: int = 10


def _mime_for(image_path: Path) -> str:
    """Tentukan MIME type gambar dari ekstensi file."""
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(image_path.suffix.lower(), "image/jpeg")


def _image_to_content(
    image_path: Path,
    *,
    max_bytes: int = _MAX_IMAGE_BYTES,
) -> ImageContent | None:
    """
    Baca file gambar dan kembalikan sebagai ImageContent (base64) agar dapat
    dikirim LANGSUNG ke model sebagai image content (model bisa melihat gambar).
    Mengembalikan None jika file tidak bisa dibaca atau terlalu besar.
    """
    try:
        data = image_path.read_bytes()
    except OSError:
        return None
    if len(data) > max_bytes:
        return None
    encoded = base64.b64encode(data).decode("ascii")
    return ImageContent(type="image", data=encoded, mime_type=_mime_for(image_path))


def _build_image_result(
    summary: dict[str, Any],
    image_paths: list[Path],
    *,
    max_images: int | None = None,
) -> list[ContentBlock]:
    """
    Susun tool result: TextContent berisi metadata ringkas, diikuti ImageContent
    dari gambar-gambar yang dikirim langsung ke model.
    Jika max_images None atau <= 0, kirim seluruh gambar tanpa batas artifisial.
    """
    content: list[ContentBlock] = [
        TextContent(type="text", text=json.dumps(summary, indent=2, ensure_ascii=False))
    ]
    target_images = (
        image_paths
        if max_images is None or max_images <= 0
        else image_paths[:max_images]
    )
    for img_path in target_images:
        ic = _image_to_content(Path(img_path))
        if ic is not None:
            content.append(ic)
    return content


# Inisialisasi Server MCP (Agent Mode)
server = MCPServer(
    name="jds-magang-doc-agent",
    description=(
        "Agent Mode Document Tools MCP Server (OCR/VLM/LLM endpoint NONAKTIF): "
        "alat bantu mekanis untuk agent vision — render dokumen (PPTX/PDF) menjadi gambar, "
        "seleksi acak non-duplikat dokumen, dan penyimpanan hasil ekstraksi Markdown buatan agent "
        "sebagai gold data example."
    ),
    instructions=(
        "Alur kerja wajib agent saat membuat gold data example:\n"
        "1. Panggil 'scan_document_folders' untuk menampilkan daftar folder dokumen yang tersedia.\n"
        "2. TANYAKAN KE USER folder mana yang datanya ingin diproses (user memilih, mis. 'input/ppt/english').\n"
        "3. TANYAKAN KE USER berapa banyak data random yang ingin dibuat (tanpa duplikasi).\n"
        "4. Panggil 'select_random_documents' dengan pilihan folder dan jumlah dari user.\n"
        "5. Untuk tiap dokumen terpilih: 'render_presentation_slides' (PPTX) atau 'convert_pdf_to_images' "
        "(PDF) — tool ini MENYALURKAN gambar LANGSUNG ke model sebagai image content, jadi model "
        "dapat melihat slide/halaman secara visual. Gunakan 'preprocess_image' bila gambar perlu "
        "diperbaiki orientasi/kontras (juga mengirim gambar hasil ke model).\n"
        "6. Tulis sendiri Markdown-nya dari gambar yang dilihat (OCR/ekstraksi teks tidak tersedia di server ini).\n"
        "7. Validasi dengan 'preview_markdown_chunks' bila perlu.\n"
        "8. Simpan hasil dengan 'save_extraction_result'.\n"
        "Jangan pernah memilih folder atau jumlah data sendiri tanpa persetujuan user."
    ),
    version="0.1.0",
)


def _scan_document_directories(root_dir: str | Path = ".") -> list[dict[str, Any]]:
    """
    Pindai root_dir dan seluruh sub-foldernya untuk mendeteksi folder berisi dokumen.

    Implementasi mandiri (ringan) agar server agent mode tidak menarik
    dependensi pipeline LLM (deep agent / graph) saat startup stdio.
    """
    root_path = Path(root_dir).resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Direktori tidak ditemukan: {root_path}")

    folder_map: dict[Path, list[Path]] = {}
    for p in root_path.rglob("*"):
        if any(part in _IGNORED_DIR_NAMES for part in p.parts):
            continue
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            folder_map.setdefault(p.parent, []).append(p)

    results: list[dict[str, Any]] = []
    for folder, files in sorted(folder_map.items(), key=lambda x: str(x[0])):
        try:
            rel = str(folder.relative_to(root_path))
            if rel == ".":
                rel = folder.name
        except ValueError:
            rel = str(folder)

        counts: dict[str, int] = {}
        for f in files:
            ext = f.suffix.lower()
            counts[ext] = counts.get(ext, 0) + 1

        results.append(
            {
                "folder_path": str(folder),
                "relative_path": rel,
                "total_documents": len(files),
                "extension_counts": counts,
                "sample_files": [f.name for f in files[:5]],
            }
        )

    return results


@server.tool(
    name="scan_document_folders",
    description=(
        "Pindai direktori (misal 'dataset', 'output', 'input', atau path khusus) dan sub-subfoldernya "
        "untuk mendeteksi keberadaan folder yang berisi file dokumen (PDF, PPTX, PPT, Gambar). "
        "Mengembalikan daftar folder yang tersedia, jumlah file per ekstensi, dan contoh nama file."
    ),
)
def scan_document_folders(
    root_dir: str = ".",
) -> str:
    """
    Pindai root_dir untuk menemukan folder-folder yang berisi file dokumen.
    """
    try:
        found_folders = _scan_document_directories(root_dir)
        return json.dumps(
            {
                "root_scanned": str(Path(root_dir).resolve()),
                "total_folders_found": len(found_folders),
                "folders": found_folders,
            },
            indent=2,
            ensure_ascii=False,
        )
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat memindai folder: {e}"


@server.tool(
    name="select_document_batch",
    description=(
        "Pilih daftar file dokumen dari satu atau banyak folder terpilih dengan kuota jumlah data "
        "(batas total 'limit' dan/atau batas per folder 'limit_per_folder') TANPA melakukan ekstraksi. "
        "Gunakan untuk menyiapkan antrean dokumen yang akan dibaca satu per satu secara visual oleh agent."
    ),
)
def select_document_batch(
    folders: str,
    limit: int | None = None,
    limit_per_folder: int | None = None,
) -> str:
    """
    Pilih daftar file dokumen dari folder-folder terpilih dengan kuota tertentu.

    Args:
        folders: Path folder atau daftar folder dipisah koma (mis. 'dataset/download/indonesian,dataset/download/english').
        limit: Batas total file yang dipilih.
        limit_per_folder: Batas file per folder.
    """
    if isinstance(folders, str):
        folder_strings = [f.strip() for f in folders.split(",") if f.strip()]
    else:
        folder_strings = [str(f).strip() for f in folders]

    per_folder: list[dict[str, Any]] = []
    selected: list[str] = []

    for f_str in folder_strings:
        p = Path(f_str).resolve()
        if not (p.exists() and p.is_dir()):
            per_folder.append(
                {"folder": f_str, "valid": False, "selected_count": 0, "files": []}
            )
            continue

        dir_files = sorted(
            f
            for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if limit_per_folder is not None and limit_per_folder > 0:
            dir_files = dir_files[:limit_per_folder]

        file_paths = [str(f) for f in dir_files]
        per_folder.append(
            {
                "folder": str(p),
                "valid": True,
                "selected_count": len(file_paths),
                "files": file_paths,
            }
        )
        selected.extend(file_paths)

    if limit is not None and limit > 0:
        selected = selected[:limit]

    return json.dumps(
        {
            "total_selected_files": len(selected),
            "selected_files": selected,
            "per_folder": per_folder,
        },
        indent=2,
        ensure_ascii=False,
    )


@server.tool(
    name="select_random_documents",
    description=(
        "Pilih 'count' file dokumen SECARA ACAK TANPA DUPLIKASI dari folder-folder terpilih oleh user. "
        "Secara default mengecualikan dokumen yang sudah memiliki hasil gold data di output_dir agar tidak "
        "diproses ganda (anti-duplikasi antar-run). Panggil tool ini SETELAH user memilih folder dan "
        "menyebutkan jumlah data random yang diinginkan."
    ),
)
def select_random_documents(
    folders: str,
    count: int,
    seed: int | None = None,
    exclude_already_extracted: bool = True,
    output_dir: str = "output/agent_gold",
) -> str:
    """
    Pilih dokumen secara acak tanpa duplikasi dari folder terpilih.

    Args:
        folders: Path folder atau daftar folder dipisah koma (pilihan user).
        count: Jumlah data random yang diminta user (tanpa duplikasi).
        seed: Seed opsional untuk seleksi acak yang dapat direproduksi.
        exclude_already_extracted: Jika True, skip dokumen yang sudah punya hasil gold di output_dir.
        output_dir: Direktori gold data untuk pengecekan anti-duplikasi.
    """
    if count is None or count <= 0:
        return "ERROR: 'count' harus berupa bilangan bulat positif (jumlah data random yang diminta user)."

    if isinstance(folders, str):
        folder_strings = [f.strip() for f in folders.split(",") if f.strip()]
    else:
        folder_strings = [str(f).strip() for f in folders]

    # Kumpulkan kandidat file unik (anti-duplikasi dalam seleksi)
    seen: set[Path] = set()
    candidates: list[Path] = []
    per_folder: list[dict[str, Any]] = []

    for f_str in folder_strings:
        p = Path(f_str).resolve()
        if not (p.exists() and p.is_dir()):
            per_folder.append(
                {
                    "folder": f_str,
                    "valid": False,
                    "available": 0,
                    "already_extracted": 0,
                    "selected": 0,
                }
            )
            continue

        dir_files = sorted(
            f
            for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        already_extracted = 0
        for f in dir_files:
            if f.resolve() in seen:
                continue
            if exclude_already_extracted and _has_existing_gold(f, output_dir):
                already_extracted += 1
                continue
            seen.add(f.resolve())
            candidates.append(f)

        per_folder.append(
            {
                "folder": str(p),
                "valid": True,
                "available": len(dir_files),
                "already_extracted": already_extracted,
                "selected": 0,
            }
        )

    valid_folders = [pf for pf in per_folder if pf["valid"]]
    if not valid_folders:
        return json.dumps(
            {
                "status": "error",
                "message": f"Tidak ada folder valid yang ditemukan dari input: {folders}",
                "selected_files": [],
                "selected_count": 0,
            },
            indent=2,
            ensure_ascii=False,
        )

    if not candidates:
        return json.dumps(
            {
                "status": "warning",
                "message": (
                    "Tidak ada kandidat dokumen tersisa: semua file sudah memiliki gold data "
                    "di output_dir atau folder kosong. Gunakan folder lain atau set "
                    "exclude_already_extracted=false untuk memproses ulang."
                ),
                "selected_files": [],
                "selected_count": 0,
                "per_folder": per_folder,
            },
            indent=2,
            ensure_ascii=False,
        )

    # Seleksi acak tanpa duplikasi
    rng = random.Random(seed)
    actual_count = min(count, len(candidates))
    picked = rng.sample(candidates, actual_count)
    picked_set = {p.resolve() for p in picked}

    # Distribusikan statistik terpilih per folder
    for pf in per_folder:
        if not pf["valid"]:
            continue
        folder_path = Path(pf["folder"])
        pf["selected"] = sum(1 for f in picked_set if f.parent == folder_path)

    return json.dumps(
        {
            "status": "success",
            "requested_count": count,
            "selected_count": actual_count,
            "excluded_already_extracted": (
                f"{sum(pf.get('already_extracted', 0) for pf in valid_folders)} dokumen dilewati "
                "karena sudah memiliki gold data"
                if exclude_already_extracted
                else 0
            ),
            "note": (
                f"Hanya {actual_count} dari {count} dokumen yang diminta tersedia."
                if actual_count < count
                else ""
            ),
            "seed_used": seed,
            "selected_files": [str(p) for p in picked],
            "per_folder": per_folder,
            "next_step": (
                "Render tiap dokumen terpilih dengan 'render_presentation_slides' (PPTX) atau "
                "'convert_pdf_to_images' (PDF), baca gambar secara visual, tulis Markdown-nya, "
                "lalu simpan dengan 'save_extraction_result'."
            ),
        },
        indent=2,
        ensure_ascii=False,
    )


def _has_existing_gold(doc_path: Path, output_dir: str | Path) -> bool:
    """Cek apakah dokumen sudah memiliki hasil gold data (untuk anti-duplikasi antar-run)."""
    out_base = Path(output_dir).resolve()
    return (out_base / f"{doc_path.stem}.md").exists()


@server.tool(
    name="render_presentation_slides",
    description=(
        "Render seluruh slide file presentasi PowerPoint (.pptx / .ppt) menjadi file gambar PNG "
        "beresolusi tinggi satu file per slide kanvas utuh, lalu KIRIM gambar-gambar tersebut LANGSUNG ke model "
        "sebagai image content agar model dapat melihat dan membaca slide secara visual. "
        "Default output: output/rendered_slides/<nama_file_tanpa_ekstensi>/. "
        "Secara otomatis mengirim seluruh slide sesuai panjang dokumen aslinya."
    ),
)
def render_presentation_slides(
    pptx_path: str,
    output_dir: str | None = None,
    max_images: int | None = None,
) -> list[ContentBlock] | str:
    """
    Render slide PPTX ke file gambar PNG per slide dan kirim gambar ke model.
    """
    path_obj = Path(pptx_path).resolve()
    if not path_obj.exists():
        return f"ERROR: File presentasi tidak ditemukan: {path_obj}"

    resolved_out = Path(
        output_dir or f"output/rendered_slides/{path_obj.stem}"
    ).resolve()
    # Hapus file gambar lama jika berisi ekstraksi parsial/stale
    if resolved_out.exists():
        for old_f in resolved_out.glob("slide_*_img_*.*"):
            try:
                old_f.unlink(missing_ok=True)
            except OSError:
                pass

    try:
        import importlib

        import app.ppt

        importlib.reload(app.ppt)
        images = app.ppt.render_presentation_slides_to_images(
            path_obj, output_dir=resolved_out
        )
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat merender slide presentasi: {e}"

    sent_count = (
        len(images)
        if (max_images is None or max_images <= 0)
        else min(max_images, len(images))
    )
    summary = {
        "pptx_file": str(path_obj),
        "total_slides_rendered": len(images),
        "rendered_image_paths": [str(p) for p in images],
        "images_sent_to_model": sent_count if images else 0,
    }
    return _build_image_result(summary, images, max_images=max_images)


@server.tool(
    name="convert_pdf_to_images",
    description=(
        "Konversi seluruh halaman file PDF menjadi file gambar beresolusi tinggi satu file per halaman, "
        "lalu KIRIM gambar-gambar tersebut LANGSUNG ke model sebagai image content agar model dapat "
        "membaca halaman PDF secara visual. Default output: output/rendered_pages/<nama_file_tanpa_ekstensi>/. "
        "Secara otomatis mengirim seluruh halaman dokumen aslinya. "
        "Gunakan 'dpi' untuk mengatur resolusi (default 200)."
    ),
)
def convert_pdf_to_images(
    pdf_path: str,
    output_dir: str | None = None,
    dpi: int = 200,
    max_images: int | None = None,
) -> list[ContentBlock] | str:
    """
    Konversi halaman-halaman PDF menjadi file gambar per halaman dan kirim gambar ke model.
    """
    path_obj = Path(pdf_path).resolve()
    if not path_obj.exists():
        return f"ERROR: File PDF tidak ditemukan: {path_obj}"
    if path_obj.suffix.lower() != ".pdf":
        return f"ERROR: File bukan PDF: {path_obj}"

    resolved_out = output_dir or f"output/rendered_pages/{path_obj.stem}"
    try:
        images = pdf_to_images(path_obj, output_dir=resolved_out, dpi=dpi)
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat mengonversi PDF ke gambar: {e}"

    sent_count = (
        len(images)
        if (max_images is None or max_images <= 0)
        else min(max_images, len(images))
    )
    summary = {
        "pdf_file": str(path_obj),
        "dpi": dpi,
        "total_pages_rendered": len(images),
        "rendered_image_paths": [str(p) for p in images],
        "images_sent_to_model": sent_count if images else 0,
    }
    return _build_image_result(summary, images, max_images=max_images)


@server.tool(
    name="extract_pdf_markdown_mupdf",
    description=(
        "Ekstrak teks dan tabel dokumen PDF digital langsung menjadi Markdown menggunakan PyMuPDF4LLM "
        "(C++ backend sangat cepat, zero-token). Berguna untuk mengambil grounding teks mentah per-halaman, "
        "tabel GFM, dan urutan baca 2-kolom sebagai referensi bantu bagi agent."
    ),
)
def extract_pdf_markdown_mupdf(
    pdf_path: str,
    page_chunks: bool = True,
) -> str:
    """
    Ekstrak konten teks & tabel PDF digital langsung ke format Markdown menggunakan PyMuPDF4LLM.
    """
    path_obj = Path(pdf_path).resolve()
    if not path_obj.exists():
        return f"ERROR: File PDF tidak ditemukan: {path_obj}"
    if path_obj.suffix.lower() != ".pdf":
        return f"ERROR: File bukan PDF: {path_obj}"

    try:
        from .pdf import extract_pdf_with_pymupdf4llm

        res = extract_pdf_with_pymupdf4llm(path_obj, page_chunks=page_chunks)
        if isinstance(res, list):
            return json.dumps(
                {
                    "status": "success",
                    "pdf_file": str(path_obj),
                    "total_pages": len(res),
                    "pages": res,
                },
                indent=2,
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "status": "success",
                "pdf_file": str(path_obj),
                "markdown": res,
            },
            indent=2,
            ensure_ascii=False,
        )
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat mengekstrak PDF dengan PyMuPDF4LLM: {e}"


@server.tool(
    name="preprocess_image",
    description=(
        "Pra-pemrosesan mekanis file gambar dokumen sebelum dibaca oleh model: perbaikan orientasi "
        "berdasarkan metadata EXIF dan peningkatan kontras untuk teks pudar / scan gelap. "
        "Hasil proses (gambar yang sudah diperbaiki) dikirim LANGSUNG ke model sebagai image content "
        "agar model dapat melihat versi gambar yang sudah bersih."
    ),
)
def preprocess_image_tool(
    image_path: str,
    contrast_factor: float = 1.2,
    output_dir: str | None = None,
) -> list[ContentBlock] | str:
    """
    Pra-pemrosesan gambar dokumen dan kirim gambar hasil ke model.
    """
    path_obj = Path(image_path).resolve()
    if not path_obj.exists():
        return f"ERROR: File gambar tidak ditemukan: {path_obj}"

    try:
        proc = preprocess_image(
            path_obj,
            auto_orient=True,
            enhance_contrast=True,
            contrast_factor=contrast_factor,
            output_dir=output_dir,
        )
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat memproses gambar: {e}"

    summary = {
        "original_path": proc.original_path,
        "processed_path": proc.processed_path,
        "is_modified": proc.is_modified,
        "dimensions": {"width": proc.dimensions[0], "height": proc.dimensions[1]},
    }
    return _build_image_result(summary, [Path(proc.processed_path)], max_images=1)


@server.tool(
    name="preview_markdown_chunks",
    description=(
        "Simulasikan pemecahan dokumen Markdown dengan splitter berbasis header (#, ##, ###) "
        "dan recursive character text splitter. Gunakan untuk memvalidasi kesiapan chunking "
        "dari Markdown yang telah ditulis agent sebelum disimpan sebagai gold data."
    ),
)
def preview_markdown_chunks(
    markdown_text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> str:
    """
    Simulasikan chunking pada teks Markdown.
    """
    try:
        chunks = sim_preview_chunks(
            markdown_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return json.dumps(
            {"total_chunks": len(chunks), "chunks": chunks},
            indent=2,
            ensure_ascii=False,
        )
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat simulasi chunking: {e}"


@server.tool(
    name="save_extraction_result",
    description=(
        "Simpan hasil ekstraksi Markdown yang telah ditulis sendiri oleh agent (hasil membaca gambar dokumen "
        "secara visual) sebagai gold data example untuk melatih agent/LLM lokal. File disimpan sebagai "
        "<nama_dokumen>.md di output_dir, disertai sidecar <nama_dokumen>.meta.json berisi metadata "
        "(file sumber, spesifikasi layout, struktur per-halaman/slide yang terdeteksi secara sistem, timestamp, statistik)."
    ),
)
def save_extraction_result(
    source_file: str,
    markdown: str,
    specs: str = "plain",
    output_dir: str = "output/agent_gold",
) -> str:
    """
    Simpan Markdown hasil ekstraksi agent beserta metadata gold data dan struktur halaman terstandar.

    Args:
        source_file: Path file dokumen sumber yang telah dibaca agent.
        markdown: Konten Markdown hasil ekstraksi yang ditulis agent.
        specs: Spesifikasi layout yang digunakan ('plain', 'markdown_hierarchy', 'bilingual_journal', 'presentation_slides', atau komposit).
        output_dir: Direktori tujuan penyimpanan gold data.
    """
    if not markdown or not markdown.strip():
        return "ERROR: Konten Markdown kosong; tidak ada yang disimpan."

    src = Path(source_file).resolve()
    if not src.exists():
        return f"ERROR: File sumber tidak ditemukan: {src}"

    try:
        from .multi_page import split_markdown_by_pages

        out_base = Path(output_dir).resolve()
        out_base.mkdir(parents=True, exist_ok=True)

        out_md = out_base / f"{src.stem}.md"
        clean_md = markdown.strip() + "\n"
        out_md.write_text(clean_md, encoding="utf-8")

        pages_parsed = split_markdown_by_pages(clean_md)
        page_structure = [
            {
                "page_number": p["page_number"],
                "type": p["type"],
                "char_count": len(p["content"]),
            }
            for p in pages_parsed
        ]

        metadata = {
            "source_file": str(src),
            "source_extension": src.suffix.lower(),
            "specs": [s.strip() for s in specs.split(",") if s.strip()],
            "extracted_by": "agent",
            "ocr_used": False,
            "saved_at": datetime.now(UTC).isoformat(),
            "markdown_path": str(out_md),
            "char_count": len(clean_md),
            "line_count": len(clean_md.splitlines()),
            "total_pages_detected": len(pages_parsed),
            "page_structure": page_structure,
        }

        out_meta = out_base / f"{src.stem}.meta.json"
        out_meta.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        return json.dumps(
            {"status": "success", **metadata}, indent=2, ensure_ascii=False
        )
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat menyimpan hasil ekstraksi: {e}"


def main() -> None:
    """Entry point untuk menjalankan MCP Server Agent Mode via stdio transport."""
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
