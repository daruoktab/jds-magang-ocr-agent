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
     (image content) agar model dapat melihat dan membaca dokumen secara visual.
     Karena library konversi (Spire.Presentation) hanya merender maksimal 10 slide per
     objek presentasi, kedua tool ini memproses DENGAN BATCH per 10 slide/halaman:
     ulangi panggilan dengan start_slide/start_page = next_start_... hingga has_more=false.
  3. Agent menulis Markdown sesuai spesifikasi layout dari gambar yang dilihat
  4. preview_markdown_chunks : validasi kesiapan chunking
  5. save_extraction_result : simpan Markdown + metadata sebagai gold data

  WAJIB: proses SATU DOKUMEN sampai SEMUA BATCH-nya selesai (has_more=false) dan
  sudah disimpan ke save_extraction_result, BARU pindah ke dokumen berikutnya.

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
from .pdf import pdf_page_count
from .ppt import count_presentation_slides
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
        "5. PROSES SATU DOKUMEN PER SATU HINGGA SELESAI SEMUA BATCH-NYA (jangan pindah ke dokumen\n"
        "   berikutnya sebelum dokumen ini disimpan):\n"
        "   a. Untuk PPTX: 'render_presentation_slides' (start_slide=1) — tool mengirim 10 slide per batch.\n"
        "      Baca gambar secara visual, tulis Markdown batch itu (sertakan penanda halaman/slide\n"
        "      standar: <!-- SLIDE: N -->). Jika summary has_more=true, ulangi panggilan dengan\n"
        "      start_slide = next_start_slide hingga has_more=false. Gabungkan Markdown kumulatif.\n"
        "   b. Untuk PDF: 'convert_pdf_to_images' (start_page=1) — sama, ulangi dengan\n"
        "      start_page = next_start_page hingga has_more=false, kumulatifkan Markdown-nya.\n"
        "      Gunakan 'preprocess_image' bila gambar perlu diperbaiki orientasi/kontras (juga mengirim\n"
        "      gambar hasil ke model).\n"
        "   c. Tulis sendiri Markdown-nya dari gambar yang dilihat (OCR/ekstraksi teks tidak tersedia\n"
        "      di server ini).\n"
        "   d. Setelah SEMUA BATCH SELESAI (has_more=false), simpan Markdown kumulatif LEBIH DARI SATU\n"
        "      HALAMAN/SLIDE ke 'save_extraction_result' (file sumber = dokumen asli, bukan gambar).\n"
        "   e. BARU setelah save_extraction_result mengembalikan status success, lanjut ke dokumen\n"
        "      berikutnya dari daftar yang dipilih.\n"
        "6. Gunakan 'preview_markdown_chunks' untuk validasi kesiapan chunking sebelum disimpan.\n"
        "7. Jangan pernah memilih folder atau jumlah data sendiri tanpa persetujuan user."
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
                "PROSES SATU DOKUMEN PER SATU HINGGA SELESAI SEMUA BATCH-NYA sebelum pindah ke "
                "dokumen berikutnya. Untuk PPTX: 'render_presentation_slides' (start_slide=1) ulangi "
                "dengan next_start_slide hingga has_more=false, kumulatifkan Markdown-nya. Untuk PDF: "
                "'convert_pdf_to_images' (start_page=1) ulangi dengan next_start_page hingga "
                "has_more=false. Baca gambar secara visual, tulis Markdown-nya, lalu simpan dengan "
                "'save_extraction_result' (file sumber = dokumen asli, bukan gambar)."
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
        "Render slide file presentasi PowerPoint (.pptx / .ppt) menjadi file gambar PNG "
        "beresolusi tinggi satu file per slide kanvas utuh, lalu KIRIM gambar-gambar tersebut "
        "LANGSUNG ke model sebagai image content agar model dapat melihat dan membaca slide secara visual. "
        "Default output: output/rendered_slides/<nama_file_tanpa_ekstensi>/. "
        "Karena Spire.Presentation (library yang digunakan) hanya bisa merender maksimal 10 slide "
        "per objek presentasi (versi Free: slide ke-11 dst. akan blank + watermark lisensi), "
        "tool ini memproses DENGAN BATCH per 10 slide: gunakan 'start_slide' (1-based, default 1) "
        "dan 'max_images' (default 10) untuk mengirim SATU BATCH per panggilan. "
        "Hasil summary mengandung 'has_more' dan 'next_start_slide': ulangi panggilan dengan "
        "start_slide = next_start_slide hingga has_more=false (seluruh slide selesai) sebelum "
        "menyimpan Markdown ke 'save_extraction_result'. Proses SATU FILE sampai selesai semua "
        "batch-nya, BARU pindah ke file berikutnya."
    ),
)
def render_presentation_slides(
    pptx_path: str,
    output_dir: str | None = None,
    start_slide: int = 1,
    max_images: int | None = None,
) -> list[ContentBlock] | str:
    """
    Render slide PPTX ke file gambar PNG per slide dan kirim SATU BATCH gambar ke model.

    Args:
        pptx_path: Path file presentasi.
        output_dir: Folder output (default: output/rendered_slides/<stem>).
        start_slide: Nomor slide 1-based dari mana batch dimulai (default 1).
        max_images: Jumlah gambar yang dikirim dalam batch ini (default 10 = batas lisensi Free).
    """
    path_obj = Path(pptx_path).resolve()
    if not path_obj.exists():
        return f"ERROR: File presentasi tidak ditemukan: {path_obj}"

    resolved_out = Path(
        output_dir or f"output/rendered_slides/{path_obj.stem}"
    ).resolve()

    try:
        total_slides = count_presentation_slides(path_obj)
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat menghitung slide presentasi: {e}"

    if total_slides <= 0:
        return f"ERROR: File presentasi tidak memiliki slide: {path_obj}"

    # Ukuran batch: default 10 (batas lisensi Spire Free), maksimal 10 per panggilan.
    batch_size = (
        DEFAULT_MAX_IMAGES if (max_images is None or max_images <= 0) else min(max_images, DEFAULT_MAX_IMAGES)
    )

    # Window slide 1-based yang akan diproses dalam batch ini.
    first = max(1, start_slide)
    last = min(first + batch_size - 1, total_slides)
    if first > total_slides:
        return f"ERROR: start_slide={start_slide} melebihi total slide ({total_slides})."

    window_indices = list(range(first - 1, last))

    # Hapus gambar stale HANYA saat memulai dokumen dari awal (batch 1),
    # agar batch sebelumnya di disk tidak ikut terhapus.
    if first == 1:
        for old_f in resolved_out.glob("slide_*.png"):
            try:
                old_f.unlink(missing_ok=True)
            except OSError:
                pass
        # Kompatibilitas mundur dengan penamaan lama (slide_N_img_M.*)
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
            path_obj,
            output_dir=resolved_out,
            slides=window_indices,
        )
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat merender slide presentasi: {e}"

    has_more = last < total_slides
    next_start = (last + 1) if has_more else None
    summary = {
        "pptx_file": str(path_obj),
        "total_slides": total_slides,
        "batch_start": first,
        "batch_end": last,
        "batch_size": batch_size,
        "rendered_image_paths": [str(p) for p in images],
        "images_sent_to_model": len(images),
        "has_more": has_more,
        "next_start_slide": next_start,
        "instruction": (
            "Batch ini selesai. Lanjutkan dengan start_slide="
            f"{next_start} hingga has_more=false, lalu simpan Markdown ke 'save_extraction_result'."
            if has_more
            else "Seluruh slide dokumen ini selesai. Sekarang simpan Markdown kumulatif ke 'save_extraction_result'."
        ),
    }
    return _build_image_result(summary, images, max_images=None)


@server.tool(
    name="convert_pdf_to_images",
    description=(
        "Konversi halaman file PDF menjadi file gambar beresolusi tinggi satu file per halaman, "
        "lalu KIRIM gambar-gambar tersebut LANGSUNG ke model sebagai image content agar model dapat "
        "membaca halaman PDF secara visual. Default output: output/rendered_pages/<nama_file_tanpa_ekstensi>/. "
        "Gunakan 'dpi' untuk mengatur resolusi (default 200), 'start_page' (1-based, default 1) untuk "
        "memulai batch, dan 'max_images' (default 10) untuk jumlah halaman per batch. "
        "Ulangi panggilan dengan start_page = next_start_page hingga has_more=false (seluruh halaman "
        "selesai) sebelum menyimpan Markdown ke 'save_extraction_result'. Proses SATU FILE sampai "
        "selesai semua batch-nya, BARU pindah ke file berikutnya."
    ),
)
def convert_pdf_to_images(
    pdf_path: str,
    output_dir: str | None = None,
    dpi: int = 200,
    start_page: int = 1,
    max_images: int | None = None,
) -> list[ContentBlock] | str:
    """
    Konversi halaman-halaman PDF menjadi file gambar per halaman dan kirim SATU BATCH gambar ke model.
    """
    path_obj = Path(pdf_path).resolve()
    if not path_obj.exists():
        return f"ERROR: File PDF tidak ditemukan: {path_obj}"
    if path_obj.suffix.lower() != ".pdf":
        return f"ERROR: File bukan PDF: {path_obj}"

    try:
        total_pages = pdf_page_count(path_obj)
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat menghitung halaman PDF: {e}"
    if total_pages <= 0:
        return f"ERROR: PDF tidak memiliki halaman: {path_obj}"

    resolved_out = output_dir or f"output/rendered_pages/{path_obj.stem}"

    # Ukuran batch: default 10 per panggilan.
    batch_size = (
        DEFAULT_MAX_IMAGES if (max_images is None or max_images <= 0) else min(max_images, DEFAULT_MAX_IMAGES)
    )

    first = max(1, start_page)
    last = min(first + batch_size - 1, total_pages)
    if first > total_pages:
        return f"ERROR: start_page={start_page} melebihi total halaman ({total_pages})."

    window_indices = list(range(first - 1, last))

    try:
        import importlib

        import app.pdf

        importlib.reload(app.pdf)
        images = app.pdf.pdf_to_images(
            path_obj,
            output_dir=resolved_out,
            dpi=dpi,
            pages=window_indices,
        )
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat mengonversi PDF ke gambar: {e}"

    has_more = last < total_pages
    next_start = (last + 1) if has_more else None
    summary = {
        "pdf_file": str(path_obj),
        "dpi": dpi,
        "total_pages": total_pages,
        "batch_start": first,
        "batch_end": last,
        "batch_size": batch_size,
        "rendered_image_paths": [str(p) for p in images],
        "images_sent_to_model": len(images),
        "has_more": has_more,
        "next_start_page": next_start,
        "instruction": (
            "Batch ini selesai. Lanjutkan dengan start_page="
            f"{next_start} hingga has_more=false, lalu simpan Markdown ke 'save_extraction_result'."
            if has_more
            else "Seluruh halaman dokumen ini selesai. Sekarang simpan Markdown kumulatif ke 'save_extraction_result'."
        ),
    }
    return _build_image_result(summary, images, max_images=None)


@server.tool(
    name="extract_pdf_markdown_mupdf",
    description=(
        "Ekstrak teks dan tabel dokumen PDF digital langsung menjadi Markdown menggunakan PyMuPDF4LLM "
        "(C++ backend sangat cepat, zero-token). Berguna untuk mengambil grounding teks mentah per-halaman, "
        "tabel GFM, dan urutan baca 2-kolom sebagai referensi bantu bagi agent. "
        "Gunakan 'max_pages' (misal 10) atau 'pages' (misal [0, 1, 2]) untuk membatasi rentang halaman pada PDF tebal."
    ),
)
def extract_pdf_markdown_mupdf(
    pdf_path: str,
    page_chunks: bool = True,
    max_pages: int | None = 10,
    pages: list[int] | None = None,
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
        import importlib

        import app.pdf

        importlib.reload(app.pdf)

        target_pages = pages
        if target_pages is None and max_pages is not None and max_pages > 0:
            target_pages = list(range(max_pages))

        res = app.pdf.extract_pdf_with_pymupdf4llm(
            path_obj,
            page_chunks=page_chunks,
            pages=target_pages,
        )
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

        clean_md = markdown.strip() + "\n"
        pages_parsed = split_markdown_by_pages(clean_md)

        # --- Gate riil: validasi kelengkapan batch (bukan hanya instruksi) ---
        # Pastikan Markdown mencakup SEMUA slide/halaman file sumber. Jika tidak,
        # agent masih ada batch yang dilewatkan (misal lupa ulangi panggilan hingga
        # has_more=false) — tolak simpan agar agent melengkapi dulu.
        ext = src.suffix.lower()
        total_pages: int | None = None
        if ext in (".pptx", ".ppt"):
            try:
                from .ppt import count_presentation_slides

                total_pages = count_presentation_slides(src)
            except Exception:  # noqa: BLE001 - jangan memblokir simpan jika hitung gagal
                total_pages = None
        elif ext == ".pdf":
            try:
                from .pdf import pdf_page_count

                total_pages = pdf_page_count(src)
            except Exception:  # noqa: BLE001
                total_pages = None

        if total_pages is not None and total_pages > 1:
            got_numbers = sorted({p["page_number"] for p in pages_parsed})
            missing = [n for n in range(1, total_pages + 1) if n not in got_numbers]
            if missing:
                return json.dumps(
                    {
                        "status": "error",
                        "message": (
                            "Markdown BELUM lengkap: file sumber punya "
                            f"{total_pages} halaman/slide, tetapi Markdown hanya mencakup "
                            f"{len(got_numbers)} (nomor yang hilang: {missing}). "
                            "Lanjutkan proses batch berikutnya (has_more=false) hingga semua "
                            "slide/halaman terbaca, lalu simpan lagi."
                        ),
                        "total_pages_in_source": total_pages,
                        "pages_in_markdown": got_numbers,
                        "missing_pages": missing,
                    },
                    indent=2,
                    ensure_ascii=False,
                )

        out_md = out_base / f"{src.stem}.md"
        out_md.write_text(clean_md, encoding="utf-8")

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
