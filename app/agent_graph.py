"""
Modul Orkestrasi Dokumen Agent Mode berbasis LangGraph StateGraph.

Menyediakan StateGraph terstruktur untuk:
1. `render_pipeline`: Resolusi path -> inspeksi dokumen -> render batch 5 slide/halaman -> optimasi payload.
2. `save_pipeline`: Ingesti Markdown per-batch -> stitching multi-halaman -> auto-ingest SQLite & double verification.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .pdf import pdf_page_count, pdf_to_images
from .ppt import count_presentation_slides, render_presentation_slides_to_images
from .tabular_db import extract_and_ingest_tables_from_markdown

logger = logging.getLogger(__name__)

# Root folder proyek
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def resolve_project_path(p: str | Path) -> Path:
    """
    Konversi path relatif menjadi path absolut terhadap root proyek (bukan C:\\Windows\\System32).
    """
    path_obj = Path(p)
    if path_obj.is_absolute():
        return path_obj.resolve()
    return (PROJECT_ROOT / path_obj).resolve()


class DocumentBatchState(TypedDict, total=False):
    """Skema State LangGraph untuk pipeline batch/per-halaman dokumen."""

    source_path: str
    resolved_path: str
    output_dir: str
    resolved_out: str
    doc_type: str  # "pptx", "pdf", "image"
    total_items: int
    current_page: int
    batch_start: int
    batch_size: int
    batch_end: int
    dpi: int
    rendered_images: list[str]
    has_more: bool
    next_start: int | None
    incoming_markdown: str | None
    specs: str
    merged_markdown: str | None
    is_complete: bool
    saved_pages: list[int]
    missing_pages: list[int]
    page_structure: list[dict[str, Any]]
    tabular_tables: list[dict[str, Any]]
    active_tables_summary: list[dict[str, Any]]
    subagent_task_directives: dict[str, Any]
    status: str
    error: str | None
    instruction: str


def get_subagent_task_directives(
    doc_type: str = "general",
    current_page: int = 1,
    total_pages: int = 1,
    active_tables: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Menyusun katalog direktif peran sub-agent (Deep Agent Role Directives) untuk halaman yang sedang diproses.
    Memaksa model pada setiap giliran untuk mengevaluasi visual dan menerapkan aturan sub-agent terkait.
    """
    return {
        "execution_protocol": (
            "EVALUASI GAMBAR HALAMAN INI: "
            "Pilih 1 atau lebih peran spesialis yang relevan di bawah ini, dan terapkan SEMUA aturan & formatnya secara ketat dalam hasil Markdown Anda."
        ),
        "available_specialist_roles": [
            "mermaid_specialist",
            "tabular_sqlite_specialist",
            "invoice_form_specialist",
            "legal_hierarchy_specialist",
            "presentation_slide_specialist",
            "scientific_math_specialist",
        ],
        "subagent_directives": {
            "mermaid_specialist": {
                "trigger": "Visual memuat alur proses, flowchart, sequence, pohon keputusan, class diagram, ERD, finite state machine, atau mindmap",
                "mandatory_action": (
                    "WAJIB diekstrak sebagai blok kode ```mermaid (misal flowchart TD / flowchart LR / sequenceDiagram). "
                    "DILARANG KERAS menggunakan panah teks biasa (↓, ->, -->) dalam daftar teks untuk diagram alir. "
                    'Selalu beri tanda kutip ganda pada label teks bersimbol/spasi, misal node_1["Langkah 1 (Input Data)"].'
                ),
            },
            "tabular_sqlite_specialist": {
                "trigger": "Visual memuat tabel data, log transaksi, laporan keuangan, neraca saldo, atau matriks pengukuran numerik",
                "mandatory_action": (
                    "Ekstrak sebagai tabel Markdown GFM bersih. "
                    "Untuk kolom angka berketerangan (misal: '0,683 (Kuat)'), pisahkan angka ke kolom nilai numerik murni dan label ke kolom kategori, "
                    "agar database SQLite dapat otomatis meng-ingest dan menjalankan query agregasi SQL (SUM, AVG, COUNT)."
                ),
                "active_sqlite_tables_snapshot": active_tables or [],
            },
            "invoice_form_specialist": {
                "trigger": "Visual memuat kwitansi, invoice faktur, formulir key-value, atau bukti transaksi",
                "mandatory_action": (
                    "Ekstrak metadata header (No Invoice, Tanggal, Pengirim, Penerima) dalam format key-value bold, "
                    "ekstrak rincian item dalam tabel tabular, dan verifikasi konsistensi aritmatika (Subtotal + Pajak = Total)."
                ),
            },
            "legal_hierarchy_specialist": {
                "trigger": "Dokumen berupa peraturan perundang-undangan, keputusan, statuta, atau regulasi hukum",
                "mandatory_action": (
                    "Patuhi tumpukan hirarki: JUDUL -> PEMBUKAAN (Menimbang/Mengingat/MEMUTUSKAN) -> BAB -> Bagian -> Paragraf -> Pasal -> Ayat -> Huruf -> Angka. "
                    "Cegah level drift agar konsisten di seluruh halaman dokumen."
                ),
            },
            "presentation_slide_specialist": {
                "trigger": "Dokumen berupa slide presentasi (PPT / PPTX / Slide deck)",
                "mandatory_action": (
                    "Gunakan `# Judul Presentasi` untuk Slide 1 (cover), dan `## Judul Slide` untuk slide-slide berikutnya. "
                    "Ekstrak poin daftar terstruktur, dan beri anotasi gambar/foto/bagan dalam blockquote `> [!NOTE] Foto/Bagan: <deskripsi>`."
                ),
            },
            "scientific_math_specialist": {
                "trigger": "Visual memuat rumus matematika, koordinat geografis, satuan fisik/kimia, atau notasi ilmiah",
                "mandatory_action": (
                    "Pertahankan notasi eksak: koordinat derajat-menit-detik (misal 7°54'13.97\" LS), satuan fisik (mdpl, ppm, μS/cm, °C), "
                    "dan rumus matematika dalam blok LaTeX $$ rumus $$."
                ),
            },
        },
    }


class AgentDocumentGraph:
    """
    Kompilasi dan orkestrasi StateGraph LangGraph untuk memproses dokumen secara deterministik.
    """

    def __init__(self) -> None:
        self.render_graph: CompiledStateGraph = self._build_render_graph()
        self.save_graph: CompiledStateGraph = self._build_save_graph()
        self.advance_graph: CompiledStateGraph = self._build_advance_graph()

    # --- Node Implementations ---

    @staticmethod
    def _node_resolve_and_inspect(state: DocumentBatchState) -> dict[str, Any]:
        """Node 1: Resolusi path absolut & inspeksi jumlah halaman/slide."""
        raw_source = state.get("source_path", "")
        if not raw_source:
            return {"status": "error", "error": "Path file sumber tidak boleh kosong"}

        resolved = resolve_project_path(raw_source)
        if not resolved.exists():
            return {
                "status": "error",
                "error": f"File dokumen tidak ditemukan: {resolved}",
            }

        ext = resolved.suffix.lower()
        if ext in (".pptx", ".ppt"):
            doc_type = "pptx"
            try:
                total_items = count_presentation_slides(resolved)
            except Exception as exc:  # noqa: BLE001
                return {
                    "status": "error",
                    "error": f"Gagal menghitung slide PPTX: {exc}",
                }
        elif ext == ".pdf":
            doc_type = "pdf"
            try:
                total_items = pdf_page_count(resolved)
            except Exception as exc:  # noqa: BLE001
                return {
                    "status": "error",
                    "error": f"Gagal menghitung halaman PDF: {exc}",
                }
        elif ext in (".png", ".jpg", ".jpeg", ".webp"):
            doc_type = "image"
            total_items = 1
        else:
            return {
                "status": "error",
                "error": f"Ekstensi file '{ext}' tidak didukung",
            }

        if total_items <= 0:
            return {
                "status": "error",
                "error": f"Dokumen tidak memiliki halaman/slide valid: {resolved}",
            }

        batch_size = max(1, state.get("batch_size", 1))
        batch_start = max(1, state.get("current_page") or state.get("batch_start") or 1)
        batch_end = min(batch_start + batch_size - 1, total_items)
        has_more = batch_end < total_items
        next_start = (batch_end + 1) if has_more else None

        default_out_sub = (
            f"output/rendered_slides/{resolved.stem}"
            if doc_type == "pptx"
            else (
                f"output/rendered_pages/{resolved.stem}"
                if doc_type == "pdf"
                else f"output/rendered_images/{resolved.stem}"
            )
        )
        out_raw = state.get("output_dir") or default_out_sub
        resolved_out = resolve_project_path(out_raw)

        return {
            "resolved_path": str(resolved),
            "doc_type": doc_type,
            "total_items": total_items,
            "current_page": batch_start,
            "batch_start": batch_start,
            "batch_size": batch_size,
            "batch_end": batch_end,
            "has_more": has_more,
            "next_start": next_start,
            "resolved_out": str(resolved_out),
            "status": "inspected",
        }

    @staticmethod
    def _node_render_batch(state: DocumentBatchState) -> dict[str, Any]:
        """Node 2: Eksekusi rendering batch gambar pada resolusi & format teroptimasi."""
        if state.get("status") == "error":
            return {}

        doc_type = state["doc_type"]
        resolved = Path(state["resolved_path"])
        resolved_out = Path(state["resolved_out"])
        resolved_out.mkdir(parents=True, exist_ok=True)

        batch_start = state["batch_start"]
        batch_end = state["batch_end"]
        window_indices = list(range(batch_start - 1, batch_end))
        dpi = state.get("dpi", 150)

        # Bersihkan file render lama pada batch pertama
        if batch_start == 1:
            for old_f in resolved_out.glob("slide_*.*"):
                try:
                    old_f.unlink(missing_ok=True)
                except OSError:
                    pass

        try:
            if doc_type == "pptx":
                rendered_paths = render_presentation_slides_to_images(
                    resolved,
                    output_dir=resolved_out,
                    slides=window_indices,
                    dpi=dpi,
                    image_ext=".jpg",
                )
            elif doc_type == "pdf":
                rendered_paths = pdf_to_images(
                    resolved,
                    output_dir=resolved_out,
                    dpi=dpi,
                    pages=window_indices,
                )
            else:
                rendered_paths = [resolved]

            str_paths = [str(p) for p in rendered_paths]
            has_more = state["has_more"]
            next_start = state["next_start"]

            inst = (
                f"Halaman/Slide {batch_start}..{batch_end} selesai dirender ({len(str_paths)} gambar). "
                f"Silakan baca gambar tersebut dan panggil tool penyimpan Markdown ('submit_page_and_get_next' atau 'save_extraction_result'). "
                f"Lanjutkan ke nomor={next_start} setelah bagian ini tersimpan."
                if has_more
                else "Seluruh dokumen selesai dirender. Simpan bagian ini untuk finalisasi dokumen."
            )

            # Ambil snapshot tabel SQLite eksisting jika ada
            from .tabular_db import TabularDatabaseManager

            db_file = (
                resolve_project_path("output/databases") / f"{resolved.stem}.sqlite"
            )
            active_tables = (
                TabularDatabaseManager(db_file).get_active_tables_summary()
                if db_file.exists()
                else []
            )

            directives = get_subagent_task_directives(
                doc_type=state.get("doc_type", "general"),
                current_page=batch_start,
                total_pages=state.get("total_items", 1),
                active_tables=active_tables,
            )

            return {
                "rendered_images": str_paths,
                "active_tables_summary": active_tables,
                "subagent_task_directives": directives,
                "status": "rendered",
                "instruction": inst,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "error": f"Gagal saat merender batch dokumen: {exc}",
            }

    @staticmethod
    def _node_save_and_stitch(state: DocumentBatchState) -> dict[str, Any]:
        """Node 3: Penggabungan Markdown inkremental, penulisan metadata, & auto-ingest SQLite."""
        if state.get("status") == "error":
            return {}

        incoming = state.get("incoming_markdown") or ""
        if not incoming.strip():
            return {
                "status": "error",
                "error": "Konten Markdown yang dikirimkan kosong",
            }

        from .multi_page import merge_and_stitch_markdown_pages
        from .tabular_db import TabularDatabaseManager

        resolved = Path(state["resolved_path"])
        out_base = resolve_project_path(state.get("output_dir") or "output/agent_gold")
        out_base.mkdir(parents=True, exist_ok=True)

        is_slide = state["doc_type"] == "pptx"
        out_md = out_base / f"{resolved.stem}.md"
        current_pg = state.get("current_page") or state.get("batch_start") or 1
        is_resume = state.get("resume", False)
        is_overwrite = state.get("overwrite", False)
        existing_md: str | None = None
        if out_md.exists() and (is_resume or (current_pg > 1 and not is_overwrite)):
            existing_md = out_md.read_text(encoding="utf-8")

        clean_incoming = incoming.strip() + "\n"
        merged_md, pages_parsed = merge_and_stitch_markdown_pages(
            existing_md,
            clean_incoming,
            default_page_number=current_pg,
            is_slide=is_slide,
        )
        out_md.write_text(merged_md, encoding="utf-8")

        total_items = state["total_items"]
        got_numbers = sorted({p["page_number"] for p in pages_parsed})
        missing = (
            [n for n in range(1, total_items + 1) if n not in got_numbers]
            if (total_items > 1)
            else []
        )
        is_complete = not bool(missing)

        page_structure = [
            {
                "page_number": p["page_number"],
                "type": p["type"],
                "char_count": len(p["content"]),
            }
            for p in pages_parsed
        ]

        # Ingest tabel transaksional ke database SQLite tunggal dokumen dengan smart schema matching & append
        tabular_info: list[dict[str, Any]] = []
        db_dir = resolve_project_path("output/databases")
        db_dir.mkdir(parents=True, exist_ok=True)
        db_file = db_dir / f"{resolved.stem}.sqlite"

        current_pg = state.get("current_page") or (
            got_numbers[-1] if got_numbers else 1
        )
        try:
            ingest_res = extract_and_ingest_tables_from_markdown(
                markdown_text=clean_incoming,
                source_file=str(resolved),
                db_path=db_file,
                page_number=current_pg,
                append_if_matching=True,
            )
            for r in ingest_res:
                tabular_info.append(r.model_dump())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gagal auto-ingest tabel transaksional ke SQLite: %s", exc)

        active_tables = TabularDatabaseManager(db_file).get_active_tables_summary()

        metadata = {
            "source_file": str(resolved),
            "source_extension": resolved.suffix.lower(),
            "specs": [
                s.strip() for s in state.get("specs", "plain").split(",") if s.strip()
            ],
            "extracted_by": "agent",
            "ocr_used": False,
            "saved_at": datetime.now(UTC).isoformat(),
            "markdown_path": str(out_md),
            "char_count": len(merged_md),
            "line_count": len(merged_md.splitlines()),
            "total_pages_detected": len(pages_parsed),
            "total_pages_in_source": total_items,
            "is_complete": is_complete,
            "pages_saved": got_numbers,
            "missing_pages": missing,
            "page_structure": page_structure,
            "tabular_database_tables": tabular_info,
            "active_database_tables": active_tables,
        }

        out_meta = out_base / f"{resolved.stem}.meta.json"
        out_meta.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        msg = (
            "Seluruh halaman/slide dokumen telah selesai diekstraksi dan disimpan secara lengkap."
            if is_complete
            else f"Halaman/Slide berhasil disimpan ({len(got_numbers)} dari {total_items} halaman/slide tersimpan). Lanjutkan ke nomor: {missing[:5]}..."
        )

        return {
            "merged_markdown": merged_md,
            "is_complete": is_complete,
            "saved_pages": got_numbers,
            "missing_pages": missing,
            "page_structure": page_structure,
            "tabular_tables": tabular_info,
            "active_tables_summary": active_tables,
            "status": "success" if is_complete else "batch_saved",
            "instruction": msg,
        }

    def _node_advance_to_next(self, state: DocumentBatchState) -> dict[str, Any]:
        """Node 4: Jika belum selesai, langsung render halaman berikutnya untuk menghemat 1 round-trip."""
        if state.get("status") == "error" or state.get("is_complete"):
            return {}

        missing = state.get("missing_pages", [])
        if not missing:
            return {"is_complete": True}

        next_page = missing[0]
        advance_state = dict(state)
        advance_state["batch_start"] = next_page
        advance_state["batch_size"] = 1
        advance_state["batch_end"] = next_page

        render_res = self._node_render_batch(cast(Any, advance_state))
        return {
            "current_page": next_page,
            "rendered_images": render_res.get("rendered_images", []),
            "subagent_task_directives": render_res.get("subagent_task_directives", {}),
            "has_more": True,
            "next_start": missing[1] if len(missing) > 1 else None,
            "instruction": (
                f"Halaman/Slide {next_page} berhasil dirender. "
                f"Evaluasi visual gambar dan jalankan peran subagent yang sesuai (Mermaid/Tabular/Hierarchy/Scientific/Slide). "
                f"Kirim hasil Markdown via 'submit_page_and_get_next'."
            ),
        }

    # --- Graph Builders ---

    def _build_render_graph(self) -> CompiledStateGraph:
        builder = StateGraph(cast(Any, DocumentBatchState))
        builder.add_node("resolve_and_inspect", self._node_resolve_and_inspect)
        builder.add_node("render_batch", self._node_render_batch)

        builder.add_edge(START, "resolve_and_inspect")
        builder.add_edge("resolve_and_inspect", "render_batch")
        builder.add_edge("render_batch", END)

        return builder.compile()

    def _build_save_graph(self) -> CompiledStateGraph:
        builder = StateGraph(cast(Any, DocumentBatchState))
        builder.add_node("resolve_and_inspect", self._node_resolve_and_inspect)
        builder.add_node("save_and_stitch", self._node_save_and_stitch)

        builder.add_edge(START, "resolve_and_inspect")
        builder.add_edge("resolve_and_inspect", "save_and_stitch")
        builder.add_edge("save_and_stitch", END)

        return builder.compile()

    def _build_advance_graph(self) -> CompiledStateGraph:
        builder = StateGraph(cast(Any, DocumentBatchState))
        builder.add_node("resolve_and_inspect", self._node_resolve_and_inspect)
        builder.add_node("save_and_stitch", self._node_save_and_stitch)
        builder.add_node("advance_to_next", self._node_advance_to_next)

        builder.add_edge(START, "resolve_and_inspect")
        builder.add_edge("resolve_and_inspect", "save_and_stitch")
        builder.add_edge("save_and_stitch", "advance_to_next")
        builder.add_edge("advance_to_next", END)

        return builder.compile()


# Global Singleton Pipeline
_agent_graph_instance: AgentDocumentGraph | None = None


def get_agent_document_graph() -> AgentDocumentGraph:
    """Mengembalikan singleton instance AgentDocumentGraph."""
    global _agent_graph_instance
    if _agent_graph_instance is None:
        _agent_graph_instance = AgentDocumentGraph()
    return _agent_graph_instance
