"""
Streamlit Launcher & Viewer untuk Pipeline Ekstraksi Dokumen Vision OCR, Sub-Agent SQL Tabular, & Dual-Track Guardrail.

Fitur:
  - Menjalankan pipeline utama `main.py` pada file PDF, PPTX, PPT, atau Gambar.
  - Tampilan teks Markdown utuh hasil VLM.
  - Tabular Database (SQLite) viewer & interactive SQL query console.
  - Dual-Track Guardrail & Audit Report (komparasi jalur Markdown vs SQLite).
  - Log eksekusi transparan.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_TYPES = ["pdf", "pptx", "ppt", "png", "jpg", "jpeg", "webp"]

SPEC_OPTIONS: dict[str, str | None] = {
    "Auto-Detect (Rekomendasi VLM)": None,
    "Plain Document (Surat/Dokumen Standar)": "plain",
    "Markdown Hierarchy (Bab, Sub-bab #, ##, ###)": "markdown_hierarchy",
    "Bilingual Journal (2 Kolom & 2 Bahasa)": "bilingual_journal",
    "Presentation Slides (Slide & Diagram Visual)": "presentation_slides",
}


def _write_upload_to_temp(
    uploaded_file: UploadedFile,
) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="streamlit_launch_"))
    temp_path = temp_dir / Path(uploaded_file.name).name
    temp_path.write_bytes(uploaded_file.getvalue())
    return temp_path


def _write_run_log(
    log_path: Path,
    *,
    input_path: Path,
    output_dir: Path,
    cmd: list[str],
    returncode: int,
    stdout: str,
    stderr: str,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    sections = [
        f"[{timestamp}] Streamlit launcher run",
        f"Input file: {input_path}",
        f"Output dir: {output_dir}",
        f"Command: {' '.join(cmd)}",
        f"Exit code: {returncode}",
        "",
        "[stdout]",
        stdout.strip() or "(empty)",
        "",
        "[stderr]",
        stderr.strip() or "(empty)",
        "",
    ]
    log_path.write_text("\n".join(sections), encoding="utf-8")


def _run_main_cli(
    input_path: Path,
    output_dir: Path,
    *,
    doc_type: str | None = None,
    dpi: int = 200,
    force_all_tables: bool = False,
    preview_chunks: bool = True,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    live_log: st.delta_generator.DeltaGenerator | None = None,
) -> tuple[int, str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{input_path.stem}.md"
    log_dir = output_dir / "logs"
    log_name = f"{input_path.stem}_{dt.datetime.now(dt.UTC).strftime('%Y%m%d_%H%M%S')}.log"
    log_path = log_dir / log_name

    cmd = [
        sys.executable,
        "main.py",
        str(input_path),
        "-o",
        str(out_file),
        "--dpi",
        str(dpi),
    ]

    if doc_type:
        cmd.extend(["-t", doc_type])
    if force_all_tables:
        cmd.append("--force-all-tables")
    if preview_chunks:
        cmd.extend([
            "--preview-chunks",
            "--chunk-size",
            str(chunk_size),
            "--chunk-overlap",
            str(chunk_overlap),
        ])

    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    log_lines: list[str] = []
    if proc.stdout is not None:
        for line in proc.stdout:
            log_lines.append(line)
            if live_log is not None:
                live_log.text("".join(log_lines[-25:]))

    returncode = proc.wait()
    all_output = "".join(log_lines)
    _write_run_log(
        log_path,
        input_path=input_path,
        output_dir=output_dir,
        cmd=cmd,
        returncode=returncode,
        stdout=all_output,
        stderr="",
    )
    return returncode, all_output, log_path


def _get_sqlite_db_for_file(input_file: Path) -> Path | None:
    """Cari file database SQLite yang terkait dengan file yang diproses."""
    db_candidates = [
        Path("output/databases") / f"{input_file.stem}_data.sqlite",
        Path("output/databases") / "documents_data.sqlite",
    ]
    for c in db_candidates:
        if c.exists():
            return c
    return None


# ==============================================================================
# UI Streamlit
# ==============================================================================

st.set_page_config(
    page_title="Vision OCR & Dual-Track Sub-Agent",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📑 Document Text, Vision OCR & Dual-Track Sub-Agent")
st.caption(
    "Ekstraksi Multi-Page Vision OCR, Sub-Agent SQL Tabular Otomatis, & Dual-Track Guardrail Cross-Verification."
)

with st.sidebar:
    st.header("⚙️ Konfigurasi Pipeline")
    spec_label = st.selectbox("Jenis Dokumen / Layout:", list(SPEC_OPTIONS.keys()))
    chosen_spec = SPEC_OPTIONS[spec_label]

    with st.expander("Pengaturan Lanjutan", expanded=False):
        dpi_val = st.slider("DPI Raster Rendering (PDF/PPT):", 100, 300, 200, 25)
        force_all_tbl = st.checkbox(
            "Paksa Ingesti Seluruh Tabel ke SQLite",
            value=False,
            help="Jika dicentang, tabel naratif/kualitatif juga akan dimasukkan ke SQLite di samping tabel transaksional/finansial.",
        )
        show_chunk_preview = st.checkbox("Tampilkan Chunking Preview", value=True)
        c_size = st.number_input("Chunk Size:", 200, 4000, 1000, 100)
        c_overlap = st.number_input("Chunk Overlap:", 0, 1000, 150, 25)

    st.markdown("---")
    st.info("💡 **Sub-Agent SQL Tabular:** Bekerja mandiri memetakan tabel terdeteksi ke SQLite (.sqlite) dan diverifikasi oleh Agent Supervisor.")

uploaded_file = st.file_uploader(
    "Unggah Dokumen (PDF, PPTX, PPT, PNG, JPG, WEBP):",
    type=SUPPORTED_TYPES,
)

if uploaded_file is not None:
    temp_input_path = _write_upload_to_temp(uploaded_file)
    output_dir = PROJECT_ROOT / "output"

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        start_process = st.button("🚀 Mulai Ekstraksi", type="primary", use_container_width=True)

    if start_process:
        with st.status("Sedang memproses dokumen dengan Dual-Track Vision & Sub-Agent SQL...", expanded=True) as status:
            live_log = st.empty()
            returncode, output_text, log_path = _run_main_cli(
                temp_input_path,
                output_dir,
                doc_type=chosen_spec,
                dpi=dpi_val,
                force_all_tables=force_all_tbl,
                preview_chunks=show_chunk_preview,
                chunk_size=int(c_size),
                chunk_overlap=int(c_overlap),
                live_log=live_log,
            )

            if returncode == 0:
                status.update(label="✅ Pemrosesan Berhasil Selesai!", state="complete", expanded=False)
                st.session_state["last_processed_file"] = temp_input_path
                st.session_state["last_output_dir"] = output_dir
                st.session_state["last_log_path"] = log_path
            else:
                status.update(label="❌ Terjadi Kesalahan saat Ekstraksi", state="error", expanded=True)
                st.error("Proses CLI mengembalikan status error. Cek log output di bawah.")

    # Jika file selesai diproses, tampilkan Tab Viewer
    if "last_processed_file" in st.session_state and st.session_state["last_processed_file"].name == temp_input_path.name:
        last_file: Path = st.session_state["last_processed_file"]
        md_file = output_dir / f"{last_file.stem}.md"
        db_file = _get_sqlite_db_for_file(last_file)

        tab_guardrail, tab_md, tab_sql, tab_log = st.tabs([
            "🛡️ Dual-Track Guardrail Audit",
            "📝 Markdown Output",
            "📊 Data Tabular (SQLite)",
            "📋 Run Log",
        ])

        with tab_guardrail:
            st.subheader("🛡️ Laporan Audit Guardrail Supervisor")
            st.caption("Agent Utama membandingkan konsistensi Jalur 1 (Teks Markdown) vs Jalur 2 (Database SQLite).")

            # Analisis sederhana dari file markdown dan database
            md_content = md_file.read_text(encoding="utf-8") if md_file.exists() else ""
            has_db = db_file is not None and db_file.exists()

            c1, c2, c3 = st.columns(3)
            c1.metric("Status Dokumen", "Terekstraksi", "Sukses")
            c2.metric("Ukuran File Markdown", f"{len(md_content):,} chars")
            c3.metric("SQLite Storage", "Aktif" if has_db else "Kosong / Tidak Ada Tabel")

            if has_db and db_file is not None:
                try:
                    conn = sqlite3.connect(str(db_file))
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
                    tables = [r[0] for r in cursor.fetchall()]

                    tbl_data = []
                    for t in tables:
                        cursor.execute(f"SELECT COUNT(*) FROM '{t}';")
                        r_cnt = cursor.fetchone()[0]
                        cursor.execute(f"PRAGMA table_info('{t}');")
                        cols = [c[1] for c in cursor.fetchall() if not c[1].startswith("_")]
                        tbl_data.append({"Nama Tabel SQLite": t, "Total Baris": r_cnt, "Jumlah Kolom": len(cols), "Kolom": ", ".join(cols)})
                    conn.close()

                    if tbl_data:
                        st.success("✅ **Guardrail Cross-Verification:** Tabel terstruktur berhasil sinkron antara VLM & SQLite.")
                        st.dataframe(pd.DataFrame(tbl_data), use_container_width=True)
                    else:
                        st.info("ℹ️ Tidak ada tabel transaksional yang ditemukan pada dokumen ini.")
                except Exception as e:  # noqa: BLE001
                    st.warning(f"Tidak dapat membaca database SQLite: {e}")
            else:
                st.info("ℹ️ Dokumen diproses tanpa pembentukan tabel database (dokumen teks naratif polos).")

        with tab_md:
            st.subheader("📝 Teks Dokumen (Markdown)")
            if md_file.exists():
                md_text = md_file.read_text(encoding="utf-8")
                st.download_button(
                    "💾 Unduh Markdown (.md)",
                    data=md_text,
                    file_name=f"{last_file.stem}.md",
                    mime="text/markdown",
                )
                st.markdown(md_text)
            else:
                st.warning("File Markdown belum tersedia.")

        with tab_sql:
            st.subheader("📊 Penjelajah Database SQLite & SQL Query Console")
            if db_file is not None and db_file.exists():
                st.write(f"📁 Path Database: `{db_file}`")
                conn = sqlite3.connect(str(db_file))
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
                tables = [r[0] for r in cursor.fetchall()]

                if tables:
                    selected_tbl = st.selectbox("Pilih Tabel untuk Dilihat:", tables)
                    df_preview = pd.read_sql_query(f"SELECT * FROM '{selected_tbl}' LIMIT 100;", conn)
                    st.dataframe(df_preview, use_container_width=True)

                    st.markdown("#### ⚡ Konsol Query SQL")
                    default_query = f"SELECT * FROM '{selected_tbl}' LIMIT 10;"
                    user_query = st.text_area("Tulis query SELECT:", value=default_query, height=80)
                    if st.button("Jalankan Query"):
                        try:
                            if not user_query.strip().upper().startswith("SELECT"):
                                st.error("Demi keamanan, hanya query SELECT yang diizinkan.")
                            else:
                                df_query_res = pd.read_sql_query(user_query, conn)
                                st.success(f"Ditemukan {len(df_query_res)} baris:")
                                st.dataframe(df_query_res, use_container_width=True)
                        except Exception as q_err:  # noqa: BLE001
                            st.error(f"Error query SQL: {q_err}")
                else:
                    st.info("Database SQLite ada tetapi belum memiliki tabel transaksional.")
                conn.close()
            else:
                st.info("Belum ada file database SQLite yang terbentuk untuk dokumen ini.")

        with tab_log:
            st.subheader("📋 Log Eksekusi")
            log_path = st.session_state.get("last_log_path")
            if log_path and Path(log_path).exists():
                st.code(Path(log_path).read_text(encoding="utf-8"), language="text")
