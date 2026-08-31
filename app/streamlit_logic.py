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
from typing import Any

import pandas as pd
import streamlit as st

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
    uploaded_file: st.runtime.uploaded_file_manager.UploadedFile,
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
    timestamp = dt.datetime.now().isoformat(timespec="seconds")
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
    log_name = f"{input_path.stem}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    captured_lines: list[str] = []
    live_lines: list[str] = []
    if live_log is not None:
        live_log.empty()

    assert proc.stdout is not None
    for raw_line in iter(proc.stdout.readline, ""):
        if not raw_line:
            break
        captured_lines.append(raw_line)
        stripped = raw_line.rstrip("\n")
        live_lines.append(stripped)
        if live_log is not None:
            tail_lines = live_lines[-30:]
            live_log.code("\n".join(tail_lines), language="text")

    returncode = proc.wait()
    combined = ["STDOUT/STDERR (merged):"]
    if captured_lines:
        combined.extend(captured_lines)
    else:
        combined.append("(empty)")
    combined.append(f"Output markdown: {out_file}")
    _write_run_log(
        log_path,
        input_path=input_path,
        output_dir=output_dir,
        cmd=cmd,
        returncode=returncode,
        stdout="".join(captured_lines),
        stderr="",
    )
    return returncode, "".join(combined), log_path


def _get_sqlite_tables_info(sqlite_path: Path) -> dict[str, Any]:
    """Baca informasi skema dan isi tabel dari SQLite database."""
    if not sqlite_path.exists():
        return {}

    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    )
    tables = [row["name"] for row in cursor.fetchall()]

    details: dict[str, Any] = {}
    for tbl in tables:
        cursor.execute(f"PRAGMA table_info('{tbl}');")
        cols = [dict(c) for c in cursor.fetchall()]

        cursor.execute(f"SELECT COUNT(*) as cnt FROM '{tbl}';")
        count = cursor.fetchone()["cnt"]

        cursor.execute(f"SELECT * FROM '{tbl}' LIMIT 500;")
        rows = [dict(r) for r in cursor.fetchall()]

        details[tbl] = {
            "columns": cols,
            "row_count": count,
            "sample_rows": rows,
        }

    conn.close()
    return details


def main() -> None:
    st.set_page_config(
        page_title="Document Vision & Tabular SQL Pipeline",
        page_icon="📄",
        layout="wide",
    )

    st.title("📄 Document Vision OCR, Sub-Agent SQL & Guardrail Pipeline")
    st.caption(
        "Arsitektur Jalur Ganda (Dual-Track): Ekstraksi Markdown VLM + Sub-Agent SQL Mandiri Per-Halaman "
        "dengan Supervisi Guardrail Cross-Verification."
    )

    with st.sidebar:
        st.header("⚙️ Konfigurasi Input")
        uploaded_file = st.file_uploader(
            "Pilih file dokumen (PDF, PPTX, PPT, Gambar)",
            type=SUPPORTED_TYPES,
            help="Upload dokumen untuk diproses secara komprehensif.",
        )

        spec_label = st.selectbox(
            "Spesifikasi Tata Letak Dokumen",
            list(SPEC_OPTIONS.keys()),
            index=0,
            help="Pilih karakteristik layout atau biarkan VLM mendeteksinya secara otomatis.",
        )

        dpi = st.slider(
            "Render DPI (PDF)",
            min_value=100,
            max_value=300,
            value=200,
            step=50,
            help="Resolusi render gambar untuk PDF.",
        )

        st.subheader("🗄️ Opsi Database Tabular")
        auto_table_db = st.checkbox(
            "Auto-ingest Tabel ke SQLite",
            value=True,
            help="Sub-Agent SQL memproses tabel otomatis ke file database .sqlite terpisah.",
        )
        force_all_tables = st.checkbox(
            "Force All Tables (Termasuk Tabel Umum/Naratif)",
            value=False,
            disabled=not auto_table_db,
            help="Jika dicentang, seluruh tabel Markdown akan dibuatkan tabel SQLite.",
        )

        st.subheader("🧩 Opsi Text Chunking")
        preview_chunks = st.checkbox("Preview Chunks", value=True)
        chunk_size = st.number_input("Chunk Size", value=1000, step=100)
        chunk_overlap = st.number_input("Chunk Overlap", value=150, step=25)

        output_dir_input = st.text_input("Direktori Output", value="output")
        process_clicked = st.button(
            "🚀 Jalankan Ekstraksi", type="primary", use_container_width=True
        )

    if not process_clicked:
        st.info("👉 Silakan upload file dokumen di sidebar dan klik **Jalankan Ekstraksi**.")
        return

    if uploaded_file is None:
        st.warning("Pilih dan upload file dokumen terlebih dahulu.")
        return

    input_ext = Path(uploaded_file.name).suffix.lower().lstrip(".")
    if input_ext not in SUPPORTED_TYPES:
        st.error(f"Ekstensi file .{input_ext} tidak didukung.")
        return

    output_dir = Path(output_dir_input).expanduser().resolve()
    temp_input = _write_upload_to_temp(uploaded_file)
    file_stem = Path(uploaded_file.name).stem

    st.subheader("⏳ Proses Eksekusi Pipeline")
    live_log = st.empty()

    with st.spinner(f"Mengekstrak '{uploaded_file.name}' via Vision VLM & Sub-Agent SQL..."):
        code, log_output, log_path = _run_main_cli(
            input_path=temp_input,
            output_dir=output_dir,
            doc_type=SPEC_OPTIONS[spec_label],
            dpi=dpi,
            force_all_tables=force_all_tables,
            preview_chunks=preview_chunks,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            live_log=live_log,
        )

    if code == 0:
        st.success(f"Ekstraksi selesai dengan sukses! (Exit code: {code})")
    else:
        st.error(f"Terjadi kesalahan dalam pemrosesan (Exit code: {code}). Periksa log.")

    # Paths Output
    out_md_file = output_dir / f"{file_stem}.md"
    out_sqlite_file = output_dir / "databases" / f"{file_stem}.sqlite"

    # Tabbed Interface
    tab_md, tab_db, tab_guardrail, tab_logs = st.tabs([
        "📝 Hasil Markdown",
        "🗄️ Tabular Database (SQLite)",
        "🛡️ Dual-Track Guardrail Audit",
        "📜 Log Eksekusi Lengkap",
    ])

    # TAB 1: Markdown Content
    with tab_md:
        if out_md_file.exists():
            md_text = out_md_file.read_text(encoding="utf-8")
            st.download_button(
                "📥 Download File Markdown (.md)",
                data=md_text.encode("utf-8"),
                file_name=f"{file_stem}.md",
                mime="text/markdown",
                key="dl_md",
            )
            st.markdown("### Pratinjau Teks Markdown:")
            st.markdown(md_text)
            with st.expander("Lihat Raw Markdown Code"):
                st.code(md_text, language="markdown")
        else:
            st.warning("File Markdown tidak ditemukan.")

    # TAB 2: Tabular Database SQLite
    with tab_db:
        if out_sqlite_file.exists():
            st.success(f"Database SQLite berhasil dibuat di: `{out_sqlite_file}`")
            sqlite_data = _get_sqlite_tables_info(out_sqlite_file)

            if sqlite_data:
                st.write(
                    f"Ditemukan **{len(sqlite_data)} tabel** tersimpan dalam database SQLite:"
                )

                st.download_button(
                    "📥 Download Database SQLite (.sqlite)",
                    data=out_sqlite_file.read_bytes(),
                    file_name=f"{file_stem}.sqlite",
                    mime="application/x-sqlite3",
                    key="dl_sqlite",
                )

                selected_table = st.selectbox(
                    "Pilih Tabel untuk Ditampilkan:", list(sqlite_data.keys())
                )

                if selected_table:
                    tbl_info = sqlite_data[selected_table]
                    st.metric(label="Total Baris Data", value=tbl_info["row_count"])

                    rows = tbl_info["sample_rows"]
                    if rows:
                        df = pd.DataFrame(rows)
                        display_cols = [c for c in df.columns if not c.startswith("_")]
                        st.dataframe(df[display_cols], use_container_width=True)
                    else:
                        st.info("Tabel tidak memiliki baris data.")

                    # Interactive SQL Query Console
                    with st.expander("🔍 Interactive SQL Query Console", expanded=False):
                        sql_input = st.text_area(
                            "Ketik Query SQL (Hanya SELECT):",
                            value=f"SELECT * FROM {selected_table} LIMIT 20;",
                            height=80,
                        )
                        if st.button("Jalankan Query SQL", key="btn_run_sql"):
                            try:
                                conn = sqlite3.connect(str(out_sqlite_file))
                                q_df = pd.read_sql_query(sql_input, conn)
                                conn.close()
                                st.write(f"Hasil Query ({len(q_df)} baris):")
                                st.dataframe(q_df, use_container_width=True)
                            except Exception as err:  # noqa: BLE001
                                st.error(f"SQL Error: {err}")
            else:
                st.info(
                    "Database SQLite ada, namun belum ada tabel yang terisi. "
                    "Pastikan dokumen memiliki tabel atau aktifkan opsi 'Force All Tables'."
                )
        else:
            st.info(
                "Tidak ada file database SQLite yang dibuat untuk dokumen ini.\n\n"
                "Kemungkinan penyebab:\n"
                "1. Dokumen tidak memiliki format tabel Markdown GFM.\n"
                "2. Tabel yang ada diklasifikasikan sebagai tabel naratif kualitatif.\n"
                "3. Anda dapat mencentang opsi **'Force All Tables'** di sidebar untuk memaksa seluruh tabel masuk ke SQLite."
            )

    # TAB 3: Dual-Track Guardrail Audit
    with tab_guardrail:
        st.markdown("### 🛡️ Master Supervisor Dual-Track Cross-Verification Report")
        st.caption(
            "Verifikasi kualitas silang membandingkan Jalur 1 (Teks Markdown) vs Jalur 2 (Tabel SQLite) "
            "untuk menjamin kelengkapan baris data dan integritas agregasi numerik."
        )

        if out_sqlite_file.exists() and out_md_file.exists():
            sqlite_data = _get_sqlite_tables_info(out_sqlite_file)
            total_sql_rows = sum(t["row_count"] for t in sqlite_data.values())

            c1, c2, c3 = st.columns(3)
            c1.metric("Status Guardrail", "PASSED" if len(sqlite_data) > 0 else "WARNING")
            c2.metric("Total Tabel SQLite", len(sqlite_data))
            c3.metric("Total Baris Data SQLite", total_sql_rows)

            st.write("#### Ringkasan Per-Tabel:")
            summary_rows = []
            for t_name, t_info in sqlite_data.items():
                cols = [c["name"] for c in t_info["columns"] if not c["name"].startswith("_")]
                summary_rows.append({
                    "Nama Tabel": t_name,
                    "Jumlah Kolom": len(cols),
                    "Kolom Terdefinisi": ", ".join(cols),
                    "Total Baris": t_info["row_count"],
                    "Status Sinkronisasi": "Verified ✅" if t_info["row_count"] > 0 else "Empty ⚠️",
                })
            if summary_rows:
                st.table(pd.DataFrame(summary_rows))
            else:
                st.info("Belum ada tabel yang terdaftar.")
        else:
            st.info("Data belum tersedia untuk audit Guardrail. Jalankan ekstraksi terlebih dahulu.")

    # TAB 4: Execution Logs
    with tab_logs:
        st.write(f"**Path Log File:** `{log_path}`")
        st.code(log_output, language="text")


if __name__ == "__main__":
    main()
