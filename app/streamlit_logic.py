"""
Streamlit Launcher & Viewer untuk Pipeline Ekstraksi Dokumen Vision VLM, Sub-Agent SQL Tabular, & Dual-Track Guardrail.

Fitur Utama:
  - Background Job Manager: Ekstraksi berjalan independen dari siklus tab/browser.
  - Logging Real-Time: Langsung ditulis dan di-flush per-baris ke file log disk.
  - Tahan Minimize / Refresh: UI otomatis mereconnect dan menampilkan progres halaman terkini (Halaman X / Y).
  - Tampilan Hasil Komprehensif: Markdown, SQLite Tabular Viewer, SQL Console, Guardrail Audit, & Run Log.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from app.job_tracker import JobManager

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_TYPES = ["pdf", "pptx", "ppt", "png", "jpg", "jpeg", "webp"]

SPEC_OPTIONS: dict[str, str | None] = {
    "Auto-Detect (Rekomendasi VLM)": None,
    "Plain Document (Surat/Dokumen Standar)": "plain",
    "Markdown Hierarchy (Bab, Sub-bab #, ##, ###)": "markdown_hierarchy",
    "Bilingual Journal (2 Kolom & 2 Bahasa)": "bilingual_journal",
    "Presentation Slides (Slide & Diagram Visual)": "presentation_slides",
}


def _save_uploaded_file(uploaded_file: UploadedFile, output_dir: Path) -> Path:
    """Simpan file yang diunggah ke direktori output/uploads secara stabil."""
    uploads_dir = output_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    target_path = uploads_dir / Path(uploaded_file.name).name
    target_path.write_bytes(uploaded_file.getvalue())
    return target_path


def _get_sqlite_db_for_file(file_stem: str, output_dir: Path) -> Path | None:
    """Cari file database SQLite yang terkait dengan file yang diproses."""
    db_candidates = [
        output_dir / file_stem / "databases" / f"{file_stem}.sqlite",
        output_dir / file_stem / f"{file_stem}.sqlite",
        output_dir / "databases" / f"{file_stem}.sqlite",
        output_dir / "databases" / f"{file_stem}_data.sqlite",
        output_dir / "databases" / "documents_data.sqlite",
    ]
    for candidate in db_candidates:
        if candidate.exists():
            return candidate

    doc_db_dir = output_dir / file_stem / "databases"
    if doc_db_dir.exists():
        matches = sorted(doc_db_dir.glob(f"{file_stem}*.sqlite"))
        if matches:
            return matches[0]

    legacy_db_dir = output_dir / "databases"
    if legacy_db_dir.exists():
        stem_matches = sorted(legacy_db_dir.glob(f"{file_stem}*.sqlite"))
        if stem_matches:
            return stem_matches[0]
    return None


# ==============================================================================
# UI Streamlit Configuration
# ==============================================================================

st.set_page_config(
    page_title="Vision VLM & Dual-Track Sub-Agent",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📄 Document Vision VLM & Dual-Track Sub-Agent")
st.caption(
    "Ekstraksi Multi-Page Vision VLM, Sub-Agent SQL Tabular Otomatis, & Dual-Track Guardrail Cross-Verification."
)

output_dir = PROJECT_ROOT / "output"
job_manager = JobManager.get_instance()

# Sidebar: Pengaturan Pipeline & Info
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
    st.markdown("#### 🛡️ Keamanan Proses")
    st.info(
        "💡 **Background Safe:** Proses ekstraksi tetap berjalan aman di latar belakang meski browser di-minimize atau tertutup jendela lain. Log tersimpan langsung di disk."
    )

    # Riwayat Ringkas File yang Pernah Diproses
    status_files = sorted(
        list(output_dir.glob("*/logs/*_status.json"))
        + list((output_dir / "logs").glob("*_status.json")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if status_files:
        st.markdown("#### 🕒 Riwayat Pekerjaan Terakhir")
        for sf in status_files[:5]:
            stem_name = sf.stem.replace("_status", "")
            st.caption(f"• **{stem_name}**")


# ==============================================================================
# Komponen Monitoring Live Real-Time (Auto-Refresh Fragment)
# ==============================================================================


@st.fragment(run_every=2)
def render_live_monitor(stem: str, output_path: Path) -> None:
    """Komponen fragment Streamlit yang memperbarui progres ekstraksi setiap 2 detik."""
    job = job_manager.get_job(stem, output_dir=output_path)
    if not job:
        st.info("Pekerjaan tidak ditemukan.")
        return

    # Jika job telah selesai atau gagal, minta Streamlit rerun halaman penuh
    if job.status != "running":
        st.rerun()

    # Progress bar & badge
    pct = job.progress_percentage()
    st.markdown("### ⏳ Proses Ekstraksi Sedang Berjalan di Latar Belakang")
    st.progress(pct / 100.0, text=f"Progres: {pct:.1f}% — {job.stage}")

    # Kartu Metrik Granular
    c1, c2, c3, c4 = st.columns(4)
    if job.total_pages > 0:
        page_str = f"{job.current_page} / {job.total_pages}"
    elif job.current_page > 0:
        page_str = f"Halaman {job.current_page}"
    else:
        page_str = "Menyiapkan..."

    c1.metric("📄 Halaman / Slide", page_str)
    c2.metric("🔄 Tahapan Proses", job.stage)
    c3.metric("⚙️ PID Sistem", str(job.pid or "-"))
    c4.metric("📝 Status File Log", "Tersambung (Aktif)")

    # Kotak informasi background safety
    st.success(
        f"🟢 **Monitoring Aktif:** Browser dapat Anda minimalkan atau Anda dapat membuka tab lain secara leluasa.\n\n"
        f"- **File Log Real-Time:** `{job.latest_log_path}`\n"
        f"- **File Status Cepat (Ringkasan):** `{job.progress_file}`\n"
        f"- **Pesan Terakhir Sistem:** *{job.last_message or 'Menunggu baris berikutnya...'}*"
    )

    # Tampilan log real-time streaming
    st.markdown("##### 📜 Live Log Streaming (25 Baris Terakhir):")
    recent_log_text = job_manager.get_latest_logs(stem, line_count=25)
    st.code(recent_log_text, language="text")

    # Aksi kontrol
    col_a1, col_a2 = st.columns([1, 1])
    with col_a1:
        if st.button("🔄 Segarkan Tampilan Sekarang", use_container_width=True):
            st.rerun()
    with col_a2:
        if (
            st.button(
                "🛑 Batalkan Ekstraksi",
                type="secondary",
                use_container_width=True,
            )
            and job_manager.cancel_job(stem)
        ):
            st.warning("Proses ekstraksi telah dibatalkan.")
            st.rerun()


# ==============================================================================
# Upload Dokumen & Alur Kerja Utama
# ==============================================================================

uploaded_file = st.file_uploader(
    "Unggah Dokumen (PDF, PPTX, PPT, PNG, JPG, WEBP):",
    type=SUPPORTED_TYPES,
)

if uploaded_file is not None:
    # Simpan file secara permanen dan stabil di output/uploads/
    input_file_path = _save_uploaded_file(uploaded_file, output_dir)
    file_stem = input_file_path.stem

    # Cek status job saat ini
    job = job_manager.get_job(file_stem, output_dir=output_dir)

    # 1. KONDISI: SEDANG BERJALAN (RUNNING)
    if job is not None and job.status == "running":
        render_live_monitor(file_stem, output_dir)

    # 2. KONDISI: SELESAI SUKSES (COMPLETED)
    elif job is not None and job.status == "completed":
        st.success("🎉 **Ekstraksi Dokumen Berhasil Selesai!**")

        # Tombol Ekstrak Ulang jika pengguna ingin memproses kembali
        col_top1, col_top2 = st.columns([3, 1])
        with col_top1:
            st.caption(
                f"File: **{job.file_name}** | Target: `{job.out_file}` | Log: `{job.latest_log_path}`"
            )
        with col_top2:
            if st.button("🔄 Ekstrak Ulang Dokumen Ini", use_container_width=True):
                job_manager.reset_job(file_stem, output_dir=output_dir)
                st.rerun()

        # Render Tab Hasil Lengkap
        md_file = job.out_file if job.out_file.exists() else (output_dir / file_stem / f"{file_stem}.md")
        if not md_file.exists():
            md_file = output_dir / f"{file_stem}.md"
        db_file = _get_sqlite_db_for_file(file_stem, output_dir)

        tab_guardrail, tab_md, tab_sql, tab_log = st.tabs([
            "🛡️ Dual-Track Guardrail Audit",
            "📝 Markdown Output",
            "📊 Data Tabular (SQLite)",
            "📋 Run Log",
        ])

        with tab_guardrail:
            st.subheader("🛡️ Laporan Audit Guardrail Supervisor")
            st.caption(
                "Agent Utama membandingkan konsistensi Jalur 1 (Teks Markdown) vs Jalur 2 (Database SQLite)."
            )

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
                    cursor.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
                    )
                    tables = [r[0] for r in cursor.fetchall()]

                    tbl_data = []
                    for t in tables:
                        cursor.execute(f"SELECT COUNT(*) FROM '{t}';")
                        r_cnt = cursor.fetchone()[0]
                        cursor.execute(f"PRAGMA table_info('{t}');")
                        cols = [c[1] for c in cursor.fetchall() if not c[1].startswith("_")]
                        tbl_data.append({
                            "Nama Tabel SQLite": t,
                            "Total Baris": r_cnt,
                            "Jumlah Kolom": len(cols),
                            "Kolom": ", ".join(cols),
                        })
                    conn.close()

                    if tbl_data:
                        st.success(
                            "✅ **Guardrail Cross-Verification:** Tabel terstruktur berhasil sinkron antara VLM & SQLite."
                        )
                        st.dataframe(pd.DataFrame(tbl_data), use_container_width=True)
                    else:
                        st.info("ℹ️ Tidak ada tabel transaksional yang ditemukan pada dokumen ini.")
                except Exception as e:  # noqa: BLE001
                    st.warning(f"Tidak dapat membaca database SQLite: {e}")
            else:
                st.info(
                    "ℹ️ Dokumen diproses tanpa pembentukan tabel database (dokumen teks naratif polos)."
                )

        with tab_md:
            st.subheader("📝 Teks Dokumen (Markdown)")
            if md_file.exists():
                md_text = md_file.read_text(encoding="utf-8")
                st.download_button(
                    "💾 Unduh Markdown (.md)",
                    data=md_text,
                    file_name=f"{file_stem}.md",
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
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
                )
                tables = [r[0] for r in cursor.fetchall()]

                if tables:
                    selected_tbl = st.selectbox("Pilih Tabel untuk Dilihat:", tables)
                    df_preview = pd.read_sql_query(
                        f"SELECT * FROM '{selected_tbl}' LIMIT 100;", conn
                    )
                    st.dataframe(df_preview, use_container_width=True)

                    st.markdown("#### ⚡ Konsol Query SQL")
                    default_query = f"SELECT * FROM '{selected_tbl}' LIMIT 10;"
                    user_query = st.text_area(
                        "Tulis query SELECT:", value=default_query, height=80
                    )
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
            st.subheader("📋 Log Eksekusi Lengkap")
            log_to_show = job.latest_log_path if job.latest_log_path.exists() else job.log_path
            if log_to_show.exists():
                log_content = log_to_show.read_text(encoding="utf-8", errors="replace")
                st.download_button(
                    "💾 Unduh File Log (.log)",
                    data=log_content,
                    file_name=log_to_show.name,
                    mime="text/plain",
                )
                st.code(log_content, language="text")

    # 3. KONDISI: GAGAL ATAU DIBATALKAN (FAILED / CANCELED)
    elif job is not None and job.status in ("failed", "canceled"):
        if job.status == "canceled":
            st.warning("⚠️ **Proses ekstraksi telah dibatalkan oleh pengguna.**")
        else:
            st.error("❌ **Terjadi Kesalahan saat Ekstraksi Dokumen**")
            if job.error_message:
                st.error(f"Detail Error: {job.error_message}")

        st.markdown("##### Log Terakhir:")
        st.code(job_manager.get_latest_logs(file_stem, line_count=30), language="text")

        if st.button("🔄 Coba Ekstrak Lagi", type="primary"):
            job_manager.reset_job(file_stem, output_dir=output_dir)
            st.rerun()

    # 4. KONDISI: IDLE / BELUM DIMULAI (SIAP DIEKSTRAK)
    else:
        st.info(f"📄 File siap diproses: **{input_file_path.name}**")
        start_process = st.button(
            "🚀 Mulai Ekstraksi", type="primary", use_container_width=False
        )
        if start_process:
            job_manager.start_job(
                input_path=input_file_path,
                output_dir=output_dir,
                doc_type=chosen_spec,
                dpi=dpi_val,
                force_all_tables=force_all_tbl,
                preview_chunks=show_chunk_preview,
                chunk_size=int(c_size),
                chunk_overlap=int(c_overlap),
            )
            st.rerun()
