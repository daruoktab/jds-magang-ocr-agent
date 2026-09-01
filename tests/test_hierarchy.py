"""Tes mesin tumpukan & auditor hirarki peraturan serta Sub-Agent SQL Dual-Track."""

from __future__ import annotations

import tempfile
from pathlib import Path

from app.hierarchy import (
    Cursor,
    Event,
    StackMachine,
    audit,
    audit_amounts,
    parse_rupiah,
    render_markdown,
)
from app.schemas import DualTrackGuardrailReport, PageTabularEvent
from app.tabular_db import (
    TabularDatabaseManager,
    cross_verify_dual_track,
    process_page_tabular_agent,
)


def ev(kind, ordinal=None, label=None, text=None, page=1):
    return Event(kind=kind, ordinal=ordinal, label=label, text=text, page=page)


def test_induk_dipulihkan_lintas_batch():
    """Batch kedua yang dibuka di tengah daftar tetap menempel pada induknya."""
    batch_1 = [
        ev("bab", "V", "TUGAS DAN FUNGSI"),
        ev("bagian", "Kesatu", "UPT"),
        ev("pasal", "5"),
        ev("huruf", "a", text="tugas pertama"),
    ]
    machine = StackMachine()
    machine.run(batch_1)
    cursor = machine.cursor_out()

    assert [s.kind for s in cursor.open_path] == ["bab", "bagian", "pasal", "huruf"]

    # Batch kedua tidak menyebut bab maupun pasal sama sekali.
    batch_2 = [ev("huruf", "b", text="tugas kedua", page=2)]
    lanjutan = StackMachine(cursor=cursor)
    lanjutan.run(batch_2)

    bab = lanjutan.root.children[0]
    pasal = bab.children[0].children[0]
    assert bab.kind == "bab" and bab.ordinal == "V"
    assert [h.ordinal for h in pasal.children] == ["a", "b"]


def test_pasal_menutup_bab_sebelumnya():
    """Bab baru menutup bab lama, bukan bersarang di dalamnya."""
    machine = StackMachine()
    machine.run([ev("bab", "I"), ev("pasal", "1"), ev("bab", "II"), ev("pasal", "2")])
    bab_i, bab_ii = machine.root.children
    assert [p.ordinal for p in bab_i.children] == ["1"]
    assert [p.ordinal for p in bab_ii.children] == ["2"]


def test_pembukaan_bukan_induk_batang_tubuh():
    """MEMUTUSKAN adalah saudara Pasal, bukan induknya."""
    machine = StackMachine()
    machine.run([ev("pembukaan", label="MEMUTUSKAN"), ev("pasal", "1")])
    assert [n.kind for n in machine.root.children] == ["pembukaan", "pasal"]


def test_ayat_menomori_ulang_di_tiap_pasal():
    """Ayat kembali ke (1) di pasal berikutnya tanpa dianggap mundur."""
    events = [
        ev("pasal", "1"),
        ev("ayat", "1"),
        ev("ayat", "2"),
        ev("pasal", "2"),
        ev("ayat", "1"),
    ]
    assert audit(events) == []


def test_ordinal_tak_terbaca_diperbaiki_bila_tunggal():
    """'L2' di antara 11 dan 13 hanya bisa berarti 12."""
    cursor = Cursor(last_seen={"pasal": "10"})
    events = [ev("pasal", "11"), ev("pasal", "L2"), ev("pasal", "13")]
    findings = audit(events, cursor=cursor)
    assert len(findings) == 1
    assert findings[0].severity == "perbaiki"
    assert findings[0].proposed == "12"


def test_huruf_salah_baca_ikut_diperbaiki():
    """Angka bukan satu-satunya: huruf pun disimpulkan dari urutan."""
    findings = audit([ev("huruf", "a"), ev("huruf", "1o"), ev("huruf", "c")])
    assert [f.proposed for f in findings] == ["b"]


def test_pasal_bersuffiks_hasil_amandemen():
    """Pasal 6A, 7B, 18B lazim pada UUD dan UU yang diamandemen."""
    urut = ["7", "7A", "7B", "7C", "8"]
    assert (
        audit([ev("pasal", o) for o in urut], cursor=Cursor(last_seen={"pasal": "6"}))
        == []
    )

    # Suffiks tetap tunduk pada urutan: 6 langsung ke 6B berarti 6A hilang.
    findings = audit(
        [ev("pasal", "6"), ev("pasal", "6B")], cursor=Cursor(last_seen={"pasal": "5"})
    )
    assert [f.severity for f in findings] == ["eskalasi"]


def test_nomor_hilang_dieskalasi_bukan_ditebak():
    """Lompatan satu nomor punya dua tafsir, jadi tidak boleh diperbaiki sendiri."""
    findings = audit([ev("pasal", "1"), ev("pasal", "3")])
    assert [f.severity for f in findings] == ["eskalasi"]


def test_cursor_membawa_kesinambungan_ordinal():
    """Pasal 30 sah bila cursor mencatat pasal terakhir 29."""
    cursor = Cursor(last_seen={"pasal": "29"})
    assert audit([ev("pasal", "30")], cursor=cursor) == []
    assert audit([ev("pasal", "35")], cursor=cursor)


def test_parse_rupiah_beragam_format():
    assert parse_rupiah("Rp 2.580.459.404.371,00") == 2580459404371.0
    assert parse_rupiah("Rp.448.081.357.000,00") == 448081357000.0
    assert parse_rupiah("Rp. (63.975.048.861,58)") == -63975048861.58
    assert parse_rupiah("Rp. NIHIL") is None


def test_auditor_rupiah_menemukan_rincian_tak_sejumlah():
    machine = StackMachine()
    machine.run(
        [
            ev("pasal", "4"),
            ev("ayat", "1", text="Pembiayaan terdiri dari:"),
            ev("huruf", "a", text="Penerimaan : Rp. 65.090.038.541,64"),
            ev(
                "ayat",
                "2",
                text="Penerimaan sebagaimana dimaksud pada ayat (1) huruf a terdiri dari:",
            ),
            ev("huruf", "a", text="SILPA : Rp. 65.090.038.541,64"),
            ev("huruf", "e", text="Penerimaan kembali pinjaman : Rp. 1.500.000.000,00"),
        ]
    )
    findings = audit_amounts(machine.root)
    assert len(findings) == 1
    assert "66,590,038,541.64" in findings[0].message


def test_indentasi_mengikuti_kedalaman_pohon():
    """Angka di bawah pembukaan rata kiri; angka di dalam ayat menjorok."""
    machine = StackMachine()
    machine.run(
        [
            ev("pembukaan", label="Mengingat"),
            ev("angka", "1", text="satu"),
            ev("pasal", "1"),
            ev("ayat", "1", text="ayat satu"),
            ev("huruf", "a", text="huruf a"),
        ]
    )
    markdown = render_markdown(machine.root, with_pages=False)
    assert "\n- 1. satu" in markdown
    assert "\n- (1) ayat satu" in markdown
    assert "\n  - a. huruf a" in markdown


def test_process_page_tabular_agent_multipage_append() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_mutasi.sqlite"

        # Halaman 1: Mutasi Rekening bagian 1
        page_1_md = """
# Laporan Mutasi Rekening Halaman 1
Berikut adalah tabel transaksi:

| Tanggal | Deskripsi | Debit | Kredit | Saldo |
|---|---|---|---|---|
| 2024-01-01 | Setoran Awal | 0 | 10,000,000 | 10,000,000 |
| 2024-01-02 | Transfer Keluar | 1,000,000 | 0 | 9,000,000 |
"""
        event1, res1 = process_page_tabular_agent(
            page_markdown=page_1_md,
            page_number=1,
            source_file="rekening_koran.pdf",
            db_path=db_path,
            table_name_prefix="rekening_koran",
            append_if_matching=True,
            force_all_tables=False,
        )

        assert isinstance(event1, PageTabularEvent)
        assert event1.page_number == 1
        assert event1.tables_detected == 1
        assert event1.rows_ingested_total == 2
        assert event1.status == "created_new_table"
        assert len(res1) == 1
        assert res1[0].verification_report is not None
        assert res1[0].verification_report.is_valid is True

        # Halaman 2: Lanjutan transaksi (skema sama)
        page_2_md = """
# Laporan Mutasi Rekening Halaman 2 (Lanjutan)

| Tanggal | Deskripsi | Debit | Kredit | Saldo |
|---|---|---|---|---|
| 2024-01-03 | Biaya Admin | 25,000 | 0 | 8,975,000 |
| 2024-01-04 | Penerimaan Bunga | 0 | 50,000 | 9,025,000 |
"""
        event2, res2 = process_page_tabular_agent(
            page_markdown=page_2_md,
            page_number=2,
            source_file="rekening_koran.pdf",
            db_path=db_path,
            table_name_prefix="rekening_koran",
            append_if_matching=True,
            force_all_tables=False,
        )

        assert event2.page_number == 2
        assert event2.tables_detected == 1
        assert event2.rows_ingested_total == 2
        assert event2.status == "appended_existing_table"
        assert len(res2) == 1

        # Cek database total baris
        db_mgr = TabularDatabaseManager(db_path)
        cnt_res = db_mgr.execute_query(
            f'SELECT COUNT(*) as cnt FROM "{res1[0].table_name}";'
        )
        assert cnt_res.rows[0]["cnt"] == 4

        # Jalankan Guardrail Cross-Verification oleh Master Supervisor
        stitched_md = f"{page_1_md}\n\n---\n\n{page_2_md}"
        guard_report = cross_verify_dual_track(
            stitched_markdown=stitched_md,
            db_path=db_path,
            source_file="rekening_koran.pdf",
            total_pages=2,
        )

        assert isinstance(guard_report, DualTrackGuardrailReport)
        assert guard_report.guardrail_status == "PASSED"
        assert guard_report.total_markdown_tables == 2
        assert guard_report.total_sqlite_tables == 1
        assert guard_report.total_sqlite_rows == 4
        assert len(guard_report.discrepancies) == 0


def test_process_page_tabular_agent_no_table() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "empty.sqlite"
        page_md = "Halaman ini hanya berisi teks narasi biasa tanpa tabel apapun."

        event, res = process_page_tabular_agent(
            page_markdown=page_md,
            page_number=1,
            source_file="surat.pdf",
            db_path=db_path,
        )

        assert event.tables_detected == 0
        assert event.status == "no_tables"
        assert len(res) == 0


def test_process_page_tabular_all_tables_ingested_including_narrative() -> None:
    """Memastikan seluruh data tabular (termasuk kualitatif / naratif) tetap di-ingest ke SQLite."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "narrative.sqlite"
        page_md = """
# Matriks Analisis Kualitatif
Berikut adalah tabel perbandingan fitur sistem:

| Fitur | Deskripsi Singkat | Status Implementasi | Catatan Tambahan |
|---|---|---|---|
| Autentikasi | Login menggunakan OAuth2 dan MFA | Selesai | Sudah lolos pentest |
| Otorisasi | RBAC berbasis peran pengguna | Dalam Proses | Menunggu approval tim security |
| Audit Trail | Pencatatan log seluruh mutasi | Selesai | Terintegrasi dengan SIEM |
"""
        event, res = process_page_tabular_agent(
            page_markdown=page_md,
            page_number=1,
            source_file="matriks_fitur.pdf",
            db_path=db_path,
            table_name_prefix="matriks_fitur",
            append_if_matching=True,
            force_all_tables=True,
        )

        assert event.tables_detected == 1
        assert event.rows_ingested_total == 3
        assert len(res) == 1
        assert event.tagged_markdown is not None
        assert "<!-- sqlite_table:" in event.tagged_markdown

        db_mgr = TabularDatabaseManager(db_path)
        cnt = db_mgr.execute_query(f'SELECT COUNT(*) as cnt FROM "{res[0].table_name}";').rows[0]["cnt"]
        assert cnt == 3


def test_multipage_bank_statement_tagged_metadata_single_sqlite_table() -> None:
    """Memastikan dokumen bank statement multi-halaman tersimpan dalam 1 tabel SQLite tunggal dan sinkron."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "bank_statement.sqlite"
        
        # Halaman 1
        p1_md = """
# Bank Statement Page 1
| Date | Description | Reference | Debit | Credit | Balance |
|---|---|---|---|---|---|
| 2024-01-01 | Opening Balance | REF001 | 0.00 | 5000000.00 | 5000000.00 |
| 2024-01-05 | ATM Withdrawal | REF002 | 500000.00 | 0.00 | 4500000.00 |
"""
        event1, _res1 = process_page_tabular_agent(
            page_markdown=p1_md,
            page_number=1,
            source_file="08_bank-statement-SYNTHETIC-1-2.pdf",
            db_path=db_path,
            table_name_prefix="08_bank_statement",
            append_if_matching=True,
        )
        assert event1.tables_detected == 1
        assert event1.rows_ingested_total == 2
        p1_tagged = event1.tagged_markdown or p1_md

        # Halaman 2 (Lanjutan)
        p2_md = """
# Bank Statement Page 2
| Date | Description | Reference | Debit | Credit | Balance |
|---|---|---|---|---|---|
| 2024-01-10 | Payroll Deposit | REF003 | 0.00 | 12000000.00 | 16500000.00 |
| 2024-01-15 | Electric Bill | REF004 | 750000.00 | 0.00 | 15750000.00 |
| 2024-01-20 | Transfer Out | REF005 | 2000000.00 | 0.00 | 13750000.00 |
"""
        event2, _res2 = process_page_tabular_agent(
            page_markdown=p2_md,
            page_number=2,
            source_file="08_bank-statement-SYNTHETIC-1-2.pdf",
            db_path=db_path,
            table_name_prefix="08_bank_statement",
            append_if_matching=True,
        )
        assert event2.tables_detected == 1
        assert event2.rows_ingested_total == 3
        assert event2.status == "appended_existing_table"
        p2_tagged = event2.tagged_markdown or p2_md

        # SQLite harus HANYA memiliki 1 tabel
        db_mgr = TabularDatabaseManager(db_path)
        tables = list(db_mgr.inspect_database()["tables"].keys())
        assert len(tables) == 1
        t_name = tables[0]
        total_rows = db_mgr.execute_query(f'SELECT COUNT(*) as cnt FROM "{t_name}";').rows[0]["cnt"]
        assert total_rows == 5

        # Cross Verification Dual Track
        stitched = f"{p1_tagged}\n\n<!-- PAGE: 2 -->\n\n{p2_tagged}"
        report = cross_verify_dual_track(
            stitched_markdown=stitched,
            db_path=db_path,
            source_file="08_bank-statement-SYNTHETIC-1-2.pdf",
            total_pages=2,
        )
        assert report.guardrail_status == "PASSED"
        assert report.total_sqlite_tables == 1
        assert report.total_sqlite_rows == 5
        assert len(report.discrepancies) == 0

