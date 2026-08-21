"""Tes mesin tumpukan & auditor hirarki peraturan."""

from __future__ import annotations

from app.hierarchy import (
    Cursor,
    Event,
    StackMachine,
    audit,
    audit_amounts,
    parse_rupiah,
    render_markdown,
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
        ev("pasal", "1"), ev("ayat", "1"), ev("ayat", "2"),
        ev("pasal", "2"), ev("ayat", "1"),
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
    assert audit([ev("pasal", o) for o in urut],
                 cursor=Cursor(last_seen={"pasal": "6"})) == []

    # Suffiks tetap tunduk pada urutan: 6 langsung ke 6B berarti 6A hilang.
    findings = audit([ev("pasal", "6"), ev("pasal", "6B")],
                     cursor=Cursor(last_seen={"pasal": "5"}))
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
    machine.run([
        ev("pasal", "4"),
        ev("ayat", "1", text="Pembiayaan terdiri dari:"),
        ev("huruf", "a", text="Penerimaan : Rp. 65.090.038.541,64"),
        ev("ayat", "2", text="Penerimaan sebagaimana dimaksud pada ayat (1) huruf a terdiri dari:"),
        ev("huruf", "a", text="SILPA : Rp. 65.090.038.541,64"),
        ev("huruf", "e", text="Penerimaan kembali pinjaman : Rp. 1.500.000.000,00"),
    ])
    findings = audit_amounts(machine.root)
    assert len(findings) == 1
    assert "66,590,038,541.64" in findings[0].message


def test_indentasi_mengikuti_kedalaman_pohon():
    """Angka di bawah pembukaan rata kiri; angka di dalam ayat menjorok."""
    machine = StackMachine()
    machine.run([
        ev("pembukaan", label="Mengingat"),
        ev("angka", "1", text="satu"),
        ev("pasal", "1"),
        ev("ayat", "1", text="ayat satu"),
        ev("huruf", "a", text="huruf a"),
    ])
    markdown = render_markdown(machine.root, with_pages=False)
    assert "\n- 1. satu" in markdown
    assert "\n- (1) ayat satu" in markdown
    assert "\n  - a. huruf a" in markdown
