"""
Hierarchy — Mesin Tumpukan & Auditor Hirarki Peraturan Perundang-undangan.

Pasangan deterministik dari `surveyor.py`. Kalau Surveyor memetakan bentuk
halaman, modul ini menyusun bentuk isinya.

Pembagian kerjanya disengaja: VLM agent hanya melaporkan apa yang dilihatnya
sebagai aliran event datar — "di sini ada Pasal 30", "di sini ada ayat (1)" —
tanpa pernah menyebut kedalaman, tanpa pernah menulis `#`. Kedalaman dan
penyarangan disusun di sini oleh mesin tumpukan, berdasarkan `level_map` yang
urutannya sudah dipatok UU 12/2011 Lampiran II.

Akibatnya dua kelas kesalahan menjadi mustahil, bukan sekadar jarang:
  - level drift: Pasal tidak bisa jadi `###` di satu batch lalu `##` di batch
    berikutnya, sebab bukan VLM yang menentukan levelnya
  - kehilangan induk: Pasal 30 otomatis menempel pada Bab yang masih terbuka,
    sebab tumpukan hanya ditutup oleh event yang levelnya lebih tinggi

Auditor memeriksa kesinambungan ordinal, dan hanya memperbaiki bila nilainya
tertentu tunggal. Begitu ada dua kemungkinan, ia mengeskalasi, tidak menebak.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

from pydantic import BaseModel, Field

# --- Peta level -------------------------------------------------------------
# Urutan mengikuti UU 12/2011 Lampiran II butir 85-95. Indeks kecil = level
# tinggi. Bagian pembukaan (menimbang/mengingat) diberi level tersendiri karena
# penomorannya punya ruang nama sendiri, terpisah dari batang tubuh.

LEVEL_ORDER: list[str] = [
    "judul",
    "pembukaan",  # Menimbang / Mengingat / Menetapkan
    "buku",
    "bab",
    "bagian",
    "paragraf",
    "pasal",
    "ayat",
    "huruf",
    "angka",
    "huruf2",
    "angka2",
]

#: Jenis penomoran tiap level, dipakai Auditor untuk memeriksa kesinambungan.
ORDINAL_KIND: dict[str, str] = {
    "buku": "romawi",
    "bab": "romawi",
    "bagian": "kata",
    "paragraf": "arab",
    "pasal": "arab",
    "ayat": "arab",
    "huruf": "alfabet",
    "angka": "arab",
    "huruf2": "alfabet",
    "angka2": "arab",
    "pembukaan": "bebas",
    "judul": "bebas",
}

#: Level yang penomorannya dimulai ulang di dalam tiap induk.
RESETS_INSIDE_PARENT = {"ayat", "huruf", "angka", "huruf2", "angka2", "paragraf", "bagian"}

#: Scope yang tidak boleh menampung level tertentu, walau levelnya lebih rendah.
#: Pembukaan (Menimbang/Mengingat/MEMUTUSKAN) adalah saudara batang tubuh, bukan
#: induknya: begitu Pasal atau Bab pertama muncul, pembukaan harus ditutup.
BLOCKED_CHILDREN: dict[str, set[str]] = {
    "pembukaan": {"buku", "bab", "bagian", "paragraf", "pasal"},
}

#: Butir 87 UU 12/2011: pembagian rincian tidak melebihi 4 tingkat.
MAX_RINCIAN_DEPTH = 4

_KATA_BILANGAN = [
    "kesatu", "kedua", "ketiga", "keempat", "kelima", "keenam",
    "ketujuh", "kedelapan", "kesembilan", "kesepuluh",
]

_ROMAN_VALUES = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
    (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]


# --- Model ------------------------------------------------------------------


class Event(BaseModel):
    """Satu pengamatan mentah dari VLM agent. Tidak memuat kedalaman."""

    kind: str = Field(..., description="Salah satu nilai di LEVEL_ORDER, atau 'teks'")
    ordinal: str | None = Field(default=None, description="Nomor apa adanya: 'I', '30', 'a', 'Kesatu'")
    label: str | None = Field(default=None, description="Judul, mis. 'KETENTUAN UMUM'")
    text: str | None = Field(default=None, description="Isi tekstual")
    page: int = Field(..., description="Halaman tempat event terlihat")


class Node(BaseModel):
    """Simpul pohon hasil susunan mesin tumpukan."""

    kind: str
    ordinal: str | None = None
    label: str | None = None
    text: str = ""
    page_start: int = 0
    page_end: int = 0
    children: list["Node"] = Field(default_factory=list)


class ScopeRef(BaseModel):
    """Satu entri jalur terbuka pada cursor."""

    kind: str
    ordinal: str | None = None
    label: str | None = None
    since_page: int = 0


class Cursor(BaseModel):
    """
    Keadaan yang dioper antar batch. Ukurannya O(kedalaman), bukan O(halaman).

    `open_path` menjawab "sedang di dalam apa"; `last_seen` menjawab "nomor
    terakhir yang terlihat di tiap level". Keduanya cukup untuk melanjutkan
    penyarangan tanpa mengirim ulang seluruh daftar isi.
    """

    doc_id: str = ""
    open_path: list[ScopeRef] = Field(default_factory=list)
    last_seen: dict[str, str] = Field(default_factory=dict)
    tail: str = Field(default="", description="Ekor teks halaman terakhir, untuk menyambung kalimat")
    catchword: str = Field(default="", description="Kata penyambung di kaki halaman terakhir")


class Violation(BaseModel):
    """Temuan Auditor."""

    severity: str = Field(..., description="'perbaiki' bila nilainya tunggal, 'eskalasi' bila ambigu")
    kind: str
    page: int
    message: str
    proposed: str | None = Field(default=None, description="Nilai perbaikan bila tertentu tunggal")


# --- Utilitas ordinal -------------------------------------------------------


def to_int(ordinal: str | None, kind: str) -> int | None:
    """Ubah ordinal apa pun menjadi bilangan urut, atau None bila tak terbaca."""
    if not ordinal:
        return None
    raw = ordinal.strip().strip(".()[]").strip()
    style = ORDINAL_KIND.get(kind, "bebas")

    if style == "romawi":
        return _roman_to_int(raw.upper())
    if style == "kata":
        low = raw.lower()
        return _KATA_BILANGAN.index(low) + 1 if low in _KATA_BILANGAN else None
    if style == "alfabet":
        low = raw.lower()
        return ord(low) - ord("a") + 1 if len(low) == 1 and low.isalpha() else None
    if style == "arab":
        # Harus seluruhnya angka. Membuang huruf secara diam-diam berbahaya:
        # "L2" (OCR untuk "12") akan terbaca 2, dan kesalahan itu lolos sebagai
        # nilai yang sah. Lebih baik dinyatakan tak terbaca agar Auditor bisa
        # menyimpulkannya dari urutan.
        return int(raw) if raw.isdigit() else None
    return None


def from_int(value: int, kind: str) -> str:
    """Kebalikan `to_int`: susun kembali ordinal sesuai gaya penomoran level."""
    style = ORDINAL_KIND.get(kind, "bebas")
    if style == "romawi":
        return _int_to_roman(value)
    if style == "kata":
        return _KATA_BILANGAN[value - 1].capitalize() if 1 <= value <= len(_KATA_BILANGAN) else str(value)
    if style == "alfabet":
        return chr(ord("a") + value - 1)
    return str(value)


def _roman_to_int(text: str) -> int | None:
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    if not text or any(c not in values for c in text):
        return None
    total, prev = 0, 0
    for char in reversed(text):
        current = values[char]
        total += -current if current < prev else current
        prev = max(prev, current)
    return total


def _int_to_roman(value: int) -> str:
    out = []
    for amount, numeral in _ROMAN_VALUES:
        while value >= amount:
            out.append(numeral)
            value -= amount
    return "".join(out)


def _level_index(kind: str) -> int:
    return LEVEL_ORDER.index(kind) if kind in LEVEL_ORDER else len(LEVEL_ORDER)


# --- Mesin tumpukan ---------------------------------------------------------


class StackMachine:
    """
    Menyusun aliran event datar menjadi pohon bersarang.

    Aturannya satu kalimat: event menutup semua scope yang levelnya sama atau
    lebih rendah, lalu menempel pada scope yang tersisa di puncak tumpukan.
    Karena itu Pasal 30 tetap berada di dalam Bab I selama belum ada event
    `bab` yang baru, walau keduanya terpisah ratusan halaman.
    """

    def __init__(self, cursor: Cursor | None = None) -> None:
        self.root = Node(kind="dokumen")
        self.stack: list[Node] = [self.root]
        self.last_seen: dict[str, str] = dict(cursor.last_seen) if cursor else {}
        self._restore(cursor)

    def _restore(self, cursor: Cursor | None) -> None:
        """Bangun ulang jalur terbuka dari cursor batch sebelumnya."""
        if not cursor:
            return
        for scope in cursor.open_path:
            node = Node(
                kind=scope.kind,
                ordinal=scope.ordinal,
                label=scope.label,
                page_start=scope.since_page,
                page_end=scope.since_page,
            )
            self.stack[-1].children.append(node)
            self.stack.append(node)

    def apply(self, event: Event) -> None:
        """Terapkan satu event ke tumpukan."""
        if event.kind == "teks":
            target = self.stack[-1]
            target.text = (target.text + " " + (event.text or "")).strip()
            target.page_end = max(target.page_end, event.page)
            return

        level = _level_index(event.kind)
        while len(self.stack) > 1 and (
            _level_index(self.stack[-1].kind) >= level
            or event.kind in BLOCKED_CHILDREN.get(self.stack[-1].kind, ())
        ):
            self.stack.pop()

        node = Node(
            kind=event.kind,
            ordinal=event.ordinal,
            label=event.label,
            text=(event.text or "").strip(),
            page_start=event.page,
            page_end=event.page,
        )
        self.stack[-1].children.append(node)
        self.stack.append(node)

        if event.ordinal:
            self.last_seen[event.kind] = event.ordinal
            # Level yang menomori ulang di dalam induknya kehilangan riwayat
            # begitu induknya berganti; kalau tidak, validator akan salah alarm.
            for lower in LEVEL_ORDER[_level_index(event.kind) + 1 :]:
                if lower in RESETS_INSIDE_PARENT:
                    self.last_seen.pop(lower, None)

    def run(self, events: Iterable[Event]) -> Node:
        """Terapkan seluruh aliran event, kembalikan akar pohon."""
        for event in events:
            self.apply(event)
        return self.root

    def cursor_out(self, doc_id: str = "", tail: str = "") -> Cursor:
        """Keadaan yang harus dioper ke batch berikutnya."""
        return Cursor(
            doc_id=doc_id,
            open_path=[
                ScopeRef(
                    kind=node.kind,
                    ordinal=node.ordinal,
                    label=node.label,
                    since_page=node.page_start,
                )
                for node in self.stack[1:]
            ],
            last_seen=dict(self.last_seen),
            tail=tail,
        )


# --- Auditor ----------------------------------------------------------------


def audit(events: Sequence[Event], cursor: Cursor | None = None) -> list[Violation]:
    """
    Periksa kesinambungan ordinal sepanjang aliran event.

    Perbaikan hanya diusulkan bila ruang nilai yang mungkin tinggal satu,
    misalnya "11, L2, 13" yang hanya bisa berarti 12. Selebihnya dieskalasi.
    """
    violations: list[Violation] = []
    last: dict[str, int] = {}
    if cursor:
        for kind, raw in cursor.last_seen.items():
            value = to_int(raw, kind)
            if value is not None:
                last[kind] = value

    depth_stack: list[str] = []

    for i, event in enumerate(events):
        if event.kind == "teks":
            continue

        # Rincian tidak boleh lebih dari 4 tingkat (butir 87 UU 12/2011).
        if event.kind in {"huruf", "angka", "huruf2", "angka2"}:
            level = _level_index(event.kind)
            while depth_stack and _level_index(depth_stack[-1]) >= level:
                depth_stack.pop()
            depth_stack.append(event.kind)
            if len(depth_stack) > MAX_RINCIAN_DEPTH:
                violations.append(
                    Violation(
                        severity="eskalasi",
                        kind=event.kind,
                        page=event.page,
                        message=f"kedalaman rincian {len(depth_stack)} melebihi batas {MAX_RINCIAN_DEPTH}",
                    )
                )
        else:
            depth_stack.clear()

        # Level yang menomori ulang: buang riwayatnya begitu induk berganti.
        for lower in LEVEL_ORDER[_level_index(event.kind) + 1 :]:
            if lower in RESETS_INSIDE_PARENT:
                last.pop(lower, None)

        if ORDINAL_KIND.get(event.kind) in {None, "bebas"}:
            continue

        value = to_int(event.ordinal, event.kind)
        previous = last.get(event.kind)

        if value is None:
            expected = previous + 1 if previous is not None else 1
            violations.append(
                Violation(
                    severity="perbaiki",
                    kind=event.kind,
                    page=event.page,
                    message=f"ordinal {event.ordinal!r} tak terbaca; urutan menuntut {from_int(expected, event.kind)}",
                    proposed=from_int(expected, event.kind),
                )
            )
            last[event.kind] = expected
            continue

        if previous is None:
            if value != 1:
                violations.append(
                    Violation(
                        severity="eskalasi",
                        kind=event.kind,
                        page=event.page,
                        message=f"{event.kind} dibuka pada {event.ordinal}, bukan awal urutan",
                    )
                )
        elif value == previous + 1:
            pass
        elif value <= previous:
            violations.append(
                Violation(
                    severity="eskalasi",
                    kind=event.kind,
                    page=event.page,
                    message=f"{event.kind} mundur: {from_int(previous, event.kind)} -> {event.ordinal}",
                )
            )
        elif value == previous + 2:
            # Satu nomor terlewat. Bisa jadi salah baca, bisa jadi memang hilang;
            # dua kemungkinan berarti tidak boleh diperbaiki sendiri.
            missing = from_int(previous + 1, event.kind)
            violations.append(
                Violation(
                    severity="eskalasi",
                    kind=event.kind,
                    page=event.page,
                    message=f"{event.kind} {missing} tidak terlihat ({from_int(previous, event.kind)} -> {event.ordinal})",
                )
            )
        else:
            violations.append(
                Violation(
                    severity="eskalasi",
                    kind=event.kind,
                    page=event.page,
                    message=f"{event.kind} melompat jauh: {from_int(previous, event.kind)} -> {event.ordinal}",
                )
            )

        last[event.kind] = value

    return violations


def check_catchword(cursor: Cursor, first_events: Sequence[Event]) -> Violation | None:
    """
    Cocokkan kata penyambung di kaki halaman sebelumnya dengan pembuka batch ini.

    UU 12/2011 mewajibkan kata penyambung, jadi dokumen menyediakan sendiri
    penanda sambungan antar halaman. Pemeriksaan ini gratis bila tersedia.
    """
    if not cursor.catchword:
        return None
    head = " ".join((e.text or e.label or e.ordinal or "") for e in first_events[:3]).lower()
    needle = re.sub(r"[^\w\s]", " ", cursor.catchword.lower()).split()
    if not needle:
        return None
    if needle[-1] not in head and needle[0] not in head:
        return Violation(
            severity="eskalasi",
            kind="seam",
            page=first_events[0].page if first_events else 0,
            message=f"kata penyambung {cursor.catchword!r} tidak muncul di awal batch",
        )
    return None


# --- Renderer ---------------------------------------------------------------

_HEADING_LEVEL = {
    "judul": 1,
    "pembukaan": 2,
    "buku": 2,
    "bab": 2,
    "bagian": 3,
    "paragraf": 4,
    "pasal": 4,
}

_LIST_LEVEL = {"ayat": 0, "huruf": 1, "angka": 2, "huruf2": 3, "angka2": 4}


def _heading_text(node: Node) -> str:
    parts = [node.kind.capitalize() if node.kind != "bab" else "BAB"]
    if node.ordinal:
        parts.append(node.ordinal.upper() if node.kind in {"bab", "buku"} else node.ordinal)
    line = " ".join(parts)
    if node.label:
        line += f" — {node.label}"
    return line


def _bullet(node: Node) -> str:
    if node.kind == "ayat":
        return f"({node.ordinal})"
    if node.kind in {"huruf", "angka"}:
        return f"{node.ordinal}."
    return f"{node.ordinal})"


def render_markdown(node: Node, *, with_pages: bool = True) -> str:
    """Susun pohon menjadi Markdown bersarang. Level heading murni turunan pohon."""
    lines: list[str] = []

    def walk(current: Node, depth: int) -> None:
        for child in current.children:
            if child.kind in _HEADING_LEVEL:
                hashes = "#" * _HEADING_LEVEL[child.kind]
                marker = f"  <!-- hal.{child.page_start} -->" if with_pages else ""
                lines.append(f"\n{hashes} {_heading_text(child)}{marker}")
                if child.text:
                    lines.append(f"\n{child.text}")
                walk(child, 0)
            elif child.kind in _LIST_LEVEL:
                # Indentasi mengikuti kedalaman sebenarnya di pohon, bukan jenis
                # levelnya. Daftar angka di bawah "Mengingat" berada di tingkat
                # teratas, sedangkan angka di dalam huruf di dalam ayat menjorok.
                body = child.text or child.label or ""
                lines.append(f"{'  ' * depth}- {_bullet(child)} {body}".rstrip())
                walk(child, depth + 1)
            else:
                if child.text:
                    lines.append(f"\n{child.text}")
                walk(child, depth)

    walk(node, 0)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip() + "\n"


def format_violations(violations: Sequence[Violation]) -> str:
    """Laporan Auditor yang enak dibaca."""
    if not violations:
        return "Auditor: tidak ada pelanggaran."
    lines = [f"Auditor: {len(violations)} temuan"]
    for v in violations:
        tag = "PERBAIKI " if v.severity == "perbaiki" else "ESKALASI "
        fix = f"  -> {v.proposed}" if v.proposed else ""
        lines.append(f"  {tag} hal.{v.page:<4} {v.kind:<9} {v.message}{fix}")
    return "\n".join(lines)


Node.model_rebuild()


# --- Auditor nilai rupiah ---------------------------------------------------
#
# Dokumen anggaran menyebutkan angka yang sama berkali-kali dari sudut berbeda:
# total di satu pasal, rinciannya di pasal lain, saling menunjuk lewat frasa
# "sebagaimana dimaksud dalam Pasal 2 huruf a". Redundansi itu bisa dipakai
# sebagai uji silang gratis, seperti cek penjumlahan pada dokumen fiskal.

# "Rp", "Rp.", "Rp 1.000,00", "Rp.1.000,00" — titik sesudah Rp opsional, dan
# kurung menandakan nilai negatif (defisit).
_RP_PATTERN = re.compile(r"Rp\.?\s*\(?\s*([\d.]+),(\d{2})\)?")
# Huruf rujukan kadang ditulis berkurung ("huruf (c)"), jadi kurungnya opsional.
_REF_PASAL = re.compile(r"Pasal\s+(\d+)\s+huruf\s*\(?([a-z])\)?", re.IGNORECASE)
_REF_AYAT = re.compile(r"ayat\s*\((\d+)\)\s*huruf\s*\(?([a-z])\)?", re.IGNORECASE)


def parse_rupiah(text: str) -> float | None:
    """Ambil nilai rupiah pertama dari sebuah teks. Kurung berarti negatif."""
    match = _RP_PATTERN.search(text or "")
    if not match:
        return None
    value = float(match.group(1).replace(".", "")) + float(match.group(2)) / 100
    return -value if "(" in match.group(0) else value


def audit_amounts(root: Node) -> list[Violation]:
    """
    Periksa apakah rincian berjumlah sama dengan total yang dirujuknya.

    Rujukan dibaca dari kalimat pengantar tiap ayat, lalu nilainya dicari di
    indeks (pasal, ayat, huruf). Hanya rincian yang benar-benar menunjuk suatu
    total yang diperiksa; sisanya dilewati tanpa keluhan.
    """
    index: dict[tuple[str, str | None, str], float] = {}
    findings: list[Violation] = []

    def walk(node: Node, pasal: str | None, ayat: str | None) -> None:
        for child in node.children:
            if child.kind == "pasal":
                walk(child, child.ordinal, None)
            elif child.kind == "ayat":
                walk(child, pasal, child.ordinal)
            else:
                if child.kind == "huruf" and pasal and child.ordinal:
                    amount = parse_rupiah(child.text)
                    if amount is not None:
                        index[(pasal, ayat, child.ordinal.lower())] = amount
                walk(child, pasal, ayat)

    walk(root, None, None)

    def check(node: Node, pasal: str | None, ayat: str | None) -> None:
        for child in node.children:
            next_pasal = child.ordinal if child.kind == "pasal" else pasal
            next_ayat = child.ordinal if child.kind == "ayat" else ayat

            if child.kind in {"pasal", "ayat"} and child.text:
                amounts = [
                    parse_rupiah(g.text)
                    for g in child.children
                    if g.kind == "huruf" and parse_rupiah(g.text) is not None
                ]
                key = None
                ref = _REF_PASAL.search(child.text)
                if ref:
                    key = (ref.group(1), None, ref.group(2).lower())
                else:
                    ref = _REF_AYAT.search(child.text)
                    if ref and next_pasal:
                        key = (next_pasal, ref.group(1), ref.group(2).lower())

                if key and len(amounts) >= 2 and key in index:
                    total = sum(a for a in amounts if a is not None)
                    expected = index[key]
                    if abs(total - expected) > 0.5:
                        findings.append(
                            Violation(
                                severity="eskalasi",
                                kind="jumlah",
                                page=child.page_start,
                                message=(
                                    f"rincian {child.kind} {child.ordinal} berjumlah "
                                    f"{total:,.2f} tetapi Pasal {key[0]} huruf {key[2]} "
                                    f"menyebut {expected:,.2f}"
                                ),
                            )
                        )
            check(child, next_pasal, next_ayat)

    check(root, None, None)
    return findings


def count_amount_checks(root: Node) -> int:
    """Berapa banyak rincian yang punya rujukan total sehingga bisa diuji silang."""
    total = 0

    def walk(node: Node, pasal: str | None) -> None:
        nonlocal total
        for child in node.children:
            next_pasal = child.ordinal if child.kind == "pasal" else pasal
            if child.kind in {"pasal", "ayat"} and child.text:
                has_ref = bool(_REF_PASAL.search(child.text) or _REF_AYAT.search(child.text))
                amounts = [g for g in child.children if g.kind == "huruf" and parse_rupiah(g.text) is not None]
                if has_ref and len(amounts) >= 2:
                    total += 1
            walk(child, next_pasal)

    walk(root, None)
    return total
