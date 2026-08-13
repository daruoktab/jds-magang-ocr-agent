"""
Registry agent ekstraksi - SATU agent untuk SATU JENIS DOKUMEN.

Setiap ExtractionAgent punya prompt tuning khusus untuk jenis dokumennya
(struk, tabel, invoice, formulir, kartu nama, dll.), plus opsi model
sendiri dan schema Pydantic sendiri. Struktur output tetap diputuskan
model - prompt tuning hanya mengarahkan fokus ekstraksi.

Menambah jenis dokumen baru = tambah satu ExtractionAgent di
AGENT_REGISTRY (dan opsional model/schema khusus). Tidak ada kode lain
yang perlu diubah - dispatcher dan pipeline otomatis mengenalinya.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Type

from pydantic import BaseModel

from .extractor import VisionExtractor
from .llm import LLMBackend
from .prompts import GENERAL_USER_PROMPT, SYSTEM_EXTRACTOR
from .schemas import DocumentExtraction


@dataclass
class ExtractionAgent:
    """Agent ekstraksi untuk satu jenis dokumen."""

    name: str                                   # nama jenis (dipakai router/klasifikasi)
    description: str                            # deskripsi untuk agent utama saat klasifikasi
    user_prompt: str                            # prompt tuning khusus jenis ini
    model: Optional[str] = None                 # model khusus (None = pakai model default)
    schema: Type[BaseModel] = DocumentExtraction
    system_prompt: str = SYSTEM_EXTRACTOR

    def build(self, backend: LLMBackend) -> VisionExtractor:
        return VisionExtractor(
            backend=backend,
            schema=self.schema,
            user_prompt=self.user_prompt,
            system_prompt=self.system_prompt,
        )

    def run(self, image_path: str, backend: LLMBackend) -> BaseModel:
        return self.build(backend).extract(image_path)


# ============================================================
# Prompt tuning per jenis dokumen.
# Struktur output tetap bebas (model yang memutuskan), prompt hanya
# mengarahkan APA yang perlu diperhatikan untuk jenis tersebut.
# ============================================================

_PROMPT_RECEIPT = """Analisis gambar STRUK BELANJA / NOTA ini secara menyeluruh.

Fokus utamamu: nama toko/merchant, nomor struk, tanggal, kasir, daftar
item belanja (nama + harga per item), subtotal, pajak, total bayar,
metode pembayaran, uang yang dibayar, dan kembalian.

Aturan lain tetap berlaku: pilih sendiri struktur JSON yang paling rapi dan
akurat (key-value, objek bersarang, atau array item), nama field deskriptif
dalam bahasa isi dokumen, angka desimal gaya Eropa ("3,75") ditulis dengan
titik (3.75), jangan menebak informasi yang tidak terlihat.

Output HANYA JSON:
{{"doc_type": "receipt", "data": {<struktur bebas>}}}
"""

_PROMPT_INVOICE = """Analisis gambar FAKTUR / INVOICE ini secara menyeluruh.

Fokus utamamu: nomor invoice, tanggal terbit, tanggal jatuh tempo, data
penjual dan pembeli (nama + alamat), daftar item/jasa (deskripsi, jumlah,
harga satuan, total), diskon, pajak/PPN, total tagihan, dan status
pembayaran bila ada.

Aturan lain tetap berlaku: pilih sendiri struktur JSON yang paling rapi dan
akurat, nama field deskriptif dalam bahasa isi dokumen, angka desimal gaya
Eropa ("3,75") ditulis dengan titik (3.75), jangan menebak informasi.

Output HANYA JSON:
{{"doc_type": "invoice", "data": {<struktur bebas>}}}
"""

_PROMPT_TABLE = """Analisis gambar TABEL DATA ini secara menyeluruh.

Fokus utamamu: representasikan seluruh isi tabel dengan akurat - header
kolom dan setiap baris datanya. Pertahankan angka dan teks persis seperti
tertulis; jangan memotong baris/kolom.

Struktur yang disarankan (pilih sendiri yang paling sesuai):
{{"columns": ["..."], "rows": [["...", "..."]]}}

Output HANYA JSON:
{{"doc_type": "table", "data": {<struktur tabel>}}}
"""

_PROMPT_FORM = """Analisis gambar FORMULIR ISIAN ini secara menyeluruh.

Fokus utamamu: setiap field/label pada formulir beserta nilainya (jika
terisi), termasuk kotak centang/checkbox dan pilihan yang ditandai.
Representasikan sebagai pasangan key-value yang deskriptif, atau objek
bersarang jika ada bagian-bagian formulir.

Aturan lain tetap berlaku: pilih sendiri struktur JSON paling tepat, nama
field deskriptif dalam bahasa isi dokumen, jangan menebak informasi yang
tidak terlihat.

Output HANYA JSON:
{{"doc_type": "form", "data": {<struktur bebas>}}}
"""

_PROMPT_BUSINESS_CARD = """Analisis gambar KARTU NAMA ini secara menyeluruh.

Fokus utamamu: nama orang, jabatan/posisi, nama perusahaan, alamat,
nomor telepon, email, website, dan informasi kontak lain yang tertera.

Aturan lain tetap berlaku: pilih sendiri struktur JSON yang paling rapi
(mis. key-value dengan nama field deskriptif), nama field dalam bahasa isi
dokumen, jangan menebak informasi yang tidak terlihat.

Output HANYA JSON:
{{"doc_type": "business_card", "data": {<struktur bebas>}}}
"""

_PROMPT_BANK_STATEMENT = """Analisis gambar LEMBARAN MUTASI REKENING BANK ini secara menyeluruh.

Fokus utamamu: nama pemilik rekening, nomor rekening, periode laporan,
daftar transaksi (tanggal, keterangan, debet, kredit, saldo), dan saldo
akhir. Jangan menebak transaksi yang tidak terbaca jelas.

Aturan lain tetap berlaku: pilih sendiri struktur JSON yang paling rapi,
nama field deskriptif, angka desimal gaya Eropa ditulis dengan titik.

Output HANYA JSON:
{{"doc_type": "bank_statement", "data": {<struktur bebas>}}}
"""

_PROMPT_LABEL = """Analisis gambar LABEL PRODUK / KEMASAN ini secara menyeluruh.

Fokus utamamu: nama produk, merek, varian, komposisi/bahan, tanggal
kadaluarsa, kode barcode/EAN, berat bersih, harga, dan informasi lain
yang tertera pada label.

Aturan lain tetap berlaku: pilih sendiri struktur JSON yang paling rapi,
nama field deskriptif dalam bahasa isi dokumen, jangan menebak informasi.

Output HANYA JSON:
{{"doc_type": "label", "data": {<struktur bebas>}}}
"""

_PROMPT_SCREENSHOT = """Analisis gambar TANGKAPAN LAYAR (screenshot aplikasi/web/chat) ini secara menyeluruh.

Fokus utamamu: konteks tampilan (aplikasi/web/chat apa), judul atau pesan
utama, elemen yang ditampilkan (tombol, menu, daftar, angka), dan semua
informasi tekstual yang penting pada layar.

Aturan lain tetap berlaku: pilih sendiri struktur JSON yang paling rapi
(mis. key-value + daftar), nama field deskriptif, jangan menebak isi yang
tidak terlihat.

Output HANYA JSON:
{{"doc_type": "screenshot", "data": {<struktur bebas>}}}
"""


# ============================================================
# Registry: nama jenis -> ExtractionAgent
# ============================================================

AGENT_REGISTRY: Dict[str, ExtractionAgent] = {
    "receipt": ExtractionAgent(
        name="receipt",
        description="Struk belanja / nota kasir (daftar item + total bayar)",
        user_prompt=_PROMPT_RECEIPT,
    ),
    "invoice": ExtractionAgent(
        name="invoice",
        description="Faktur / invoice (tagihan penjualan, item, pajak)",
        user_prompt=_PROMPT_INVOICE,
    ),
    "table": ExtractionAgent(
        name="table",
        description="Tabel data murni (baris dan kolom)",
        user_prompt=_PROMPT_TABLE,
    ),
    "form": ExtractionAgent(
        name="form",
        description="Formulir isian (field + nilai, checkbox)",
        user_prompt=_PROMPT_FORM,
    ),
    "business_card": ExtractionAgent(
        name="business_card",
        description="Kartu nama (nama, jabatan, perusahaan, kontak)",
        user_prompt=_PROMPT_BUSINESS_CARD,
    ),
    "bank_statement": ExtractionAgent(
        name="bank_statement",
        description="Lembaran mutasi rekening bank (daftar transaksi)",
        user_prompt=_PROMPT_BANK_STATEMENT,
    ),
    "label": ExtractionAgent(
        name="label",
        description="Label produk / kemasan (nama, bahan, kadaluarsa, barcode)",
        user_prompt=_PROMPT_LABEL,
    ),
    "screenshot": ExtractionAgent(
        name="screenshot",
        description="Tangkapan layar aplikasi/web/chat",
        user_prompt=_PROMPT_SCREENSHOT,
    ),
    # Fallback: jenis tak dikenal -> prompt general (model bebas menentukan).
    "generic": ExtractionAgent(
        name="generic",
        description="Dokumen lain / tidak termasuk kategori di atas",
        user_prompt=GENERAL_USER_PROMPT,
    ),
}


def get_agent(name: str) -> ExtractionAgent:
    """Ambil agent berdasarkan nama jenis. Tak dikenal -> agent generic."""
    return AGENT_REGISTRY.get(name, AGENT_REGISTRY["generic"])
