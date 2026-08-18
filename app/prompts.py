"""
Template prompt untuk ekstraksi & klasifikasi dokumen.

Desain: GENERAL untuk ekstraksi (satu prompt memberi kebebasan penuh ke VLM),
ditambah prompt tuning per jenis dokumen untuk mengarahkan fokus ekstraksi.
Struktur output tetap diputuskan model, tidak dipaksa oleh prompt.
"""

# --- Peran model (system prompt) ekstraksi -----------------------------------
SYSTEM_EXTRACTOR = (
    "Kamu adalah asisten ekstraksi dokumen yang sangat teliti dan terstruktur. "
    "Kamu menganalisis gambar dokumen, memahami isinya secara menyeluruh, dan "
    "mengembalikan satu objek JSON yang merepresentasikan informasi tersebut "
    "dengan struktur yang paling tepat. Kamu yang memutuskan struktur JSON-nya. "
    "Jangan menambahkan penjelasan di luar JSON, jangan menerjemahkan isi dokumen, "
    "dan jangan menebak informasi yang tidak ada di gambar."
)

# Satu-satunya user prompt general - berlaku untuk SEMUA jenis dokumen.
GENERAL_USER_PROMPT = """Analisis gambar dokumen berikut secara menyeluruh dan ubah menjadi satu objek JSON.

Aturan:
1. Pahami jenis dokumen yang ada di gambar (struk belanja, tabel, formulir,
   kartu nama, kuitansi, label produk, dan lain-lain).
2. Kamu yang memutuskan struktur representasi yang paling tepat, rapi, dan
   akurat. Bebas memilih dan mengombinasikan pola berikut:
   - pasangan key-value untuk atribut tunggal;
   - objek bersarang untuk entitas (mis. alamat, penjual, pembeli);
   - array objek untuk daftar item/baris yang berulang;
   - {"columns": [...], "rows": [[...]]} untuk tabel;
   - atau pola lain yang menurutmu paling sesuai.
3. Beri nama field yang deskriptif, ringkas, dan konsisten, dalam bahasa isi
   dokumen. Jangan menerjemahkan isi dokumen.
4. Nilai ditulis persis seperti yang terlihat di gambar. Untuk angka yang
   memakai koma desimal gaya Eropa (mis. tertulis "3,75"), gunakan titik (3.75).
5. Jangan menebak atau menambahkan informasi yang tidak terlihat di gambar.

Output HANYA satu objek JSON (tanpa markdown, tanpa teks lain) dengan kerangka:
{"doc_type": "<jenis dokumen>", "data": {<struktur bebas yang kamu pilih>}}
"""

# --- Klasifikasi jenis dokumen -----------------------------------------------
CLASSIFY_SYSTEM = (
    "Kamu adalah pengklasifikasi jenis dokumen yang teliti. Kamu melihat "
    "sebuah gambar dokumen, memutuskan jenisnya, dan mengembalikan satu "
    "objek JSON berisi nama jenis tersebut. Tidak ada teks lain."
)

CLASSIFY_PROMPT = """Analisis gambar dokumen berikut dan klasifikasikan jenisnya.

Pilih SATU jenis dari daftar berikut - gunakan persis nama yang tertera:
{types}

Output HANYA JSON:
{{"doc_type": "<nama jenis>"}}
"""

# --- Prompt tuning per jenis dokumen -----------------------------------------
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


def build_extraction_user_prompt(
    base_prompt: str,
    ocr_text: str | None = None,
    critique: str | None = None,
) -> str:
    """
    Gabungkan base prompt jenis dokumen dengan OCR auxiliary context dan
    catatan perbaikan refleksi bila tersedia.
    """
    parts = [base_prompt.strip()]

    if ocr_text and ocr_text.strip():
        parts.append(
            "\n--- AUXILIARY OCR TEXT (REFERENSI TEKS TAMBAHAN) ---\n"
            "Teks mentah berikut diekstrak dari gambar menggunakan model OCR beresolusi tinggi. "
            "Gunakan teks ini untuk membantu memastikan ejaan kata, angka nominal, atau kode yang kecil/pudar, "
            "tetapi gambar dokumen tetaplah sumber kebenaran visual utama:\n"
            f"```\n{ocr_text.strip()}\n```"
        )

    if critique and critique.strip():
        parts.append(
            "\n--- CATATAN PERBAIKAN REFLEKSI (SELF-REFLECTION FEEDBACK) ---\n"
            f"{critique.strip()}"
        )

    return "\n\n".join(parts)

