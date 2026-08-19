"""
Template prompt untuk Ekstraksi Dokumen Vision OCR -> Markdown Siap Chunking.

Mencakup 4 spesifikasi kemampuan ekstraksi dokumen:
  1. Dokumen Standar / Biasa (plain text & tables)
  2. Kontinuitas Hierarki Markdown (#, ##, ###)
  3. Dokumen 2 Kolom & 2 Bahasa (Jurnal Ilmiah, column-aware reading order)
  4. Dokumen Presentasi / PPT Slides (Slide headers, bullet points, deskripsi diagram/visual)
"""
from __future__ import annotations

# --- System Prompt Utama -----------------------------------------------------
SYSTEM_DOCUMENT_EXTRACTOR = (
    "Kamu adalah AI ahli ekstraksi dokumen multimodal berpresisi tinggi. "
    "Tugasmu adalah menganalisis gambar dokumen dan mengubah seluruh isinya "
    "menjadi format MARKDOWN BERSIH yang terstruktur dan siap langsung di-chunking "
    "oleh text splitter.\n\n"
    "Aturan Umum:\n"
    "1. Pertahankan ejaan, angka, istilah teknis, dan bahasa asli persis seperti di dokumen.\n"
    "2. Gunakan sintaks Markdown standar (heading #, ##, ###; bullet list -, 1.; tabel GFM | Col | Col |; bold **text**).\n"
    "3. Jangan menambahkan penjelasan pengantar ('Berikut adalah hasil ekstraksi...') atau penutup. Outputkan HANYA konten dokumen dalam Markdown.\n"
    "4. Jika ada tabel, konversi ke format Markdown Table yang valid.\n"
    "5. Jika ada teks OCR tambahan, gunakan sebagai referensi untuk memastikan akurasi ejaan/angka, namun gambar tetap acuan visual utama."
)

# --- 1. Prompt Dokumen Biasa / Standar ---------------------------------------
_PROMPT_PLAIN_DOCUMENT = """Ekstrak seluruh teks dan isi dari gambar dokumen ini menjadi Markdown yang rapi dan terstruktur.

Panduan:
- Susun teks dalam paragraf yang rapi dan logis.
- Jika ada judul atau subjudul, gunakan heading Markdown (`##`, `###`).
- Jika ada daftar item, gunakan bullet points (`- `) atau penomoran (`1. `).
- Jika ada formulir atau data atribut, gunakan format key-value tebal (mis. `**Nama:** Budi`).
- Jika ada tabel, ubah menjadi format tabel Markdown.

Outputkan HANYA teks Markdown dokumen.
"""

# --- 2. Prompt Kontinuitas Hierarki Markdown ---------------------------------
_PROMPT_MARKDOWN_HIERARCHY = """Ekstrak dokumen ini dengan fokus utama menjaga HIERARKI HEADING MARKDOWN (#, ##, ###) yang runtut dan ketat.

Panduan:
- Identifikasi judul utama dokumen sebagai `# Judul Dokumen`.
- Identifikasi bab / bagian utama sebagai `## Bagian Utama`.
- Identifikasi sub-bab sebagai `### Sub-bagian` dan `#### Rincian`.
- Pastikan urutan dan kedalaman heading tidak melompat sembarangan (mis. jangan langsung ### tanpa ada ##).
- Pertahankan daftar berpoin bertingkat dengan indentasi spasi yang konsisten.
- Jika halaman ini merupakan kelanjutan dari bagian sebelumnya (tanpa judul baru), lanjutkan isi kontennya langsung dengan paragraf/poin yang sesuai.

Outputkan HANYA teks Markdown dengan hierarki heading yang presisi.
"""

# --- 3. Prompt Jurnal 2 Kolom & 2 Bahasa -------------------------------------
_PROMPT_BILINGUAL_JOURNAL = """Ekstrak dokumen ilmiah / artikel ini dengan membaca urutan KOLOM SECARA BENAR (Column-Aware Reading Order) dan menjaga struktur DUA BAHASA (Bilingual).

Panduan Khusus Jurnal 2 Kolom & Bilingual:
1. URUTAN BACA 2 KOLOM:
   - Baca dan tuntaskan seluruh teks di KOLOM KIRI dari atas sampai bawah terlebih dahulu.
   - Setelah kolom kiri selesai, baru lanjutkan membaca KOLOM KANAN dari atas sampai bawah.
   - JANGAN membaca melintang horizontal memotong antar kolom!
2. FORMAT DOKUMEN ILMIAH:
   - Judul artikel, nama penulis, dan afiliasi diletakkan di bagian atas.
   - Bagian Abstrak / Abstract (seringkali 2 bahasa, mis. Bahasa Indonesia & English) dipisahkan dengan heading yang jelas (`### Abstrak`, `### Abstract`).
   - Bagian utama (Pendahuluan, Metode, Hasil, Pembahasan, Kesimpulan, Referensi) disusun dengan heading `##`.
3. RUMUS & TABEL:
   - Rumus matematika ditulis jelas (format LaTeX/teks bersih).
   - Tabel dan grafik 2 kolom diposisikan secara logis sesuai alur teks.

Outputkan HANYA teks Markdown jurnal yang runtut sesuai urutan baca kolom yang benar.
"""

# --- 4. Prompt Slide Presentasi / PPT ---------------------------------------
_PROMPT_PRESENTATION_SLIDES = """Ekstrak slide presentasi ini menjadi Markdown terstruktur yang optimal untuk representasi slide.

Panduan Slide:
1. Awali dengan judul slide: `## Slide: [Judul Slide]` (atau `## [Judul Slide]`).
2. Sajikan pesan utama dan poin-poin presentasi menggunakan bullet points berjenjang (`- Poin utama`, `  - Sub-poin penjelasan`).
3. Jika terdapat visual, diagram alur, grafik angka, atau bagan:
   - Deskripsikan informasi atau relasi bagan tersebut dalam blockquote: `> **[Diagram/Grafik]:** [Penjelasan isi bagan, relasi panah, dan angka utama]`.
4. Jika ada tabel atau perbandingan metrik, representasikan dalam tabel Markdown.

Outputkan HANYA teks Markdown slide presentasi.
"""

# --- Klasifikasi Karakteristik Dokumen --------------------------------------
CLASSIFY_SYSTEM = (
    "Kamu adalah pengklasifikasi layout dokumen. Tugasmu menentukan karakteristik "
    "tata letak dokumen untuk memilih strategi ekstraksi Markdown yang paling tepat."
)

CLASSIFY_PROMPT = """Analisis layout gambar dokumen berikut dan tentukan satu karakteristik utama dari pilihan berikut:

- `plain`: Dokumen umum, surat, memo, formulir, atau teks standar biasa.
- `markdown_hierarchy`: Dokumen berstruktur resmi, laporan teknis, buku, atau SOP dengan hierarki bab (#, ##, ###) yang ketat.
- `bilingual_journal`: Jurnal ilmiah, makalah akademik, artikel 2 kolom, atau dokumen hukum 2 bahasa berdampingan.
- `presentation_slides`: Slide presentasi (PowerPoint/Keynote/PDF landscape) berisi poin-poin dan diagram/visual.

Pilih SATU nama di atas. Outputkan HANYA JSON:
{{"doc_type": "<plain | markdown_hierarchy | bilingual_journal | presentation_slides>"}}
"""


def build_extraction_prompt(
    doc_type: str = "plain",
    ocr_text: str | None = None,
    previous_page_context: str | None = None,
) -> str:
    """
    Bangun user prompt ekstraksi berdasarkan karakteristik dokumen, OCR teks tambahan,
    dan konteks heading dari halaman sebelumnya bila multi-halaman.
    """
    doc_lower = (doc_type or "plain").lower()

    if "journal" in doc_lower or "bilingual" in doc_lower or "2col" in doc_lower:
        base_prompt = _PROMPT_BILINGUAL_JOURNAL
    elif "hierarchy" in doc_lower or "markdown" in doc_lower or "report" in doc_lower:
        base_prompt = _PROMPT_MARKDOWN_HIERARCHY
    elif "slide" in doc_lower or "ppt" in doc_lower or "presentation" in doc_lower:
        base_prompt = _PROMPT_PRESENTATION_SLIDES
    else:
        base_prompt = _PROMPT_PLAIN_DOCUMENT

    parts = [base_prompt.strip()]

    if previous_page_context and previous_page_context.strip():
        parts.append(
            "\n--- KONTEKS HALAMAN SEBELUMNYA (KONTINUITAS HEADING) ---\n"
            f"Dokumen ini merupakan halaman lanjutan. Konteks akhir halaman sebelumnya adalah:\n"
            f"```markdown\n{previous_page_context.strip()[-500:]}\n```\n"
            "Pastikan level heading (#, ##, ###) dan kelanjutan kalimat pada halaman ini menyambung secara selaras."
        )

    if ocr_text and ocr_text.strip():
        parts.append(
            "\n--- AUXILIARY OCR TEXT (REFERENSI TEKS TAMBAHAN) ---\n"
            "Teks mentah berikut diekstrak dari gambar menggunakan model OCR beresolusi tinggi. "
            "Gunakan untuk membantu memvalidasi ejaan, istilah teknis, atau angka kecil:\n"
            f"```\n{ocr_text.strip()}\n```"
        )

    return "\n\n".join(parts)
