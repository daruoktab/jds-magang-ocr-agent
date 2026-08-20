"""
Template prompt modular untuk Ekstraksi Dokumen Vision OCR -> Markdown Siap Chunking.

Mendukung penyusunan prompt modular (Composable Prompting) untuk dokumen yang
memiliki lebih dari 1 spesifikasi layout secara bersamaan:
  1. `plain`                : Dokumen biasa / standar
  2. `markdown_hierarchy`   : Kontinuitas hierarki heading (#, ##, ###)
  3. `bilingual_journal`    : Jurnal ilmiah 2-kolom & 2-bahasa (column-aware reading order)
  4. `presentation_slides`  : Slide presentasi (bullet points, deskripsi diagram/visual)
"""

from __future__ import annotations

import re

# --- System Prompt Utama -----------------------------------------------------
SYSTEM_DOCUMENT_EXTRACTOR: str = (
    "Kamu adalah AI ahli ekstraksi dokumen multimodal berpresisi tinggi. "
    "Tugasmu adalah menganalisis gambar dokumen dan mengubah seluruh isinya "
    "menjadi format MARKDOWN BERSIH yang terstruktur dan siap langsung di-chunking "
    "oleh text splitter.\n\n"
    "Aturan Standar:\n"
    "1. Pertahankan ejaan, angka, istilah teknis, rumus, dan bahasa asli persis seperti di dokumen.\n"
    "2. Gunakan sintaks Markdown standar (heading #, ##, ###; bullet list -, 1.; tabel GFM | Col |; bold **text**).\n"
    "3. Jangan menambahkan penjelasan pengantar ('Berikut adalah hasil ekstraksi...') atau penutup. Outputkan HANYA konten dokumen dalam Markdown.\n"
    "4. Jika ada tabel, konversi ke format Markdown Table (GFM) yang valid.\n"
    "5. Jika ada teks OCR tambahan, gunakan sebagai referensi untuk memastikan akurasi ejaan/angka, namun gambar tetap acuan visual utama."
)

# --- Modul Aturan Komposisional (Composable Rule Modules) --------------------

_RULE_BASE: str = """Ekstrak seluruh teks dan isi dari gambar dokumen ini menjadi Markdown yang bersih, rapi, dan terstruktur."""

_RULE_COLUMN_AWARE: str = """### PANDUAN TATA LETAK 2-KOLOM & BILINGUAL (Column-Aware Reading Order):
- URUTAN BACA KOLOM: Baca dan tuntaskan seluruh teks di KOLOM KIRI dari atas sampai bawah terlebih dahulu. Setelah kolom kiri selesai, baru lanjutkan membaca KOLOM KANAN dari atas sampai bawah. JANGAN membaca melintang horizontal memotong antar kolom!
- STRUKTUR BILINGUAL: Judul, abstrak (Bahasa Indonesia & English), dan teks dwibahasa harus dipisahkan dengan heading yang jelas (mis. `### Abstrak`, `### Abstract`).
- Rumus matematika dan sitasi dalam kolom ditulis dengan jelas dan rapi."""

_RULE_MARKDOWN_HIERARCHY: str = """### PANDUAN HIERARKI HEADING MARKDOWN (#, ##, ###):
- Identifikasi judul utama sebagai `# Judul Dokumen`.
- Identifikasi bab/bagian utama sebagai `## Bagian Utama`.
- Identifikasi sub-bab sebagai `### Sub-bagian` dan `#### Rincian`.
- Pastikan urutan dan kedalaman heading runtut dan tidak melompat sembarangan (mis. jangan langsung ### tanpa ada ##).
- Pertahankan daftar berpoin bertingkat dengan indentasi spasi yang konsisten.
- Jika dokumen ini merupakan kelanjutan dari bagian sebelumnya (tanpa judul baru), lanjutkan isi kontennya langsung dengan paragraf/poin yang sesuai."""

_RULE_PRESENTATION_SLIDES: str = """### PANDUAN SLIDE PRESENTASI & ELEMEN VISUAL:
- Untuk setiap slide, awali dengan penanda slide standar sistem: `<!-- SLIDE: <nomor_slide> -->` diikuti judul slide: `## [Judul Slide]`.
- Sajikan poin-poin presentasi menggunakan bullet points berjenjang (`- Poin utama`, `  - Sub-poin penjelasan`).
- Jika terdapat visual, diagram alur, grafik angka, atau bagan: Deskripsikan informasi atau relasi bagan tersebut dalam blockquote: `> **[Diagram/Visual]:** [Penjelasan isi bagan, relasi panah, dan angka utama]`.
- Konversi tabel ringkas atau perbandingan metrik ke tabel Markdown."""

_RULE_PLAIN_DOCUMENT: str = """### PANDUAN DOKUMEN STANDAR:
- Susun teks dalam paragraf yang rapi dan logis.
- Jika ada formulir atau data atribut, gunakan format key-value tebal (mis. `**Nama:** Budi`).
- Jika ada tabel data, ubah menjadi format tabel Markdown."""

# --- Klasifikasi Karakteristik Dokumen --------------------------------------
CLASSIFY_SYSTEM: str = (
    "Kamu adalah pengklasifikasi layout dokumen tingkat lanjut. Tugasmu menganalisis "
    "gambar dokumen dan mendeteksi SEMUA karakteristik spesifikasi tata letak yang relevan "
    "(bisa lebih dari satu / multi-label)."
)

CLASSIFY_PROMPT: str = """Analisis layout gambar dokumen berikut dan tentukan SEMUA karakteristik spesifikasi yang ada:

Pilihan spesifikasi:
- `plain`: Dokumen umum, surat, memo, formulir, atau teks standar biasa.
- `markdown_hierarchy`: Dokumen berstruktur resmi, laporan, buku, atau SOP dengan penomoran bab (#, ##, ###) yang bertingkat.
- `bilingual_journal`: Jurnal ilmiah, makalah akademik, format 2 kolom, atau dokumen hukum 2 bahasa berdampingan.
- `presentation_slides`: Slide presentasi (PowerPoint/PDF landscape), sarat bullet points, atau memiliki diagram/visual bagan.

Sebuah dokumen DAPAT memiliki lebih dari 1 spesifikasi (misal jurnal ilmiah multi-halaman memiliki `bilingual_journal` DAN `markdown_hierarchy`).

Outputkan HANYA format JSON list:
{{"specs": ["<spec1>", "<spec2>"]}}
"""


def normalize_specs(specs: list[str] | str | None) -> list[str]:
    """
    Normalisasi input spesifikasi menjadi list of valid spec keys.
    Mendukung input berupa list, string tunggal, atau string koma ('journal,hierarchy').
    """
    if not specs:
        return ["plain"]

    raw_items: list[str] = []
    if isinstance(specs, str):
        # Pisahkan jika ada koma atau spasi
        raw_items = [s.strip() for s in re.split(r"[,|+]", specs) if s.strip()]
    elif isinstance(specs, (list, tuple, set)):
        for item in specs:
            if isinstance(item, str):
                raw_items.extend(
                    [s.strip() for s in re.split(r"[,|+]", item) if s.strip()]
                )

    matched_specs: list[str] = []
    for item in raw_items:
        key = item.lower()
        if (
            "bilingual" in key or "journal" in key or "academic" in key
        ) and "bilingual_journal" not in matched_specs:
            matched_specs.append("bilingual_journal")
        elif (
            "hierarchy" in key or "markdown" in key or "report" in key
        ) and "markdown_hierarchy" not in matched_specs:
            matched_specs.append("markdown_hierarchy")
        elif (
            "slide" in key or "ppt" in key or "presentation" in key
        ) and "presentation_slides" not in matched_specs:
            matched_specs.append("presentation_slides")
        elif (
            "plain" in key or "generic" in key or "standard" in key
        ) and "plain" not in matched_specs:
            matched_specs.append("plain")

    return matched_specs or ["plain"]


def build_extraction_prompt(
    specs: list[str] | str | None = None,
    ocr_text: str | None = None,
    previous_page_context: str | None = None,
) -> str:
    """
    Bangun prompt ekstraksi komposit modular yang menggabungkan seluruh aturan spesifikasi aktif.
    """
    active_specs = normalize_specs(specs)
    prompt_blocks: list[str] = [_RULE_BASE]

    # Gabungkan modul-modul aturan yang aktif
    if "bilingual_journal" in active_specs:
        prompt_blocks.append(_RULE_COLUMN_AWARE)

    if "markdown_hierarchy" in active_specs:
        prompt_blocks.append(_RULE_MARKDOWN_HIERARCHY)

    if "presentation_slides" in active_specs:
        prompt_blocks.append(_RULE_PRESENTATION_SLIDES)

    if "plain" in active_specs and len(active_specs) == 1:
        prompt_blocks.append(_RULE_PLAIN_DOCUMENT)

    # Sisipkan konteks heading halaman sebelumnya jika ada
    if previous_page_context and previous_page_context.strip():
        prompt_blocks.append(
            "### KONTEKS HALAMAN SEBELUMNYA (KONTINUITAS HEADING):\n"
            "Dokumen ini merupakan halaman lanjutan. Konteks akhir halaman sebelumnya adalah:\n"
            f"```markdown\n{previous_page_context.strip()[-500:]}\n```\n"
            "Pastikan level heading (#, ##, ###) dan kelanjutan kalimat pada halaman ini menyambung secara selaras dengan konteks di atas."
        )

    # Sisipkan teks mentah OCR jika ada
    if ocr_text and ocr_text.strip():
        prompt_blocks.append(
            "### AUXILIARY OCR TEXT (REFERENSI TEKS TAMBAHAN):\n"
            "Teks mentah berikut diekstrak dari gambar menggunakan model OCR beresolusi tinggi. "
            "Gunakan untuk membantu memverifikasi ejaan, istilah teknis, simbol, atau angka kecil:\n"
            f"```\n{ocr_text.strip()}\n```"
        )

    prompt_blocks.append(
        "Outputkan HANYA konten dokumen dalam format Markdown yang bersih dan terstruktur."
    )
    return "\n\n".join(prompt_blocks)
