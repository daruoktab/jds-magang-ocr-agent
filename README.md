# jds-magang — Vision OCR & Document Extractor (Ready for Chunking)

Sistem ekstraksi dokumen multimodal (PDF, PPT/PPTX, Scan Gambar) menjadi **Markdown bersih dan terstruktur yang siap langsung di-chunking** untuk pipeline RAG downstream.

> ℹ️ **Catatan Branch:** 
> Fitur lengkap pipeline RAG end-to-end (Embedding `Qwen3-VL-Embedding`, Reranker `Qwen3-VL-Reranker`, dan Vector Store) tersimpan di branch `end-to-end`. Branch `main` difokuskan pada pipeline ekstraksi dokumen ke format Markdown siap chunking.

---

## 🎯 4 Spesifikasi Karakteristik Dokumen

| No | Spesifikasi Layout | Karakteristik & Perilaku Ekstraksi | Target Output |
|:---:|:---|:---|:---|
| **1** | `plain` | **Dokumen Biasa / Standar**<br>Dokumen umum (memo, formulir, surat, nota) yang tidak memerlukan perlakuan hierarki khusus. OCR langsung diekstrak menjadi teks/markdown bersih. | Paragraf rapi, tabel standar GFM |
| **2** | `markdown_hierarchy` | **Hierarki Markdown Berkelanjutan**<br>Dokumen bertingkat (`#`, `##`, `###`). Menjaga konsistensi judul ketika berpindah halaman (multi-page) dan menyambungkan kalimat terpotong tanpa merusak struktur. | Hierarki heading utuh, eliminasi page header/footer berulang |
| **3** | `bilingual_journal` | **Jurnal Ilmiah / Dokumen 2-Kolom & 2-Bahasa**<br>Membaca kolom kiri dari atas ke bawah sampai selesai, lalu melanjutkan ke kolom kanan. Menjaga koherensi teks bilingual berdampingan. | Urutan baca logis berurutan (tidak melompat antar-kolom) |
| **4** | `presentation_slides` | **Slide Presentasi (PPT / PPTX / Slide PDF)**<br>Slide yang sarat poin-poin/bullet list, tabel ringkas, speaker notes, serta penanda visual/diagram `[Diagram: ...]`. | Markdown per-slide yang siap dipartisi per topik |

---

## 🏗️ Alur Pipeline Ekstraksi

```mermaid
flowchart TD
    DOC["Input: PDF / PPTX / Gambar Scan"] --> ROUTE{"Jenis File?"}
    
    ROUTE -- "PPTX / PPT" --> PPT["app/ppt.py: Struktur Slide, Bullets, Tables, Notes"]
    ROUTE -- "PDF / Image" --> PRE["app/preprocess.py: Auto-rotate EXIF & Contrast"]
    
    PRE --> OCR["app/ocr.py: ocr-lighton Auxiliary Raw Text"]
    PRE --> CLS["app/extractor.py: Layout Classifier (4 Spesifikasi)"]
    
    CLS --> VLM["app/extractor.py: VLM Markdown Extraction dengan Prompt Spesialisasi"]
    OCR --> VLM
    
    VLM --> MULTI["app/multi_page.py: Header Stitching & Page Boundary Cleanup"]
    PPT --> CHUNK
    MULTI --> CHUNK["app/multi_page.py: Markdown Chunking Splitter Ready"]
    
    CHUNK --> OUT["Output: Markdown Siap Chunking / Chunk Preview"]
```

---

## 📁 Struktur Modul

```
jds-magang/
├── app/
│   ├── config.py          # Pengaturan model VLM, OCR, dan API Key
│   ├── preprocess.py      # Auto-orientasi EXIF & optimasi kontras dokumen
│   ├── llm.py             # Builder ChatOpenAI & helper base64 image data URI
│   ├── ocr.py             # Model OCR tuned (ocr-lighton) untuk pembacaan teks mentah
│   ├── prompts.py         # Prompt spesialisasi 4 karakteristik dokumen
│   ├── schemas.py         # Pydantic schemas (ExtractedDocument, DocumentPage, ChunkItem)
│   ├── extractor.py       # Ekstraksi multimodal VLM -> Markdown
│   ├── agents.py          # Registry Agent untuk 4 spesifikasi tata letak
│   ├── ppt.py             # Parser presentasi PowerPoint (.pptx)
│   ├── pdf.py             # Multi-page PDF renderer & stitcher
│   ├── multi_page.py      # Penyambung halaman kontinu & simulasi chunking
│   ├── graph.py           # Pipeline LangGraph DocumentExtractionPipeline
│   └── deep_agent.py      # Harness Deep Agents dengan subagents spesialis
├── main.py                # Antarmuka CLI utama
├── pyproject.toml         # Konfigurasi dependensi
└── README.md
```

---

## ⚙️ Instalasi & Persiapan

Gunakan `uv` untuk instalasi paket:

```powershell
# 1. Buat virtual environment
uv venv --prompt magang-jds .venv

# 2. Aktifkan venv & install dependensi
.venv\Scripts\Activate.ps1
uv pip install -U -r pyproject.toml
```

Buat file `.env` (salin dari template bila perlu):
```env
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.together.xyz/v1
VLM_MODEL=qwen-35b-vision
OCR_MODEL=ocr-lighton
```

---

## 🚀 Panduan Penggunaan CLI

### 1. Ekstraksi Dokumen ke Teks Markdown

```powershell
# Ekstrak PDF multi-halaman
python main.py dokumen.pdf

# Ekstrak file presentasi PowerPoint
python main.py presentasi.pptx

# Ekstrak gambar hasil scan / foto dokumen
python main.py scan_dokumen.jpg
```

### 2. Menyimpan Output ke File Markdown (`-o` / `--out`)

```powershell
python main.py laporan_tahunan.pdf -o output/laporan.md
```

### 3. Simulasi & Preview Chunking LangChain (`--preview-chunks`)

Menampilkan bagaimana teks Markdown yang diekstrak akan dipecah oleh `MarkdownHeaderTextSplitter` dan `RecursiveCharacterTextSplitter`:

```powershell
python main.py dokumen.pdf --preview-chunks --chunk-size 1000 --chunk-overlap 150
```

### 4. Memilih Spesifikasi Tata Letak Tertentu (`-t` / `--type`)

Secara default, pipeline akan melakukan klasifikasi layout secara otomatis. Anda juga dapat memaksa spesifikasi tertentu:

```powershell
# Mode 1: Dokumen biasa tanpa hierarki rumit
python main.py formulir.pdf --type plain

# Mode 2: Dokumen ber-heading penting dengan kontinuitas antar halaman
python main.py buku_panduan.pdf --type markdown_hierarchy

# Mode 3: Jurnal ilmiah 2-kolom & 2-bahasa (urutan baca kolom kiri lalu kanan)
python main.py jurnal_penelitian.pdf --type bilingual_journal

# Mode 4: Slide presentasi
python main.py presentasi.pdf --type presentation_slides
```

### 5. Melihat Daftar Spesifikasi yang Didukung

```powershell
python main.py --list-types
```

---

## 🧪 Pengujian & Verifikasi

Untuk menjalankan script pengujian unit test dan verifikasi chunking:

```powershell
.venv\Scripts\Activate.ps1 ; python test_extraction_specs.py
```
