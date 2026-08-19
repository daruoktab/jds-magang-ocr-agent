# jds-magang-ocr-agent — Vision OCR & Document Extractor (Ready for Chunking)

Sistem ekstraksi dokumen multimodal (PDF, PPT/PPTX, Scan Gambar) menjadi **Markdown bersih dan terstruktur yang siap langsung di-chunking** untuk pipeline RAG downstream.

> ℹ️ **Catatan Branch:** 
> Fitur lengkap pipeline RAG end-to-end (Embedding `Qwen3-VL-Embedding`, Reranker `Qwen3-VL-Reranker`, dan Vector Store) tersimpan di branch `end-to-end`. Branch `main` difokuskan pada pipeline ekstraksi dokumen ke format Markdown siap chunking.

---

## 🎯 4 Spesifikasi Karakteristik Dokumen (Mendukung Multi-Spesifikasi Komposit)

Sistem mendukung ekstraksi dengan satu atau **beberapa spesifikasi sekaligus secara komposit** (*Composable Prompts*), misalnya jurnal ilmiah multi-halaman yang membutuhkan aturan 2-kolom sekaligus kontinuitas heading antar-halaman (`bilingual_journal` + `markdown_hierarchy`):

| No | Spesifikasi Layout | Karakteristik & Perilaku Ekstraksi | Target Output |
|:---:|:---|:---|:---|
| **1** | `plain` | **Dokumen Biasa / Standar**<br>Dokumen umum (memo, formulir, surat, nota) yang tidak memerlukan perlakuan hierarki khusus. OCR langsung diekstrak menjadi teks/markdown bersih. | Paragraf rapi, tabel standar GFM |
| **2** | `markdown_hierarchy` | **Hierarki Markdown Berkelanjutan**<br>Dokumen bertingkat (`#`, `##`, `###`). Menjaga konsistensi judul ketika berpindah halaman (multi-page) dan menyambungkan kalimat terpotong tanpa merusak struktur. | Hierarki heading utuh, eliminasi page header/footer berulang |
| **3** | `bilingual_journal` | **Jurnal Ilmiah / Dokumen 2-Kolom & 2-Bahasa**<br>Membaca kolom kiri dari atas ke bawah sampai selesai, lalu melanjutkan ke kolom kanan. Menjaga koherensi teks bilingual berdampingan. | Urutan baca logis berurutan (tidak melompat antar-kolom) |
| **4** | `presentation_slides` | **Slide Presentasi (PPT / PPTX / Slide PDF)**<br>Slide yang sarat poin-poin/bullet list, tabel ringkas, speaker notes, serta penanda visual/diagram `[Diagram: ...]`. | Markdown per-slide yang siap dipartisi per topik |

---

## 🤖 Arsitektur Sub-Agent (Deep Agents Harness)

Proyek ini dilengkapi dengan **Master Agent dan 6 Sub-Agent Spesialis** ([app/deep_agent.py](file:///c:/Users/HYPE%20AMD/Documents/Coding/jds-magang/app/deep_agent.py)) yang mendelegasikan tugas secara otonom:

| Nama Sub-Agent | Peran & Spesialisasi | Tool Utama |
|:---|:---|:---|
| `ocr-specialist` | Pembacaan teks mentah literal tingkat tinggi tanpa halusinasi | `ocr_document` (`ocr-lighton`) |
| `layout-classifier` | Deteksi multi-trait tata letak dokumen (kolom, hierarki, slide, tabel) | `classify_layout` |
| `markdown-extractor` | Ekstraksi gambar multimodal ke Markdown bersih berbasis spesifikasi komposit | `extract_to_markdown` |
| `presentation-specialist` | Ekstraksi slide PowerPoint (.pptx) dengan hierarki bullet, tabel, dan notes | `extract_presentation_pptx` |
| `pdf-orchestrator` | Orkestrasi pemrosesan PDF multi-halaman & penyambungan kontinuitas heading | `extract_pdf_document` |
| `chunking-simulator` | Evaluasi kesiapan partisi Markdown dengan header splitter & recursive splitter | `preview_chunks` |

---

## 🔌 Model Context Protocol (MCP) Server

Proyek ini menyediakan server MCP berstandar resmi **MCP Python SDK v2.0** ([app/mcp_server.py](file:///c:/Users/HYPE%20AMD/Documents/Coding/jds-magang/app/mcp_server.py)), mengekspos kapabilitas Vision OCR dan ekstrasi dokumen untuk AI Assistant (seperti Antigravity IDE, Claude Desktop, atau Cursor):

### Daftar MCP Tools yang Disediakan:
1. `extract_document_to_markdown`: Ekstraksi PDF, PPTX, gambar ke Markdown siap chunking.
2. `ocr_image`: Grounding teks mentah presisi tinggi via model OCR.
3. `classify_document_layout`: Deteksi multi-trait spesifikasi tata letak dokumen.
4. `extract_presentation_pptx`: Parser file PowerPoint (.pptx/.ppt) ke Markdown.
5. `preview_markdown_chunks`: Simulasi partisi teks Markdown dengan header metadata.
6. `run_deep_reasoning_agent`: Eksekusi agen penalaran dokumen multimodal otonom.

### Konfigurasi `mcp_config.json`:
```json
{
  "mcpServers": {
    "jds-magang-ocr-agent": {
      "command": "c:\\Users\\HYPE AMD\\Documents\\Coding\\jds-magang\\.venv\\Scripts\\python.exe",
      "args": [
        "-m",
        "app.mcp_server"
      ],
      "cwd": "c:\\Users\\HYPE AMD\\Documents\\Coding\\jds-magang",
      "env": {
        "LLM_BASE_URL": "http://localhost:1234/v1",
        "LLM_API_KEY": "lm-studio",
        "VLM_MODEL": "qwen-35b-vision",
        "OCR_MODEL": "ocr-lighton"
      }
    }
  }
}
```

---

## 🏗️ Alur Pipeline Ekstraksi

```mermaid
flowchart TD
    DOC["Input: PDF / PPTX / Gambar Scan"] --> ROUTE{"Jenis File?"}
    
    ROUTE -- "PPTX / PPT" --> PPT["app/ppt.py: Struktur Slide, Bullets, Tables, Notes"]
    ROUTE -- "PDF / Image" --> PRE["app/preprocess.py: Auto-rotate EXIF & Contrast"]
    
    PRE --> OCR["app/ocr.py: ocr-lighton Auxiliary Raw Text"]
    PRE --> CLS["app/extractor.py: Multi-Spec Classifier"]
    
    CLS --> VLM["app/extractor.py: Composable Prompt VLM Extraction"]
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
│   ├── prompts.py         # Modul prompt modular berlapis (Composable Prompts)
│   ├── schemas.py         # Pydantic schemas (ExtractedDocument, DocumentPage, ChunkItem)
│   ├── extractor.py       # Ekstraksi multimodal VLM -> Markdown (Multi-label)
│   ├── agents.py          # Registry Agent komposit untuk kombinasi spesifikasi
│   ├── ppt.py             # Parser presentasi PowerPoint (.pptx)
│   ├── pdf.py             # Multi-page PDF renderer & stitcher
│   ├── multi_page.py      # Penyambung halaman kontinu & simulasi chunking
│   ├── graph.py           # Pipeline LangGraph DocumentExtractionPipeline
│   ├── deep_agent.py      # Harness Deep Agents dengan 6 subagents spesialis
│   └── mcp_server.py      # Server MCP berstandar SDK v2.0
├── main.py                # Antarmuka CLI utama
├── mcp_server.py          # Entrypoint runner MCP Server
├── mcp_config.template.json # Template konfigurasi MCP JSON
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
LLM_BASE_URL=http://localhost:1234/v1
VLM_MODEL=qwen-35b-vision
OCR_MODEL=ocr-lighton
```

---

## 🚀 Panduan Penggunaan CLI

### 1. Ekstraksi Dokumen ke Teks Markdown (Default: Deep Reasoning Agent)

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

```powershell
python main.py dokumen.pdf --preview-chunks --chunk-size 1000 --chunk-overlap 150
```

### 4. Memilih Spesifikasi Tunggal / Multi-Spesifikasi Komposit (`-t` / `--type`)

```powershell
# Spesifikasi Komposit (Jurnal 2-Kolom + Hierarki Bab Kontinu):
python main.py jurnal_lengkap.pdf --type journal,hierarchy
```
