# jds-magang-ocr-agent — Vision OCR & Document Extractor (Ready for Chunking, Tabular SQLite Database, & Mermaid Diagrams)

Sistem ekstraksi dokumen multimodal (PDF, PPT/PPTX, Scan Gambar) menjadi **Markdown bersih dan terstruktur yang siap langsung di-chunking** untuk pipeline RAG downstream, **Engine Data Tabular Transaksional ke SQLite** untuk data log mutasi/rekening koran/faktur yang memerlukan kalkulasi agregat berpresisi 100% (SUM, AVG, COUNT, filter tanggal), serta **Sub-Agent Spesialis Diagram** untuk mengevaluasi secara selektif dan mengekstrak diagram visual menjadi kode **Mermaid.js** yang valid.

> ℹ️ **Catatan Branch:** 
> Fitur lengkap pipeline RAG end-to-end (Embedding `Qwen3-VL-Embedding`, Reranker `Qwen3-VL-Reranker`, dan Vector Store) tersimpan di branch `end-to-end`. Branch `main` difokuskan pada pipeline ekstraksi dokumen ke format Markdown siap chunking, Tabular SQLite Database, dan Diagram Mermaid.

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

## 🖼️ Rendering Presentasi (PPT/PPTX)

Presentasi di-render menjadi gambar PNG per slide lalu dikirim ke model vision:

```text
PPT/PPTX -> LibreOffice headless -> PDF -> PyMuPDF -> PNG per slide
```

Pipeline memakai **LibreOffice headless** sebagai satu-satunya renderer di semua platform (Windows, Linux, macOS) — tidak ada lagi dependensi Spire. Instalasi:

- **Debian/Ubuntu:** `sudo apt-get install -y libreoffice`
- **Windows:** installer LibreOffice resmi (pipelinenya otomatis mendeteksi lokasi instalasi standar).
- Jika executable tidak berada di `PATH` atau tidak di lokasi standar, set env `LIBREOFFICE_BIN` ke path `soffice`/`libreoffice`.

Batching: rendering diproses bertahap per 10 slide (selaras strategi batch di pipeline PDF) untuk mengontrol memori, progres logging, dan keseragaman unit kerja — bukan karena batasan lisensi.

MCP tool: `render_presentation_slides` (lihat §MCP Tools di bawah).

---

## 📊 Ekstraksi Selektif Diagram ke Mermaid.js (`app/diagram.py`)

Tidak semua gambar visual pada dokumen cocok diubah menjadi diagram Mermaid. Modul analisis diagram mengevaluasi kelayakan secara selektif sebelum melakukan konversi sintaks:

1. **Kategori yang Cocok untuk Mermaid (`is_convertible = True`)**:
   - `flowchart`: Alur proses bisnis, pohon keputusan, alur logika (`flowchart TD` / `LR`).
   - `sequence_diagram`: Interaksi antar aktor, sistem, API, atau servis (`sequenceDiagram`).
   - `class_diagram`: Diagram kelas UML, struktur atribut & method (`classDiagram`).
   - `state_diagram`: State machine, siklus hidup status (`stateDiagram-v2`).
   - `er_diagram`: Entity Relationship Diagram data relasional (`erDiagram`).
   - `mindmap`: Peta konsep hierarkis bertingkat (`mindmap`).
   - `gantt_chart`: Jadwal linimasa proyek & milestone (`gantt`).
   - `block_architecture`: Arsitektur blok/komponen sistem (`flowchart` dengan `subgraph`).
2. **Kategori yang Tidak Cocok untuk Mermaid (`is_convertible = False`)**:
   - Grafik numerik kontinu padat (scatter plot, multi-series line chart, histogram, heatmap).
   - Peta geografis / denah spasial.
   - Foto realistis, gambar anatomi biologis, atau seni bebas.
   - Skematik sirkuit elektrik mikro atau CAD mekanik rumit.
   - *Penanganan:* Dikonversi ke format representasi yang lebih tepat (tabel Markdown atau ringkasan struktural), bukan dipaksakan ke kode Mermaid yang rusak.

---

## 🗄️ Mekanisme Tabular Transaksional vs Narrative Chunking (SQLite + Double-Verification)

Dokumen seperti **rekening koran, mutasi bank (Doc 8), ledger kas, slip transaksi, dan tabel keuangan** tidak cocok di-chunking ke Vector DB karena Vector Search tidak dapat melakukan kalkulasi agregat (SUM, AVG, filter tanggal).

Sistem kini dilengkapi modul cerdas ([`app/tabular_db.py`](file:///c:/Users/HYPE%20AMD/Documents/Coding/jds-magang/app/tabular_db.py)):
1. **Deteksi & Klasifikasi Heuristik + LLM**:
   - Membedakan tabel transaksional (`transactional_log`, `financial_statement`) vs tabel naratif (`narrative_matrix`).
   - Menganalisis rasio numerik, format tanggal ISO, keyword perbankan/akuntansi, dan kepadatan teks sel.
2. **Pydantic Validation**:
   - Skema tabel divalidasi ketat menggunakan Pydantic (`TableClassificationResult`, `TableSchema`, `TableColumnSchema`, `TableVerificationReport`, `TabularQueryResult`).
3. **SQLite Ingestion**:
   - Data otomatis dikonversi ke tipe data SQL murni (`REAL`, `INTEGER`, `DATE`, `TEXT`), membersihkan pemisah ribuan internasional (`4,218,640,517.32`) maupun format Indonesia (`4.218.640.517,32`).
4. **Double-Verification (Verifikasi Ganda)**:
   - **Row Count Integrity Check**: Memastikan tidak ada baris yang terlewat.
   - **Numeric & Aggregation Sanity Test**: Mengeksekusi query `SELECT SUM(...)` & `SELECT AVG(...)` untuk membuktikan query SQL bebas syntax/type error.
   - **Balance Continuity Check**: Menguji konsistensi aritmatika saldo berjalan ($Saldo_{n-1} + Kredit - Debit = Saldo_n$).
   - **LLM Reflection**: Mengonfirmasi akurasi ekstraksi terhadap sampel data mentah.

---

## 🤖 Arsitektur Sub-Agent (Deep Agents Harness)

Proyek ini dilengkapi dengan **Master Agent dan 8 Sub-Agent Spesialis** ([app/deep_agent.py](file:///c:/Users/HYPE%20AMD/Documents/Coding/jds-magang/app/deep_agent.py)) yang mendelegasikan tugas secara otonom:

| Nama Sub-Agent | Peran & Spesialisasi | Tool Utama |
|:---|:---|:---|
| `ocr-specialist` | Pembacaan teks mentah literal tingkat tinggi tanpa halusinasi | `ocr_document` (`ocr-lighton`) |
| `layout-classifier` | Deteksi multi-trait tata letak dokumen (kolom, hierarki, slide, tabel) | `classify_layout` |
| `markdown-extractor` | Ekstraksi gambar multimodal ke Markdown bersih berbasis spesifikasi komposit | `extract_to_markdown` |
| `diagram-mermaid-specialist` | Evaluasi selektif & ekstraksi diagram visual ke kode Mermaid.js yang valid | `classify_diagram_suitability`, `extract_diagram_to_mermaid` |
| `presentation-specialist` | Ekstraksi slide PowerPoint (.pptx/.ppt): render gambar per slide, lalu dibaca VLM menjadi Markdown | `extract_presentation_pptx` |
| `pdf-orchestrator` | Orkestrasi pemrosesan PDF multi-halaman & penyambungan kontinuitas heading | `extract_pdf_document` |
| `chunking-simulator` | Evaluasi kesiapan partisi Markdown dengan header splitter & recursive splitter | `preview_chunks` |
| `tabular-db-specialist` | Deteksi tabel transaksional, simpan ke SQLite, verifikasi ganda, & eksekusi query SQL | `classify_table_storage`, `ingest_table_to_sqlite`, `verify_sqlite_table`, `query_tabular_database` |

---

## 🔌 Model Context Protocol (MCP) Server & Batch Document Discovery

Server MCP berstandar resmi **MCP Python SDK v2.0** ([app/mcp_server.py](file:///c:/Users/HYPE%20AMD/Documents/Coding/jds-magang/app/mcp_server.py)) menyediakan MCP Tools lengkap untuk AI Assistant:

### Daftar MCP Tools:
1. **`scan_document_folders`**: Pindai direktori (mis. `dataset`, `input`, `output`) dan seluruh subfolder untuk mendeteksi folder dokumen.
2. **`batch_extract_documents`**: Ekstraksi dokumen massal dari folder terpilih dengan opsi kuota batas jumlah data.
3. **`extract_document_to_markdown`**: Ekstraksi file dokumen tunggal (PDF, PPTX, gambar) ke Markdown siap chunking.
4. **`ocr_image`**: Grounding teks mentah presisi tinggi via model OCR `ocr-lighton`.
5. **`classify_document_layout`**: Deteksi multi-trait spesifikasi tata letak dokumen.
6. **`classify_diagram_convertibility`**: Evaluasi kelayakan visual diagram untuk konversi ke Mermaid.js.
7. **`extract_diagram_to_mermaid`**: Ekstraksi diagram visual ke kode Mermaid.js yang valid secara sintaks.
8. **`extract_presentation_pptx`**: Ekstraksi PowerPoint (.pptx/.ppt) ke Markdown terstruktur.
9. **`render_presentation_slides`**: Render slide presentasi (.pptx/.ppt) menjadi file gambar PNG beresolusi tinggi (LibreOffice headless).
10. **`preview_markdown_chunks`**: Simulasi partisi teks Markdown dengan header metadata.
11. **`ingest_markdown_tables_to_sqlite`**: Ingesti otomatis tabel transaksional ke SQLite dengan double-verification.
12. **`query_tabular_database`**: Eksekusi SQL query untuk kalkulasi numerik (SUM, AVG, COUNT, date-filter) berpresisi 100%.
13. **`inspect_tabular_database`**: Inspeksi daftar tabel, skema kolom, dan jumlah baris di SQLite.
14. **`run_deep_reasoning_agent`**: Eksekusi Master Deep Reasoning Agent otonom dengan 8 sub-agent.

### Konfigurasi `mcp_config.json`:
```json
{
  "mcpServers": {
    "jds-magang-doc-agent": {
      "command": "python",
      "args": ["-m", "app.mcp_server"],
      "env": {
        "OPENAI_API_KEY": "your_api_key_here",
        "VLM_MODEL": "google/gemini-2.5-flash"
      }
    }
  }
}
```

---

## 🚀 Penggunaan CLI

### 1. Ekstraksi Dokumen Tunggal (PDF / Gambar / Scan)
```bash
# Ekstraksi PDF dengan layout terdeteksi otomatis
python main.py dataset/sample.pdf --output-dir output

# Ekstraksi dengan spesifikasi komposit jurnal ilmiah 2-kolom & hierarki bab
python main.py dataset/journal.pdf --specs "bilingual_journal,markdown_hierarchy"

# Ekstraksi disertai preview chunking untuk validasi RAG
python main.py dataset/document.pdf --preview-chunks --chunk-size 800 --chunk-overlap 100
```

### 2. Ekstraksi Presentasi PowerPoint (.pptx / .ppt)
```bash
# Pipeline VLM (default) - me-render slide ke gambar via LibreOffice headless
python main.py dataset/presentation.pptx --output-dir output

# Mode native teks (hanya membaca shape & text frame)
python main.py dataset/presentation.pptx --ppt-native
```

### 3. Pemindaian Dataset & Ekstraksi Massal (Batch Mode)
```bash
# Pindai struktur folder dataset
python main.py --scan-dir dataset

# Ekstraksi batch seluruh dokumen di dalam subfolder terpilih
python main.py --batch-folders "dataset/01_Kamus,dataset/02_Jurnal" --output-dir output/batch_results

# Ekstraksi batch dengan batas kuota dokumen
python main.py --batch-folders "dataset/03_Presentasi" --max-docs 5 --output-dir output/batch_sample
```

### 4. Eksekusi Deep Reasoning Agent
```bash
python main.py dataset/sample.pdf --deep-agent --preview-chunks
```
