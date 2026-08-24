# jds-magang-ocr-agent — Vision OCR & Document Extractor (Ready for Chunking & Tabular SQLite Database)

Sistem ekstraksi dokumen multimodal (PDF, PPT/PPTX, Scan Gambar) menjadi **Markdown bersih dan terstruktur yang siap langsung di-chunking** untuk pipeline RAG downstream, serta **Engine Data Tabular Transaksional ke SQLite** untuk data log mutasi/rekening koran/faktur yang memerlukan kalkulasi agregat berpresisi 100% (SUM, AVG, COUNT, filter tanggal).

> ℹ️ **Catatan Branch:** 
> Fitur lengkap pipeline RAG end-to-end (Embedding `Qwen3-VL-Embedding`, Reranker `Qwen3-VL-Reranker`, dan Vector Store) tersimpan di branch `end-to-end`. Branch `main` difokuskan pada pipeline ekstraksi dokumen ke format Markdown siap chunking dan Tabular SQLite Database.

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

Proyek ini dilengkapi dengan **Master Agent dan 7 Sub-Agent Spesialis** ([app/deep_agent.py](file:///c:/Users/HYPE%20AMD/Documents/Coding/jds-magang/app/deep_agent.py)) yang mendelegasikan tugas secara otonom:

| Nama Sub-Agent | Peran & Spesialisasi | Tool Utama |
|:---|:---|:---|
| `ocr-specialist` | Pembacaan teks mentah literal tingkat tinggi tanpa halusinasi | `ocr_document` (`ocr-lighton`) |
| `layout-classifier` | Deteksi multi-trait tata letak dokumen (kolom, hierarki, slide, tabel) | `classify_layout` |
| `markdown-extractor` | Ekstraksi gambar multimodal ke Markdown bersih berbasis spesifikasi komposit | `extract_to_markdown` |
| `presentation-specialist` | Ekstraksi slide PowerPoint (.pptx) dengan hierarki bullet, tabel, dan notes | `extract_presentation_pptx` |
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
6. **`extract_presentation_pptx`**: Parser file PowerPoint (.pptx/.ppt) ke Markdown terstruktur.
7. **`preview_markdown_chunks`**: Simulasi partisi teks Markdown dengan header metadata.
8. **`ingest_markdown_tables_to_sqlite`**: Ingesti otomatis tabel transaksional ke SQLite dengan double-verification.
9. **`query_tabular_database`**: Eksekusi SQL query untuk kalkulasi numerik (SUM, AVG, COUNT, date-filter) berpresisi 100%.
10. **`inspect_tabular_database`**: Inspeksi daftar tabel, skema kolom, dan jumlah baris di SQLite.
11. **`run_deep_reasoning_agent`**: Eksekusi Master Deep Reasoning Agent otonom.

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
        "PYTHONPATH": "c:\\Users\\HYPE AMD\\Documents\\Coding\\jds-magang",
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

## 🤖 Agent Mode (`app/mcp_agent_server.py`) — Tanpa OCR/VLM/LLM Endpoint

Server MCP khusus untuk **agent model berbasis vision** (Gemini CLI, Claude Desktop, Cursor, dsb.) yang mengerjakan ekstraksi **secara mandiri tanpa memanggil endpoint model apa pun**. OCR, VLM, maupun LLM **dimatikan** — agent-lah yang membaca gambar dokumen dan menulis Markdown-nya sendiri; server hanya menyediakan alat bantu mekanis. Hasil yang disimpan melalui `save_extraction_result` otomatis mendeteksi tabel transaksional dan meng-ingest-nya ke database SQLite dengan verifikasi ganda.
