# jds-magang — TableRAG × Vision OCR (RAG Multimodal Lokal)

Proyek pengembangan **RAG multimodal 100% lokal** untuk reasoning di atas dokumen heterogen
(teks, tabel, dan gambar dokumen). Dibangun di atas dua repo upstream — **TableRAG**
(RAG hybrid: SQL execution + textual retrieval) dan **qwen3-vl-embedding** (embedding
multimodal VLM via llama.cpp) — lalu dikembangkan dengan ide sendiri
(**multi-agent VLM document extraction**).

Semua komponen LLM memakai **Ollama lokal** (`qwen3.5:4b`): backbone agent, NL2SQL, dan VLM ekstraksi gambar.

---

## 📁 Struktur

```
jds-magang/
├── TableRAG/                    # Fork modifikasi dari yxh-y/TableRAG
│   ├── offline_data_ingestion_and_query_interface/   # Ingestion Excel→DB + service NL2SQL (Flask)
│   └── online_inference/        # Agent loop (solve_subquery) + retriever hybrid
├── vision_ocr/                  # IDE SENDIRI: multi-agent VLM document extraction
│   ├── dispatcher.py            #   Agent utama: klasifikasi jenis dokumen → routing
│   ├── agents.py                #   Registry 9 ExtractionAgent (prompt tuning per jenis)
│   ├── extractor.py             #   Pipeline gambar → VLM → JSON → validasi Pydantic
│   ├── llm.py                   #   Backend modular (Ollama native / OpenAI-compatible)
│   ├── schemas.py               #   Pydantic: DocumentExtraction (struktur bebas)
│   ├── embedder.py              #   VisionEmbedder: wrapper llama-vl-embedding (Qwen3-VL-Embedding)
│   └── cli.py                   #   CLI: auto-dispatch / paksa agent / klasifikasi saja
├── qwen3-vl-embedding/          # Fork ceveyne/qwen3-vl-embedding (setup belum selesai)
│   └── scripts/                 #   Convert + regression script
├── input/                       # Gambar input pengguna (git-ignored)
└── .gitignore
```

---

## ✅ Status Sekarang

| Komponen | Status |
|---|---|
| **TableRAG pipeline** | Dimodifikasi: backend DB → **SQLite** (MySQL tetap didukung via config), LLM → **Ollama qwen3.5:4b**, prompt NL2SQL → SQLite syntax, ~15 bug runtime & type-checker diperbaiki (`ty check` bersih) |
| **vision_ocr** | Multi-agent VLM extraction **berfungsi & teruji**: klasifikasi jenis (receipt/table/...) + ekstraksi struktur bebas, validasi Pydantic, `ty check` bersih |
| **Model lokal** | Ollama `qwen3.5:4b` (VLM + tools + thinking) — tool calling `solve_subquery` terverifikasi |
| **qwen3-vl-embedding** | Fork siap; **belum di-setup**: submodule belum init, binary `llama-vl-embedding` belum di-build, model 2B/8B belum di-download |
| **Repo** | Satu git tunggal, private: `github.com/daruoktab/jds-magang` |

### Hasil uji `vision_ocr` (qwen3.5:4b)
- Struk Euro → `receipt`: merchant, tanggal ISO, 6 item (harga numerik + `currency: EUR`), `subtotal/tax/total/change` akurat.
- Tabel → `table`: `{columns, rows}` akurat.
- Mode generic: model menentukan sendiri key-value & struktur terbaik.

---

## 💡 Ide: Combine VLM + Embedding Multimodal (pola TableRAG)

Prinsip yang dipinjam dari TableRAG: **simpan 2 representasi, recall murah dulu, understand mahal belakangan**.

```
OFFLINE (dua representasi per gambar):
  Gambar ──▶ Qwen3-VL-Embedding (llama-vl-embedding) ──▶ FAISS visual   (retrieval cepat)
  Gambar ──▶ VLM qwen3.5:4b (vision_ocr) ────────────▶ SQLite + teks    (understanding detail)

ONLINE:
  Query ──▶ embed sekali (Qwen3-VL-Embedding) ──▶ FAISS search ──▶ top-k gambar relevan
                └──▶ VLM hanya untuk gambar terpilih ──▶ struktur ──▶ COMBINE_PROMPT / SQL
```

- **Embedding multimodal** (bge diganti total): satu model untuk teks, tabel markdown, dan gambar.
- **VLM** hanya dipanggil untuk top-k hasil retrieval — hemat token/waktu (persis alasan TableRAG pakai SQL).
- Extension agent: tool baru `solve_image` di agent loop untuk "visual subquery".

---

## 🗺️ Plan Sementara (Roadmap)

1. **Setup `qwen3-vl-embedding`**
   - `git submodule update --init --recursive`
   - Build `llama-vl-embedding` (Windows: `cmake --build llama.cpp/build --target llama-vl-embedding`)
   - Download Qwen3-VL-Embedding-2B/8B, convert GGUF (main `f16` + `--mmproj f16`)
2. **`vision_ingestor.py`** — offline: hasil `vision_ocr` → `TableData` jadi DataFrame → SQLite + schema JSON; key-value → chunk. Gambar ikut pipeline SQL seperti `.xlsx`.
3. **Retriever multimodal** — extend `MixedDocRetriever`: `VisionEmbedder` (Qwen3-VL) menggantikan bge-m3; index FAISS visual paralel; query di-embed sekali → fusion skor → top-k.
4. **Tool `solve_image`** — agent memanggil `vision_ocr` untuk subquery visual; hasil masuk `COMBINE_PROMPT`.
5. **Uji end-to-end** — ingestion `dev_excel.zip` + gambar → service SQL → agent loop.

---

## 🚀 Menjalankan

```bash
# 1. Environment
conda activate magang-jds          # Python 3.12; torch diinstall manual
uv pip install -r TableRAG/requirements.txt
ollama serve                       # + ollama pull qwen3.5:4b

# 2. Vision OCR (gambar → struktur)
python -m vision_ocr.cli input/gambar.jpg                 # auto: klasifikasi + agent terbaik
python -m vision_ocr.cli input/gambar.jpg --agent receipt  # paksa agent tertentu
python -m vision_ocr.cli --list-agents

# 3. TableRAG offline (ingestion + service SQL)
cd TableRAG/offline_data_ingestion_and_query_interface/src
python data_persistent.py          # Excel → SQLite (config: database_config.json)
python interface.py                # Flask :5000 (NL2SQL via qwen3.5:4b)

# 4. Agent loop
cd TableRAG/online_inference
python main.py --data_file_path <cases.json> --doc_dir <docs> --excel_dir <excels> --backbone v3
```

> Catatan: `git push` via protokol `github.com` tidak stabil di jaringan ini; push rutin sebaiknya via
> `git@ssh.github.com:443` (SSH) atau Git Data API.
