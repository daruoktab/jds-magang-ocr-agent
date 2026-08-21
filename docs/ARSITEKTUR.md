# Arsitektur, Logika, dan Algoritma

Dokumen rujukan untuk seluruh logika di repo ini: apa yang dikerjakan tiap modul,
algoritma apa yang dipakai, dari mana asalnya, dan bagian mana dari sitasi itu
yang benar-benar dipakai.

Status per 21 Agustus 2026, cabang `main`.

---

## 1. Prinsip inti

Satu kalimat yang menurunkan hampir semua keputusan desain di sini:

> **Perbesar yang bisa dipastikan program, perkecil yang harus ditebak model —
> dan jangan pernah biarkan model menebak apa yang program sudah tahu.**

Turunannya yang berlaku di seluruh kode:

| Aturan | Penerapan |
|---|---|
| Program mengukur bentuk, VLM membaca isi | `surveyor.py` tidak pernah memanggil LLM |
| OCR dipakai untuk koordinat, bukan untuk teks | `surveyor.py::_detect_row_start` hanya memakai posisi token |
| VLM tidak menentukan kedalaman | `hierarchy.py::StackMachine` yang menyusun pohon |
| Yang bisa dihitung program tidak boleh diwariskan | Identitas baris dari Surveyor, bukan dari laporan batch sebelumnya |
| Menolak menebak | Zona "ragu" pada keterisian kolom; perbaikan ordinal hanya bila nilainya tunggal |
| Kontrak dibekukan, direvisi lewat jalur terpisah | `level_map` dan tanda tangan schema tidak berubah di tengah span |

---

## 2. Peta modul

Repo ini berisi **dua subsistem** yang berdampingan di atas komponen bersama.

### 2.1 Jalur A — pipeline VLM per halaman (sudah jalan)

Untuk dokumen normal: PDF digital, hasil scan biasa, PPT.

```
main.py / mcp_server.py
   └── graph.py  DocumentExtractionPipeline (LangGraph)
         START → preprocess → ocr → classify → extract_markdown → END
```

| Modul | Tugas | Catatan algoritma |
|---|---|---|
| `config.py` | Loader `.env` minimal tanpa dependensi | `Settings` beku (`frozen=True`), singleton lazy |
| `preprocess.py` | Auto-orientasi EXIF, autocontrast, paksa RGB | `ImageOps.autocontrast(cutoff=0.5)` + `ImageEnhance.Contrast(1.2)` |
| `llm.py` | Builder `ChatOpenAI` untuk endpoint OpenAI-compatible | `enable_thinking` dikirim lewat `extra_body`, bukan `model_kwargs` |
| `ocr.py` | OCR grounding lewat VLM kecil | Dipakai sebagai *referensi ejaan*, bukan sumber utama |
| `prompts.py` | Prompt komposisional | 4 modul aturan digabung sesuai spec aktif |
| `extractor.py` | Klasifikasi layout + ekstraksi Markdown | Parsing JSON dengan fallback regex multi-label |
| `agents.py` | Registry agent per spec, mendukung kombinasi | Spec majemuk membentuk agent komposit `a+b` |
| `graph.py` | Orkestrasi 4 node | Tiap node menangkap exception dan turun ke fallback |
| `pdf.py` | Render & orkestrasi PDF multi-halaman | Batch 10 halaman, konteks 400 karakter terakhir dioper |
| `ppt.py` | Ekstraksi PPTX/PPT | Batch 10 slide (workaround limit lisensi Spire Free) |
| `multi_page.py` | Jahit halaman, bersihkan artefak, simulasi chunking | Regex nomor halaman, penyambungan paragraf terpotong |
| `batch.py` | Pemindaian folder & ekstraksi massal | — |
| `deep_agent.py` | Harness `deepagents` dengan 6 sub-agent | Master orchestrator mendelegasi per jenis file |
| `mcp_server.py`, `mcp_agent_server.py` | Antarmuka MCP | 935 baris untuk agent server (tooling anotasi/gold set) |

**Algoritma penting di jalur ini**

*Prompt komposisional* (`prompts.py::build_extraction_prompt`) — alih-alih satu
prompt raksasa, aturan disusun dari modul yang aktif: `_RULE_COLUMN_AWARE`,
`_RULE_MARKDOWN_HIERARCHY`, `_RULE_PRESENTATION_SLIDES`, `_RULE_PLAIN_DOCUMENT`.
Dokumen bisa punya lebih dari satu karakteristik sekaligus, sehingga
klasifikasinya multi-label, bukan pilihan tunggal.

*Konteks lintas halaman* (`pdf.py::process_multipage_pdf`) — 400 karakter
terakhir halaman sebelumnya dioper sebagai `previous_page_context`. Ini bentuk
paling awal dari cursor; lihat §4 untuk penggantinya yang bertipe.

*Penjahitan halaman* (`multi_page.py`) — membuang running header/footer lewat
pola regex nomor halaman, lalu menyambung paragraf yang terpotong bila blok
sebelumnya tidak diakhiri tanda baca penutup dan blok berikutnya tidak dimulai
penanda struktur.

### 2.2 Jalur B — perancah deterministik (baru)

Untuk dokumen hasil scan berskala besar dengan tabel bergaris — kasus yang
membuat jalur A tidak memadai.

```
surveyor.py   (geometri, tanpa LLM)  →  SurveyReport
hierarchy.py  (struktur, tanpa LLM)  →  pohon + audit + Markdown
```

Belum tersambung ke `graph.py`; keduanya masih berdiri sendiri dan dijalankan
lewat CLI. Rencana penyatuannya ada di §7.

---

## 3. `surveyor.py` — pemetaan geometri

Tidak memanggil LLM. Seluruh keluarannya diturunkan dari citra halaman.

### 3.1 Pengambilan peta tinta

`_page_ink` — bila halaman berupa **satu citra utuh** (ciri khas dokumen scan),
citra aslinya diambil langsung lewat `doc.extract_image()` tanpa render ulang.
Pada PP 28/2025 citra aslinya CCITT G4 bilevel 400 DPI berukuran ~167 KB;
me-render ulang ke 300 DPI justru menurunkan resolusi sekaligus memperlambat.

Pengaman: skala x dan y harus konsisten dalam 2% sebelum citra dipercaya
sebidang dengan halaman. Kalau tidak, jatuh ke `get_pixmap(dpi=300)`.

### 3.2 Deteksi grid — `_detect_grid`

Projection profile pada citra biner:

1. **Garis horizontal panjang**: baris piksel dengan tinta > 35% lebar halaman,
   lalu digabung bila berjarak ≤ 8 piksel.
2. **Badan tabel**: dimulai di **celah besar pertama**, bukan yang terlebar.
   Alasannya empiris — pada halaman yang memuat awal baris, badan tabel
   terpotong sekat sub-baris sehingga celah terlebar justru jatuh *di dalam*
   badan, dan nomor barisnya terbuang ke luar zona pencarian.
3. **Blok header**: seluruh wilayah dari garis teratas sampai garis pembuka
   badan, diukur **sekaligus**. Mengukur per pasang garis gagal pada halaman
   miring: satu garis kerap terdeteksi ganda, dan pita tipis di antaranya
   membuat goresan huruf ikut terhitung sebagai sekat kolom (pernah
   menghasilkan 108 kolom pada satu halaman).
4. **Sekat kolom**: kolom piksel dengan tinta > 85% tinggi blok header.
5. Sekat di 1% tepi citra dibuang — itu bekas tepi hitam scan, bukan sekat.

### 3.3 Koreksi kemiringan — `_estimate_skew`

Dikerjakan **hanya setelah** deteksi murah gagal, dan hanya bila halaman punya
garis horizontal panjang sama sekali. Citra diperkecil 4×, diputar pada rentang
±1,2° dengan langkah 0,1°, lalu diambil sudut yang memaksimalkan puncak
projection.

Yang diukur adalah garis grid yang dicetak ulang di setiap halaman — objek yang
bentuknya sudah pasti lurus, bukan tebakan dari isi teks.

### 3.4 Peta keterisian kolom — `_column_density` + `_classify_occupancy`

Kerapatan tinta per kolom di badan tabel, dikali 1000. Pemisahannya lebar:
kolom kosong ±5, kolom berisi 80–110.

```
DENSITY_EMPTY_MAX  = 9.0   → "kosong"
DENSITY_FILLED_MIN = 13.0  → "isi"
di antaranya               → "ragu"   (dieskalasi, tidak ditebak)
```

Gunanya menghapus satu kelas halusinasi seluruhnya: pertanyaan *"kolom ini
kosong, atau modelnya lupa?"* berubah dari tak terjawab menjadi fakta yang sudah
diketahui sebelum VLM dipanggil.

### 3.5 Deteksi awal baris — `_detect_row_start`

Mencari token berangka di rentang x kolom pertama, **dibatasi pada badan tabel**.
Pembatasan y wajib: kode pengaman `SK No 122554 C` di kaki halaman berada pada
rentang x yang sama dan akan salah terbaca sebagai awal baris.

Toleransi 4 pt ke atas karena nomor baris kerap tercetak sedikit di atas garis.

Di sinilah text layer OCR dipakai — **posisinya**, bukan isinya. Pada PP 28/2025
karakter OCR sering salah (`L2` untuk `12`, `03l 14` untuk `03114`) sementara
posisinya tetap benar. Akurasi terukur: 41/42 halaman pada rentang uji, satu-
satunya meleset adalah halaman judul lampiran yang punya aturan tersendiri.

### 3.6 Tanda tangan schema & segmentasi span — `segment_spans`

Tanda tangan = `(jumlah kolom, posisi sekat relatif terhadap lebar halaman)`,
dicocokkan dengan toleransi 0,03.

Dua pendekatan lain sudah diuji dan **gagal**:

- *pHash pita header* — beda piksel 25–60% antar halaman yang schema-nya
  identik, karena tiap halaman adalah hasil scan terpisah.
- *Pencocokan persis posisi sekat* — menghasilkan 35 kelas palsu pada 232
  halaman sampel, karena posisi bergoyang ±0,01–0,02 akibat skew dan crop.

Dengan toleransi, 232 halaman sampel PP 28/2025 melebur jadi **dua** kelas:
13 kolom untuk seluruh Lampiran I, 8 kolom untuk Lampiran II.

Konsekuensi biaya: agent penentu schema cukup dipanggil 2–5 kali untuk 11.591
halaman, bukan sekali per halaman. Header tabel dicetak ulang di setiap halaman,
jadi *kemunculan header bukan penanda tabel baru*.

### 3.7 Pewarisan grid

Bila deteksi gagal di satu halaman tetapi halaman sebelumnya bergrid, grid
diwarisi dan ditandai `source="warisan"`. Warisan diputus saat **orientasi
berubah**, bukan saat satu halaman gagal — kegagalan satu halaman di tengah
tabel justru alasan utama adanya warisan.

### 3.8 Rencana batch — `plan_batches`

Ukuran nominal 10 halaman, tetapi batas digeser sampai 2 halaman agar jatuh
tepat di awal baris logis. Batch yang dimulai di awal baris membuka jauh lebih
sedikit scope dan bisa dijalankan ulang sendirian.

### 3.9 Degradasi mulus

Halaman landscape tanpa grid **bukan anomali** — bisa slide presentasi, bagan,
atau gambar selebar halaman. Mode tetap `"teks"`, tanpa peringatan. Peringatan
hanya muncul bila halaman sebelumnya bergrid, sebab hanya di situ grid memang
seharusnya ada.

Terverifikasi pada PDF digital biasa, slide landscape, dan bagan organisasi
Perbup Malang: semuanya `teks`, nol peringatan.

---

## 4. `hierarchy.py` — mesin tumpukan & auditor

Pasangan deterministik Surveyor. Kalau Surveyor memetakan bentuk halaman, modul
ini menyusun bentuk isinya.

### 4.1 Model event datar

VLM agent **hanya** melaporkan apa yang dilihatnya:

```python
Event(kind="pasal", ordinal="30", page=11)
Event(kind="ayat",  ordinal="1", text="...", page=11)
```

Tidak ada `#`. Tidak ada kedalaman. Tidak ada penyarangan.

### 4.2 Peta level

`LEVEL_ORDER` mengikuti UU 12/2011 Lampiran II butir 85–95:

```
judul → pembukaan → buku → bab → bagian → paragraf → pasal → ayat
      → huruf → angka → huruf2 → angka2
```

Karena `LEVEL_ORDER` adalah list, kedalaman tidak terbatas — menambah level
berarti menambah satu entri.

`ORDINAL_KIND` menetapkan gaya penomoran tiap level (romawi, kata bilangan,
alfabet, arab), dipakai Auditor untuk memeriksa kesinambungan.

### 4.3 Mesin tumpukan — `StackMachine`

Aturannya satu kalimat: **event menutup semua scope yang levelnya sama atau
lebih rendah, lalu menempel pada scope yang tersisa di puncak tumpukan.**

Akibatnya dua kelas kesalahan menjadi *mustahil*, bukan sekadar jarang:

- **Level drift** — Pasal tidak bisa jadi `###` di satu batch lalu `##` di batch
  berikutnya, sebab bukan VLM yang menentukan levelnya.
- **Kehilangan induk** — Pasal 30 otomatis menempel pada Bab yang masih terbuka,
  walau keduanya terpisah ratusan halaman.

Satu pengecualian eksplisit, `BLOCKED_CHILDREN`: pembukaan (Menimbang /
Mengingat / MEMUTUSKAN) tidak boleh menampung Bab atau Pasal. Ia saudara batang
tubuh, bukan induknya. Tanpa aturan ini seluruh batang tubuh bersarang di dalam
`MEMUTUSKAN`.

### 4.4 Cursor

Yang dioper antar batch berukuran **O(kedalaman), bukan O(halaman)**:

```json
{
  "open_path": [{"kind":"bab","ordinal":"V"}, {"kind":"bagian","ordinal":"Ketiga"},
                {"kind":"pasal","ordinal":"8"}, {"kind":"huruf","ordinal":"d"}],
  "last_seen": {"bab":"V","bagian":"Ketiga","pasal":"8","huruf":"d"},
  "tail": "...", "catchword": "..."
}
```

Terukur 347–644 byte pada semua dokumen uji. Ini koreksi terhadap rancangan awal
yang mengoper daftar isi yang menumpuk — daftar isi tumbuh linear terhadap
panjang dokumen, cursor tidak.

`_restore` membangun ulang jalur terbuka di awal batch berikutnya, sehingga
batch yang dibuka di tengah daftar tetap menempel pada induknya.

### 4.5 Auditor ordinal — `audit`

Memeriksa kesinambungan penomoran, dengan satu aturan yang menentukan:

> **Perbaiki hanya bila ruang nilai yang mungkin tinggal satu.**

```
last=11, terbaca "L2", berikutnya 13   → 12 tertentu tunggal  → PERBAIKI
last=03113, "03l 14", lalu 03115       → 03114 tunggal         → PERBAIKI
last=6, terbaca 8                      → 7 hilang, dua tafsir  → ESKALASI
```

`to_int` **menolak** ordinal yang tidak seluruhnya angka. Membuang huruf
diam-diam berbahaya: `"L2"` akan terbaca 2, dan kesalahan itu lolos sebagai
nilai sah. Lebih baik dinyatakan tak terbaca supaya urutan yang menyimpulkan.

**Nomor bersuffiks hasil amandemen** (Pasal 6A, 7B, 18B) disimpan sebagai
pecahan perseratus agar urutan `6 < 6A < 6B < 7` terjaga. `_successor_ok`
menerima tiga bentuk lanjutan: naik satu tingkat, varian bersuffiks dari nomor
yang sama, dan kembali ke nomor bulat berikutnya sesudah rangkaian suffiks.
Ketegasannya tidak berkurang — `6 → 6B` tetap ditandai karena 6A hilang.

Level yang menomori ulang di dalam induknya (`RESETS_INSIDE_PARENT`) kehilangan
riwayat begitu induknya berganti, supaya ayat (1) di Pasal berikutnya tidak
dianggap mundur.

Kedalaman rincian dibatasi 4 tingkat sesuai butir 87 UU 12/2011.

### 4.6 Auditor rupiah — `audit_amounts`

Dokumen anggaran menyebutkan angka yang sama berkali-kali dari sudut berbeda:
total di satu pasal, rinciannya di pasal lain, saling menunjuk lewat frasa
*"sebagaimana dimaksud pada ayat (1) huruf a"*. Redundansi itu dipakai sebagai
uji silang gratis.

Algoritmanya: bangun indeks `(pasal, ayat, huruf) → nilai`, lalu untuk setiap
ayat yang punya rujukan, jumlahkan nilai anak-anaknya dan bandingkan.

Detail yang penting: `parse_rupiah` harus menerima `Rp` maupun `Rp.` (Perda
Tangsel memakai yang pertama, Perda Kerinci yang kedua), dan rujukan `huruf (c)`
berkurung. Tanpa keduanya, satu dokumen menghasilkan **nol** uji silang.

**Hasil nyata**: pada Perda Kabupaten Kerinci 1/2008 auditor menemukan rincian
Pasal 4 ayat (2) berjumlah Rp66.590.038.541,64 sedangkan ayat (1) huruf a
menyebut Rp65.090.038.541,64 — selisih persis sebesar butir penerimaan kembali
pinjaman, dan Pasal 1 dokumen itu memakai angka yang pertama. Salah ketik asli
di dokumen, ditemukan tanpa LLM.

### 4.7 Renderer — `render_markdown`

Level heading murni turunan pohon. Indentasi daftar mengikuti **kedalaman
sebenarnya di pohon**, bukan jenis levelnya — daftar angka di bawah "Mengingat"
berada di tingkat teratas, sedangkan angka di dalam huruf di dalam ayat menjorok.

### 4.8 Kata penyambung — `check_catchword`

UU 12/2011 mewajibkan kata penyambung di kaki halaman. Dokumen menyediakan
sendiri penanda sambungan antar halaman, jadi pemeriksaan seam ini gratis bila
tersedia.

---

## 5. Konstanta kalibrasi dan asal-usulnya

Semua nilai di `surveyor.py` dikalibrasi pada dokumen peraturan hasil scan
bilevel 300–400 DPI. **Inilah batas keberlakuannya** (lihat §8).

| Konstanta | Nilai | Asal |
|---|---|---|
| `H_LINE_FRAC` | 0.35 | garis horizontal grid menutupi ≥35% lebar halaman |
| `V_LINE_FRAC` | 0.85 | sekat kolom menembus ≥85% tinggi blok header |
| `GROUP_GAP` | 8 px | tebal garis + jitter scan |
| `COL_PAD_FRAC` | 0.010 | padding agar sisa garis vertikal tidak terhitung tinta |
| `DENSITY_EMPTY_MAX` / `FILLED_MIN` | 9.0 / 13.0 | dari sebaran terukur: kosong ±5, berisi 80–110 |
| `SIGNATURE_TOL` | 0.03 | goyangan posisi sekat akibat skew dan crop |
| `SKEW_MAX_DEG` | 1.2° | rentang kemiringan yang teramati |
| `ROW_TOKEN_TOLERANCE_PT` | 4.0 | nomor baris kerap tercetak sedikit di atas garis |
| `MAX_RINCIAN_DEPTH` | 4 | UU 12/2011 Lampiran II butir 87 |
| `PDF_PAGE_BATCH` | 10 | ukuran unit kerja per iterasi |

---

## 6. Sitasi dan bagian yang dipakai

Ditandai: **[P]** sudah diimplementasikan · **[R]** dirancang, belum ditulis ·
**[X]** ditimbang lalu ditolak.

### 6.1 Sumber normatif

**UU 12/2011 Lampiran II — Teknik Penyusunan Peraturan Perundang-undangan** · **[P]**
<https://id.wikisource.org/wiki/Undang-Undang_Republik_Indonesia_Nomor_12_Tahun_2011/Lampiran/2>

| Bagian yang dipakai | Di mana |
|---|---|
| Butir 85–95: urutan Bab → Bagian → Paragraf → Pasal → Ayat → rincian | `hierarchy.py::LEVEL_ORDER` |
| Gaya penomoran tiap level (Romawi, kata bilangan, angka Arab, huruf) | `hierarchy.py::ORDINAL_KIND` |
| Butir 87: rincian maksimal 4 tingkat | `hierarchy.py::MAX_RINCIAN_DEPTH` |
| Kewajiban kata penyambung di kaki halaman | `hierarchy.py::check_catchword` |

Inilah alasan hirarki tidak perlu ditebak model: strukturnya **diatur
undang-undang**, jadi `level_map` bisa dipatok di kode.

### 6.2 Paper yang diimplementasikan

**Information Extraction From Fiscal Documents Using LLMs** · arXiv [2511.10659](https://arxiv.org/html/2511.10659) · **[P]**

| Bagian yang dipakai | Di mana |
|---|---|
| *Sequential context*: tiap halaman menerima hasil ekstraksi halaman sebelumnya | `hierarchy.py::Cursor`; bentuk awalnya `pdf.py::previous_page_context` |
| Render ke citra dan buang text metadata, paksa model ber-OCR sendiri | Prinsip "text layer bukan sumber teks" di `surveyor.py` |
| Validasi konsistensi numerik: anak harus menjumlah ke induk (84% lolos) | `hierarchy.py::audit_amounts` |
| Meta-prompting: LLM menulis sendiri prompt ekstraksinya | Belum dipakai |
| TEDS untuk kebenaran struktur (73–97%) | Belum dipakai — celah evaluasi |

**The Format Tax** · arXiv [2604.03616](https://arxiv.org/abs/2604.03616) · **[P]**

| Bagian yang dipakai | Di mana |
|---|---|
| Biaya dominan output terstruktur ada di **instruksi format pada prompt** yang menekan penalaran, bukan di constrained decoding | Pemisahan dua pass: VLM menalar bebas, lalu mencetak event datar |
| Memisahkan penalaran dari pemformatan memulihkan sebagian besar akurasi | Seluruh pemformatan dipindahkan ke kode (`render_markdown`) |

**Detect-Order-Construct** · arXiv [2401.11874](https://arxiv.org/pdf/2401.11874) · **[P]**

| Bagian yang dipakai | Di mana |
|---|---|
| Pohon dibangun lewat **prediksi relasi**, bukan prediksi kedalaman absolut | `hierarchy.py::StackMachine` — model melapor event, kode menyusun pohon |

**Extracting Variable-Depth Logical Document Hierarchy from Long Documents** · arXiv [2105.09297](https://arxiv.org/pdf/2105.09297) · **[P]**

| Bagian yang dipakai | Di mana |
|---|---|
| Kedalaman hirarki bervariasi dan tidak boleh dipatok angka tetap | `LEVEL_ORDER` berupa list, kedalaman tak terbatas |

**TASER: Table Agents for Schema-guided Extraction and Recommendation** · arXiv [2508.13404](https://arxiv.org/abs/2508.13404) · **[P]** sebagian

| Bagian yang dipakai | Di mana |
|---|---|
| Schema dibekukan di depan, ekstraksi mengikutinya | Tanda tangan span di `segment_spans` |
| Bukti empiris tabel membentang banyak halaman (terpanjang 44 halaman dari 22.584 halaman) | Justifikasi seluruh mekanisme cursor & scope terbuka |
| *Recommender Agent* mengusulkan revisi schema dari keluaran yang tidak cocok, dijalankan batch bukan inline | **[R]** dirancang, belum ditulis |

### 6.3 Paper yang dirancang, belum ditulis

**Executable Schema Contracts** · arXiv [2606.05415](https://arxiv.org/abs/2606.05415) · **[R]**

| Bagian yang dipakai | Rencana |
|---|---|
| *Closed-world field catalog*: penemuan schema oleh LLM dibatasi pada field yang benar-benar terbukti ada | Aturan `propose_schema_change` hanya boleh menyebut level/kolom yang teramati |
| Identity key & foreign key diturunkan secara deterministik, bukan ditanyakan ke model | Sejalan dengan prinsip inti |

**Schema Key Wording as an Instruction Channel** · arXiv [2604.14862](https://arxiv.org/pdf/2604.14862) · **[R]**

| Bagian yang dipakai | Rencana |
|---|---|
| Nama field berfungsi sebagai kanal instruksi saat constrained decoding; nama semantik mengungguli label generik | "Kartu kolom": nama + komentar + 2–3 contoh nilai sebagai keluaran agent schema, dipakai ulang ribuan halaman |

**Tabular Data Understanding with LLMs: A Survey** · arXiv [2508.00217](https://arxiv.org/pdf/2508.00217)
dan **A Closer Look into LLMs for Table Understanding** · arXiv [2603.15402](https://arxiv.org/pdf/2603.15402) · **[R]**

| Bagian yang dipakai | Rencana |
|---|---|
| Markdown/CSV hemat token (HTML ~3× lebih boros); header eksplisit + hint tipe menaikkan akurasi di semua format | Format tabel dua lapis: DDL sebagai kontrak semantik, CSV sebagai baris |

**HiChunk / HiCBench** · ACL 2026 Long [1372](https://aclanthology.org/2026.acl-long.1372/) · **[R]**

| Bagian yang dipakai | Rencana |
|---|---|
| Chunking hirarkis + algoritma *Auto-Merge* saat retrieval | Tahap sesudah pohon jadi; saat ini `multi_page.preview_markdown_chunks` masih placeholder berbasis header |

**PubTables-v2** · arXiv [2512.10888](https://arxiv.org/abs/2512.10888) · **[R]**
Dataset multi-halaman sebagai rujukan bahwa tabel lintas halaman adalah kasus normal, bukan pinggiran.

**HPD-Parsing** · arXiv [2607.18839](https://arxiv.org/abs/2607.18839) · **[R]**
Hierarchical parallel decoding sebagai rute penskalaan bila 10 halaman per proses terlalu lambat untuk dokumen 11.591 halaman.

**MinerU-Popo** · arXiv [2605.24973](https://arxiv.org/html/2605.24973v1) · **[R]**
Model post-processing untuk memperbaiki hasil parsing terstruktur; calon lapisan repair sesudah Auditor menandai.

**Semantic Evaluation of PDF Table Extraction** · arXiv [2603.18652](https://arxiv.org/html/2603.18652v1) · **[R]**

| Bagian yang dipakai | Rencana |
|---|---|
| LLM-as-judge berkorelasi r=0,93 dengan manusia, TEDS hanya r=0,68 | Evaluasi **sampel luring** saja — terlalu mahal dan tidak deterministik untuk inline |

**OmniDocBench** · CVPR 2025 · <https://github.com/opendatalab/OmniDocBench> · **[R]**
Benchmark end-to-end (1.651 halaman, anotasi tabel LaTeX dan HTML) untuk mengukur pipeline setelah gold set tersedia.

**Akoma Ntoso / LegalDocML** · [OASIS](https://docs.oasis-open.org/legaldocml/akn-core/v1.0/akn-core-v1.0-part1-vocabulary.html) · **[R]**
Kandidat format keluaran menggantikan Markdown. Heading Markdown kehilangan identitas level: `###` tidak tahu dirinya "Pasal".

**Schema-Driven Information Extraction from Heterogeneous Tables** · arXiv [2305.14336](https://arxiv.org/abs/2305.14336) · **[R]**
Definisi tugas: schema tulisan manusia → record terstruktur. Landasan konseptual jalur tabel.

### 6.4 Ditimbang lalu ditolak

| Sumber | Alasan ditolak |
|---|---|
| **POTATR** · arXiv [2606.09788](https://arxiv.org/pdf/2606.09788) · **[X]** | Classifier kontinuasi terlatih butuh data latih yang tidak kita punya. Sinyal tinta di kolom pertama sudah deterministik dan lebih akurat di dokumen ini. |
| **OTSL** · arXiv [2305.03393](https://arxiv.org/pdf/2305.03393) dan **DocTags/SmolDocling** · arXiv [2503.11576](https://arxiv.org/html/2503.11576v1) · **[X]** | Dirancang memulihkan struktur yang belum diketahui. Schema di sini sudah beku dan terverifikasi geometri; OTSL hanya menambah kosakata tanpa manfaat. |
| Embedding similarity untuk pencocokan schema · **[X]** | Non-deterministik untuk pertanyaan yang geometri sudah jawab persis. |
| pHash pita header · **[X]** | Diuji: beda piksel 25–60% antar halaman yang schema-nya identik. |
| **Annotares** · arXiv [2608.03898](https://arxiv.org/html/2608.03898v1) · **[X]** | Sempat dikira tentang struktur hirarkis; ternyata anotasi Tatbestand/Rechtsfolge di level kalimat. Tidak relevan. |

---

## 7. Rencana penyatuan dua jalur

Jalur B belum tersambung ke `graph.py`. Titik sambungnya sudah tersedia:

| Kait yang sudah ada | Perubahan |
|---|---|
| `specs` multi-label di `prompts.py` | Tambah spec `ruled_table` dan `legal_hierarchy` + modul `_RULE_*` |
| `previous_page_context: str \| None` di `extractor.py` | Naikkan jadi `Cursor` bertipe, render ke string di batas pemanggilan |
| Node `classify` di `graph.py` | Surveyor menjawab deterministik; klasifikator VLM jadi fallback. Menghapus 1 panggilan VLM per halaman |
| Node baru `survey` sebelum `classify` | Menyuplai fakta geometri |

Yang **tidak** akan disatukan: orkestrasinya. `graph.py` bekerja per halaman;
jalur sulit bekerja per batch dengan cursor. Memaksa graf per-halaman melakukan
batch-dengan-cursor akan merusak kasus sederhana yang sudah jalan. Rencananya
dua orkestrator di atas komponen bersama, dengan `SurveyReport` sebagai router.

---

## 8. Status: terbukti vs belum

### Terbukti terukur

| Klaim | Bukti |
|---|---|
| Surveyor andal pada dialeknya | 200 halaman Lampiran I.F PP 28/2025 (sektor yang belum pernah diuji): 1 span, **0 kegagalan grid** |
| Kecepatan memadai | 104 ms/halaman → ~20 menit untuk 11.591 halaman |
| Deteksi awal baris | 5/5 tepat pada rentang uji; 41/42 pada uji primitif |
| Segmentasi span | Batas batang tubuh↔lampiran dan Lampiran I↔II benar |
| Degradasi mulus | PDF digital, slide landscape, bagan organisasi: `teks`, nol peringatan |
| Mesin tumpukan lintas batch | 4 dokumen berstruktur sangat berbeda, termasuk tabel pivot, tanpa modifikasi kode |
| Cursor berukuran tetap | 347–644 byte di semua dokumen uji |
| Auditor menemukan kesalahan nyata | Salah ketik Rp1,5 miliar di Perda Kerinci 1/2008 |
| Tes regresi | 12/12 lulus (`tests/test_hierarchy.py`) |

### Belum terbukti

- **Belum ada satu pun panggilan VLM sungguhan pada jalur B.** Peran Reader/Scribe
  dikerjakan manual saat pengujian. Rancangannya terbukti utuh; akurasinya belum
  terukur.
- **Separuh pipeline tabel belum pernah jalan ujung ke ujung.** Agent penentu
  schema, model delta OPEN/APPEND/COMMIT, dan Briefer belum ditulis.
- **Tidak ada gold set.** Tanpa halaman berlabel manual, tidak ada satu angka
  akurasi isi pun yang bisa disebut.
- **Panel verifikator masih hipotesis.** Uji pertama justru negatif: sekat yang
  benar pada PP 28/2025 tetap memotong kata 21% karena bounding box OCR
  menyeberang batas sel, sementara hipotesis yang kurang sekat lolos 0%.

---

## 9. Batas yang diketahui

Surveyor hanya andal pada **satu dialek tabel**: scan bilevel bergrid, satu tabel
per halaman, satu baris satu record. Di luar itu ia gagal — dan tambalan ambang
yang membetulkan satu dokumen merusak dokumen lain.

Terbukti pada `kupdf.net_tabel-praktis-uud-1945.pdf`:

| Asumsi yang patah | Gejala |
|---|---|
| Header tidak bergabung | Baris BAB membentang beberapa kolom Pasal → hanya 5 dari 8 sekat terdeteksi |
| Satu tabel per halaman | Dua tabel bertumpuk; aturan "celah besar pertama" merentang menutupi keduanya |
| `get_drawings()` memberi garis tabel | Word hanya menerbitkan rect untuk **sel berwarna**; sekat kolom interior muncul di 1–3 dari 18 baris |
| Satu baris = satu record | Tabel pivot: kolom = Pasal, baris = Ayat. Bahkan dengan grid benar, briefing akan salah |

Percobaan perbaikan (pecah blok + ukur sekat di badan) membetulkan UUD menjadi
8 dan 7 kolom, tetapi membuat halaman teks biasa Raperda melaporkan **88 kolom**.
Tidak dimasukkan ke repo.

Arah perbaikannya bukan menyetel ambang, melainkan: hipotesis bersaing +
panel verifikator dengan penilaian relatif, dialek tabel sebagai parameter yang
ditemukan sekali per span (bukan konstanta global), dan karantina untuk halaman
yang tidak lolos verifikasi. Detailnya belum ditulis di sini karena belum diuji.

---

## 10. Langkah berikutnya

Berurut menurut nilai per usaha:

1. **Gold set** — 2 baris utuh Lampiran I (±20 halaman) dan 20 halaman batang
   tubuh, dilabeli manual. Sehari kerja, dan itu bedanya antara berharap dan tahu.
2. **Briefer** — perakit briefing yang menyatukan `SurveyReport` + cursor menjadi
   konteks yang benar-benar dikirim ke agent.
3. **Panel verifikator** — dua pemeriksa (potong-kata + cakupan-kolom), lalu ukur
   apakah penilaian relatif memilih kandidat yang tepat pada empat dokumen yang
   sudah ada. Kalau tidak diskriminatif, seluruh rancangan dialek tidak berguna —
   dan lebih baik tahu sebelum menulisnya.
