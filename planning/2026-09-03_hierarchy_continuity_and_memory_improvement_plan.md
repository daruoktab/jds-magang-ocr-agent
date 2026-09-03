# Rencana Pengembangan: Arsitektur Kontinuitas Hierarki Linear & Terukur Secara Otomatis (Auto-Scaling & Quality-First)
**Tanggal:** 3 September 2026  
**Status:** Dokumen Perencanaan Teknis (*Technical Architecture Plan — Auto-Scaling Revision*)  
**Prinsip Utama:** **KUALITAS UTAMA (*Quality-First*) & PENSKALAAN OTOMATIS SECARA LINEAR (*Linear Auto-Scaling Architecture*)**  
**Cakupan Dokumen:** Dari Dokumen Ringkas (5–20 Halaman), Menengah (50–100 Halaman), hingga Ribuan Halaman (1.000+ Halaman) dengan Beban Token yang Beradaptasi Dinamis Sesuai Kapasitas Hardware / Context Window Model.  
**Target Komponen:** `app/config.py`, `app/multi_page.py`, `app/pdf.py`, `app/prompts.py`, `app/graph.py`, `app/deep_agent.py`

---

## 1. Visi Arsitektur: Linear, Otomatis, & Bebas Hardcode

Angka "2.000 token" yang dibahas sebelumnya adalah representasi **baseline estimasi**. Pada kenyataannya, dokumen di dunia industri sangat bervariasi:
* Dokumen 10 halaman (memo/jurnal) tidak membutuhkan 2.000 token hierarki.
* Dokumen 89 halaman (*PIC16F84A*) sangat optimal di ~1.500–2.500 token hierarki.
* Dokumen 1.000+ halaman (manual spesifikasi, regulasi, standar hukum) membutuhkan investasi hardware/model ber-context window lebih besar (32k–128k), sehingga jatah budget konteks hierarki harus **bisa otomatis naik secara linear** untuk mempertahankan detail struktural buku secara utuh.

### Nilai Inti yang Dibangun:
1. **Zero Hardcoded Constraint:** Besaran token hierarki tidak dipaku mati pada angka statis, melainkan dikendalikan oleh **`HierarchyContextGovernor`** yang elastis.
2. **Linear Auto-Scaling:** Budget dan kedalaman outline menyesuaikan secara linear terhadap:
   - Kapasitas total context window model aktif (`context_window`).
   - Panjang total dokumen dan progres halaman saat ini (`current_page / total_pages`).
   - Kepadatan hierarki (*heading density*).
3. **Stateless Execution:** Berapapun tebalnya dokumen (hingga ribuan halaman), eksekusi per-halaman tetap murni *stateless* (tidak ada kebocoran VRAM/memory leak), sementara kesinambungan struktur dijembatani oleh *dynamic outline payload*.

---

## 2. Formula Penskalaan Otomatis (*Auto-Scaling Formulation*)

Modul pengatur konteks (`HierarchyContextGovernor`) menghitung jatah budget token per halaman secara dinamis menggunakan formula adaptif:

```
                                  ┌─────────────────────────────────────────────────┐
                                  │      KAPASITAS CONTEXT WINDOW MODEL (CW)        │
                                  │   (misal: 8.192, 16.384, 32.768, atau 128.000)   │
                                  └──────────────────────┬──────────────────────────┘
                                                         │
                                                         ▼
                                          [Rasio Alokasi Hierarki: 15% - 25%]
                                                         │
                                                         ▼
                                        ┌─────────────────────────────────┐
                                        │ BUDGET ELASTIS (AdaptiveBudget) │
                                        └────────────────┬────────────────┘
                                                         │
                    ┌────────────────────────────────────┼────────────────────────────────────┐
                    ▼                                    ▼                                    ▼
       [Pilar 1: Peta Struktur]             [Pilar 2: Breadcrumb & State]         [Pilar 3: Trailing Buffer]
       Alokasi: 40% dari Budget               Alokasi: 10% dari Budget             Alokasi: 50% dari Budget
  (Adaptive Hierarchical Folding)               (Jalur Aktif & Counter)              (2-4 Paragraf Lengkap)
```

### Rumus Matematis Runtime Budget:

$$\text{Budget}_{\text{runtime}} = \text{clamp}\Big(\alpha \times \text{Window}_{\text{VLM}}, \;\; \text{MinBudget}, \;\; \text{MaxBudgetCap}\Big)$$

* **$\alpha$ (Rasio Hierarki):** Default `0.20` (20% dari total context window model).
* **$\text{MinBudget}$:** Default `800` token (menjamin kualitas dokumen pendek tetap prima).
* **$\text{MaxBudgetCap}$:** Dapat diatur via env `HIERARCHY_MAX_TOKEN_BUDGET` (default `4096` token, atau tidak terbatas jika user memiliki modal hardware besar untuk dokumen ribuan halaman).

### Contoh Perilaku Skalabilitas Otomatis Berdasarkan Model & Dokumen:

| Kasus Dokumen & Model | Kapasitas Window Model | Jatah Budget Konteks Hierarki | Karakteristik Output Konteks |
| :--- | :--- | :--- | :--- |
| **Dokumen 10 Hal (SLM 8k Window)** | 8.192 token | **~1.200 token** | Zero-pruning, outline lengkap dari halaman 1–10, 2 paragraf ekor. |
| **Dokumen 89 Hal (PIC16F84A, 16k Window)** | 16.384 token | **~2.500 token** | Resolusi H1–H3 tajam untuk seluruh bab terkait, 3 paragraf ekor. |
| **Dokumen 500 Hal (VLM 32k Window)** | 32.768 token | **~5.000 token** | Outline pohon mencakup ratusan sub-bab, counter tabel/gambar global. |
| **Dokumen 2.000+ Hal (Enterprise 128k)** | 131.072 token | **~15.000 token** | Kerangka utuh satu buku tebal termuat lengkap di prompt tanpa folding kasar. |

---

## 3. Algoritma *Hierarchical Distance Folding* (Penskalaan Pohon Bab)

Jika sebuah dokumen memiliki ribuan halaman namun berjalan pada model dengan window terbatas, sistem menerapkan algoritma **Linear Distance Degradation**:

```
Tingkat Kedalaman Detail (Resolution)
  ▲
100% │ [ZONA 1: Bab Aktif Saat Ini]  --> Resolusi Penuh (H1, H2, H3, H4)
     │
 60% │ [ZONA 2: Bab Berdampingan]    --> Resolusi Menengah (H1, H2 saja)
     │
 20% │ [ZONA 3: Bab Lampau / Jauh]    --> Milestone Ringkas (H1 + Range Hal)
  0% └─────────────────────────────────────────────────────────────► Jarak Halaman (Distance Δ)
```

### Logika Algoritmik:
1. **Zona 1 (Bab Aktif):** Seluruh seksi di bawah Bab yang sedang berjalan dipertahankan 100% (hingga H3/H4).
2. **Zona 2 (Bab Tetangga, $\Delta \le 2$ Bab):** Ditampilkan hingga level H2 agar transisi antar-bab mulus.
3. **Zona 3 (Bab Lampau, $\Delta > 2$ Bab):** Otomatis di-ringkas (*folded*) menjadi satu baris penanda milestone:
   ```markdown
   # 1.0 GENERAL OVERVIEW (Hal 1-18) [Selesai]
   # 2.0 PIN CONFIGURATION & PORTS (Hal 19-35) [Selesai]
   # 3.0 MEMORY ARCHITECTURE (Hal 36-saat ini) [Aktif]
     ## 3.1 Program Memory
     ## 3.2 Data RAM & SFR
       ### 3.2.1 Status Register
   ```
Algoritma ini menjamin bahwa **pohon heading selalu muat dalam alokasi budget berapapun tebalnya buku**.

---

## 4. Desain Modul & Konfigurasi Sistem

### A. Penambahan Konfigurasi di `app/config.py`
Menjadikan seluruh parameter terukur dan dapat dikonfigurasi melalui *environment variables*:

```python
# app/config.py
HIERARCHY_AUTO_SCALE: bool = _bool_env("HIERARCHY_AUTO_SCALE", "true")
HIERARCHY_BUDGET_RATIO: float = _float_env("HIERARCHY_BUDGET_RATIO", "0.20")
HIERARCHY_MIN_BUDGET: int = _int_env("HIERARCHY_MIN_BUDGET", "800")
HIERARCHY_MAX_BUDGET: int = _int_env("HIERARCHY_MAX_BUDGET", "4096")
VLM_CONTEXT_WINDOW: int = _int_env("VLM_CONTEXT_WINDOW", "16384")
```

### B. Arsitektur Engine: `HierarchyContextGovernor` di `app/multi_page.py`
```python
class HierarchyContextGovernor:
    """
    Governor cerdas untuk menghitung, melacak, dan menskalakan konteks hierarki
    secara otomatis dan linear dari 1 halaman hingga 10.000+ halaman.
    """
    def __init__(
        self,
        context_window: int = 16384,
        budget_ratio: float = 0.20,
        min_budget: int = 800,
        max_budget: int = 4096,
    ):
        self.context_window = context_window
        self.target_budget = int(
            min(max_budget, max(min_budget, context_window * budget_ratio))
        )
        self.outline_store: list[OutlineNode] = []
        self.active_chapter: OutlineNode | None = None
        self.active_path: list[str] = []
        self.last_counters: dict[str, Any] = {"list": None, "table": None, "figure": None}
        self.trailing_buffer: str = ""

    def register_page_result(self, page_number: int, markdown_text: str) -> None:
        """Parse dan update pohon struktur secara incremental."""
        ...

    def generate_adaptive_prompt_context(self, current_page: int) -> str:
        """
        Rakit payload prompt dengan algoritma distance folding
        sehingga tepat mengisi target_budget secara efisien.
        """
        ...
```

---

## 5. Roadmap Implementasi Bertahap

### Fase 1: Governor & Formula Scaling Engine
* [ ] Tambahkan konfigurasi `HIERARCHY_*` pada [`app/config.py`](file:///d:/Codings/jds-magang/app/config.py).
* [ ] Bangun `HierarchyContextGovernor` di [`app/multi_page.py`](file:///d:/Codings/jds-magang/app/multi_page.py) dengan AST parser dan formula budget dinamis.
* [ ] Unit test di `tests/test_hierarchy_scaling.py`:
  - Uji penskalaan elastis: simulasi model window 8k vs 32k vs 128k.
  - Uji distance folding: simulasi dokumen 100 halaman dan 1.000 halaman tetap patuh pada target budget.

### Fase 2: Integrasi Pipeline Multi-Halaman
* [ ] Hubungkan `HierarchyContextGovernor` ke loop dokumen multi-halaman di [`app/pdf.py`](file:///d:/Codings/jds-magang/app/pdf.py).
* [ ] Hubungkan ke modul presentasi di [`app/ppt.py`](file:///d:/Codings/jds-magang/app/ppt.py).
* [ ] Hubungkan format prompt adaptif ke [`app/prompts.py`](file:///d:/Codings/jds-magang/app/prompts.py).

### Fase 3: Post-Processing Reconciliation & Guardrails
* [ ] Tambahkan verifikasi silang pada tahap akhir (`stitch_pages_to_markdown`) untuk memastikan heading yang dihasilkan tidak ada yang bertentangan dengan master tree outline.
* [ ] Normalisasi section numbering otomatis (`1.0` $\rightarrow$ `#`, `1.1` $\rightarrow$ `##`, `1.1.1` $\rightarrow$ `###`).

### Fase 4: Validasi & Audit Mutu Kode
* [ ] Uji coba ekstraksi dokumen nyata berskala menengah (`PIC16F84A.pdf`).
* [ ] Verifikasi kestabilan konsumsi token di LM Studio / local server.
* [ ] Jalankan audit kode: `ruff check --fix ; ty check`.

---

## 6. Kesimpulan

Dengan perencanaan ini, sistem kita memiliki fondasi yang **linear, elastis, dan scalable secara otomatis**:
1. **Tidak Ada Angka Kaku:** Budget secara dinamis menyesuaikan kemampuan model dan ketebalan dokumen.
2. **Kesiapan Dokumen Ribuan Halaman:** Ketika user memproses dokumen ribuan halaman dan menaikkan context window (misal ke 32k atau 128k), sistem secara otomatis memanfaatkan kapasitas ekstra tersebut untuk memperkaya detail outline tanpa perlu penulisan ulang kode.
3. **Kualitas Ekstraksi Berkelas Enterprise:** Keteraturan hierarki (#, ##, ###), penomoran bab, kelanjutan tabel, dan kalimat gantung selalu terjaga dengan presisi tertinggi.
