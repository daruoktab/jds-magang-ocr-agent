"""
Modul Analisis & Ekstraksi Diagram Multimodal ke Mermaid.js.

Mendeteksi dan mengevaluasi secara selektif apakah diagram pada dokumen cocok
dikonversi menjadi kode Mermaid yang valid (flowchart, sequence, ERD, state, class, mindmap, block architecture),
atau lebih tepat disajikan dalam bentuk deskripsi terstruktur / tabel Markdown (grafik kontinu numerik, peta, foto, skematik mikro).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .llm import image_data_uri
from .preprocess import preprocess_image
from .schemas import (
    DiagramConvertibilityResult,
    DiagramExtractionResult,
    DiagramFormatRecommendation,
    DiagramTypeLiteral,
)

logger = logging.getLogger("app.diagram")

# ==============================================================================
# Prompts
# ==============================================================================

DIAGRAM_CLASSIFY_SYSTEM = (
    "Kamu adalah AI spesialis evaluasi visual diagram dan arsitektur Mermaid.js. "
    "Tugasmu: Menganalisis gambar dan mengevaluasi secara ketat dan selektif apakah visual "
    "tersebut cocok dikonversi menjadi kode diagram Mermaid yang valid."
)

DIAGRAM_CLASSIFY_PROMPT = """Analisis elemen visual/diagram pada gambar ini secara objektif.

Evaluasi apakah diagram ini COCOK untuk dikonversi menjadi kode Mermaid.js:

Kategori yang COCOK untuk Mermaid:
1. `flowchart`: Alur proses bisnis, pohon keputusan, workflow logika (`flowchart TD` / `LR`).
2. `sequence_diagram`: Interaksi antar aktor/komponen/sistem/API (`sequenceDiagram`).
3. `class_diagram`: Diagram kelas UML, struktur atribut & method (`classDiagram`).
4. `state_diagram`: Finite state machine, transisi status (`stateDiagram-v2`).
5. `er_diagram`: Entity Relationship Diagram, entitas & relasi kunci asing (`erDiagram`).
6. `mindmap`: Peta konsep hierarkis bertingkat (`mindmap`).
7. `gantt_chart`: Linimasa proyek, jadwal fase & milestone (`gantt`).
8. `block_architecture`: Arsitektur sistem/komponen blok terhubung (`flowchart TB/LR` dengan subgraph).
9. `git_graph`: Alur percabangan git branch & commit (`gitGraph`).

Kategori yang TIDAK COCOK untuk Mermaid (JANGAN paksakan ke Mermaid):
- `unsuitable_statistical_chart`: Grafik data numerik kontinu padat (scatter plot, line chart banyak titik, histogram, boxplot, radar chart, heatmap). Format rekomendasi: 'markdown_table' atau 'text_description'.
- `unsuitable_map_or_spatial`: Peta geografis, denah ruangan (floorplan), visualisasi spasial. Format rekomendasi: 'text_description'.
- `unsuitable_photo_or_illustration`: Foto realistis, gambar anatomi biologis, rendering 3D, ilustrasi seni. Format rekomendasi: 'text_description'.
- `unsuitable_complex_schematic`: Skematik sirkuit elektrik mikro, gambar teknik CAD mekanik rumit. Format rekomendasi: 'text_description'.
- `non_diagram`: Dokumen teks murni, formulir polos, atau tabel tanpa bagan visual. Format rekomendasi: 'none' atau 'text_description'.

Outputkan HANYA JSON persis dengan format berikut:
{
  "is_convertible": true/false,
  "diagram_type": "<flowchart|sequence_diagram|class_diagram|state_diagram|er_diagram|mindmap|gantt_chart|block_architecture|git_graph|unsuitable_statistical_chart|unsuitable_map_or_spatial|unsuitable_photo_or_illustration|unsuitable_complex_schematic|non_diagram|generic_diagram>",
  "recommended_format": "<mermaid|markdown_table|text_description|none>",
  "mermaid_type": "<mis. 'flowchart TD', 'sequenceDiagram', 'erDiagram', 'stateDiagram-v2', 'classDiagram', 'mindmap', 'gantt' atau null>",
  "confidence": 0.0 - 1.0,
  "reasoning": "<penjelasan ringkas mengapa cocok atau tidak cocok>",
  "nodes_or_entities": ["<nama node/entitas utama>", ...]
}
"""

DIAGRAM_EXTRACTION_SYSTEM = (
    "Kamu adalah AI ahli visualisasi diagram dan sintaks Mermaid.js tingkat mahir. "
    "Tugasmu: Mengekstrak diagram visual dari gambar menjadi kode Mermaid yang 100% VALID dan bersih, "
    "disertai penjelasan kontekstual yang jelas."
)

DIAGRAM_EXTRACTION_PROMPT = """Ekstrak diagram dari gambar ini menjadi kode Mermaid.js yang valid dan terstruktur.

Aturan Penting Sintaks Mermaid:
1. Gunakan tipe diagram yang tepat (mis. `flowchart TD`, `flowchart LR`, `sequenceDiagram`, `classDiagram`, `erDiagram`, `stateDiagram-v2`, `mindmap`, `gantt`).
2. Berikan identifier node yang bersih (alfanumerik tanpa spasi, misal `node_A`, `step_1`, `DB_User`).
3. Selalu beri tanda kutip ganda pada label teks yang mengandung spasi, tanda kurung, simbol, atau tanda baca (mis. `node_1["Langkah 1 (Input Data)"]`).
4. JANGAN gunakan tag HTML `<br>` atau formatting yang dapat merusak renderer Mermaid CLI.
5. Jika ada sub-alur atau modul terpisah, kelompokkan menggunakan `subgraph Judul ... end`.
6. Jika visual ternyata bukan diagram yang cocok (mis. grafik statistik numerik kontinu), berikan ringkasan tabel Markdown atau deskripsi struktural dalam blockquote alih-alih kode Mermaid palsu.

Format Output yang Diharapkan:
```mermaid
<kode mermaid di sini>
```

**Ringkasan Diagram:**
<Deskripsi singkat makna, alur utama, dan entitas dalam diagram>
"""


# ==============================================================================
# Helper Functions & Syntax Sanitizer
# ==============================================================================

MERMAID_KEYWORDS = (
    "flowchart",
    "graph",
    "sequencediagram",
    "classdiagram",
    "erdiagram",
    "statediagram",
    "mindmap",
    "gantt",
    "gitgraph",
    "pie",
    "quadrantchart",
    "journey",
)


def sanitize_mermaid_code(raw_code: str) -> str:
    """
    Sanitasi dan bersihkan kode Mermaid agar valid dan tidak menyebabkan error render.
    """
    cleaned = raw_code.strip()

    # Ekstrak dari blok ```mermaid jika ada
    match = re.search(
        r"```(?:mermaid)?\s*\n(.*?)\n\s*```", cleaned, re.DOTALL | re.IGNORECASE
    )
    if match:
        cleaned = match.group(1).strip()

    lines = cleaned.splitlines()
    filtered_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Hapus sisa markdown formatting pengantar jika ada
        if stripped.startswith("```"):
            continue
        filtered_lines.append(line)

    code_body = "\n".join(filtered_lines).strip()
    return f"```mermaid\n{code_body}\n```"


# ==============================================================================
# Core Functions
# ==============================================================================


def classify_diagram_convertibility(
    image_path: str | Path,
    llm: BaseChatModel,
) -> DiagramConvertibilityResult:
    """
    Analisis gambar dokumen/diagram dan tentukan apakah cocok dikonversi ke kode Mermaid.
    """
    path_obj = Path(image_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"File gambar tidak ditemukan: {path_obj}")

    proc = preprocess_image(path_obj)
    data_uri = image_data_uri(proc.processed_path)

    content: list[dict[str, Any]] = [
        {"type": "text", "text": DIAGRAM_CLASSIFY_PROMPT},
        {"type": "image_url", "image_url": {"url": data_uri}},
    ]
    messages = [
        SystemMessage(content=DIAGRAM_CLASSIFY_SYSTEM),
        HumanMessage(content=cast(Any, content)),
    ]

    try:
        resp = llm.invoke(messages)
        text_resp = str(resp.content).strip()
        logger.debug("[Diagram:Classify] Respon mentah VLM: %s", text_resp)

        # Cari JSON blok
        match = re.search(r"\{.*\}", text_resp, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                dtype = data.get("diagram_type", "generic_diagram")
                rec_fmt = data.get("recommended_format", "text_description")
                is_conv = bool(data.get("is_convertible", False))
                mtype = data.get("mermaid_type")
                conf = float(data.get("confidence", 0.9))
                reason = data.get("reasoning", "")
                nodes = data.get("nodes_or_entities", [])

                return DiagramConvertibilityResult(
                    is_convertible=is_conv,
                    diagram_type=cast(DiagramTypeLiteral, dtype),
                    recommended_format=cast(DiagramFormatRecommendation, rec_fmt),
                    mermaid_type=mtype,
                    confidence=conf,
                    reasoning=reason,
                    nodes_or_entities=nodes if isinstance(nodes, list) else [],
                )
            except (json.JSONDecodeError, ValueError) as json_err:
                logger.warning(
                    "[Diagram:Classify] Gagal decode JSON (%s): %s", json_err, text_resp
                )

        # Fallback heuristik berbasis teks
        lower_resp = text_resp.lower()
        if (
            "flowchart" in lower_resp
            or "alur" in lower_resp
            or "workflow" in lower_resp
        ):
            return DiagramConvertibilityResult(
                is_convertible=True,
                diagram_type="flowchart",
                recommended_format="mermaid",
                mermaid_type="flowchart TD",
                reasoning="Terdeteksi alur proses / flowchart dari respon VLM",
            )
        if "sequence" in lower_resp:
            return DiagramConvertibilityResult(
                is_convertible=True,
                diagram_type="sequence_diagram",
                recommended_format="mermaid",
                mermaid_type="sequenceDiagram",
                reasoning="Terdeteksi alur pesan / sequence diagram",
            )
        if (
            "er_diagram" in lower_resp
            or "erd" in lower_resp
            or "entity relationship" in lower_resp
        ):
            return DiagramConvertibilityResult(
                is_convertible=True,
                diagram_type="er_diagram",
                recommended_format="mermaid",
                mermaid_type="erDiagram",
                reasoning="Terdeteksi relasi entitas basis data / ERD",
            )
        if "mindmap" in lower_resp:
            return DiagramConvertibilityResult(
                is_convertible=True,
                diagram_type="mindmap",
                recommended_format="mermaid",
                mermaid_type="mindmap",
                reasoning="Terdeteksi peta konsep / mindmap hierarkis",
            )

        return DiagramConvertibilityResult(
            is_convertible=False,
            diagram_type="non_diagram",
            recommended_format="text_description",
            mermaid_type=None,
            reasoning="Tidak terdeteksi struktur diagram yang cocok untuk Mermaid",
        )
    except Exception:
        logger.exception("[Diagram:Classify] Error saat memanggil VLM")
        return DiagramConvertibilityResult(
            is_convertible=False,
            diagram_type="generic_diagram",
            recommended_format="text_description",
            mermaid_type=None,
            reasoning="Gagal mengevaluasi kelayakan diagram",
        )


def extract_diagram_to_mermaid(
    image_path: str | Path,
    llm: BaseChatModel,
    *,
    forced_diagram_type: str | None = None,
) -> DiagramExtractionResult:
    """
    Ekstrak gambar diagram visual menjadi kode Mermaid yang valid atau deskripsi terstruktur.
    """
    path_obj = Path(image_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"File gambar tidak ditemukan: {path_obj}")

    # 1. Evaluasi kelayakan jika tidak dipaksa
    convertibility: DiagramConvertibilityResult | None = None
    if not forced_diagram_type:
        convertibility = classify_diagram_convertibility(path_obj, llm)
        if not convertibility.is_convertible:
            logger.info(
                "[Diagram:Extract] Visual '%s' tidak cocok untuk Mermaid (tipe: %s, alasan: %s)",
                path_obj.name,
                convertibility.diagram_type,
                convertibility.reasoning,
            )
            # Jika tidak cocok, buat deskripsi struktural langsung
            return DiagramExtractionResult(
                status="unsuitable",
                is_mermaid=False,
                diagram_type=convertibility.diagram_type,
                mermaid_code=None,
                text_summary=f"> **[Visual Note - {convertibility.diagram_type}]:** {convertibility.reasoning}",
                reasoning=convertibility.reasoning,
                convertibility=convertibility,
            )

    # 2. Ekstraksi kode Mermaid via VLM
    proc = preprocess_image(path_obj)
    data_uri = image_data_uri(proc.processed_path)

    custom_prompt = DIAGRAM_EXTRACTION_PROMPT
    if forced_diagram_type or (convertibility and convertibility.mermaid_type):
        chosen_type = forced_diagram_type or (
            convertibility.mermaid_type if convertibility else "flowchart TD"
        )
        custom_prompt += f"\n\nFormat diagram yang disarankan: `{chosen_type}`."

    content: list[dict[str, Any]] = [
        {"type": "text", "text": custom_prompt},
        {"type": "image_url", "image_url": {"url": data_uri}},
    ]
    messages = [
        SystemMessage(content=DIAGRAM_EXTRACTION_SYSTEM),
        HumanMessage(content=cast(Any, content)),
    ]

    try:
        resp = llm.invoke(messages)
        text_resp = str(resp.content).strip()

        # Pisahkan kode mermaid dan deskripsi/summary
        mermaid_match = re.search(
            r"```(?:mermaid)?\s*\n(.*?)\n\s*```", text_resp, re.DOTALL | re.IGNORECASE
        )
        mermaid_block: str | None = None
        summary_text: str = ""

        if mermaid_match:
            raw_mermaid = mermaid_match.group(1).strip()
            mermaid_block = sanitize_mermaid_code(raw_mermaid)
            # Ambil sisa teks sebagai summary
            summary_text = text_resp.replace(mermaid_match.group(0), "").strip()
        else:
            # Periksa apakah seluruh respon atau bagian respon dimulai dengan kata kunci Mermaid
            stripped_resp = text_resp.strip()
            first_line = (
                stripped_resp.splitlines()[0].strip().lower() if stripped_resp else ""
            )
            if any(first_line.startswith(kw) for kw in MERMAID_KEYWORDS):
                mermaid_block = sanitize_mermaid_code(stripped_resp)
                summary_text = "Diagram Mermaid berhasil diekstrak."
            else:
                summary_text = text_resp

        diag_type = forced_diagram_type or (
            convertibility.diagram_type if convertibility else "flowchart"
        )

        return DiagramExtractionResult(
            status="success" if mermaid_block else "unsuitable",
            is_mermaid=bool(mermaid_block),
            diagram_type=diag_type,
            mermaid_code=mermaid_block,
            text_summary=summary_text
            or f"Diagram tipe {diag_type} berhasil diekstrak.",
            reasoning=convertibility.reasoning
            if convertibility
            else "Ekstraksi visual berhasil dijalankan.",
            convertibility=convertibility,
        )

    except Exception:
        logger.exception(
            "[Diagram:Extract] Gagal mengekstrak Mermaid dari '%s'",
            path_obj.name,
        )
        return DiagramExtractionResult(
            status="error",
            is_mermaid=False,
            diagram_type=forced_diagram_type or "generic_diagram",
            mermaid_code=None,
            text_summary="Gagal melakukan ekstraksi diagram visual.",
            reasoning="Terjadi error saat ekstraksi visual diagram.",
            convertibility=convertibility,
        )
