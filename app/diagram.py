"""
Modul Spesialis Diagram & Visual Artifacts untuk Ekstraksi Vision VLM -> Mermaid.js.

Fitur:
  1. Klasifikasi kelayakan konversi diagram ke sintaks Mermaid.js.
  2. Ekstraksi visual diagram ke kode Mermaid (flowchart, sequence, ERD, class, state, mindmap, block architecture, dsb).
  3. Sanitasi dan validasi sintaks Mermaid.js.
  4. Fallback ke representasi deskriptif terstruktur jika diagram tidak cocok untuk Mermaid (grafik statistik kontinu, peta spasial, skematik sirkuit mikro, foto).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import cast

from langchain_core.language_models.chat_models import BaseChatModel

from .llm import encode_image_to_base64 as encode_image_base64
from .schemas import (
    DiagramConvertibilityResult,
    DiagramExtractionResult,
    DiagramFormatRecommendation,
    DiagramTypeLiteral,
)

logger = logging.getLogger(__name__)

MERMAID_KEYWORDS: tuple[str, ...] = (
    "flowchart",
    "graph",
    "sequencediagram",
    "classdiagram",
    "statediagram",
    "erdiagram",
    "mindmap",
    "gantt",
    "pie",
    "gitgraph",
    "c4context",
    "block-beta",
)


def sanitize_mermaid_code(raw_text: str) -> str | None:
    """
    Ekstrak dan bersihkan blok kode Mermaid dari output model AI.
    Menghilangkan wrapper Markdown (```mermaid ... ```) dan memperbaiki anomali formatting umum.
    """
    if not raw_text or not raw_text.strip():
        return None

    cleaned = raw_text.strip()

    # Ekstrak dari blok markdown ```mermaid ... ```
    mermaid_match = re.search(r"```(?:mermaid)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if mermaid_match:
        cleaned = mermaid_match.group(1).strip()

    # Periksa apakah baris pertama diawali kata kunci Mermaid yang valid
    lines = [line for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return None

    first_line = lines[0].strip().lower().replace(" ", "").replace("-", "")
    is_valid_header = any(first_line.startswith(kw.replace("-", "")) for kw in MERMAID_KEYWORDS)

    if not is_valid_header:
        # Coba perbaiki jika ada preamble teks sebelum diagram
        for idx, line in enumerate(lines):
            line_clean = line.strip().lower().replace(" ", "").replace("-", "")
            if any(line_clean.startswith(kw.replace("-", "")) for kw in MERMAID_KEYWORDS):
                cleaned = "\n".join(lines[idx:]).strip()
                break
        else:
            logger.warning("[Diagram:Sanitize] Tidak ditemukan header Mermaid valid.")
            return None

    # Sanitasi karakter yang sering merusak parsing Mermaid di web UI
    cleaned = cleaned.replace("\r\n", "\n")
    return cleaned


def validate_mermaid_syntax(mermaid_code: str) -> tuple[bool, str | None]:
    """
    Validasi dasar sintaks Mermaid secara statis / heuristik.
    Mengembalikan (is_valid, error_message).
    """
    if not mermaid_code or not mermaid_code.strip():
        return False, "Kode Mermaid kosong"

    lines = [l.strip() for l in mermaid_code.splitlines() if l.strip()]
    if not lines:
        return False, "Kode Mermaid tidak memiliki baris instruksi"

    # Periksa header
    first_line = lines[0].lower().replace(" ", "").replace("-", "")
    if not any(first_line.startswith(kw.replace("-", "")) for kw in MERMAID_KEYWORDS):
        return False, f"Header Mermaid tidak dikenali: '{lines[0]}'"

    # Cek keseimbangan kurung atau simbol relasi umum
    open_brackets = mermaid_code.count("[")
    close_brackets = mermaid_code.count("]")
    open_braces = mermaid_code.count("{")
    close_braces = mermaid_code.count("}")
    open_parens = mermaid_code.count("(")
    close_parens = mermaid_code.count(")")

    # Toleransi kecil jika ada label khusus, namun beri warning jika selisih banyak
    if abs(open_brackets - close_brackets) > 2:
        return False, f"Ketidakseimbangan tanda kurung siku: [{open_brackets} vs ]{close_brackets}"
    if abs(open_braces - close_braces) > 2:
        return False, f"Ketidakseimbangan kurung kurawal: {{{open_braces}}} vs {{{close_braces}}}"
    if abs(open_parens - close_parens) > 2:
        return False, f"Ketidakseimbangan tanda kurung biasa: ({open_parens} vs ){close_parens}"

    return True, None


def get_diagram_recommendation(diagram_type: DiagramTypeLiteral) -> DiagramFormatRecommendation:
    """Berikan rekomendasi format ekstraksi berdasarkan kategori diagram."""
    mermaid_compatible = {
        "flowchart": ("flowchart TD", "mermaid_code", "Cocok untuk alur kerja terstruktur"),
        "sequence_diagram": ("sequenceDiagram", "mermaid_code", "Cocok untuk urutan pesan/proses"),
        "class_diagram": ("classDiagram", "mermaid_code", "Cocok untuk hierarki kelas OOP"),
        "state_diagram": ("stateDiagram-v2", "mermaid_code", "Cocok untuk transisi state finite"),
        "er_diagram": ("erDiagram", "mermaid_code", "Cocok untuk skema relasi entitas/database"),
        "mindmap": ("mindmap", "mermaid_code", "Cocok untuk hierarki konsep & taksonomi"),
        "gantt_chart": ("gantt", "mermaid_code", "Cocok untuk jadwal & lini masa proyek"),
        "block_architecture": ("block-beta", "mermaid_code", "Cocok untuk diagram blok arsitektur"),
        "git_graph": ("gitGraph", "mermaid_code", "Cocok untuk visualisasi alur branching git"),
        "generic_diagram": ("flowchart LR", "mermaid_code", "Diagram umum, gunakan representasi flowchart"),
    }

    if diagram_type in mermaid_compatible:
        syntax, fmt, rationale = mermaid_compatible[diagram_type]
        return DiagramFormatRecommendation(
            diagram_type=diagram_type,
            recommended_format=fmt,
            is_mermaid_compatible=True,
            suggested_syntax=syntax,
            rationale=rationale,
        )

    unsuitable_reasons = {
        "unsuitable_statistical_chart": "Grafik statistik kontinu (bar, line, scatter) lebih baik diekstrak ke tabel data Markdown dan deskripsi tren.",
        "unsuitable_map_or_spatial": "Peta geografis atau spasial tidak dapat diwakili oleh simpul graf Mermaid.",
        "unsuitable_photo_or_illustration": "Foto atau ilustrasi artistik memerlukan deskripsi visual naratif.",
        "unsuitable_complex_schematic": "Skematik sirkuit elektrik mikro atau CAD terlalu rumit untuk abstraksi Mermaid.",
        "non_diagram": "Konten berupa teks standar atau halaman polos.",
    }

    rationale = unsuitable_reasons.get(
        diagram_type,
        "Format visual tidak kompatibel dengan generator kode diagram relasional.",
    )
    return DiagramFormatRecommendation(
        diagram_type=diagram_type,
        recommended_format="text_description",
        is_mermaid_compatible=False,
        suggested_syntax=None,
        rationale=rationale,
    )


def classify_diagram_convertibility(
    image_path: str | Path,
    llm: BaseChatModel,
) -> DiagramConvertibilityResult:
    """
    Evaluasi visual: Apakah gambar mengandung diagram yang cocok dikonversi ke kode Mermaid?
    Mengembalikan `DiagramConvertibilityResult` berisi keputusan konvertibilitas dan format yang dianjurkan.
    """
    path_obj = Path(image_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"File citra diagram tidak ditemukan: {path_obj}")

    b64_img = encode_image_base64(path_obj)

    prompt = (
        "Analisis gambar ini dengan teliti untuk mengevaluasi apakah gambar ini berisi DIAGRAM yang cocok "
        "dikonversi ke sintaks kode diagram relasional (Mermaid.js).\n\n"
        "Kategori diagram yang SANGAT COCOK untuk Mermaid:\n"
        "  - Flowchart / Alur Proses (flowchart TD/LR)\n"
        "  - Sequence Diagram / Interaksi Pesan (sequenceDiagram)\n"
        "  - Entity Relationship Diagram / ERD (erDiagram)\n"
        "  - Class Diagram / OOP Hierarchy (classDiagram)\n"
        "  - State Machine / State Diagram (stateDiagram-v2)\n"
        "  - Mindmap / Pohon Konsep (mindmap)\n"
        "  - Block Architecture / Blok Sistem Komponen (block-beta / flowchart)\n"
        "  - Gantt Chart / Timeline (gantt)\n\n"
        "Kategori yang TIDAK COCOK untuk Mermaid:\n"
        "  - Grafik statistik data kuantitatif / kurva / bar chart (unsuitable_statistical_chart)\n"
        "  - Peta spasial / denah / geografis (unsuitable_map_or_spatial)\n"
        "  - Foto / Gambar ilustrasi bebas (unsuitable_photo_or_illustration)\n"
        "  - Skematik sirkuit mikro / blueprint CAD (unsuitable_complex_schematic)\n"
        "  - Dokumen teks biasa tanpa diagram (non_diagram)\n\n"
        "Berikan output HANYA dalam format JSON valid berikut tanpa penjelasan tambahan:\n"
        "{\n"
        '  "is_convertible": true/false,\n'
        '  "diagram_type": "flowchart" | "sequence_diagram" | "class_diagram" | "state_diagram" | "er_diagram" | "mindmap" | "gantt_chart" | "block_architecture" | "unsuitable_statistical_chart" | "unsuitable_map_or_spatial" | "unsuitable_photo_or_illustration" | "unsuitable_complex_schematic" | "non_diagram",\n'
        '  "recommended_format": "mermaid" | "text_description" | "markdown_table",\n'
        '  "mermaid_type": "flowchart" | "sequenceDiagram" | "erDiagram" | "classDiagram" | "stateDiagram-v2" | "mindmap" | "gantt" | "block-beta" | null,\n'
        '  "confidence": 0.0 - 1.0,\n'
        '  "reasoning": "Alasan singkat dalam bahasa Indonesia",\n'
        '  "nodes_or_entities": ["Entitas1", "Entitas2", ...]\n'
        "}"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_img}"},
                },
            ],
        }
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
                    recommended_format=str(rec_fmt),
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
                mermaid_type="flowchart",
                confidence=0.8,
                reasoning="Terdeteksi pola alur proses/flowchart dari analisis teks model.",
            )
        if "sequence" in lower_resp or "urutan" in lower_resp:
            return DiagramConvertibilityResult(
                is_convertible=True,
                diagram_type="sequence_diagram",
                recommended_format="mermaid",
                mermaid_type="sequenceDiagram",
                confidence=0.8,
                reasoning="Terdeteksi urutan interaksi aktor/komponen.",
            )

        return DiagramConvertibilityResult(
            is_convertible=False,
            diagram_type="generic_diagram",
            recommended_format="text_description",
            mermaid_type=None,
            confidence=0.6,
            reasoning="Diagram tidak memiliki struktur relasional diskrit yang terbukti cocok untuk Mermaid.",
        )

    except Exception:
        logger.exception(
            "[Diagram:Classify] Gagal mengklasifikasikan gambar diagram '%s'",
            path_obj.name,
        )
        return DiagramConvertibilityResult(
            is_convertible=False,
            diagram_type="generic_diagram",
            recommended_format="text_description",
            mermaid_type=None,
            confidence=0.0,
            reasoning="Terjadi error saat memanggil Vision LLM classifier.",
        )


def extract_diagram_to_mermaid(
    image_path: str | Path,
    llm: BaseChatModel,
    *,
    forced_diagram_type: str | None = None,
) -> DiagramExtractionResult:
    """
    Ekstrak citra diagram menjadi kode Mermaid.js terstruktur yang siap dirender,
    atau deskripsi teks komprehensif jika diagram tidak cocok untuk Mermaid.
    """
    path_obj = Path(image_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"File citra tidak ditemukan: {path_obj}")

    b64_img = encode_image_base64(path_obj)

    # 1. Evaluasi kelayakan jika tidak dipaksa
    convertibility: DiagramConvertibilityResult | None = None
    if not forced_diagram_type:
        convertibility = classify_diagram_convertibility(path_obj, llm)
        if not convertibility.is_convertible:
            desc_prompt = (
                "Gambar ini berisi elemen visual/grafik/diagram yang tidak cocok dijadikan diagram kode Mermaid. "
                "Berikan deskripsi terstruktur mengenai gambar ini mencakup:\n"
                "1. Judul / Tema Visual\n"
                "2. Komponen / Data Utama yang digambarkan\n"
                "3. Tren / Hubungan / Kesimpulan utama dari gambar\n"
                "Sajikan dalam format Markdown bersih."
            )
            desc_resp = llm.invoke(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": desc_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64_img}"
                                },
                            },
                        ],
                    }
                ]
            )
            return DiagramExtractionResult(
                status="unsuitable",
                is_mermaid=False,
                diagram_type=convertibility.diagram_type,
                mermaid_code=None,
                text_description=str(desc_resp.content).strip(),
                text_summary=str(desc_resp.content).strip(),
                reasoning=convertibility.reasoning,
                convertibility=convertibility,
            )

    # 2. Prompt Ekstraksi Mermaid.js
    syntax_hint = ""
    if forced_diagram_type:
        syntax_hint = f"Paksa gunakan tipe diagram: {forced_diagram_type}."
    elif convertibility and convertibility.mermaid_type:
        syntax_hint = f"Gunakan tipe diagram Mermaid: {convertibility.mermaid_type}."

    extract_prompt = (
        "Tugas Anda adalah mengekstrak seluruh elemen diagram, simpul (nodes), label teks, dan relasi berarah "
        "dari gambar ini menjadi KODE MERMAID.JS YANG VALID dan LENGKAP.\n\n"
        f"{syntax_hint}\n\n"
        "Panduan Pembuatan Kode Mermaid:\n"
        "1. Mulai dengan deklarasi tipe diagram di baris pertama (contoh: 'flowchart TD', 'sequenceDiagram', 'erDiagram').\n"
        "2. Gunakan ID simpul alfanumerik yang bersih (misal: A, B, node1, Step1) dan letakkan label teks asli di dalam kurung siku [Teks], lingkaran ((Teks)), atau belah ketupat {Teks}.\n"
        "3. Pertahankan semua label relasi/panah secara akurat (contoh: A -->|Label| B).\n"
        "4. Hindari karakter khusus yang merusak sintaks (gunakan tanda kutip dua untuk teks kompleks di dalam label).\n"
        "5. Berikan output di dalam blok markdown ```mermaid\\n...\\n```.\n"
        "6. Tambahkan ringkasan 1-2 kalimat di bawah blok kode yang menjelaskan arti diagram tersebut."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": extract_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_img}"},
                },
            ],
        }
    ]

    try:
        resp = llm.invoke(messages)
        text_resp = str(resp.content).strip()

        # Ekstrak blok mermaid
        mermaid_match = re.search(
            r"```(?:mermaid)?\s*([\s\S]*?)\s*```", text_resp, re.IGNORECASE
        )
        mermaid_block: str | None = None
        summary_text: str | None = None

        if mermaid_match:
            mermaid_block = sanitize_mermaid_code(mermaid_match.group(1))
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

        diag_type = cast(
            DiagramTypeLiteral,
            forced_diagram_type
            or (convertibility.diagram_type if convertibility else "flowchart"),
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
            diagram_type=cast(DiagramTypeLiteral, forced_diagram_type or "generic_diagram"),
            mermaid_code=None,
            text_summary="Gagal melakukan ekstraksi diagram visual.",
            reasoning="Terjadi error saat ekstraksi visual diagram.",
            convertibility=convertibility,
        )


__all__ = [
    "MERMAID_KEYWORDS",
    "classify_diagram_convertibility",
    "extract_diagram_to_mermaid",
    "get_diagram_recommendation",
    "sanitize_mermaid_code",
    "validate_mermaid_syntax",
]
