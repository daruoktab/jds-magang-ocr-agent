"""
Pipeline ekstraksi: gambar dokumen -> VLM -> Markdown Bersih Siap Chunking.
Mendukung multi-spesifikasi karakteristik tata letak dokumen secara komposit dengan logging transparan.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .llm import image_data_uri
from .prompts import (
    CLASSIFY_PROMPT,
    CLASSIFY_SYSTEM,
    SYSTEM_DOCUMENT_EXTRACTOR,
    build_extraction_prompt,
    normalize_specs,
)

logger = logging.getLogger("app.extractor")


class VisionExtractor:
    """Ekstraktor dokumen multimodal: Mengubah gambar dokumen menjadi Markdown siap chunking."""

    def __init__(
        self,
        llm: BaseChatModel,
        system_prompt: str = SYSTEM_DOCUMENT_EXTRACTOR,
    ) -> None:
        self.llm: BaseChatModel = llm
        self.system_prompt: str = system_prompt

    def classify(self, image_path: str) -> list[str]:
        """
        Klasifikasikan satu atau lebih karakteristik layout dokumen yang ada pada gambar:
        ['plain'], ['bilingual_journal', 'markdown_hierarchy'], dsb.
        """
        logger.debug(
            "[Extractor:Classify] Menyiapkan payload klasifikasi untuk: %s", image_path
        )
        content: list[dict[str, Any]] = [
            {"type": "text", "text": CLASSIFY_PROMPT},
            {"type": "image_url", "image_url": {"url": image_data_uri(image_path)}},
        ]
        messages = [
            SystemMessage(content=CLASSIFY_SYSTEM),
            HumanMessage(content=cast(Any, content)),
        ]

        resp = self.llm.invoke(messages)
        text_resp = str(resp.content).strip()
        logger.debug("[Extractor:Classify] Respon mentah VLM: %s", text_resp)

        # Parse JSON output jika ada
        match = re.search(r"\{.*?\}", text_resp, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                raw_specs = data.get("specs") or [data.get("doc_type")]
                normalized = normalize_specs(raw_specs)
                logger.info(
                    "[Extractor:Classify] Berhasil parse JSON layout: %s", normalized
                )
                return normalized
            except (json.JSONDecodeError, TypeError, ValueError, KeyError) as e:
                logger.warning(
                    "[Extractor:Classify] Gagal parse JSON layout (%s): '%s'",
                    e,
                    match.group(0),
                )

        # Fallback multi-matching via regex
        text_lower = text_resp.lower()
        detected: list[str] = []
        if "journal" in text_lower or "bilingual" in text_lower or "2col" in text_lower:
            detected.append("bilingual_journal")
        if (
            "hierarchy" in text_lower
            or "markdown" in text_lower
            or "heading" in text_lower
        ):
            detected.append("markdown_hierarchy")
        if "slide" in text_lower or "presentation" in text_lower or "ppt" in text_lower:
            detected.append("presentation_slides")

        final_specs = detected if detected else ["plain"]
        logger.info("[Extractor:Classify] Fallback regex layout: %s", final_specs)
        return final_specs

    def inspect_page(self, image_path: str) -> dict[str, Any]:
        """
        Inspeksi komprehensif layout dokumen & elemen visual (diagram, tabel, hierarki).
        Mengembalikan dict berisi specs, has_diagram, diagram_type, dan has_table.
        """
        inspect_prompt = (
            "Analisis gambar dokumen ini secara menyeluruh untuk mendeteksi karakteristik tata letak dan elemen visual.\n\n"
            "Evaluasi hal berikut:\n"
            "1. specs: Karakteristik dokumen (pilih dari: 'plain', 'markdown_hierarchy', 'bilingual_journal', 'presentation_slides').\n"
            "2. has_diagram: true jika terdapat diagram visual (flowchart, alur proses, sequence diagram, ERD, arsitektur blok, mindmap, state diagram, org chart), false jika hanya teks biasa atau foto polos.\n"
            "3. diagram_type: Tipe diagram jika has_diagram=true (contoh: 'flowchart', 'sequence_diagram', 'er_diagram', 'block_architecture', 'mindmap', dll., atau null jika tidak ada).\n"
            "4. has_table: true jika terdapat tabel data/baris kolom.\n\n"
            "Outputkan HANYA format JSON valid tanpa pengantar:\n"
            "{\n"
            '  "specs": ["spec1", "spec2"],\n'
            '  "has_diagram": true/false,\n'
            '  "diagram_type": "string" atau null,\n'
            '  "has_table": true/false\n'
            "}"
        )

        content: list[dict[str, Any]] = [
            {"type": "text", "text": inspect_prompt},
            {"type": "image_url", "image_url": {"url": image_data_uri(image_path)}},
        ]
        messages = [
            SystemMessage(content=CLASSIFY_SYSTEM),
            HumanMessage(content=cast(Any, content)),
        ]

        try:
            resp = self.llm.invoke(messages)
            text_resp = str(resp.content).strip()
            match = re.search(r"\{.*?\}", text_resp, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                raw_specs = data.get("specs") or [data.get("doc_type")]
                norm_specs = normalize_specs(raw_specs)
                has_diag = bool(data.get("has_diagram", False))
                diag_type = data.get("diagram_type")
                has_tbl = bool(data.get("has_table", False))
                return {
                    "specs": norm_specs,
                    "has_diagram": has_diag,
                    "diagram_type": diag_type,
                    "has_table": has_tbl,
                }
        except Exception as e:  # noqa: BLE001
            logger.warning("[Extractor:Inspect] Gagal inspect JSON (%s), fallback ke classify biasa.", e)

        # Fallback
        specs = self.classify(image_path)
        return {
            "specs": specs,
            "has_diagram": "presentation_slides" in specs,
            "diagram_type": "generic_diagram" if "presentation_slides" in specs else None,
            "has_table": False,
        }

    def judge_and_refine(
        self,
        image_path: str,
        draft_markdown: str,
        *,
        specs: list[str] | str | None = None,
    ) -> str:
        """
        Tahap Aggregator Judge & Self-Correction (Koreksi Ulang):
        Membandingkan draft gabungan Markdown (teks + Mermaid + tabel) terhadap citra asli dokumen.
        Memperbaiki kekurangan atau mempertahankan draft jika sudah akurat dan lengkap.
        """
        if not draft_markdown or not draft_markdown.strip():
            return draft_markdown

        judge_prompt = (
            "Anda adalah AI Chief Quality Auditor & Document Aggregator Verifier.\n\n"
            "Tugas Anda: Bandingkan DRAFT MARKDOWN di bawah ini dengan GAMBAR ASLI DOKUMEN.\n\n"
            "Evaluasi dan lakukan koreksi ulang dengan panduan:\n"
            "1. KELENGKAPAN: Pastikan seluruh teks, judul, poin-poin, dan data angka pada gambar telah tercakup dalam Markdown.\n"
            "2. DIAGRAM VISUAL (MERMAID): Jika pada gambar terdapat diagram alur/relasi/arsitektur/proses, pastikan sudah direpresentasikan dengan blok kode ```mermaid yang valid dan lengkap atau deskripsi terstruktur.\n"
            "3. INTEGRITAS TABEL: Pastikan tabel diformat sebagai tabel Markdown (GFM) yang rapi.\n"
            "4. KEBERSIHAN: Hapus duplikasi atau ketidakkonsistenan antarseksi.\n"
            "5. JIKA DRAFT SUDAH BENAR DAN LENGKAP: Kembalikan teks Markdown tersebut secara presisi tanpa merusak format.\n\n"
            f"[DRAFT MARKDOWN SEBELUM KOREKSI]:\n'''markdown\n{draft_markdown}\n'''\n\n"
            "Outputkan HANYA teks Markdown hasil perbaikan akhir tanpa basa-basi pengantar atau penutup."
        )

        content: list[dict[str, Any]] = [
            {"type": "text", "text": judge_prompt},
            {"type": "image_url", "image_url": {"url": image_data_uri(image_path)}},
        ]
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=cast(Any, content)),
        ]

        try:
            resp = self.llm.invoke(messages)
            refined_md = str(resp.content).strip()
            # Bersihkan wrapper code fence jika ada
            if refined_md.startswith("```markdown") and refined_md.endswith("```"):
                refined_md = refined_md[len("```markdown") : -3].strip()
            elif refined_md.startswith("```md") and refined_md.endswith("```"):
                refined_md = refined_md[len("```md") : -3].strip()
            elif (
                refined_md.startswith("```")
                and refined_md.endswith("```")
                and not refined_md.startswith("```mermaid")
            ):
                refined_md = refined_md[3:-3].strip()

            return refined_md if refined_md else draft_markdown
        except Exception as e:  # noqa: BLE001
            logger.warning("[Extractor:Judge] Terjadi kesalahan pada tahap judge & refine (%s). Menggunakan draft awal.", e)
            return draft_markdown

    def extract_markdown(
        self,
        image_path: str,
        *,
        specs: list[str] | str | None = None,
        previous_page_context: str | None = None,
    ) -> str:
        """
        Ekstrak gambar dokumen menjadi teks Markdown bersih sesuai spesifikasi layout komposit.

        Args:
            image_path: Path ke file gambar dokumen.
            specs: Satu atau daftar karakteristik dokumen ('plain', 'markdown_hierarchy', 'bilingual_journal', 'presentation_slides').
            previous_page_context: Konteks halaman sebelumnya untuk menjaga kontinuitas header.

        Returns:
            Teks Markdown terstruktur siap chunking.
        """
        user_prompt = build_extraction_prompt(
            specs=specs,
            previous_page_context=previous_page_context,
        )

        logger.debug(
            "[Extractor:Markdown] Menyusun prompt ekstraksi (panjang prompt: %d karakter, konteks lalu: %s)",
            len(user_prompt),
            bool(previous_page_context),
        )

        content: list[dict[str, Any]] = [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": image_data_uri(image_path)}},
        ]

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=cast(Any, content)),
        ]

        response = self.llm.invoke(messages)
        md_text = str(response.content).strip()

        # Bersihkan pembungkus markdown block ```markdown ... ``` jika VLM membungkusnya
        if md_text.startswith("```markdown") and md_text.endswith("```"):
            md_text = md_text[len("```markdown") : -3].strip()
        elif md_text.startswith("```md") and md_text.endswith("```"):
            md_text = md_text[len("```md") : -3].strip()
        elif md_text.startswith("```") and md_text.endswith("```"):
            md_text = md_text[3:-3].strip()

        return md_text
