"""
Registry agent ekstraksi - SATU agent untuk SATU JENIS DOKUMEN.

Setiap `ExtractionAgent` punya prompt tuning untuk jenis dokumennya, plus opsi
model dan schema Pydantic sendiri. Menambah jenis baru = tambah satu entri di
`AGENT_REGISTRY`; tidak ada kode lain yang perlu berubah.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Type

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from .extractor import VisionExtractor
from .prompts import (
    GENERAL_USER_PROMPT,
    SYSTEM_EXTRACTOR,
    _PROMPT_BANK_STATEMENT,
    _PROMPT_BUSINESS_CARD,
    _PROMPT_FORM,
    _PROMPT_INVOICE,
    _PROMPT_LABEL,
    _PROMPT_RECEIPT,
    _PROMPT_SCREENSHOT,
    _PROMPT_TABLE,
)
from .schemas import DocumentExtraction


@dataclass
class ExtractionAgent:
    """Agent ekstraksi untuk satu jenis dokumen."""

    name: str
    description: str
    user_prompt: str
    model: Optional[str] = None
    schema: Type[BaseModel] = DocumentExtraction
    system_prompt: str = SYSTEM_EXTRACTOR

    def build(self, llm: BaseChatModel) -> VisionExtractor:
        return VisionExtractor(
            llm=llm,
            schema=self.schema,
            user_prompt=self.user_prompt,
            system_prompt=self.system_prompt,
        )

    def run(self, image_path: str, llm: BaseChatModel) -> BaseModel:
        return self.build(llm).extract(image_path)


AGENT_REGISTRY: Dict[str, ExtractionAgent] = {
    "receipt": ExtractionAgent(
        name="receipt",
        description="Struk belanja / nota kasir (daftar item + total bayar)",
        user_prompt=_PROMPT_RECEIPT,
    ),
    "invoice": ExtractionAgent(
        name="invoice",
        description="Faktur / invoice (tagihan penjualan, item, pajak)",
        user_prompt=_PROMPT_INVOICE,
    ),
    "table": ExtractionAgent(
        name="table",
        description="Tabel data murni (baris dan kolom)",
        user_prompt=_PROMPT_TABLE,
    ),
    "form": ExtractionAgent(
        name="form",
        description="Formulir isian (field + nilai, checkbox)",
        user_prompt=_PROMPT_FORM,
    ),
    "business_card": ExtractionAgent(
        name="business_card",
        description="Kartu nama (nama, jabatan, perusahaan, kontak)",
        user_prompt=_PROMPT_BUSINESS_CARD,
    ),
    "bank_statement": ExtractionAgent(
        name="bank_statement",
        description="Lembaran mutasi rekening bank (daftar transaksi)",
        user_prompt=_PROMPT_BANK_STATEMENT,
    ),
    "label": ExtractionAgent(
        name="label",
        description="Label produk / kemasan (nama, bahan, kadaluarsa, barcode)",
        user_prompt=_PROMPT_LABEL,
    ),
    "screenshot": ExtractionAgent(
        name="screenshot",
        description="Tangkapan layar aplikasi/web/chat",
        user_prompt=_PROMPT_SCREENSHOT,
    ),
    "generic": ExtractionAgent(
        name="generic",
        description="Dokumen lain / tidak termasuk kategori di atas",
        user_prompt=GENERAL_USER_PROMPT,
    ),
}


def get_agent(name: str) -> ExtractionAgent:
    """Ambil agent berdasarkan nama jenis. Tak dikenal -> agent generic."""
    return AGENT_REGISTRY.get(name, AGENT_REGISTRY["generic"])
