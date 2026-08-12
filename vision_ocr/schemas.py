"""
Schema Pydantic untuk hasil ekstraksi dokumen.

Modular: secara default hasil ditampung sebagai struktur bebas
(DocumentExtraction) karena model yang memutuskan bentuk representasinya.
Jika Anda butuh schema yang kaku (mis. untuk integrasi dengan sistem lain),
buat model Pydantic sendiri dan teruskan ke VisionExtractor/create_extractor
melalui parameter `schema` - pipeline tidak berubah.
"""
from typing import Any, Dict

from pydantic import BaseModel, Field


class DocumentExtraction(BaseModel):
    """
    Hasil ekstraksi generik.

    - doc_type : jenis dokumen, diputuskan sendiri oleh VLM
      (mis. "receipt", "table", "form", "business_card").
    - data     : struktur bebas hasil ekstraksi - seluruh isi dokumen
      direpresentasikan model dalam format yang dianggapnya paling tepat
      (dict bersarang, list, atau kombinasi).
    """

    doc_type: str = Field(description="Jenis dokumen yang diputuskan model")
    data: Dict[str, Any] = Field(
        description="Struktur bebas hasil ekstraksi, ditentukan oleh model"
    )
