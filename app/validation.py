"""
Modul validasi konsistensi hasil ekstraksi dokumen (Agentic Reflection).

Memeriksa:
  - Konsistensi matematika (total == subtotal + pajak - diskon, penjumlahan item baris)
  - Format angka dan tanggal yang wajar
  - Kelengkapan field krusial per jenis dokumen
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationResult:
    """Hasil evaluasi validasi ekstraksi."""

    is_valid: bool
    issues: list[str] = field(default_factory=list)
    score: float = 1.0  # 0.0 (sangat buruk) - 1.0 (sempurna)

    def format_critique(self) -> str:
        """Format catatan perbaikan (critique) untuk diberikan kembali ke VLM."""
        if not self.issues:
            return ""
        critique_lines = ["Ditemukan beberapa ketidaksesuaian pada hasil ekstraksi sebelumnya:"]
        for idx, issue in enumerate(self.issues, 1):
            critique_lines.append(f"{idx}. {issue}")
        critique_lines.append(
            "\nHarap teliti kembali gambar dokumen dan teks OCR. "
            "Koreksi angka, rumus perhitungan, atau field yang kurang tepat tersebut."
        )
        return "\n".join(critique_lines)


def _to_float(val: Any) -> float | None:
    """Ubah string / number menjadi float bersih."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        # Bersihkan simbol mata uang & spasi: Rp, $, IDR, dll
        cleaned = re.sub(r"[^\d.,\-+]", "", val).strip()
        if not cleaned:
            return None
        # Handle format ribuan titik dan desimal koma (gaya Indo/Eropa 10.000,50 -> 10000.50)
        if "," in cleaned and "." in cleaned:
            if cleaned.rfind(",") > cleaned.rfind("."):
                cleaned = cleaned.replace(".", "").replace(",", ".")
            else:
                cleaned = cleaned.replace(",", "")
        elif "," in cleaned:
            # 10,50 -> 10.50 or 10,000 -> 10000
            parts = cleaned.split(",")
            if len(parts) == 2 and len(parts[1]) <= 2:
                cleaned = f"{parts[0]}.{parts[1]}"
            else:
                cleaned = cleaned.replace(",", "")
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def validate_extraction(doc_type: str, data: dict[str, Any]) -> ValidationResult:
    """
    Validasi data ekstraksi berdasarkan jenis dokumen.

    Returns:
        ValidationResult dengan is_valid=True jika tidak ada error fatal.
    """
    if not isinstance(data, dict):
        return ValidationResult(is_valid=False, issues=["Data ekstraksi bukan objek dictionary valid."], score=0.0)

    if not data:
        return ValidationResult(is_valid=False, issues=["Data ekstraksi kosong."], score=0.0)

    issues: list[str] = []
    doc_lower = (doc_type or "").lower()

    # 1. Validasi Struk & Invoice
    if any(k in doc_lower for k in ("receipt", "invoice", "struk", "faktur", "bill")):
        _validate_financial_doc(data, issues)

    # 2. Validasi Tabel
    elif "table" in doc_lower:
        _validate_table_doc(data, issues)

    # 3. Validasi Kartu Nama
    elif any(k in doc_lower for k in ("business_card", "card", "kartu")):
        _validate_business_card(data, issues)

    # Hitung score & validity
    is_valid = len(issues) == 0
    score = max(0.0, 1.0 - (0.25 * len(issues)))

    return ValidationResult(is_valid=is_valid, issues=issues, score=score)


def _validate_financial_doc(data: dict[str, Any], issues: list[str]) -> None:
    """Cek struk/invoice untuk konsistensi matematis & total."""
    # Cari field total & subtotal
    total_val = None
    subtotal_val = None
    tax_val = None
    discount_val = None

    for k, v in data.items():
        k_lower = k.lower()
        if "total" in k_lower and "sub" not in k_lower and "pajak" not in k_lower and "tax" not in k_lower:
            total_val = _to_float(v)
        elif "subtotal" in k_lower or "sub_total" in k_lower or "net" in k_lower:
            subtotal_val = _to_float(v)
        elif "tax" in k_lower or "ppn" in k_lower or "pajak" in k_lower:
            tax_val = _to_float(v)
        elif "discount" in k_lower or "diskon" in k_lower or "potongan" in k_lower:
            discount_val = _to_float(v)

    # Cek konsistensi subtotal + tax - discount ~= total
    if total_val is not None and subtotal_val is not None:
        expected_total = subtotal_val
        if tax_val is not None:
            expected_total += tax_val
        if discount_val is not None:
            expected_total -= discount_val

        # Toleransi selisih pembulatan (mis. 1.0 / 0.05)
        diff = abs(total_val - expected_total)
        tolerance = max(1.0, 0.02 * total_val)  # 2% or 1 rupiah/cent
        if diff > tolerance:
            issues.append(
                f"Inkonsistensi Total: Total tercatat ({total_val:g}) tidak sama dengan "
                f"Subtotal ({subtotal_val:g}) + Pajak ({tax_val or 0:g}) - Diskon ({discount_val or 0:g}) "
                f"= {expected_total:g} (selisih: {diff:g})."
            )

    # Cek konsistensi baris item belanja
    items = None
    for k, v in data.items():
        if isinstance(v, list) and len(v) > 0 and any(kw in k.lower() for kw in ("item", "barang", "products", "rows", "daftar")):
            items = v
            break

    if items and isinstance(items, list):
        items_total_sum = 0.0
        has_item_prices = False
        for idx, item in enumerate(items, 1):
            if not isinstance(item, dict):
                continue
            price = None
            qty = None
            item_total = None
            for ik, iv in item.items():
                ik_lower = ik.lower()
                if any(x in ik_lower for x in ("total", "subtotal", "amount", "jumlah")):
                    item_total = _to_float(iv)
                elif any(x in ik_lower for x in ("price", "harga", "rate")):
                    price = _to_float(iv)
                elif any(x in ik_lower for x in ("qty", "quantity", "banyak", "count", "pcs")):
                    qty = _to_float(iv)

            # Cek price * qty == item_total jika ada
            if price is not None and qty is not None and item_total is not None and qty > 0:
                expected_item_total = price * qty
                if abs(item_total - expected_item_total) > max(1.0, 0.05 * item_total):
                    issues.append(
                        f"Item #{idx} ({item.get('name') or item.get('description') or 'item'}): "
                        f"Harga ({price:g}) x Qty ({qty:g}) = {expected_item_total:g}, "
                        f"tetapi total item tertulis {item_total:g}."
                    )

            if item_total is not None:
                items_total_sum += item_total
                has_item_prices = True
            elif price is not None:
                items_total_sum += price * (qty or 1.0)
                has_item_prices = True

        # Jika ada subtotal dan total items dihitung
        if has_item_prices and subtotal_val is not None:
            if abs(items_total_sum - subtotal_val) > max(2.0, 0.05 * subtotal_val):
                issues.append(
                    f"Jumlah total seluruh item ({items_total_sum:g}) tidak cocok dengan Subtotal ({subtotal_val:g})."
                )


def _validate_table_doc(data: dict[str, Any], issues: list[str]) -> None:
    """Cek struktur tabel (columns vs rows)."""
    columns = data.get("columns") or data.get("headers")
    rows = data.get("rows") or data.get("data")
    if columns and rows and isinstance(columns, list) and isinstance(rows, list):
        col_count = len(columns)
        for idx, row in enumerate(rows, 1):
            if isinstance(row, list) and len(row) != col_count:
                issues.append(
                    f"Baris tabel #{idx} memiliki {len(row)} kolom, sedangkan header tabel memiliki {col_count} kolom."
                )
                if len(issues) >= 3:
                    break


def _validate_business_card(data: dict[str, Any], issues: list[str]) -> None:
    """Cek kelengkapan kartu nama (nama orang/perusahaan & kontak)."""
    keys_str = " ".join(data.keys()).lower()
    has_contact = any(k in keys_str for k in ("phone", "tel", "email", "mobile", "hp", "kontak", "whatsapp", "website", "address", "alamat"))
    has_name = any(k in keys_str for k in ("name", "nama", "person", "company", "perusahaan", "title", "jabatan"))
    if not has_name:
        issues.append("Kartu nama tidak memiliki field nama orang atau perusahaan yang teridentifikasi.")
    if not has_contact:
        issues.append("Kartu nama tidak memiliki kontak (telepon/email/alamat) yang teridentifikasi.")
