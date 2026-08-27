"""
Modul Pemrosesan Data Tabular Transaksional ke Database SQLite & Double-Verification.

Fitur:
  1. Deteksi & pemisahan tabel transaksional (rekening koran, log mutasi keuangan, ledger, invoice)
     vs tabel naratif / matriks kualitatif.
  2. Parsing tabel Markdown GFM dengan pembersihan & normalisasi angka (IDR/USD/EUR, pemisah ribuan).
  3. Inferensi skema kolom SQL dinamis (TEXT, REAL, INTEGER, DATE) tervalidasi Pydantic.
  4. Penyimpanan ke SQLite lokal (`output/databases/{nama_dokumen}.sqlite`).
  5. Mekanisme Double-Verification (Verifikasi Ganda):
     - Pengecekan integritas baris (row count match).
     - Validasi keberadaan kolom & tipe data numerik.
     - Pengujian query agregasi (SUM, AVG, COUNT).
     - Pengujian kontinuitas aritmatika saldo (Balance[n-1] + Kredit - Debit = Balance[n]).
     - Refleksi model untuk mengonfirmasi kesesuaian data.
  6. Eksekusi query SQL dengan presisi 100%.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from .schemas import (
    TableClassificationResult,
    TableColumnSchema,
    TableIngestionResult,
    TableSchema,
    TableTypeLiteral,
    TableVerificationReport,
    TabularQueryResult,
    VerificationCheck,
)

# Kata kunci header yang mengindikasikan tabel transaksional / finansial
TRANSACTIONAL_HEADER_KEYWORDS: set[str] = {
    "date",
    "tanggal",
    "posting date",
    "value date",
    "description",
    "deskripsi",
    "keterangan",
    "transaksi",
    "transaction",
    "reference",
    "ref no",
    "cheque",
    "debit",
    "debit amount",
    "credit",
    "credit amount",
    "kredit",
    "balance",
    "saldo",
    "amount",
    "nominal",
    "total",
    "subtotal",
    "price",
    "harga",
    "qty",
    "quantity",
    "unit",
    "rate",
    "tax",
    "pajak",
    "fee",
    "biaya",
    "mutasi",
}

# ==============================================================================
# Helper Parsing & Normalisasi Nilai
# ==============================================================================


def sanitize_identifier(name: str) -> str:
    """Bersihkan nama tabel atau kolom agar valid sebagai identifier SQL."""
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", name.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"col_{cleaned}"
    return cleaned


def clean_cell_text(cell: str) -> str:
    """Bersihkan teks sel dari spasi berlebih dan karakter markdown formatting."""
    return re.sub(r"\s+", " ", cell).strip()


def parse_numeric_value(val_str: str) -> float | int | None:
    """
    Ekstrak nilai numerik dari string (mata uang, pemisah ribuan, format negatif akuntansi).
    Mendukung format:
      - Standar US: 1,234,567.89
      - Standar Indo/Eropa: 1.234.567,89
      - Format negatif akuntansi: (1,234.50) -> -1234.50
      - Nilai dengan simbol mata uang: Rp 50.000, $120.50, IDR 1,000,000
    """
    if not val_str or not isinstance(val_str, str):
        return None

    raw = clean_cell_text(val_str)
    if not raw or raw in ("-", "--", "N/A", "NA", "null", "None"):
        return None

    # Cek format negatif akuntansi dalam tanda kurung, misal (1,234.50)
    is_negative = False
    if raw.startswith("(") and raw.endswith(")"):
        is_negative = True
        raw = raw[1:-1].strip()
    elif raw.startswith("-"):
        is_negative = True
        raw = raw[1:].strip()

    # Bersihkan simbol mata uang & karakter non-angka kecuali titik, koma, minus
    cleaned = re.sub(r"[^\d.,]", "", raw).strip()
    if not cleaned:
        return None

    # Format Internasional: 1,234,567.89 (koma ribuan, titik desimal)
    if re.match(r"^\d{1,3}(,\d{3})*(\.\d+)?$", cleaned):
        try:
            num = float(cleaned.replace(",", ""))
            return -num if is_negative else num
        except ValueError:
            pass

    # Format Eropa/Indonesia: 1.234.567,89 (titik ribuan, koma desimal)
    if re.match(r"^\d{1,3}(\.\d{3})*(,\d+)?$", cleaned):
        try:
            num = float(cleaned.replace(".", "").replace(",", "."))
            return -num if is_negative else num
        except ValueError:
            pass

    # Angka polos atau desimal standar: 12345.67
    if re.match(r"^\d+(\.\d+)?$", cleaned):
        try:
            num = float(cleaned)
            return -num if is_negative else num
        except ValueError:
            pass

    return None


def parse_date_value(val_str: str) -> str | None:
    """
    Cek dan parsing format tanggal umum (YYYYMMDD, YYYY-MM-DD, DD.MM.YYYY, DD/MM/YYYY).
    Mengembalikan format ISO 'YYYY-MM-DD' atau None jika bukan tanggal.
    """
    if not val_str or not isinstance(val_str, str):
        return None
    raw = clean_cell_text(val_str).strip()

    # Format YYYYMMDD (misal: 20240201)
    if re.match(r"^(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])$", raw):
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"

    # Format YYYY-MM-DD
    if re.match(r"^(19|20)\d{2}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$", raw):
        return raw

    # Format DD.MM.YYYY atau DD/MM/YYYY atau DD-MM-YYYY
    m = re.match(
        r"^(0[1-9]|[12]\d|3[01])[./-]((0[1-9]|1[0-2]))[./-]((19|20)\d{2})$", raw
    )
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"

    return None


def parse_markdown_tables(markdown_text: str) -> list[dict[str, Any]]:
    """
    Ekstrak dan parse seluruh tabel format GFM Markdown dari teks dokumen.
    Mengembalikan daftar objek berisi header asli, baris terurai, dan teks konteks di atas tabel.
    """
    lines = markdown_text.splitlines()
    tables: list[dict[str, Any]] = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Deteksi awal tabel Markdown: baris yang diawali '|' dan baris berikutnya adalah separator '|---|'
        if line.startswith("|") and line.endswith("|") and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if next_line.startswith("|") and re.match(
                r"^\|(\s*:?-+:?\s*\|)+$", next_line
            ):
                # Header row
                header_raw = [c.strip() for c in line.strip("|").split("|")]
                rows_raw: list[list[str]] = []

                # Konteks teks sebelum tabel (3 baris sebelumnya)
                context_lines = [
                    lines[k].strip()
                    for k in range(max(0, i - 3), i)
                    if lines[k].strip()
                ]
                context = " ".join(context_lines)

                # Baca baris data
                j = i + 2
                while j < len(lines):
                    row_line = lines[j].strip()
                    if not row_line.startswith("|") or not row_line.endswith("|"):
                        break
                    row_cells = [c.strip() for c in row_line.strip("|").split("|")]
                    if len(row_cells) < len(header_raw):
                        row_cells.extend([""] * (len(header_raw) - len(row_cells)))
                    elif len(row_cells) > len(header_raw):
                        row_cells = row_cells[: len(header_raw)]
                    rows_raw.append(row_cells)
                    j += 1

                if rows_raw:
                    tables.append(
                        {
                            "headers": header_raw,
                            "rows": rows_raw,
                            "context": context,
                            "line_start": i,
                            "line_end": j,
                        }
                    )
                i = j
                continue
        i += 1

    return tables


# ==============================================================================
# Classifier: Heuristik & AI Classification
# ==============================================================================


def classify_table_heuristic(
    headers: list[str],
    rows: list[list[str]],
    context: str = "",
) -> TableClassificationResult:
    """
    Klasifikasi heuristik untuk menentukan apakah tabel bertipe transaksional / finansial
    (yang harus masuk SQLite) atau tabel naratif (bisa di-chunking ke Vector RAG).
    """
    total_cols = len(headers)
    total_rows = len(rows)

    if total_cols == 0 or total_rows == 0:
        return TableClassificationResult(
            table_id="empty",
            is_transactional=False,
            table_type="generic_table",
            recommended_storage="vector_rag",
            confidence=1.0,
            reasoning="Tabel kosong.",
            numeric_density=0.0,
            date_density=0.0,
            total_rows=0,
            total_columns=0,
            columns_detected=[],
        )

    # 1. Analisis Header
    matched_keywords = []
    for h in headers:
        clean_h = sanitize_identifier(h).replace("_", " ").lower()
        for kw in TRANSACTIONAL_HEADER_KEYWORDS:
            if kw in clean_h:
                matched_keywords.append(h)
                break

    keyword_ratio = len(matched_keywords) / total_cols

    # 2. Analisis Kepadatan Numerik & Tanggal pada Sel Data
    numeric_cells = 0
    date_cells = 0
    total_cells = total_rows * total_cols
    total_char_len = 0

    for r in rows:
        for cell in r:
            total_char_len += len(cell)
            if parse_numeric_value(cell) is not None:
                numeric_cells += 1
            elif parse_date_value(cell) is not None:
                date_cells += 1

    numeric_density = numeric_cells / max(1, total_cells)
    date_density = date_cells / max(1, total_cells)
    avg_cell_len = total_char_len / max(1, total_cells)

    # 3. Klasifikasi Semantik
    is_transactional = False
    table_type: TableTypeLiteral = "generic_table"
    reasoning_points: list[str] = []

    has_financial_headers = any(
        kw in [sanitize_identifier(h) for h in headers]
        for kw in [
            "debit",
            "kredit",
            "credit",
            "saldo",
            "balance",
            "amount",
            "nominal",
            "mutasi",
        ]
    )
    has_date_headers = any(
        kw in [sanitize_identifier(h) for h in headers]
        for kw in ["posting_date", "tanggal", "date", "value_date"]
    )

    if has_financial_headers and has_date_headers:
        is_transactional = True
        table_type = "financial_statement" if total_rows >= 5 else "transactional_log"
        reasoning_points.append(
            f"Header mengandung kombinasi tanggal dan finansial (Debit/Kredit/Saldo): {matched_keywords}."
        )
    elif (
        keyword_ratio >= 0.4
        and (numeric_density + date_density >= 0.3)
        and total_rows >= 3
    ):
        is_transactional = True
        table_type = "transactional_log"
        reasoning_points.append(
            f"Rasio keyword transaksional tinggi ({keyword_ratio:.1%}) dan kepadatan angka/tanggal {numeric_density + date_density:.1%}. "
        )
    elif avg_cell_len > 70 or (numeric_density < 0.15 and not has_financial_headers):
        is_transactional = False
        table_type = "narrative_matrix"
        reasoning_points.append(
            f"Rata-rata teks per sel panjang ({avg_cell_len:.1f} karakter) dan dominan narasi kualitatif."
        )
    else:
        is_transactional = False
        table_type = "generic_table"
        reasoning_points.append(
            f"Tabel umum tanpa pola transaksi dominan (numeric density: {numeric_density:.1%})."
        )

    recommended_storage = "sqlite_database" if is_transactional else "vector_rag"
    confidence = min(
        0.98, max(0.65, keyword_ratio * 0.5 + (numeric_density + date_density) * 0.5)
    )

    return TableClassificationResult(
        table_id=f"table_auto_{total_rows}r_{total_cols}c",
        is_transactional=is_transactional,
        table_type=table_type,
        recommended_storage=recommended_storage,
        confidence=round(confidence, 2),
        reasoning="; ".join(reasoning_points),
        numeric_density=round(numeric_density, 3),
        date_density=round(date_density, 3),
        total_rows=total_rows,
        total_columns=total_cols,
        columns_detected=headers,
    )


def infer_table_schema(
    table_name: str,
    headers: list[str],
    rows: list[list[str]],
    source_file: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> TableSchema:
    """
    Infer skema kolom SQLite dan tipe data (TEXT, REAL, INTEGER, DATE) dari data baris.
    """
    col_schemas: list[TableColumnSchema] = []
    total_rows = len(rows)

    used_names: set[str] = set()

    for col_idx, raw_header in enumerate(headers):
        clean_name = sanitize_identifier(raw_header)
        # Hindari duplikasi nama kolom
        base_name = clean_name
        suffix = 1
        while clean_name in used_names:
            clean_name = f"{base_name}_{suffix}"
            suffix += 1
        used_names.add(clean_name)

        # Analisis sampel nilai kolom
        col_values = [
            rows[r][col_idx]
            for r in range(min(total_rows, 50))
            if col_idx < len(rows[r])
        ]
        non_empty = [v for v in col_values if clean_cell_text(v)]

        numeric_count = sum(1 for v in non_empty if parse_numeric_value(v) is not None)
        date_count = sum(1 for v in non_empty if parse_date_value(v) is not None)

        sql_type: Any = "TEXT"
        if non_empty and numeric_count / len(non_empty) >= 0.7:
            # Cek apakah integer atau real (ada desimal)
            has_float = any(
                isinstance(parse_numeric_value(v), float)
                and not float(parse_numeric_value(v) or 0).is_integer()
                for v in non_empty
            )
            sql_type = "REAL" if has_float else "NUMERIC"
        elif non_empty and date_count / len(non_empty) >= 0.7:
            sql_type = "DATE"
        elif any(
            k in clean_name
            for k in [
                "amount",
                "saldo",
                "balance",
                "debit",
                "kredit",
                "credit",
                "nominal",
                "fee",
                "harga",
                "price",
            ]
        ):
            sql_type = "REAL"
        elif any(k in clean_name for k in ["date", "tanggal", "posting"]):
            sql_type = "DATE"

        col_schemas.append(
            TableColumnSchema(
                name=clean_name,
                original_name=raw_header,
                sql_type=sql_type,
                is_nullable=True,
                sample_values=non_empty[:3],
            )
        )

    return TableSchema(
        table_name=sanitize_identifier(table_name),
        source_file=source_file,
        columns=col_schemas,
        metadata=metadata or {},
    )


# ==============================================================================
# Database Manager: SQLite Management & Safe Execution
# ==============================================================================


class TabularDatabaseManager:
    """
    Pengelola Database SQLite untuk Ingesti & Querying Data Tabular Transaksional Dokumen.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            default_dir = Path("output/databases").resolve()
            default_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = default_dir / "documents_data.sqlite"
        else:
            self.db_path = Path(db_path).resolve()
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Buka koneksi SQLite dengan row factory dict."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def create_table(
        self, schema: TableSchema, if_not_exists: bool = True, replace: bool = False
    ) -> str:
        """Buat tabel SQLite berdasarkan skema terstruktur."""
        with self._get_connection() as conn:
            if replace:
                conn.execute(f'DROP TABLE IF EXISTS "{schema.table_name}";')

            col_defs = [f'"{col.name}" {col.sql_type}' for col in schema.columns]
            col_defs.insert(0, '"_row_id" INTEGER PRIMARY KEY AUTOINCREMENT')
            col_defs.append('"_source_doc" TEXT')
            col_defs.append('"_ingested_at" TEXT')

            exist_clause = "" if replace else ("IF NOT EXISTS" if if_not_exists else "")
            sql = (
                f'CREATE TABLE {exist_clause} "{schema.table_name}" (\n  '
                + ",\n  ".join(col_defs)
                + "\n);"
            )

            conn.execute(sql)
            # Buat indeks untuk kolom tanggal atau referensi jika ada
            for col in schema.columns:
                if col.sql_type == "DATE" or "ref" in col.name or "account" in col.name:
                    idx_name = f"idx_{schema.table_name}_{col.name}"
                    conn.execute(
                        f'CREATE INDEX IF NOT EXISTS "{idx_name}" ON "{schema.table_name}" ("{col.name}");'
                    )
            conn.commit()

        return schema.table_name

    def ingest_records(
        self,
        table_name: str,
        schema: TableSchema,
        headers: list[str],
        rows: list[list[str]],
        source_doc: str = "",
        replace: bool = True,
    ) -> int:
        """
        Ingest baris-baris data mentah ke tabel SQLite dengan transformasi tipe data otomatis.
        """
        self.create_table(schema, if_not_exists=True, replace=replace)

        col_names = [col.name for col in schema.columns]
        col_types = {col.name: col.sql_type for col in schema.columns}

        placeholders = ", ".join(["?"] * (len(col_names) + 2))
        insert_cols = ", ".join(
            [f'"{c}"' for c in col_names] + ['"_source_doc"', '"_ingested_at"']
        )
        insert_sql = (
            f'INSERT INTO "{schema.table_name}" ({insert_cols}) VALUES ({placeholders})'
        )

        ingested_at = datetime.now(UTC).isoformat()
        prepared_rows: list[tuple[Any, ...]] = []

        for row_idx, r in enumerate(rows):
            row_values: list[Any] = []
            for c_idx, col in enumerate(schema.columns):
                raw_val = r[c_idx] if c_idx < len(r) else ""
                sql_t = col_types.get(col.name, "TEXT")

                if sql_t in ("REAL", "NUMERIC", "INTEGER"):
                    num = parse_numeric_value(raw_val)
                    row_values.append(num)
                elif sql_t == "DATE":
                    dt = parse_date_value(raw_val)
                    row_values.append(dt or clean_cell_text(raw_val))
                else:
                    row_values.append(clean_cell_text(raw_val))

            row_values.append(source_doc)
            row_values.append(ingested_at)
            prepared_rows.append(tuple(row_values))

        with self._get_connection() as conn:
            conn.executemany(insert_sql, prepared_rows)
            conn.commit()

        return len(prepared_rows)

    def execute_query(self, sql_query: str, max_rows: int = 100) -> TabularQueryResult:
        """
        Eksekusi query SELECT aman pada database SQLite.
        """
        t0 = time.perf_counter()
        stripped = sql_query.strip().rstrip(";")
        # Keamanan: batasi hanya SELECT
        if not re.match(r"^\s*SELECT\b", stripped, re.IGNORECASE):
            return TabularQueryResult(
                query=sql_query,
                status="error",
                columns=[],
                rows=[],
                row_count=0,
                execution_time_ms=0.0,
                error_message="Hanya query 'SELECT' yang diizinkan untuk keamanan data.",
            )

        try:
            with self._get_connection() as conn:
                cursor = conn.execute(stripped)
                col_names = (
                    [d[0] for d in cursor.description] if cursor.description else []
                )
                raw_rows = cursor.fetchmany(max_rows)
                rows_dict = [dict(r) for r in raw_rows]
                elapsed = round((time.perf_counter() - t0) * 1000, 2)
                return TabularQueryResult(
                    query=sql_query,
                    status="success",
                    columns=col_names,
                    rows=rows_dict,
                    row_count=len(rows_dict),
                    execution_time_ms=elapsed,
                )
        except Exception as e:  # noqa: BLE001
            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            return TabularQueryResult(
                query=sql_query,
                status="error",
                columns=[],
                rows=[],
                row_count=0,
                execution_time_ms=elapsed,
                error_message=f"SQL Error: {e}",
            )

    def inspect_database(self) -> dict[str, Any]:
        """Ambil daftar seluruh tabel dan skema kolom yang ada di database."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            )
            tables = [row["name"] for row in cursor.fetchall()]

            table_details = {}
            for t in tables:
                col_cur = conn.execute(f"PRAGMA table_info('{t}');")
                cols = [dict(c) for c in col_cur.fetchall()]
                count_cur = conn.execute(f"SELECT COUNT(*) as cnt FROM '{t}';")
                cnt = count_cur.fetchone()["cnt"]
                sample_cur = conn.execute(f"SELECT * FROM '{t}' LIMIT 3;")
                samples = [dict(r) for r in sample_cur.fetchall()]
                table_details[t] = {
                    "columns": cols,
                    "row_count": cnt,
                    "samples": samples,
                }

        return {
            "database_path": str(self.db_path),
            "tables": table_details,
        }


# ==============================================================================
# Double-Verification Mechanism (Integritas, Agregasi, Refleksi)
# ==============================================================================


class TabularVerifier:
    """
    Engine Verifikasi Ganda (Double-Verification):
      1. Integritas baris & skema kolom.
      2. Uji coba query kalkulasi agregat (SUM & AVG).
      3. Uji konsistensi aritmatika saldo rekening (Balance[n-1] + Credit - Debit = Balance[n]).
      4. Refleksi verifikasi model LLM bila tersedia.
    """

    def __init__(
        self, db_manager: TabularDatabaseManager, llm: BaseChatModel | None = None
    ) -> None:
        self.db = db_manager
        self.llm = llm

    def verify_table(
        self,
        table_name: str,
        expected_row_count: int | None = None,
        source_markdown_sample: str | None = None,
    ) -> TableVerificationReport:
        checks: list[VerificationCheck] = []
        test_queries: list[dict[str, Any]] = []

        # 1. Pengecekan Integritas Baris (Row Count Integrity)
        count_res = self.db.execute_query(f'SELECT COUNT(*) as cnt FROM "{table_name}"')
        actual_rows = count_res.rows[0]["cnt"] if count_res.rows else 0
        test_queries.append(
            {
                "query": f'SELECT COUNT(*) FROM "{table_name}"',
                "result": actual_rows,
                "success": count_res.status == "success",
            }
        )

        if expected_row_count is not None:
            row_count_ok = actual_rows == expected_row_count
            checks.append(
                VerificationCheck(
                    check_name="row_count_integrity",
                    passed=row_count_ok,
                    details=f"SQLite memiliki {actual_rows} baris (Ekspektasi: {expected_row_count} baris).",
                    metric_value=actual_rows,
                )
            )
        else:
            checks.append(
                VerificationCheck(
                    check_name="row_count_present",
                    passed=actual_rows > 0,
                    details=f"Tabel memiliki {actual_rows} baris tersimpan.",
                    metric_value=actual_rows,
                )
            )

        # 2. Pengecekan Skema Kolom
        inspect = self.db.inspect_database()
        t_info = inspect["tables"].get(table_name, {})
        cols_info = t_info.get("columns", [])
        col_names = [c["name"] for c in cols_info if not c["name"].startswith("_")]

        checks.append(
            VerificationCheck(
                check_name="columns_registered",
                passed=len(col_names) > 0,
                details=f"Tabel memiliki {len(col_names)} kolom terdefinisi: {col_names}",
                metric_value=len(col_names),
            )
        )

        # 3. Numeric Aggregation Sanity Test (Uji SUM & AVG)
        numeric_cols = [
            c["name"]
            for c in cols_info
            if c["type"] in ("REAL", "NUMERIC", "INTEGER")
            and not c["name"].startswith("_")
        ]
        agg_passed = True
        agg_details: Any = {}

        if numeric_cols:
            agg_selects = []
            for nc in numeric_cols:
                agg_selects.append(f'SUM("{nc}") as sum_{nc}')
                agg_selects.append(f'AVG("{nc}") as avg_{nc}')
            agg_sql = f'SELECT {", ".join(agg_selects)} FROM "{table_name}";'

            agg_res = self.db.execute_query(agg_sql)
            if agg_res.status == "error":
                agg_passed = False
                agg_details = {"error": agg_res.error_message}
                test_queries.append(
                    {
                        "query": agg_sql,
                        "result": str(agg_res.error_message),
                        "success": False,
                    }
                )
            else:
                agg_details = agg_res.rows[0] if agg_res.rows else {}
                test_queries.append(
                    {"query": agg_sql, "result": agg_details, "success": True}
                )

            checks.append(
                VerificationCheck(
                    check_name="numeric_aggregation_sanity",
                    passed=agg_passed,
                    details=f"Query kalkulasi agregat (SUM & AVG) berjalan sukses tanpa error: {agg_details}"
                    if agg_passed
                    else f"Gagal menjalankan query agregat: {agg_details}",
                )
            )

        # 4. Bank Statement Balance Continuity Check (Balance[n-1] + Kredit - Debit ≈ Balance[n])
        has_debit = any("debit" in c.lower() for c in col_names)
        has_credit = any(
            "credit" in c.lower() or "kredit" in c.lower() for c in col_names
        )
        has_balance = any(
            "balance" in c.lower() or "saldo" in c.lower() for c in col_names
        )

        if has_debit and has_credit and has_balance:
            debit_col = next(c for c in col_names if "debit" in c.lower())
            credit_col = next(
                c for c in col_names if "credit" in c.lower() or "kredit" in c.lower()
            )
            balance_col = next(
                c for c in col_names if "balance" in c.lower() or "saldo" in c.lower()
            )

            sample_rows_res = self.db.execute_query(
                f'SELECT "_row_id", "{debit_col}", "{credit_col}", "{balance_col}" FROM "{table_name}" ORDER BY "_row_id" ASC LIMIT 20;'
            )
            s_rows = sample_rows_res.rows
            continuity_mismatches = 0
            if len(s_rows) >= 2:
                for idx in range(1, len(s_rows)):
                    prev_bal = s_rows[idx - 1].get(balance_col) or 0.0
                    curr_deb = s_rows[idx].get(debit_col) or 0.0
                    curr_crd = s_rows[idx].get(credit_col) or 0.0
                    curr_bal = s_rows[idx].get(balance_col) or 0.0

                    expected_bal = prev_bal + curr_crd - curr_deb
                    # Toleransi selisih floating point 1.0 (karena pembulatan sen)
                    if abs(expected_bal - curr_bal) > 1.0:
                        continuity_mismatches += 1

            checks.append(
                VerificationCheck(
                    check_name="balance_arithmetic_consistency",
                    passed=continuity_mismatches == 0,
                    details=f"Pemeriksaan kontinuitas saldo (Balance[n-1] + Kredit - Debit = Balance[n]): {continuity_mismatches} ketidaksesuaian dari {len(s_rows)} baris sampel.",
                    metric_value=continuity_mismatches,
                )
            )

        # 5. LLM Reflection (Optional)
        reflection_notes: str | None = None
        if self.llm is not None and source_markdown_sample:
            try:
                sample_db_rows = self.db.execute_query(
                    f'SELECT * FROM "{table_name}" LIMIT 5;'
                ).rows
                prompt = (
                    "Lakukan verifikasi refleksi antara teks sampel Markdown asli dan record data SQLite hasil ekstraksi berikut.\n"
                    f"Sample Markdown Dokumen:\n```\n{source_markdown_sample[:1000]}\n```\n\n"
                    f"Sample Database SQLite Rows:\n```json\n{json.dumps(sample_db_rows, indent=2)}\n```\n\n"
                    "Apakah ada kolom atau angka penting yang terpotong/salah letak? Jawab ringkas 2-3 kalimat."
                )
                resp = self.llm.invoke(prompt)
                if hasattr(resp, "content"):
                    raw_c = resp.content
                    reflection_notes = (
                        raw_c
                        if isinstance(raw_c, str)
                        else json.dumps(raw_c, ensure_ascii=False)
                    )
                else:
                    reflection_notes = str(resp)
            except Exception as e:  # noqa: BLE001
                reflection_notes = f"LLM Reflection skipped: {e}"

        all_passed = all(c.passed for c in checks)
        passed_ratio = sum(1 for c in checks if c.passed) / max(1, len(checks))
        confidence_score = (
            round(0.5 + (passed_ratio * 0.5), 2)
            if all_passed
            else round(passed_ratio * 0.75, 2)
        )
        status_lit: Any = (
            "verified"
            if all_passed
            else ("needs_revision" if passed_ratio >= 0.6 else "rejected")
        )

        return TableVerificationReport(
            table_name=table_name,
            database_path=str(self.db.db_path),
            is_valid=all_passed,
            verification_status=status_lit,
            confidence_score=confidence_score,
            checks=checks,
            verified_row_count=actual_rows,
            test_queries=test_queries,
            llm_reflection=reflection_notes,
            summary=f"Verifikasi data tabel '{table_name}' selesai dengan status '{status_lit}'. Total {len(checks)} pemeriksaan dijalankan ({sum(1 for c in checks if c.passed)} passed).",
        )


# ==============================================================================
# High-Level Pipeline Functions
# ==============================================================================


def extract_and_ingest_tables_from_markdown(
    markdown_text: str,
    source_file: str = "",
    db_path: str | Path | None = None,
    table_name_prefix: str | None = None,
    llm: BaseChatModel | None = None,
) -> list[TableIngestionResult]:
    """
    Ekstrak semua tabel dari Markdown, filter tabel transaksional, simpan ke database SQLite,
    dan jalankan double-verification otomatis.
    """
    parsed_tables = parse_markdown_tables(markdown_text)
    if not parsed_tables:
        return []

    src_stem = Path(source_file).stem if source_file else "doc"
    base_prefix = sanitize_identifier(table_name_prefix or src_stem)

    db_manager = TabularDatabaseManager(db_path)
    verifier = TabularVerifier(db_manager, llm=llm)
    results: list[TableIngestionResult] = []

    for idx, tbl in enumerate(parsed_tables, start=1):
        headers = tbl["headers"]
        rows = tbl["rows"]
        context = tbl["context"]

        # 1. Klasifikasi
        classification = classify_table_heuristic(headers, rows, context=context)

        # 2. Jika bukan transaksional (misal tabel naratif), lewati dari SQLite (tetap di RAG)
        if not classification.is_transactional:
            continue

        # 3. Buat skema SQLite
        table_name = f"{base_prefix}_t{idx}"
        schema = infer_table_schema(
            table_name=table_name,
            headers=headers,
            rows=rows,
            source_file=source_file,
            metadata={"context": context, "original_index": idx},
        )

        # 4. Ingest data
        rows_ingested = db_manager.ingest_records(
            table_name=table_name,
            schema=schema,
            headers=headers,
            rows=rows,
            source_doc=source_file,
            replace=True,
        )

        # 5. Double-Verification
        sample_md = "\n".join(["|".join(headers)] + ["|".join(r) for r in rows[:5]])
        verif_report = verifier.verify_table(
            table_name=table_name,
            expected_row_count=len(rows),
            source_markdown_sample=sample_md,
        )

        status = "success" if verif_report.is_valid else "warning"

        # Ambil sampel 3 data
        sample_data = db_manager.execute_query(
            f'SELECT * FROM "{table_name}" LIMIT 3;'
        ).rows

        results.append(
            TableIngestionResult(
                status=status,
                table_name=table_name,
                database_path=str(db_manager.db_path),
                total_rows_ingested=rows_ingested,
                columns=[col.name for col in schema.columns],
                verification_report=verif_report,
                sample_data=sample_data,
                message=f"Tabel '{table_name}' berhasil disimpan ({rows_ingested} baris).",
            )
        )

    return results


def query_sqlite(
    sql_query: str, db_path: str | Path | None = None
) -> TabularQueryResult:
    """Eksekusi query SQL praktis pada database dokumen."""
    manager = TabularDatabaseManager(db_path)
    return manager.execute_query(sql_query)
