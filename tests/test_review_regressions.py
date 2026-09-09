"""Regresi untuk temuan review menyeluruh aplikasi dan dokumentasi."""

import tempfile
import unittest
from pathlib import Path

from app.agent_graph import AgentDocumentGraph
from app.tabular_db import (
    TabularDatabaseManager,
    cross_verify_dual_track,
    extract_document_header_info,
    parse_markdown_tables,
    parse_numeric_value,
    process_page_tabular_agent,
)


class TestReviewRegressions(unittest.TestCase):
    def test_parse_indonesian_and_decimal_numeric_values(self):
        self.assertEqual(parse_numeric_value("Rp 50.000"), 50000.0)
        self.assertEqual(parse_numeric_value("1.234"), 1234.0)
        self.assertEqual(parse_numeric_value("1234,56"), 1234.56)
        self.assertEqual(parse_numeric_value("1,234,567.89"), 1234567.89)

    def test_header_fingerprint_uses_source_and_is_stable_per_document(self):
        first = extract_document_header_info("Teks tanpa metadata", "first.pdf")
        same_document = extract_document_header_info("Halaman lain tanpa metadata", "first.pdf")
        second = extract_document_header_info("Teks tanpa metadata", "second.pdf")
        self.assertEqual(first.fingerprint_hash, same_document.fingerprint_hash)
        self.assertNotEqual(first.fingerprint_hash, second.fingerprint_hash)

    def test_adjacent_markdown_tables_stay_separate(self):
        markdown = """| Produk | Warna |
|---|---|
| Kursi | Biru |

| Kota | Cuaca |
|---|---|
| Bandung | Cerah |
"""
        tables = parse_markdown_tables(markdown)
        self.assertEqual(len(tables), 2)
        self.assertEqual(tables[0]["headers"], ["Produk", "Warna"])
        self.assertEqual(tables[1]["headers"], ["Kota", "Cuaca"])

    def test_general_tables_require_force_all_tables(self):
        markdown = """| Produk | Qty | Harga |
|---|---|---|
| Kursi | 2 | 10000 |
| Meja | 3 | 20000 |
| Lampu | 1 | 30000 |
"""
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "tables.sqlite"
            event, results = process_page_tabular_agent(
                markdown, 1, source_file="inventory.pdf", db_path=db_path
            )
            self.assertEqual(event.rows_ingested_total, 0)
            self.assertEqual(results, [])

            _event, forced_results = process_page_tabular_agent(
                markdown,
                1,
                source_file="inventory.pdf",
                db_path=db_path,
                force_all_tables=True,
            )
            self.assertEqual(forced_results[0].total_rows_ingested, 3)

    def test_guardrail_flags_missing_signatories(self):
        markdown = """| Pihak | Nama | Tanda Tangan |
|---|---|---|
| Pihak Pertama | Sari | Signed |
| Pihak Kedua | Bima | Signed |
"""
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "signatories.sqlite"
            process_page_tabular_agent(
                markdown, 1, source_file="agreement.pdf", db_path=db_path
            )
            manager = TabularDatabaseManager(db_path)
            with manager._get_connection() as connection:
                connection.execute("DELETE FROM document_signatories;")
            report = cross_verify_dual_track(markdown, db_path, "agreement.pdf")
            self.assertEqual(report.guardrail_status, "WARNING")
            self.assertTrue(report.discrepancies)

    def test_batch_save_preserves_existing_chunks_without_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory) / "output"
            source = Path(directory) / "document.pdf"
            graph = AgentDocumentGraph()
            common = {
                "resolved_path": str(source),
                "resolved_out": str(out_dir),
                "total_items": 3,
                "specs": "plain",
            }
            graph._node_save_and_stitch(
                {**common, "current_page": 1, "incoming_markdown": "Halaman satu"}
            )
            graph._node_save_and_stitch(
                {**common, "current_page": 3, "incoming_markdown": "Halaman tiga"}
            )
            saved_pages = sorted((out_dir / "chunks").glob("page_*.md"))
            self.assertEqual([page.name for page in saved_pages], ["page_0001.md", "page_0003.md"])

    def test_save_can_skip_automatic_tabular_ingestion(self):
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory) / "output"
            source = Path(directory) / "document.pdf"
            AgentDocumentGraph()._node_save_and_stitch(
                {
                    "resolved_path": str(source),
                    "resolved_out": str(out_dir),
                    "total_items": 1,
                    "current_page": 1,
                    "incoming_markdown": "| Item | Nilai |\n|---|---|\n| A | 1 |",
                    "ingest_transactional_tables": False,
                }
            )
            self.assertFalse((out_dir / "databases" / "document.sqlite").exists())


if __name__ == "__main__":
    unittest.main()
