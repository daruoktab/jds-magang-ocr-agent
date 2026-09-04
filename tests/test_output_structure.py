import shutil
import sys
import tempfile
import unittest
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.cleanup import clean_all_outputs, clean_document_output, migrate_legacy_output


class TestOutputStructure(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_output_struct_"))

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_migrate_legacy_output(self):
        # Buat dummy struktur legacy:
        # output/docA.md
        # output/databases/docA.sqlite
        # output/pdf_pages/docA/page_0001.png
        # output/logs/docA_latest.log
        (self.test_dir / "docA.md").write_text("# Doc A Content", encoding="utf-8")
        
        legacy_db_dir = self.test_dir / "databases"
        legacy_db_dir.mkdir(parents=True, exist_ok=True)
        (legacy_db_dir / "docA.sqlite").write_text("sqlite dummy", encoding="utf-8")

        legacy_pdf_dir = self.test_dir / "pdf_pages" / "docA"
        legacy_pdf_dir.mkdir(parents=True, exist_ok=True)
        (legacy_pdf_dir / "page_0001.png").write_text("png dummy", encoding="utf-8")

        legacy_log_dir = self.test_dir / "logs"
        legacy_log_dir.mkdir(parents=True, exist_ok=True)
        (legacy_log_dir / "docA_latest.log").write_text("log dummy", encoding="utf-8")

        # Jalankan migrasi
        migrate_legacy_output(self.test_dir)

        # Verifikasi struktur baru:
        docA_dir = self.test_dir / "docA"
        self.assertTrue(docA_dir.exists())
        self.assertTrue((docA_dir / "docA.md").exists())
        self.assertTrue((docA_dir / "databases" / "docA.sqlite").exists())
        self.assertTrue((docA_dir / "pages" / "page_0001.png").exists())
        self.assertTrue((docA_dir / "logs" / "docA_latest.log").exists())

        # Pastikan file loose dan folder legacy kosong telah bersih
        self.assertFalse((self.test_dir / "docA.md").exists())
        self.assertFalse((self.test_dir / "databases").exists())
        self.assertFalse((self.test_dir / "pdf_pages").exists())
        self.assertFalse((self.test_dir / "logs").exists())

    def test_clean_document_output(self):
        docB_dir = self.test_dir / "docB"
        docB_dir.mkdir(parents=True, exist_ok=True)
        (docB_dir / "docB.md").write_text("test", encoding="utf-8")

        clean_document_output("docB", output_dir=self.test_dir)
        self.assertFalse(docB_dir.exists())

    def test_clean_all_outputs(self):
        (self.test_dir / ".gitkeep").write_text("", encoding="utf-8")
        docC_dir = self.test_dir / "docC"
        docC_dir.mkdir(parents=True, exist_ok=True)
        (docC_dir / "docC.md").write_text("test", encoding="utf-8")

        clean_all_outputs(output_dir=self.test_dir)
        self.assertTrue((self.test_dir / ".gitkeep").exists())
        self.assertFalse(docC_dir.exists())

    def test_export_to_csv(self):
        from app.tabular_db import (
            TableColumnSchema,
            TableSchema,
            TabularDatabaseManager,
        )

        db_file = self.test_dir / "databases" / "test_doc.sqlite"
        mgr = TabularDatabaseManager(db_file)
        schema = TableSchema(
            table_name="transaksi",
            source_file="test_doc.pdf",
            columns=[
                TableColumnSchema(
                    name="tanggal",
                    original_name="Tanggal",
                    sql_type="TEXT",
                    is_nullable=False,
                    description="",
                    sample_values=[],
                ),
                TableColumnSchema(
                    name="nominal",
                    original_name="Nominal",
                    sql_type="REAL",
                    is_nullable=False,
                    description="",
                    sample_values=[],
                ),
            ],
            primary_key=None,
        )
        mgr.ingest_records(
            table_name="transaksi",
            schema=schema,
            headers=["tanggal", "nominal"],
            rows=[["2026-01-01", "150000"], ["2026-01-02", "250000"]],
        )

        csv_dir = self.test_dir / "csv"
        exported = mgr.export_to_csv(output_dir=csv_dir)
        self.assertEqual(len(exported), 1)
        self.assertTrue((csv_dir / "transaksi.csv").exists())
        csv_content = (csv_dir / "transaksi.csv").read_text(encoding="utf-8-sig")
        self.assertIn("tanggal", csv_content)
        self.assertIn("150000", csv_content)


if __name__ == "__main__":
    unittest.main()

