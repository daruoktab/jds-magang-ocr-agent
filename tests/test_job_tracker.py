"""
Unit tests untuk Job Tracker & Background Process Manager.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from app.job_tracker import JobInfo, JobManager, is_pid_alive


class TestJobTracker(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_job_tracker_"))
        self.log_dir = self.temp_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_progress_percentage(self) -> None:
        job = JobInfo(
            job_id="test_doc",
            file_name="test_doc.pdf",
            input_path=self.temp_dir / "test_doc.pdf",
            output_dir=self.temp_dir,
            out_file=self.temp_dir / "test_doc.md",
            db_file=self.temp_dir / "databases" / "test_doc.sqlite",
            log_path=self.log_dir / "test_doc_20260902.log",
            latest_log_path=self.log_dir / "test_doc_latest.log",
            status_file=self.log_dir / "test_doc_status.json",
            progress_file=self.log_dir / "test_doc_progress.txt",
            status="running",
            current_page=3,
            total_pages=10,
        )
        self.assertEqual(job.progress_percentage(), 30.0)

        job.status = "completed"
        self.assertEqual(job.progress_percentage(), 100.0)

    def test_save_status_and_progress_file(self) -> None:
        status_file = self.log_dir / "doc1_status.json"
        progress_file = self.log_dir / "doc1_progress.txt"

        job = JobInfo(
            job_id="doc1",
            file_name="doc1.pdf",
            input_path=self.temp_dir / "doc1.pdf",
            output_dir=self.temp_dir,
            out_file=self.temp_dir / "doc1.md",
            db_file=None,
            log_path=self.log_dir / "doc1_123.log",
            latest_log_path=self.log_dir / "doc1_latest.log",
            status_file=status_file,
            progress_file=progress_file,
            status="running",
            current_page=4,
            total_pages=8,
            stage="Ekstraksi Halaman 4 / 8",
            last_message="Memproses Halaman 4 / 8 dari 'doc1.pdf'...",
            pid=99999,
        )
        job.save_status()

        self.assertTrue(status_file.exists())
        self.assertTrue(progress_file.exists())

        # Verifikasi JSON
        data = json.loads(status_file.read_text(encoding="utf-8"))
        self.assertEqual(data["current_page"], 4)
        self.assertEqual(data["total_pages"], 8)
        self.assertEqual(data["status"], "running")
        self.assertEqual(data["progress_pct"], 50.0)

        # Verifikasi TXT
        txt_content = progress_file.read_text(encoding="utf-8")
        self.assertIn("Halaman 4 / 8 (50.0%)", txt_content)
        self.assertIn("Ekstraksi Halaman 4 / 8", txt_content)
        self.assertIn("File Dokumen   : doc1.pdf", txt_content)

    def test_parse_line_progress_pdf_and_ppt(self) -> None:
        manager = JobManager.get_instance()
        job = JobInfo(
            job_id="parse_test",
            file_name="parse_test.pdf",
            input_path=self.temp_dir / "parse_test.pdf",
            output_dir=self.temp_dir,
            out_file=self.temp_dir / "parse_test.md",
            db_file=None,
            log_path=self.log_dir / "parse_test.log",
            latest_log_path=self.log_dir / "parse_test_latest.log",
            status_file=self.log_dir / "parse_test_status.json",
            progress_file=self.log_dir / "parse_test_progress.txt",
        )

        # Uji log Halaman PDF
        manager._parse_line_progress(
            job,
            "10:15:20 | INFO | Memproses Halaman 5 / 20 dari 'parse_test.pdf'...",
        )
        self.assertEqual(job.current_page, 5)
        self.assertEqual(job.total_pages, 20)
        self.assertIn("Halaman 5 / 20", job.stage)

        # Uji log Slide PPT
        manager._parse_line_progress(
            job, "[Vision PPT] [Slide 7/15] Memproses slide 'slide_07.png'..."
        )
        self.assertEqual(job.current_page, 7)
        self.assertEqual(job.total_pages, 15)
        self.assertIn("Slide 7 / 15", job.stage)

        # Uji log Sub-Agent SQL
        manager._parse_line_progress(
            job,
            "Sub-Agent SQL [Halaman 7]: Terdeteksi 1 tabel, mengevaluasi ingesti ke SQLite...",
        )
        self.assertIn("Sub-Agent SQL", job.stage)

        # Uji log Guardrail Cross-Verification
        manager._parse_line_progress(
            job,
            "Guardrail Cross-Verification: Memeriksa integritas teks Markdown vs SQLite...",
        )
        self.assertIn("Guardrail Supervisor", job.stage)

    def test_reconstruct_job_from_disk(self) -> None:
        manager = JobManager.get_instance()
        stem = "disk_recon"
        status_file = self.log_dir / f"{stem}_status.json"
        progress_file = self.log_dir / f"{stem}_progress.txt"
        latest_log = self.log_dir / f"{stem}_latest.log"
        latest_log.write_text("Line 1\nLine 2\nLine 3\n", encoding="utf-8")

        status_data = {
            "job_id": stem,
            "file_name": "disk_recon.pdf",
            "input_path": str(self.temp_dir / "disk_recon.pdf"),
            "output_dir": str(self.temp_dir),
            "out_file": str(self.temp_dir / "disk_recon.md"),
            "db_file": None,
            "log_path": str(latest_log),
            "latest_log_path": str(latest_log),
            "status_file": str(status_file),
            "progress_file": str(progress_file),
            "status": "completed",
            "current_page": 10,
            "total_pages": 10,
            "stage": "Selesai",
            "last_message": "Selesai sukses",
            "pid": 1234,
        }
        status_file.write_text(json.dumps(status_data), encoding="utf-8")

        # Pastikan tidak ada di memori
        manager.reset_job(stem, output_dir=self.temp_dir)

        # Tulis ulang status file karena reset_job menghapusnya
        status_file.write_text(json.dumps(status_data), encoding="utf-8")

        # Panggil get_job
        reconstructed = manager.get_job(stem, output_dir=self.temp_dir)
        self.assertIsNotNone(reconstructed)
        assert reconstructed is not None
        self.assertEqual(reconstructed.status, "completed")
        self.assertEqual(reconstructed.total_pages, 10)
        self.assertEqual(reconstructed.current_page, 10)

    def test_reconstruct_fallback_without_status_file(self) -> None:
        manager = JobManager.get_instance()
        stem = "cli_doc_without_status"
        doc_dir = self.temp_dir / stem
        doc_dir.mkdir(parents=True, exist_ok=True)
        md_file = doc_dir / f"{stem}.md"
        md_file.write_text("# Laporan CLI\nIsi laporan.", encoding="utf-8")

        manager.reset_job(stem, output_dir=self.temp_dir)
        job = manager.get_job(stem, output_dir=self.temp_dir)
        self.assertIsNotNone(job)
        assert job is not None
        self.assertEqual(job.status, "completed")
        self.assertTrue(job.out_file.exists())

    def test_list_all_documents(self) -> None:
        manager = JobManager.get_instance()
        stem1 = "doc_alpha"
        doc_dir1 = self.temp_dir / stem1
        doc_dir1.mkdir(parents=True, exist_ok=True)
        (doc_dir1 / f"{stem1}.md").write_text("# Alpha", encoding="utf-8")

        stem2 = "doc_beta"
        doc_dir2 = self.temp_dir / stem2
        doc_dir2.mkdir(parents=True, exist_ok=True)
        (doc_dir2 / f"{stem2}.md").write_text("# Beta", encoding="utf-8")
        pages_dir = doc_dir2 / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
        (pages_dir / "page_0001.png").write_bytes(b"dummy")

        docs = manager.list_all_documents(output_dir=self.temp_dir)
        stems = [d["stem"] for d in docs]
        self.assertIn(stem1, stems)
        self.assertIn(stem2, stems)
        beta_doc = next(d for d in docs if d["stem"] == stem2)
        self.assertEqual(beta_doc["page_count"], 1)

    def test_is_pid_alive(self) -> None:
        current_pid = os.getpid()
        self.assertTrue(is_pid_alive(current_pid))
        self.assertFalse(is_pid_alive(None))
        self.assertFalse(is_pid_alive(-1))


if __name__ == "__main__":
    unittest.main()
