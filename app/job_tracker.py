"""
Job Tracker & Background Process Manager untuk Streamlit Document Extraction.

Fitur:
  - Eksekusi background thread & subprocess independen dari lifecycle tab browser.
  - Logging real-time langsung di-flush baris demi baris ke file:
      * output/logs/{stem}_latest.log (selalu menunjuk ke run terakhir)
      * output/logs/{stem}_{timestamp}.log (arsip riwayat)
  - Tracking status & progres granular ke file:
      * output/logs/{stem}_status.json (metadata terstruktur)
      * output/logs/{stem}_progress.txt (ringkasan 6-baris ramah dibaca manusia)
  - Deteksi otomatis nomor halaman saat ini vs total halaman dari output stream.
  - Tahan terhadap minimize window, tab sleep, atau refresh UI Streamlit.
"""

from __future__ import annotations

import collections
import datetime as dt
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def is_pid_alive(pid: int | None) -> bool:
    """Cek apakah proses dengan PID tertentu masih aktif di sistem operasi."""
    if pid is None or pid <= 0:
        return False
    try:
        import psutil

        return psutil.pid_exists(pid)
    except Exception:  # noqa: BLE001, S110
        pass

    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except (ProcessLookupError, OSError):
        return False
    else:
        return True


@dataclass
class JobInfo:
    job_id: str
    file_name: str
    input_path: Path
    output_dir: Path
    out_file: Path
    db_file: Path | None
    log_path: Path
    latest_log_path: Path
    status_file: Path
    progress_file: Path
    status: str = "running"  # "running" | "completed" | "failed" | "canceled"
    current_page: int = 0
    total_pages: int = 0
    stage: str = "Memulai ekstraksi..."
    last_message: str = ""
    started_at: str = field(
        default_factory=lambda: dt.datetime.now(dt.UTC).isoformat(
            timespec="seconds"
        )
    )
    updated_at: str = field(
        default_factory=lambda: dt.datetime.now(dt.UTC).isoformat(
            timespec="seconds"
        )
    )
    pid: int | None = None
    returncode: int | None = None
    error_message: str | None = None
    recent_logs: collections.deque[str] = field(
        default_factory=lambda: collections.deque(maxlen=200)
    )

    def progress_percentage(self) -> float:
        """Hitung persentase progres 0.0 - 100.0."""
        if self.status == "completed":
            return 100.0
        if self.total_pages > 0 and self.current_page > 0:
            pct = (self.current_page / self.total_pages) * 100.0
            return min(98.0, max(1.0, round(pct, 1)))
        if self.status == "running":
            return 5.0
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "file_name": self.file_name,
            "input_path": str(self.input_path),
            "output_dir": str(self.output_dir),
            "out_file": str(self.out_file),
            "db_file": str(self.db_file) if self.db_file else None,
            "log_path": str(self.log_path),
            "latest_log_path": str(self.latest_log_path),
            "status_file": str(self.status_file),
            "progress_file": str(self.progress_file),
            "status": self.status,
            "current_page": self.current_page,
            "total_pages": self.total_pages,
            "progress_pct": self.progress_percentage(),
            "stage": self.stage,
            "last_message": self.last_message,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "pid": self.pid,
            "returncode": self.returncode,
            "error_message": self.error_message,
        }

    def save_status(self) -> None:
        """Simpan status JSON dan ringkasan TXT ke disk secara atomic/aman."""
        try:
            self.status_file.parent.mkdir(parents=True, exist_ok=True)
            self.updated_at = dt.datetime.now(dt.UTC).isoformat(
                timespec="seconds"
            )

            # 1. Simpan JSON terstruktur
            self.status_file.write_text(
                json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            # 2. Simpan TXT ringkas yang mudah dimonitor via Notepad / terminal
            pct_str = f"{self.progress_percentage():.1f}%"
            page_str = (
                f"{self.current_page} / {self.total_pages}"
                if self.total_pages > 0
                else f"{self.current_page} (menghitung total...)"
            )
            txt_content = (
                "==================================================\n"
                "STATUS EKSTRAKSI DOKUMEN (REAL-TIME MONITOR)\n"
                "==================================================\n"
                f"File Dokumen   : {self.file_name}\n"
                f"Status         : {self.status.upper()} (PID: {self.pid or '-'})\n"
                f"Progres        : Halaman {page_str} ({pct_str})\n"
                f"Tahapan Saat Ini: {self.stage}\n"
                f"Pesan Terakhir : {self.last_message}\n"
                f"Waktu Mulai    : {self.started_at}\n"
                f"Update Terakhir: {self.updated_at}\n"
                f"File Log Utama : {self.latest_log_path}\n"
                f"File Markdown  : {self.out_file}\n"
                f"Database SQLite: {self.db_file or '-'}\n"
                "==================================================\n"
            )
            self.progress_file.write_text(txt_content, encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            logger.warning("Gagal menyimpan file status job: %s", e)


class JobManager:
    """Manajer singleton in-process untuk menjalankan dan melacak job ekstraksi."""

    _instance: ClassVar[JobManager | None] = None
    _jobs: ClassVar[dict[str, JobInfo]] = {}
    _processes: ClassVar[dict[str, subprocess.Popen]] = {}
    _threads: ClassVar[dict[str, threading.Thread]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def get_instance(cls) -> JobManager:
        if cls._instance is None:
            cls._instance = JobManager()
        return cls._instance

    def start_job(
        self,
        input_path: Path,
        output_dir: Path,
        *,
        doc_type: str | None = None,
        dpi: int = 200,
        force_all_tables: bool = False,
        preview_chunks: bool = True,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ) -> JobInfo:
        """Mulai proses ekstraksi baru di latar belakang jika belum ada yang berjalan."""
        with self._lock:
            stem = input_path.stem
            existing_job = self.get_job(stem, output_dir=output_dir)
            if existing_job and existing_job.status == "running":
                # Sudah berjalan, kembalikan job yang sedang aktif
                return existing_job

            doc_output_dir = output_dir / stem
            doc_output_dir.mkdir(parents=True, exist_ok=True)
            log_dir = doc_output_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)

            timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
            log_path = log_dir / f"{stem}_{timestamp}.log"
            latest_log_path = log_dir / f"{stem}_latest.log"
            status_file = log_dir / f"{stem}_status.json"
            progress_file = log_dir / f"{stem}_progress.txt"
            out_file = doc_output_dir / f"{stem}.md"
            db_dir = doc_output_dir / "databases"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_file = db_dir / f"{stem}.sqlite"

            # Tulis header awal ke file log agar langsung tersedia
            start_iso = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
            header = (
                "================================================================================\n"
                f"LOG EKSTRAKSI DOKUMEN: {input_path.name}\n"
                f"Waktu Mulai: {start_iso}\n"
                f"Target Markdown: {out_file}\n"
                f"Target SQLite: {db_file}\n"
                "================================================================================\n\n"
            )
            log_path.write_text(header, encoding="utf-8")
            latest_log_path.write_text(header, encoding="utf-8")

            job = JobInfo(
                job_id=stem,
                file_name=input_path.name,
                input_path=input_path,
                output_dir=output_dir,
                out_file=out_file,
                db_file=db_file,
                log_path=log_path,
                latest_log_path=latest_log_path,
                status_file=status_file,
                progress_file=progress_file,
                status="running",
                stage="Menginisialisasi proses CLI...",
                last_message="Memulai pipeline ekstraksi...",
                started_at=start_iso,
            )
            job.save_status()
            self._jobs[stem] = job

            # Susun perintah CLI
            cmd = [
                sys.executable,
                "main.py",
                str(input_path),
                "-o",
                str(out_file),
                "--dpi",
                str(dpi),
                "--log-file",
                str(log_path),
            ]
            if doc_type:
                cmd.extend(["-t", doc_type])
            if force_all_tables:
                cmd.append("--force-all-tables")
            if preview_chunks:
                cmd.extend([
                    "--preview-chunks",
                    "--chunk-size",
                    str(chunk_size),
                    "--chunk-overlap",
                    str(chunk_overlap),
                ])

            # Jalankan worker di background thread terpisah
            worker = threading.Thread(
                target=self._run_worker,
                args=(job, cmd),
                daemon=True,
                name=f"Worker-{stem}",
            )
            self._threads[stem] = worker
            worker.start()
            return job

    def _run_worker(self, job: JobInfo, cmd: list[str]) -> None:
        """Worker background yang membaca stdout proses dan mengalirkan log ke disk."""
        start_time = time.time()
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            with self._lock:
                self._processes[job.job_id] = proc
                job.pid = proc.pid
                job.stage = "Proses CLI aktif, menunggu parsing halaman..."
                job.save_status()

            # Buka kedua file log dalam mode append dengan autoflush
            with (
                open(job.log_path, "a", encoding="utf-8", buffering=1) as f_hist,
                open(
                    job.latest_log_path, "a", encoding="utf-8", buffering=1
                ) as f_latest,
            ):
                if proc.stdout is not None:
                    for raw_line in proc.stdout:
                        f_hist.write(raw_line)
                        f_latest.write(raw_line)
                        f_hist.flush()
                        f_latest.flush()

                        line = raw_line.strip()
                        if not line:
                            continue

                        job.recent_logs.append(line)
                        self._parse_line_progress(job, line)

            returncode = proc.wait()
            elapsed = time.time() - start_time
            mins, secs = divmod(int(elapsed), 60)
            duration_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"

            footer = (
                "\n================================================================================\n"
                f"STATUS: {'SELESAI SUKSES' if returncode == 0 else f'GAGAL (Exit Code: {returncode})'}\n"
                f"Waktu Selesai : {dt.datetime.now(dt.UTC).isoformat(timespec='seconds')}\n"
                f"Total Durasi  : {duration_str}\n"
                "================================================================================\n"
            )
            with open(job.log_path, "a", encoding="utf-8") as f:
                f.write(footer)
            with open(job.latest_log_path, "a", encoding="utf-8") as f:
                f.write(footer)

            with self._lock:
                job.returncode = returncode
                if returncode == 0:
                    job.status = "completed"
                    job.stage = "Selesai"
                    job.last_message = f"Ekstraksi sukses dalam {duration_str}!"
                elif job.status != "canceled":
                    job.status = "failed"
                    job.stage = "Gagal"
                    job.error_message = (
                        f"Proses berhenti dengan kode error {returncode}."
                    )
                    job.last_message = job.error_message

                job.save_status()

        except Exception as exc:
            logger.exception("Kesalahan pada worker background")
            with self._lock:
                job.status = "failed"
                job.stage = "Error Sistem"
                job.error_message = str(exc)
                job.last_message = f"Exception: {exc}"
                job.save_status()

    def _parse_line_progress(self, job: JobInfo, line: str) -> None:
        """Deteksi pola progres (halaman X / Y, slide, SQL, guardrail) dari baris log."""
        updated = False

        # 1. Pola Halaman PDF: "Memproses Halaman 3 / 10 dari 'dokumen.pdf'..."
        m_page = re.search(
            r"Memproses Halaman\s+(\d+)\s*/\s*(\d+)", line, re.IGNORECASE
        )
        if m_page:
            job.current_page = int(m_page.group(1))
            job.total_pages = int(m_page.group(2))
            job.stage = (
                f"Ekstraksi VLM Halaman {job.current_page} / {job.total_pages}"
            )
            job.last_message = line
            updated = True

        # 2. Pola Slide PPT: "[Vision PPT] [Slide 2/8] Memproses slide..."
        m_slide = re.search(
            r"\[Slide\s+(\d+)\s*/\s*(\d+)\]", line, re.IGNORECASE
        )
        if m_slide:
            job.current_page = int(m_slide.group(1))
            job.total_pages = int(m_slide.group(2))
            job.stage = (
                f"Ekstraksi VLM Slide {job.current_page} / {job.total_pages}"
            )
            job.last_message = line
            updated = True

        # 3. Pola Total Slide selesai dirender: "Selesai render 8 slide gambar"
        m_slides_total = re.search(
            r"Selesai render\s+(\d+)\s+slide gambar", line, re.IGNORECASE
        )
        if m_slides_total:
            job.total_pages = int(m_slides_total.group(1))
            job.stage = (
                f"Rendering selesai ({job.total_pages} slide), memulai VLM..."
            )
            job.last_message = line
            updated = True

        # 4. Tahap Preprocessing & Rendering
        if "Mengekstrak PDF multi-halaman" in line:
            job.stage = "Inisialisasi PDF & Rendering Halaman"
            job.last_message = line
            updated = True
        elif "Memulai rendering slide menjadi gambar" in line:
            job.stage = "Rendering Slide Presentasi (LibreOffice)"
            job.last_message = line
            updated = True

        # 5. Tahap Sub-Agent SQL Tabular
        elif "Sub-Agent SQL" in line:
            job.stage = f"Sub-Agent SQL Ingesti (Halaman {job.current_page or 1})"
            job.last_message = line
            updated = True

        # 6. Tahap Guardrail Cross-Verification
        elif "Guardrail Cross-Verification" in line:
            job.stage = "Audit Guardrail Supervisor (Markdown vs SQLite)"
            job.last_message = line
            updated = True

        # 7. Tahap Simulasi Chunking
        elif "SIMULASI PEMBAGIAN CHUNKING" in line:
            job.stage = "Simulasi Chunking Siap RAG"
            job.last_message = line
            updated = True

        # 8. Markdown Berhasil Disimpan
        elif "Hasil Markdown berhasil disimpan ke:" in line:
            job.stage = "Penyimpanan Markdown Selesai"
            job.last_message = line
            updated = True

        if updated:
            job.save_status()

    def get_job(
        self, stem: str, output_dir: Path | None = None
    ) -> JobInfo | None:
        """Ambil info job dari memori atau rekonstruksi dari status.json di disk."""
        with self._lock:
            # 1. Cek dari memori
            if stem in self._jobs:
                job = self._jobs[stem]
                # Verifikasi jika status masih running, apakah proses OS benar-benar aktif
                if job.status == "running":
                    proc = self._processes.get(stem)
                    if proc is not None and proc.poll() is not None:
                        # Subprocess sudah selesai
                        returncode = proc.poll()
                        job.returncode = returncode
                        job.status = (
                            "completed" if returncode == 0 else "failed"
                        )
                        job.save_status()
                    elif not is_pid_alive(job.pid):
                        job.status = "failed"
                        job.stage = "Proses Berhenti Tak Terduga"
                        job.error_message = "Proses sistem telah terhenti."
                        job.save_status()
                return job

        # 2. Jika tidak ada di memori (misal server streamlit sempat restart), coba load dari disk
        target_dir = output_dir or (PROJECT_ROOT / "output")
        status_file = target_dir / stem / "logs" / f"{stem}_status.json"
        if not status_file.exists():
            status_file = target_dir / "logs" / f"{stem}_status.json"
        if status_file.exists():
            try:
                data = json.loads(status_file.read_text(encoding="utf-8"))
                latest_log = Path(
                    data.get(
                        "latest_log_path",
                        target_dir / "logs" / f"{stem}_latest.log",
                    )
                )
                recent = collections.deque(maxlen=200)
                if latest_log.exists():
                    lines = latest_log.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()
                    recent.extend(lines[-50:])

                job = JobInfo(
                    job_id=data.get("job_id", stem),
                    file_name=data.get("file_name", f"{stem}"),
                    input_path=Path(data.get("input_path", "")),
                    output_dir=Path(data.get("output_dir", target_dir)),
                    out_file=Path(
                        data.get("out_file", target_dir / f"{stem}.md")
                    ),
                    db_file=Path(data["db_file"])
                    if data.get("db_file")
                    else None,
                    log_path=Path(data.get("log_path", latest_log)),
                    latest_log_path=latest_log,
                    status_file=status_file,
                    progress_file=Path(
                        data.get(
                            "progress_file",
                            target_dir / "logs" / f"{stem}_progress.txt",
                        )
                    ),
                    status=data.get("status", "failed"),
                    current_page=data.get("current_page", 0),
                    total_pages=data.get("total_pages", 0),
                    stage=data.get("stage", "Tidak diketahui"),
                    last_message=data.get("last_message", ""),
                    started_at=data.get("started_at", ""),
                    updated_at=data.get("updated_at", ""),
                    pid=data.get("pid"),
                    returncode=data.get("returncode"),
                    error_message=data.get("error_message"),
                    recent_logs=recent,
                )

                # Validasi jika status tersimpan "running" tapi PID sudah mati
                if job.status == "running" and not is_pid_alive(job.pid):
                    job.status = "failed"
                    job.stage = "Proses Terhenti"
                    job.error_message = "Proses tidak aktif lagi di sistem."
                    job.save_status()

                with self._lock:
                    self._jobs[stem] = job
                return job
            except Exception as e:  # noqa: BLE001
                logger.warning("Gagal membaca status job dari disk: %s", e)

        return None

    def cancel_job(self, stem: str) -> bool:
        """Hentikan proses ekstraksi paksa jika sedang berjalan."""
        with self._lock:
            job = self._jobs.get(stem)
            proc = self._processes.get(stem)

        if not job or job.status != "running":
            return False

        # Matikan proses sistem operasi
        if proc and proc.poll() is None:
            try:
                # Di Windows, gunakan taskkill tree agar proses anak (soffice, python) ikut mati
                if sys.platform == "win32" and job.pid:
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(job.pid)],
                        capture_output=True,
                        check=False,
                    )
                proc.terminate()
            except Exception as e:  # noqa: BLE001
                logger.warning("Gagal mematikan proses: %s", e)

        with self._lock:
            job.status = "canceled"
            job.stage = "Dibatalkan oleh pengguna"
            job.last_message = "Ekstraksi dibatalkan."
            job.save_status()

        return True

    def reset_job(self, stem: str, output_dir: Path | None = None) -> None:
        """Hapus referensi job dari memori dan bersihkan file status agar dapat diekstrak ulang."""
        with self._lock:
            self._jobs.pop(stem, None)
            self._processes.pop(stem, None)
            self._threads.pop(stem, None)

        target_dir = output_dir or (PROJECT_ROOT / "output")
        candidate_files = [
            target_dir / stem / "logs" / f"{stem}_status.json",
            target_dir / "logs" / f"{stem}_status.json",
            target_dir / stem / "logs" / f"{stem}_progress.txt",
            target_dir / "logs" / f"{stem}_progress.txt",
        ]
        for f in candidate_files:
            try:
                if f.exists():
                    f.unlink()
            except Exception as e:  # noqa: BLE001
                logger.warning("Gagal menghapus file status saat reset_job: %s", e)

    def get_latest_logs(self, stem: str, line_count: int = 40) -> str:
        """Ambil potongan baris log terakhir (dari memori atau langsung dari file)."""
        job = self.get_job(stem)
        if job and job.recent_logs:
            return "\n".join(list(job.recent_logs)[-line_count:])

        # Fallback baca dari file disk
        log_file = PROJECT_ROOT / "output" / stem / "logs" / f"{stem}_latest.log"
        if not log_file.exists():
            log_file = PROJECT_ROOT / "output" / "logs" / f"{stem}_latest.log"
        if log_file.exists():
            try:
                lines = log_file.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
                return "\n".join(lines[-line_count:])
            except Exception:  # noqa: BLE001, S110
                pass
        return "(Belum ada log)"
