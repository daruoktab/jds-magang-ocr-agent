"""Streamlit launcher untuk menjalankan `main.py` pada file PDF/PPT."""

from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_TYPES = ["pdf", "ppt", "pptx"]


def _write_upload_to_temp(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="streamlit_launch_"))
    temp_path = temp_dir / Path(uploaded_file.name).name
    temp_path.write_bytes(uploaded_file.getvalue())
    return temp_path


def _write_run_log(
    log_path: Path,
    *,
    input_path: Path,
    output_dir: Path,
    cmd: list[str],
    returncode: int,
    stdout: str,
    stderr: str,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().isoformat(timespec="seconds")
    sections = [
        f"[{timestamp}] Streamlit launcher run",
        f"Input file: {input_path}",
        f"Output dir: {output_dir}",
        f"Command: {' '.join(cmd)}",
        f"Exit code: {returncode}",
        "",
        "[stdout]",
        stdout.strip() or "(empty)",
        "",
        "[stderr]",
        stderr.strip() or "(empty)",
        "",
    ]
    log_path.write_text("\n".join(sections), encoding="utf-8")


def _run_main_cli(
    input_path: Path,
    output_dir: Path,
    *,
    live_log: st.delta_generator.DeltaGenerator | None = None,
) -> tuple[int, str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{input_path.stem}.md"
    log_dir = output_dir / "logs"
    log_name = f"{input_path.stem}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = log_dir / log_name
    cmd = [
        sys.executable,
        "main.py",
        str(input_path),
        "-o",
        str(out_file),
    ]
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
    captured_lines: list[str] = []
    live_lines: list[str] = []
    if live_log is not None:
        live_log.empty()

    assert proc.stdout is not None
    for raw_line in iter(proc.stdout.readline, ""):
        if not raw_line:
            break
        captured_lines.append(raw_line)
        stripped = raw_line.rstrip("\n")
        live_lines.append(stripped)
        print(stripped, flush=True)
        if live_log is not None:
            live_log.code("\n".join(live_lines), language="text")

    returncode = proc.wait()
    combined = ["STDOUT/STDERR (merged):"]
    if captured_lines:
        combined.extend(captured_lines)
    else:
        combined.append("(empty)")
    combined.append(f"Output markdown: {out_file}")
    _write_run_log(
        log_path,
        input_path=input_path,
        output_dir=output_dir,
        cmd=cmd,
        returncode=returncode,
        stdout="".join(captured_lines),
        stderr="",
    )
    return returncode, "".join(combined), log_path


def main() -> None:
    st.set_page_config(page_title="Main CLI Launcher", page_icon="D", layout="wide")
    st.title("Main CLI Launcher")
    st.caption("Upload PDF/PPT lalu jalankan `main.py` dengan output folder yang dipilih.")

    with st.sidebar:
        st.header("Input")
        uploaded_file = st.file_uploader(
            "Pilih file PDF atau PPT",
            type=SUPPORTED_TYPES,
            help="File akan dijalankan lewat `main.py`.",
        )
        output_dir_input = st.text_input("Folder output", value="output")
        process_clicked = st.button("Run main.py", type="primary", use_container_width=True)

    if not process_clicked:
        st.info("Upload file lalu klik Run main.py.")
        return

    if uploaded_file is None:
        st.warning("Pilih file terlebih dahulu.")
        return

    input_ext = Path(uploaded_file.name).suffix.lower().lstrip(".")
    if input_ext not in SUPPORTED_TYPES:
        st.error("Hanya file PDF, PPT, atau PPTX yang didukung.")
        return

    output_dir = Path(output_dir_input).expanduser().resolve()
    temp_input = _write_upload_to_temp(uploaded_file)

    st.subheader("Hasil Eksekusi")
    live_log = st.empty()
    with st.spinner("Menjalankan main.py..."):
        code, output, log_path = _run_main_cli(
            temp_input,
            output_dir,
            live_log=live_log,
        )

    st.write(f"Exit code: {code}")
    st.write(f"Log file: {log_path}")
    st.code(output, language="text")

    out_file = output_dir / f"{Path(uploaded_file.name).stem}.md"
    if out_file.exists():
        st.success(f"Markdown tersimpan di: {out_file}")
        st.download_button(
            "Download Markdown",
            data=out_file.read_bytes(),
            file_name=out_file.name,
            mime="text/markdown",
        )


if __name__ == "__main__":
    main()
