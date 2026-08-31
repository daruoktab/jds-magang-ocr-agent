"""Streamlit UI for the existing document extraction pipeline.

This module is intentionally kept as an adapter layer. Domain logic remains in
the existing ``app`` package and the CLI entrypoint is not modified.
"""

from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

from app.config import get_settings, setup_logging
from app.graph import DocumentExtractionPipeline
from app.multi_page import preview_markdown_chunks
from app.ocr import build_ocr_extractor
from app.pdf import pdf_to_images, process_multipage_pdf

SUPPORTED_TYPES = ["pdf", "png", "jpg", "jpeg"]
SPEC_OPTIONS = {
    "Auto-detect": None,
    "Plain document": "plain",
    "Markdown hierarchy": "markdown_hierarchy",
    "Bilingual journal": "bilingual_journal",
    "Presentation slides": "presentation_slides",
}


class _StreamlitLogHandler(logging.Handler):
    """Collect pipeline logs so they can be shown in the UI."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(self.format(record))


def _model_to_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return dict(value)


def _extract_document(
    input_path: Path,
    *,
    mode: str,
    dpi: int,
    forced_specs: str | None,
) -> dict[str, Any]:
    """Run the existing extraction/OCR APIs and normalize their UI output."""
    settings = get_settings()

    if mode == "OCR only":
        ocr = build_ocr_extractor(settings)
        if input_path.suffix.lower() == ".pdf":
            page_images = pdf_to_images(input_path, output_dir=input_path.parent, dpi=dpi)
            page_texts = [
                f"--- Halaman {index} ---\n{ocr.extract(str(page)).text}"
                for index, page in enumerate(page_images, start=1)
            ]
            markdown = "\n\n".join(page_texts)
            total_pages = len(page_images)
        else:
            markdown = ocr.extract(str(input_path)).text
            total_pages = 1

        return {
            "markdown_content": markdown,
            "specs": ["ocr"],
            "total_pages": total_pages,
            "pages": [],
            "metadata": {"mode": mode, "dpi": dpi},
        }

    pipeline = DocumentExtractionPipeline(settings)
    if input_path.suffix.lower() == ".pdf":
        extracted = process_multipage_pdf(
            pdf_path=input_path,
            pipeline=pipeline,
            output_dir=input_path.parent,
            dpi=dpi,
            forced_specs=forced_specs,
        )
        result = _model_to_dict(extracted)
    else:
        result = pipeline.run(str(input_path), forced_specs=forced_specs)

    return {
        "markdown_content": result.get("markdown_content", ""),
        "specs": result.get("specs", ["plain"]),
        "total_pages": result.get("total_pages", 1),
        "pages": result.get("pages", []),
        "metadata": result.get("metadata", {}),
        "ocr_text": result.get("ocr_text"),
    }


def _run_with_logs(
    file_name: str,
    file_bytes: bytes,
    *,
    mode: str,
    dpi: int,
    forced_specs: str | None,
) -> tuple[dict[str, Any], list[str]]:
    log_handler = _StreamlitLogHandler()
    log_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | [%(name)s] %(message)s", "%H:%M:%S")
    )
    setup_logging()
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)

    try:
        with tempfile.TemporaryDirectory(prefix="streamlit_ocr_") as temp_dir:
            input_path = Path(temp_dir) / Path(file_name).name
            input_path.write_bytes(file_bytes)
            result = _extract_document(
                input_path,
                mode=mode,
                dpi=dpi,
                forced_specs=forced_specs,
            )
        return result, log_handler.records
    finally:
        root_logger.removeHandler(log_handler)


def _render_chunks(markdown: str, chunk_size: int, chunk_overlap: int) -> None:
    if not markdown.strip():
        st.info("Belum ada teks untuk dipecah menjadi chunk.")
        return

    chunks = preview_markdown_chunks(
        markdown,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    st.caption(f"{len(chunks)} chunk ditemukan")
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        title = f"Chunk #{chunk['chunk_index']} - {chunk['char_count']} karakter"
        with st.expander(title):
            if metadata:
                st.json(metadata)
            st.code(chunk["content"], language="markdown")


def main() -> None:
    st.set_page_config(
        page_title="Document Vision Extractor",
        page_icon="D",
        layout="wide",
    )
    st.title("Document Vision Extractor")
    st.caption("Streamlit adapter untuk pipeline OCR dan ekstraksi Markdown existing")

    with st.sidebar:
        st.header("Pengaturan")
        uploaded_file = st.file_uploader(
            "Pilih dokumen",
            type=SUPPORTED_TYPES,
            help="MVP mendukung satu PDF atau gambar per proses.",
        )
        mode = st.radio("Mode proses", ["Extract Markdown", "OCR only"])
        spec_label = st.selectbox("Spesifikasi layout", list(SPEC_OPTIONS))
        dpi = st.slider("DPI PDF", min_value=100, max_value=300, value=200, step=25)
        chunk_size = st.number_input(
            "Ukuran chunk", min_value=100, max_value=10000, value=1000, step=100
        )
        chunk_overlap = st.number_input(
            "Overlap chunk", min_value=0, max_value=5000, value=150, step=50
        )
        show_logs = st.checkbox("Tampilkan log detail", value=True)
        process_clicked = st.button("Process Document", type="primary", use_container_width=True)

    if process_clicked:
        if uploaded_file is None:
            st.warning("Pilih file terlebih dahulu.")
        elif chunk_overlap >= chunk_size:
            st.warning("Overlap chunk harus lebih kecil dari ukuran chunk.")
        else:
            with st.spinner("Memproses dokumen..."):
                try:
                    st.session_state.pop("error", None)
                    st.session_state.pop("result", None)
                    result, logs = _run_with_logs(
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        mode=mode,
                        dpi=dpi,
                        forced_specs=SPEC_OPTIONS[spec_label],
                    )
                    st.session_state["result"] = result
                    st.session_state["logs"] = logs
                    st.session_state["filename"] = uploaded_file.name
                    st.session_state.pop("error", None)
                except Exception as exc:  # noqa: BLE001
                    st.session_state.pop("result", None)
                    st.session_state["error"] = str(exc)

    if st.session_state.get("error"):
        st.error(st.session_state["error"])

    result = st.session_state.get("result")
    logs = st.session_state.get("logs", [])
    if not result:
        st.info("Upload dokumen lalu klik Process Document untuk memulai.")
        return

    markdown = result.get("markdown_content", "")
    specs = ", ".join(result.get("specs") or ["plain"])
    col1, col2, col3 = st.columns(3)
    col1.metric("Halaman", result.get("total_pages", 1))
    col2.metric("Karakter Markdown", len(markdown))
    col3.metric("Layout", specs)

    extract_tab, chunks_tab, logs_tab = st.tabs(["Extract", "Chunks", "Logs"])
    with extract_tab:
        st.subheader(st.session_state.get("filename", "Hasil ekstraksi"))
        if markdown.strip():
            st.markdown(markdown)
        else:
            st.info("Pipeline menghasilkan teks kosong.")
        st.download_button(
            "Download Markdown",
            data=io.BytesIO(markdown.encode("utf-8")),
            file_name=f"{Path(st.session_state.get('filename', 'document')).stem}.md",
            mime="text/markdown",
        )
        with st.expander("Lihat Markdown mentah"):
            st.code(markdown, language="markdown")

    with chunks_tab:
        _render_chunks(markdown, int(chunk_size), int(chunk_overlap))

    with logs_tab:
        if show_logs and logs:
            st.code("\n".join(logs), language="text")
        else:
            st.info("Log detail tidak ditampilkan atau belum tersedia.")


if __name__ == "__main__":
    main()
