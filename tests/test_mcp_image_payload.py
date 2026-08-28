"""Unit tests untuk optimasi gambar & proteksi payload buffer MCP di app/mcp_agent_server.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app.mcp_agent_server import (
    _build_image_result,
    _image_to_content,
    _mime_for,
)


def test_mime_for_extensions():
    assert _mime_for(Path("test.png")) == "image/png"
    assert _mime_for(Path("test.jpg")) == "image/jpeg"
    assert _mime_for(Path("test.jpeg")) == "image/jpeg"
    assert _mime_for(Path("test.webp")) == "image/webp"
    assert _mime_for(Path("test.unknown")) == "image/jpeg"


def test_image_to_content_optimizes_png(tmp_path: Path):
    # Buat PNG besar buatan dengan gradien/konten
    png_path = tmp_path / "test_slide.png"
    img = Image.new("RGBA", (2600, 1500), (240, 240, 240, 255))
    img.save(png_path, format="PNG")

    content = _image_to_content(png_path, max_dimension=2400, jpeg_quality=90)
    assert content is not None
    assert content.type == "image"
    # Harus diubah ke JPEG berkualitas tinggi dan dimensi ter-downscale
    assert content.mime_type == "image/jpeg"
    # Base64 string tidak boleh kosong
    assert len(content.data) > 0


def test_image_to_content_transparent_png(tmp_path: Path):
    # Buat PNG dengan transparansi
    png_path = tmp_path / "transparent.png"
    img = Image.new("RGBA", (200, 200), (255, 0, 0, 0))  # Transparan
    img.save(png_path, format="PNG")

    content = _image_to_content(png_path)
    assert content is not None
    assert len(content.data) > 0


def test_build_image_result_payload_guard(tmp_path: Path):
    # Buat 3 file gambar
    img_paths: list[Path] = []
    for i in range(3):
        p = tmp_path / f"img_{i}.jpg"
        img = Image.new("RGB", (800, 600), (i * 50, 100, 150))
        img.save(p, format="JPEG", quality=85)
        img_paths.append(p)

    summary = {"total_images": 3}

    # Uji saat batas payload normal (semua masuk)
    results = _build_image_result(summary.copy(), img_paths)
    # 1 TextContent + 3 ImageContent = 4 blocks
    assert len(results) == 4
    assert "payload_warning" not in summary

    # Uji saat batas payload sangat kecil (terjadi throttling aman)
    with patch("app.mcp_agent_server.MAX_TOTAL_PAYLOAD_BYTES", 1000):
        throttled_summary = {"total_images": 3}
        results_throttled = _build_image_result(throttled_summary, img_paths)
        # Kurang dari 4 blocks karena dibatasi pengaman
        assert len(results_throttled) < 4
        assert "payload_warning" in throttled_summary


def test_select_document_batch_files_and_folders(tmp_path: Path):
    import json

    from app.mcp_agent_server import select_document_batch

    folder = tmp_path / "folder_a"
    folder.mkdir()
    f1 = folder / "doc1.pdf"
    f2 = folder / "doc2.pptx"
    f1.touch()
    f2.touch()

    f_standalone = tmp_path / "single.pdf"
    f_standalone.touch()

    # Panggil dengan campuran folder dan file
    input_str = f"{folder}, {f_standalone}"
    res = json.loads(select_document_batch(input_str))

    assert res["total_selected_files"] == 3
    assert str(f_standalone.resolve()) in res["selected_files"]
    assert str(f1.resolve()) in res["selected_files"]
    assert str(f2.resolve()) in res["selected_files"]

    # Test file_indices
    res_idx = json.loads(select_document_batch(str(folder), file_indices="2"))
    assert res_idx["total_selected_files"] == 1
    assert str(f2.resolve()) in res_idx["selected_files"]

    # Test pattern
    res_pat = json.loads(select_document_batch(str(folder), pattern="*doc1*"))
    assert res_pat["total_selected_files"] == 1
    assert str(f1.resolve()) in res_pat["selected_files"]

    # Test limit slicing across per_source
    res_lim = json.loads(select_document_batch(str(folder), limit=1))
    assert res_lim["total_selected_files"] == 1
    assert len(res_lim["selected_files"]) == 1
    assert res_lim["per_source"][0]["selected_count"] == 1
    assert len(res_lim["per_source"][0]["files"]) == 1


def test_find_documents_keyword_search(tmp_path: Path):
    import json

    from app.mcp_agent_server import find_documents

    target_dir = tmp_path / "my_ppt_folder"
    target_dir.mkdir()
    (target_dir / "44_PPT UJIAN SKIRPSI CINDI FATIKASARI.pptx").touch()
    (target_dir / "10_LAPORAN KEUANGAN.pdf").touch()

    # Search by keyword "cindi"
    res_raw = find_documents(query="cindi", root_dir=str(tmp_path))
    res = json.loads(res_raw)
    assert res["status"] == "success"
    assert res["total_matches"] == 1
    assert "CINDI" in res["matches"][0]["file_name"]

    # Search by number "44"
    res_num_raw = find_documents(query="44", root_dir=str(tmp_path))
    res_num = json.loads(res_num_raw)
    assert res_num["status"] == "success"
    assert res_num["total_matches"] == 1
    assert "44_PPT" in res_num["matches"][0]["file_name"]


def test_select_random_documents_files_and_folders(tmp_path: Path):
    import json

    from app.mcp_agent_server import select_random_documents

    folder = tmp_path / "folder_b"
    folder.mkdir()
    f1 = folder / "doc1.pdf"
    f1.touch()

    f_single = tmp_path / "single.png"
    f_single.touch()

    input_str = f"{folder}, {f_single}"
    res = json.loads(select_random_documents(input_str, count=2, seed=42))

    assert res["status"] == "success"
    assert res["selected_count"] == 2
    assert len(res["selected_files"]) == 2


def test_open_file_dialog_success_and_cancel(tmp_path: Path):
    import json
    from unittest.mock import patch

    from app.mcp_agent_server import open_file_dialog

    # Test success
    fake_file = str((tmp_path / "test.pdf").resolve())
    with patch(
        "app.mcp_agent_server._show_open_file_dialog_native", return_value=[fake_file]
    ):
        res = json.loads(open_file_dialog())
        assert res["status"] == "success"
        assert res["selected_count"] == 1
        assert res["selected_files"] == [fake_file]

    # Test cancel
    with patch("app.mcp_agent_server._show_open_file_dialog_native", return_value=[]):
        res = json.loads(open_file_dialog())
        assert res["status"] == "cancelled"
        assert res["selected_count"] == 0


def test_open_folder_dialog_success_and_cancel(tmp_path: Path):
    import json
    from unittest.mock import patch

    from app.mcp_agent_server import open_folder_dialog

    folder = tmp_path / "my_docs"
    folder.mkdir()
    (folder / "sample.pdf").touch()

    # Test success
    with patch(
        "app.mcp_agent_server._show_open_folder_dialog_native",
        return_value=str(folder.resolve()),
    ):
        res = json.loads(open_folder_dialog())
        assert res["status"] == "success"
        assert res["selected_folder"] == str(folder.resolve())
        assert res["total_documents"] == 1

    # Test cancel
    with patch(
        "app.mcp_agent_server._show_open_folder_dialog_native", return_value=None
    ):
        res = json.loads(open_folder_dialog())
        assert res["status"] == "cancelled"
        assert res["selected_folder"] is None


def test_agent_document_graph_render_and_save(tmp_path: Path):
    import json

    from app.mcp_agent_server import process_document_batch, save_extraction_result

    # Buat file gambar uji
    img_file = tmp_path / "test_doc.png"
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    img.save(img_file, format="PNG")

    # Uji process_document_batch via LangGraph
    blocks = process_document_batch(str(img_file), output_dir=str(tmp_path / "out"))
    assert isinstance(blocks, list)
    # 1 Text summary + 1 Image content
    assert len(blocks) == 2

    # Uji save_extraction_result via LangGraph
    md_text = "# Test Header\n\nSample content for extraction testing."
    save_res_str = save_extraction_result(
        str(img_file),
        markdown=md_text,
        output_dir=str(tmp_path / "gold_out"),
    )
    save_res = json.loads(save_res_str)
    assert save_res["status"] == "success"
    assert save_res["is_complete"] is True
    assert (tmp_path / "gold_out" / "test_doc.md").exists()


def test_tabular_multi_page_append_continuity(tmp_path: Path):
    from app.tabular_db import (
        TabularDatabaseManager,
        extract_and_ingest_tables_from_markdown,
    )

    db_file = tmp_path / "test_ledger.sqlite"
    doc_src = str(tmp_path / "rekening_koran.pdf")

    # Page 1 Markdown Table
    md_page1 = """
# Rekening Koran Halaman 1
| Tanggal | Uraian Transaksi | Debit | Kredit | Saldo |
| --- | --- | --- | --- | --- |
| 2024-01-01 | Setoran Awal | 0 | 100,000,000 | 100,000,000 |
| 2024-01-02 | Pembayaran Sewa | 25,000,000 | 0 | 75,000,000 |
"""
    res1 = extract_and_ingest_tables_from_markdown(
        md_page1,
        source_file=doc_src,
        db_path=db_file,
        page_number=1,
        append_if_matching=True,
    )
    assert len(res1) == 1
    assert res1[0].total_rows_ingested == 2

    # Page 2 Markdown Table (Lanjutan tabel yang sama pada halaman 2)
    md_page2 = """
# Rekening Koran Halaman 2 (Lanjutan)
| Tanggal | Uraian Transaksi | Debit | Kredit | Saldo |
| --- | --- | --- | --- | --- |
| 2024-01-05 | Penerimaan Invoice #102 | 0 | 50,000,000 | 125,000,000 |
| 2024-01-10 | Biaya Operasional | 10,000,000 | 0 | 115,000,000 |
| 2024-01-12 | Pembelian Peralatan | 15,000,000 | 0 | 100,000,000 |
"""
    res2 = extract_and_ingest_tables_from_markdown(
        md_page2,
        source_file=doc_src,
        db_path=db_file,
        page_number=2,
        append_if_matching=True,
    )
    assert len(res2) == 1
    assert res2[0].total_rows_ingested == 3
    # Harus di-append ke tabel yang sama, bukan membuat tabel baru
    assert res2[0].table_name == res1[0].table_name

    # Verifikasi total baris di database SQLite tunggal
    db_mgr = TabularDatabaseManager(db_file)
    active_summary = db_mgr.get_active_tables_summary()
    assert len(active_summary) == 1
    assert active_summary[0]["total_rows"] == 5

    # Query agregasi SQL
    q_res = db_mgr.execute_query(
        f'SELECT COUNT(*) as total_count FROM "{res1[0].table_name}"'
    )
    assert q_res.rows[0]["total_count"] == 5


def test_start_and_submit_page_chaining(tmp_path: Path):
    import json

    from app.mcp_agent_server import start_document_extraction, submit_page_and_get_next

    # Buat file gambar uji 1 halaman
    img_file = tmp_path / "single_doc.png"
    img = Image.new("RGB", (300, 300), (255, 255, 255))
    img.save(img_file, format="PNG")

    # Inisialisasi ekstraksi
    start_blocks = start_document_extraction(
        str(img_file), output_dir=str(tmp_path / "out")
    )
    assert isinstance(start_blocks, list)
    assert len(start_blocks) == 2

    # Submit halaman 1 (finalisasi dokumen karena 1 halaman)
    submit_res_str = submit_page_and_get_next(
        source_file=str(img_file),
        page_number=1,
        markdown="# Halaman 1 Selesai\n\nTeks konten dokumen.",
        output_dir=str(tmp_path / "agent_gold"),
    )
    assert isinstance(submit_res_str, str)
    submit_res = json.loads(submit_res_str)
    assert submit_res["status"] == "success"
    assert submit_res["is_complete"] is True
    assert (tmp_path / "agent_gold" / "single_doc.md").exists()


def test_multi_page_sequential_advance(tmp_path: Path):
    import json
    from unittest.mock import patch

    from app.mcp_agent_server import start_document_extraction, submit_page_and_get_next

    # Buat file mock presentasi
    pptx_file = tmp_path / "test_presentation.pptx"
    pptx_file.touch()

    # Mock count_presentation_slides = 3, render_presentation_slides_to_images = [img1, img2, img3]
    fake_img1 = tmp_path / "slide_1.jpg"
    fake_img2 = tmp_path / "slide_2.jpg"
    fake_img3 = tmp_path / "slide_3.jpg"
    for f in [fake_img1, fake_img2, fake_img3]:
        Image.new("RGB", (200, 200), (255, 255, 255)).save(f, format="JPEG")

    def mock_render(src, output_dir, slides, dpi, image_ext):
        mapping = {0: fake_img1, 1: fake_img2, 2: fake_img3}
        return [mapping[s] for s in slides if s in mapping]

    with (
        patch("app.agent_graph.count_presentation_slides", return_value=3),
        patch(
            "app.agent_graph.render_presentation_slides_to_images",
            side_effect=mock_render,
        ),
    ):
        # 1. Start extraction (Page 1)
        res_start = start_document_extraction(
            str(pptx_file), output_dir=str(tmp_path / "slides_out")
        )
        assert isinstance(res_start, list)
        assert len(res_start) == 2

        from mcp.types import TextContent

        # 2. Submit Slide 1 (Raw Markdown without <!-- SLIDE: 1 --> tag)
        res_sub1 = submit_page_and_get_next(
            source_file=str(pptx_file),
            page_number=1,
            markdown="# JUDUL PRESENTASI\n\nPenulis: Cindi",
            output_dir=str(tmp_path / "gold_out"),
        )
        assert isinstance(res_sub1, list)
        assert isinstance(res_sub1[0], TextContent)
        summary1 = json.loads(res_sub1[0].text)
        assert summary1["status"] == "in_progress"
        assert summary1["page_saved"] == 1
        assert summary1["current_page"] == 2
        assert summary1["pages_saved_so_far"] == [1]
        assert summary1["missing_pages"] == [2, 3]
        assert "subagent_directives" in summary1["subagent_task_directives"]
        assert (
            "mermaid_specialist"
            in summary1["subagent_task_directives"]["subagent_directives"]
        )

        # 3. Submit Slide 2 (Raw Markdown without <!-- SLIDE: 2 --> tag)
        res_sub2 = submit_page_and_get_next(
            source_file=str(pptx_file),
            page_number=2,
            markdown="# POKOK PEMBAHASAN\n\n1. Latar Belakang\n2. Tujuan",
            output_dir=str(tmp_path / "gold_out"),
        )
        assert isinstance(res_sub2, list)
        assert isinstance(res_sub2[0], TextContent)
        summary2 = json.loads(res_sub2[0].text)
        assert summary2["status"] == "in_progress"
        assert summary2["page_saved"] == 2
        assert summary2["current_page"] == 3
        assert summary2["pages_saved_so_far"] == [1, 2]
        assert summary2["missing_pages"] == [3]
        assert (
            "tabular_sqlite_specialist"
            in summary2["subagent_task_directives"]["subagent_directives"]
        )

        # 4. Submit Slide 3 (Final Slide)
        res_sub3 = submit_page_and_get_next(
            source_file=str(pptx_file),
            page_number=3,
            markdown="# KESIMPULAN\n\nHasil penelitian valid.",
            output_dir=str(tmp_path / "gold_out"),
        )
        assert isinstance(res_sub3, str)
        summary3 = json.loads(res_sub3)
        assert summary3["status"] == "success"
        assert summary3["is_complete"] is True
        assert summary3["pages_saved"] == [1, 2, 3]

        # 5. Verifikasi isi file Markdown hasil stitching
        final_md_path = tmp_path / "gold_out" / "test_presentation.md"
        assert final_md_path.exists()
        final_md_text = final_md_path.read_text(encoding="utf-8")
        assert "<!-- SLIDE: 1 -->" in final_md_text
        assert "JUDUL PRESENTASI" in final_md_text
        assert "<!-- SLIDE: 2 -->" in final_md_text
        assert "POKOK PEMBAHASAN" in final_md_text
        assert "<!-- SLIDE: 3 -->" in final_md_text
        assert "KESIMPULAN" in final_md_text

        # 6. Test read_document_extraction tool
        from app.mcp_agent_server import read_document_extraction

        read_res = json.loads(
            read_document_extraction(
                str(pptx_file), output_dir=str(tmp_path / "gold_out")
            )
        )
        assert read_res["status"] == "success"
        assert read_res["is_complete"] is True
        assert "JUDUL PRESENTASI" in read_res["markdown_content"]

        # 7. Fresh re-extraction starting from Page 1 on the same file does NOT inherit prior completion
        res_fresh_sub1 = submit_page_and_get_next(
            source_file=str(pptx_file),
            page_number=1,
            markdown="# JUDUL BARU\n\nRe-ekstraksi bersih.",
            output_dir=str(tmp_path / "gold_out"),
        )
        assert isinstance(res_fresh_sub1, list)
        assert isinstance(res_fresh_sub1[0], TextContent)
        fresh_summary = json.loads(res_fresh_sub1[0].text)
        assert fresh_summary["status"] == "in_progress"
        assert fresh_summary["is_complete"] is False
        assert fresh_summary["page_saved"] == 1
        assert fresh_summary["current_page"] == 2
        assert fresh_summary["pages_saved_so_far"] == [1]
        assert fresh_summary["missing_pages"] == [2, 3]
