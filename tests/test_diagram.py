"""
Unit tests untuk Diagram Mermaid & Visual Artifacts Specialist Module.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from PIL import Image

from app.diagram import (
    classify_diagram_convertibility,
    extract_diagram_to_mermaid,
    get_diagram_recommendation,
    sanitize_mermaid_code,
    validate_mermaid_syntax,
)
from app.schemas import DiagramConvertibilityResult, DiagramExtractionResult


def _create_dummy_image(path: Path) -> Path:
    """Helper untuk membuat file gambar dummy untuk testing."""
    img = Image.new("RGB", (300, 200), color=(255, 255, 255))
    img.save(str(path))
    return path


def test_sanitize_mermaid_code():
    raw_with_block = """
    Berikut adalah hasil ekstraksinya:
    ```mermaid
    flowchart TD
        A[Start] --> B[Process]
        B --> C[End]
    ```
    Semoga membantu!
    """
    cleaned = sanitize_mermaid_code(raw_with_block)
    assert cleaned is not None
    assert cleaned.startswith("flowchart TD")
    assert "--> C[End]" in cleaned

    # Raw string tanpa code fence
    raw_plain = """flowchart LR
    A --> B
    """
    cleaned_plain = sanitize_mermaid_code(raw_plain)
    assert cleaned_plain is not None
    assert cleaned_plain.startswith("flowchart LR")


def test_validate_mermaid_syntax():
    valid_code = """flowchart TD
    A[User] -->|Auth| B(API Gateway)
    B --> C{Decision}
    C -->|Yes| D[Database]
    """
    is_valid, err = validate_mermaid_syntax(valid_code)
    assert is_valid is True
    assert err is None

    invalid_header = """randomHeader
    A --> B
    """
    is_valid_bad, err_bad = validate_mermaid_syntax(invalid_header)
    assert is_valid_bad is False
    assert "Header Mermaid tidak dikenali" in str(err_bad)


def test_get_diagram_recommendation():
    rec_flow = get_diagram_recommendation("flowchart")
    assert rec_flow.is_mermaid_compatible is True
    assert rec_flow.recommended_format == "mermaid_code"
    assert rec_flow.suggested_syntax == "flowchart TD"

    rec_unsuitable = get_diagram_recommendation("unsuitable_statistical_chart")
    assert rec_unsuitable.is_mermaid_compatible is False
    assert rec_unsuitable.recommended_format == "text_description"


def test_classify_diagram_convertibility(tmp_path):
    img_file = tmp_path / "diagram_test.png"
    _create_dummy_image(img_file)

    mock_llm = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = json.dumps(
        {
            "is_convertible": True,
            "diagram_type": "flowchart",
            "recommended_format": "mermaid",
            "mermaid_type": "flowchart",
            "confidence": 0.95,
            "reasoning": "Diagram alur proses login",
            "nodes_or_entities": ["User", "Login", "Dashboard"],
        }
    )
    mock_llm.invoke.return_value = mock_resp

    res = classify_diagram_convertibility(img_file, mock_llm)
    assert isinstance(res, DiagramConvertibilityResult)
    assert res.is_convertible is True
    assert res.diagram_type == "flowchart"
    assert res.mermaid_type == "flowchart"
    assert "User" in res.nodes_or_entities


def test_extract_diagram_to_mermaid_success(tmp_path):
    img_file = tmp_path / "seq_test.png"
    _create_dummy_image(img_file)

    mock_llm = MagicMock()

    # Step 1: Classify
    resp_classify = MagicMock()
    resp_classify.content = """
    {
      "is_convertible": true,
      "diagram_type": "sequence_diagram",
      "recommended_format": "mermaid",
      "mermaid_type": "sequenceDiagram",
      "confidence": 0.98,
      "reasoning": "Sequence interaksi user dan backend API",
      "nodes_or_entities": ["Client", "Server", "DB"]
    }
    """

    # Step 2: Extract
    resp_extract = MagicMock()
    resp_extract.content = """
    Berikut adalah kode Mermaid:
    ```mermaid
    sequenceDiagram
        autonumber
        Client->>Server: POST /login
        Server->>DB: Query User
        DB-->>Server: User Data
        Server-->>Client: JWT Access Token
    ```
    Alur autentikasi token JWT dari client ke database.
    """

    mock_llm.invoke.side_effect = [resp_classify, resp_extract]

    res = extract_diagram_to_mermaid(img_file, mock_llm)
    assert isinstance(res, DiagramExtractionResult)
    assert res.status == "success"
    assert res.is_mermaid is True
    assert res.diagram_type == "sequence_diagram"
    assert res.mermaid_code is not None
    assert "sequenceDiagram" in res.mermaid_code
    assert "JWT Access Token" in res.mermaid_code
    assert res.text_summary is not None and "Alur autentikasi token JWT" in res.text_summary


def test_extract_diagram_unsuitable_fallback(tmp_path):
    img_file = tmp_path / "map_test.png"
    _create_dummy_image(img_file)

    mock_llm = MagicMock()
    resp_classify = MagicMock()
    resp_classify.content = """
    {
      "is_convertible": false,
      "diagram_type": "unsuitable_map_or_spatial",
      "recommended_format": "text_description",
      "mermaid_type": null,
      "confidence": 0.99,
      "reasoning": "Peta topografi wilayah Jawa Barat dengan kontur ketinggian",
      "nodes_or_entities": []
    }
    """
    mock_llm.invoke.return_value = resp_classify

    res = extract_diagram_to_mermaid(img_file, mock_llm)
    assert res.status == "unsuitable"
    assert res.is_mermaid is False
    assert res.mermaid_code is None
    assert res.text_summary is not None and "Peta topografi wilayah" in res.text_summary
