"""
Unit tests untuk modul analisis diagram & ekstraksi Mermaid.js (app/diagram.py).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from PIL import Image

from app.diagram import (
    classify_diagram_convertibility,
    extract_diagram_to_mermaid,
    sanitize_mermaid_code,
)
from app.schemas import (
    DiagramConvertibilityResult,
    DiagramExtractionResult,
)


def _create_dummy_image(path):
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    img.save(path, "PNG")


def test_sanitize_mermaid_code():
    raw = """
    ```mermaid
    flowchart TD
      A["Mulai"] --> B["Proses Data"]
      B --> C{"Valid?"}
      C -- Ya --> D["Selesai"]
      C -- Tidak --> B
    ```
    """
    sanitized = sanitize_mermaid_code(raw)
    assert sanitized.startswith("```mermaid\n")
    assert sanitized.endswith("\n```")
    assert "flowchart TD" in sanitized
    assert 'A["Mulai"] --> B["Proses Data"]' in sanitized


def test_classify_diagram_convertibility_flowchart(tmp_path):
    img_file = tmp_path / "flowchart_test.png"
    _create_dummy_image(img_file)

    mock_llm = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = """
    {
      "is_convertible": true,
      "diagram_type": "flowchart",
      "recommended_format": "mermaid",
      "mermaid_type": "flowchart TD",
      "confidence": 0.95,
      "reasoning": "Diagram alur proses logika login pengguna dengan keputusan cabang",
      "nodes_or_entities": ["Input Akun", "Validasi Password", "Dashboard"]
    }
    """
    mock_llm.invoke.return_value = mock_resp

    result = classify_diagram_convertibility(img_file, mock_llm)
    assert isinstance(result, DiagramConvertibilityResult)
    assert result.is_convertible is True
    assert result.diagram_type == "flowchart"
    assert result.mermaid_type == "flowchart TD"
    assert result.recommended_format == "mermaid"
    assert len(result.nodes_or_entities) == 3


def test_classify_diagram_unsuitable_chart(tmp_path):
    img_file = tmp_path / "scatter_plot.png"
    _create_dummy_image(img_file)

    mock_llm = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = """
    {
      "is_convertible": false,
      "diagram_type": "unsuitable_statistical_chart",
      "recommended_format": "markdown_table",
      "mermaid_type": null,
      "confidence": 0.98,
      "reasoning": "Grafik scatter plot data regresi 100 titik acak kontinu",
      "nodes_or_entities": []
    }
    """
    mock_llm.invoke.return_value = mock_resp

    result = classify_diagram_convertibility(img_file, mock_llm)
    assert result.is_convertible is False
    assert result.diagram_type == "unsuitable_statistical_chart"
    assert result.recommended_format == "markdown_table"
    assert result.mermaid_type is None


def test_extract_diagram_to_mermaid_success(tmp_path):
    img_file = tmp_path / "sequence_test.png"
    _create_dummy_image(img_file)

    mock_llm = MagicMock()

    # Call 1: classify
    resp_classify = MagicMock()
    resp_classify.content = """
    {
      "is_convertible": true,
      "diagram_type": "sequence_diagram",
      "recommended_format": "mermaid",
      "mermaid_type": "sequenceDiagram",
      "confidence": 0.92,
      "reasoning": "Sequence diagram autentikasi OAuth2 antara Client, AuthServer, dan ResourceServer",
      "nodes_or_entities": ["Client", "AuthServer", "ResourceServer"]
    }
    """

    # Call 2: extract
    resp_extract = MagicMock()
    resp_extract.content = """
    ```mermaid
    sequenceDiagram
      actor User
      participant Client as Web App
      participant Auth as Auth Server
      User->>Client: Login
      Client->>Auth: Request Token
      Auth-->>Client: JWT Access Token
    ```

    **Ringkasan Diagram:**
    Alur autentikasi token JWT antara User, Client Web, dan Server Otentikasi.
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
    assert "Alur autentikasi token JWT" in res.text_summary


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
    assert "Peta topografi wilayah" in res.text_summary
