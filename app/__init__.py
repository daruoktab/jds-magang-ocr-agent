"""
Vision VLM & Document Text Extractor Package.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "AGENT_REGISTRY",
    "DocumentExtractionAgent",
    "get_agent",
    "batch_extract_documents",
    "scan_document_directories",
    "Settings",
    "get_settings",
    "build_deep_agent",
    "run_deep_reasoning_agent",
    "classify_diagram_convertibility",
    "extract_diagram_to_mermaid",
    "sanitize_mermaid_code",
    "VisionExtractor",
    "DocumentExtractionPipeline",
    "DocumentExtractionState",
    "format_page_delimiter",
    "preview_markdown_chunks",
    "split_markdown_by_pages",
    "stitch_pages_to_markdown",
    "convert_pdf_to_images",
    "extract_pdf",
    "pdf_to_images",
    "extract_presentation",
    "extract_presentation_native",
    "process_presentation",
    "process_presentation_vision",
    "ImagePreprocessor",
    "preprocess_document_image",
    "cross_verify_dual_track",
    "detect_transactional_tables",
    "export_tables_to_csv",
    "get_table_data",
    "ingest_markdown_tables_to_sqlite",
    "list_tables",
    "query_sqlite",
]

_MODULE_MAP: dict[str, str] = {
    "AGENT_REGISTRY": ".agents",
    "DocumentExtractionAgent": ".agents",
    "get_agent": ".agents",
    "batch_extract_documents": ".batch",
    "scan_document_directories": ".batch",
    "Settings": ".config",
    "get_settings": ".config",
    "build_deep_agent": ".deep_agent",
    "run_deep_reasoning_agent": ".deep_agent",
    "classify_diagram_convertibility": ".diagram",
    "extract_diagram_to_mermaid": ".diagram",
    "sanitize_mermaid_code": ".diagram",
    "VisionExtractor": ".extractor",
    "DocumentExtractionPipeline": ".graph",
    "DocumentExtractionState": ".graph",
    "format_page_delimiter": ".multi_page",
    "preview_markdown_chunks": ".multi_page",
    "split_markdown_by_pages": ".multi_page",
    "stitch_pages_to_markdown": ".multi_page",
    "convert_pdf_to_images": ".pdf",
    "extract_pdf": ".pdf",
    "pdf_to_images": ".pdf",
    "extract_presentation": ".ppt",
    "extract_presentation_native": ".ppt",
    "process_presentation": ".ppt",
    "process_presentation_vision": ".ppt",
    "ImagePreprocessor": ".preprocess",
    "preprocess_document_image": ".preprocess",
    "cross_verify_dual_track": ".tabular_db",
    "detect_transactional_tables": ".tabular_db",
    "export_tables_to_csv": ".tabular_db",
    "get_table_data": ".tabular_db",
    "ingest_markdown_tables_to_sqlite": ".tabular_db",
    "list_tables": ".tabular_db",
    "query_sqlite": ".tabular_db",
}


def __getattr__(name: str) -> Any:
    if name in _MODULE_MAP:
        mod = importlib.import_module(_MODULE_MAP[name], __name__)
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
