"""
Unit tests untuk sanitasi thinking process model reasoning dan normalisasi tabel Markdown / SQLite.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.multi_page import strip_thinking_process
from app.tabular_db import parse_markdown_tables, sanitize_markdown_tables


def test_strip_thinking_process_complete():
    text = """
<think>
Wait, I need to check if there are any discrepancies.
Draft Table 4-2 Row 1: | 05h | PORTA | ...
Let me re-read carefully: ...
</think>

# PIC16F84A

## TABLE 4-1: PORTA FUNCTIONS
| Name | Bit0 |
| :--- | :--- |
| RA0 | bit0 |
"""
    cleaned = strip_thinking_process(text)
    assert "<think>" not in cleaned
    assert "</think>" not in cleaned
    assert "Wait, I need to check" not in cleaned
    assert cleaned.startswith("# PIC16F84A")
    assert "TABLE 4-1: PORTA FUNCTIONS" in cleaned


def test_strip_thinking_process_dangling_end():
    text = """
Wait, I need to check if there are any discrepancies.
Draft Table 4-2 Row 1: | 05h | PORTA | ...
</think>

# PIC16F84A
konten bersih
"""
    cleaned = strip_thinking_process(text)
    assert "</think>" not in cleaned
    assert "Wait, I need to check" not in cleaned
    assert cleaned.startswith("# PIC16F84A")


def test_strip_thinking_process_unclosed():
    text = """<think>
Thinking process:
Let's verify this table..."""
    cleaned = strip_thinking_process(text)
    assert cleaned == ""


def test_sanitize_markdown_tables_broken_subheaders_and_blank_lines():
    """
    Uji perbaikan tabel Table 2-1 (PIC16F84A):
    - Baris subheader '**Bank 1** | ...' tanpa pipa pembuka
    - Baris kosong sebelum Bank 1
    - Baris 00h/80h dengan fake separator pipes '|-----|'
    """
    raw_broken_table = """
<!-- sqlite_table: pic16f84a_t1 -->
| Addr | Name | Bit 7 | Bit 6 | Bit 5 | Bit 4 | Bit 3 | Bit 2 | Bit 1 | Bit 0 | Value on Power-on RESET | Details on page |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Bank 0** | | | | | | | | | | | |
| 00h | INDF | Uses contents of FSR | ----- | ----- | | | | | | ----- | ----- | |-----| |-----| |-----| | 11 |
| 01h | TMR0 | 8-bit Real-Time Clock/Counter | | | | | | | | xxxx | xxxx | 20 |
| 0Bh | INTCON | GIE | EEIE | T0IE | INTE | RBIE | T0IF | INTF | RBIF | 0000 | 000x | 10 |

**Bank 1** | | | | | | | | | | | |
| 80h | INDF | Uses Contents of FSR | ----- | ----- | | | | | | ----- | ----- | |-----| |-----| |-----| | 11 |
| 81h | OPTION_REG | RBPU | INTEDG | T0CS | T0SE | PSA | PS2 | PS1 | PS0 | 1111 | 1111 | 9 |
| 0Bh | INTCON | GIE | EEIE | T0IE | INTE | RBIE | T0IF | INTF | RBIF | 0000 | 000x | 10 |
"""
    sanitized = sanitize_markdown_tables(raw_broken_table)

    # 1. Pastikan Bank 1 diawali pipa
    assert "| **Bank 1** |" in sanitized
    # 2. Pastikan baris kosong di antara Bank 0 dan Bank 1 dihilangkan sehingga tabel tidak putus
    lines = [l.strip() for l in sanitized.splitlines() if l.strip().startswith("|")]
    # Header + Delimiter + Bank 0 + 00h + 01h + 0Bh + Bank 1 + 80h + 81h + 0Bh = 10 baris
    assert len(lines) == 10

    # 3. Uji ekstraksi SQLite: pastikan parse_markdown_tables membaca seluruh 8 baris data
    tables = parse_markdown_tables(raw_broken_table)
    assert len(tables) == 1
    table_dict = tables[0]
    assert len(table_dict["headers"]) == 12
    # Rows harus mencakup baris Bank 0 DAN Bank 1
    assert len(table_dict["rows"]) == 8

    # Cek bahwa baris Bank 1 masuk dan tidak terpotong
    row_bank1_header = table_dict["rows"][4]
    assert "**Bank 1**" in row_bank1_header[0]

    row_80h = table_dict["rows"][5]
    assert row_80h[0] == "80h"
    assert row_80h[1] == "INDF"

    row_81h = table_dict["rows"][6]
    assert row_81h[0] == "81h"
    assert row_81h[1] == "OPTION_REG"


if __name__ == "__main__":
    test_strip_thinking_process_complete()
    print("✓ test_strip_thinking_process_complete passed")
    test_strip_thinking_process_dangling_end()
    print("✓ test_strip_thinking_process_dangling_end passed")
    test_strip_thinking_process_unclosed()
    print("✓ test_strip_thinking_process_unclosed passed")
    test_sanitize_markdown_tables_broken_subheaders_and_blank_lines()
    print("✓ test_sanitize_markdown_tables_broken_subheaders_and_blank_lines passed")
    print("All tests in test_sanitization_and_tables.py passed successfully!")
