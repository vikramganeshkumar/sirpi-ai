#!/usr/bin/env python3
"""
Shared helper functions for the PDF RAG system.

Anything used in more than one file lives here so it's defined once:
API key resolution, PDF text extraction, text truncation, console
formatting, JSON saving, and logging setup.
"""
import json
import logging
import os
import sys
from typing import Optional

try:
    from pypdf import PdfReader
except ImportError:
    print("Error: pypdf not installed. Run: pip install -r requirements.txt")
    sys.exit(1)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract and concatenate text from every page of a PDF."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(pdf_path)
    logging.info("PDF loaded: %d pages", len(reader.pages))

    pages_text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages_text.append(page_text)
    return "\n".join(pages_text)


def truncate_text(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, breaking at the last whitespace when possible
    so words / sentences aren't chopped mid-token before hitting the model."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_space = cut.rfind(" ")
    return cut[:last_space] if last_space > max_chars * 0.8 else cut


def print_section(title: str, char: str = "=", width: int = 80) -> None:
    """Consistent section headers instead of each script hand-rolling '='*80."""
    print(char * width)
    print(title)
    print(char * width)


def save_json(data, path: str) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logging.info("Saved results to %s", path)


def setup_logging(log_file: str = "rag_system.log", level: int = logging.INFO) -> None:
    """Idempotent logging setup -- safe to call from multiple entry points
    without duplicating handlers."""
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
