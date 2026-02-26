"""
Igris AI Agent — Document Reader Skill  (Issue #3 — document reading)

Supports reading:
- PDF files (via PyPDF2)
- DOCX files (via python-docx)
- TXT / MD files (plain text)
- CSV files (via csv module)

Each reader extracts text content that can then be fed to the LLM for
summarization, Q&A, or processing.
"""

import os
import csv
from langchain.tools import tool

# Lazy imports for optional dependencies
def _read_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        return "PyPDF2 not installed. Run: pip install PyPDF2"

    reader = PdfReader(file_path)
    text_parts = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text_parts.append(f"--- Page {i + 1} ---\n{page_text}")
    if not text_parts:
        return "PDF appears to contain no extractable text (may be scanned/image-based)."
    return "\n\n".join(text_parts)


def _read_docx(file_path: str) -> str:
    """Extract text from a DOCX file."""
    try:
        from docx import Document
    except ImportError:
        return "python-docx not installed. Run: pip install python-docx"

    doc = Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    if not paragraphs:
        return "DOCX appears to be empty."
    return "\n\n".join(paragraphs)


def _read_csv(file_path: str) -> str:
    """Read a CSV file and return it as formatted text."""
    rows = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i >= 100:  # limit to first 100 rows
                rows.append(f"... (truncated, {i}+ rows total)")
                break
            rows.append(" | ".join(row))
    return "\n".join(rows) if rows else "CSV file is empty."


def _read_text(file_path: str) -> str:
    """Read a plain text or markdown file."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    if len(content) > 50_000:
        return content[:50_000] + f"\n\n... (truncated, total {len(content):,} characters)"
    return content


@tool
def read_document(file_path: str) -> str:
    """Read a document file (PDF, DOCX, TXT, CSV, MD) and return its text content.
    Use this when the user provides a document to read, summarize, or analyze."""
    try:
        file_path = os.path.abspath(file_path)
        if not os.path.exists(file_path):
            return f"File not found: {file_path}"

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            content = _read_pdf(file_path)
        elif ext == ".docx":
            content = _read_docx(file_path)
        elif ext == ".csv":
            content = _read_csv(file_path)
        elif ext in (".txt", ".md", ".log", ".py", ".js", ".json", ".xml", ".html"):
            content = _read_text(file_path)
        else:
            return f"Unsupported file type: {ext}. Supported: PDF, DOCX, TXT, CSV, MD"

        return f"📄 Document: {os.path.basename(file_path)} ({ext})\n\n{content}"
    except Exception as e:
        return f"Error reading document: {e}"


DOCUMENT_READER_TOOLS = [read_document]
