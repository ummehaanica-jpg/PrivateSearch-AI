"""Text extraction for PDF, DOCX, TXT and Markdown.

Files are only ever *parsed as data*. Nothing from an uploaded document is
executed, imported or evaluated.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from .config import ALLOWED_EXTENSIONS


class ExtractionError(Exception):
    """Raised with a user-friendly message when a file cannot be read."""


@dataclass
class ExtractedPage:
    page: int | None  # 1-based page number, None when the format has no pages
    text: str


@dataclass
class ExtractedDocument:
    pages: list[ExtractedPage]
    n_pages: int | None


def sanitize_filename(name: str) -> str:
    base = os.path.basename((name or "").replace("\\", "/"))
    base = re.sub(r"[\x00-\x1f\x7f]", "", base).strip()
    return base[:150] or "document"


def extension_of(name: str) -> str:
    return Path(name).suffix.lower()


def validate_filename(name: str) -> str:
    ext = extension_of(name)
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(ALLOWED_EXTENSIONS)
        raise ExtractionError(f"Unsupported file type '{ext or 'none'}'. Supported types: {allowed}.")
    return ext


def clean_text(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join words hyphenated across lines
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_markdown(text: str) -> str:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"^\s*```.*$", "", text, flags=re.M)
    text = re.sub(r"(\*\*|__|\*|`)", "", text)
    return text


def _read_text_file(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def _extract_pdf(path: Path, max_pages: int) -> ExtractedDocument:
    try:
        import pymupdf as fitz  # PyMuPDF >= 1.24.3
    except ImportError:  # pragma: no cover - older releases
        import fitz  # type: ignore
    try:
        doc = fitz.open(path)
    except Exception as exc:
        raise ExtractionError(f"Could not open the PDF: {exc}") from exc
    with doc:
        if doc.needs_pass:
            raise ExtractionError("This PDF is password-protected. Remove the password and try again.")
        if doc.page_count > max_pages:
            raise ExtractionError(f"This PDF has {doc.page_count} pages; the limit is {max_pages}.")
        pages = []
        for number, page in enumerate(doc, start=1):
            text = clean_text(page.get_text("text"))
            if text:
                pages.append(ExtractedPage(number, text))
        return ExtractedDocument(pages, doc.page_count)


def _extract_docx(path: Path) -> ExtractedDocument:
    try:
        from docx import Document
    except ImportError as exc:  # pragma: no cover
        raise ExtractionError("python-docx is not installed. Run: pip install python-docx") from exc
    try:
        document = Document(str(path))
    except Exception as exc:
        raise ExtractionError(f"Could not open the Word document: {exc}") from exc
    blocks = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                blocks.append(" | ".join(cells))
    text = clean_text("\n\n".join(blocks))
    return ExtractedDocument([ExtractedPage(None, text)] if text else [], None)


def extract_text(path: Path, max_pdf_pages: int = 2000) -> ExtractedDocument:
    ext = validate_filename(path.name)
    if ext == ".pdf":
        result = _extract_pdf(path, max_pdf_pages)
    elif ext == ".docx":
        result = _extract_docx(path)
    else:
        text = _read_text_file(path)
        if ext in (".md", ".markdown"):
            text = _strip_markdown(text)
        text = clean_text(text)
        result = ExtractedDocument([ExtractedPage(None, text)] if text else [], None)
    if not result.pages:
        raise ExtractionError("No readable text was found. Scanned PDFs need OCR, which is not included.")
    return result
