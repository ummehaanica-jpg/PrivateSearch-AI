import pymupdf
import pytest
from docx import Document

from privatesearch.chunking import chunk_pages
from privatesearch.extraction import ExtractedPage, ExtractionError, extract_text, sanitize_filename, validate_filename


def test_txt_and_markdown(tmp_path):
    (tmp_path / "a.txt").write_text("Hello world.\n\nSecond paragraph.", encoding="utf-8")
    (tmp_path / "b.md").write_text("# Title\n\nSome **bold** [link](http://x.y) text.", encoding="utf-8")
    assert "Second paragraph" in extract_text(tmp_path / "a.txt").pages[0].text
    md = extract_text(tmp_path / "b.md").pages[0].text
    assert "bold" in md and "**" not in md and "http" not in md and "#" not in md


def test_docx(tmp_path):
    doc = Document()
    doc.add_paragraph("Quantum computing basics.")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text, table.rows[0].cells[1].text = "qubit", "superposition"
    doc.save(tmp_path / "x.docx")
    text = extract_text(tmp_path / "x.docx").pages[0].text
    assert "Quantum computing" in text and "qubit | superposition" in text


def test_pdf_page_numbers(tmp_path):
    pdf = pymupdf.open()
    for body in ("Tomatoes need sun.", "Entanglement links particles."):
        pdf.new_page().insert_text((72, 100), body)
    pdf.save(tmp_path / "x.pdf")
    doc = extract_text(tmp_path / "x.pdf")
    assert [p.page for p in doc.pages] == [1, 2] and doc.n_pages == 2
    assert "Entanglement" in doc.pages[1].text


def test_blank_pdf_and_bad_files(tmp_path):
    blank = pymupdf.open()
    blank.new_page()
    blank.save(tmp_path / "blank.pdf")
    with pytest.raises(ExtractionError):
        extract_text(tmp_path / "blank.pdf")
    (tmp_path / "broken.pdf").write_bytes(b"not a pdf")
    with pytest.raises(ExtractionError):
        extract_text(tmp_path / "broken.pdf")
    with pytest.raises(ExtractionError):
        validate_filename("malware.exe")


def test_filename_sanitising():
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("C:\\Users\\me\\notes.txt") == "notes.txt"
    assert sanitize_filename("") == "document"


def test_chunking_properties():
    sentences = [f"This is sentence number {i} about topic {i % 3}." for i in range(80)]
    pages = [ExtractedPage(1, " ".join(sentences)), ExtractedPage(2, "Short page here.")]
    chunks = chunk_pages(pages, target_words=60, overlap_words=15)
    assert len(chunks) > 3
    assert all(c.text.strip() for c in chunks)
    assert all(c.n_words <= 75 for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert chunks[-1].page == 2 and all(c.page == 1 for c in chunks[:-1])
    # overlap: consecutive chunks on the same page share text
    assert chunks[0].text.split(". ")[-1][:15] in chunks[1].text
    sentence_text = " ".join(sentences)
    assert all(c.text in sentence_text for c in chunks[:-1])


def test_giant_sentence_is_split():
    chunks = chunk_pages([ExtractedPage(None, "word " * 1000)], target_words=100, overlap_words=10)
    assert len(chunks) >= 10 and all(c.n_words <= 110 for c in chunks)
