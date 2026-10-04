import fitz
import pytest


@pytest.fixture
def sample_pdf_path(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "This is a sample PDF for testing the RAG pipeline.")
    doc.save(pdf_path)
    doc.close()
    return str(pdf_path)
