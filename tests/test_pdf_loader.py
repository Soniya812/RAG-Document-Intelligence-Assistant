from src.pdf_loader import load_pdf_text, normalize_text


def test_load_pdf_text_missing(tmp_path):
    p = tmp_path / "empty.pdf"
    p.write_bytes(b"%PDF-1.4\n%EOF\n")
    try:
        _ = load_pdf_text(str(p))
        assert False, "Expected ValueError for empty PDF"
    except ValueError:
        assert True


def test_load_pdf_text_sample(sample_pdf_path):
    text = load_pdf_text(sample_pdf_path)
    assert isinstance(text, str)
    assert len(text) > 0


def test_normalize_text_removes_extra_whitespace():
    raw = "  This   is\n\n\n a   test.\t\nAnother  line.  "
    cleaned = normalize_text(raw)
    assert "  " not in cleaned
    assert "\n\n" not in cleaned
    assert cleaned.startswith("This")
    assert "Another line." in cleaned
