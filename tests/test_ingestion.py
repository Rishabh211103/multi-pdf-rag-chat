import pytest

from app.ingestion import IngestionError, clean_text, compute_doc_id, make_splitter, parse_pdf


def test_doc_id_is_deterministic(make_pdf):
    data = make_pdf(["hello world"])
    assert compute_doc_id(data) == compute_doc_id(data)
    assert compute_doc_id(data) != compute_doc_id(make_pdf(["something else"]))


def test_rejects_non_pdf():
    with pytest.raises(IngestionError, match="not a valid PDF"):
        parse_pdf(b"just some text", "notes.txt", make_splitter(200, 20))


def test_rejects_pdf_without_text(make_pdf):
    with pytest.raises(IngestionError, match="No extractable text"):
        parse_pdf(make_pdf(["", ""]), "scan.pdf", make_splitter(200, 20))


def test_chunks_carry_page_metadata(make_pdf):
    page_one = "\n".join(f"Line {i}: the mitochondria is the powerhouse of the cell." for i in range(20))
    data = make_pdf([page_one, "", "Final page about photosynthesis."])

    parsed = parse_pdf(data, "bio.pdf", make_splitter(200, 20))

    assert parsed.pages == 3
    assert len(parsed.chunks) > 2  # page one is split into several chunks
    assert {c.metadata["page"] for c in parsed.chunks} == {1, 3}  # empty page 2 skipped
    assert all(c.metadata["source"] == "bio.pdf" for c in parsed.chunks)
    assert [c.metadata["chunk"] for c in parsed.chunks] == list(range(len(parsed.chunks)))


def test_clean_text_joins_hyphenation_and_whitespace():
    assert clean_text("retrie-\nval   augmented\n\n\n\ngeneration") == "retrieval augmented\n\ngeneration"
