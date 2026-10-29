"""PDF parsing and chunking."""

import hashlib
import re
from dataclasses import dataclass

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class IngestionError(ValueError):
    """Raised when an uploaded file cannot be turned into searchable chunks."""


@dataclass
class ParsedPDF:
    doc_id: str
    filename: str
    pages: int
    chunks: list[Document]


def make_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )


def compute_doc_id(data: bytes) -> str:
    """Content hash, so the same file uploaded twice maps to the same document."""
    return hashlib.sha256(data).hexdigest()[:16]


def extract_pages(data: bytes) -> list[str]:
    # The PDF spec allows the header anywhere in the first 1024 bytes.
    if b"%PDF" not in data[:1024]:
        raise IngestionError("File is not a valid PDF.")
    try:
        pdf = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise IngestionError(f"Could not read PDF: {exc}") from exc

    with pdf:
        if pdf.needs_pass:
            raise IngestionError("Password-protected PDFs are not supported.")
        return [page.get_text() for page in pdf]


def clean_text(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join words hyphenated across lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_pdf(data: bytes, filename: str, splitter: RecursiveCharacterTextSplitter) -> ParsedPDF:
    doc_id = compute_doc_id(data)
    pages = extract_pages(data)

    # Split page by page so every chunk maps to exactly one page for citations.
    page_docs = [
        Document(page_content=text, metadata={"doc_id": doc_id, "source": filename, "page": number})
        for number, raw in enumerate(pages, start=1)
        if (text := clean_text(raw))
    ]
    if not page_docs:
        raise IngestionError(
            "No extractable text found. Scanned/image-only PDFs need OCR, which is not supported yet."
        )

    chunks = splitter.split_documents(page_docs)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk"] = index

    return ParsedPDF(doc_id=doc_id, filename=filename, pages=len(pages), chunks=chunks)
