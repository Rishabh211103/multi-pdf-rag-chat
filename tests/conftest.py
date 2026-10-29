"""Shared fixtures. Uses fake embeddings and a fake LLM, so tests need no API keys or model downloads."""

import pymupdf
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.config import Settings
from app.documents import DocumentService, build_vector_store
from app.ingestion import make_splitter
from app.rag import RAGService
from app.registry import DocumentRegistry


@pytest.fixture
def make_pdf():
    def _make(pages: list[str]) -> bytes:
        pdf = pymupdf.open()
        for text in pages:
            page = pdf.new_page()
            if text:
                page.insert_text((72, 72), text, fontsize=10)
        data = pdf.tobytes()
        pdf.close()
        return data

    return _make


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path,
        chunk_size=200,
        chunk_overlap=20,
        retrieval_k=3,
        retrieval_fetch_k=10,
    )


@pytest.fixture
def store(settings):
    return build_vector_store(settings, DeterministicFakeEmbedding(size=64))


@pytest.fixture
def document_service(settings, store) -> DocumentService:
    return DocumentService(
        store=store,
        registry=DocumentRegistry(settings.registry_path),
        splitter=make_splitter(settings.chunk_size, settings.chunk_overlap),
    )


@pytest.fixture
def make_rag(settings, store):
    def _make(responses: list[str]) -> RAGService:
        return RAGService(FakeListChatModel(responses=responses), store, settings)

    return _make
