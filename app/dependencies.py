"""Lazily-built singletons, injected into routes with FastAPI `Depends`.

Tests swap these out via `app.dependency_overrides`.
"""

from functools import lru_cache

from langchain_chroma import Chroma

from app.config import get_settings
from app.documents import DocumentService, build_vector_store
from app.ingestion import make_splitter
from app.providers import build_chat_model, build_embeddings
from app.rag import RAGService
from app.registry import DocumentRegistry


@lru_cache
def get_vector_store() -> Chroma:
    settings = get_settings()
    return build_vector_store(settings, build_embeddings(settings))


@lru_cache
def get_registry() -> DocumentRegistry:
    return DocumentRegistry(get_settings().registry_path)


@lru_cache
def get_document_service() -> DocumentService:
    settings = get_settings()
    return DocumentService(
        store=get_vector_store(),
        registry=get_registry(),
        splitter=make_splitter(settings.chunk_size, settings.chunk_overlap),
    )


@lru_cache
def get_rag_service() -> RAGService:
    settings = get_settings()
    return RAGService(build_chat_model(settings), get_vector_store(), settings)
