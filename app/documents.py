"""Document lifecycle: ingest into Chroma + registry, list, delete."""

import logging
from datetime import datetime, timezone

from langchain_chroma import Chroma
from langchain_text_splitters import TextSplitter

from app.config import Settings
from app.ingestion import compute_doc_id, parse_pdf
from app.registry import DocumentRegistry
from app.schemas import DocumentInfo

logger = logging.getLogger(__name__)

# Chroma rejects very large single inserts; stay well under its limit.
_ADD_BATCH_SIZE = 500


def build_vector_store(settings: Settings, embeddings) -> Chroma:
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


class DocumentService:
    def __init__(self, store: Chroma, registry: DocumentRegistry, splitter: TextSplitter):
        self.store = store
        self.registry = registry
        self.splitter = splitter

    def ingest(self, data: bytes, filename: str) -> tuple[DocumentInfo, bool]:
        """Index a PDF. Returns (info, created); created is False for duplicates."""
        doc_id = compute_doc_id(data)
        if existing := self.registry.get(doc_id):
            return existing, False

        parsed = parse_pdf(data, filename, self.splitter)
        # Deterministic IDs make a retried ingest overwrite instead of duplicating chunks.
        ids = [f"{doc_id}-{i}" for i in range(len(parsed.chunks))]
        for start in range(0, len(ids), _ADD_BATCH_SIZE):
            end = start + _ADD_BATCH_SIZE
            self.store.add_documents(parsed.chunks[start:end], ids=ids[start:end])

        info = DocumentInfo(
            doc_id=doc_id,
            filename=filename,
            pages=parsed.pages,
            chunks=len(parsed.chunks),
            size_bytes=len(data),
            uploaded_at=datetime.now(timezone.utc),
        )
        self.registry.add(info)
        logger.info("Indexed %s (%s): %d pages, %d chunks", filename, doc_id, info.pages, info.chunks)
        return info, True

    def delete(self, doc_id: str) -> bool:
        if not self.registry.get(doc_id):
            return False
        ids = self.store.get(where={"doc_id": doc_id}, include=[])["ids"]
        if ids:
            self.store.delete(ids=ids)
        self.registry.delete(doc_id)
        logger.info("Deleted %s (%d chunks)", doc_id, len(ids))
        return True
