import json
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from app.config import Settings, get_settings
from app.dependencies import get_document_service, get_rag_service, get_registry, get_vector_store
from app.documents import DocumentService
from app.ingestion import IngestionError
from app.providers import embedding_model_name, llm_model_name
from app.rag import RAGService
from app.registry import DocumentRegistry
from app.schemas import ChatRequest, ChatResponse, DocumentInfo, UploadResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("docuchat")


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load the embedding model before accepting requests, so the first
    # upload/chat doesn't stall (and time out) while the model loads.
    logger.info("Loading embedding model...")
    await run_in_threadpool(get_vector_store)
    logger.info("Embedding model ready.")
    yield


app = FastAPI(
    title="DocuChat API",
    description="Chat with multiple PDFs using Retrieval-Augmented Generation.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(settings: Settings = Depends(get_settings)) -> dict:
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "llm_model": llm_model_name(settings),
        "embedding_provider": settings.embedding_provider,
        "embedding_model": embedding_model_name(settings),
    }


@app.post("/documents", response_model=list[UploadResult])
async def upload_documents(
    files: list[UploadFile] = File(...),
    service: DocumentService = Depends(get_document_service),
    settings: Settings = Depends(get_settings),
) -> list[UploadResult]:
    """Upload and index one or more PDFs. Each file reports its own status."""
    max_bytes = settings.max_upload_mb * 1024 * 1024
    results = []
    for upload in files:
        filename = upload.filename or "document.pdf"
        data = await upload.read()
        if len(data) > max_bytes:
            results.append(
                UploadResult(
                    filename=filename,
                    status="failed",
                    error=f"File exceeds the {settings.max_upload_mb} MB limit.",
                )
            )
            continue
        try:
            # Parsing + embedding is CPU-bound; keep it off the event loop.
            info, created = await run_in_threadpool(service.ingest, data, filename)
        except IngestionError as exc:
            results.append(UploadResult(filename=filename, status="failed", error=str(exc)))
            continue
        results.append(
            UploadResult(
                filename=filename,
                status="indexed" if created else "duplicate",
                document=info,
            )
        )
    return results


@app.get("/documents", response_model=list[DocumentInfo])
def list_documents(registry: DocumentRegistry = Depends(get_registry)) -> list[DocumentInfo]:
    return registry.list_all()


@app.delete("/documents/{doc_id}", status_code=204)
def delete_document(doc_id: str, service: DocumentService = Depends(get_document_service)) -> Response:
    if not service.delete(doc_id):
        raise HTTPException(status_code=404, detail="Document not found.")
    return Response(status_code=204)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, rag: RAGService = Depends(get_rag_service)) -> ChatResponse:
    try:
        return rag.answer(request.question, request.history, request.doc_ids)
    except Exception as exc:
        logger.exception("Chat failed")
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}") from exc


@app.post("/chat/stream")
def chat_stream(request: ChatRequest, rag: RAGService = Depends(get_rag_service)) -> StreamingResponse:
    """Stream the answer as newline-delimited JSON events (sources, token..., done)."""

    def events():
        try:
            for event in rag.stream(request.question, request.history, request.doc_ids):
                yield json.dumps(event) + "\n"
        except Exception as exc:
            logger.exception("Streaming chat failed")
            yield json.dumps({"type": "error", "data": str(exc)}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")
