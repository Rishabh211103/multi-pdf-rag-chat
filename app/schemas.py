from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    pages: int
    chunks: int
    size_bytes: int
    uploaded_at: datetime


class UploadResult(BaseModel):
    filename: str
    status: Literal["indexed", "duplicate", "failed"]
    document: DocumentInfo | None = None
    error: str | None = None


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[ChatTurn] = Field(default_factory=list)
    doc_ids: list[str] | None = Field(
        default=None, description="Restrict retrieval to these documents. Omit to search all."
    )


class Source(BaseModel):
    index: int
    doc_id: str
    filename: str
    page: int
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    standalone_question: str
