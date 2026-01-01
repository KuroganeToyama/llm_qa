"""Data models for the QA system."""
from typing import List, Optional
from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    """Metadata for a document chunk."""
    source: str
    page: Optional[int] = None
    chunk_id: str
    images: Optional[List[str]] = None


class RetrievedChunk(BaseModel):
    """A chunk retrieved from the vector store."""
    content: str
    metadata: DocumentMetadata
    score: float


class QAResponse(BaseModel):
    """Response from the QA system."""
    answer: str
    sources: Optional[List[str]] = None


class QueryRequest(BaseModel):
    """A query request to the QA system."""
    question: str
    return_sources: bool = False
