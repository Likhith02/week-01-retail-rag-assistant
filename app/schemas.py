from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, description="User question to ask the assistant.")
    top_k: int | None = Field(default=None, ge=1, le=10)


class SourceChunk(BaseModel):
    chunk_id: str
    source: str
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    confidence: float
    low_confidence: bool
    sources: List[SourceChunk]


class HealthResponse(BaseModel):
    status: str
    docs_loaded: int
    chunks_loaded: int
