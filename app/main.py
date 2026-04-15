from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.rag_engine import SimpleRAGEngine
from app.schemas import AskRequest, AskResponse, HealthResponse, SourceChunk
from app.settings import Settings, get_settings
from app.telegram_notifier import send_telegram_message


settings: Settings = get_settings()
engine = SimpleRAGEngine(settings.docs_dir)

app = FastAPI(
    title="Week 01 - Retail RAG Assistant",
    description=(
        "Beginner-friendly Retrieval-Augmented Generation (RAG) demo with FastAPI "
        "and optional Telegram alerts on low-confidence questions."
    ),
    version="1.0.0",
)


@app.on_event("startup")
def startup_event() -> None:
    engine.rebuild()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        docs_loaded=engine.doc_count,
        chunks_loaded=engine.chunk_count,
    )


@app.post("/reindex")
def reindex() -> JSONResponse:
    engine.rebuild()
    return JSONResponse(
        {
            "message": "Index rebuilt successfully.",
            "docs_loaded": engine.doc_count,
            "chunks_loaded": engine.chunk_count,
        }
    )


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    top_k = request.top_k or settings.top_k
    results = engine.search(request.question, top_k=top_k)
    answer = engine.answer_question(request.question, results)

    confidence = float(results[0].score) if results else 0.0
    low_confidence = confidence < settings.low_confidence_threshold

    if low_confidence:
        msg = (
            "[LOW CONFIDENCE QUESTION]\n"
            f"Question: {request.question}\n"
            f"Confidence: {confidence:.4f}\n"
            f"Top source: {(results[0].source if results else 'none')}"
        )
        await send_telegram_message(
            bot_token=settings.telegram_bot_token,
            chat_id=settings.telegram_chat_id,
            text=msg,
        )

    source_chunks = [
        SourceChunk(
            chunk_id=item.chunk_id,
            source=item.source,
            score=round(item.score, 4),
            text=item.text,
        )
        for item in results
    ]

    return AskResponse(
        answer=answer,
        confidence=round(confidence, 4),
        low_confidence=low_confidence,
        sources=source_chunks,
    )
