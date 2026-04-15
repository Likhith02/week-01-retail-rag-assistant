# Retail RAG Assistant

FastAPI service that answers policy questions using local document retrieval (RAG-style) and returns source evidence.

## Features

- Document chunking and TF-IDF retrieval
- Cosine-similarity ranking with top-k sources
- `POST /ask` with confidence score and source chunks
- Optional Telegram alert for low-confidence responses

## Tech Stack

- Python
- FastAPI
- Retrieval pipeline (TF-IDF + cosine similarity)
- Telegram Bot API (optional)

## Quick Start

```powershell
cd "C:\Users\likhi\OneDrive\Documents\New project\week-01-retail-rag-assistant"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload
```

Open:
- API docs: `http://127.0.0.1:8000/docs`

## API

- `GET /health` - service status and index counts
- `POST /ask` - ask a question
- `POST /reindex` - rebuild document index after doc updates

Example request:

```json
{
  "question": "Can I return a final sale item?"
}
```

## Configuration

`.env` values:

- `LOW_CONFIDENCE_THRESHOLD` (default `0.14`)
- `TOP_K` (default `3`)
- `TELEGRAM_BOT_TOKEN` (optional)
- `TELEGRAM_CHAT_ID` (optional)

## Repository Layout

```text
app/
  main.py
  rag_engine.py
  schemas.py
  settings.py
  telegram_notifier.py
data/docs/
scripts/demo_client.py
```
