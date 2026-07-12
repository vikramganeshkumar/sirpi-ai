# Margin — PDF Chatbot UI

A responsive web chat UI on top of your existing guardrailed PDF RAG pipeline
(`pdf_rag_guardrails.py`). Upload a PDF, chat about it, and every question is
checked by the same four-stage guardrail pipeline as the CLI version — rate
limit → length → injection pattern → Gemma topic relevance — with the result
shown as a trace strip under each of your messages.

## What's new vs. the CLI version

- **`app.py`** — FastAPI backend. Wraps `RAGWithGuardrails` unchanged; adds
  HTTP endpoints for upload, chat, and session management.
- **`static/`** — the frontend (plain HTML/CSS/JS, no build step).
- **`pdf_rag_guardrails.py`** — one addition: `check_query_with_trace()`,
  which returns a per-stage trace alongside the existing `(allowed, reason)`
  result. `check_query()` still works exactly as before for the CLI.
- **`chat_memory.py`, `helpers.py`, `model_client.py`** — unchanged, copied
  in as-is.

## Memory model

- Each PDF upload creates a **session** in SQLite (`chat_history.db` by
  default — set `CHAT_DB_PATH` in `.env` to change the location).
- A session stores the extracted document text, its Gemini-generated
  summary, and the full message history.
- On each request, the backend keeps a `RAGWithGuardrails` instance per
  session in memory (so its rate limiter is scoped per-conversation). If the
  server restarts, the instance is rebuilt from SQLite via
  `load_from_memory()` — no PDF re-upload or re-summarization needed.
- Every answer is generated with the session's document text plus the last
  few turns of conversation history, so follow-up questions stay grounded
  in the same PDF.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in your real values
```

Required in `.env`:
```
GOOGLE_API_KEY=...       # Gemini API key
GEMINI_MODEL=gemini-flash-latest
GEMMA=gemma3:12b         # your Ollama model name
GEMMA_URL=https://...    # your Ollama /api/generate endpoint
```

## Run

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000** in a browser.

The CLI script still works unchanged:
```bash
python pdf_rag_guardrails.py document.pdf "What is this document about?"
```

## API endpoints

| Method | Path                              | Purpose                                  |
|--------|------------------------------------|-------------------------------------------|
| POST   | `/api/sessions/upload`            | Upload a PDF, creates a session           |
| GET    | `/api/sessions`                   | List recent sessions                      |
| GET    | `/api/sessions/{id}`              | Get a session's doc info + full history   |
| DELETE | `/api/sessions/{id}`              | Delete a session and its messages         |
| POST   | `/api/sessions/{id}/messages`     | Send a message, runs guardrails + answers |

## Notes

- CORS is wide open (`allow_origins=["*"]`) for local development. If you
  deploy the frontend on a different origin than the backend, narrow this
  in `app.py` before going live.
- The frontend is served by the same FastAPI process (`static/` mounted at
  `/static`, `index.html` served at `/`) so there's a single process to run
  locally. The frontend code itself has no dependency on FastAPI and could
  be split onto its own static host later if you want, as long as it can
  reach `/api/*`.
