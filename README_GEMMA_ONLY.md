# Margin — PDF Chatbot (All Local, Gemma-Powered)

A responsive web chat UI for talking to your PDFs, powered entirely by **local Gemma/Ollama**. No external APIs, no cloud costs, no keys needed. Everything runs on your machine.

## What changed from the original (Gemini-based) version

**Original:** Gemini for summarization + answering, Gemma for guardrails only.  
**This version:** Gemma for **everything** — summarization, answering, relevance checks. Same guardrail pipeline, same four-stage trace, fully local.

**Benefits:**
- No `GOOGLE_API_KEY` required
- No external API calls (privacy ✓)
- No rate limits from cloud services
- Instant local responses
- Runs completely offline once Gemma is deployed

## Files to use from this folder

Replace these three files in your `sirpi.ai` folder with the versions from here:

- `pdf_rag_guardrails_gemma_only.py` → replace `pdf_rag_guardrails.py`
- `app_gemma_only.py` → replace `app.py`
- `requirements_gemma_only.txt` → replace `requirements.txt`
- `.env_gemma_only.example` → replace `.env.example`

Then delete or ignore the old files.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Your `.env` only needs two things:
```dotenv
GEMMA=gemma3:12b
GEMMA_URL=https://gemma-ocr.slicearrow.com/api/generate
CHAT_DB_PATH=chat_history.db
```

## Run

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000** in a browser.

The CLI script still works:
```bash
python pdf_rag_guardrails.py document.pdf "What is this document about?"
```

## How it works

1. Upload a PDF → Gemma summarizes it locally
2. Ask a question → runs through the 4-stage guardrail pipeline (all local):
   - Rate limit ✓
   - Length check ✓
   - Injection pattern filter ✓
   - Gemma relevance check ✓
3. If it passes → Gemma answers based on the document text + conversation history
4. Every message shows a trace strip with which stages it passed/blocked

## API endpoints

| Method | Path                              | Purpose                                  |
|--------|------------------------------------|-------------------------------------------|
| POST   | `/api/sessions/upload`            | Upload a PDF, creates a session           |
| GET    | `/api/sessions`                   | List recent sessions                      |
| GET    | `/api/sessions/{id}`              | Get a session's doc info + full history   |
| DELETE | `/api/sessions/{id}`              | Delete a session and its messages         |
| POST   | `/api/sessions/{id}/messages`     | Send a message, runs guardrails + answers |

## Notes

- Everything is local — the Gemma endpoint can be on the same machine, another machine on your network, or a cloud instance (still no external authentication needed).
- The frontend is served by FastAPI from `/static`, so there's a single process to run locally. You can split it to a separate static host later if desired.
- Sessions and full conversation history are stored in `chat_history.db` (SQLite), which survives server restarts.
