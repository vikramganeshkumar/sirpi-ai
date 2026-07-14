# Migration Guide: Gemini → All-Gemma Setup

If you want to replace all Gemini calls with local Gemma (no external APIs), follow these steps.

## What changed

| Component | Original | All-Gemma |
|-----------|----------|-----------|
| Document summarization | Gemini API | Gemma (local) |
| Question answering | Gemini API | Gemma (local) |
| Relevance check | Gemma (local) | Gemma (local) |
| **Required keys** | `GOOGLE_API_KEY` | None |

## Step 1: Backup your current setup

If you want to keep your current Gemini-based version around, rename the files:
```bash
cd sirpi.ai
mv pdf_rag_guardrails.py pdf_rag_guardrails_gemini.py
mv app.py app_gemini.py
mv requirements.txt requirements_gemini.txt
mv .env.example .env.example.gemini
```

## Step 2: Copy in the Gemma-only versions

From the outputs folder you downloaded these files from, copy:
- `pdf_rag_guardrails_gemma_only.py` → `pdf_rag_guardrails.py`
- `app_gemma_only.py` → `app.py`
- `requirements_gemma_only.txt` → `requirements.txt`
- `.env_gemma_only.example` → `.env.example`

```bash
cp /path/to/outputs/pdf_rag_guardrails_gemma_only.py pdf_rag_guardrails.py
cp /path/to/outputs/app_gemma_only.py app.py
cp /path/to/outputs/requirements_gemma_only.txt requirements.txt
cp /path/to/outputs/.env_gemma_only.example .env.example
```

## Step 3: Update your `.env`

If you already have a `.env` file, you can simplify it by removing the Gemini key:

**Before:**
```dotenv
GOOGLE_API_KEY=AIzaSy...your_key...
GEMINI_MODEL=gemini-flash-latest

GEMMA=gemma3:12b
GEMMA_URL=https://gemma-ocr.slicearrow.com/api/generate

CHAT_DB_PATH=chat_history.db
```

**After:**
```dotenv
GEMMA=gemma3:12b
GEMMA_URL=https://gemma-ocr.slicearrow.com/api/generate

CHAT_DB_PATH=chat_history.db
```

## Step 4: Reinstall dependencies

The `google-genai` package is no longer needed:
```bash
pip install -r requirements.txt
```

## Step 5: Restart the server

If it's currently running, stop it (Ctrl+C), then:
```bash
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## Step 6: Test

- Upload a PDF — it should now summarize using local Gemma (not Gemini)
- Ask a question — it should be answered by local Gemma
- You should still see the four-stage guardrail trace (Rate · Length · Pattern · Topic)

## What stayed the same

- All guardrail logic (rate limiting, length checks, injection filtering, relevance classification)
- The four-stage guardrail pipeline and trace visualization
- Session persistence and chat history (SQLite)
- Frontend UI (same HTML/CSS/JS)
- API endpoints

## Rollback

If you saved the Gemini versions with `_gemini` suffix:
```bash
mv pdf_rag_guardrails.py pdf_rag_guardrails_gemma.py
mv app.py app_gemma.py
mv pdf_rag_guardrails_gemini.py pdf_rag_guardrails.py
mv app_gemini.py app.py
mv requirements.txt requirements_gemma.txt
mv requirements_gemini.txt requirements.txt
# Restore your .env with GOOGLE_API_KEY
pip install -r requirements.txt
```

## Questions

**Q: Will responses be slower?**  
A: Depends on your Gemma endpoint. If it's on the same machine, probably similar speed. If it's a remote Ollama server with network latency, could be slower.

**Q: Can I use a different Gemma model?**  
A: Yes. Change `GEMMA=gemma3:12b` to whatever model you have running on your Ollama instance (e.g. `GEMMA=llama2` or `GEMMA=mistral`). Make sure it supports text generation and can handle JSON-structured output for the relevance check.

**Q: What if I want to keep both versions?**  
A: You can — just keep the `_gemini` files around and swap the main files based on which version you want to run.
