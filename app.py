#!/usr/bin/env python3
"""
FastAPI backend for the PDF chatbot UI.

Wraps the existing RAGWithGuardrails pipeline (pdf_rag_guardrails.py)
unchanged -- this file only adds HTTP endpoints, temp-file handling for
uploads, and per-session instance caching. All guardrail logic, retries,
and Gemini/Gemma calls still live where they already did.

Session/message persistence is handled entirely by chat_memory.py (SQLite),
so a conversation survives a server restart -- on restart, a session's
document text + summary are loaded back into a fresh RAGWithGuardrails
instance via load_from_memory() instead of re-uploading the PDF.
"""
import logging
import os
import tempfile
from typing import Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import chat_memory
from helpers import get_api_key
from pdf_rag_guardrails import RAGWithGuardrails

app = FastAPI(title="PDF RAG Chatbot")

# Dev-friendly CORS. If you split the frontend onto its own origin/port,
# narrow this to that origin before deploying anywhere public.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One RAGWithGuardrails instance per active session, cached in-process.
# Each instance keeps its own rate limiter, so rate limits are per-session
# rather than shared across every user of the server.
_rag_instances: Dict[str, RAGWithGuardrails] = {}
_api_key: str = ""


def _get_or_load_instance(session_id: str) -> RAGWithGuardrails:
    """Return the cached RAG instance for a session, or rebuild it from
    SQLite (document text + summary) if the server restarted since."""
    if session_id in _rag_instances:
        return _rag_instances[session_id]

    session = chat_memory.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    rag = RAGWithGuardrails(api_key=_api_key)
    rag.load_from_memory(session["document_text"], session["document_summary"] or "")
    _rag_instances[session_id] = rag
    return rag


class MessageIn(BaseModel):
    message: str


@app.on_event("startup")
def startup() -> None:
    global _api_key
    chat_memory.init_db()
    _api_key = get_api_key()
    logging.info("PDF chatbot backend started")


@app.post("/api/sessions/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        rag = RAGWithGuardrails(api_key=_api_key)
        rag.load_document(tmp_path)  # extracts text + generates summary via Gemini
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to process PDF: {e}")
    finally:
        os.remove(tmp_path)

    session_id = chat_memory.create_session(
        pdf_name=file.filename, document_text=rag.text, document_summary=rag.summary
    )
    _rag_instances[session_id] = rag

    return {"session_id": session_id, "pdf_name": file.filename, "summary": rag.summary}


@app.get("/api/sessions")
def list_sessions():
    return chat_memory.list_sessions()


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    session = chat_memory.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    history = chat_memory.get_history(session_id)
    return {
        "id": session["id"],
        "pdf_name": session["pdf_name"],
        "summary": session["document_summary"],
        "created_at": session["created_at"],
        "history": history,
    }


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    chat_memory.delete_session(session_id)
    _rag_instances.pop(session_id, None)
    return {"deleted": session_id}


@app.post("/api/sessions/{session_id}/messages")
def send_message(session_id: str, body: MessageIn):
    rag = _get_or_load_instance(session_id)
    query = body.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    allowed, reason, trace = rag.check_query_with_trace(query)
    chat_memory.save_message(session_id, "user", query)

    if not allowed:
        answer = f"Blocked: {reason}"
        chat_memory.save_message(session_id, "assistant", answer)
        return {"answer": answer, "blocked": True, "reason": reason, "trace": trace}

    history = chat_memory.get_history(session_id)[:-1]  # exclude the message just saved
    try:
        answer = rag.answer_question(query, history=history)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Generation failed: {e}")

    chat_memory.save_message(session_id, "assistant", answer)
    return {"answer": answer, "blocked": False, "reason": reason, "trace": trace}


# Serve the frontend last, so /api/* routes above take precedence.
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")
