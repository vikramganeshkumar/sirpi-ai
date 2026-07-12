#!/usr/bin/env python3
"""
Persistent conversation memory for the PDF chatbot.

Stores sessions (one per loaded PDF) and their message history in SQLite,
so a conversation survives an app restart. Each session also stores the
document's extracted text and summary, so resuming a session doesn't
require re-uploading the PDF or re-calling the summarization API.
"""
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Dict, List, Optional

DB_PATH = os.environ.get("CHAT_DB_PATH", "chat_history.db")


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create tables if they don't exist yet. Safe to call on every startup."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                pdf_name TEXT NOT NULL,
                document_text TEXT NOT NULL,
                document_summary TEXT,
                created_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            )
            """
        )


def create_session(pdf_name: str, document_text: str, document_summary: str = "") -> str:
    session_id = str(uuid.uuid4())
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO sessions (id, pdf_name, document_text, document_summary, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, pdf_name, document_text, document_summary, time.time()),
        )
    return session_id


def save_message(session_id: str, role: str, content: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, time.time()),
        )


def get_history(session_id: str) -> List[Dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def list_sessions(limit: int = 20) -> List[Dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, pdf_name, created_at FROM sessions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [{"id": r["id"], "pdf_name": r["pdf_name"], "created_at": r["created_at"]} for r in rows]


def get_session(session_id: str) -> Optional[Dict]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, pdf_name, document_text, document_summary, created_at "
            "FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
    return dict(row) if row else None


def delete_session(session_id: str) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
