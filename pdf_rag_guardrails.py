#!/usr/bin/env python3
"""
PDF RAG system with guardrails -- a hybrid setup:
  - Gemini (Google API)   -> document summarization + answering questions
  - Gemma (self-hosted, via Ollama) -> guardrail relevance classification

Guardrail pipeline for every incoming query, in order (cheapest checks first
so obvious junk never reaches a model at all):
  1. Rate limiting        (in-memory sliding window)
  2. Length limit          (reject oversized queries before they're sent anywhere)
  3. Local injection filter(regex pre-filter for common jailbreak phrasing)
  4. Gemma relevance check (structured JSON classification, fail-closed by default)

Note: this system answers from a truncated slice of the raw document text --
there's no chunking/embedding/vector search here, so "top-k retrieval" claims
in older docs for this repo don't apply to this implementation.
"""
import json
import logging
import os
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

from helpers import extract_text_from_pdf, get_api_key, print_section, setup_logging, truncate_text
from model_client import call_ollama

try:
    import google.genai as genai
except ImportError:
    print("Error: google-genai not installed. Run: pip install -r requirements.txt")
    sys.exit(1)

load_dotenv()

GEMMA_MODEL = os.environ.get("GEMMA")
GEMMA_URL = os.environ.get("GEMMA_URL")
# Gemini has been retiring model IDs quickly (gemini-2.5-flash started 404ing
# for new API keys in July 2026 ahead of its official Oct 2026 EOL date).
# Keeping this in .env means a retirement is a config change, not a code edit.
# 'gemini-flash-latest' is a Google-maintained alias that auto-points to a
# current, non-deprecated flash model.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")


# Cheap local pre-filter for common prompt-injection / jailbreak phrasing.
# This is NOT a substitute for the Gemma relevance check below -- it's a
# fast, free first pass that blocks the cheapest attacks without a model call.
INJECTION_PATTERNS = [
    re.compile(r"ignore (all|the|any) (previous|prior|above) instructions?", re.I),
    re.compile(r"disregard (all|the|any) (previous|prior|above)", re.I),
    re.compile(r"you are now", re.I),
    re.compile(r"reveal (your|the) (system prompt|instructions)", re.I),
    re.compile(r"act as (if|though)", re.I),
    re.compile(r"jailbreak", re.I),
    re.compile(r"pretend (you|to) (are|be)", re.I),
    re.compile(r"new instructions?:", re.I),
]

# Ordered list of guardrail stage names, used by check_query_with_trace so
# the web UI can render a consistent pipeline strip regardless of where a
# query was stopped.
GUARDRAIL_STAGES = ["rate_limit", "length", "injection_pattern", "topic_relevance"]


@dataclass
class GuardrailConfig:
    max_query_length: int = 500
    max_retries: int = 2
    retry_backoff_seconds: float = 5.0
    request_timeout_seconds: int = 180
    fail_open: bool = False  # if the relevance check errors, block by default (safe)
    rate_limit_window_seconds: float = 60.0
    rate_limit_max_requests: int = 20
    log_file: str = "guardrail_events.log"


class RateLimiter:
    """In-memory sliding-window rate limiter, scoped to one process/instance."""

    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: deque = deque()

    def allow(self) -> bool:
        now = time.time()
        while self._timestamps and now - self._timestamps[0] > self.window_seconds:
            self._timestamps.popleft()
        if len(self._timestamps) >= self.max_requests:
            return False
        self._timestamps.append(now)
        return True


class RAGWithGuardrails:
    def __init__(
        self,
        api_key: str,
        config: Optional[GuardrailConfig] = None,
        gemini_model: str = GEMINI_MODEL,
        gemma_model: Optional[str] = None,
        gemma_url: Optional[str] = None,
    ):
        # Gemini -- summary + answering
        self.client = genai.Client(api_key=api_key)
        self.gemini_model = gemini_model

        # Gemma -- guardrail relevance check
        self.gemma_model = gemma_model or GEMMA_MODEL
        self.gemma_url = gemma_url or GEMMA_URL
        if not self.gemma_model or not self.gemma_url:
            raise EnvironmentError(
                "GEMMA and GEMMA_URL must be set in .env (needed for the "
                "guardrail relevance check), or pass gemma_model/gemma_url directly"
            )

        self.config = config or GuardrailConfig()
        self.text = ""
        self.summary = ""
        self._rate_limiter = RateLimiter(
            self.config.rate_limit_max_requests, self.config.rate_limit_window_seconds
        )
        setup_logging(self.config.log_file)

    # ---------- document loading ----------

    def load_document(self, pdf_path: str) -> None:
        self.text = extract_text_from_pdf(pdf_path)
        self.summary = self._create_document_summary()
        logging.info("Document summary: %s", self.summary[:200])

    def load_from_memory(self, document_text: str, document_summary: str) -> None:
        """Restore document context without re-extracting the PDF or
        re-calling the summarization model -- used when resuming a persisted
        session (see chat_memory.py)."""
        self.text = document_text
        self.summary = document_summary

    def _create_document_summary(self) -> str:
        prompt = (
            "Briefly summarize what this document is about in 2-3 sentences:\n\n"
            f"{truncate_text(self.text, 2000)}\n\nSummary:"
        )
        return self._generate_with_gemini(prompt).strip()

    # ---------- guardrails (Gemma) ----------

    def _local_injection_check(self, query: str) -> Optional[str]:
        """Fast local pre-filter. Returns a block reason, or None if clean."""
        for pattern in INJECTION_PATTERNS:
            if pattern.search(query):
                return "Query matched a blocked pattern (possible prompt injection)"
        return None

    def _llm_topic_check(self, query: str) -> Tuple[bool, str]:
        """Ask Gemma to classify relevance and return it as structured JSON
        instead of parsing free text (a plain 'YES' in result.upper() approach
        misclassifies anything that merely contains the substring 'yes')."""
        prompt = (
            f"Document summary: {self.summary}\n\n"
            f'User question: "{query}"\n\n'
            "Decide if this question is asking about the document's content. "
            "Respond with ONLY valid JSON, no other text, in this exact format:\n"
            '{"on_topic": true or false, "reason": "short explanation"}'
        )
        raw = call_ollama(
            prompt=prompt,
            model_name=self.gemma_model,
            model_url=self.gemma_url,
            retries=self.config.max_retries + 1,
            timeout=self.config.request_timeout_seconds,
            backoff_base_seconds=self.config.retry_backoff_seconds,
        )
        if raw is None:
            raise RuntimeError(f"Gemma relevance check failed (model: {self.gemma_model})")

        cleaned = raw.strip().strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        parsed = json.loads(cleaned)  # raises on malformed output -> caller handles it
        return bool(parsed.get("on_topic", False)), str(parsed.get("reason", ""))

    def check_query_with_trace(self, query: str) -> Tuple[bool, str, List[Dict]]:
        """Same pipeline as check_query, but also returns a per-stage trace
        (stage name, passed/skipped, detail) for surfacing in a UI. This is
        the source of truth -- check_query is a thin wrapper around it so
        existing callers (CLI, process()) keep their original signature.
        """
        trace: List[Dict] = []

        def remaining_skipped(from_index: int) -> None:
            for stage in GUARDRAIL_STAGES[from_index:]:
                trace.append({"stage": stage, "status": "skipped", "detail": ""})

        # 1. rate limit
        if not self._rate_limiter.allow():
            trace.append({"stage": "rate_limit", "status": "blocked", "detail": "Rate limit exceeded"})
            remaining_skipped(1)
            logging.warning("BLOCKED (rate limit): %r", query)
            return False, "Rate limit exceeded, try again shortly", trace
        trace.append({"stage": "rate_limit", "status": "passed", "detail": ""})

        # 2. length
        if len(query) > self.config.max_query_length:
            detail = f"{len(query)} chars, max {self.config.max_query_length}"
            trace.append({"stage": "length", "status": "blocked", "detail": detail})
            remaining_skipped(2)
            logging.warning("BLOCKED (length): %r", query[:100])
            return False, f"Query too long ({detail})", trace
        trace.append({"stage": "length", "status": "passed", "detail": ""})

        # 3. local injection pattern filter
        injection_reason = self._local_injection_check(query)
        if injection_reason:
            trace.append({"stage": "injection_pattern", "status": "blocked", "detail": injection_reason})
            remaining_skipped(3)
            logging.warning("BLOCKED (injection pattern): %r", query)
            return False, injection_reason, trace
        trace.append({"stage": "injection_pattern", "status": "passed", "detail": ""})

        # 4. Gemma relevance check
        try:
            on_topic, reason = self._llm_topic_check(query)
        except Exception as e:
            logging.error("Guardrail relevance check failed: %s", e)
            if self.config.fail_open:
                trace.append({"stage": "topic_relevance", "status": "passed", "detail": "check failed, fail_open"})
                return True, "Relevance check failed; allowed per fail_open config", trace
            trace.append({"stage": "topic_relevance", "status": "blocked", "detail": "relevance check errored"})
            return False, "Relevance check failed; blocked by default (fail-closed)", trace

        if not on_topic:
            trace.append({"stage": "topic_relevance", "status": "blocked", "detail": reason})
            logging.info("BLOCKED (off-topic): %r | reason=%s", query, reason)
            return False, reason, trace

        trace.append({"stage": "topic_relevance", "status": "passed", "detail": reason})
        return True, reason, trace

    def check_query(self, query: str) -> Tuple[bool, str]:
        """Run the full guardrail pipeline on a query. Returns (is_allowed, reason).
        Kept for CLI / backward compatibility -- wraps check_query_with_trace."""
        allowed, reason, _trace = self.check_query_with_trace(query)
        return allowed, reason

    # ---------- generation (Gemini) ----------

    def _generate_with_gemini(self, prompt: str) -> str:
        last_error = None
        for attempt in range(1, self.config.max_retries + 2):
            try:
                response = self.client.models.generate_content(model=self.gemini_model, contents=prompt)
                return response.text
            except Exception as e:
                last_error = e
                logging.warning("Gemini generation attempt %d failed: %s", attempt, e)
                if attempt <= self.config.max_retries:
                    time.sleep(self.config.retry_backoff_seconds * attempt)
        raise RuntimeError(f"Gemini generation failed after {self.config.max_retries} retries: {last_error}")

    def answer_question(self, query: str, history: Optional[list] = None) -> str:
        """Answer a question about the loaded document, via Gemini.

        `history` is an optional list of {"role": "user"|"assistant", "content": str}
        dicts, most-recent-last. Only the last few turns are included in the
        prompt to keep it bounded -- full history still lives wherever the
        caller persists it (e.g. chat_memory.py).
        """
        history_text = ""
        if history:
            turns = []
            for turn in history[-6:]:
                speaker = "User" if turn["role"] == "user" else "Assistant"
                turns.append(f"{speaker}: {turn['content']}")
            history_text = "Conversation so far:\n" + "\n".join(turns) + "\n\n"

        prompt = (
            f"Based on this document:\n\n{truncate_text(self.text, 5000)}\n\n"
            f"{history_text}"
            f"Answer this question: {query}\n\n"
            'If the answer is not in the document, say "I don\'t find this information in the document."\n\n'
            "Answer:"
        )
        return self._generate_with_gemini(prompt)

    def process(self, pdf_path: str, query: str) -> str:
        """Load a PDF and answer one question, running it through guardrails first."""
        self.load_document(pdf_path)
        allowed, reason = self.check_query(query)
        if not allowed:
            return (
                f"Blocked: {reason}\n\n"
                f"Your question: '{query}'\n"
                f"This document is about: {self.summary}"
            )
        return self.answer_question(query)

    def interactive_mode(self, pdf_path: str) -> None:
        self.load_document(pdf_path)
        print_section("Ready for Q&A -- type 'exit' to quit")
        print(f"Document summary: {self.summary}\n")

        while True:
            try:
                query = input("Your question: ").strip()
                if query.lower() == "exit":
                    print("Goodbye!")
                    break
                if not query:
                    continue

                allowed, reason = self.check_query(query)
                if not allowed:
                    print(f"Blocked: {reason}\n")
                    continue

                answer = self.answer_question(query)
                print_section("Answer")
                print(answer)
                print()

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                logging.error("Unhandled error in interactive loop: %s", e)
                print(f"Error: {e}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single question: python pdf_rag_guardrails.py <pdf> <question>")
        print("  Interactive:     python pdf_rag_guardrails.py <pdf> --interactive")
        sys.exit(1)

    pdf_path = sys.argv[1]
    api_key = get_api_key()

    try:
        rag = RAGWithGuardrails(api_key=api_key)

        if len(sys.argv) > 2 and sys.argv[2] == "--interactive":
            rag.interactive_mode(pdf_path)
        elif len(sys.argv) > 2:
            query = " ".join(sys.argv[2:])
            answer = rag.process(pdf_path, query)
            print_section("Result")
            print(answer)
        else:
            print("Error: provide a question or use --interactive")
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
