#!/usr/bin/env python3
"""
PDF RAG system with guardrails.

Guardrail pipeline for every incoming query, in order (cheapest checks first
so obvious junk never reaches the API):
  1. Rate limiting        (in-memory sliding window)
  2. Length limit          (reject oversized queries before they're sent anywhere)
  3. Local injection filter(regex pre-filter for common jailbreak phrasing)
  4. LLM relevance check   (structured JSON classification, fail-closed by default)

Note: this system answers from a truncated slice of the raw document text --
there's no chunking/embedding/vector search here, so "top-k retrieval" claims
in older docs for this repo don't apply to this implementation.
"""
import json
import logging
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

from helpers import extract_text_from_pdf, get_api_key, print_section, setup_logging, truncate_text

try:
    import google.genai as genai
except ImportError:
    print("Error: google-genai not installed. Run: pip install -r requirements.txt")
    sys.exit(1)


# Cheap local pre-filter for common prompt-injection / jailbreak phrasing.
# This is NOT a substitute for the LLM relevance check below -- it's a fast,
# free first pass that blocks the cheapest attacks without an API call.
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


@dataclass
class GuardrailConfig:
    max_query_length: int = 500
    max_retries: int = 2
    retry_backoff_seconds: float = 1.5
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
        model: str = "gemini-2.0-flash",
    ):
        self.client = genai.Client(api_key=api_key)
        self.model = model
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

    def _create_document_summary(self) -> str:
        prompt = (
            "Briefly summarize what this document is about in 2-3 sentences:\n\n"
            f"{truncate_text(self.text, 2000)}\n\nSummary:"
        )
        return self._generate_with_retry(prompt).strip()

    # ---------- guardrails ----------

    def _local_injection_check(self, query: str) -> Optional[str]:
        """Fast local pre-filter. Returns a block reason, or None if clean."""
        for pattern in INJECTION_PATTERNS:
            if pattern.search(query):
                return "Query matched a blocked pattern (possible prompt injection)"
        return None

    def _llm_topic_check(self, query: str) -> Tuple[bool, str]:
        """Ask the model to classify relevance and return it as structured JSON
        instead of parsing free text (the old 'YES' in result.upper() approach
        misclassifies anything that merely contains the substring 'yes')."""
        prompt = (
            f"Document summary: {self.summary}\n\n"
            f'User question: "{query}"\n\n'
            "Decide if this question is asking about the document's content. "
            "Respond with ONLY valid JSON, no other text, in this exact format:\n"
            '{"on_topic": true or false, "reason": "short explanation"}'
        )
        raw = self._generate_with_retry(prompt)
        cleaned = raw.strip().strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        parsed = json.loads(cleaned)  # raises on malformed output -> caller handles it
        return bool(parsed.get("on_topic", False)), str(parsed.get("reason", ""))

    def check_query(self, query: str) -> Tuple[bool, str]:
        """Run the full guardrail pipeline on a query. Returns (is_allowed, reason)."""
        if not self._rate_limiter.allow():
            logging.warning("BLOCKED (rate limit): %r", query)
            return False, "Rate limit exceeded, try again shortly"

        if len(query) > self.config.max_query_length:
            logging.warning("BLOCKED (length): %r", query[:100])
            return False, f"Query too long ({len(query)} chars, max {self.config.max_query_length})"

        injection_reason = self._local_injection_check(query)
        if injection_reason:
            logging.warning("BLOCKED (injection pattern): %r", query)
            return False, injection_reason

        try:
            on_topic, reason = self._llm_topic_check(query)
        except Exception as e:
            logging.error("Guardrail relevance check failed: %s", e)
            if self.config.fail_open:
                return True, "Relevance check failed; allowed per fail_open config"
            return False, "Relevance check failed; blocked by default (fail-closed)"

        if not on_topic:
            logging.info("BLOCKED (off-topic): %r | reason=%s", query, reason)
        return on_topic, reason

    # ---------- generation ----------

    def _generate_with_retry(self, prompt: str) -> str:
        last_error = None
        for attempt in range(1, self.config.max_retries + 2):
            try:
                response = self.client.models.generate_content(model=self.model, contents=prompt)
                return response.text
            except Exception as e:
                last_error = e
                logging.warning("Generation attempt %d failed: %s", attempt, e)
                if attempt <= self.config.max_retries:
                    time.sleep(self.config.retry_backoff_seconds * attempt)
        raise RuntimeError(f"Generation failed after {self.config.max_retries} retries: {last_error}")

    def answer_question(self, query: str) -> str:
        prompt = (
            f"Based on this document:\n\n{truncate_text(self.text, 5000)}\n\n"
            f"Answer this question: {query}\n\n"
            'If the answer is not in the document, say "I don\'t find this information in the document."\n\n'
            "Answer:"
        )
        return self._generate_with_retry(prompt)

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
    api_key = get_api_key()

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single question: python pdf_rag_guardrails.py <pdf> <question>")
        print("  Interactive:     python pdf_rag_guardrails.py <pdf> --interactive")
        sys.exit(1)

    pdf_path = sys.argv[1]
    rag = RAGWithGuardrails(api_key=api_key)

    try:
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