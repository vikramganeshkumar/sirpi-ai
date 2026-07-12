#!/usr/bin/env python3
"""
Standalone diagnostic for the Gemma/Ollama endpoint used by the guardrail
relevance check. Run this directly (no FastAPI, no PDF needed) to see
exactly what GEMMA_URL returns -- this is the same call check_query_with_trace
makes under the hood, minus everything else in the pipeline.

Usage:
    python test_gemma.py
"""
import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

GEMMA_MODEL = os.environ.get("GEMMA")
GEMMA_URL = os.environ.get("GEMMA_URL")

print(f"GEMMA model: {GEMMA_MODEL!r}")
print(f"GEMMA_URL:   {GEMMA_URL!r}")
print("-" * 60)

if not GEMMA_MODEL or not GEMMA_URL:
    print("ERROR: GEMMA and/or GEMMA_URL not set in .env")
    raise SystemExit(1)

# --- Test 1: plain generation, no JSON constraint -----------------------
print("\n[Test 1] Basic generation ('Say hello')")
payload = {
    "model": GEMMA_MODEL,
    "prompt": "Say hello in one short sentence.",
    "stream": False,
    "options": {"temperature": 0, "num_predict": 100},
}
try:
    resp = requests.post(GEMMA_URL, json=payload, timeout=60)
    print(f"HTTP status: {resp.status_code}")
    print(f"Raw body:    {resp.text[:1000]}")
    resp.raise_for_status()
    data = resp.json()
    print(f"Parsed 'response' field: {data.get('response', '')!r}")
except requests.exceptions.Timeout:
    print("FAILED: request timed out (60s). Endpoint is unreachable or too slow.")
except requests.exceptions.ConnectionError as e:
    print(f"FAILED: connection error -- endpoint unreachable.\n{e}")
except requests.exceptions.HTTPError as e:
    print(f"FAILED: HTTP error.\n{e}")
except Exception as e:
    print(f"FAILED: unexpected error.\n{type(e).__name__}: {e}")

# --- Test 2: the exact relevance-check prompt shape used by the app -----
print("\n" + "-" * 60)
print("[Test 2] Relevance-check style prompt (must return strict JSON)")
relevance_prompt = (
    'Document summary: This is a resume.\n\n'
    'User question: "What is this document about?"\n\n'
    "Decide if this question is asking about the document's content. "
    "Respond with ONLY valid JSON, no other text, in this exact format:\n"
    '{"on_topic": true or false, "reason": "short explanation"}'
)
payload2 = {
    "model": GEMMA_MODEL,
    "prompt": relevance_prompt,
    "stream": False,
    "options": {"temperature": 0, "num_predict": 2048},
}
try:
    resp = requests.post(GEMMA_URL, json=payload2, timeout=60)
    print(f"HTTP status: {resp.status_code}")
    resp.raise_for_status()
    data = resp.json()
    raw = data.get("response", "")
    print(f"Raw model output:\n{raw!r}")

    cleaned = raw.strip().strip("`")
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()

    try:
        parsed = json.loads(cleaned)
        print(f"\nJSON parsed OK: {parsed}")
    except json.JSONDecodeError as e:
        print(f"\nFAILED TO PARSE AS JSON: {e}")
        print("This is almost certainly your 'Relevance check failed' bug --")
        print("the model isn't returning strict JSON (e.g. wrapped in prose,")
        print("markdown fences with a language tag, or truncated).")
except requests.exceptions.Timeout:
    print("FAILED: request timed out (60s).")
except requests.exceptions.ConnectionError as e:
    print(f"FAILED: connection error.\n{e}")
except requests.exceptions.HTTPError as e:
    print(f"FAILED: HTTP error.\n{e}")
except Exception as e:
    print(f"FAILED: unexpected error.\n{type(e).__name__}: {e}")

print("\n" + "-" * 60)
print("Done.")