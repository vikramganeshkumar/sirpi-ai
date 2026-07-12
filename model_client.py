#!/usr/bin/env python3
"""
Shared client for calling self-hosted Ollama models (text or vision) over
HTTP. Used by both the OCR pipeline (ocr_client.py) and the main RAG
pipeline (pdf_rag_guardrails.py) so retry/backoff/error-handling logic for
talking to an Ollama endpoint lives in exactly one place.
"""
import logging
import time
from typing import Optional

import requests


def call_ollama(
    prompt: str,
    model_name: str,
    model_url: str,
    image_b64: Optional[str] = None,
    retries: int = 3,
    timeout: int = 180,
    num_predict: int = 2048,
    temperature: float = 0,
    backoff_base_seconds: float = 5.0,
) -> Optional[str]:
    """Call an Ollama-style /api/generate endpoint.

    Pass image_b64 for vision models (OCR); omit it for pure text
    generation. Returns the model's text response, or None if every retry
    attempt fails.
    """
    if not model_name or not model_url:
        raise EnvironmentError(
            "model_name and model_url must both be set -- check your .env "
            "has the matching pair defined (e.g. GEMMA and GEMMA_URL)"
        )

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    if image_b64:
        payload["images"] = [image_b64]

    for attempt in range(1, retries + 1):
        logging.info("[%s] Attempt %d/%d (timeout: %ds)", model_name, attempt, retries, timeout)
        try:
            response = requests.post(model_url, json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            text = result.get("response", "")
            logging.info("[%s] Success on attempt %d", model_name, attempt)
            return text

        except requests.exceptions.Timeout as e:
            logging.warning("[%s] Timeout on attempt %d: %s", model_name, attempt, e)
        except requests.exceptions.ConnectionError as e:
            logging.warning("[%s] Connection error on attempt %d: %s", model_name, attempt, e)
        except requests.exceptions.HTTPError as e:
            logging.warning("[%s] HTTP error on attempt %d: %s", model_name, attempt, e)
            if e.response is not None and e.response.status_code < 500:
                # Client error (4xx) won't fix itself on retry -- fail fast
                return None
        except Exception as e:
            logging.error("[%s] Unexpected error on attempt %d: %s", model_name, attempt, e)

        if attempt < retries:
            wait_time = backoff_base_seconds * attempt
            logging.info("[%s] Retrying in %.0fs...", model_name, wait_time)
            time.sleep(wait_time)

    logging.error("[%s] All %d attempts failed", model_name, retries)
    return None
