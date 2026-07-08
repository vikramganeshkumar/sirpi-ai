#!/usr/bin/env python3
"""
OCR client for image-to-text extraction via self-hosted Ollama vision models
(Qwen, Gemma, etc.) over HTTP.

Model name/URL pairs are loaded from .env -- never hardcode endpoints or
credentials here. See README for the required .env keys.
"""
import logging
import os
import time
from typing import Optional

import requests
from dotenv import load_dotenv

from helpers import setup_logging

load_dotenv()

QWEN = os.environ.get("QWEN")
QWEN_URL = os.environ.get("QWEN_URL")
GEMMA = os.environ.get("GEMMA")
GEMMA_URL = os.environ.get("GEMMA_URL")

DEFAULT_PROMPT = (
    "Extract all text from this image exactly as it appears, verbatim. "
    "Do not summarize, organize, or add commentary — just output the raw text."
)
setup_logging()


def call_model_with_image(
    image_b64: str,
    model_name: str,
    model_url: str,
    prompt: str = DEFAULT_PROMPT,
    retries: int = 3,
    timeout: int = 360,
) -> Optional[str]:
    """Send a base64 image + prompt to an Ollama-style /api/generate endpoint
    and return the model's text response, or None if all retries fail."""
    if not model_name or not model_url:
        raise EnvironmentError(
            "model_name and model_url must be set (check your .env has "
            "the matching *_MODEL / *_URL pair defined)"
        )

    logging.info("Calling %s with prompt: %s", model_name, prompt[:80])

    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 2048},    
        }

    for attempt in range(1, retries + 1):
        logging.info("[%s] Attempt %d/%d (timeout: %ds)", model_name, attempt, retries, timeout)
        try:
            response = requests.post(model_url, json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            ocr_text = result.get("response", "")
            print("RAW RESPONSE:", result)

            logging.info("[%s] Success on attempt %d", model_name, attempt)
            return ocr_text

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
            wait_time = 5 * attempt
            logging.info("[%s] Retrying in %ds...", model_name, wait_time)
            time.sleep(wait_time)

    logging.error("[%s] All %d attempts failed", model_name, retries)
    return None


def call_qwen(image_b64: str, prompt: str = DEFAULT_PROMPT, **kwargs) -> Optional[str]:
    return call_model_with_image(image_b64, model_name=QWEN, model_url=QWEN_URL, prompt=prompt, **kwargs)


def call_gemma(image_b64: str, prompt: str = DEFAULT_PROMPT, **kwargs) -> Optional[str]:
    return call_model_with_image(image_b64, model_name=GEMMA, model_url=GEMMA_URL, prompt=prompt, **kwargs)


if __name__ == "__main__":
    import base64
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ocr_client.py <image_path> [qwen|gemma]")
        sys.exit(1)

    image_path = sys.argv[1]
    model_choice = sys.argv[2] if len(sys.argv) > 2 else "gemma"

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    caller = call_qwen if model_choice == "qwen" else call_gemma
    text = caller(image_b64)

    if text is None:
        print("OCR failed -- check the log output above.")
        sys.exit(1)

    print("\n--- Extracted text ---\n")
    print(text)