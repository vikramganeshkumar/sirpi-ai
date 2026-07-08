# PDF RAG System with Guardrails (Google Gemini)

A Python application that answers questions about a PDF document using
Google's Gemini API, with a guardrail pipeline that filters queries before
they reach the model.

> **Note on "RAG":** this implementation summarizes the document and answers
> questions from a truncated slice of its raw text -- there's no
> chunking/embedding/vector search step. If you need true semantic retrieval
> over long documents, that's a separate enhancement, not what's shipped here.

## Project Structure

| File | Purpose |
|---|---|
| `helpers.py` | Shared functions: API key resolution, PDF text extraction, truncation, logging, JSON saving |
| `pdf_rag_guardrails.py` | Core `RAGWithGuardrails` class + CLI entry point |
| `ocr_client.py` | OCR client for scanned/image content via self-hosted Ollama vision models (Qwen, Gemma) |
| `examples.py` | Programmatic usage examples |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Keeps secrets, generated PDFs/images, and logs out of git |
| `.env` | Local-only config: API keys and OCR model endpoints (never committed) |

## Features

- **PDF text extraction** via `pypdf`
- **Document summarization** used as grounding context for the relevance check
- **Guardrail pipeline** run on every query, in order:
  1. Rate limiting (sliding window, in-memory)
  2. Query length limit
  3. Local regex pre-filter for common prompt-injection phrasing
  4. LLM-based relevance check (structured JSON output, fail-closed by default)
- **Retry with backoff** on generation calls
- **Interactive** and **single-question** CLI modes
- **Guardrail event logging** to file for later review
- **OCR fallback** (`ocr_client.py`) for scanned/image content via self-hosted
  Ollama vision models (Qwen, Gemma), with retry/backoff on the HTTP call

## Prerequisites

- Python 3.8+
- Google API key with Gemini API access

## Setup

### 1. Get a Google API key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click "Create API key"
3. Save it securely -- never commit it to git (see `.gitignore`)

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
google-genai>=0.3.0
pypdf>=4.0.0
python-dotenv>=1.0.0
requests>=2.31.0
```

### 3. Set your API key

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

Or pass it per-run with `--api-key` (supported anywhere `get_api_key()` from
`helpers.py` is used).

### 4. Configure OCR model endpoints (optional, only needed for `ocr_client.py`)

Create a `.env` file in the project root (never commit this -- it's covered
by `.gitignore`):

```
GOOGLE_API_KEY=your-api-key-here
QWEN=qwen2.5vl:7b
QWEN_URL=https://your-qwen-endpoint
GEMMA=gemma3:12b
GEMMA_URL=https://ocr-ollama.slicearrow.com/api/generate
```

`ocr_client.py` loads these via `python-dotenv` and will raise a clear
`EnvironmentError` if a model's name/URL pair isn't set when you try to use it.

## Usage

### Single question

```bash
python pdf_rag_guardrails.py document.pdf "What is the main topic of this document?"
```

### Interactive mode

```bash
python pdf_rag_guardrails.py document.pdf --interactive
```

```
Ready for Q&A -- type 'exit' to quit
Document summary: [generated summary]

Your question: What are the key findings?
Answer: [generated answer]

Your question: exit
Goodbye!
```

### Programmatic usage

```python
from pdf_rag_guardrails import RAGWithGuardrails, GuardrailConfig

rag = RAGWithGuardrails(api_key="your-api-key")
answer = rag.process("document.pdf", "What is the main topic?")
print(answer)
```

See `examples.py` for more, including a fail-open custom config, off-topic
blocking, saving results to JSON, and batch processing a folder of PDFs.

### OCR (scanned/image content)

Standalone CLI:

```bash
python ocr_client.py path/to/image.png gemma
```

Second argument selects `qwen` or `gemma` (defaults to `gemma`).

Programmatic usage:

```python
import base64
from ocr_client import call_gemma, call_qwen

with open("scan.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

text = call_gemma(image_b64)
if text is None:
    print("OCR failed after retries -- check logs")
else:
    print(text)
```

Both `call_gemma()` and `call_qwen()` wrap `call_model_with_image()`, which
retries on timeout/connection errors, retries on 5xx server errors, but
fails fast (no retry) on 4xx client errors since those won't resolve on
their own.

**Prompt tuning matters here.** The default prompt asks for verbatim text
extraction. A vague prompt (e.g. "what do you see?") tends to produce a
narrative summary instead of raw OCR text -- fine for a quick sanity check,
but not what you want feeding into a downstream pipeline. If output comes
back empty or stuck on repeated whitespace, increase `num_predict` in the
`options` payload inside `ocr_client.py` (the model likely hit a token cap
mid-generation).

## Guardrail Configuration

`GuardrailConfig` (in `pdf_rag_guardrails.py`) controls the pipeline:

| Field | Default | Meaning |
|---|---|---|
| `max_query_length` | 500 | Reject queries longer than this |
| `max_retries` | 2 | Retries for generation calls before failing |
| `retry_backoff_seconds` | 1.5 | Backoff multiplier between retries |
| `fail_open` | `False` | If the LLM relevance check errors, block (`False`) or allow (`True`) |
| `rate_limit_window_seconds` | 60 | Sliding window size for rate limiting |
| `rate_limit_max_requests` | 20 | Max queries allowed per window |
| `log_file` | `guardrail_events.log` | Where blocked/allowed events are logged |

```python
config = GuardrailConfig(fail_open=True, rate_limit_max_requests=50)
rag = RAGWithGuardrails(api_key="your-api-key", config=config)
```

**Fail-open vs fail-closed:** by default, if the relevance-check API call
itself fails (network error, malformed response, etc.), the query is
**blocked**. Set `fail_open=True` only for trusted/internal use cases where
availability matters more than strict filtering.

## Known Limitations

- **Not true RAG** -- no embeddings or vector search; answers come from a
  truncated slice of the document text, so very long documents may lose
  relevant context.
- **Local injection filter is a keyword/regex pre-filter**, not a robust
  jailbreak classifier -- it catches obvious attempts, not novel phrasing.
- **Rate limiting is per-process/in-memory** -- it resets on restart and
  doesn't coordinate across multiple processes or instances.
- **Scanned PDFs** (image-only) aren't automatically routed through OCR yet
  -- `ocr_client.py` exists as a standalone tool but isn't wired into the
  main PDF extraction flow in `helpers.py`.
- **Token limits** apply to Gemini's context window; very long documents may
  be truncated in the answer-generation step.
- **OCR endpoints are self-hosted and external** -- `ocr_client.py` depends
  on a third-party Ollama server being reachable; no guardrail pipeline runs
  on OCR requests (that pipeline currently only covers the Gemini Q&A flow).

## Security

- Never commit your API key -- use `GOOGLE_API_KEY` or `--api-key`, not hardcoded values
- `.gitignore` excludes `.env`, credential files, PDFs, and common image
  formats (`.png`/`.jpg`/`.jpeg`) by default -- useful since OCR testing
  tends to leave test images lying around
- Guardrail events (blocked queries, injection attempts) are logged to
  `guardrail_events.log` -- review periodically if deploying this beyond local use
- PDFs and generated text aren't persisted outside the running process except
  where you explicitly call `save_json`

## Troubleshooting

**`FileNotFoundError: PDF not found`** -- check the path is correct and the file exists.

**`Error: set the GOOGLE_API_KEY environment variable`** -- export it or pass `--api-key`.

**Guardrail blocks a question that seems on-topic** -- check
`guardrail_events.log` for the reason; the LLM relevance check can be wrong
occasionally. Consider `fail_open=True` if this happens often in a trusted
context, or refine the injection regex list in `pdf_rag_guardrails.py`.

**OCR response comes back empty or full of repeated whitespace** -- this
usually means the model hit its output token cap mid-generation (the raw
Ollama response will show `"done": false` when this happens). Fix by raising
`num_predict` in the `options` dict inside `call_model_with_image()` in
`ocr_client.py` (e.g. to `2048` or higher for dense images).

**OCR response is a narrative summary instead of raw text** -- the prompt is
too vague. Use a prompt that explicitly says "verbatim" / "do not summarize"
-- see `DEFAULT_PROMPT` in `ocr_client.py`.

**`EnvironmentError: model_name and model_url must be set`** -- your `.env`
is missing `QWEN`/`QWEN_URL` or `GEMMA`/`GEMMA_URL`. Check with `type .env`
(Windows) or `cat .env` (Mac/Linux) that both values in the pair you're
using are present.

## License

Apache License 2.0 -- see `LICENSE`.