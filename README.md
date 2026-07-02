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
| `examples.py` | Programmatic usage examples |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Keeps secrets, generated PDFs, and logs out of git |

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
```

### 3. Set your API key

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

Or pass it per-run with `--api-key` (supported anywhere `get_api_key()` from
`helpers.py` is used).

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
- **Scanned PDFs** (image-only) aren't supported without OCR.
- **Token limits** apply to Gemini's context window; very long documents may
  be truncated in the answer-generation step.

## Security

- Never commit your API key -- use `GOOGLE_API_KEY` or `--api-key`, not hardcoded values
- `.gitignore` excludes `.env`, credential files, and PDFs by default
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

## License

Apache License 2.0 -- see `LICENSE`.