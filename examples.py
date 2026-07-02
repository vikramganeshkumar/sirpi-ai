#!/usr/bin/env python3
"""
Example usage of the PDF RAG system with guardrails.
Demonstrates using RAGWithGuardrails programmatically.
"""
import os
import sys
from pathlib import Path

from helpers import get_api_key, print_section, save_json
from pdf_rag_guardrails import GuardrailConfig, RAGWithGuardrails


def example_single_query():
    print_section("EXAMPLE 1: Single Query")
    api_key = get_api_key()
    rag = RAGWithGuardrails(api_key=api_key)

    demo_pdf = "sample.pdf"
    if not os.path.exists(demo_pdf):
        print(f"Note: {demo_pdf} not found. Provide a real PDF to run this example.")
        return

    answer = rag.process(demo_pdf, "What is this document about?")
    print(answer)


def example_off_topic_blocked():
    """Shows the guardrail rejecting a question unrelated to the document."""
    print_section("EXAMPLE 2: Off-topic Question Gets Blocked")
    api_key = get_api_key()
    rag = RAGWithGuardrails(api_key=api_key)

    demo_pdf = "sample.pdf"
    if not os.path.exists(demo_pdf):
        print(f"Note: {demo_pdf} not found.")
        return

    rag.load_document(demo_pdf)
    allowed, reason = rag.check_query("What's the weather like today?")
    print(f"Allowed: {allowed} | Reason: {reason}")


def example_custom_config():
    """A fail-open, higher-throughput config -- useful for a trusted internal tool
    where you'd rather degrade gracefully than block on a flaky guardrail call."""
    print_section("EXAMPLE 3: Custom Guardrail Config")
    api_key = get_api_key()
    config = GuardrailConfig(fail_open=True, rate_limit_max_requests=50, max_query_length=1000)
    rag = RAGWithGuardrails(api_key=api_key, config=config)

    demo_pdf = "sample.pdf"
    if not os.path.exists(demo_pdf):
        print(f"Note: {demo_pdf} not found.")
        return

    answer = rag.process(demo_pdf, "Summarize the key points of this document")
    print(answer)


def example_save_results():
    print_section("EXAMPLE 4: Save Results to JSON")
    api_key = get_api_key()
    rag = RAGWithGuardrails(api_key=api_key)

    demo_pdf = "sample.pdf"
    if not os.path.exists(demo_pdf):
        print(f"Note: {demo_pdf} not found.")
        return

    query = "Summarize the key points of this document"
    answer = rag.process(demo_pdf, query)
    save_json({"query": query, "answer": answer}, "rag_results.json")
    print("Saved to rag_results.json")


def example_batch_processing():
    print_section("EXAMPLE 5: Batch Processing Multiple PDFs")
    api_key = get_api_key()
    rag = RAGWithGuardrails(api_key=api_key)

    pdf_dir = Path("documents")
    pdfs = list(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
    if not pdfs:
        print(f"Note: no PDFs found in {pdf_dir}/. Add some to run this example.")
        return

    results = []
    for pdf_path in pdfs:
        print(f"Processing {pdf_path.name}...")
        try:
            answer = rag.process(str(pdf_path), "What is the main topic of this document?")
            results.append({"file": pdf_path.name, "answer": answer})
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")

    save_json(results, "batch_results.json")
    print(f"Processed {len(results)} PDFs. Saved to batch_results.json")


EXAMPLES = {
    "1": example_single_query,
    "2": example_off_topic_blocked,
    "3": example_custom_config,
    "4": example_save_results,
    "5": example_batch_processing,
}


def show_menu():
    print_section("PDF RAG SYSTEM - USAGE EXAMPLES")
    print(
        """
Examples:
  1. Single Query            - Answer one question from a PDF
  2. Off-topic Blocked       - See the guardrail reject an unrelated question
  3. Custom Guardrail Config - Fail-open mode, custom rate limits
  4. Save Results            - Export results to JSON
  5. Batch Processing        - Process multiple PDFs

Before running:
  export GOOGLE_API_KEY="your-key"

Run all:      python examples.py
Run specific: python examples.py 1
"""
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in EXAMPLES:
        EXAMPLES[sys.argv[1]]()
    else:
        show_menu()