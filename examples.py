#!/usr/bin/env python3
"""
Example usage of the PDF RAG system with Google Gemini.
This demonstrates how to use the system programmatically in your own code.
"""

import os
import json
from pdf_rag_gemini import PDFRAGSystem


def example_single_query():
    """Example 1: Process a PDF and ask a single question"""
    print("=" * 80)
    print("EXAMPLE 1: Single Query")
    print("=" * 80)
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: Set GOOGLE_API_KEY environment variable")
        return
    
    # Initialize RAG system
    rag = PDFRAGSystem(api_key=api_key)
    
    # For demo purposes, we'll create a simple text file
    demo_pdf = "sample.pdf"
    
    if not os.path.exists(demo_pdf):
        print(f"Note: {demo_pdf} not found. Please provide a valid PDF file.")
        print("\nUsage example:")
        print("  result = rag.process_pdf_and_answer('your_document.pdf', 'Your question here?')")
        return
    
    # Process PDF and get answer
    result = rag.process_pdf_and_answer(
        pdf_path=demo_pdf,
        query="What is this document about?"
    )
    
    print("\n" + "-" * 80)
    print("QUERY:")
    print(result["query"])
    print("\n" + "-" * 80)
    print("ANSWER:")
    print(result["answer"])
    print("\n" + "-" * 80)
    print(f"Total chunks in document: {result['total_chunks']}")
    print(f"Chunks retrieved: {len(result['retrieved_chunks'])}")


def example_multiple_queries():
    """Example 2: Load a PDF once and ask multiple questions"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Multiple Queries (Single PDF Load)")
    print("=" * 80)
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: Set GOOGLE_API_KEY environment variable")
        return
    
    rag = PDFRAGSystem(api_key=api_key)
    
    demo_pdf = "sample.pdf"
    if not os.path.exists(demo_pdf):
        print(f"Note: {demo_pdf} not found.")
        print("\nExample code:")
        print("""
    # Load and process PDF once
    text = rag.extract_text_from_pdf('document.pdf')
    chunks = rag.chunk_text(text)
    rag.embed_chunks(chunks)
    
    # Ask multiple questions without reprocessing
    queries = [
        "What are the main topics?",
        "Who are the authors?",
        "What are the conclusions?",
    ]
    
    for query in queries:
        relevant_chunks = rag.retrieve_relevant_chunks(query)
        answer = rag.generate_answer(query, relevant_chunks)
        print(f"Q: {query}")
        print(f"A: {answer}\\n")
        """)
        return
    
    try:
        print(f"\nLoading {demo_pdf}...")
        text = rag.extract_text_from_pdf(demo_pdf)
        chunks = rag.chunk_text(text)
        rag.embed_chunks(chunks)
        
        # Example questions
        queries = [
            "What is the main topic of this document?",
            "What are the key findings?",
            "Who are the authors or contributors?",
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"\n--- Query {i} ---")
            print(f"Q: {query}")
            
            relevant_chunks = rag.retrieve_relevant_chunks(query, top_k=3)
            answer = rag.generate_answer(query, relevant_chunks)
            
            print(f"A: {answer}")
    
    except FileNotFoundError:
        print(f"PDF file {demo_pdf} not found")


def example_with_custom_settings():
    """Example 3: Use custom parameters"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Custom Settings")
    print("=" * 80)
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: Set GOOGLE_API_KEY environment variable")
        return
    
    # Use a different model
    rag = PDFRAGSystem(api_key=api_key, model="gemini-1.5-pro")
    
    demo_pdf = "sample.pdf"
    if not os.path.exists(demo_pdf):
        print(f"Note: {demo_pdf} not found.")
        print("\nExample with custom settings:")
        print("""
    # Use different model
    rag = PDFRAGSystem(api_key=api_key, model="gemini-1.5-pro")
    
    # Extract with custom chunk size (larger for more context)
    text = rag.extract_text_from_pdf('document.pdf')
    chunks = rag.chunk_text(text, chunk_size=1000, overlap=200)
    rag.embed_chunks(chunks)
    
    # Retrieve more chunks for better context
    relevant = rag.retrieve_relevant_chunks("Your question", top_k=5)
    answer = rag.generate_answer("Your question", relevant)
        """)
        return
    
    try:
        print("\nUsing model: gemini-1.5-pro")
        print("Chunk size: 800 characters")
        print("Overlap: 200 characters")
        print("Top-K retrieval: 5 chunks")
        
        text = rag.extract_text_from_pdf(demo_pdf)
        chunks = rag.chunk_text(text, chunk_size=800, overlap=200)
        rag.embed_chunks(chunks)
        
        query = "Provide a comprehensive answer to this: What are the main themes?"
        relevant_chunks = rag.retrieve_relevant_chunks(query, top_k=5)
        answer = rag.generate_answer(query, relevant_chunks)
        
        print(f"\nQ: {query}")
        print(f"\nA: {answer}")
    
    except FileNotFoundError:
        print(f"PDF file {demo_pdf} not found")


def example_save_results():
    """Example 4: Save results to JSON"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Save Results to JSON")
    print("=" * 80)
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: Set GOOGLE_API_KEY environment variable")
        return
    
    rag = PDFRAGSystem(api_key=api_key)
    
    demo_pdf = "sample.pdf"
    if not os.path.exists(demo_pdf):
        print(f"Note: {demo_pdf} not found.")
        print("\nExample code to save results:")
        print("""
    result = rag.process_pdf_and_answer('document.pdf', 'Your question?')
    
    # Save to JSON
    with open('rag_results.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print("Results saved to rag_results.json")
        """)
        return
    
    try:
        # Process PDF
        result = rag.process_pdf_and_answer(
            pdf_path=demo_pdf,
            query="Summarize the key points of this document"
        )
        
        # Prepare data for JSON (remove embeddings if needed)
        output_data = {
            "query": result["query"],
            "answer": result["answer"],
            "total_chunks": result["total_chunks"],
            "retrieved_chunks_count": len(result["retrieved_chunks"]),
        }
        
        # Save to file
        output_file = "rag_results.json"
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Results saved to {output_file}")
        print(json.dumps(output_data, indent=2))
    
    except FileNotFoundError:
        print(f"PDF file {demo_pdf} not found")


def example_batch_processing():
    """Example 5: Process multiple PDFs"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Batch Processing Multiple PDFs")
    print("=" * 80)
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: Set GOOGLE_API_KEY environment variable")
        return
    
    print("\nExample code for batch processing:")
    print("""
from pathlib import Path
import json

api_key = os.environ.get("GOOGLE_API_KEY")
rag = PDFRAGSystem(api_key=api_key)

# Find all PDFs in a directory
pdf_dir = Path("documents")
pdfs = list(pdf_dir.glob("*.pdf"))

results = []

for pdf_path in pdfs:
    print(f"Processing {pdf_path.name}...")
    
    try:
        result = rag.process_pdf_and_answer(
            pdf_path=str(pdf_path),
            query="What is the main topic of this document?"
        )
        
        results.append({
            "file": pdf_path.name,
            "query": result["query"],
            "answer": result["answer"],
            "chunks_used": len(result["retrieved_chunks"])
        })
    
    except Exception as e:
        print(f"Error processing {pdf_path.name}: {e}")

# Save batch results
with open("batch_results.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"\\nProcessed {len(results)} PDFs successfully")
    """)


def show_menu():
    """Display example menu"""
    print("\n" + "=" * 80)
    print("PDF RAG SYSTEM - USAGE EXAMPLES")
    print("=" * 80)
    print("""
This script demonstrates different ways to use the PDF RAG system.

Examples included:
  1. Single Query - Answer one question from a PDF
  2. Multiple Queries - Ask many questions from one loaded PDF
  3. Custom Settings - Use different models and parameters
  4. Save Results - Export results to JSON
  5. Batch Processing - Process multiple PDFs

Before running:
  - Set your Google API key: export GOOGLE_API_KEY="your-key"
  - Have a PDF file ready (or create a sample.pdf)

To run all examples:
  python examples.py

To run specific example:
  python examples.py 1
  python examples.py 2
  etc.
    """)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num == "1":
            example_single_query()
        elif example_num == "2":
            example_multiple_queries()
        elif example_num == "3":
            example_with_custom_settings()
        elif example_num == "4":
            example_save_results()
        elif example_num == "5":
            example_batch_processing()
        else:
            show_menu()
    else:
        show_menu()
        print("\nNote: To actually run examples, you need:")
        print("  - GOOGLE_API_KEY environment variable set")
        print("  - A valid PDF file")
        print("\nThen run: python examples.py 1")