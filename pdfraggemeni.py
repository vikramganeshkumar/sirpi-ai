#!/usr/bin/env python3
"""
PDF-based RAG (Retrieval-Augmented Generation) with Google Gemini
This script extracts text from a PDF, chunks it, embeds it, and uses Gemini for Q&A.
"""

import os
import sys
import argparse
from typing import List, Tuple
from pathlib import Path

try:
    import google.generativeai as genai
    from google.generativeai.types import EmbedContent
except ImportError:
    print("Error: google-generativeai not installed. Install with:")
    print("  pip install google-generativeai")
    sys.exit(1)

try:
    from pypdf import PdfReader
except ImportError:
    print("Error: pypdf not installed. Install with:")
    print("  pip install pypdf")
    sys.exit(1)


class PDFRAGSystem:
    """
    A Retrieval-Augmented Generation system using Google Gemini.
    """

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        """
        Initialize the RAG system with Google Gemini.
        
        Args:
            api_key: Google API key for accessing Gemini
            model: Gemini model to use for generation (default: gemini-1.5-flash)
        """
        genai.configure(api_key=api_key)
        self.generation_model = model
        self.embedding_model = "models/embedding-001"
        
        # Store document chunks and their embeddings
        self.chunks: List[str] = []
        self.embeddings: List[List[float]] = []
        self.metadata: List[dict] = []

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract all text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text from the PDF
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            reader = PdfReader(pdf_path)
            print(f"[INFO] PDF loaded: {len(reader.pages)} pages")
            
            text = ""
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            
            return text
        except Exception as e:
            raise ValueError(f"Error reading PDF: {str(e)}")

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[Tuple[str, dict]]:
        """
        Split text into overlapping chunks for better context preservation.
        
        Args:
            text: Input text to chunk
            chunk_size: Approximate size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of (chunk, metadata) tuples
        """
        chunks = []
        sentences = text.split('.')
        
        current_chunk = ""
        chunk_num = 0
        page_num = 1
        
        for sentence in sentences:
            sentence = sentence.strip() + "."
            
            # Track page numbers from markers in text
            if "--- Page" in sentence:
                page_num = int(sentence.split("Page ")[1].split()[0])
                continue
            
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += " " + sentence
            else:
                if current_chunk.strip():
                    chunks.append((
                        current_chunk.strip(),
                        {"chunk_num": chunk_num, "page": page_num}
                    ))
                    chunk_num += 1
                
                # Create overlap by keeping the end of the previous chunk
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append((
                current_chunk.strip(),
                {"chunk_num": chunk_num, "page": page_num}
            ))
        
        print(f"[INFO] Text split into {len(chunks)} chunks")
        return chunks

    def embed_chunks(self, chunks: List[Tuple[str, dict]]) -> None:
        """
        Generate embeddings for all text chunks using Gemini.
        
        Args:
            chunks: List of (text, metadata) tuples
        """
        print(f"[INFO] Generating embeddings for {len(chunks)} chunks...")
        
        try:
            for i, (chunk_text, metadata) in enumerate(chunks):
                # Generate embedding using Gemini
                embedding = genai.embed_content(
                    model=self.embedding_model,
                    content=chunk_text,
                    task_type="RETRIEVAL_DOCUMENT"
                )
                
                self.chunks.append(chunk_text)
                self.embeddings.append(embedding["embedding"])
                self.metadata.append(metadata)
                
                if (i + 1) % 5 == 0:
                    print(f"[INFO] Embedded {i + 1}/{len(chunks)} chunks")
            
            print(f"[INFO] Embedding complete. Total chunks stored: {len(self.chunks)}")
        except Exception as e:
            raise RuntimeError(f"Error generating embeddings: {str(e)}")

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        """
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a ** 2 for a in vec1) ** 0.5
        norm2 = sum(b ** 2 for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve the most relevant chunks for a given query.
        
        Args:
            query: The user's question
            top_k: Number of top chunks to retrieve
            
        Returns:
            List of relevant chunks
        """
        # Generate embedding for the query
        query_embedding = genai.embed_content(
            model=self.embedding_model,
            content=query,
            task_type="RETRIEVAL_QUERY"
        )["embedding"]
        
        # Calculate similarity scores
        scores = [
            (i, self.cosine_similarity(query_embedding, emb))
            for i, emb in enumerate(self.embeddings)
        ]
        
        # Sort by similarity and get top-k
        top_indices = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
        
        relevant_chunks = []
        for idx, score in top_indices:
            relevant_chunks.append(
                f"[Page {self.metadata[idx]['page']}, Chunk {self.metadata[idx]['chunk_num']}] "
                f"(Relevance: {score:.2f})\n{self.chunks[idx]}"
            )
        
        return relevant_chunks

    def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        """
        Generate an answer using Gemini based on retrieved context.
        
        Args:
            query: The user's question
            context_chunks: Retrieved relevant chunks
            
        Returns:
            Generated answer from Gemini
        """
        # Prepare the context
        context = "\n\n".join(context_chunks)
        
        # Create the prompt for Gemini
        prompt = f"""Based on the following document excerpts, answer the user's question. 
If the answer cannot be found in the provided context, say "I cannot find this information in the provided document."

Document excerpts:
{context}

User question: {query}

Please provide a clear and concise answer based on the document content above."""
        
        try:
            response = genai.GenerativeModel(self.generation_model).generate_content(prompt)
            return response.text
        except Exception as e:
            raise RuntimeError(f"Error generating answer: {str(e)}")

    def process_pdf_and_answer(self, pdf_path: str, query: str) -> dict:
        """
        Complete pipeline: Extract PDF → Chunk → Embed → Retrieve → Generate Answer
        
        Args:
            pdf_path: Path to the PDF file
            query: User's question
            
        Returns:
            Dictionary with answer and retrieved chunks
        """
        # Extract text from PDF
        print(f"\n[STEP 1] Extracting text from {pdf_path}...")
        text = self.extract_text_from_pdf(pdf_path)
        
        # Chunk the text
        print("\n[STEP 2] Chunking text...")
        chunks = self.chunk_text(text)
        
        # Generate embeddings
        print("\n[STEP 3] Generating embeddings...")
        self.embed_chunks(chunks)
        
        # Retrieve relevant chunks
        print("\n[STEP 4] Retrieving relevant chunks...")
        relevant_chunks = self.retrieve_relevant_chunks(query)
        
        print(f"[INFO] Retrieved {len(relevant_chunks)} most relevant chunks")
        
        # Generate answer
        print("\n[STEP 5] Generating answer...")
        answer = self.generate_answer(query, relevant_chunks)
        
        return {
            "query": query,
            "answer": answer,
            "retrieved_chunks": relevant_chunks,
            "total_chunks": len(self.chunks)
        }

    def interactive_qa(self, pdf_path: str) -> None:
        """
        Start an interactive Q&A session with a PDF.
        
        Args:
            pdf_path: Path to the PDF file
        """
        # Extract text and prepare embeddings once
        print(f"\n[INITIALIZATION] Loading {pdf_path}...")
        text = self.extract_text_from_pdf(pdf_path)
        chunks = self.chunk_text(text)
        self.embed_chunks(chunks)
        
        print("\n[READY] PDF loaded and indexed. You can now ask questions.")
        print("Type 'exit' to quit.\n")
        
        while True:
            try:
                query = input("Your question: ").strip()
                if query.lower() == 'exit':
                    print("Goodbye!")
                    break
                
                if not query:
                    continue
                
                # Retrieve and generate answer
                relevant_chunks = self.retrieve_relevant_chunks(query)
                answer = self.generate_answer(query, relevant_chunks)
                
                print(f"\nAnswer:\n{answer}\n")
                print("-" * 80 + "\n")
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {str(e)}\n")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="PDF-based RAG system with Google Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Answer a specific question
  python pdf_rag_gemini.py document.pdf "What is the main topic?" --api-key YOUR_API_KEY
  
  # Interactive mode
  python pdf_rag_gemini.py document.pdf --interactive --api-key YOUR_API_KEY
  
  # Custom model
  python pdf_rag_gemini.py document.pdf "Your question" --model gemini-1.5-pro --api-key YOUR_API_KEY
        """
    )
    
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("query", nargs="?", help="Question to ask (optional, use --interactive instead)")
    parser.add_argument("--api-key", help="Google API key (or set GOOGLE_API_KEY env variable)")
    parser.add_argument("--model", default="gemini-1.5-flash", help="Gemini model to use")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive Q&A mode")
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size in characters")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: Google API key required.")
        print("Provide via --api-key or set GOOGLE_API_KEY environment variable")
        sys.exit(1)
    
    # Initialize the system
    rag_system = PDFRAGSystem(api_key=api_key, model=args.model)
    
    try:
        if args.interactive:
            rag_system.interactive_qa(args.pdf_path)
        elif args.query:
            result = rag_system.process_pdf_and_answer(args.pdf_path, args.query)
            
            print("\n" + "=" * 80)
            print("ANSWER")
            print("=" * 80)
            print(result["answer"])
            print("\n" + "=" * 80)
            print(f"Retrieved {len(result['retrieved_chunks'])} chunks from {result['total_chunks']} total chunks")
        else:
            parser.print_help()
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()