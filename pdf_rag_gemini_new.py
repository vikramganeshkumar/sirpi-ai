#!/usr/bin/env python3
import os
import sys

try:
    import google.genai as genai
except ImportError:
    print("Error: google-genai not installed")
    sys.exit(1)

try:
    from pypdf import PdfReader
except ImportError:
    print("Error: pypdf not installed")
    sys.exit(1)

class SimpleRAG:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.text = ""

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        reader = PdfReader(pdf_path)
        print(f"[INFO] PDF loaded: {len(reader.pages)} pages")
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text

    def answer_question(self, query: str) -> str:
        prompt = f"""Read this document and answer the question:

DOCUMENT:
{self.text[:5000]}

QUESTION: {query}

ANSWER:"""
        
        response = self.client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
        )
        return response.text

    def process(self, pdf_path: str, query: str):
        print("[STEP 1] Loading PDF...")
        self.text = self.extract_text_from_pdf(pdf_path)
        
        print("[STEP 2] Generating answer...")
        answer = self.answer_question(query)
        
        return answer

if __name__ == "__main__":
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: Set GOOGLE_API_KEY")
        sys.exit(1)
    
    if len(sys.argv) < 3:
        print("Usage: python pdf_rag_gemini_new.py <pdf> <question>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    query = sys.argv[2]
    
    try:
        rag = SimpleRAG(api_key=api_key)
        answer = rag.process(pdf_path, query)
        print("\n" + "="*80)
        print("ANSWER:")
        print("="*80)
        print(answer)
        print("="*80)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)