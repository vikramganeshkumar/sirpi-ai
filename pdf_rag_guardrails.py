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

class RAGWithGuardrails:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.text = ""
        self.summary = ""

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

    def create_document_summary(self) -> str:
        """Create a summary of the document for guardrail checking"""
        prompt = f"""Briefly summarize what this document is about in 2-3 sentences:

{self.text[:2000]}

Summary:"""
        
        response = self.client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
        )
        return response.text

    def is_on_topic(self, query: str) -> tuple[bool, str]:
        """
        Check if the user's question is related to the document.
        Returns: (is_on_topic: bool, reason: str)
        """
        prompt = f"""Document is about: {self.summary}

User question: "{query}"

Is this question asking about the document topic? Answer with YES or NO, then briefly explain why.

Format your response as:
ANSWER: YES or NO
REASON: [explanation]"""
        
        try:
            response = self.client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt,
            )
            result = response.text.strip()
            
            # Parse the response
            is_on_topic = "YES" in result.upper()
            reason = result.split("REASON:")[-1].strip() if "REASON:" in result else "Unable to determine relevance"
            
            return is_on_topic, reason
        except Exception as e:
            print(f"[WARNING] Could not validate relevance: {e}")
            return True, "Could not validate"  # Default to allowing if check fails

    def answer_question(self, query: str) -> str:
        """Generate an answer based on the document"""
        prompt = f"""Based on this document:

{self.text[:5000]}

Answer this question: {query}

If the answer is not in the document, say "I don't find this information in the document."

Answer:"""
        
        response = self.client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
        )
        return response.text

    def process(self, pdf_path: str, query: str):
        """Process PDF and answer question with guardrails"""
        print("[STEP 1] Loading PDF...")
        self.text = self.extract_text_from_pdf(pdf_path)
        
        print("[STEP 2] Creating document summary...")
        self.summary = self.create_document_summary()
        print(f"[INFO] Document is about: {self.summary[:100]}...")
        
        print("[STEP 3] Checking if question is on-topic...")
        is_on_topic, reason = self.is_on_topic(query)
        
        if not is_on_topic:
            print(f"\n[⚠️  GUARDRAIL] Off-topic question detected!")
            print(f"Reason: {reason}")
            return f"❌ Off-topic question!\n\nYour question: '{query}'\n\nThis document is about: {self.summary}\n\nPlease ask a question related to the document."
        
        print("[STEP 4] Generating answer...")
        answer = self.answer_question(query)
        
        return answer

    def interactive_mode(self, pdf_path: str):
        """Interactive Q&A with guardrails"""
        print("[STEP 1] Loading PDF...")
        self.text = self.extract_text_from_pdf(pdf_path)
        
        print("[STEP 2] Creating document summary...")
        self.summary = self.create_document_summary()
        print(f"\n📄 Document Summary: {self.summary}")
        print("\n" + "="*80)
        print("Ready for Q&A! Ask questions about the document.")
        print("Type 'exit' to quit.")
        print("="*80 + "\n")
        
        while True:
            try:
                query = input("Your question: ").strip()
                if query.lower() == 'exit':
                    print("Goodbye!")
                    break
                
                if not query:
                    continue
                
                print("\n[Checking relevance...]")
                is_on_topic, reason = self.is_on_topic(query)
                
                if not is_on_topic:
                    print(f"⚠️  Off-topic: {reason}")
                    print(f"This document is about: {self.summary}\n")
                    continue
                
                print("[Generating answer...]")
                answer = self.answer_question(query)
                
                print(f"\n{'='*80}")
                print(f"Answer:\n{answer}")
                print(f"{'='*80}\n")
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}\n")

if __name__ == "__main__":
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: Set GOOGLE_API_KEY")
        sys.exit(1)
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single question: python script.py <pdf> <question>")
        print("  Interactive: python script.py <pdf> --interactive")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    try:
        rag = RAGWithGuardrails(api_key=api_key)
        
        if len(sys.argv) > 2 and sys.argv[2] == "--interactive":
            # Interactive mode
            rag.interactive_mode(pdf_path)
        elif len(sys.argv) > 2:
            # Single question
            query = " ".join(sys.argv[2:])
            answer = rag.process(pdf_path, query)
            print("\n" + "="*80)
            print(answer)
            print("="*80)
        else:
            print("Error: Please provide a question or use --interactive")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
