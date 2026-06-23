# PDF-based RAG System with Google Gemini

A Python application that implements Retrieval-Augmented Generation (RAG) for PDF documents using Google's Gemini API. This system extracts text from PDFs, creates semantic embeddings, and answers questions based on the document content.

## Features

- **PDF Text Extraction**: Automatically extracts and processes text from PDF files
- **Intelligent Chunking**: Splits documents into overlapping chunks for better context preservation
- **Semantic Search**: Uses Google Gemini embeddings to find relevant document sections
- **Context-Aware Answers**: Generates answers using Gemini with retrieved document context
- **Interactive Mode**: Ask multiple questions about the same document
- **Single-Query Mode**: Get quick answers to specific questions
- **Detailed Metadata**: Track page numbers and chunk positions for retrieved content

## Prerequisites

- Python 3.8+
- Google API key with Gemini API access

## Installation

### 1. Get a Google API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click "Create API key"
3. Save your API key securely

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install google-generativeai>=0.3.0 pypdf>=4.0.0
```

## Configuration

### Option A: Environment Variable (Recommended)

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

### Option B: Command Line Argument

Pass the API key directly with `--api-key`:

```bash
python pdf_rag_gemini.py document.pdf "Your question" --api-key "your-api-key-here"
```

## Usage

### Basic Usage - Answer a Single Question

```bash
python pdf_rag_gemini.py document.pdf "What is the main topic of this document?"
```

### Interactive Mode - Ask Multiple Questions

```bash
python pdf_rag_gemini.py document.pdf --interactive
```

In interactive mode, you can ask multiple questions about the same document without reprocessing:

```
[READY] PDF loaded and indexed. You can now ask questions.
Type 'exit' to quit.

Your question: What are the key findings?
Answer: [Generated answer based on document context]

Your question: Who conducted this research?
Answer: [Another answer]

Your question: exit
Goodbye!
```

### Advanced Options

```bash
# Use a different Gemini model
python pdf_rag_gemini.py document.pdf "Your question" --model gemini-1.5-pro

# Customize chunk size (default: 500 characters)
python pdf_rag_gemini.py document.pdf "Your question" --chunk-size 1000

# Retrieve more context chunks (default: 3)
python pdf_rag_gemini.py document.pdf "Your question" --top-k 5

# Specify API key directly
python pdf_rag_gemini.py document.pdf "Your question" --api-key "your-api-key"
```

## How It Works

### 1. **Text Extraction**
   - Reads the PDF file page by page
   - Extracts and cleans text content
   - Tracks page numbers for reference

### 2. **Text Chunking**
   - Splits text into overlapping chunks (~500 characters by default)
   - Maintains context by overlapping chunk boundaries
   - Preserves page information in metadata

### 3. **Embedding Generation**
   - Uses Google Gemini's embedding model (`embedding-001`)
   - Creates semantic embeddings for each chunk
   - Stores embeddings for similarity search

### 4. **Semantic Search**
   - Embeds the user's query using the same model
   - Calculates cosine similarity with all chunks
   - Retrieves top-K most relevant chunks (default: 3)

### 5. **Answer Generation**
   - Creates a prompt with retrieved context and user query
   - Uses Gemini to generate contextual answers
   - Ensures answers are grounded in the document

## Output Example

```
================================================================================
ANSWER
================================================================================
Based on the document, the main findings indicate that [answer text]. This is
supported by evidence from pages 2-5, showing that [relevant quote or summary].

================================================================================
Retrieved 3 chunks from 47 total chunks
```

## Supported Gemini Models

- `gemini-1.5-flash` (default, faster and efficient)
- `gemini-1.5-pro` (more capable, slower)
- `gemini-2.0-flash` (latest, if available)

Check [Google AI Studio](https://aistudio.google.com) for the latest available models.

## Customization

### Modify Chunk Size

For longer documents with complex context, increase chunk size:

```bash
python pdf_rag_gemini.py document.pdf "Your question" --chunk-size 1000
```

### Retrieve More Context

For better accuracy, retrieve more chunks:

```bash
python pdf_rag_gemini.py document.pdf "Your question" --top-k 5
```

### Use in Your Code

```python
from pdf_rag_gemini import PDFRAGSystem

# Initialize
rag = PDFRAGSystem(api_key="your-api-key")

# Process PDF and answer
result = rag.process_pdf_and_answer(
    pdf_path="document.pdf",
    query="What is the main topic?"
)

print(result["answer"])
```

## Limitations

- **PDF Format**: Works best with text-based PDFs. Scanned PDFs (images) require OCR
- **Large PDFs**: Very large documents may take longer to process
- **Token Limits**: Google Gemini has token limits; very long context may be truncated
- **Context Window**: The system retrieves limited chunks; questions requiring synthesis across many chapters may need higher `--top-k`

## Troubleshooting

### "FileNotFoundError: PDF file not found"
```bash
# Make sure the PDF path is correct
python pdf_rag_gemini.py /full/path/to/document.pdf "Your question"
```

### "Error generating embeddings: Invalid API key"
```bash
# Verify your API key is valid and has Gemini API access
# Check environment variable or use --api-key flag
export GOOGLE_API_KEY="your-valid-api-key"
```

### "Error reading PDF: Invalid PDF"
- Ensure the file is a valid PDF
- The file isn't corrupted or password-protected
- For scanned PDFs, consider using OCR tools first

### Slow Processing
- Use `gemini-1.5-flash` instead of pro for faster responses
- Reduce chunk size to process fewer embeddings
- For interactive mode, chunking/embedding happens once

## API Costs

- **Embedding API**: Free tier available (check current limits)
- **Gemini Generation**: Variable cost depending on model and tokens used
- Monitor your API usage at [Google Cloud Console](https://console.cloud.google.com)

## Performance Tips

1. **Reuse the system**: In interactive mode, embeddings are calculated once
2. **Optimize chunks**: Balance between too small (many chunks) and too large (lost context)
3. **Select model wisely**: `gemini-1.5-flash` is ideal for most use cases
4. **Cache embeddings**: For repeated queries on same PDF, embeddings are stored in memory

## Security

- Never commit your API key to version control
- Use environment variables for sensitive information
- The script doesn't store PDFs or embeddings outside memory
- Clear embeddings after each session

## License

This script is provided as-is for educational and commercial use.

## Support

For issues with:
- **Google API**: Check [Gemini API Documentation](https://ai.google.dev/docs)
- **pypdf**: See [pypdf GitHub](https://github.com/py-pdf/pypdf)
- **This script**: Review the code comments and function docstrings

## Example Workflows

### Research Paper Analysis
```bash
export GOOGLE_API_KEY="your-key"
python pdf_rag_gemini.py research_paper.pdf --interactive

# Ask questions like:
# - What methodology was used?
# - What were the main results?
# - What limitations did the authors mention?
```

### Document Q&A
```bash
python pdf_rag_gemini.py contract.pdf "What are the payment terms?" --api-key your-key
```

### Batch Processing
```bash
for pdf in *.pdf; do
  python pdf_rag_gemini.py "$pdf" "Summarize this document" > "${pdf%.pdf}_summary.txt"
done
```

---

**Last Updated**: June 2026