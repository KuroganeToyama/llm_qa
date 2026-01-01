# LLM QA System

A simple LLM QA pipeline for answering questions based on your local documents. Because my friend is actually making a proper one for law QA and I just simply got interested in the concept.

## Features

- Support for PDF, Markdown, and TXT documents
- Vector-based semantic search using FAISS
- Powered by OpenAI GPT-4 and embeddings
- Simple CLI interface
- Grounded answers with optional source citations

## Prerequisites

- Python 3.10 or higher
- OpenAI API key

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd llm_qa
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure environment variables:**
   ```bash
   copy .env.example .env
   ```
   
   Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_actual_api_key_here
   ```

## Usage

### 1. Add Your Documents

Place your documents (PDF, Markdown, or TXT files) in the `data/raw_docs/` directory:

```bash
data/raw_docs/
├── syllabus.pdf
├── notes.md
└── readme.txt
```

### 2. Build the Index

Run the ingestion script to process your documents and create the vector store:

```bash
python scripts/rebuild_index.py
```

This will:
- Load all documents from `data/raw_docs/`
- Split them into chunks
- Create embeddings
- Save associated images in `data/images/`
- Save the vector store to `data/vector_store/`

### 3. Ask Questions

Use the CLI to ask questions about your documents:

```bash
python -m app.main "What is the grading policy?"
```

**With source citations:**
```bash
python -m app.main "What is the grading policy?" --sources
```

## Configuration

You can customize the system behavior by editing `.env`:

```env
# Model Configuration
MODEL_NAME=gpt-4-turbo-preview
EMBEDDING_MODEL=text-embedding-3-small

# Retrieval Configuration
TOP_K=5                    # Number of chunks to retrieve
CHUNK_SIZE=600            # Size of each text chunk
CHUNK_OVERLAP=100         # Overlap between chunks

# LLM Configuration
TEMPERATURE=0.2           # Lower = more deterministic
```

## Project Structure

```
llm_qa/
├── app/
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── types.py           # Data models
│   ├── prompts.py         # Prompt templates
│   ├── ingest.py          # Document ingestion pipeline
│   ├── query.py           # Question answering pipeline
│   └── main.py            # CLI interface
│
├── data/
│   ├── raw_docs/          # Place your documents here
│   └── vector_store/      # Generated vector store (auto-created)
│
├── scripts/
│   └── rebuild_index.py   # Script to rebuild the index
│
├── .env                   # Environment variables (you create this)
├── .env.example           # Example environment file
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## How It Works

### 1. Ingestion Phase

**Document Loading:**
- Documents are loaded from `data/raw_docs/`
- PDF, Markdown, and TXT formats supported

**PDF Page Rendering:**
- Each PDF page is rendered as a high-resolution image (144 DPI)
- Images saved to `data/images/` for later use
- Pages are associated with their rendered images

**Image-Only Page Handling:**
- Pages with minimal text (< 50 characters) are detected
- GPT-4o generates detailed descriptions of these pages
- Descriptions include charts, diagrams, tables, and visual elements
- Makes image-heavy pages searchable via AI-generated content

**Text Chunking:**
- Text is split into overlapping chunks (default: 800 characters)
- Overlap (default: 150 characters) prevents information loss
- Each chunk stores metadata: source file, page number, associated images

**Embedding & Storage:**
- Text chunks are embedded using OpenAI embeddings
- Embeddings stored in FAISS vector store
- Image references preserved in chunk metadata

### 2. Query Phase

**Text Retrieval:**
- User question is embedded using the same embedding model
- Vector similarity search retrieves top-K relevant text chunks
- Both normal text and AI-generated image descriptions are searchable

**Image Detection:**
- System checks if retrieved chunks have associated page images
- Images are automatically loaded from disk

**Multi-Modal Generation:**
- **If images present + USE_VISION=true:** 
  - Uses GPT-4o vision model
  - Sends text context + up to 5 page images
  - Model analyzes both text and visual content
  - Generates comprehensive answer incorporating visual information
  
- **If no images or USE_VISION=false:**
  - Uses standard text-only model (GPT-4)
  - Generates answer from text context only

**Response:**
- Answer is grounded in retrieved context (text + images)
- Optional source citations showing which documents/pages were used
- Vision model can reference specific visual elements from page images

## Examples

**Basic query:**
```bash
python app/main.py "What are the main topics covered?"
```

**Query with sources:**
```bash
python app/main.py "What are the main topics covered?" --sources
```

**Output:**
```
Question: What are the main topics covered?

Processing...

============================================================
ANSWER:
============================================================
The main topics covered include machine learning fundamentals,
neural networks, and natural language processing.

============================================================
SOURCES:
============================================================
1. syllabus.pdf:page 2
2. lecture_notes.md
```
