"""Document ingestion pipeline."""
import uuid
import base64
import shutil
from pathlib import Path
from typing import List, Dict

import fitz

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from app import config


def describe_page_image(image_path: str) -> str:
    """
    Generate a description of a page image using vision model.
    
    Args:
        image_path: Path to page image file
        
    Returns:
        Text description of the image content
    """
    try:
        # Encode image to base64
        with open(image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Set up vision LLM
        llm = ChatOpenAI(
            model=config.VISION_MODEL,
            temperature=0.1,
            openai_api_key=config.OPENAI_API_KEY,
            max_tokens=500
        )
        
        # Prompt for description
        prompt = """Describe the content of this document page in detail. 
Include all text, diagrams, charts, tables, images, and any other visual elements.
Be comprehensive and factual. This description will be used for document search."""
        
        message_content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
            }
        ]
        
        response = llm.invoke([HumanMessage(content=message_content)])
        return response.content
        
    except Exception as e:
        print(f"  Warning: Could not describe image {image_path}: {e}")
        return "Image content (description unavailable)"


def render_pdf_pages_as_images(pdf_path: Path, output_dir: Path) -> Dict[int, str]:
    """
    Render each PDF page as an image.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save page images
        
    Returns:
        Dictionary mapping page numbers to image file path
    """
    pdf_name = pdf_path.stem
    pages_by_number = {}
    
    try:
        doc = fitz.open(str(pdf_path))
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Render page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
            
            # Create filename
            image_filename = f"{pdf_name}_page{page_num + 1}.png"
            image_path = output_dir / image_filename
            
            # Save as PNG
            pix.save(str(image_path))
            
            pages_by_number[page_num + 1] = str(image_path)
        
        doc.close()
        print(f"  Rendered {len(pages_by_number)} pages as images")
        
    except Exception as e:
        print(f"  Error rendering pages from {pdf_path.name}: {e}")
    
    return pages_by_number


def load_documents(docs_dir: Path) -> tuple[List[Document], Dict[str, Dict[int, str]]]:
    """
    Load all documents from the specified directory and render PDF pages as images.
    
    Args:
        docs_dir: Path to directory containing documents
        
    Returns:
        Tuple of (documents list, page images mapping by source and page number)
    """
    documents = []
    all_images = {}  # {source_filename: {page_num: image_path}}
    
    # Clear and recreate images directory
    images_dir = config.DATA_DIR / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
        print("Cleared existing images directory")
    images_dir.mkdir(parents=True, exist_ok=True)
    
    for file_path in docs_dir.rglob("*"):
        if not file_path.is_file():
            continue
            
        try:
            if file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()
                print(f"\nLoaded PDF: {file_path.name} ({len(docs)} pages)")
                
                # Render PDF pages as images
                print(f"Rendering pages from {file_path.name}...")
                pages_as_images = render_pdf_pages_as_images(file_path, images_dir)
                if pages_as_images:
                    all_images[file_path.name] = pages_as_images
                
                # Check for image-only pages and generate descriptions
                if config.DESCRIBE_IMAGE_PAGES:
                    print(f"Checking for image-only pages...")
                    for doc in docs:
                        page_text = doc.page_content.strip()
                        page_num = doc.metadata.get('page', 0)
                        
                        # If page has very little text, it's likely image-only
                        if len(page_text) < config.MIN_TEXT_LENGTH:
                            page_image = pages_as_images.get(page_num)
                            if page_image:
                                print(f"  Page {page_num}: Low text ({len(page_text)} chars), generating description...")
                                description = describe_page_image(page_image)
                                # Replace sparse text with AI description
                                doc.page_content = f"[AI-Generated Description]\n{description}\n\n[Original Text]\n{page_text}"
                
            elif file_path.suffix.lower() == ".md":
                loader = UnstructuredMarkdownLoader(str(file_path))
                docs = loader.load()
                print(f"\nLoaded Markdown: {file_path.name}")
                
            elif file_path.suffix.lower() == ".txt":
                loader = TextLoader(str(file_path), encoding='utf-8')
                docs = loader.load()
                print(f"\nLoaded Text: {file_path.name}")
                
            else:
                print(f"\nSkipping unsupported file: {file_path.name}")
                continue
            
            # Add source to metadata
            for doc in docs:
                doc.metadata["source"] = file_path.name
                
            documents.extend(docs)
            
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
            continue
    
    return documents, all_images


def chunk_documents(documents: List[Document], images_mapping: Dict[str, Dict[int, str]]) -> List[Document]:
    """
    Split documents into chunks and associate page images with chunks.
    
    Args:
        documents: List of documents to chunk
        images_mapping: Dictionary mapping source files and pages to page image path
        
    Returns:
        List of chunked documents with page image metadata
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Add chunk IDs and associate images
    for chunk in chunks:
        chunk.metadata["chunk_id"] = str(uuid.uuid4())
        
        # Associate images from the same page
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        
        if source and page and source in images_mapping:
            page_image = images_mapping[source].get(page)
            if page_image:
                chunk.metadata["images"] = [page_image]
    
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    chunks_with_images = sum(1 for c in chunks if c.metadata.get("images"))
    if chunks_with_images:
        print(f"{chunks_with_images} chunks have associated page images")
    
    return chunks


def create_vector_store(chunks: List[Document]) -> FAISS:
    """
    Create and persist a FAISS vector store from document chunks.
    
    Args:
        chunks: List of document chunks
        
    Returns:
        FAISS vector store
    """
    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=config.OPENAI_API_KEY
    )
    
    print(f"Creating embeddings for {len(chunks)} chunks...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Save to disk
    vector_store.save_local(str(config.VECTOR_STORE_DIR))
    print(f"Vector store saved to {config.VECTOR_STORE_DIR}")
    
    return vector_store


def ingest_documents() -> None:
    """
    Main ingestion pipeline: load, chunk, embed, and store documents.
    """
    print("=" * 60)
    print("Starting document ingestion...")
    print("=" * 60)
    
    # Check if raw_docs directory exists and has files
    if not config.RAW_DOCS_DIR.exists():
        raise FileNotFoundError(f"Directory not found: {config.RAW_DOCS_DIR}")
    
    files = list(config.RAW_DOCS_DIR.rglob("*"))
    doc_files = [f for f in files if f.is_file() and f.suffix.lower() in ['.pdf', '.md', '.txt']]
    
    if not doc_files:
        raise ValueError(f"No documents found in {config.RAW_DOCS_DIR}")
    
    print(f"Found {len(doc_files)} documents to process")
    print()
    
    # Load documents and extract images
    documents, images_mapping = load_documents(config.RAW_DOCS_DIR)
    if not documents:
        raise ValueError("No documents were successfully loaded")
    
    print()
    
    # Chunk documents and associate images
    chunks = chunk_documents(documents, images_mapping)
    print()
    
    # Create and save vector store
    create_vector_store(chunks)
    
    print()
    print("=" * 60)
    print("Ingestion complete!")
    print("=" * 60)


if __name__ == "__main__":
    ingest_documents()
