"""Query pipeline for answering questions."""
import base64
from pathlib import Path
from typing import List, Optional

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from app import config
from app.prompts import SYSTEM_PROMPT, get_user_prompt_with_images, format_context
from app.models import QAResponse


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to base64 string.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Base64 encoded string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def load_vector_store() -> FAISS:
    """
    Load the FAISS vector store from disk.
    
    Returns:
        FAISS vector store
        
    Raises:
        FileNotFoundError: If vector store doesn't exist
    """
    if not config.VECTOR_STORE_DIR.exists():
        raise FileNotFoundError(
            f"Vector store not found at {config.VECTOR_STORE_DIR}. "
            "Please run ingestion first: python scripts/rebuild_index.py"
        )
    
    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=config.OPENAI_API_KEY
    )
    
    vector_store = FAISS.load_local(
        str(config.VECTOR_STORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    return vector_store


def retrieve_context(vector_store: FAISS, question: str, k: int = None) -> List[Document]:
    """
    Retrieve relevant context chunks for a question.
    
    Args:
        vector_store: FAISS vector store
        question: User's question
        k: Number of chunks to retrieve (default from config)
        
    Returns:
        List of relevant document chunks
    """
    if k is None:
        k = config.TOP_K
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    
    chunks = retriever.invoke(question)
    return chunks


def generate_answer(question: str, context_chunks: List[Document]) -> str:
    """
    Generate an answer using the LLM (with vision support if enabled).
    
    Args:
        question: User's question
        context_chunks: Retrieved context chunks
        
    Returns:
        Generated answer
    """
    # Collect images from chunks
    all_images = []
    for chunk in context_chunks:
        images = chunk.metadata.get("images", [])
        if images:
            all_images.extend(images)
    
    # Use vision model if images are present and vision is enabled
    if all_images and config.USE_VISION:
        return generate_answer_with_vision(question, context_chunks, all_images)
    
    # Otherwise use standard text-only generation
    context = format_context(context_chunks)
    user_prompt = get_user_prompt_with_images(context, question, has_images=False)
    
    llm = ChatOpenAI(
        model=config.MODEL_NAME,
        temperature=config.TEMPERATURE,
        openai_api_key=config.OPENAI_API_KEY
    )
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    return response.content


def generate_answer_with_vision(question: str, context_chunks: List[Document], image_paths: List[str]) -> str:
    """
    Generate an answer using vision-enabled LLM.
    
    Args:
        question: User's question
        context_chunks: Retrieved context chunks
        image_paths: List of paths to relevant images
        
    Returns:
        Generated answer
    """
    # Format text context
    context = format_context(context_chunks)
    
    # Prepare image content
    image_contents = []
    for img_path in image_paths[:5]:  # Limit to 5 images to avoid token limits
        if Path(img_path).exists():
            try:
                base64_image = encode_image_to_base64(img_path)
                # Determine image type from extension
                ext = Path(img_path).suffix.lower()
                mime_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else f"image/{ext[1:]}"
                
                image_contents.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    }
                })
            except Exception as e:
                print(f"Warning: Could not load image {img_path}: {e}")
    
    # Create multi-modal prompt
    user_prompt_text = get_user_prompt_with_images(context, question, has_images=True)
    
    # Build message content with text and images
    message_content = [{"type": "text", "text": user_prompt_text}]
    message_content.extend(image_contents)
    
    # Initialize vision-capable LLM
    llm = ChatOpenAI(
        model=config.VISION_MODEL,
        temperature=config.TEMPERATURE,
        openai_api_key=config.OPENAI_API_KEY,
        max_tokens=4096
    )
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=message_content)
    ]
    
    response = llm.invoke(messages)
    return response.content


def extract_sources(chunks: List[Document]) -> List[str]:
    """
    Extract source information from chunks.
    
    Args:
        chunks: Retrieved document chunks
        
    Returns:
        List of source strings
    """
    sources = []
    seen = set()
    
    for chunk in chunks:
        source = chunk.metadata.get('source', 'unknown')
        page = chunk.metadata.get('page')
        
        if page:
            source_str = f"{source}:page {page}"
        else:
            source_str = source
        
        if source_str not in seen:
            sources.append(source_str)
            seen.add(source_str)
    
    return sources


def answer_question(question: str, return_sources: bool = False) -> QAResponse:
    """
    Answer a question using the RAG pipeline.
    
    Args:
        question: User's question
        return_sources: Whether to include sources in response
        
    Returns:
        QAResponse with answer and optional sources
    """
    try:
        # Load vector store
        vector_store = load_vector_store()
        
        # Retrieve context
        context_chunks = retrieve_context(vector_store, question)
        
        # Handle empty retrieval
        if not context_chunks:
            return QAResponse(
                answer="I don't know. No relevant information was found in the documents.",
                sources=[] if return_sources else None
            )
        
        # Generate answer
        answer = generate_answer(question, context_chunks)
        
        # Extract sources if requested
        sources = extract_sources(context_chunks) if return_sources else None
        
        return QAResponse(answer=answer, sources=sources)
        
    except FileNotFoundError as e:
        return QAResponse(
            answer=str(e),
            sources=None
        )
    except Exception as e:
        return QAResponse(
            answer=f"An error occurred: {str(e)}",
            sources=None
        )
