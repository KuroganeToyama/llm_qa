"""Prompt templates for the QA system."""

SYSTEM_PROMPT = """You are a helpful assistant.
Answer ONLY using the provided context (text and images).
If the answer cannot be found in the context, say "I don't know."
Do not make up information or use external knowledge.
When images are provided, analyze them carefully and incorporate relevant visual information in your answer."""


def get_user_prompt_with_images(context: str, question: str, has_images: bool = False) -> str:
    """
    Generate the user prompt with context and question (with optional image awareness).
    
    Args:
        context: The retrieved context chunks
        question: The user's question
        has_images: Whether images are included in the context
        
    Returns:
        Formatted prompt string
    """
    image_note = "\n\nNote: Images from the relevant pages are included below. Please analyze them along with the text." if has_images else ""
    
    return f"""Context:
{context}{image_note}

Question:
{question}

Answer:"""


def format_context(chunks: list) -> str:
    """
    Format retrieved chunks into a context string.
    
    Args:
        chunks: List of retrieved document chunks
        
    Returns:
        Formatted context string
    """
    if not chunks:
        return "No relevant context found."
    
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        content = chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
        source = chunk.metadata.get('source', 'unknown') if hasattr(chunk, 'metadata') else 'unknown'
        page = chunk.metadata.get('page', '') if hasattr(chunk, 'metadata') else ''
        images = chunk.metadata.get('images', []) if hasattr(chunk, 'metadata') else []
        
        source_info = f"{source}"
        if page:
            source_info += f" (page {page})"
        
        # Add note if page image is present
        if images:
            source_info += f" [Page image available]"
            
        context_parts.append(f"[{i}] Source: {source_info}\n{content}")
    
    return "\n\n".join(context_parts)
