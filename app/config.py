"""Configuration management for the QA system."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_docs"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

# Ensure directories exist
RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set in .env file")

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4-turbo-preview")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Vision model for multi-modal support
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o")
USE_VISION = os.getenv("USE_VISION", "true").lower() == "true"

# Image-only page handling
DESCRIBE_IMAGE_PAGES = os.getenv("DESCRIBE_IMAGE_PAGES", "true").lower() == "true"
MIN_TEXT_LENGTH = int(os.getenv("MIN_TEXT_LENGTH", "50"))

# Retrieval Configuration
TOP_K = int(os.getenv("TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# LLM Configuration
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

# Vector store settings
VECTOR_STORE_INDEX_NAME = "faiss_index"
