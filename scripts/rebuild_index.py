"""Script to rebuild the vector store index."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingest import ingest_documents


if __name__ == "__main__":
    ingest_documents()
