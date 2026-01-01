"""CLI interface for the Local Documents QA system."""
import sys
from app.query import answer_question


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python -m app.main \"Your question here\" [--sources]")
        print("\nOptions:")
        print("  --sources    Include source citations in the response")
        print("\nExample:")
        print('  python -m app.main "What is the grading policy?"')
        print('  python -m app.main "What is the grading policy?" --sources')
        sys.exit(1)
    
    # Parse arguments
    question = sys.argv[1]
    return_sources = "--sources" in sys.argv
    
    # Get answer
    print(f"\nQuestion: {question}")
    print("\nProcessing...\n")
    
    response = answer_question(question, return_sources=return_sources)
    
    # Display results
    print("=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(response.answer)
    
    if response.sources:
        print("\n" + "=" * 60)
        print("SOURCES:")
        print("=" * 60)
        for i, source in enumerate(response.sources, 1):
            print(f"{i}. {source}")
    
    print("\n")


if __name__ == "__main__":
    main()
