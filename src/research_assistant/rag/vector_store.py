from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from research_assistant.rag.document_loader import load_papers
from research_assistant.rag.embeddings import get_embeddings


PROJECT_ROOT = Path(__file__).resolve().parents[3]
INDEX_DIR = PROJECT_ROOT / "vector_store" / "faiss_index"


def build_index() -> None:
    """Load papers, split text, embed chunks, and save FAISS."""

    documents = load_papers()

    print(f"\nTotal pages loaded: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(documents)

    # Remove chunks containing only whitespace.
    chunks = [
        chunk
        for chunk in chunks
        if chunk.page_content.strip()
    ]

    if not chunks:
        raise ValueError(
            "No extractable text found in the PDFs."
        )

    print(f"Total chunks created: {len(chunks)}")

    embeddings = get_embeddings()

    print("\nGenerating embeddings and building FAISS index...")

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    INDEX_DIR.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_store.save_local(str(INDEX_DIR))

    print(f"\nFAISS index saved to: {INDEX_DIR}")
    print("Indexing completed successfully!")


def load_vector_store() -> FAISS:
    """Load the existing FAISS index."""

    if not (INDEX_DIR / "index.faiss").exists():
        raise FileNotFoundError(
            f"FAISS index not found at {INDEX_DIR}. "
            "Run scripts/build_index.py first."
        )

    embeddings = get_embeddings()

    vector_store = FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return vector_store