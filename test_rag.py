from pathlib import Path
import argparse

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Project paths
PAPER_DIR = Path("data/papers")
INDEX_DIR = Path("vector_store/faiss_index")

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    """Initialize the embedding model."""

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


def load_papers():
    """Load all PDFs and preserve their page metadata."""

    pdf_files = sorted(PAPER_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            "No PDF files found in data/papers/"
        )

    documents = []

    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path.name}")

        loader = PyMuPDFLoader(str(pdf_path))
        pages = loader.load()

        # Store a relative source path for each page.
        for page in pages:
            page.metadata["source"] = str(pdf_path)

        documents.extend(pages)

        print(f"  Pages loaded: {len(pages)}")

    return documents


def build_index():
    """Load papers, split text, embed chunks, and save FAISS."""

    documents = load_papers()

    print(f"\nTotal pages loaded: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(documents)

    # Remove chunks that contain only whitespace.
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


def search_papers(question, k=4):
    """Retrieve the most relevant passages for a question."""

    if not INDEX_DIR.exists():
        raise FileNotFoundError(
            "FAISS index not found. Run build first."
        )

    embeddings = get_embeddings()

    # Only load an index that you created and trust.
    vector_store = FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    results = vector_store.similarity_search(
        question,
        k=k,
    )

    print(f"\nQuestion: {question}")
    print(f"Retrieved passages: {len(results)}")

    for rank, document in enumerate(results, start=1):
        source = document.metadata.get(
            "source", "Unknown source"
        )

        # PyMuPDFLoader normally uses zero-based page metadata.
        page = document.metadata.get("page")

        if page is not None:
            page = int(page) + 1

        print("\n" + "=" * 70)
        print(f"Result {rank}")
        print(f"Source: {source}")
        print(f"Page: {page if page is not None else 'Unknown'}")
        print("-" * 70)
        print(document.page_content)


def main():
    parser = argparse.ArgumentParser(
        description="Research Paper RAG"
    )

    parser.add_argument(
        "command",
        choices=["build", "search"],
        help="Build the FAISS index or search it",
    )

    parser.add_argument(
        "--question",
        type=str,
        help="Question to search for",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of passages to retrieve",
    )

    args = parser.parse_args()

    if args.command == "build":
        build_index()

    elif args.command == "search":
        if not args.question:
            parser.error(
                "The search command requires --question"
            )

        if args.k < 1:
            parser.error("--k must be at least 1")

        search_papers(
            question=args.question,
            k=args.k,
        )


if __name__ == "__main__":
    main()