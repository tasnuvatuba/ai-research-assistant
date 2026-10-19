from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PAPER_DIR = PROJECT_ROOT / "data" / "papers"


def load_papers() -> list[Document]:
    """Load all PDFs and preserve source and page metadata."""

    pdf_files = sorted(PAPER_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {PAPER_DIR}"
        )

    documents = []

    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path.name}")

        loader = PyMuPDFLoader(str(pdf_path))
        pages = loader.load()

        # Store paths relative to the project root.
        for page in pages:
            page.metadata["source"] = str(
                pdf_path.relative_to(PROJECT_ROOT)
            )

        documents.extend(pages)

        print(f"  Pages loaded: {len(pages)}")

    return documents