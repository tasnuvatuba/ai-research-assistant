from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


PAPER_DIR = Path("data/papers")

pdf_files = list(PAPER_DIR.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError(
        "No PDF files found in data/papers/"
    )

# Load the first PDF
pdf_path = pdf_files[0]

loader = PyMuPDFLoader(str(pdf_path))
documents = loader.load()

print(f"Paper: {pdf_path.name}")
print(f"Pages loaded: {len(documents)}")

# Split the extracted text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
)

chunks = splitter.split_documents(documents)

print(f"Chunks created: {len(chunks)}")

# Inspect the first chunk
first_chunk = chunks[0]

print("\nFirst chunk:")
print(first_chunk.page_content[:800])

print("\nMetadata:")
print(first_chunk.metadata)

# Inspect the second chunk
second_chunk = chunks[1]

print("\nSecond chunk:")
print(second_chunk.page_content[:500])

print("\nMetadata:")
print(second_chunk.metadata)