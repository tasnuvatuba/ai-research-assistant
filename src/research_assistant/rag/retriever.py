from langchain_core.documents import Document

from research_assistant.rag.vector_store import load_vector_store


class PaperRetriever:
    """Retrieve relevant passages from the research paper index."""

    def __init__(self):
        # Load the index once when the retriever is initialized.
        self.vector_store = load_vector_store()

    def retrieve_documents(
        self,
        question: str,
        k: int = 4,
    ) -> list[Document]:
        """Return the k most relevant passages."""

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if k < 1:
            raise ValueError("k must be at least 1.")

        return self.vector_store.similarity_search(
            question,
            k=k,
        )