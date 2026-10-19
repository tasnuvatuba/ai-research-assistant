from pathlib import Path

from research_assistant.rag.retriever import PaperRetriever


def main():
    print("Initializing research paper retriever...")

    # Load the embedding model and FAISS index once.
    retriever = PaperRetriever()

    print("Retriever ready!")

    while True:
        question = input(
            "\nEnter a question (or type 'exit'): "
        ).strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            documents = retriever.retrieve_documents(
                question,
                k=4,
            )

            print(f"\nQuestion: {question}")
            print(f"Retrieved passages: {len(documents)}")

            for rank, document in enumerate(
                documents,
                start=1,
            ):
                source = document.metadata.get(
                    "source",
                    "Unknown source",
                )

                page = document.metadata.get("page")

                # PyMuPDFLoader normally uses zero-based pages.
                if page is not None:
                    page = int(page) + 1

                print("\n" + "=" * 70)
                print(f"Result {rank}")
                print(f"Source: {Path(source).name}")
                print(
                    f"Page: "
                    f"{page if page is not None else 'Unknown'}"
                )
                print("-" * 70)
                print(document.page_content)

        except Exception as error:
            print(f"An error occurred: {error}")


if __name__ == "__main__":
    main()