from pathlib import Path

from research_assistant.application.rag_pipeline import RAGPipeline


def main():
    print("Initializing RAG pipeline...")
    pipeline = RAGPipeline(top_k=4)
    print("RAG pipeline ready!")

    while True:
        question = input("\nAsk a research question (or type 'exit'): ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            result = pipeline.answer_question(question)

            print("\n" + "=" * 70)
            print("ANSWER")
            print("=" * 70)
            print(result["answer"])

            print("\n" + "=" * 70)
            print("SOURCES")
            print("=" * 70)

            if not result["sources"]:
                print("No sources found.")
            else:
                for source in result["sources"]:
                    filename = Path(source["source"]).name
                    page = source["page"]

                    if page is not None:
                        print(f"- {filename}, page {page}")
                    else:
                        print(f"- {filename}")

        except Exception as error:
            print(f"An error occurred: {error}")


if __name__ == "__main__":
    main()