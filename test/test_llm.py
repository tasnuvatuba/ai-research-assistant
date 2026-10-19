from research_assistant.llm.local_llm import LocalLLM


def main():
    # Load the model once.
    llm = LocalLLM()

    while True:
        question = input(
            "\nAsk a question (or type 'exit'): "
        ).strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue

        answer = llm.generate_response(question)

        print("\nModel response:")
        print(answer)


if __name__ == "__main__":
    main()