"""Local-first chatbot using Ollama models via anyllm.

Usage: python local_chat.py

Requires a running Ollama server with a model pulled:
    ollama pull llama3
    python local_chat.py

Install: pip install anyai[llm]
"""
import sys

import anyllm


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "ollama/llama3"
    print(f"Chat with {model} (type 'quit' to exit)")
    print("=" * 50)

    # Create a conversation to maintain context
    conv = anyllm.Conversation(model=model)
    conv.add("system", "You are a helpful assistant. Keep answers concise.")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        conv.add("user", user_input)
        try:
            response = conv.send()
            print(f"\nBot: {response.content}")
        except Exception as e:
            print(f"\nError: {e}")
            print("Make sure Ollama is running: ollama serve")
            break

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
