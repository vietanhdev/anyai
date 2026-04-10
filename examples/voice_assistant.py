"""Voice AI assistant using anyrobo.

Usage: python voice_assistant.py

Requires: Ollama running locally, a microphone, and speakers.
Install:  pip install anyrobo[whisper]
"""

import anyrobo


def main():
    # Create an assistant with a custom personality
    assistant = anyrobo.Assistant(
        config=anyrobo.AssistantConfig(
            stt_backend="whisper",
            stt_model="base",
            tts_backend="pyttsx3",
            llm_backend="ollama",
            llm_model="llama3",
        ),
        personality=anyrobo.Personality(
            name="Aria",
            system_prompt=(
                "You are Aria, a friendly and helpful voice assistant. "
                "Keep responses short and conversational since they will "
                "be spoken aloud."
            ),
        ),
    )

    # Register custom tools the assistant can use
    @assistant.tool("get_time")
    def get_time() -> str:
        """Get the current time."""
        from datetime import datetime
        return datetime.now().strftime("%I:%M %p")

    @assistant.tool("get_date")
    def get_date() -> str:
        """Get today's date."""
        from datetime import datetime
        return datetime.now().strftime("%A, %B %d, %Y")

    print("Aria is ready! Say something (Ctrl+C to stop).")
    print("=" * 50)

    try:
        assistant.run()
    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install: pip install anyrobo[whisper]")
        print("Also need: ollama serve && ollama pull llama3")
