"""Quick start -- demonstrate the unified AnyAI API.

AnyAI is the meta-package that proxies to specialised packages
(anycv, anyocr, anyllm, etc.) so you only need one import.

Requirements:
    pip install anyai[cv,ocr,llm]
"""

import anyai

# -- Check which optional backends are installed --
print("Installed backends:", anyai.available_backends())
print("Version info:", anyai.version_info())

# -- Computer vision (requires anyai[cv]) --
# Detect objects in an image with a single call
# detections = anyai.detect("photo.jpg")
# for det in detections:
#     print(f"  {det.label}: {det.confidence:.0%}")

# -- OCR (requires anyai[ocr]) --
# Read text from a document image
# result = anyai.ocr("receipt.png")
# print(result)

# -- LLM chat (requires anyai[llm]) --
# Chat with any model via a unified API
# response = anyai.chat("Summarise the benefits of open source in one sentence.")
# print(response)

# -- Pipelines: chain steps together --
pipeline = anyai.pipeline(
    [
        ("greet", lambda name: f"Hello, {name}!"),
        ("upper", lambda text: text.upper()),
    ]
)
print(pipeline("world"))  # -> HELLO, WORLD!
