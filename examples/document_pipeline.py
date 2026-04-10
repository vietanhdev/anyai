"""Extract, summarize, and analyze text from document images.

Usage: python document_pipeline.py document.jpg

Pipeline: OCR -> Summarize -> Extract entities -> Sentiment
Install:  pip install anyai[ocr,nlp]
"""
import sys

from anyai import pipeline


def ocr_step(image_path: str) -> str:
    """Extract text from an image using anyocr."""
    import anyocr
    result = anyocr.read(image_path)
    print(f"  [OCR] {len(result.lines)} lines, confidence {result.confidence:.0%}")
    return result.text


def summarize_step(text: str) -> str:
    """Summarize extracted text using anynlp."""
    import anynlp
    summary = anynlp.summarize(text, ratio=0.3)
    print(f"  [Summary] {len(summary.text)} chars")
    return summary.text


def sentiment_step(text: str) -> dict:
    """Analyze sentiment and return combined result."""
    import anynlp
    sent = anynlp.sentiment(text)
    print(f"  [Sentiment] {sent.label} ({sent.score:.2f})")
    return {"summary": text, "sentiment": sent.label, "score": sent.score}


# Build the pipeline: OCR -> Summarize -> Sentiment
doc_analyzer = pipeline(
    ("ocr", ocr_step),
    ("summarize", summarize_step),
    ("sentiment", sentiment_step),
)


if __name__ == "__main__":
    image = sys.argv[1] if len(sys.argv) > 1 else "document.jpg"
    print(f"Analyzing: {image}")
    print(f"Pipeline: {doc_analyzer}")
    print()
    try:
        result = doc_analyzer(image)
        print(f"\nResult: {result}")
    except Exception as e:
        print(f"\nDemo mode (install deps to run): {e}")
        print("Install: pip install anyai[ocr,nlp]")
