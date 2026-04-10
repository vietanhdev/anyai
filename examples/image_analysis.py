"""CV pipeline: detect objects, classify, describe with VLM.

Usage: python image_analysis.py photo.jpg

Install: pip install anyai[cv]
"""
import sys

from anyai import pipeline


def detect_step(image_path: str) -> dict:
    """Detect objects in the image."""
    import anycv
    detections = anycv.detect(image_path)
    labels = [f"{d.label} ({d.confidence:.0%})" for d in detections]
    print(f"  [Detect] Found: {', '.join(labels) or 'nothing'}")
    return {"image": image_path, "detections": detections}


def classify_step(data: dict) -> dict:
    """Classify the overall image content."""
    import anycv
    classes = anycv.classify(data["image"], top_k=3)
    data["classes"] = classes
    top = classes[0] if classes else None
    print(f"  [Classify] Top: {top.label} ({top.confidence:.0%})" if top else "")
    return data


def describe_step(data: dict) -> dict:
    """Describe the image using a vision-language model."""
    import anycv
    desc = anycv.describe(data["image"])
    data["description"] = desc.text
    print(f"  [VLM] {desc.text[:80]}...")
    return data


# Build the CV pipeline
image_pipeline = pipeline(
    ("detect", detect_step),
    ("classify", classify_step),
    ("describe", describe_step),
)

if __name__ == "__main__":
    image = sys.argv[1] if len(sys.argv) > 1 else "photo.jpg"
    print(f"Analyzing: {image}")
    print(f"Pipeline: {image_pipeline}")
    print()
    try:
        result = image_pipeline(image)
        print(f"\nDetections: {len(result['detections'])}")
        print(f"Top class: {result['classes'][0].label}")
        print(f"Description: {result['description']}")
    except Exception as e:
        print(f"\nDemo mode (install deps to run): {e}")
        print("Install: pip install anyai[cv]")
