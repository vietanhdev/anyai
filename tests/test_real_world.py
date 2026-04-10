"""Real-world integration tests using actual text data."""
import anyai


def test_text_summarize_real():
    text = """Machine learning is a branch of artificial intelligence that focuses on building
    systems that learn from data. Unlike traditional programming where rules are explicitly
    coded, ML systems identify patterns in data to make predictions or decisions. Common
    approaches include supervised learning, unsupervised learning, and reinforcement learning.
    Deep learning, a subset of ML, uses neural networks with many layers to process complex
    patterns in large datasets."""
    result = anyai.text.summarize(text)
    assert len(result) > 0
    assert len(result) < len(text)


def test_text_sentiment_real():
    assert anyai.text.sentiment("I love this amazing product!")["label"] == "positive"
    assert anyai.text.sentiment("This is terrible and broken")["label"] == "negative"


def test_text_keywords_real():
    text = "Python is a programming language used for machine learning and data science"
    keywords = anyai.text.extract_keywords(text)
    assert len(keywords) > 0


def test_pipeline_real():
    from anyai import pipeline

    chain = pipeline(
        ("upper", lambda x: x.upper()),
        ("split", lambda x: x.split()),
    )
    result = chain("hello world")
    assert result == ["HELLO", "WORLD"]


def test_about():
    info = anyai.about()
    assert "AnyAI" in info
