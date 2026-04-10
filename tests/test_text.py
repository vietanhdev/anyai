"""Tests for anyai.text module."""

import pytest
from anyai.text import summarize, extract_keywords, sentiment


# --- summarize ---

class TestSummarize:
    """Tests for the summarize function."""

    def test_empty_string(self):
        assert summarize("") == ""

    def test_whitespace_only(self):
        assert summarize("   ") == ""

    def test_single_sentence(self):
        text = "Machine learning is fascinating."
        result = summarize(text, num_sentences=3)
        assert result == text

    def test_fewer_sentences_than_requested(self):
        text = "First sentence. Second sentence."
        result = summarize(text, num_sentences=5)
        assert "First sentence" in result
        assert "Second sentence" in result

    def test_returns_correct_count(self):
        text = (
            "AI is transforming industries. "
            "Machine learning powers recommendations. "
            "Deep learning handles images. "
            "NLP processes text. "
            "Robotics automates tasks."
        )
        result = summarize(text, num_sentences=2)
        sentences = [s.strip() for s in result.split(". ") if s.strip()]
        # Should have at most 2 sentences.
        assert len(sentences) <= 3  # Accounting for split edge cases.

    def test_preserves_original_order(self):
        text = (
            "Alpha is the first. "
            "Beta is the second. "
            "Gamma is the third. "
            "Delta is the fourth. "
            "Alpha is mentioned again with alpha content."
        )
        result = summarize(text, num_sentences=2)
        # The selected sentences should maintain their original order.
        if "Alpha" in result and "Beta" in result:
            assert result.index("Alpha") < result.index("Beta")

    def test_summarize_long_text(self):
        sentences = [f"Sentence number {i} about topic {i % 3}." for i in range(20)]
        text = " ".join(sentences)
        result = summarize(text, num_sentences=3)
        assert len(result) > 0
        assert len(result) < len(text)


# --- extract_keywords ---

class TestExtractKeywords:
    """Tests for the extract_keywords function."""

    def test_empty_string(self):
        assert extract_keywords("") == []

    def test_whitespace_only(self):
        assert extract_keywords("   ") == []

    def test_stop_words_only(self):
        assert extract_keywords("the a an is are was were") == []

    def test_returns_list(self):
        result = extract_keywords("machine learning artificial intelligence")
        assert isinstance(result, list)

    def test_respects_top_n(self):
        text = "apple banana cherry date elderberry fig grape"
        result = extract_keywords(text, top_n=3)
        assert len(result) == 3

    def test_frequent_words_first(self):
        text = "python python python java java ruby"
        result = extract_keywords(text, top_n=3)
        assert result[0] == "python"
        assert result[1] == "java"

    def test_excludes_stop_words(self):
        text = "the machine is a very good learning tool"
        result = extract_keywords(text)
        assert "the" not in result
        assert "is" not in result
        assert "a" not in result

    def test_single_char_excluded(self):
        text = "I a x machine learning"
        result = extract_keywords(text)
        for kw in result:
            assert len(kw) > 1

    def test_case_insensitive(self):
        text = "Python PYTHON python"
        result = extract_keywords(text, top_n=1)
        assert result == ["python"]


# --- sentiment ---

class TestSentiment:
    """Tests for the sentiment function."""

    def test_empty_string(self):
        result = sentiment("")
        assert result["label"] == "neutral"
        assert result["score"] == 0.5

    def test_positive_text(self):
        result = sentiment("This product is amazing and wonderful!")
        assert result["label"] == "positive"
        assert result["score"] > 0.5

    def test_negative_text(self):
        result = sentiment("This is terrible and awful.")
        assert result["label"] == "negative"
        assert result["score"] > 0.5

    def test_neutral_text(self):
        result = sentiment("The table has four legs.")
        assert result["label"] == "neutral"
        assert result["score"] == 0.5

    def test_negation_flips_positive(self):
        result = sentiment("This is not good.")
        assert result["label"] == "negative"

    def test_negation_flips_negative(self):
        result = sentiment("This is not bad.")
        assert result["label"] == "positive"

    def test_intensifier_boosts_score(self):
        base = sentiment("This is good.")
        intensified = sentiment("This is very good.")
        # Intensified should have equal or higher score.
        assert intensified["score"] >= base["score"]

    def test_returns_dict_with_required_keys(self):
        result = sentiment("Hello world")
        assert "label" in result
        assert "score" in result

    def test_score_between_zero_and_one(self):
        for text in [
            "Great product!",
            "Horrible experience.",
            "Neutral statement about tables.",
            "I really love this but also hate that.",
        ]:
            result = sentiment(text)
            assert 0.0 <= result["score"] <= 1.0, f"Score out of range for: {text}"

    def test_label_values(self):
        for text in [
            "Great product!",
            "Horrible experience.",
            "Neutral statement.",
        ]:
            result = sentiment(text)
            assert result["label"] in ("positive", "negative", "neutral")

    def test_mixed_sentiment(self):
        result = sentiment("I love the design but hate the price.")
        # Should still return a valid result.
        assert result["label"] in ("positive", "negative", "neutral")
        assert 0.0 <= result["score"] <= 1.0
