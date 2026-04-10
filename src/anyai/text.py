"""Basic text AI utilities with no heavy ML dependencies.

Provides extractive summarization, keyword extraction, and rule-based
sentiment analysis using only the Python standard library.
"""

import math
import re
import string
from collections import Counter
from typing import Dict, List, Optional, Tuple

# Common English stop words for filtering.
_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "ought", "used", "it", "its", "this", "that", "these", "those", "i",
    "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "they", "them", "their", "theirs",
    "themselves", "what", "which", "who", "whom", "when", "where", "why",
    "how", "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "just", "because", "as", "until", "while",
    "about", "between", "through", "during", "before", "after", "above",
    "below", "up", "down", "out", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "also", "if",
})

# Simple sentiment lexicon: word -> score in [-1, 1].
_POSITIVE_WORDS = frozenset({
    "good", "great", "excellent", "amazing", "wonderful", "fantastic",
    "awesome", "outstanding", "superb", "brilliant", "love", "loved",
    "loving", "happy", "glad", "pleased", "delighted", "enjoy", "enjoyed",
    "enjoying", "beautiful", "perfect", "best", "better", "nice",
    "impressive", "remarkable", "exceptional", "magnificent", "terrific",
    "marvelous", "positive", "recommend", "recommended", "like", "liked",
    "helpful", "pleasant", "satisfied", "exciting", "excited",
})

_NEGATIVE_WORDS = frozenset({
    "bad", "terrible", "horrible", "awful", "worst", "worse", "poor",
    "disappointing", "disappointed", "hate", "hated", "hating", "ugly",
    "boring", "bored", "sad", "unhappy", "angry", "annoyed", "annoying",
    "frustrating", "frustrated", "useless", "waste", "broken", "fail",
    "failed", "failing", "failure", "problem", "problems", "wrong",
    "negative", "dislike", "disliked", "unpleasant", "painful", "dreadful",
    "miserable", "pathetic", "mediocre", "inferior",
})

_NEGATION_WORDS = frozenset({
    "not", "no", "never", "neither", "nobody", "nothing", "nowhere",
    "nor", "cannot", "can't", "won't", "don't", "doesn't", "didn't",
    "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't",
    "hadn't", "wouldn't", "couldn't", "shouldn't",
})

_INTENSIFIERS = frozenset({
    "very", "really", "extremely", "absolutely", "incredibly", "highly",
    "truly", "completely", "totally", "utterly", "thoroughly",
})


def _tokenize(text: str) -> List[str]:
    """Split text into lowercase word tokens, stripping punctuation.

    Args:
        text: Input text string.

    Returns:
        A list of lowercase word tokens.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s'-]", " ", text)
    tokens = text.split()
    tokens = [t.strip(string.punctuation) for t in tokens]
    return [t for t in tokens if t]


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using basic punctuation rules.

    Args:
        text: Input text string.

    Returns:
        A list of sentence strings.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def summarize(text: str, num_sentences: int = 3) -> str:
    """Produce an extractive summary by selecting top-scoring sentences.

    Sentences are scored by the sum of their non-stop-word term frequencies,
    normalized by sentence length. The top ``num_sentences`` are returned
    in their original order.

    Args:
        text: The input text to summarize.
        num_sentences: Maximum number of sentences in the summary.
            Defaults to 3.

    Returns:
        A string containing the selected summary sentences joined by spaces.
        Returns the original text if it has fewer sentences than requested.

    Examples:
        >>> summarize("AI is great. It helps people. The weather is nice.", 2)
        'AI is great. It helps people.'
    """
    if not text or not text.strip():
        return ""

    sentences = _split_sentences(text)
    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    # Build word frequency map (excluding stop words).
    all_tokens = _tokenize(text)
    word_freq: Counter = Counter()
    for token in all_tokens:
        if token not in _STOP_WORDS and len(token) > 1:
            word_freq[token] += 1

    # Score each sentence.
    scored: List[Tuple[int, float, str]] = []
    for idx, sentence in enumerate(sentences):
        tokens = _tokenize(sentence)
        content_tokens = [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]
        if not content_tokens:
            scored.append((idx, 0.0, sentence))
            continue
        score = sum(word_freq.get(t, 0) for t in content_tokens)
        # Normalize by length to avoid bias toward long sentences.
        score /= len(content_tokens)
        scored.append((idx, score, sentence))

    # Select top sentences and return in original order.
    scored.sort(key=lambda x: x[1], reverse=True)
    selected = sorted(scored[:num_sentences], key=lambda x: x[0])
    return " ".join(s[2] for s in selected)


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """Extract keywords using TF-like scoring.

    Words are scored by their frequency after removing stop words and
    single-character tokens. The top ``top_n`` words are returned.

    Args:
        text: The input text.
        top_n: Number of keywords to return. Defaults to 10.

    Returns:
        A list of keyword strings ordered by descending score.

    Examples:
        >>> extract_keywords("machine learning is a type of machine intelligence", 3)
        ['machine', 'learning', 'type']
    """
    if not text or not text.strip():
        return []

    tokens = _tokenize(text)
    # Filter stop words and short tokens.
    content_tokens = [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]

    if not content_tokens:
        return []

    freq = Counter(content_tokens)
    # Return top_n keywords sorted by frequency then alphabetically for ties.
    keywords = sorted(freq.keys(), key=lambda w: (-freq[w], w))
    return keywords[:top_n]


def sentiment(text: str) -> Dict[str, object]:
    """Perform rule-based sentiment analysis.

    Uses a simple lexicon approach with negation handling and intensifier
    detection. Returns a label (``'positive'``, ``'negative'``, or
    ``'neutral'``) and a confidence score between 0 and 1.

    Args:
        text: The input text to analyze.

    Returns:
        A dictionary with keys:
        - ``'label'``: one of ``'positive'``, ``'negative'``, ``'neutral'``
        - ``'score'``: float confidence between 0.0 and 1.0

    Examples:
        >>> sentiment("This product is absolutely wonderful!")
        {'label': 'positive', 'score': ...}
        >>> sentiment("This is terrible and broken.")
        {'label': 'negative', 'score': ...}
    """
    if not text or not text.strip():
        return {"label": "neutral", "score": 0.5}

    tokens = _tokenize(text)
    if not tokens:
        return {"label": "neutral", "score": 0.5}

    positive_score = 0.0
    negative_score = 0.0
    token_count = len(tokens)

    for i, token in enumerate(tokens):
        multiplier = 1.0

        # Check for preceding intensifier.
        if i > 0 and tokens[i - 1] in _INTENSIFIERS:
            multiplier = 1.5

        # Check for preceding negation (within 3 words).
        negated = False
        lookback = max(0, i - 3)
        for j in range(lookback, i):
            if tokens[j] in _NEGATION_WORDS:
                negated = True
                break

        if token in _POSITIVE_WORDS:
            if negated:
                negative_score += multiplier
            else:
                positive_score += multiplier
        elif token in _NEGATIVE_WORDS:
            if negated:
                positive_score += multiplier
            else:
                negative_score += multiplier

    total = positive_score + negative_score
    if total == 0:
        return {"label": "neutral", "score": 0.5}

    if positive_score > negative_score:
        raw = positive_score / total
        score = 0.5 + raw * 0.5  # Map to [0.5, 1.0]
        return {"label": "positive", "score": round(score, 2)}
    elif negative_score > positive_score:
        raw = negative_score / total
        score = 0.5 + raw * 0.5
        return {"label": "negative", "score": round(score, 2)}
    else:
        return {"label": "neutral", "score": 0.5}
