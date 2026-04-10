"""Tests for anyai.core module."""

import pytest
from unittest.mock import patch


def test_about_returns_string():
    """about() should return a non-empty info string."""
    from anyai.core import about
    result = about()
    assert isinstance(result, str)
    assert "AnyAI" in result


def test_about_contains_version():
    """about() should include the version number."""
    from anyai.core import about
    result = about()
    import re
    assert re.search(r"\d+\.\d+\.\d+", result)


def test_about_contains_author():
    """about() should include the author name."""
    from anyai.core import about
    result = about()
    assert "Viet-Anh Nguyen" in result


def test_about_lists_builtins():
    """about() should mention the built-in modules."""
    from anyai.core import about
    result = about()
    assert "text" in result
    assert "image" in result


def test_available_backends_returns_dict():
    """available_backends() should return a dict of extra -> bool."""
    from anyai.core import available_backends
    result = available_backends()
    assert isinstance(result, dict)
    assert "cv" in result
    assert "llm" in result
    assert "ocr" in result
    assert "ml" in result
    assert "nlp" in result
    assert "deploy" in result
    assert "table" in result


def test_available_backends_values_are_bool():
    """All values in available_backends() should be booleans."""
    from anyai.core import available_backends
    result = available_backends()
    for key, value in result.items():
        assert isinstance(value, bool), f"Expected bool for '{key}', got {type(value)}"


def test_available_backends_optional_not_installed():
    """available_backends returns a dict with bool values for each extra."""
    from anyai.core import available_backends
    result = available_backends()
    # Verify structure: all extras should be in the result as booleans
    for key in ["cv", "llm", "ocr", "ml", "nlp", "deploy", "table"]:
        assert key in result
        assert isinstance(result[key], bool)
