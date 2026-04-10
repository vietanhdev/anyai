"""Tests for anyai proxy functions and pipeline API."""

import types
import pytest
from unittest.mock import MagicMock, patch

import anyai
from anyai.core import _make_proxy
from anyai.pipeline import Pipeline, pipeline


class TestProxyFunctions:
    """Tests for lazy proxy functions."""

    def test_proxy_raises_import_error_when_missing(self):
        """Proxy should raise ImportError with install hint when sub-package is absent."""
        proxy = _make_proxy("detect", "cv", "anycv", "detect")
        with patch("importlib.import_module", side_effect=ImportError("No module")):
            with pytest.raises(ImportError, match=r"pip install anyai\[cv\]"):
                proxy("image.jpg")

    def test_proxy_error_mentions_function_name(self):
        """Error message should include the proxy function name."""
        proxy = _make_proxy("ocr", "ocr", "anyocr", "read")
        with patch("importlib.import_module", side_effect=ImportError("No module")):
            with pytest.raises(ImportError, match=r"anyai\.ocr\(\)"):
                proxy("image.jpg")

    def test_proxy_delegates_when_installed(self):
        """Proxy should call the target function when the package is importable."""
        fake_mod = types.ModuleType("anycv")
        fake_mod.detect = MagicMock(return_value=["cat", "dog"])

        with patch("importlib.import_module", return_value=fake_mod) as mock_import:
            result = anyai.detect("image.jpg", conf=0.5)

        mock_import.assert_called_once_with("anycv")
        fake_mod.detect.assert_called_once_with("image.jpg", conf=0.5)
        assert result == ["cat", "dog"]

    def test_proxy_attribute_error(self):
        """Proxy should raise AttributeError if target attr is missing from module."""
        fake_mod = types.ModuleType("anycv")
        # No 'detect' attribute on the module.

        with patch("importlib.import_module", return_value=fake_mod):
            with pytest.raises(AttributeError, match="detect"):
                anyai.detect("image.jpg")

    @pytest.mark.parametrize("name,extra,package,attr", [
        ("detect", "cv", "anycv", "detect"),
        ("ocr", "ocr", "anyocr", "read"),
        ("chat", "llm", "anyllm", "chat"),
        ("classify", "ml", "anyml", "classify"),
        ("profile", "table", "tableai", "profile"),
        ("summarize", "nlp", "anynlp", "summarize"),
        ("deploy", "deploy", "anydeploy", "export"),
    ])
    def test_all_proxies_give_clear_errors(self, name, extra, package, attr):
        """Every registered proxy should give a clear install hint on ImportError."""
        proxy = _make_proxy(name, extra, package, attr)
        with patch("importlib.import_module", side_effect=ImportError("No module")):
            with pytest.raises(ImportError) as exc_info:
                proxy()
            msg = str(exc_info.value)
            assert f"pip install anyai[{extra}]" in msg
            assert f"anyai.{name}()" in msg

    def test_proxies_are_accessible_from_anyai_module(self):
        """All proxy names should be importable from anyai."""
        for name in ("detect", "ocr", "chat", "classify", "profile", "summarize", "deploy"):
            assert hasattr(anyai, name), f"anyai.{name} not found"
            assert callable(getattr(anyai, name))

    def test_proxies_in_all(self):
        """Proxy names should appear in anyai.__all__."""
        for name in ("detect", "ocr", "chat", "classify", "profile", "summarize", "deploy"):
            assert name in anyai.__all__


class TestPipeline:
    """Tests for anyai.pipeline() chaining."""

    def test_single_step(self):
        """Pipeline with one step passes through args and kwargs."""
        step = MagicMock(return_value="out")
        pipe = pipeline([step])
        result = pipe("a", key="b")
        step.assert_called_once_with("a", key="b")
        assert result == "out"

    def test_two_steps_chained(self):
        """Output of step 1 becomes input to step 2."""
        step1 = MagicMock(return_value="intermediate")
        step2 = MagicMock(return_value="final")
        pipe = pipeline([step1, step2])
        result = pipe("input")
        step1.assert_called_once_with("input")
        step2.assert_called_once_with("intermediate")
        assert result == "final"

    def test_three_steps_chained(self):
        """Pipeline chains three steps sequentially."""
        pipe = pipeline([
            lambda x: x + 1,
            lambda x: x * 2,
            lambda x: x - 3,
        ])
        assert pipe(5) == 9  # (5+1)*2 - 3

    def test_pipeline_empty_raises(self):
        """pipeline() with no steps should raise ValueError."""
        with pytest.raises(ValueError, match="at least one step"):
            pipeline([])

    def test_pipeline_with_proxy_functions(self):
        """Pipeline should work with proxy functions."""
        fake_ocr = types.ModuleType("anyocr")
        fake_ocr.read = MagicMock(return_value="extracted text")
        fake_nlp = types.ModuleType("anynlp")
        fake_nlp.summarize = MagicMock(return_value="summary")

        def mock_import(name):
            return {"anyocr": fake_ocr, "anynlp": fake_nlp}[name]

        with patch("importlib.import_module", side_effect=mock_import):
            pipe = pipeline([anyai.ocr, anyai.summarize])
            result = pipe("document.jpg")

        fake_ocr.read.assert_called_once_with("document.jpg")
        fake_nlp.summarize.assert_called_once_with("extracted text")
        assert result == "summary"

    def test_pipeline_in_anyai_all(self):
        """pipeline should be in anyai.__all__."""
        assert "pipeline" in anyai.__all__
