"""Tests for the anyai MCP server."""

from unittest.mock import MagicMock, patch

import pytest

from anyai.mcp_server import _serialize, _try_import, create_server


class TestSerialize:
    def test_string_passthrough(self):
        assert _serialize("hello") == "hello"

    def test_dict_to_json(self):
        result = _serialize({"key": "value"})
        assert '"key"' in result
        assert '"value"' in result

    def test_list_to_json(self):
        result = _serialize([1, 2, 3])
        assert "[" in result
        assert "1" in result

    def test_non_serializable_falls_back_to_str(self):
        obj = object()
        result = _serialize(obj)
        assert isinstance(result, str)


class TestTryImport:
    def test_import_existing_module(self):
        mod = _try_import("json", "test")
        assert hasattr(mod, "dumps")

    def test_import_missing_module(self):
        with pytest.raises(RuntimeError, match="not installed"):
            _try_import("nonexistent_package_xyz", "test")


class TestCreateServer:
    def test_server_creation(self):
        server = create_server()
        assert server is not None
        assert server.name == "anyai"

    def test_server_has_tools(self):
        server = create_server()
        tools = server._tool_manager._tools
        tool_names = set(tools.keys())
        expected = {
            "detect_objects",
            "read_text",
            "chat",
            "summarize_text",
            "analyze_sentiment",
            "profile_data",
            "classify_data",
        }
        assert expected == tool_names

    @patch("anyai.mcp_server._try_import")
    def test_detect_objects_calls_anycv(self, mock_import):
        mock_mod = MagicMock()
        mock_mod.detect.return_value = [{"label": "cat", "confidence": 0.9}]
        mock_import.return_value = mock_mod

        server = create_server()
        tool_fn = server._tool_manager._tools["detect_objects"].fn
        result = tool_fn(image="test.jpg", model=None)

        mock_import.assert_called_with("anycv", "cv")
        mock_mod.detect.assert_called_once_with("test.jpg")
        assert "cat" in result

    @patch("anyai.mcp_server._try_import")
    def test_detect_objects_with_model(self, mock_import):
        mock_mod = MagicMock()
        mock_mod.detect.return_value = []
        mock_import.return_value = mock_mod

        server = create_server()
        tool_fn = server._tool_manager._tools["detect_objects"].fn
        tool_fn(image="test.jpg", model="yolov8n")

        mock_mod.detect.assert_called_once_with("test.jpg", model="yolov8n")

    @patch("anyai.mcp_server._try_import")
    def test_read_text_calls_anyocr(self, mock_import):
        mock_mod = MagicMock()
        mock_mod.read.return_value = "Hello world"
        mock_import.return_value = mock_mod

        server = create_server()
        tool_fn = server._tool_manager._tools["read_text"].fn
        result = tool_fn(document="doc.pdf", backend=None)

        mock_import.assert_called_with("anyocr", "ocr")
        mock_mod.read.assert_called_once_with("doc.pdf")
        assert result == "Hello world"

    @patch("anyai.mcp_server._try_import")
    def test_read_text_with_backend(self, mock_import):
        mock_mod = MagicMock()
        mock_mod.read.return_value = "text"
        mock_import.return_value = mock_mod

        server = create_server()
        tool_fn = server._tool_manager._tools["read_text"].fn
        tool_fn(document="doc.png", backend="surya")

        mock_mod.read.assert_called_once_with("doc.png", backend="surya")

    @patch("anyai.mcp_server._try_import")
    def test_chat_calls_anyllm(self, mock_import):
        mock_mod = MagicMock()
        mock_mod.chat.return_value = "Hi there!"
        mock_import.return_value = mock_mod

        server = create_server()
        tool_fn = server._tool_manager._tools["chat"].fn
        result = tool_fn(message="Hello", model=None)

        mock_import.assert_called_with("anyllm", "llm")
        mock_mod.chat.assert_called_once_with("Hello")
        assert result == "Hi there!"

    @patch("anyai.mcp_server._try_import")
    def test_summarize_text_calls_anynlp(self, mock_import):
        mock_mod = MagicMock()
        mock_mod.summarize.return_value = "Summary here."
        mock_import.return_value = mock_mod

        server = create_server()
        tool_fn = server._tool_manager._tools["summarize_text"].fn
        result = tool_fn(text="Long text...", num_sentences=2)

        mock_import.assert_called_with("anynlp", "nlp")
        mock_mod.summarize.assert_called_once_with("Long text...", num_sentences=2)
        assert result == "Summary here."

    @patch("anyai.mcp_server._try_import")
    def test_analyze_sentiment_calls_anynlp(self, mock_import):
        mock_mod = MagicMock()
        mock_mod.sentiment.return_value = {"label": "positive", "score": 0.95}
        mock_import.return_value = mock_mod

        server = create_server()
        tool_fn = server._tool_manager._tools["analyze_sentiment"].fn
        result = tool_fn(text="Great product!")

        mock_import.assert_called_with("anynlp", "nlp")
        mock_mod.sentiment.assert_called_once_with("Great product!")
        assert "positive" in result

    @patch("anyai.mcp_server._try_import")
    def test_profile_data_calls_tableai(self, mock_import):
        mock_mod = MagicMock()
        mock_mod.profile.return_value = {"rows": 100, "cols": 5}
        mock_import.return_value = mock_mod

        server = create_server()
        tool_fn = server._tool_manager._tools["profile_data"].fn
        result = tool_fn(data="data.csv")

        mock_import.assert_called_with("tableai", "table")
        mock_mod.profile.assert_called_once_with("data.csv")
        assert "100" in result

    @patch("anyai.mcp_server._try_import")
    def test_classify_data_calls_anyml(self, mock_import):
        mock_mod = MagicMock()
        mock_mod.classify.return_value = {"score": 0.92}
        mock_import.return_value = mock_mod

        server = create_server()
        tool_fn = server._tool_manager._tools["classify_data"].fn
        result = tool_fn(data="data.csv", target="label")

        mock_import.assert_called_with("anyml", "ml")
        mock_mod.classify.assert_called_once_with("data.csv", target="label")
        assert "0.92" in result
