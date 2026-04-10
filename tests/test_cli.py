"""Tests for the anyai CLI."""

import types
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from anyai.cli import main


@pytest.fixture
def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# Help and version
# ---------------------------------------------------------------------------

class TestHelpAndVersion:
    """Basic smoke tests for --help and --version."""

    def test_main_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "AnyAI" in result.output

    def test_main_no_args_shows_help(self, runner):
        result = runner.invoke(main, [])
        assert result.exit_code == 0
        assert "AnyAI" in result.output

    def test_version(self, runner):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        import re
        assert re.search(r"\d+\.\d+\.\d+", result.output)


# ---------------------------------------------------------------------------
# Sub-command help
# ---------------------------------------------------------------------------

class TestSubcommandHelp:
    """Every sub-command should have --help that works."""

    @pytest.mark.parametrize("cmd", [
        "detect", "classify", "segment",
        "ocr",
        "chat",
        "summarize", "sentiment", "entities", "keywords",
        "train", "predict",
        "profile", "clean",
        "export", "serve", "benchmark",
        "info", "doctor",
    ])
    def test_subcommand_help(self, runner, cmd):
        result = runner.invoke(main, [cmd, "--help"])
        assert result.exit_code == 0
        assert "--help" in result.output or cmd in result.output


# ---------------------------------------------------------------------------
# Info and doctor
# ---------------------------------------------------------------------------

class TestInfoCommand:
    """Tests for the 'info' command."""

    def test_info_runs(self, runner):
        result = runner.invoke(main, ["info"])
        assert result.exit_code == 0
        assert "anyai" in result.output.lower()

    def test_info_shows_version(self, runner):
        result = runner.invoke(main, ["info"])
        import re
        assert re.search(r"\d+\.\d+\.\d+", result.output)


class TestDoctorCommand:
    """Tests for the 'doctor' command."""

    def test_doctor_runs(self, runner):
        result = runner.invoke(main, ["doctor"])
        assert result.exit_code == 0

    def test_doctor_lists_packages(self, runner):
        result = runner.invoke(main, ["doctor"])
        # Should list at least some of the sub-packages
        assert "anycv" in result.output or "anyocr" in result.output or "anyllm" in result.output

    def test_doctor_shows_install_hints(self, runner):
        result = runner.invoke(main, ["doctor"])
        assert "pip install" in result.output


# ---------------------------------------------------------------------------
# Missing sub-package gives clear install instructions
# ---------------------------------------------------------------------------

class TestMissingPackageErrors:
    """Commands that require missing packages should exit with install hints."""

    def test_detect_missing_anycv(self, runner):
        with patch("anyai.cli.importlib.import_module", side_effect=ImportError):
            result = runner.invoke(main, ["detect", "img.jpg"])
        assert result.exit_code != 0
        assert "pip install anyai[cv]" in result.output

    def test_ocr_missing_anyocr(self, runner):
        with patch("anyai.cli.importlib.import_module", side_effect=ImportError):
            result = runner.invoke(main, ["ocr", "doc.pdf"])
        assert result.exit_code != 0
        assert "pip install anyai[ocr]" in result.output

    def test_chat_missing_anyllm(self, runner):
        with patch("anyai.cli.importlib.import_module", side_effect=ImportError):
            result = runner.invoke(main, ["chat", "hello"])
        assert result.exit_code != 0
        assert "pip install anyai[llm]" in result.output

    def test_entities_missing_anynlp(self, runner):
        with patch("anyai.cli.importlib.import_module", side_effect=ImportError):
            result = runner.invoke(main, ["entities", "John works at Google"])
        assert result.exit_code != 0
        assert "pip install anyai[nlp]" in result.output

    def test_train_missing_anyml(self, runner):
        with patch("anyai.cli.importlib.import_module", side_effect=ImportError):
            result = runner.invoke(main, ["train", "data.csv", "--target", "label"])
        assert result.exit_code != 0
        assert "pip install anyai[ml]" in result.output

    def test_profile_missing_tableai(self, runner):
        with patch("anyai.cli.importlib.import_module", side_effect=ImportError):
            result = runner.invoke(main, ["profile", "data.csv"])
        assert result.exit_code != 0
        assert "pip install anyai[table]" in result.output

    def test_export_missing_anydeploy(self, runner):
        with patch("anyai.cli.importlib.import_module", side_effect=ImportError):
            result = runner.invoke(main, ["export", "model.pt"])
        assert result.exit_code != 0
        assert "pip install anyai[deploy]" in result.output


# ---------------------------------------------------------------------------
# Built-in NLP commands (no extra package required)
# ---------------------------------------------------------------------------

class TestBuiltinNLP:
    """Summarize, sentiment, and keywords use built-in anyai.text."""

    def test_summarize(self, runner):
        text = "AI is transforming the world. Machine learning is powerful. Deep learning is a subset."
        result = runner.invoke(main, ["summarize", text])
        assert result.exit_code == 0
        assert "Summary" in result.output

    def test_sentiment_positive(self, runner):
        result = runner.invoke(main, ["sentiment", "I love this product!"])
        assert result.exit_code == 0
        assert "Positive" in result.output or "positive" in result.output.lower()

    def test_sentiment_negative(self, runner):
        result = runner.invoke(main, ["sentiment", "This is terrible and awful."])
        assert result.exit_code == 0
        assert "Negative" in result.output or "negative" in result.output.lower()

    def test_keywords(self, runner):
        result = runner.invoke(main, ["keywords", "python machine learning artificial intelligence python"])
        assert result.exit_code == 0
        assert "python" in result.output.lower()

    def test_keywords_top_n(self, runner):
        result = runner.invoke(main, ["keywords", "--top", "2", "python java ruby python java"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Chat requires a message or -i
# ---------------------------------------------------------------------------

class TestChatValidation:
    """Chat command should require a message or interactive flag."""

    def test_chat_no_message_no_interactive_errors(self, runner):
        fake_mod = types.ModuleType("anyllm")
        fake_mod.chat = MagicMock()
        with patch("anyai.cli.importlib.import_module", return_value=fake_mod):
            result = runner.invoke(main, ["chat"])
        assert result.exit_code != 0

    def test_chat_with_message(self, runner):
        fake_mod = types.ModuleType("anyllm")
        fake_mod.chat = MagicMock(return_value="Hello!")
        with patch("anyai.cli.importlib.import_module", return_value=fake_mod):
            result = runner.invoke(main, ["chat", "hi"])
        assert result.exit_code == 0
        fake_mod.chat.assert_called_once()

    def test_chat_with_model_option(self, runner):
        fake_mod = types.ModuleType("anyllm")
        fake_mod.chat = MagicMock(return_value="Hi there")
        with patch("anyai.cli.importlib.import_module", return_value=fake_mod):
            result = runner.invoke(main, ["chat", "--model", "ollama/llama3", "hello"])
        assert result.exit_code == 0
        fake_mod.chat.assert_called_once_with("hello", model="ollama/llama3")

    def test_chat_stream_flag(self, runner):
        fake_mod = types.ModuleType("anyllm")
        fake_mod.chat = MagicMock(return_value="Streamed response")
        with patch("anyai.cli.importlib.import_module", return_value=fake_mod):
            result = runner.invoke(main, ["chat", "--stream", "hello"])
        assert result.exit_code == 0
        fake_mod.chat.assert_called_once_with("hello", stream=True)


# ---------------------------------------------------------------------------
# Interactive chat mode
# ---------------------------------------------------------------------------

class TestInteractiveChat:
    """Tests for the interactive chat REPL."""

    def test_interactive_quit(self, runner):
        fake_mod = types.ModuleType("anyllm")
        fake_mod.chat = MagicMock()
        with patch("anyai.cli.importlib.import_module", return_value=fake_mod):
            result = runner.invoke(main, ["chat", "-i"], input="quit\n")
        assert result.exit_code == 0
        assert "Bye" in result.output

    def test_interactive_chat_exchange(self, runner):
        fake_mod = types.ModuleType("anyllm")
        fake_mod.chat = MagicMock(return_value="I am an AI.")
        with patch("anyai.cli.importlib.import_module", return_value=fake_mod):
            result = runner.invoke(main, ["chat", "-i"], input="Who are you?\nquit\n")
        assert result.exit_code == 0
        fake_mod.chat.assert_called_once()

    def test_interactive_model_switch(self, runner):
        fake_mod = types.ModuleType("anyllm")
        fake_mod.chat = MagicMock(return_value="Response")
        with patch("anyai.cli.importlib.import_module", return_value=fake_mod):
            result = runner.invoke(
                main, ["chat", "-i"],
                input="/model gpt-4\nhello\nquit\n",
            )
        assert result.exit_code == 0
        assert "gpt-4" in result.output
        fake_mod.chat.assert_called_once_with("hello", model="gpt-4")


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

class TestOutputFormatting:
    """Verify that rich output is used."""

    def test_detect_json_format(self, runner):
        fake_mod = types.ModuleType("anycv")
        fake_mod.detect = MagicMock(return_value={"objects": ["cat", "dog"]})
        with patch("anyai.cli.importlib.import_module", return_value=fake_mod):
            result = runner.invoke(main, ["detect", "img.jpg", "--format", "json"])
        assert result.exit_code == 0
        assert "cat" in result.output
        assert "dog" in result.output

    def test_detect_text_format(self, runner):
        fake_mod = types.ModuleType("anycv")
        fake_mod.detect = MagicMock(return_value={"objects": ["cat"]})
        with patch("anyai.cli.importlib.import_module", return_value=fake_mod):
            result = runner.invoke(main, ["detect", "img.jpg", "--format", "text"])
        assert result.exit_code == 0

    def test_detect_table_format(self, runner):
        fake_mod = types.ModuleType("anycv")
        fake_mod.detect = MagicMock(return_value={"objects": ["cat"]})
        with patch("anyai.cli.importlib.import_module", return_value=fake_mod):
            result = runner.invoke(main, ["detect", "img.jpg", "--format", "table"])
        assert result.exit_code == 0

    def test_output_to_file(self, runner, tmp_path):
        fake_mod = types.ModuleType("anycv")
        fake_mod.detect = MagicMock(return_value={"objects": ["cat"]})
        outfile = str(tmp_path / "out.json")
        with patch("anyai.cli.importlib.import_module", return_value=fake_mod):
            result = runner.invoke(main, ["detect", "img.jpg", "--format", "json", "-o", outfile])
        assert result.exit_code == 0
        with open(outfile) as f:
            content = f.read()
        assert "cat" in content
