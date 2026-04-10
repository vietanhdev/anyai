"""Tests for anyai.logging module."""

import logging

import pytest

from anyai.logging import enable_debug, get_logger, set_log_level


class TestSetLogLevel:
    def test_set_debug(self):
        set_log_level("DEBUG")
        root = logging.getLogger("anyai")
        assert root.level == logging.DEBUG

    def test_set_warning(self):
        set_log_level("WARNING")
        root = logging.getLogger("anyai")
        assert root.level == logging.WARNING

    def test_case_insensitive(self):
        set_log_level("info")
        root = logging.getLogger("anyai")
        assert root.level == logging.INFO

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError, match="Invalid log level"):
            set_log_level("VERBOSE")

    def test_set_error(self):
        set_log_level("ERROR")
        root = logging.getLogger("anyai")
        assert root.level == logging.ERROR


class TestEnableDebug:
    def test_sets_debug_level(self):
        enable_debug()
        root = logging.getLogger("anyai")
        assert root.level == logging.DEBUG


class TestGetLogger:
    def test_returns_child_logger(self):
        lg = get_logger("cv")
        assert lg.name == "anyai.cv"

    def test_returns_logger_instance(self):
        lg = get_logger("ocr")
        assert isinstance(lg, logging.Logger)

    def test_child_inherits_level(self):
        set_log_level("DEBUG")
        lg = get_logger("test_child")
        assert lg.getEffectiveLevel() == logging.DEBUG

    def test_root_has_handler(self):
        get_logger("dummy")
        root = logging.getLogger("anyai")
        assert len(root.handlers) >= 1


class TestTopLevelAccess:
    def test_importable_from_anyai(self):
        import anyai

        assert hasattr(anyai, "set_log_level")
        assert hasattr(anyai, "enable_debug")
        assert hasattr(anyai, "get_logger")
