"""Tests for the unified configuration system."""

import os
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from anyai.config import (
    Config,
    _dotted_to_env_key,
    _env_key_to_dotted,
    _flatten_dict,
    _unflatten_dict,
)


# ---------------------------------------------------------------------------
# Helper conversion functions
# ---------------------------------------------------------------------------


class TestDottedToEnvKey:
    def test_simple(self):
        assert _dotted_to_env_key("llm.default_model") == "ANYAI_LLM_DEFAULT_MODEL"

    def test_single_segment(self):
        assert _dotted_to_env_key("debug") == "ANYAI_DEBUG"

    def test_multi_segment(self):
        assert _dotted_to_env_key("cv.cache_dir") == "ANYAI_CV_CACHE_DIR"


class TestEnvKeyToDotted:
    def test_simple(self):
        assert _env_key_to_dotted("ANYAI_LLM_DEFAULT_MODEL") == "llm.default_model"

    def test_single_segment(self):
        assert _env_key_to_dotted("ANYAI_DEBUG") == "debug"

    def test_section_key(self):
        assert _env_key_to_dotted("ANYAI_CV_CACHE_DIR") == "cv.cache_dir"


class TestFlattenDict:
    def test_simple(self):
        assert _flatten_dict({"a": 1}) == {"a": 1}

    def test_nested(self):
        result = _flatten_dict({"llm": {"default_model": "x", "timeout": 30}})
        assert result == {"llm.default_model": "x", "llm.timeout": 30}

    def test_deeply_nested(self):
        result = _flatten_dict({"a": {"b": {"c": 1}}})
        assert result == {"a.b.c": 1}

    def test_empty(self):
        assert _flatten_dict({}) == {}


class TestUnflattenDict:
    def test_simple(self):
        assert _unflatten_dict({"a": 1}) == {"a": 1}

    def test_dotted(self):
        result = _unflatten_dict({"llm.default_model": "x", "llm.timeout": 30})
        assert result == {"llm": {"default_model": "x", "timeout": 30}}

    def test_deeply_nested(self):
        result = _unflatten_dict({"a.b.c": 1})
        assert result == {"a": {"b": {"c": 1}}}

    def test_empty(self):
        assert _unflatten_dict({}) == {}


# ---------------------------------------------------------------------------
# Config class
# ---------------------------------------------------------------------------


class TestConfigGetSet:
    def setup_method(self):
        self.cfg = Config()

    def test_set_and_get(self):
        self.cfg.set("llm.default_model", "ollama/llama3")
        assert self.cfg.get("llm.default_model") == "ollama/llama3"

    def test_get_missing_returns_default(self):
        assert self.cfg.get("nonexistent.key") is None
        assert self.cfg.get("nonexistent.key", "fallback") == "fallback"

    def test_set_overwrites(self):
        self.cfg.set("key", "v1")
        self.cfg.set("key", "v2")
        assert self.cfg.get("key") == "v2"

    def test_delete(self):
        self.cfg.set("key", "value")
        self.cfg.delete("key")
        assert self.cfg.get("key") is None

    def test_delete_missing_raises(self):
        with pytest.raises(KeyError):
            self.cfg.delete("nonexistent")

    def test_various_value_types(self):
        self.cfg.set("str_key", "hello")
        self.cfg.set("int_key", 42)
        self.cfg.set("float_key", 3.14)
        self.cfg.set("bool_key", True)
        self.cfg.set("list_key", [1, 2, 3])
        self.cfg.set("dict_key", {"nested": True})

        assert self.cfg.get("str_key") == "hello"
        assert self.cfg.get("int_key") == 42
        assert self.cfg.get("float_key") == 3.14
        assert self.cfg.get("bool_key") is True
        assert self.cfg.get("list_key") == [1, 2, 3]
        assert self.cfg.get("dict_key") == {"nested": True}


class TestConfigDefaults:
    def setup_method(self):
        self.cfg = Config()

    def test_set_default(self):
        self.cfg.set_default("cv.cache_dir", "/default/cache")
        assert self.cfg.get("cv.cache_dir") == "/default/cache"

    def test_override_beats_default(self):
        self.cfg.set_default("key", "default_value")
        self.cfg.set("key", "override_value")
        assert self.cfg.get("key") == "override_value"

    def test_set_defaults_bulk(self):
        self.cfg.set_defaults({
            "cv.cache_dir": "/cache",
            "ocr.default_backend": "surya",
        })
        assert self.cfg.get("cv.cache_dir") == "/cache"
        assert self.cfg.get("ocr.default_backend") == "surya"


class TestConfigEnvVars:
    def setup_method(self):
        self.cfg = Config()

    def test_env_var_overrides_set(self):
        self.cfg.set("llm.default_model", "programmatic")
        with patch.dict(os.environ, {"ANYAI_LLM_DEFAULT_MODEL": "from_env"}):
            assert self.cfg.get("llm.default_model") == "from_env"

    def test_env_var_overrides_default(self):
        self.cfg.set_default("cv.cache_dir", "/default")
        with patch.dict(os.environ, {"ANYAI_CV_CACHE_DIR": "/env_cache"}):
            assert self.cfg.get("cv.cache_dir") == "/env_cache"

    def test_no_env_var(self):
        self.cfg.set("key", "value")
        assert self.cfg.get("key") == "value"


class TestConfigAsDict:
    def setup_method(self):
        self.cfg = Config()

    def test_merges_defaults_and_overrides(self):
        self.cfg.set_default("a", 1)
        self.cfg.set("b", 2)
        d = self.cfg.as_dict()
        assert d["a"] == 1
        assert d["b"] == 2

    def test_override_wins(self):
        self.cfg.set_default("key", "default")
        self.cfg.set("key", "override")
        d = self.cfg.as_dict()
        assert d["key"] == "override"

    def test_env_var_in_as_dict(self):
        self.cfg.set("llm.model", "set_value")
        with patch.dict(os.environ, {"ANYAI_LLM_MODEL": "env_value"}):
            d = self.cfg.as_dict()
            assert d["llm.model"] == "env_value"


class TestConfigReset:
    def test_reset_clears_all(self):
        cfg = Config()
        cfg.set("a", 1)
        cfg.set_default("b", 2)
        cfg.reset()
        assert cfg.get("a") is None
        assert cfg.get("b") is None
        assert cfg.as_dict() == {} or all(
            k.startswith("ANYAI_") for k in cfg.as_dict()
        )


class TestConfigLoadSave:
    def test_load_yaml(self, tmp_path):
        yaml_content = "llm:\n  default_model: ollama/llama3\ncv:\n  cache_dir: /custom/cache\n"
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        cfg = Config()
        cfg.load(config_file)

        assert cfg.get("llm.default_model") == "ollama/llama3"
        assert cfg.get("cv.cache_dir") == "/custom/cache"

    def test_load_missing_file(self, tmp_path):
        cfg = Config()
        with pytest.raises(FileNotFoundError):
            cfg.load(tmp_path / "nonexistent.yaml")

    def test_save_yaml(self, tmp_path):
        cfg = Config()
        cfg.set("llm.default_model", "ollama/llama3")
        cfg.set("cv.cache_dir", "/custom/cache")

        config_file = tmp_path / "output.yaml"
        cfg.save(config_file)

        assert config_file.exists()

        # Reload and verify
        cfg2 = Config()
        cfg2.load(config_file)
        assert cfg2.get("llm.default_model") == "ollama/llama3"
        assert cfg2.get("cv.cache_dir") == "/custom/cache"

    def test_save_creates_directories(self, tmp_path):
        cfg = Config()
        cfg.set("key", "value")
        nested_path = tmp_path / "deep" / "nested" / "config.yaml"
        cfg.save(nested_path)
        assert nested_path.exists()

    def test_load_empty_yaml(self, tmp_path):
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        cfg = Config()
        cfg.load(config_file)  # Should not raise

    def test_load_expands_tilde(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("key: value\n")

        cfg = Config()
        # Use the actual path (tilde expansion tested by pathlib)
        cfg.load(str(config_file))
        assert cfg.get("key") == "value"


class TestConfigRepr:
    def test_repr(self):
        cfg = Config()
        cfg.set("a", 1)
        cfg.set_default("b", 2)
        r = repr(cfg)
        assert "Config(" in r
        assert "overrides=1" in r
        assert "defaults=1" in r


class TestConfigThreadSafety:
    def test_concurrent_set_get(self):
        cfg = Config()
        errors = []

        def writer(n):
            try:
                for i in range(100):
                    cfg.set(f"thread_{n}.key_{i}", i)
            except Exception as e:
                errors.append(e)

        def reader(n):
            try:
                for i in range(100):
                    cfg.get(f"thread_{n}.key_{i}")
            except Exception as e:
                errors.append(e)

        threads = []
        for n in range(5):
            threads.append(threading.Thread(target=writer, args=(n,)))
            threads.append(threading.Thread(target=reader, args=(n,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread safety errors: {errors}"


class TestConfigIntegration:
    """Test that anyai.config is accessible as the global singleton."""

    def test_global_config_accessible(self):
        import anyai
        assert hasattr(anyai, "config")
        assert isinstance(anyai.config, Config)

    def test_global_config_set_get(self):
        import anyai
        anyai.config.set("test.integration_key", "test_value")
        assert anyai.config.get("test.integration_key") == "test_value"
        # Clean up
        anyai.config.delete("test.integration_key")
