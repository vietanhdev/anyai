"""Tests for the anyai.models cross-package model registry (Iteration 43)."""

import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from anyai import models
from anyai import core as _core


@pytest.fixture(autouse=True)
def _clean_registry():
    """Clear the registry before each test."""
    saved = dict(models._registry)
    models._registry.clear()
    yield
    models._registry.clear()
    models._registry.update(saved)


class TestRegister:
    """Tests for models.register()."""

    def test_register_adds_entry(self):
        models.register("mypkg", "/tmp/mypkg_cache")
        assert "mypkg" in models._registry
        assert models._registry["mypkg"] == Path("/tmp/mypkg_cache")

    def test_register_overwrites(self):
        models.register("mypkg", "/a")
        models.register("mypkg", "/b")
        assert models._registry["mypkg"] == Path("/b")


class TestList:
    """Tests for models.list()."""

    def test_empty_when_no_cache_dirs(self, tmp_path):
        # Point to a non-existent dir
        models.register("testpkg", str(tmp_path / "nonexistent"))
        with patch("anyai.models._known_dirs", return_value=models._registry):
            result = models.list()
        assert result == []

    def test_finds_files_in_cache(self, tmp_path):
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "model.bin").write_bytes(b"x" * 100)

        models.register("testpkg", str(cache))
        with patch("anyai.models._known_dirs", return_value={"testpkg": cache}):
            result = models.list()

        assert len(result) == 1
        assert result[0]["package"] == "testpkg"
        assert result[0]["size"] == 100
        assert "model.bin" in result[0]["path"]
        assert "B" in result[0]["size_human"]

    def test_finds_nested_files(self, tmp_path):
        cache = tmp_path / "cache"
        sub = cache / "subdir"
        sub.mkdir(parents=True)
        (sub / "a.pt").write_bytes(b"a" * 50)
        (cache / "b.pt").write_bytes(b"b" * 75)

        with patch("anyai.models._known_dirs", return_value={"pkg": cache}):
            result = models.list()

        assert len(result) == 2
        sizes = sorted(e["size"] for e in result)
        assert sizes == [50, 75]


class TestTotalSize:
    """Tests for models.total_size()."""

    def test_empty_cache(self, tmp_path):
        with patch("anyai.models._known_dirs", return_value={"pkg": tmp_path / "nope"}):
            assert models.total_size() == 0

    def test_sums_all_files(self, tmp_path):
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "a.bin").write_bytes(b"x" * 100)
        (cache / "b.bin").write_bytes(b"y" * 200)

        with patch("anyai.models._known_dirs", return_value={"pkg": cache}):
            assert models.total_size() == 300


class TestClear:
    """Tests for models.clear()."""

    def test_removes_cache_dirs(self, tmp_path):
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "model.bin").write_bytes(b"data")

        with patch("anyai.models._known_dirs", return_value={"pkg": cache}):
            cleared = models.clear()

        assert cleared == 1
        assert not cache.exists()

    def test_returns_zero_when_nothing_to_clear(self, tmp_path):
        with patch("anyai.models._known_dirs", return_value={"pkg": tmp_path / "nope"}):
            cleared = models.clear()
        assert cleared == 0


class TestDownloadAll:
    """Tests for models.download_all()."""

    def test_calls_download_defaults(self):
        mock_mod = MagicMock()
        mock_mod.download_defaults = MagicMock()

        with patch.object(_core, "_SUB_PACKAGES", {"cv": "anycv"}):
            with patch("anyai.models.importlib.import_module", return_value=mock_mod):
                result = models.download_all()

        mock_mod.download_defaults.assert_called_once()
        assert "anycv" in result

    def test_skips_packages_without_download_defaults(self):
        mock_mod = MagicMock(spec=[])  # no download_defaults attribute

        with patch.object(_core, "_SUB_PACKAGES", {"cv": "anycv"}):
            with patch("anyai.models.importlib.import_module", return_value=mock_mod):
                result = models.download_all()

        assert result == []

    def test_skips_uninstalled_packages(self):
        with patch.object(_core, "_SUB_PACKAGES", {"cv": "anycv"}):
            with patch("anyai.models.importlib.import_module", side_effect=ImportError):
                result = models.download_all()

        assert result == []

    def test_on_progress_callback(self):
        mock_mod = MagicMock()
        mock_mod.download_defaults = MagicMock()
        progress_calls = []

        def on_progress(pkg, msg):
            progress_calls.append((pkg, msg))

        with patch.object(_core, "_SUB_PACKAGES", {"cv": "anycv"}):
            with patch("anyai.models.importlib.import_module", return_value=mock_mod):
                models.download_all(on_progress=on_progress)

        assert len(progress_calls) == 2
        assert progress_calls[0] == ("anycv", "downloading default models...")
        assert progress_calls[1] == ("anycv", "done")
