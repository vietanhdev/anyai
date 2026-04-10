"""Tests for the extended anyai.utils functions (Iteration 41)."""

import os
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from anyai.utils import download_file, get_device, check_memory, format_size


# ---------------------------------------------------------------------------
# format_size
# ---------------------------------------------------------------------------


class TestFormatSize:
    """Tests for format_size()."""

    def test_zero_bytes(self):
        assert format_size(0) == "0 B"

    def test_bytes(self):
        assert format_size(500) == "500 B"

    def test_one_byte(self):
        assert format_size(1) == "1 B"

    def test_kilobytes(self):
        assert format_size(1024) == "1.00 KB"

    def test_megabytes(self):
        assert format_size(1048576) == "1.00 MB"

    def test_gigabytes(self):
        assert format_size(1073741824) == "1.00 GB"

    def test_terabytes(self):
        assert format_size(1099511627776) == "1.00 TB"

    def test_fractional(self):
        result = format_size(1536)
        assert "KB" in result
        assert "1.50" in result

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            format_size(-1)

    def test_returns_string(self):
        assert isinstance(format_size(42), str)


# ---------------------------------------------------------------------------
# get_device
# ---------------------------------------------------------------------------


class TestGetDevice:
    """Tests for get_device()."""

    def test_returns_string(self):
        result = get_device()
        assert isinstance(result, str)

    def test_returns_valid_device(self):
        result = get_device()
        assert result in ("cuda", "mps", "cpu")

    def test_cuda_when_available(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert get_device() == "cuda"

    def test_mps_when_available(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert get_device() == "mps"

    def test_cpu_when_no_gpu(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert get_device() == "cpu"

    def test_cpu_when_no_torch(self):
        with patch.dict("sys.modules", {"torch": None}):
            assert get_device() == "cpu"


# ---------------------------------------------------------------------------
# check_memory
# ---------------------------------------------------------------------------


class TestCheckMemory:
    """Tests for check_memory()."""

    def test_returns_dict(self):
        result = check_memory()
        assert isinstance(result, dict)

    def test_has_expected_keys(self):
        result = check_memory()
        for key in ("ram_total", "ram_available", "vram_total", "vram_available"):
            assert key in result

    def test_ram_values_non_negative(self):
        result = check_memory()
        assert result["ram_total"] >= 0
        assert result["ram_available"] >= 0

    def test_vram_values_non_negative(self):
        result = check_memory()
        assert result["vram_total"] >= 0
        assert result["vram_available"] >= 0

    def test_values_are_integers(self):
        result = check_memory()
        for key in ("ram_total", "ram_available", "vram_total", "vram_available"):
            assert isinstance(result[key], int)


# ---------------------------------------------------------------------------
# download_file
# ---------------------------------------------------------------------------


class TestDownloadFile:
    """Tests for download_file()."""

    def test_returns_cached_file(self, tmp_path):
        """If file already exists in cache, it should be returned directly."""
        cache = tmp_path / "cache"
        cache.mkdir()
        existing = cache / "abc123.txt"
        existing.write_text("cached data")

        result = download_file(
            "https://example.com/file.txt",
            cache_dir=str(cache),
            filename="abc123.txt",
        )
        assert result == existing

    def test_creates_cache_dir(self, tmp_path):
        """Cache directory should be created if it doesn't exist."""
        cache = tmp_path / "new_cache"
        assert not cache.exists()

        # Mock urlretrieve so we don't actually download
        with patch("anyai.utils.urllib.request.urlretrieve") as mock_dl:
            def fake_download(url, dest):
                Path(dest).write_text("data")
            mock_dl.side_effect = fake_download
            result = download_file(
                "https://example.com/model.bin",
                cache_dir=str(cache),
            )

        assert cache.exists()
        assert result.exists()

    def test_filename_derived_from_url(self, tmp_path):
        """When no filename given, one is derived from the URL hash."""
        cache = tmp_path / "cache"

        with patch("anyai.utils.urllib.request.urlretrieve") as mock_dl:
            def fake_download(url, dest):
                Path(dest).write_text("data")
            mock_dl.side_effect = fake_download
            result = download_file(
                "https://example.com/weights.bin",
                cache_dir=str(cache),
            )

        assert result.suffix == ".bin"
        assert result.exists()

    def test_download_failure_cleans_tmp(self, tmp_path):
        """If download fails, temporary file should be removed."""
        cache = tmp_path / "cache"

        with patch("anyai.utils.urllib.request.urlretrieve", side_effect=OSError("fail")):
            with pytest.raises(OSError):
                download_file(
                    "https://example.com/bad.bin",
                    cache_dir=str(cache),
                )

        # No .tmp files should remain
        if cache.exists():
            tmp_files = list(cache.glob("*.tmp"))
            assert len(tmp_files) == 0
