"""Tests for anyai.utils module."""

import pytest
from anyai.utils import check_deps, require_deps, get_version


class TestCheckDeps:
    """Tests for the check_deps function."""

    def test_stdlib_module_available(self):
        assert check_deps("json") is True

    def test_stdlib_os_available(self):
        assert check_deps("os") is True

    def test_nonexistent_package(self):
        assert check_deps("nonexistent_package_xyz_123") is False

    def test_pillow_available(self):
        assert check_deps("PIL") is True

    def test_returns_bool(self):
        result = check_deps("json")
        assert isinstance(result, bool)

    def test_optional_packages_not_available(self):
        assert check_deps("nonexistent_package_xyz_123") is False
        assert check_deps("another_fake_pkg_456") is False


class TestRequireDeps:
    """Tests for the require_deps function."""

    def test_available_module_no_error(self):
        require_deps("json")  # Should not raise.

    def test_missing_module_raises(self):
        with pytest.raises(ImportError, match="not installed"):
            require_deps("nonexistent_package_xyz")

    def test_missing_module_with_extra_hint(self):
        with pytest.raises(ImportError, match="anyai\\[cv\\]"):
            require_deps("nonexistent_cv_pkg_xyz", extra="cv")

    def test_missing_module_without_extra_hint(self):
        with pytest.raises(ImportError, match="pip install nonexistent_package"):
            require_deps("nonexistent_package")


class TestGetVersion:
    """Tests for the get_version function."""

    def test_returns_string(self):
        result = get_version()
        assert isinstance(result, str)

    def test_returns_correct_version(self):
        result = get_version()
        import re
        assert re.match(r"\d+\.\d+\.\d+", result)

    def test_version_format(self):
        result = get_version()
        parts = result.split(".")
        assert len(parts) == 3
        for part in parts:
            assert part.isdigit()
