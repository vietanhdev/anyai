"""Tests for anyai.image module."""

import os
import tempfile

import pytest
from PIL import Image

from anyai.image import describe, classify


@pytest.fixture
def sample_jpeg(tmp_path):
    """Create a temporary JPEG image for testing."""
    img = Image.new("RGB", (100, 50), color="red")
    path = str(tmp_path / "test.jpg")
    img.save(path, format="JPEG")
    return path


@pytest.fixture
def sample_png(tmp_path):
    """Create a temporary PNG image for testing."""
    img = Image.new("RGBA", (200, 150), color=(0, 128, 255, 255))
    path = str(tmp_path / "test.png")
    img.save(path, format="PNG")
    return path


# --- describe ---

class TestDescribe:
    """Tests for the describe function."""

    def test_returns_dict(self, sample_jpeg):
        result = describe(sample_jpeg)
        assert isinstance(result, dict)

    def test_contains_required_keys(self, sample_jpeg):
        result = describe(sample_jpeg)
        required = {"path", "format", "width", "height", "mode", "file_size_bytes",
                     "has_exif", "ai_description"}
        assert required.issubset(result.keys())

    def test_jpeg_format(self, sample_jpeg):
        result = describe(sample_jpeg)
        assert result["format"] == "JPEG"

    def test_png_format(self, sample_png):
        result = describe(sample_png)
        assert result["format"] == "PNG"

    def test_dimensions_jpeg(self, sample_jpeg):
        result = describe(sample_jpeg)
        assert result["width"] == 100
        assert result["height"] == 50

    def test_dimensions_png(self, sample_png):
        result = describe(sample_png)
        assert result["width"] == 200
        assert result["height"] == 150

    def test_mode_rgb(self, sample_jpeg):
        result = describe(sample_jpeg)
        assert result["mode"] == "RGB"

    def test_mode_rgba(self, sample_png):
        result = describe(sample_png)
        assert result["mode"] == "RGBA"

    def test_file_size_positive(self, sample_jpeg):
        result = describe(sample_jpeg)
        assert result["file_size_bytes"] > 0

    def test_ai_description_none_without_backend(self, sample_jpeg):
        result = describe(sample_jpeg)
        assert result["ai_description"] is None

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            describe("/nonexistent/path/image.jpg")

    def test_invalid_image(self, tmp_path):
        bad_file = str(tmp_path / "bad.jpg")
        with open(bad_file, "w") as f:
            f.write("this is not an image")
        with pytest.raises(ValueError):
            describe(bad_file)

    def test_path_in_result(self, sample_jpeg):
        result = describe(sample_jpeg)
        assert result["path"] == sample_jpeg


# --- classify ---

class TestClassify:
    """Tests for the classify function."""

    def test_returns_dict(self, sample_jpeg):
        result = classify(sample_jpeg)
        assert isinstance(result, dict)

    def test_contains_required_keys(self, sample_jpeg):
        result = classify(sample_jpeg)
        assert "path" in result
        assert "format_type" in result
        assert "ai_classification" in result

    def test_jpeg_classified_as_photo(self, sample_jpeg):
        result = classify(sample_jpeg)
        assert result["format_type"] == "photo"

    def test_png_classified_as_graphic(self, sample_png):
        result = classify(sample_png)
        assert result["format_type"] == "graphic"

    def test_ai_classification_none_without_backend(self, sample_jpeg):
        result = classify(sample_jpeg)
        assert result["ai_classification"] is None

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            classify("/nonexistent/path/image.jpg")
