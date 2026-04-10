"""Tests for the anyai.errors exception hierarchy (Iteration 45)."""

import pytest

from anyai.errors import (
    AnyAIError,
    BackendNotAvailableError,
    ModelNotFoundError,
    PrivacyModeError,
    ValidationError,
)


class TestAnyAIError:
    """Tests for the base AnyAIError."""

    def test_is_exception(self):
        assert issubclass(AnyAIError, Exception)

    def test_can_raise_and_catch(self):
        with pytest.raises(AnyAIError):
            raise AnyAIError("test error")

    def test_message_preserved(self):
        err = AnyAIError("something went wrong")
        assert str(err) == "something went wrong"

    def test_caught_by_exception(self):
        with pytest.raises(Exception):
            raise AnyAIError("test")


class TestModelNotFoundError:
    """Tests for ModelNotFoundError."""

    def test_is_anyai_error(self):
        assert issubclass(ModelNotFoundError, AnyAIError)

    def test_can_raise_and_catch(self):
        with pytest.raises(ModelNotFoundError):
            raise ModelNotFoundError("llama3 not cached")

    def test_caught_by_base(self):
        with pytest.raises(AnyAIError):
            raise ModelNotFoundError("model missing")

    def test_message(self):
        err = ModelNotFoundError("yolov8n not found")
        assert "yolov8n" in str(err)


class TestBackendNotAvailableError:
    """Tests for BackendNotAvailableError."""

    def test_is_anyai_error(self):
        assert issubclass(BackendNotAvailableError, AnyAIError)

    def test_can_raise_and_catch(self):
        with pytest.raises(BackendNotAvailableError):
            raise BackendNotAvailableError("torch not installed")

    def test_caught_by_base(self):
        with pytest.raises(AnyAIError):
            raise BackendNotAvailableError("missing backend")


class TestPrivacyModeError:
    """Tests for PrivacyModeError."""

    def test_is_anyai_error(self):
        assert issubclass(PrivacyModeError, AnyAIError)

    def test_can_raise_and_catch(self):
        with pytest.raises(PrivacyModeError):
            raise PrivacyModeError("network blocked in privacy mode")

    def test_caught_by_base(self):
        with pytest.raises(AnyAIError):
            raise PrivacyModeError("blocked")


class TestValidationError:
    """Tests for ValidationError."""

    def test_is_anyai_error(self):
        assert issubclass(ValidationError, AnyAIError)

    def test_can_raise_and_catch(self):
        with pytest.raises(ValidationError):
            raise ValidationError("invalid input")

    def test_caught_by_base(self):
        with pytest.raises(AnyAIError):
            raise ValidationError("bad data")

    def test_message(self):
        err = ValidationError("field 'name' is required")
        assert "name" in str(err)


class TestHierarchy:
    """Test that the hierarchy is correct and all errors are distinct."""

    def test_all_subclass_anyai_error(self):
        for cls in (ModelNotFoundError, BackendNotAvailableError,
                    PrivacyModeError, ValidationError):
            assert issubclass(cls, AnyAIError)

    def test_errors_are_distinct(self):
        classes = [ModelNotFoundError, BackendNotAvailableError,
                   PrivacyModeError, ValidationError]
        for i, a in enumerate(classes):
            for b in classes[i + 1:]:
                assert not issubclass(a, b)
                assert not issubclass(b, a)

    def test_catch_specific_not_others(self):
        with pytest.raises(ModelNotFoundError):
            try:
                raise ModelNotFoundError("test")
            except BackendNotAvailableError:
                pytest.fail("Should not be caught by BackendNotAvailableError")

    def test_importable_from_anyai(self):
        import anyai
        assert anyai.AnyAIError is AnyAIError
        assert anyai.ModelNotFoundError is ModelNotFoundError
        assert anyai.BackendNotAvailableError is BackendNotAvailableError
        assert anyai.PrivacyModeError is PrivacyModeError
        assert anyai.ValidationError is ValidationError
