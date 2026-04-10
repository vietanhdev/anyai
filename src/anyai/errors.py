"""Common exception hierarchy for the AnyAI ecosystem.

All AnyAI packages should raise exceptions from this hierarchy (or
compatible subclasses) so that callers can catch errors consistently.

Usage::

    from anyai.errors import ModelNotFoundError

    raise ModelNotFoundError("llama3 is not cached; run anyai models download-all")
"""


class AnyAIError(Exception):
    """Base exception for all AnyAI errors."""


class ModelNotFoundError(AnyAIError):
    """Raised when a requested model is not cached or available."""


class BackendNotAvailableError(AnyAIError):
    """Raised when a required backend package is not installed."""


class PrivacyModeError(AnyAIError):
    """Raised when a network call is blocked by privacy/offline mode."""


class ValidationError(AnyAIError):
    """Raised when input data fails validation."""
