"""Privacy mode support for the AnyAI ecosystem.

When privacy mode is enabled, all packages should:
- Disable any network calls
- Use only local models/backends
- Log a warning if a network call would be needed

Privacy mode can be enabled via:
- ``anyai.config.set("privacy_mode", True)``
- Environment variable ``ANYAI_PRIVACY=1``
- Passing ``privacy_mode=True`` to individual functions

Usage::

    import anyai

    # Enable globally
    anyai.config.set("privacy_mode", True)

    # Or via environment variable:
    #   export ANYAI_PRIVACY=1
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger("anyai.privacy")


# Re-export for convenience. Prefer PrivacyModeError from anyai.errors.
try:
    from anyai.errors import PrivacyModeError
except ImportError:
    class PrivacyModeError(RuntimeError):  # type: ignore[no-redef]
        """Raised when an operation would violate privacy mode restrictions."""
        pass

PrivacyError = PrivacyModeError  # backward compat alias


def is_privacy_mode(override: Optional[bool] = None) -> bool:
    """Check whether privacy mode is currently active.

    Resolution order:
    1. Explicit ``override`` parameter (if not None)
    2. Environment variable ``ANYAI_PRIVACY`` (``"1"`` or ``"true"``)
    3. ``anyai.config.get("privacy_mode")``
    4. Default: False

    Args:
        override: Explicit privacy mode flag. If provided, takes precedence
            over all other settings.

    Returns:
        True if privacy mode is active.
    """
    if override is not None:
        return override

    # Check environment variable
    env_val = os.environ.get("ANYAI_PRIVACY", "").lower()
    if env_val in ("1", "true", "yes", "on"):
        return True
    if env_val in ("0", "false", "no", "off"):
        return False

    # Check anyai config (lazy import to avoid circular dependencies)
    try:
        from anyai.config import Config
        # Use a standalone check rather than importing the global singleton
        # to avoid circular imports at module load time
        import anyai
        val = anyai.config.get("privacy_mode", False)
        if isinstance(val, str):
            return val.lower() in ("1", "true", "yes", "on")
        return bool(val)
    except (ImportError, AttributeError):
        pass

    return False


def check_privacy(operation: str, override: Optional[bool] = None) -> None:
    """Check privacy mode and raise if a network operation is attempted.

    Call this before any operation that would send data over the network.

    Args:
        operation: Description of the network operation (for the error message).
        override: Explicit privacy mode flag.

    Raises:
        PrivacyModeError: If privacy mode is active and the operation requires
            network access.
    """
    if is_privacy_mode(override):
        raise PrivacyModeError(
            f"Privacy mode is enabled. Cannot perform network operation: {operation}. "
            f"Disable privacy mode or use a local-only alternative."
        )


def warn_privacy(operation: str, override: Optional[bool] = None) -> bool:
    """Log a warning if privacy mode is on and a network call would be needed.

    Unlike :func:`check_privacy`, this does not raise an exception.

    Args:
        operation: Description of the network operation.
        override: Explicit privacy mode flag.

    Returns:
        True if privacy mode is active (caller should avoid the operation).
    """
    if is_privacy_mode(override):
        logger.warning(
            "Privacy mode is enabled. Network operation would be needed: %s",
            operation,
        )
        return True
    return False
