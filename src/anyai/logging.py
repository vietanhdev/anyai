"""Shared logging configuration for all anyai packages.

Provides a simple interface to control log levels across all ``anyai.*``
loggers at once, using standard Python :mod:`logging`.

Usage::

    import anyai

    # Configure logging level
    anyai.set_log_level("DEBUG")  # DEBUG, INFO, WARNING, ERROR
    anyai.set_log_level("WARNING")  # Default - quiet

    # Shortcut for debug mode
    anyai.enable_debug()

    # Each package uses standard Python logging
    import logging
    logger = logging.getLogger("anyai.cv")
"""

import logging
from typing import Optional

# The root logger for all anyai packages
_ROOT_LOGGER_NAME = "anyai"

# Valid level names
_VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def _get_root_logger() -> logging.Logger:
    """Return the root ``anyai`` logger, creating a handler if needed."""
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        # Default: WARNING (quiet)
        logger.setLevel(logging.WARNING)
    return logger


def set_log_level(level: str) -> None:
    """Set the logging level for all ``anyai.*`` loggers.

    Args:
        level: One of ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``,
            or ``"CRITICAL"`` (case-insensitive).

    Raises:
        ValueError: If *level* is not a recognized level name.

    Example::

        import anyai
        anyai.set_log_level("DEBUG")
    """
    level_upper = level.upper()
    if level_upper not in _VALID_LEVELS:
        raise ValueError(
            f"Invalid log level '{level}'. "
            f"Choose from: {', '.join(sorted(_VALID_LEVELS))}"
        )
    root = _get_root_logger()
    root.setLevel(getattr(logging, level_upper))


def enable_debug() -> None:
    """Shortcut to set all ``anyai.*`` loggers to DEBUG level.

    Example::

        import anyai
        anyai.enable_debug()
    """
    set_log_level("DEBUG")


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the ``anyai`` namespace.

    This ensures that the root handler is initialised and returns a logger
    named ``anyai.<name>``.

    Args:
        name: Logger suffix (e.g. ``"cv"``, ``"ocr"``).

    Returns:
        A :class:`logging.Logger` instance.
    """
    _get_root_logger()  # ensure handler exists
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
