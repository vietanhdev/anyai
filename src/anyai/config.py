"""Unified configuration system for the AnyAI ecosystem.

Provides a thread-safe, hierarchical configuration store that supports:

- Dotted key access (``"llm.default_model"``)
- Environment variable overrides via the ``ANYAI_`` prefix
- YAML config file loading and saving
- Per-sub-package defaults

Usage::

    import anyai

    # Set values programmatically
    anyai.config.set("llm.default_model", "ollama/llama3")
    anyai.config.set("cv.cache_dir", "/custom/cache")

    # Read values
    model = anyai.config.get("llm.default_model")

    # From environment variables:
    #   ANYAI_LLM_DEFAULT_MODEL=ollama/llama3
    #   ANYAI_CV_CACHE_DIR=/custom/cache

    # Load / save YAML files
    anyai.config.load("~/.anyai/config.yaml")
    anyai.config.save("~/.anyai/config.yaml")
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional


_ENV_PREFIX = "ANYAI_"


class Config:
    """Thread-safe, hierarchical configuration store.

    Keys use dotted notation (e.g. ``"cv.cache_dir"``).  Values are stored
    in a flat dictionary keyed by the full dotted path.

    Resolution order (highest priority first):

    1. Environment variables (``ANYAI_<SECTION>_<KEY>``)
    2. Programmatic ``set()`` calls
    3. Values loaded from a YAML config file
    4. Built-in defaults registered by sub-packages
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._data: Dict[str, Any] = {}
        self._defaults: Dict[str, Any] = {}

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a configuration value.

        Checks environment variables first (``ANYAI_`` prefix), then
        programmatic overrides, then registered defaults.

        Args:
            key: Dotted configuration key, e.g. ``"llm.default_model"``.
            default: Fallback value when the key is not found anywhere.

        Returns:
            The resolved value, or *default* if not found.
        """
        # 1. Check environment variables
        env_val = self._from_env(key)
        if env_val is not None:
            return env_val

        with self._lock:
            # 2. Programmatic overrides
            if key in self._data:
                return self._data[key]

            # 3. Registered defaults
            if key in self._defaults:
                return self._defaults[key]

        return default

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value.

        Args:
            key: Dotted configuration key.
            value: The value to store.
        """
        with self._lock:
            self._data[key] = value

    def delete(self, key: str) -> None:
        """Remove a programmatic override for *key*.

        Does not affect defaults or environment variables.

        Args:
            key: Dotted configuration key.

        Raises:
            KeyError: If the key has no programmatic override.
        """
        with self._lock:
            if key not in self._data:
                raise KeyError(f"Configuration key not set: {key!r}")
            del self._data[key]

    def set_default(self, key: str, value: Any) -> None:
        """Register a default value for *key*.

        Defaults have the lowest priority and are typically set by
        sub-packages during initialization.

        Args:
            key: Dotted configuration key.
            value: Default value.
        """
        with self._lock:
            self._defaults[key] = value

    def set_defaults(self, defaults: Dict[str, Any]) -> None:
        """Register multiple default values at once.

        Args:
            defaults: Mapping of dotted keys to default values.
        """
        with self._lock:
            self._defaults.update(defaults)

    def load(self, path: str | Path) -> None:
        """Load configuration from a YAML file.

        Loaded values are treated as programmatic overrides (i.e. they
        take precedence over defaults but not environment variables).

        The YAML file is expected to have a nested structure that maps
        to dotted keys::

            llm:
              default_model: ollama/llama3
            cv:
              cache_dir: /custom/cache

        Args:
            path: Path to the YAML file.  ``~`` is expanded.

        Raises:
            FileNotFoundError: If the file does not exist.
            ImportError: If PyYAML is not installed.
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required to load config files. "
                "Install it with: pip install pyyaml"
            )

        resolved = Path(path).expanduser()
        if not resolved.exists():
            raise FileNotFoundError(f"Config file not found: {resolved}")

        with open(resolved) as fh:
            raw = yaml.safe_load(fh)

        if not isinstance(raw, dict):
            return

        flat = _flatten_dict(raw)
        with self._lock:
            self._data.update(flat)

    def save(self, path: str | Path) -> None:
        """Save the current configuration to a YAML file.

        Only programmatic overrides are saved (not defaults or env vars).

        Args:
            path: Destination path.  ``~`` is expanded.  Parent
                directories are created automatically.

        Raises:
            ImportError: If PyYAML is not installed.
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required to save config files. "
                "Install it with: pip install pyyaml"
            )

        resolved = Path(path).expanduser()
        resolved.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            nested = _unflatten_dict(self._data)

        with open(resolved, "w") as fh:
            yaml.safe_dump(nested, fh, default_flow_style=False)

    def as_dict(self) -> Dict[str, Any]:
        """Return all effective configuration as a flat dictionary.

        Merges defaults, programmatic overrides, and environment
        variables (env wins).

        Returns:
            A dict mapping dotted keys to their resolved values.
        """
        with self._lock:
            merged = dict(self._defaults)
            merged.update(self._data)

        # Layer in env vars
        for key in list(merged.keys()):
            env_val = self._from_env(key)
            if env_val is not None:
                merged[key] = env_val

        # Also pick up env vars that were not in merged
        prefix = _ENV_PREFIX
        for env_key, env_val in os.environ.items():
            if env_key.startswith(prefix):
                dotted = _env_key_to_dotted(env_key)
                if dotted not in merged:
                    merged[dotted] = env_val

        return merged

    def reset(self) -> None:
        """Clear all programmatic overrides and defaults."""
        with self._lock:
            self._data.clear()
            self._defaults.clear()

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    @staticmethod
    def _from_env(key: str) -> Optional[str]:
        """Check for an environment variable matching *key*.

        The mapping is::

            "llm.default_model"  ->  ANYAI_LLM_DEFAULT_MODEL

        Returns:
            The env var value, or ``None`` if not set.
        """
        env_name = _dotted_to_env_key(key)
        return os.environ.get(env_name)

    def __repr__(self) -> str:
        with self._lock:
            n_overrides = len(self._data)
            n_defaults = len(self._defaults)
        return f"Config(overrides={n_overrides}, defaults={n_defaults})"


# ---------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------


def _dotted_to_env_key(key: str) -> str:
    """Convert a dotted key to an environment variable name.

    ``"llm.default_model"`` -> ``"ANYAI_LLM_DEFAULT_MODEL"``
    """
    return _ENV_PREFIX + key.replace(".", "_").upper()


def _env_key_to_dotted(env_key: str) -> str:
    """Convert an environment variable name back to a dotted key.

    ``"ANYAI_LLM_DEFAULT_MODEL"`` -> ``"llm.default_model"``

    Note: since we cannot distinguish section separators from
    underscores within a section, this uses a simple heuristic:
    the first segment before the first underscore (after removing
    the prefix) is treated as the section.
    """
    stripped = env_key[len(_ENV_PREFIX):].lower()
    parts = stripped.split("_", 1)
    if len(parts) == 2:
        return f"{parts[0]}.{parts[1]}"
    return stripped


def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested dictionary into dotted-key format.

    ``{"llm": {"default_model": "x"}}`` -> ``{"llm.default_model": "x"}``
    """
    result: Dict[str, Any] = {}
    for k, v in d.items():
        full_key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            result.update(_flatten_dict(v, full_key))
        else:
            result[full_key] = v
    return result


def _unflatten_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Unflatten a dotted-key dictionary into a nested structure.

    ``{"llm.default_model": "x"}`` -> ``{"llm": {"default_model": "x"}}``
    """
    result: Dict[str, Any] = {}
    for key, value in d.items():
        parts = key.split(".")
        current = result
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result
