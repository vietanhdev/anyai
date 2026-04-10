"""Cross-package model registry for the AnyAI ecosystem.

Provides a central view of all cached models across all sub-packages so
users can list, inspect, and clear model caches from one place.

Usage::

    from anyai import models

    # Show every cached model file across all registered packages
    for entry in models.list():
        print(entry["package"], entry["path"], entry["size"])

    # Total cache footprint
    print(models.total_size())

    # Wipe everything
    models.clear()
"""

from __future__ import annotations

import importlib
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from anyai.utils import format_size

# ---------------------------------------------------------------------------
# Registry of per-package cache directories
# ---------------------------------------------------------------------------

_registry: Dict[str, Path] = {}

# Default cache root -- packages that do not explicitly register will be
# looked up under  ~/.cache/anyai/<package_name>
_DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "anyai"


def register(package_name: str, cache_dir: str | Path) -> None:
    """Register a package's model cache directory.

    Args:
        package_name: Short name of the package (e.g. ``'anycv'``).
        cache_dir: Absolute path to the cache directory.
    """
    _registry[package_name] = Path(cache_dir)


def _known_dirs() -> Dict[str, Path]:
    """Return all known cache directories (registered + defaults)."""
    from anyai.core import _SUB_PACKAGES

    dirs: Dict[str, Path] = dict(_registry)
    for _extra, pkg_name in _SUB_PACKAGES.items():
        if pkg_name not in dirs:
            default = _DEFAULT_CACHE_ROOT / pkg_name
            dirs[pkg_name] = default
    # Include the core anyai cache
    if "anyai" not in dirs:
        dirs["anyai"] = _DEFAULT_CACHE_ROOT
    return dirs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list() -> List[Dict[str, Any]]:
    """List all cached model files across every registered package.

    Returns:
        A list of dicts, each with keys ``package``, ``path`` (str),
        ``size`` (int, bytes), and ``size_human`` (str).
    """
    entries: List[Dict[str, Any]] = []
    for pkg, cache_dir in _known_dirs().items():
        if not cache_dir.is_dir():
            continue
        for f in sorted(cache_dir.rglob("*")):
            if f.is_file():
                sz = f.stat().st_size
                entries.append({
                    "package": pkg,
                    "path": str(f),
                    "size": sz,
                    "size_human": format_size(sz),
                })
    return entries


def clear() -> int:
    """Remove all cached model files across every registered package.

    Returns:
        The number of directories that were cleared.
    """
    cleared = 0
    for _pkg, cache_dir in _known_dirs().items():
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir)
            cleared += 1
    return cleared


def total_size() -> int:
    """Return the total size (in bytes) of all cached model files."""
    total = 0
    for entry in list():
        total += entry["size"]
    return total


def download_all(
    on_progress: Optional[Callable[[str, str], None]] = None,
) -> List[str]:
    """Download all default models for offline use.

    Each sub-package may expose a ``download_defaults()`` function that
    fetches its standard models.  This helper calls them all.

    Args:
        on_progress: Optional callback ``(package_name, message)`` for
            progress reporting.

    Returns:
        List of package names that successfully downloaded models.
    """
    from anyai.core import _SUB_PACKAGES

    downloaded: List[str] = []
    for _extra, pkg_name in _SUB_PACKAGES.items():
        try:
            mod = importlib.import_module(pkg_name)
        except ImportError:
            continue

        fn = getattr(mod, "download_defaults", None)
        if fn is None:
            continue

        if on_progress:
            on_progress(pkg_name, "downloading default models...")
        try:
            fn()
            downloaded.append(pkg_name)
            if on_progress:
                on_progress(pkg_name, "done")
        except Exception:
            if on_progress:
                on_progress(pkg_name, "failed")
    return downloaded
