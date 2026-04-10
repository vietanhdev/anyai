"""Helper utilities for AnyAI: dependency checking, configuration, etc."""

import hashlib
import importlib
import urllib.request
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse


def check_deps(module_name: str) -> bool:
    """Check whether an optional dependency is installed and importable.

    Args:
        module_name: The Python package/module name to check
            (e.g. ``'anycv'``, ``'torch'``, ``'PIL'``).

    Returns:
        ``True`` if the module can be imported, ``False`` otherwise.

    Examples:
        >>> check_deps("json")  # stdlib, always available
        True
        >>> check_deps("nonexistent_package_xyz")
        False
    """
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def require_deps(module_name: str, extra: Optional[str] = None) -> None:
    """Raise a helpful error if a required dependency is missing.

    Args:
        module_name: The Python package/module name to check.
        extra: The pip extra to suggest in the error message
            (e.g. ``'cv'`` suggests ``pip install anyai[cv]``).

    Raises:
        ImportError: If the module cannot be imported, with an
            installation hint in the message.
    """
    if not check_deps(module_name):
        hint = f"pip install anyai[{extra}]" if extra else f"pip install {module_name}"
        raise ImportError(
            f"'{module_name}' is required but not installed. "
            f"Install it with: {hint}"
        )


def get_version() -> str:
    """Return the current AnyAI version string.

    Returns:
        The version string, e.g. ``'0.1.0'``.
    """
    from anyai import __version__
    return __version__


# ---------------------------------------------------------------------------
# Cached file download
# ---------------------------------------------------------------------------

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "anyai"


def download_file(
    url: str,
    cache_dir: Optional[str] = None,
    filename: Optional[str] = None,
) -> Path:
    """Download a file with caching.

    If the file already exists in *cache_dir* it is returned immediately
    without re-downloading.

    Args:
        url: The URL to download.
        cache_dir: Directory to cache the downloaded file.  Defaults to
            ``~/.cache/anyai``.
        filename: Optional filename override.  If not given, derived from
            the URL via a hash to avoid collisions.

    Returns:
        Path to the cached file.

    Raises:
        ValueError: If the URL scheme is not http or https.
        OSError: If the download fails.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"Only http and https URLs are supported, got: {parsed.scheme!r}"
        )

    cache = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)

    if filename is None:
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        # Keep the original extension if present
        url_path = url.split("?")[0]
        suffix = Path(url_path).suffix or ""
        filename = f"{url_hash}{suffix}"

    dest = cache / filename
    if dest.exists():
        return dest

    # Download to a temporary file first, then rename for atomicity
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(url, str(tmp))
        tmp.rename(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    return dest


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------


def get_device() -> str:
    """Detect the best available compute device.

    Checks for CUDA, then Apple MPS, then falls back to CPU.

    Returns:
        One of ``'cuda'``, ``'mps'``, or ``'cpu'``.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass

    return "cpu"


# ---------------------------------------------------------------------------
# Memory checking
# ---------------------------------------------------------------------------


def check_memory() -> dict:
    """Check available system RAM and (optionally) GPU VRAM.

    Returns:
        A dict with keys:

        - ``ram_total`` -- total RAM in bytes
        - ``ram_available`` -- available RAM in bytes
        - ``vram_total`` -- total VRAM in bytes (0 if no GPU)
        - ``vram_available`` -- available VRAM in bytes (0 if no GPU)
    """
    info: dict = {
        "ram_total": 0,
        "ram_available": 0,
        "vram_total": 0,
        "vram_available": 0,
    }

    # System RAM via /proc/meminfo (Linux) or psutil
    try:
        import psutil

        mem = psutil.virtual_memory()
        info["ram_total"] = mem.total
        info["ram_available"] = mem.available
    except ImportError:
        # Fallback: read /proc/meminfo on Linux
        meminfo_path = Path("/proc/meminfo")
        if meminfo_path.exists():
            lines = meminfo_path.read_text().splitlines()
            for line in lines:
                if line.startswith("MemTotal:"):
                    info["ram_total"] = int(line.split()[1]) * 1024
                elif line.startswith("MemAvailable:"):
                    info["ram_available"] = int(line.split()[1]) * 1024

    # GPU VRAM via torch
    try:
        import torch

        if torch.cuda.is_available():
            info["vram_total"] = torch.cuda.get_device_properties(0).total_mem
            info["vram_available"] = info["vram_total"] - torch.cuda.memory_allocated(0)
    except (ImportError, Exception):
        pass

    return info


# ---------------------------------------------------------------------------
# Human-readable sizes
# ---------------------------------------------------------------------------


def format_size(size_bytes: int) -> str:
    """Format a byte count as a human-readable string.

    Args:
        size_bytes: Size in bytes (non-negative integer).

    Returns:
        A string like ``'1.23 GB'`` or ``'456 B'``.

    Examples:
        >>> format_size(0)
        '0 B'
        >>> format_size(1024)
        '1.00 KB'
        >>> format_size(1048576)
        '1.00 MB'
    """
    if size_bytes < 0:
        raise ValueError("size_bytes must be non-negative")
    if size_bytes == 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(size_bytes)
    for unit in units:
        if abs(size) < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} B"
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} {units[-1]}"  # pragma: no cover
