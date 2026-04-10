"""Core utilities shared across AnyAI modules."""

import importlib
import sys
from typing import Any, Callable, Dict, List, Tuple

from anyai.utils import check_deps

# Registry of known sub-packages and their import names.
_SUB_PACKAGES: Dict[str, str] = {
    "cv": "anycv",
    "ocr": "anyocr",
    "llm": "anyllm",
    "ml": "anyml",
    "nlp": "anynlp",
    "deploy": "anydeploy",
    "table": "tableai",
    "robo": "anyrobo",
    "voice": "anyrobo",
    "traincv": "traincv",
}

# Maps proxy function name -> (extra name, package name, callable attr)
_PROXY_REGISTRY: Dict[str, Tuple[str, str, str]] = {
    "detect": ("cv", "anycv", "detect"),
    "ocr": ("ocr", "anyocr", "read"),
    "chat": ("llm", "anyllm", "chat"),
    "classify": ("ml", "anyml", "classify"),
    "profile": ("table", "tableai", "profile"),
    "summarize": ("nlp", "anynlp", "summarize"),
    "deploy": ("deploy", "anydeploy", "export"),
}


def _make_proxy(name: str, extra: str, package: str, attr: str) -> Callable:
    """Create a lazy proxy function that delegates to a sub-package."""

    def proxy(*args: Any, **kwargs: Any) -> Any:
        try:
            mod = importlib.import_module(package)
        except ImportError:
            raise ImportError(
                f"'{package}' is required for anyai.{name}() but not installed. "
                f"Install with: pip install anyai[{extra}]"
            )
        fn = getattr(mod, attr, None)
        if fn is None:
            raise AttributeError(
                f"Module '{package}' has no attribute '{attr}'"
            )
        return fn(*args, **kwargs)

    proxy.__name__ = name
    proxy.__qualname__ = f"anyai.{name}"
    proxy.__doc__ = (
        f"Proxy to {package}.{attr}(). "
        f"Requires: pip install anyai[{extra}]"
    )
    return proxy


# Build all proxy functions.
_proxies: Dict[str, Callable] = {}
for _name, (_extra, _pkg, _attr) in _PROXY_REGISTRY.items():
    _proxies[_name] = _make_proxy(_name, _extra, _pkg, _attr)


def about() -> str:
    """Print package information and return it as a string.

    Displays the AnyAI version, author information, available built-in
    modules, and the status of optional sub-packages.

    Returns:
        A formatted string containing package information.
    """
    from anyai import __version__, __author__, __email__, __url__

    backends = available_backends()
    installed = [b for b, status in backends.items() if status]
    missing = [b for b, status in backends.items() if not status]

    lines = [
        f"AnyAI v{__version__}",
        f"Author: {__author__} <{__email__}>",
        f"URL: {__url__}",
        "",
        "Built-in modules: text, image, utils",
        "",
        f"Installed extras: {', '.join(installed) if installed else 'none'}",
        f"Available extras: {', '.join(missing) if missing else 'all installed'}",
    ]
    info = "\n".join(lines)
    print(info)
    return info


def available_backends() -> Dict[str, bool]:
    """List the installation status of all optional sub-packages.

    Returns:
        A dictionary mapping extra name (e.g. ``'cv'``, ``'llm'``) to a
        boolean indicating whether the corresponding package is importable.
    """
    result: Dict[str, bool] = {}
    for extra_name, package_name in _SUB_PACKAGES.items():
        result[extra_name] = check_deps(package_name)
    return result
