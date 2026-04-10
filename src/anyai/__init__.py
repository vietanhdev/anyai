"""AnyAI - One-liner AI for everyone.

A unified gateway to AI capabilities. Works standalone with built-in
rule-based functionality and supports optional extras for advanced AI.
"""

import importlib
from typing import Dict

__version__ = "0.2.4"
__author__ = "Viet-Anh Nguyen"
__email__ = "vietanh.dev@gmail.com"
__url__ = "https://github.com/vietanhdev/anyai"

from anyai import models, text, image, utils
from anyai.config import Config
from anyai.core import about, available_backends, _proxies, _SUB_PACKAGES
from anyai.errors import (
    AnyAIError,
    BackendNotAvailableError,
    ModelNotFoundError,
    PrivacyModeError,
    ValidationError,
)
from anyai.logging import set_log_level, enable_debug, get_logger
from anyai.pipeline import pipeline, Pipeline, ParallelPipeline, PipelineStepError
from anyai.privacy import is_privacy_mode, check_privacy, warn_privacy

# Global configuration singleton
config = Config()

# Expose proxy functions at the top level.
detect = _proxies["detect"]
ocr = _proxies["ocr"]
chat = _proxies["chat"]
classify = _proxies["classify"]
profile = _proxies["profile"]
summarize = _proxies["summarize"]
deploy = _proxies["deploy"]


def version_info() -> Dict[str, str]:
    """Show versions of anyai and all installed sub-packages.

    Returns:
        A dictionary mapping package names to their version strings.
        Packages that are not installed are omitted.
    """
    info: Dict[str, str] = {"anyai": __version__}
    for _extra, pkg_name in _SUB_PACKAGES.items():
        try:
            mod = importlib.import_module(pkg_name)
            ver = getattr(mod, "__version__", "installed (unknown version)")
            info[pkg_name] = ver
        except ImportError:
            pass
    return info


__all__ = [
    "about",
    "available_backends",
    "config",
    "Config",
    "version_info",
    "pipeline",
    "Pipeline",
    "ParallelPipeline",
    "PipelineStepError",
    "detect",
    "ocr",
    "chat",
    "classify",
    "profile",
    "summarize",
    "deploy",
    "text",
    "image",
    "models",
    "utils",
    "set_log_level",
    "enable_debug",
    "get_logger",
    "AnyAIError",
    "BackendNotAvailableError",
    "ModelNotFoundError",
    "PrivacyModeError",
    "ValidationError",
    "is_privacy_mode",
    "check_privacy",
    "warn_privacy",
    "__version__",
]
