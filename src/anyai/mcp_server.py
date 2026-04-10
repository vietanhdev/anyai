"""MCP server exposing all AnyAI capabilities as tools for AI agents.

Start with::

    anyai mcp-server

Or programmatically::

    from anyai.mcp_server import create_server
    server = create_server()
    server.run()
"""

from __future__ import annotations

import importlib
import json
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP


def _try_import(package: str, extra: str):
    """Import a sub-package or raise a clear error."""
    try:
        return importlib.import_module(package)
    except ImportError:
        raise RuntimeError(
            f"'{package}' is not installed. "
            f"Install with: pip install anyai[{extra}]"
        )


def _serialize(obj: Any) -> str:
    """Convert an arbitrary result to a JSON string for MCP responses."""
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, indent=2, default=str)
    except (TypeError, ValueError):
        return str(obj)


def create_server() -> FastMCP:
    """Create and return the AnyAI MCP server with all tools registered."""
    mcp = FastMCP("anyai")

    # ------------------------------------------------------------------
    # Computer Vision
    # ------------------------------------------------------------------

    @mcp.tool()
    def detect_objects(
        image: str,
        model: Optional[str] = None,
    ) -> str:
        """Detect objects in an image.

        Args:
            image: Path to the image file.
            model: Optional model name (e.g. 'yolov8n').

        Returns:
            JSON string of detected objects with labels and bounding boxes.
        """
        mod = _try_import("anycv", "cv")
        kwargs: Dict[str, Any] = {}
        if model is not None:
            kwargs["model"] = model
        results = mod.detect(image, **kwargs)
        return _serialize(results)

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------

    @mcp.tool()
    def read_text(
        document: str,
        backend: Optional[str] = None,
    ) -> str:
        """Read text from an image or PDF using OCR.

        Args:
            document: Path to the image or PDF file.
            backend: Optional OCR backend (e.g. 'surya', 'easyocr').

        Returns:
            Extracted text content.
        """
        mod = _try_import("anyocr", "ocr")
        kwargs: Dict[str, Any] = {}
        if backend is not None:
            kwargs["backend"] = backend
        result = mod.read(document, **kwargs)
        return str(result)

    # ------------------------------------------------------------------
    # LLM Chat
    # ------------------------------------------------------------------

    @mcp.tool()
    def chat(
        message: str,
        model: Optional[str] = None,
    ) -> str:
        """Chat with an LLM.

        Args:
            message: The user message to send.
            model: Optional model identifier (e.g. 'ollama/llama3').

        Returns:
            The LLM response text.
        """
        mod = _try_import("anyllm", "llm")
        kwargs: Dict[str, Any] = {}
        if model is not None:
            kwargs["model"] = model
        response = mod.chat(message, **kwargs)
        return str(response)

    # ------------------------------------------------------------------
    # NLP
    # ------------------------------------------------------------------

    @mcp.tool()
    def summarize_text(
        text: str,
        num_sentences: int = 3,
    ) -> str:
        """Summarize text into key sentences.

        Args:
            text: The text to summarize.
            num_sentences: Number of sentences in the summary. Defaults to 3.

        Returns:
            A summarized version of the input text.
        """
        mod = _try_import("anynlp", "nlp")
        result = mod.summarize(text, num_sentences=num_sentences)
        return str(result)

    @mcp.tool()
    def analyze_sentiment(
        text: str,
    ) -> str:
        """Analyze the sentiment of text.

        Args:
            text: The text to analyze.

        Returns:
            JSON string with 'label' (positive/negative/neutral) and 'score'.
        """
        mod = _try_import("anynlp", "nlp")
        result = mod.sentiment(text)
        return _serialize(result)

    # ------------------------------------------------------------------
    # Table / Data
    # ------------------------------------------------------------------

    @mcp.tool()
    def profile_data(
        data: str,
    ) -> str:
        """Profile a CSV dataset, returning statistics and metadata.

        Args:
            data: Path to a CSV file.

        Returns:
            JSON string of the data profile.
        """
        mod = _try_import("tableai", "table")
        result = mod.profile(data)
        return _serialize(result)

    # ------------------------------------------------------------------
    # ML Classification
    # ------------------------------------------------------------------

    @mcp.tool()
    def classify_data(
        data: str,
        target: str,
    ) -> str:
        """Auto-train and classify tabular data.

        Args:
            data: Path to a CSV file.
            target: Name of the target column.

        Returns:
            JSON string of classification results including score.
        """
        mod = _try_import("anyml", "ml")
        result = mod.classify(data, target=target)
        return _serialize(result)

    return mcp


def main(transport: str = "stdio") -> None:
    """Entry point for the MCP server."""
    server = create_server()
    server.run(transport=transport)


if __name__ == "__main__":
    main()
