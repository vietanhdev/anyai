# Contributing to anyai

Thank you for considering contributing to anyai, the unified gateway for the AnyAI ecosystem! This guide will help you get started.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/vietanhdev/anyai.git
cd anyai

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install in development mode with all extras
pip install -e ".[all,dev]"

# Run tests
pytest
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=anyai

# Run specific test file
pytest tests/test_utils.py -v
```

All tests must pass without any optional sub-packages installed. Use mocks for any functionality that depends on optional packages.

## Project Structure

```
src/anyai/
    __init__.py       # Public API and re-exports
    cli.py            # Click-based CLI (anyai command)
    config.py         # Unified configuration system
    core.py           # Sub-package registry and proxy functions
    errors.py         # Common exception hierarchy
    image.py          # Built-in image utilities
    logging.py        # Logging configuration
    models.py         # Cross-package model registry
    pipeline.py       # Pipeline orchestration
    text.py           # Built-in text utilities (NLP)
    utils.py          # Shared helper functions
```

## Adding a New CLI Command

1. Add the command function to `src/anyai/cli.py` using the `@main.command()` decorator.
2. If the command requires a sub-package, use `_require_extra()` to provide a helpful install message.
3. Add tests in `tests/test_cli.py`.

## Adding Shared Utilities

1. Add the function to `src/anyai/utils.py`.
2. Export it from `src/anyai/__init__.py` if it is part of the public API.
3. Add tests in `tests/test_utils.py` or `tests/test_utils_extended.py`.

## Adding a New Sub-Package Integration

1. Add the package to `_SUB_PACKAGES` in `src/anyai/core.py`.
2. Add a proxy function in `_PROXY_REGISTRY` if the package exposes a one-liner.
3. Add the optional dependency to `pyproject.toml` under `[project.optional-dependencies]`.
4. Add a CLI command in `src/anyai/cli.py`.
5. Add tests for the CLI integration.

## Code Style

- We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting.
- Use type hints for all function signatures.
- Write docstrings for all public classes and methods.
- Keep line length under 100 characters.

```bash
# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/
```

## Pull Request Process

1. Fork the repository and create a feature branch.
2. Write tests for your changes.
3. Ensure all tests pass: `pytest`.
4. Ensure code passes linting: `ruff check src/ tests/`.
5. Update documentation if needed.
6. Submit a pull request with a clear description of your changes.

## Reporting Issues

Please use [GitHub Issues](https://github.com/vietanhdev/anyai/issues) to report bugs or request features. Include:

- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
