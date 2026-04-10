"""AnyAI unified CLI -- one command for all AI capabilities."""

import importlib
import json
import sys
from typing import Optional

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from anyai import __version__
from anyai.core import _SUB_PACKAGES, _PROXY_REGISTRY

console = Console()
error_console = Console(stderr=True)


def _install_hint(extra: str) -> str:
    return f"pip install anyai[{extra}]"


def _require_extra(package: str, extra: str):
    """Try to import *package*; on failure print a rich error and exit."""
    try:
        return importlib.import_module(package)
    except ImportError:
        hint = _install_hint(extra)
        error_console.print(
            f"[bold red]Missing dependency:[/bold red] {package} is not installed.\n"
            f"Install it with: [bold cyan]{hint.replace('[', '\\[')}[/bold cyan]"
        )
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Main group
# ---------------------------------------------------------------------------

@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="anyai")
@click.pass_context
def main(ctx):
    """AnyAI - One-liner AI for everyone."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# ---------------------------------------------------------------------------
# Computer Vision commands
# ---------------------------------------------------------------------------

@main.command()
@click.argument("image")
@click.option("--model", default=None, help="Model name (e.g. yolov8n).")
@click.option("--output", "-o", default=None, help="Save results to file.")
@click.option(
    "--format", "fmt",
    type=click.Choice(["json", "table", "text"]),
    default="table",
    help="Output format.",
)
def detect(image, model, output, fmt):
    """Detect objects in an image."""
    mod = _require_extra("anycv", "cv")
    kwargs = {}
    if model is not None:
        kwargs["model"] = model
    results = mod.detect(image, **kwargs)
    _render_results(results, fmt, output, title="Detection Results")


@main.command()
@click.argument("image")
@click.option("--output", "-o", default=None, help="Save results to file.")
@click.option(
    "--format", "fmt",
    type=click.Choice(["json", "table", "text"]),
    default="table",
    help="Output format.",
)
def classify(image, output, fmt):
    """Classify an image."""
    mod = _require_extra("anyml", "ml")
    results = mod.classify(image)
    _render_results(results, fmt, output, title="Classification Results")


@main.command()
@click.argument("image")
@click.option("--output", "-o", default=None, help="Save results to file.")
@click.option(
    "--format", "fmt",
    type=click.Choice(["json", "table", "text"]),
    default="table",
    help="Output format.",
)
def segment(image, output, fmt):
    """Segment objects in an image."""
    mod = _require_extra("anycv", "cv")
    results = mod.segment(image)
    _render_results(results, fmt, output, title="Segmentation Results")


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

@main.command()
@click.argument("document")
@click.option("--backend", default=None, help="OCR backend (e.g. surya).")
@click.option("--output", "-o", default=None, help="Save results to file.")
def ocr(document, backend, output):
    """Read text from an image or PDF."""
    mod = _require_extra("anyocr", "ocr")
    kwargs = {}
    if backend is not None:
        kwargs["backend"] = backend
    text = mod.read(document, **kwargs)
    if output:
        _write_output(text, output)
    else:
        console.print(Panel(str(text), title="OCR Result", border_style="green"))


# ---------------------------------------------------------------------------
# LLM Chat
# ---------------------------------------------------------------------------

@main.command()
@click.argument("message", required=False)
@click.option("--model", default=None, help="Model identifier (e.g. ollama/llama3).")
@click.option("--stream", "stream_mode", is_flag=True, help="Enable streaming output.")
@click.option("-i", "interactive", is_flag=True, help="Interactive chat mode (REPL).")
def chat(message, model, stream_mode, interactive):
    """Chat with an LLM."""
    mod = _require_extra("anyllm", "llm")

    if interactive:
        _interactive_chat(mod, model)
        return

    if not message:
        error_console.print(
            "[bold red]Error:[/bold red] Provide a message or use [bold]-i[/bold] for interactive mode."
        )
        raise SystemExit(1)

    kwargs = {}
    if model is not None:
        kwargs["model"] = model
    if stream_mode:
        kwargs["stream"] = True

    response = mod.chat(message, **kwargs)
    console.print(Markdown(str(response)))


def _interactive_chat(mod, model: Optional[str]):
    """Run an interactive chat REPL."""
    model_name = model or "default"
    console.print(
        Panel(
            f"AnyAI Chat (model: {model_name}) | Type [bold]quit[/bold] to exit",
            border_style="blue",
        )
    )

    kwargs = {}
    if model is not None:
        kwargs["model"] = model

    while True:
        try:
            user_input = console.input("[bold green]> [/bold green]")
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye!")
            break

        stripped = user_input.strip()
        if stripped.lower() in ("quit", "exit"):
            console.print("Bye!")
            break

        if stripped.startswith("/model "):
            model = stripped.split(" ", 1)[1].strip()
            kwargs["model"] = model
            console.print(f"Switched to [bold cyan]{model}[/bold cyan]")
            continue

        if not stripped:
            continue

        try:
            response = mod.chat(stripped, **kwargs)
            console.print(Markdown(str(response)))
        except Exception as exc:
            error_console.print(f"[red]Error: {exc}[/red]")


# ---------------------------------------------------------------------------
# NLP commands
# ---------------------------------------------------------------------------

@main.command()
@click.argument("text")
@click.option("--sentences", "-n", default=3, help="Number of summary sentences.")
def summarize(text, sentences):
    """Summarize text."""
    from anyai.text import summarize as _summarize

    result = _summarize(text, num_sentences=sentences)
    console.print(Panel(result, title="Summary", border_style="cyan"))


@main.command()
@click.argument("text")
def sentiment(text):
    """Analyse sentiment of text."""
    from anyai.text import sentiment as _sentiment

    result = _sentiment(text)
    label = result["label"]
    score = result["score"]
    color = {"positive": "green", "negative": "red", "neutral": "yellow"}.get(label, "white")
    table = Table(title="Sentiment Analysis", show_header=True)
    table.add_column("Label", style=f"bold {color}")
    table.add_column("Confidence", justify="right")
    table.add_row(label.capitalize(), f"{score:.2f}")
    console.print(table)


@main.command()
@click.argument("text")
def entities(text):
    """Extract named entities from text."""
    mod = _require_extra("anynlp", "nlp")
    results = mod.entities(text)
    _render_results(results, "table", None, title="Named Entities")


@main.command()
@click.argument("text")
@click.option("--top", "-n", default=10, help="Number of keywords.")
def keywords(text, top):
    """Extract keywords from text."""
    from anyai.text import extract_keywords

    kws = extract_keywords(text, top_n=top)
    table = Table(title="Keywords", show_header=True)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Keyword", style="bold cyan")
    for i, kw in enumerate(kws, 1):
        table.add_row(str(i), kw)
    console.print(table)


# ---------------------------------------------------------------------------
# ML commands
# ---------------------------------------------------------------------------

@main.command()
@click.argument("data")
@click.option("--target", required=True, help="Target column name.")
def train(data, target):
    """Auto-train an ML model on a CSV dataset."""
    mod = _require_extra("anyml", "ml")
    result = mod.train(data, target=target)
    _render_results(result, "table", None, title="Training Results")


@main.command()
@click.argument("model_path")
@click.argument("data")
@click.option("--output", "-o", default=None, help="Save predictions to file.")
def predict(model_path, data, output):
    """Predict with a trained model."""
    mod = _require_extra("anyml", "ml")
    results = mod.predict(model_path, data)
    _render_results(results, "table", output, title="Predictions")


# ---------------------------------------------------------------------------
# Table / Data commands
# ---------------------------------------------------------------------------

@main.command()
@click.argument("data")
def profile(data):
    """Profile a CSV dataset."""
    mod = _require_extra("tableai", "table")
    results = mod.profile(data)
    _render_results(results, "table", None, title="Data Profile")


@main.command()
@click.argument("data")
@click.option("--output", "-o", default=None, help="Save cleaned data to file.")
def clean(data, output):
    """Auto-clean a CSV dataset."""
    mod = _require_extra("tableai", "table")
    results = mod.clean(data)
    _render_results(results, "table", output, title="Cleaned Data")


# ---------------------------------------------------------------------------
# Deployment commands
# ---------------------------------------------------------------------------

@main.command(name="export")
@click.argument("model_path")
@click.option("--format", "fmt", default="onnx", help="Export format (e.g. onnx).")
@click.option("--output", "-o", default=None, help="Output file path.")
def export_model(model_path, fmt, output):
    """Export a model to a deployment format."""
    mod = _require_extra("anydeploy", "deploy")
    result = mod.export(model_path, format=fmt, output=output)
    console.print(f"[green]Model exported successfully.[/green]")
    if result:
        console.print(str(result))


@main.command()
@click.argument("model_path")
@click.option("--host", default="0.0.0.0", help="Host to bind.")
@click.option("--port", default=8000, type=int, help="Port to bind.")
def serve(model_path, host, port):
    """Serve a model via HTTP."""
    mod = _require_extra("anydeploy", "deploy")
    console.print(f"Serving [bold]{model_path}[/bold] on {host}:{port} ...")
    mod.serve(model_path, host=host, port=port)


@main.command()
@click.argument("package", required=False)
@click.option("--model", default=None, help="Model name for CV benchmark.")
@click.option("--models", default=None, help="Comma-separated model list for LLM benchmark.")
@click.option("--backends", default=None, help="Comma-separated backend list for OCR benchmark.")
@click.option("--num-images", default=50, type=int, help="Number of images for CV benchmark.")
def benchmark(package, model, models, backends, num_images):
    """Benchmark a package's performance.

    Usage: anyai benchmark [cv|llm|ocr]
    """
    if package is None:
        error_console.print(
            "[bold red]Error:[/bold red] Specify a package to benchmark: "
            "[bold]cv[/bold], [bold]llm[/bold], or [bold]ocr[/bold].\n"
            "Example: anyai benchmark cv"
        )
        raise SystemExit(1)

    package = package.lower()

    if package in ("cv", "anycv"):
        mod = _require_extra("anycv", "cv")
        kwargs = {"num_images": num_images}
        if model is not None:
            kwargs["model"] = model
        result = mod.benchmark(**kwargs)
        console.print(str(result))
        _render_results(result.to_dict(), "table", None, title="CV Benchmark Results")

    elif package in ("llm", "anyllm"):
        mod = _require_extra("anyllm", "llm")
        kwargs = {}
        if models is not None:
            kwargs["models"] = [m.strip() for m in models.split(",")]
        result = mod.benchmark(**kwargs)
        console.print(str(result))

    elif package in ("ocr", "anyocr"):
        mod = _require_extra("anyocr", "ocr")
        kwargs = {}
        if backends is not None:
            kwargs["backends"] = [b.strip() for b in backends.split(",")]
        result = mod.benchmark(**kwargs)
        console.print(str(result))

    else:
        error_console.print(
            f"[bold red]Error:[/bold red] Unknown package '{package}'. "
            "Supported: cv, llm, ocr."
        )
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Meta commands
# ---------------------------------------------------------------------------

@main.command()
def info():
    """Show installed AnyAI packages and status."""
    from anyai import version_info as _version_info

    versions = _version_info()
    table = Table(title="AnyAI Package Info", show_header=True, border_style="blue")
    table.add_column("Package", style="bold")
    table.add_column("Version", justify="right")
    for pkg, ver in versions.items():
        table.add_row(pkg, ver)
    console.print(table)


@main.command(name="mcp-server")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="MCP transport type.",
)
def mcp_server(transport):
    """Start the AnyAI MCP tool server for AI agents."""
    from anyai.mcp_server import main as mcp_main

    mcp_main(transport=transport)


@main.command()
def doctor():
    """Check system health: Python, devices, packages, disk, and memory."""
    import shutil
    import platform
    from pathlib import Path

    from anyai.core import available_backends
    from anyai.utils import get_device, check_memory, format_size

    # --- System overview table ---
    sys_table = Table(
        title="AnyAI System Health",
        show_header=True,
        border_style="blue",
        title_style="bold blue",
    )
    sys_table.add_column("Check", style="bold")
    sys_table.add_column("Status", justify="center")
    sys_table.add_column("Details")

    # Python version
    py_ver = platform.python_version()
    py_ok = sys.version_info >= (3, 8)
    sys_table.add_row(
        "Python version",
        Text("OK", style="bold green") if py_ok else Text("WARN", style="bold yellow"),
        f"{py_ver} ({'compatible' if py_ok else 'requires >= 3.8'})",
    )

    # Compute device
    device = get_device()
    device_style = {
        "cuda": "bold green",
        "mps": "bold green",
        "cpu": "bold yellow",
    }.get(device, "bold yellow")
    sys_table.add_row(
        "Compute device",
        Text(device.upper(), style=device_style),
        {
            "cuda": "NVIDIA GPU detected",
            "mps": "Apple Metal GPU detected",
            "cpu": "No GPU detected (CPU only)",
        }.get(device, device),
    )

    # Memory
    mem = check_memory()
    ram_avail = format_size(mem["ram_available"])
    ram_total = format_size(mem["ram_total"])
    ram_ok = mem["ram_available"] > 512 * 1024 * 1024  # > 512 MB
    sys_table.add_row(
        "System memory",
        Text("OK", style="bold green") if ram_ok else Text("LOW", style="bold red"),
        f"{ram_avail} available / {ram_total} total",
    )

    if mem["vram_total"] > 0:
        vram_avail = format_size(mem["vram_available"])
        vram_total = format_size(mem["vram_total"])
        sys_table.add_row(
            "GPU memory",
            Text("OK", style="bold green"),
            f"{vram_avail} available / {vram_total} total",
        )

    # Disk space for model cache
    cache_dir = str(Path.home() / ".cache" / "anyai")
    try:
        disk = shutil.disk_usage(Path.home())
        disk_free = format_size(disk.free)
        disk_total = format_size(disk.total)
        disk_ok = disk.free > 1024 * 1024 * 1024  # > 1 GB
        sys_table.add_row(
            "Disk (model cache)",
            Text("OK", style="bold green") if disk_ok else Text("LOW", style="bold red"),
            f"{disk_free} free / {disk_total} total ({cache_dir})",
        )
    except Exception:
        sys_table.add_row(
            "Disk (model cache)",
            Text("??", style="bold yellow"),
            "Unable to check disk space",
        )

    # Ollama server status
    ollama_ok = False
    ollama_detail = "Not reachable"
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                ollama_ok = True
                ollama_detail = "Running on localhost:11434"
    except Exception:
        pass
    sys_table.add_row(
        "Ollama server",
        Text("OK", style="bold green") if ollama_ok else Text("OFF", style="dim"),
        ollama_detail,
    )

    console.print(sys_table)
    console.print()

    # --- Installed backends table ---
    backends = available_backends()

    pkg_table = Table(
        title="Installed Backends",
        show_header=True,
        border_style="blue",
        title_style="bold blue",
    )
    pkg_table.add_column("Package", style="bold")
    pkg_table.add_column("Status", justify="center")
    pkg_table.add_column("Version", justify="right")
    pkg_table.add_column("Install", style="dim")

    for extra, pkg_name in _SUB_PACKAGES.items():
        installed = backends.get(extra, False)
        if installed:
            try:
                mod = importlib.import_module(pkg_name)
                ver = getattr(mod, "__version__", "unknown")
            except Exception:
                ver = "unknown"
            status = Text("installed", style="bold green")
        else:
            ver = "-"
            status = Text("missing", style="bold red")
        install_cmd = f"pip install anyai\\[{extra}]"
        pkg_table.add_row(pkg_name, status, ver, install_cmd)

    console.print(pkg_table)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render_results(results, fmt: str, output: Optional[str], *, title: str = "Results"):
    """Render results in the requested format."""
    if fmt == "json":
        text = json.dumps(results, indent=2, default=str)
        if output:
            _write_output(text, output)
        else:
            console.print_json(text)
    elif fmt == "text":
        text = str(results)
        if output:
            _write_output(text, output)
        else:
            console.print(text)
    else:
        # table (default)
        if isinstance(results, dict):
            table = Table(title=title, show_header=True)
            table.add_column("Key", style="bold cyan")
            table.add_column("Value")
            for k, v in results.items():
                table.add_row(str(k), str(v))
            console.print(table)
        elif isinstance(results, (list, tuple)):
            if results and isinstance(results[0], dict):
                table = Table(title=title, show_header=True)
                cols = list(results[0].keys())
                for c in cols:
                    table.add_column(str(c))
                for row in results:
                    table.add_row(*(str(row.get(c, "")) for c in cols))
                console.print(table)
            else:
                for item in results:
                    console.print(str(item))
        else:
            console.print(Panel(str(results), title=title))

        if output:
            _write_output(json.dumps(results, indent=2, default=str), output)


def _write_output(content: str, path: str):
    """Write content to a file and confirm."""
    with open(path, "w") as f:
        f.write(content)
    console.print(f"[green]Output saved to {path}[/green]")


if __name__ == "__main__":
    main()
