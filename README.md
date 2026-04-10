<h1 align="center">anyai</h1>
<p align="center"><em>The unified gateway to the any* AI ecosystem — every AI task, one import.</em></p>

<p align="center">
<img src="https://img.shields.io/pypi/v/anyai.svg" alt="PyPI">
<img src="https://img.shields.io/pypi/pyversions/anyai.svg" alt="Python">
<img src="https://img.shields.io/pypi/l/anyai.svg" alt="License">
</p>

**anyai** is the umbrella meta-package for the `any*` ecosystem. It provides a single import and a unified one-liner API for computer vision, OCR, LLMs, NLP, tabular ML, and deployment. The core package ships with zero heavy dependencies and includes useful rule-based implementations out of the box, while every sister package (`anycv`, `anyocr`, `anyllm`, `anynlp`, `anyml`, `tableai`, `traincv`, `anydeploy`, `anyrobo`) can be unlocked as an optional extra.

Built by [Viet-Anh Nguyen](https://github.com/vietanhdev) at [NRL.ai](https://www.nrl.ai).

## Why anyai?

- **One-liner API** — Get started in 3 lines of code for any AI task
- **Plugin architecture** — Discovers and delegates to installed `any*` sister packages automatically
- **Local-first** — Built-in rule-based methods work offline with no downloads
- **Minimal core deps** — Only `pillow` and `pyyaml` required; heavy ML is optional
- **Production-ready** — Type hints, tests, dataclass result types, graceful degradation

## Installation

```bash
pip install anyai
```

For optional features:

```bash
pip install anyai[cv]        # + anycv (object detection, classification)
pip install anyai[ocr]       # + anyocr (text extraction from images)
pip install anyai[llm]       # + anyllm (LLM abstraction layer)
pip install anyai[nlp]       # + anynlp (NER, sentiment, summarization)
pip install anyai[ml]        # + anyml (AutoML for tabular data)
pip install anyai[table]     # + tableai (DataFrame profiling & cleaning)
pip install anyai[deploy]    # + anydeploy (ONNX/TFLite export + serving)
pip install anyai[robo]      # + anyrobo (voice agent framework)
pip install anyai[all]       # everything
```

**Python 3.8+ supported** (tested on 3.8, 3.9, 3.10, 3.11, 3.12, 3.13)

## Quick Start

```python
import anyai

# 1. Text summarization (built-in, zero deps — extractive sentence scoring)
summary = anyai.summarize(
    "Long article text here...",
    max_sentences=3,
)
print(summary)

# 2. Sentiment analysis (built-in AFINN-style lexicon with negation handling)
sentiment = anyai.sentiment("I absolutely love this library!")
print(sentiment.label, sentiment.score)   # "positive" 0.87

# 3. Keyword extraction (built-in TF-IDF-like scoring)
keywords = anyai.keywords("Machine learning is transforming software.", top_k=5)

# 4. Image metadata (built-in via Pillow)
info = anyai.image_info("photo.jpg")
print(info.width, info.height, info.format)

# 5. Delegated tasks (require sister packages)
labels = anyai.detect("photo.jpg")        # needs anyai[cv]
text   = anyai.ocr("scan.png")            # needs anyai[ocr]
reply  = anyai.chat("Explain RAG in 1 line.")  # needs anyai[llm]
```

## Models & Methods

### Built-in (zero-dependency) implementations

| Task | Method | Notes |
|---|---|---|
| `summarize` | Extractive scoring: sentence position + word-frequency + length penalty | Pure Python, no models downloaded |
| `sentiment` | AFINN-style lexicon lookup with negation windows (`not good` -> negative) | ~2,500 scored English words baked in |
| `keywords` | TF-IDF-like term weighting with English stopword removal | No external corpus required |
| `image_info` | Pillow `Image.open()` + EXIF parsing | Returns `ImageMetadata` dataclass |
| `pipeline` | Chain built-in and sister-package ops into a DAG | Lazy evaluation |
| `config` | YAML + environment variable resolver | Respects `ANYAI_*` env vars |

### Delegated backends (via optional extras)

When you call a task that requires a sister package, `anyai` dynamically imports the registered backend:

- `anyai.detect / classify / segment` -> [anycv](https://pypi.org/project/anycv) (YOLOv8 / MobileNetV2 / DeepLabV3 via ONNX Runtime)
- `anyai.ocr` -> [anyocr](https://pypi.org/project/anyocr) (Surya / EasyOCR / PaddleOCR / Tesseract / Vision-LLM)
- `anyai.chat / embed / tools` -> [anyllm](https://pypi.org/project/anyllm) (Ollama / llama.cpp / OpenAI / Anthropic / HF)
- `anyai.ner / classify_text` -> [anynlp](https://pypi.org/project/anynlp)
- `anyai.automl` -> [anyml](https://pypi.org/project/anyml) (sklearn + XGBoost/LightGBM)
- `anyai.profile_df / clean_df` -> [tableai](https://pypi.org/project/tableai)
- `anyai.export / serve` -> [anydeploy](https://pypi.org/project/anydeploy)

## API Reference

| Function | Purpose |
|---|---|
| `anyai.summarize(text, max_sentences=3)` | Extractive summary |
| `anyai.sentiment(text)` | `SentimentResult(label, score)` |
| `anyai.keywords(text, top_k=10)` | Ranked keyword list |
| `anyai.image_info(path)` | `ImageMetadata` dataclass |
| `anyai.detect(image, model="yolov8n")` | Object detection (requires `[cv]`) |
| `anyai.classify(image)` | Image classification (requires `[cv]`) |
| `anyai.ocr(image)` | Text extraction (requires `[ocr]`) |
| `anyai.chat(prompt, model="auto")` | LLM completion (requires `[llm]`) |
| `anyai.Pipeline([...])` | Chain tasks across packages |
| `anyai.Config.from_yaml(path)` | Load project-wide config |

## CLI Usage

```bash
anyai summarize article.txt --sentences 5
anyai sentiment "I really enjoyed the film"
anyai keywords document.txt --top 10
anyai info photo.jpg
anyai version
```

## Examples

### Build a pipeline that spans packages

```python
from anyai import Pipeline

# Each step delegates to the right sister package (if installed)
pipe = Pipeline([
    ("ocr",       {"backend": "auto"}),    # anyocr
    ("summarize", {"max_sentences": 3}),   # built-in
    ("sentiment", {}),                     # built-in
])

result = pipe.run("scanned_report.png")
print(result["summary"], result["sentiment"])
```

### Use config-driven defaults

```yaml
# anyai.yaml
llm:
  provider: ollama
  model: llama3.1:8b
cv:
  model: yolov8n
```

```python
import anyai

# All subsequent calls inherit these defaults
anyai.Config.load("anyai.yaml")
anyai.chat("Hello")   # routed to ollama/llama3.1:8b
```

### Graceful degradation

```python
import anyai

# If anyllm is not installed, fall back to the built-in extractive summary
try:
    summary = anyai.summarize_llm(long_text)   # abstractive (needs anyai[llm])
except anyai.BackendNotAvailable:
    summary = anyai.summarize(long_text)       # extractive fallback (built-in)
```

## License

MIT (c) Viet-Anh Nguyen
