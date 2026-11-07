# ichkil — Arabic Tashkeel for Python

> Vocalize bare Arabic text with a tiny, self-contained ONNX model (≈22 MB, CPU-only,
> no GPU, no tokenizer, no server). One call, fully offline after the first run.

[![CI](https://github.com/Ichkil/ichkil-python/actions/workflows/ci.yml/badge.svg)](https://github.com/Ichkil/ichkil-python/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ichkil)](https://pypi.org/project/ichkil/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ichkil)](https://pypi.org/project/ichkil/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Hugging Face model](https://img.shields.io/badge/%F0%9F%A4%97_model-ichkil%2Fichkil-yellow)](https://huggingface.co/ichkil/ichkil)
[![Try it in your browser](https://img.shields.io/badge/%F0%9F%A4%97_try%20it-ichkil%2Fichkil%2Ddemo-blue)](https://huggingface.co/spaces/ichkil/ichkil-demo)

`ichkil` downloads its model from [Hugging Face](https://huggingface.co/ichkil/ichkil)
on first use, verifies its SHA-256, and runs inference with
[ONNX Runtime](https://onnxruntime.ai/). After that it is 100 % offline.

```python
import ichkil

ichkil.diacritize("محمد قرأ الكتاب في المدرسة")
# -> 'مُحَمَّدٌ قَرَأَ الْكِتَابَ فِي الْمَدْرَسَةِ'
```

## Install

```bash
pip install ichkil
# or
uv add ichkil
```

Requires Python ≥ 3.9 (tested on 3.9 – 3.12). Dependencies: `huggingface_hub`,
`onnxruntime` (CPU).

## Quickstart

```python
import ichkil

# one-liner (lazy model download on first call, then cached)
ichkil.diacritize("العلم نور والجهل ظلمة")
# -> 'الْعِلْمُ نُورٌ وَالْجَهْلُ ظُلْمَةٌ'

# explicit control
from ichkil import Diacritizer

d = Diacritizer()                      # downloads + verifies + caches the model
d.diacritize("القهوة جميلة في الصباح")
# -> 'الْقَهْوَةُ جَمِيلَةٌ فِي الصَّبَاحِ'

labels = d.predict_labels("كتاب")      # [4, 2, 1, 3]  (one label id per character, 0..15)

# fully offline: point at local files
d = Diacritizer.from_file("./model.onnx")
```

### Command line

```bash
pip install ichkil
ichkil "محمد قرأ الكتاب"
# مُحَمَّدٌ قَرَأَ الْكِتَابَ

ichkil --json "العلم نور"
# {"input": "العلم نور", "output": "الْعِلْمِ نُورٌ"}

cat text.txt | ichkil                  # read from stdin
ichkil --model ./model.onnx "محمد"    # offline mode
```

## API

| Item | Description |
| --- | --- |
| `ichkil.diacritize(text: str) -> str` | Vocalize text (module-level convenience, lazy default model). |
| `ichkil.predict_labels(text: str) -> list[int]` | Label id (0–15) per character. |
| `Diacritizer` | Constructor: `repo_id`, `revision` (default `"v0.3.0"`), `cache_dir`, `token`, `verify_checksum`, `num_threads`. |
| `Diacritizer.from_file(model_path, config_path=None)` | Build from local `model.onnx` (+ `config.json` beside it). |
| `Diacritizer.diacritize(text)` / `.predict_labels(text)` | Instance methods (same behavior). |
| `ichkil.pipeline` | Pure text helpers: `normalize_text`, `strip_diacritics`, `encode`, `decode`, `attach`, label tables. |
| Errors | `IchkilError` ← `ChecksumError`, `ModelLoadError`, `InferenceError`. |
| `ichkil.reset_default()` | Drop the lazily-created default instance (useful in tests). |

### Label contract

The model predicts exactly one label per input character:

| id | meaning | id | meaning |
| --- | --- | --- | --- |
| 0 | non-letter token / padding | 7 | shadda + fatha (ّـَ) |
| 1 | letter, no diacritic | 8 | shadda + damma (ّـُ) |
| 2 | fatha (ـَ) | 9 | shadda + kasra (ّـِ) |
| 3 | damma (ـُ) | 10 | tanwin fatha (ـً) |
| 4 | kasra (ـِ) | 11 | tanwin damma (ـٌ) |
| 5 | sukun (ـْ) | 12 | tanwin kasra (ـٍ) |
| 6 | shadda (ـّ) | 13 | shadda + tanwin fatha (ـًّ) |
| | | 14 | shadda + tanwin damma (ـٌّ) |
| | | 15 | shadda + tanwin kasra (ـٍّ) |

Input is normalized first (Unicode NFC, ٱ→ا, ی→ي, ک→ك, zero-width characters
removed, whitespace collapsed) and existing marks are stripped, so vocalized and
bare input give the same result. Labels predicted on non-letters are
unspecified and never attached.
Non-Arabic characters (Latin letters, digits, punctuation, spaces) pass through
unchanged.

## Model provenance

- Repository: [`ichkil/ichkil`](https://huggingface.co/ichkil/ichkil), pinned to the
  **`v0.3.0`** tag by default (`Diacritizer(revision="v0.1.0")` loads the previous
  13-label model).
- SHA-256 of `model.onnx` (v0.3.0): `9779e1aca89cebc77a9b92037b7d700c422cdf6c3774d4883f4652f641f3f705`
- Contract: `input_ids` int64 `[batch, seq]` → `logits` float32 `[batch, seq, 16]`;
  dynamic length, tested up to 1024 characters per call.
- Quality (v0.3.0): DER 2.32 % on Fadel et al. 2019 (classical), 2.14 % on WikiNews
  (modern Arabic); the ONNX artifact scores identically to the PyTorch reference.
- The cross-runtime golden vectors in `tests/golden.json` are a verbatim copy of the
  core repository's `golden/vectors.json`.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                   # pure pipeline tests (no network)
pytest -m integration                     # + model tests (downloads once, ~22 MB)
ruff check . && ruff format --check .    # lint
```

## Publishing

`v*` tags are built and published to PyPI via **GitHub trusted publishing (OIDC)** —
no API token is stored anywhere. One-time setup on
[pypi.org/project/ichkil](https://pypi.org/project/ichkil/): *Project settings →
Publishing → Add a publisher* with `github.com/Ichkil/ichkil-python`,
`workflow: publish.yml`.

## Sibling libraries

- [ichkil-js](https://github.com/Ichkil/ichkil-js) — npm package (Node.js + browser)
- [ichkil-go](https://github.com/Ichkil/ichkil-go) — Go module
- [Core repo](https://github.com/Ichkil/ichkil) — training, evaluation, model spec,
  golden vectors

## License

MIT — [LICENSE](LICENSE) · [github.com/Ichkil/ichkil-python](https://github.com/Ichkil/ichkil-python) · © 2026 Maaouia BenHamed
