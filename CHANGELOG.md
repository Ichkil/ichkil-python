# Changelog

All notable changes to the `ichkil` Python package are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-01

### Added

- Initial public release.
- `Diacritizer` — one-call Arabic Tashkeel backed by a ≈5 MB ONNX model
  (`input_ids` int64 `[batch, seq]` → `logits` float32 `[batch, seq, 13]`),
  downloaded from Hugging Face (`ichkil/ichkil`), SHA-256-verified, cached
  locally; `Diacritizer.from_file()` for fully offline use.
- Module-level convenience API: `ichkil.diacritize(text)`,
  `ichkil.predict_labels(text)`, `ichkil.reset_default()`.
- Pure text pipeline (`ichkil.pipeline`): `strip_diacritics`, `encode`,
  `decode`, `attach`, label tables (13 labels), tatweel normalization —
  shared with and kept in sync with the JS and Go runtimes.
- Typed API (`py.typed`), `IchkilError` hierarchy
  (`ChecksumError`, `ModelLoadError`, `InferenceError`).
- CLI: `ichkil` (positional text or stdin, `--model` offline mode,
  `--json`, `--repo`, `--revision`, `--version`).
- Cross-runtime golden vector suite (`tests/golden.json`, copied from the
  core repository) plus pure-pipeline and integration tests.
- CI: lint (ruff), test matrix Python 3.9 – 3.12, sdist + wheel build.
- PyPI publishing via GitHub trusted publishing (OIDC) on `v*` tags.
