"""Model acquisition: download (once) from Hugging Face, then use the local cache.

Files fetched from the model repository (``ichkil/ichkil`` by default):

* ``model.onnx``   — the ONNX artifact (int64 ``input_ids`` -> float32 ``logits``)
* ``config.json``  — vocabulary (``sym2id``), ``num_tags``, ``max_seq_len``
* ``SHA256SUMS``   — expected checksum of ``model.onnx`` (verified by default)

Everything is cached under ``~/.cache/huggingface`` (or a custom ``cache_dir``),
so only the first call per revision touches the network.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

from .errors import ChecksumError, ModelLoadError

__all__ = [
    "DEFAULT_REPO_ID",
    "DEFAULT_REVISION",
    "ModelBundle",
    "load_config",
    "resolve_model",
    "sha256_of_file",
    "verify_checksum",
]

DEFAULT_REPO_ID = "ichkil/ichkil"
DEFAULT_REVISION = "main"

_SHA256_BLOCK_SIZE = 1 << 20


@dataclass(frozen=True)
class ModelBundle:
    """A resolved, verified local model artifact plus its runtime config."""

    onnx_path: str
    config: dict
    repo_id: str
    revision: str


def sha256_of_file(path: str | Path, chunk_size: int = _SHA256_BLOCK_SIZE) -> str:
    """Hex SHA-256 digest of a local file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def _verify_file(path: str | Path, expected_sha256: str) -> None:
    """Raise :class:`ChecksumError` if the file's SHA-256 differs from ``expected_sha256``."""
    actual = sha256_of_file(path)
    if actual.lower() != expected_sha256.strip().lower():
        raise ChecksumError(
            f"model checksum mismatch for {path}: expected {expected_sha256.strip().lower()}, "
            f"got {actual}. Delete the local cache file or pass verify_checksum=False to bypass."
        )


def verify_checksum(path: str | Path, expected_sha256: str) -> None:
    """Public API: verify a local file against an expected SHA-256.

    Raises
    ------
    ChecksumError
        If the file's SHA-256 differs from ``expected_sha256``.
    """
    _verify_file(path, expected_sha256)


def load_config(config_path: str | Path) -> dict:
    """Load and minimally validate a ``config.json``."""
    with open(config_path, encoding="utf-8") as fh:
        config = json.load(fh)
    for key in ("sym2id", "num_tags"):
        if key not in config:
            raise ModelLoadError(
                f"config.json is missing required key '{key}' (from {config_path})"
            )
    return config


def _parse_sha256_sums(text: str, filename: str) -> str | None:
    """Extract the expected SHA-256 for ``filename`` from a ``SHA256SUMS`` document."""
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[-1].strip().lstrip("*") == filename:
            return parts[0]
    return None


def resolve_model(
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
    cache_dir: str | None = None,
    token: str | None = None,
    verify_checksum: bool = True,
) -> ModelBundle:
    """Download (or reuse the cached) model files from Hugging Face.

    Parameters
    ----------
    repo_id:
        Hugging Face model repository (default ``"ichkil/ichkil"``).
    revision:
        Branch, tag or commit to fetch (default ``"main"``).
    cache_dir:
        Cache directory; defaults to the standard HF cache
        (``~/.cache/huggingface``).
    token:
        Optional Hugging Face token (only needed for gated / private repos).
    verify_checksum:
        Verify ``model.onnx`` against ``SHA256SUMS`` (default ``True``).
    """
    common = {
        "repo_id": repo_id,
        "repo_type": "model",
        "revision": revision,
        "token": token,
        "cache_dir": cache_dir,
    }
    onnx_path = hf_hub_download(filename="model.onnx", **common)
    config_path = hf_hub_download(filename="config.json", **common)

    config = load_config(config_path)

    expected: str | None = None
    if verify_checksum:
        try:
            sums_path = hf_hub_download(filename="SHA256SUMS", **common)
            expected = _parse_sha256_sums(Path(sums_path).read_text(encoding="utf-8"), "model.onnx")
        except Exception:  # noqa: BLE001 - SHA256SUMS may be absent on some revisions
            expected = None
        if expected is None:
            expected = config.get("sha256")
        if expected is None:
            raise ModelLoadError(
                "cannot verify model checksum: the repository has neither SHA256SUMS "
                "nor config.sha256 (pass verify_checksum=False to bypass)"
            )
        _verify_file(onnx_path, expected)

    return ModelBundle(
        onnx_path=str(onnx_path),
        config=config,
        repo_id=repo_id,
        revision=revision,
    )
