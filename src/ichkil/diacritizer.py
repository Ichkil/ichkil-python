"""The :class:`Diacritizer` — a ready-to-use Arabic diacritizer backed by ONNX."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from . import pipeline as P
from .download import (
    DEFAULT_REPO_ID,
    DEFAULT_REVISION,
    load_config,
    resolve_model,
)
from .errors import InferenceError, ModelLoadError

__all__ = ["Diacritizer"]


class Diacritizer:
    """Arabic Tashkeel (diacritization) with a one-call API.

    The model (``model.onnx`` + ``config.json``) is downloaded from Hugging
    Face on first use and cached under ``~/.cache/huggingface``; afterwards
    construction is offline.

    Parameters
    ----------
    repo_id:
        Hugging Face model repository (default ``"ichkil/ichkil"``).
    revision:
        Branch, tag or commit (default ``"main"``).
    cache_dir:
        Model cache directory (default: the standard HF cache).
    token:
        Optional HF token for gated/private repositories.
    verify_checksum:
        Verify the artifact against ``SHA256SUMS`` (default ``True``).
    num_threads:
        ONNX Runtime intra-op thread count (default: ORT heuristic).
    """

    def __init__(
        self,
        *,
        repo_id: str = DEFAULT_REPO_ID,
        revision: str = DEFAULT_REVISION,
        cache_dir: str | None = None,
        token: str | None = None,
        verify_checksum: bool = True,
        num_threads: int | None = None,
    ) -> None:
        bundle = resolve_model(
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            verify_checksum=verify_checksum,
        )
        self._init_from_paths(bundle.onnx_path, bundle.config, num_threads)
        self.repo_id = bundle.repo_id
        self.revision = bundle.revision

    @classmethod
    def from_file(
        cls,
        model_path: str | Path,
        config_path: str | Path | None = None,
        num_threads: int | None = None,
    ) -> Diacritizer:
        """Build a diacritizer from local ``model.onnx`` + ``config.json`` files.

        ``config_path`` defaults to ``config.json`` next to the model.
        """
        model_path = Path(model_path)
        if not model_path.is_file():
            raise ModelLoadError(f"model file not found: {model_path}")
        if config_path is None:
            config_path = model_path.parent / "config.json"
        config_path = Path(config_path)
        if not config_path.is_file():
            raise ModelLoadError(f"config file not found: {config_path}")
        config = load_config(config_path)
        inst = cls.__new__(cls)
        inst._init_from_paths(str(model_path), config, num_threads)
        inst.repo_id = "<local>"
        inst.revision = model_path.name
        return inst

    # ------------------------------------------------------------------ #
    def _init_from_paths(self, onnx_path: str, config: dict, num_threads: int | None) -> None:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if num_threads:
            options.intra_op_num_threads = int(num_threads)
        try:
            self._session = ort.InferenceSession(
                onnx_path, sess_options=options, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:  # noqa: BLE001 - wrap ORT errors in our error type
            raise ModelLoadError(f"failed to load ONNX model {onnx_path}: {exc}") from exc

        self._sym2id: dict = dict(config["sym2id"])
        self.num_labels: int = int(config["num_tags"])
        self.max_seq_len: int = int(config.get("max_seq_len", 0)) or 0
        self._unk_id: int = int(self._sym2id.get("<unk>", 1))

    # ------------------------------------------------------------------ #
    def predict_labels(self, text: str) -> list[int]:
        """Predicted label id (0..12) for every character of ``text``."""
        if text is None:
            raise TypeError("text must be a string")
        base = P.strip_diacritics(str(text))
        if not base:
            return []
        ids = P.encode(base, self._sym2id, self._unk_id)
        input_ids = np.asarray([ids], dtype=np.int64)
        try:
            (logits,) = self._session.run(["logits"], {"input_ids": input_ids})
        except Exception as exc:  # noqa: BLE001
            raise InferenceError(f"inference failed: {exc}") from exc
        flat = logits[0].reshape(-1).tolist()
        n = len(base)
        return [P.argmax(flat[i * self.num_labels : (i + 1) * self.num_labels]) for i in range(n)]

    def diacritize(self, text: str) -> str:
        """Turn bare (or already vocalized) Arabic into fully-vocalized text.

        >>> diacritize("محمد قرأ الكتاب")
        'مُحَمَّدٌ قَرَأَ الْكِتَابَ'
        """
        raw = "" if text is None else str(text)
        raw = raw.strip()
        if not raw:
            return ""
        base = P.strip_diacritics(raw)
        if not base:
            return raw
        labels = self.predict_labels(base)
        return P.attach(base, labels)

    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Diacritizer(repo_id={self.repo_id!r}, revision={self.revision!r}, "
            f"num_labels={self.num_labels})"
        )
