"""ichkil — Arabic Tashkeel (diacritization) in Python.

A tiny, self-contained ONNX model (≈5 MB, CPU-only) that assigns harakat,
shadda and tanwin to bare Arabic text. The model is downloaded from
Hugging Face on first use and cached locally.

Quickstart::

    import ichkil

    ichkil.diacritize("محمد قرأ الكتاب")
    # -> 'مُحَمَّدٌ قَرَأَ الْكِتَابَ'

Or, for explicit control::

    from ichkil import Diacritizer

    d = Diacritizer()                       # downloads + caches the model
    d.diacritize("الكتاب على الطاولة")

    d = Diacritizer.from_file("model.onnx")  # fully offline
"""

from __future__ import annotations

from typing import Callable

from .diacritizer import Diacritizer
from .download import DEFAULT_REPO_ID, DEFAULT_REVISION
from .errors import ChecksumError, IchkilError, InferenceError, ModelLoadError
from .pipeline import (
    LABEL_NON_LETTER,
    LABEL_NONE,
    NUM_LABELS,
    attach,
    combo_to_label,
    decode,
    encode,
    is_arabic_letter,
    label_to_combo,
    strip_diacritics,
)
from .version import __version__

__all__ = [
    "Diacritizer",
    "diacritize",
    "predict_labels",
    "IchkilError",
    "ChecksumError",
    "ModelLoadError",
    "InferenceError",
    "DEFAULT_REPO_ID",
    "DEFAULT_REVISION",
    "NUM_LABELS",
    "LABEL_NON_LETTER",
    "LABEL_NONE",
    "label_to_combo",
    "combo_to_label",
    "strip_diacritics",
    "encode",
    "decode",
    "attach",
    "is_arabic_letter",
    "__version__",
]

# Module-level convenience wrapper: one lazy default Diacritizer shared by all
# calls. Explicit construction (``Diacritizer(...)``) is always available for
# advanced use (pinned revisions, offline files, thread tuning).
_default: Diacritizer | None = None


def _default_diacritizer() -> Diacritizer:
    global _default
    if _default is None:
        _default = Diacritizer()
    return _default


def diacritize(text: str) -> str:
    """Vocalize bare Arabic text using the default (lazy) model instance.

    The model is downloaded from Hugging Face on the first call and cached
    under ``~/.cache/huggingface``; later calls are offline.
    """
    return _default_diacritizer().diacritize(text)


def predict_labels(text: str) -> list:
    """Predicted label id (0..12) per character, using the default instance."""
    return _default_diacritizer().predict_labels(text)


def reset_default() -> None:
    """Drop the lazily-created default instance (mainly useful in tests)."""
    global _default
    _default = None


def get_factory() -> Callable[[], Diacritizer]:
    """The factory used by :func:`diacritize` (for testing/custom defaults)."""
    return _default_diacritizer
