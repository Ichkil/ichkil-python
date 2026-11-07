"""Pure text pipeline for Arabic Tashkeel — no model, no third-party imports.

The model emits exactly one label per input character:

    0        non-letter token (space / punctuation / digit / Latin) or padding
    1        letter carries no diacritic
    2 .. 15  the harakat / shadda / tanwin combinations
             (13..15 = shadda + tanwin, added in model v0.3)

This module is the single source of truth shared by inference and the tests,
and it mirrors the reference implementation in the core repository
(``ichkil/constants.py``) and the JS pipeline used by the Hugging Face Space.
Keep the three in sync.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence

__all__ = [
    "AR_LETTER_RE",
    "AR_DIACRITIC_RE",
    "TATWEEL",
    "LABEL_NON_LETTER",
    "LABEL_NONE",
    "NUM_LABELS",
    "LABEL_TO_COMBO",
    "COMBO_TO_LABEL",
    "is_arabic_letter",
    "label_to_combo",
    "combo_to_label",
    "normalize_text",
    "strip_diacritics",
    "encode",
    "argmax",
    "decode",
    "attach",
]

# A single Arabic letter (the characters the model may decorate).
AR_LETTER_RE = re.compile(r"[\u0621-\u063A\u0641-\u064A]")

# Any Arabic diacritic / harakat code point (stripped from input first).
AR_DIACRITIC_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")

# Tatweel (horizontal line), removed from input before encoding.
TATWEEL = "\u0640"

LABEL_NON_LETTER = 0
LABEL_NONE = 1
NUM_LABELS = 16

# label id -> canonical diacritic combination string
LABEL_TO_COMBO: tuple[str, ...] = (
    "",  # 0  non-letter token
    "",  # 1  letter carries no diacritic
    "\u064e",  # 2  fatha
    "\u064f",  # 3  damma
    "\u0650",  # 4  kasra
    "\u0652",  # 5  sukun
    "\u0651",  # 6  shadda
    "\u0651\u064e",  # 7  shadda + fatha
    "\u0651\u064f",  # 8  shadda + damma
    "\u0651\u0650",  # 9  shadda + kasra
    "\u064b",  # 10 tanwin fatha
    "\u064c",  # 11 tanwin damma
    "\u064d",  # 12 tanwin kasra
    "\u0651\u064b",  # 13 shadda + tanwin fatha
    "\u0651\u064c",  # 14 shadda + tanwin damma
    "\u0651\u064d",  # 15 shadda + tanwin kasra
)

COMBO_TO_LABEL: dict[str, int] = {combo: idx for idx, combo in enumerate(LABEL_TO_COMBO) if combo}


def is_arabic_letter(ch: str) -> bool:
    """Return ``True`` if ``ch`` is an Arabic letter (U+0621..U+063A, U+0641..U+064A)."""
    return bool(AR_LETTER_RE.fullmatch(ch))


def label_to_combo(label: int) -> str:
    """Map a label id to its diacritic combination (``""`` for unknown ids)."""
    if 0 <= label < NUM_LABELS:
        return LABEL_TO_COMBO[label]
    return ""


def combo_to_label(combo: str) -> int:
    """Map a diacritic combination (or ``""``) to its label id (``0`` if unknown)."""
    return COMBO_TO_LABEL.get(combo, LABEL_NON_LETTER)


_CHAR_MAP = {
    "\u0671": "\u0627",  # alef wasla -> alef
    "\u06cc": "\u064a",  # farsi yeh -> yeh
    "\u06a9": "\u0643",  # keheh -> kaf
    "\u06c1": "\u0647",  # heh goal -> heh
    "\u00a0": " ",  # no-break space
}
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\ufeff]")
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Canonical form used by the model (mirror of the core ``normalize_text``).

    Unicode NFC, presentation variants mapped to base Arabic letters,
    zero-width characters removed, whitespace runs collapsed, trimmed.
    Diacritics are kept (see :func:`strip_diacritics`).
    """
    text = unicodedata.normalize("NFC", text)
    text = "".join(_CHAR_MAP.get(c, c) for c in text)
    text = _ZERO_WIDTH_RE.sub("", text)
    return _WS_RE.sub(" ", text).strip()


def strip_diacritics(text: str) -> str:
    """Normalize ``text``, then remove all diacritics and tatweel.

    Letters, spaces and punctuation are preserved so the result stays aligned
    with the characters the model sees.
    """
    text = AR_DIACRITIC_RE.sub("", normalize_text(text))
    if TATWEEL in text:
        text = text.replace(TATWEEL, "")
    return text


def encode(text: str, sym2id: Mapping[str, int], unk_id: int) -> list[int]:
    """Map each character to its vocabulary id (out-of-vocabulary -> ``unk_id``)."""
    return [sym2id.get(ch, unk_id) for ch in text]


def argmax(values: Sequence[float]) -> int:
    """Index of the maximum value (first one wins ties), matching ONNX Runtime."""
    best = 0
    best_v = values[0]
    for i in range(1, len(values)):
        v = values[i]
        if v > best_v:
            best_v = v
            best = i
    return best


def decode(base: str, logits: Sequence[float], num_labels: int) -> str:
    """Attach predicted diacritics onto ``base`` from flat logits.

    ``logits`` is the model output row for ``base`` in row-major order
    (``len(logits) == len(base) * num_labels``). Non-letter characters pass
    through unchanged.
    """
    chars = list(base)
    out: list[str] = []
    for i, ch in enumerate(chars):
        if is_arabic_letter(ch):
            off = i * num_labels
            out.append(ch + label_to_combo(argmax(logits[off : off + num_labels])))
        else:
            out.append(ch)
    return "".join(out)


def attach(base: str, labels: Sequence[int]) -> str:
    """Attach predicted diacritics onto bare letters of ``base``.

    ``labels`` is one entry per character of ``base``. Non-letter characters
    pass through unchanged.
    """
    out: list[str] = []
    for ch, lab in zip(base, labels):
        if is_arabic_letter(ch):
            out.append(ch + label_to_combo(lab))
        else:
            out.append(ch)
    return "".join(out)
