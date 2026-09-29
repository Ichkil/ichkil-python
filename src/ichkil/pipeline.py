"""Pure text pipeline for Arabic Tashkeel — no model, no third-party imports.

The model emits exactly one label per input character:

    0        non-letter token (space / punctuation) or padding
    1        letter carries no diacritic
    2 .. 12  the eleven harakat / shadda / tanwin combinations

This module is the single source of truth shared by inference and the tests,
and it mirrors the reference implementation in the core repository
(``ichkil/constants.py``) and the JS pipeline used by the Hugging Face Space.
Keep the three in sync.
"""

from __future__ import annotations

import re
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
    "strip_diacritics",
    "encode",
    "argmax",
    "decode",
    "attach",
]

# A single Arabic letter (the characters the model may decorate).
AR_LETTER_RE = re.compile(r"[\u0621-\u064A]")

# Any Arabic diacritic / harakat code point (stripped from input first).
AR_DIACRITIC_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")

# Tatweel (horizontal line), removed from input before encoding.
TATWEEL = "\u0640"

LABEL_NON_LETTER = 0
LABEL_NONE = 1
NUM_LABELS = 13

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
)

COMBO_TO_LABEL: dict[str, int] = {combo: idx for idx, combo in enumerate(LABEL_TO_COMBO) if combo}


def is_arabic_letter(ch: str) -> bool:
    """Return ``True`` if ``ch`` is a single Arabic letter (U+0621..U+064A)."""
    return bool(AR_LETTER_RE.fullmatch(ch))


def label_to_combo(label: int) -> str:
    """Map a label id to its diacritic combination (``""`` for unknown ids)."""
    if 0 <= label < NUM_LABELS:
        return LABEL_TO_COMBO[label]
    return ""


def combo_to_label(combo: str) -> int:
    """Map a diacritic combination (or ``""``) to its label id (``0`` if unknown)."""
    return COMBO_TO_LABEL.get(combo, LABEL_NON_LETTER)


def strip_diacritics(text: str) -> str:
    """Return ``text`` with all diacritics and tatweel removed.

    Letters, spaces and punctuation are preserved so the result stays aligned
    with the original character positions.
    """
    text = AR_DIACRITIC_RE.sub("", text)
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
