"""Pure-pipeline tests — no model, no network, no third-party dependencies.

These lock in the text contract (label table, strip/encode/decode/attach)
shared by every ichkil client runtime. The golden vectors in ``golden.json``
are copied verbatim from the core repository (``golden/vectors.json``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ichkil import pipeline as P

GOLDEN = json.loads((Path(__file__).parent / "golden.json").read_text(encoding="utf-8"))
VECTORS = GOLDEN["vectors"]


def test_label_table_covers_all_13_labels():
    assert P.NUM_LABELS == 13
    assert len(P.LABEL_TO_COMBO) == 13
    # labels 0 and 1 carry no diacritic
    assert P.LABEL_TO_COMBO[0] == ""
    assert P.LABEL_TO_COMBO[1] == ""
    # labels 2..12 are the eleven harakat / shadda / tanwin combinations
    assert all(P.LABEL_TO_COMBO[i] for i in range(2, 13))
    # shadda+haraka combos contain the shadda mark
    assert "\u0651" in P.LABEL_TO_COMBO[7]
    assert "\u0651" in P.LABEL_TO_COMBO[8]
    assert "\u0651" in P.LABEL_TO_COMBO[9]
    # tanwin combos
    assert P.LABEL_TO_COMBO[10] == "\u064b"
    assert P.LABEL_TO_COMBO[11] == "\u064c"
    assert P.LABEL_TO_COMBO[12] == "\u064d"


@pytest.mark.parametrize("label", [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
def test_label_to_combo_roundtrip(label: int):
    combo = P.label_to_combo(label)
    assert combo
    assert P.combo_to_label(combo) == label


def test_combo_to_label_unknown_defaults_to_zero():
    assert P.combo_to_label("") == P.LABEL_NON_LETTER
    assert P.combo_to_label("\u0670") == P.LABEL_NON_LETTER


def test_label_to_combo_out_of_range():
    assert P.label_to_combo(-1) == ""
    assert P.label_to_combo(13) == ""


@pytest.mark.parametrize(
    ("ch", "expected"),
    [
        ("\u0627", True),  # alif
        ("\u0628", True),  # ba
        ("\u0649", True),  # yeh
        ("\u064a", True),  # yeh (final range)
        ("a", False),
        ("1", False),
        (" ", False),
        # tatweel is inside the reference letter range (core + JS both use
        # [\u0621-\u064A]); strip_diacritics removes it from input before attach.
        ("\u0640", True),
        ("\u0650", False),  # kasra is a diacritic, not a letter
    ],
)
def test_is_arabic_letter(ch: str, expected: bool):
    assert P.is_arabic_letter(ch) is expected


def test_strip_diacritics_removes_harakat_and_tatweel():
    vocalized = "مُحَمَّدٌ قَرَأَ الْكِتَابَ"
    assert P.strip_diacritics(vocalized) == "محمد قرأ الكتاب"
    # tatweel is removed too
    assert P.strip_diacritics("مـحـمـد") == "محمد"
    # letters, spaces, punctuation are preserved
    assert P.strip_diacritics("a b, 123") == "a b, 123"
    # empty stays empty
    assert P.strip_diacritics("") == ""


def test_encode_uses_sym2id_and_falls_back_to_unk():
    sym2id = {"a": 10, "b": 11}
    unk = 1
    assert P.encode("ab", sym2id, unk) == [10, 11]
    assert P.encode("ac", sym2id, unk) == [10, 1]
    assert P.encode("", sym2id, unk) == []


def test_argmax_returns_first_index_on_ties():
    assert P.argmax([0.5, 0.5, 0.5]) == 0
    assert P.argmax([1.0, 2.0, 2.0]) == 1
    assert P.argmax([-3.0]) == 0
    assert P.argmax([0.1, -1.0, 0.05]) == 0


def _logits_with_labels(labels: list[int], num_labels: int = P.NUM_LABELS) -> list[float]:
    """Flat logits in which each position's argmax is the given label."""
    flat: list[float] = []
    for lab in labels:
        row = [0.0] * num_labels
        row[lab] = 1.0
        flat.extend(row)
    return flat


def test_decode_matches_attach_with_synthetic_logits():
    base = "محمد"
    labels = [3, 2, 7, 11]
    expected = P.attach(base, labels)
    assert P.decode(base, _logits_with_labels(labels), P.NUM_LABELS) == expected


def test_decode_leaves_non_letters_untouched():
    base = "a, b"
    assert P.decode(base, _logits_with_labels([1, 1, 1, 2]), P.NUM_LABELS) == "a, b"
    assert P.attach(base, [1, 1, 1, 2]) == "a, b"


def test_attach_golden_vectors():
    """``attach(input, labels) == expected_text`` for every core golden vector."""
    assert VECTORS, "golden.json is empty"
    for vec in VECTORS:
        assert P.attach(vec["input"], vec["labels"]) == vec["expected_text"], (
            f"{vec['id']} ({vec['category']}): attach(input, labels) != expected_text"
        )


def test_attach_length_mismatch_is_tolerated_like_reference():
    # attach() zips; extra labels are ignored, missing labels stop the pass-through.
    assert P.attach("abc", [1, 1]) == "ab"
    assert P.attach("ab", [1, 1, 1]) == "ab"


def test_contract_max_seq_len_is_1024():
    assert GOLDEN["contract"]["max_seq_len"] == 1024
    for vec in VECTORS:
        assert 1 <= len(vec["input"]) <= GOLDEN["contract"]["max_seq_len"]
