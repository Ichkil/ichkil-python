"""Integration tests for :class:`ichkil.Diacritizer`.

Marked ``integration`` because they download the model from Hugging Face on
the first run (≈5 MB, cached under ``~/.cache/huggingface`` afterwards).
Run with::

    pytest -m integration

The golden vectors come from ``tests/golden.json`` — a verbatim copy of the
core repository's ``golden/vectors.json`` (cross-runtime contract).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import ichkil
from ichkil import ChecksumError, Diacritizer, IchkilError
from ichkil.download import resolve_model, verify_checksum

# The whole module exercises the real model (the ``diacritizer`` fixture
# downloads it on first use). Keep it out of the pure-pipeline job so CI can
# run ``pytest -m "not integration"`` without network access.
pytestmark = pytest.mark.integration

GOLDEN = json.loads((Path(__file__).parent / "golden.json").read_text(encoding="utf-8"))
VECTORS = GOLDEN["vectors"]
LONG_VECTOR = next(v for v in VECTORS if v["category"] == "long")


@pytest.fixture(scope="module")
def diacritizer() -> Diacritizer:
    return Diacritizer()


def test_golden_vectors_diacritize(diacritizer: Diacritizer):
    """The model must reproduce every core golden vector exactly."""
    for vec in VECTORS:
        got = diacritizer.diacritize(vec["input"])
        assert got == vec["expected_text"], (
            f"{vec['id']} ({vec['category']})\n"
            f"input    : {vec['input']}\n"
            f"expected : {vec['expected_text']}\n"
            f"got      : {got}"
        )


def test_golden_vectors_predict_labels(diacritizer: Diacritizer):
    """argmax(labels) must equal the core golden labels for every vector."""
    for vec in VECTORS:
        if vec["category"] == "long":
            continue  # covered by test_max_seq_len_contract
        got = diacritizer.predict_labels(vec["input"])
        assert got == vec["labels"], (
            f"{vec['id']} ({vec['category']})\n"
            f"expected labels: {vec['labels']}\n"
            f"got      labels: {got}"
        )


def test_max_seq_len_contract(diacritizer: Diacritizer):
    """The 1024-char vector (== contract max_seq_len) must pass unchanged."""
    assert len(LONG_VECTOR["input"]) == GOLDEN["contract"]["max_seq_len"]
    assert diacritizer.diacritize(LONG_VECTOR["input"]) == LONG_VECTOR["expected_text"]


def test_empty_and_whitespace_input(diacritizer: Diacritizer):
    assert diacritizer.diacritize("") == ""
    assert diacritizer.diacritize("   \n\t ") == ""
    assert diacritizer.predict_labels("") == []


def test_non_arabic_passthrough(diacritizer: Diacritizer):
    assert diacritizer.diacritize("hello world") == "hello world"
    assert diacritizer.diacritize("a, b") == "a, b"
    assert diacritizer.diacritize("price 100") == "price 100"


def test_mixed_text_only_decorates_arabic_letters(diacritizer: Diacritizer):
    got = diacritizer.diacritize("محمد 2024")
    assert got.startswith("مُحَمَّدٌ ")
    assert got.endswith("2024")


@pytest.mark.parametrize(
    "vocalized",
    [
        "مُحَمَّدٌ قَرَأَ الْكِتَابَ فِي الْمَدْرَسَةِ",
        "الْعِلْمِ نُورٌ وَالْجَهْلُ ظُلْمَةُ",
        "مَدْرَسَةٌ",
    ],
)
def test_idempotent_on_vocalized_text(diacritizer: Diacritizer, vocalized: str):
    """Re-diacerizing already-vocalized text must not change it."""
    assert diacritizer.diacritize(vocalized) == vocalized


def test_input_with_diacritics_is_normalized(diacritizer: Diacritizer):
    # stripping then re-attaching yields the canonical golden form
    assert diacritizer.diacritize("مُحَمَّدٌ") == "مُحَمَّدٌ"
    assert diacritizer.diacritize("كِتَابِ") == "كِتَابِ"


def test_predict_labels_shape_and_range(diacritizer: Diacritizer):
    text = "العلم نور"
    labels = diacritizer.predict_labels(text)
    assert len(labels) == len(text)
    assert all(0 <= lab < 13 for lab in labels)
    # non-letter characters get "no diacritic" (0 or 1); attach() ignores them anyway
    assert labels[text.index(" ")] in (0, 1)


def test_diacritize_none_raises_type_error(diacritizer: Diacritizer):
    with pytest.raises(TypeError):
        diacritizer.predict_labels(None)  # type: ignore[arg-type]


def test_from_file_offline(diacritizer: Diacritizer):
    """``from_file`` must behave identically to the HF-backed constructor."""
    bundle = resolve_model()  # reuses the HF cache; no second download
    offline = Diacritizer.from_file(bundle.onnx_path)
    for vec in VECTORS[:8]:
        assert offline.diacritize(vec["input"]) == vec["expected_text"]
    assert offline.repo_id == "<local>"


def test_from_file_missing_model_raises():
    with pytest.raises(IchkilError):
        Diacritizer.from_file("/nonexistent/model.onnx")


def test_verify_checksum_raises_on_mismatch(tmp_path: Path):
    blob = tmp_path / "model.onnx"
    blob.write_bytes(b"fake model bytes")
    with pytest.raises(ChecksumError):
        verify_checksum(blob, "0" * 64)
    # and accepts the real digest
    import hashlib

    real = hashlib.sha256(b"fake model bytes").hexdigest()
    verify_checksum(blob, real)  # must not raise
    # case-insensitive comparison
    verify_checksum(blob, real.upper())


def test_version_and_public_api():
    assert ichkil.__version__ == "1.0.0"
    assert ichkil.NUM_LABELS == 13
    # the module-level convenience API exists and delegates to the same code
    assert callable(ichkil.diacritize)
    assert callable(ichkil.predict_labels)
    assert callable(ichkil.reset_default)
