"""Quickstart example for ichkil (Python).

Run:
    python examples/quickstart.py

The model is downloaded from Hugging Face on first run (~5 MB) and cached
under ``~/.cache/huggingface``; afterwards this script runs offline.
"""

from __future__ import annotations

import ichkil
from ichkil import Diacritizer

EXAMPLES = [
    "محمد قرأ الكتاب في المدرسة",
    "العلم نور والجهل ظلمة",
    "القهوة جميلة في الصباح",
    "مرحبا، كيف حالك؟",
    "hello world — mixed input",
]


def main() -> None:
    print(f"ichkil {ichkil.__version__}\n")

    for text in EXAMPLES:
        print(f"  {text}")
        print(f"  -> {ichkil.diacritize(text)}\n")

    d = Diacritizer()
    print("labels for 'كتاب':", d.predict_labels("كتاب"))
    print("idempotent check:", d.diacritize(d.diacritize("مدرسة")) == d.diacritize("مدرسة"))


if __name__ == "__main__":
    main()
