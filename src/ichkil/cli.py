"""Command-line interface.

Usage:
    ichkil "محمد قرأ الكتاب"
    ichkil --json "الكتاب على الطاولة"
    cat text.txt | ichkil
    ichkil --model ./model.onnx "محمد"     # offline
"""

from __future__ import annotations

import argparse
import json
import sys

from .diacritizer import Diacritizer
from .download import DEFAULT_REPO_ID, DEFAULT_REVISION
from .version import __version__

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ichkil",
        description="Arabic Tashkeel (diacritization) — vocalizes bare Arabic text.",
    )
    parser.add_argument("text", nargs="?", help="bare Arabic text (reads stdin if omitted)")
    parser.add_argument("--model", default=None, help="path to a local model.onnx (offline mode)")
    parser.add_argument(
        "--repo", default=DEFAULT_REPO_ID, help="Hugging Face model repo (default: %(default)s)"
    )
    parser.add_argument(
        "--revision", default=DEFAULT_REVISION, help="branch/tag/commit (default: %(default)s)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit a JSON object instead of plain text",
    )
    parser.add_argument("--version", action="version", version=f"ichkil {__version__}")
    args = parser.parse_args(argv)

    text = args.text
    if text is None:
        if sys.stdin.isatty():
            parser.print_usage(sys.stderr)
            return 2
        text = sys.stdin.read()
    if not text.strip():
        print("error: no input text", file=sys.stderr)
        return 2

    try:
        if args.model:
            diacritizer = Diacritizer.from_file(args.model)
        else:
            diacritizer = Diacritizer(repo_id=args.repo, revision=args.revision)
        output = diacritizer.diacritize(text)
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report any failure
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.as_json:
        print(json.dumps({"input": text.strip(), "output": output}, ensure_ascii=False))
    else:
        print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
