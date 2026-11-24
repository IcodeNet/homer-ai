"""Check data/homer/ (UTF-8, NFC, no stray spaces) and print line/character counts.

Run from the project folder:

    uv run python -I scripts/corpus/check.py
"""

import collections
import sys
import unicodedata
from pathlib import Path

root = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path(__file__).resolve().parents[2] / "data" / "homer"
)
for poem in ("odyssey", "iliad"):
    for lang in ("greek", "english"):
        d = root / poem / lang
        rows, chars = [], collections.Counter()
        for f in sorted(d.glob("book_*.txt")):
            b = f.read_bytes()
            t = b.decode("utf-8")  # raises if not valid UTF-8
            assert unicodedata.is_normalized("NFC", t), f
            assert "\r" not in t and "\t" not in t and "  " not in t, f
            assert all(line.strip() == line and line for line in t.split("\n")[:-1]), f
            rows.append((f.stem, t.count("\n"), len(t)))
            chars.update(t)
        print(
            poem,
            lang,
            "lines",
            sum(r[1] for r in rows),
            "chars",
            sum(r[2] for r in rows),
        )
        print("  per book:", " ".join(f"{r[0][-2:]}:{r[1]}/{r[2]}" for r in rows))
        odd = {c: n for c, n in chars.items() if not (c.isalpha() or c in " \n")}
        print(
            "  non-letter chars:",
            " ".join(
                f"{c!r}({unicodedata.name(c, '?')}):{n}" for c, n in sorted(odd.items())
            ),
        )
