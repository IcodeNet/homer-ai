"""Build data/homer/ from Perseus XML (Greek) and Gutenberg text (English).

Run from the project folder, after scripts/corpus/download.py:

    uv run python -I scripts/corpus/build.py

It writes one clean text file per book. See data/homer/README.md for every
cleaning step.
"""

import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

TEI = "{http://www.tei-c.org/ns/1.0}"
ROMAN = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
    "VIII": 8,
    "IX": 9,
    "X": 10,
    "XI": 11,
    "XII": 12,
    "XIII": 13,
    "XIV": 14,
    "XV": 15,
    "XVI": 16,
    "XVII": 17,
    "XVIII": 18,
    "XIX": 19,
    "XX": 20,
    "XXI": 21,
    "XXII": 22,
    "XXIII": 23,
    "XXIV": 24,
}


def clean(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    return re.sub(r"\s+", " ", s).strip()


def greek_books(xml_path: Path) -> dict[int, list[str]]:
    root = ET.parse(xml_path).getroot()
    body = root.find(f"{TEI}text/{TEI}body")
    books = {}
    for div in body.iter(f"{TEI}div"):
        if (div.get("subtype") or "").lower() != "book":
            continue
        n = int(div.get("n"))
        lines = []
        for verse in div.iter(f"{TEI}l"):
            # itertext() includes text inside <del> (editor-bracketed lines are kept)
            # but <note> elements never sit inside <l> in these files.
            text = clean("".join(verse.itertext()))
            # 4 lines use U+2019 for elision; the rest use U+02BC. Unify.
            text = text.replace("’", "ʼ")
            if text:
                lines.append(text)
        books[n] = lines
    return books


def english_books(txt_path: Path) -> dict[int, list[str]]:
    raw = txt_path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    start = raw.index("*** START OF THE PROJECT GUTENBERG EBOOK")
    start = raw.index("\n", start) + 1
    end = raw.index("*** END OF THE PROJECT GUTENBERG EBOOK")
    body = raw[start:end]
    # Odyssey: drop the translator's footnotes section at the end.
    fn = re.search(r"^FOOTNOTES:\s*$", body, flags=re.M)
    if fn:
        body = body[: fn.start()]
    lines = body.split("\n")
    heads = [
        (i, ROMAN[m.group(1)])
        for i, line in enumerate(lines)
        if (m := re.fullmatch(r"BOOK ([IVXL]+)\.?", line.strip()))
        and line
        == line.lstrip()  # real headings start at column 0; contents list is indented
    ]
    assert [b for _, b in heads] == list(range(1, 25)), heads
    books = {}
    for k, (i, n) in enumerate(heads):
        j = heads[k + 1][0] if k + 1 < len(heads) else len(lines)
        chunk = "\n".join(lines[i + 1 : j])
        paras = [clean(p) for p in re.split(r"\n\s*\n", chunk)]
        paras = [p for p in paras if p]
        # The first paragraph is Butler's summary of the book: drop it.
        paras = paras[1:]
        out = []
        for p in paras:
            if p == "[Illustration]":
                continue
            # Footnote markers like "strap.16" or "others 10—". Checked: every
            # digit in the book text is a footnote marker (Odyssey 1..187;
            # the Iliad has none).
            p = re.sub(r"\s*\d+", "", p)
            p = p.replace("[", "").replace(
                "]", ""
            )  # Butler's brackets for suspect passages: keep words
            p = p.replace("_", "")  # Gutenberg italics markers
            p = clean(p)
            if p:
                out.append(p)
        # Gutenberg lost the big first letter on a few books: capitalise it.
        if out[0][0].islower():
            out[0] = out[0][0].upper() + out[0][1:]
        books[n] = out
    return books


def write(books: dict[int, list[str]], folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    assert sorted(books) == list(range(1, 25)), sorted(books)
    for n, lines in books.items():
        (folder / f"book_{n:02d}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    raw = root / "data" / "raw"
    gdir, pgdir, out = raw / "greek", raw / "english", root / "data" / "homer"
    if len(sys.argv) == 4:
        gdir, pgdir, out = (Path(a) for a in sys.argv[1:4])
    write(
        greek_books(gdir / "tlg0012.tlg002.perseus-grc2.xml"), out / "odyssey" / "greek"
    )
    write(
        greek_books(gdir / "tlg0012.tlg001.perseus-grc2.xml"), out / "iliad" / "greek"
    )
    write(english_books(pgdir / "pg1727.txt"), out / "odyssey" / "english")
    write(english_books(pgdir / "pg2199.txt"), out / "iliad" / "english")


if __name__ == "__main__":
    main()
