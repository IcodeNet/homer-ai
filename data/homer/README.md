# Homer: Odyssey and Iliad, Greek and English

Here's the full text of Homer's two poems, the *Odyssey* and the
*Iliad*, in Ancient Greek and in English. Each poem has 24 books, and each
book is one plain text file.

The training code (`src/homer_ai/train_homer.py`) reads from this folder. By default
that's just `odyssey/greek/book_01.txt`, Book 1 of the Odyssey, but you can point it at more
books, at the Iliad, or at the English text.

```
data/homer/
  odyssey/greek/book_01.txt … book_24.txt     one verse per line
  odyssey/english/book_01.txt … book_24.txt   one paragraph per line (prose)
  iliad/greek/book_01.txt … book_24.txt       one verse per line
  iliad/english/book_01.txt … book_24.txt     one paragraph per line (prose)
```

## File format

- Plain text, UTF-8 encoding, NFC-normalised (one fixed way of storing
  accented letters, so the same letter is always the same bytes).
- No line numbers, no headings, no markup, no footnotes.
- Every line ends with a newline, including the last one.
- **Greek:** one verse (one line of the poem) per line.
- **English:** the translation is prose, not verse. So one line is one
  paragraph, and a line can be long. English lines do **not** match Greek
  lines one to one.

## Counts

| Poem    | Language | Files | Lines                 | Characters |
|---------|----------|------:|----------------------:|-----------:|
| Odyssey | Greek    | 24    | 12,107 verses         | 538,303    |
| Odyssey | English  | 24    | 1,027 paragraphs      | 606,758    |
| Iliad   | Greek    | 24    | 15,687 verses         | 690,573    |
| Iliad   | English  | 24    | 1,113 paragraphs      | 803,456    |

Characters are Unicode characters (what Python's `len()` counts), including
the newline at the end of each line.

## Sources and licences

The short version is in [LICENSE.md](LICENSE.md). The details and the credit are here.

### Greek: Perseus Digital Library

From the Perseus Digital Library, Tufts University (https://www.perseus.tufts.edu/),
GitHub repository https://github.com/PerseusDL/canonical-greekLit
(downloaded November 2026, commit `dacb049`).

- **Odyssey:** `data/tlg0012/tlg002/tlg0012.tlg002.perseus-grc2.xml`.
  Greek text as edited by A. T. Murray, *Homer: The Odyssey, with an English
  Translation*, Loeb Classical Library, 1919.
- **Iliad:** `data/tlg0012/tlg001/tlg0012.tlg001.perseus-grc2.xml`.
  Greek text as edited by D. B. Monro and T. W. Allen, *Homeri Opera*,
  Oxford Classical Texts, Clarendon Press, 1908–1920. (Note: this is a
  different editor from the Odyssey file.)

Licence: Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0),
https://creativecommons.org/licenses/by-sa/4.0/ . Perseus asks that reuse
credits the Perseus Digital Library with a link to https://www.perseus.tufts.edu/
and to the source GitHub repository, as above. "ShareAlike" means that if you
share changed versions of this text, you must share them under the same licence.

### English: Samuel Butler's prose translations, Project Gutenberg

- **Odyssey:** Samuel Butler, *The Odyssey, rendered into English prose for
  the use of those who cannot read the original* (first published 1900;
  the Gutenberg file is the second edition of 1922, seen through the press
  by Henry Festing Jones, whose preface is dated December 1921).
  Project Gutenberg eBook #1727, https://www.gutenberg.org/ebooks/1727
- **Iliad:** Samuel Butler, *The Iliad of Homer, rendered into English prose
  for the use of those who cannot read the original* (first published 1898).
  Project Gutenberg eBook #2199, https://www.gutenberg.org/ebooks/2199

Licence: **public domain.** Butler died in 1902 (and Jones in 1928), and
both books were published before 1930, so they are public domain in the United States and in
countries where copyright lasts the author's life plus 70 years or less.
The Project Gutenberg header, footer and licence text were removed; the
"Project Gutenberg" name is not used for this cleaned copy.

Butler uses Latin names for many gods and some heroes: Jove (Zeus),
Minerva (Athena), Juno (Hera), Neptune (Poseidon), Mercury (Hermes),
Ulysses (Odysseus). The Greek text uses the Greek names.

Why Butler? Perseus has English versions too, but its Iliad "Murray" file
is modernised wording that isn't the same as Murray's 1924 printing, and
its Butler files are a modern revision (by T. Power and G. Nagy) with Greek
words in brackets. The original Butler text is clearly public domain, it's
one translator for both poems, and it's clean prose.

## What was changed from the sources

Greek:
- Took the text of every `<l>` (verse line) element, in order. Dropped all
  XML tags. Squeezed runs of spaces to one space.
- Four elision marks written as `’` (U+2019) were changed to `ʼ` (U+02BC),
  the mark used everywhere else in these files (Iliad 8.548, 8.550, 8.551,
  Odyssey 23.283).

English:
- Removed the Project Gutenberg header and footer, the prefaces, the table
  of contents, the book headings, and Butler's one-paragraph summary at the
  top of each book.
- Removed Butler's footnotes in the Odyssey (the notes section at the end and
  the 187 footnote numbers in the text, like `strap.16`).
- Butler put square brackets `[ ]` around Odyssey passages he thought were
  added later. The brackets were removed; the words were kept.
- Removed `_` italic markers and `[Illustration]` lines.
- Books 3 and 4 of the Odyssey start with a lower-case word in the Gutenberg
  file (the big first letter was lost); it was changed to a capital.
- Joined each paragraph onto one line and NFC-normalised the text.

## Known differences from the "standard" line count

The usual totals are about 12,110 verses for the Odyssey and 15,693 for the
Iliad. These editions come up a few lines short, because the editors left some
lines out (they thought those lines were added later):

- Odyssey (3 lines fewer): 10.456, 16.101, 23.49 are absent. Also, lines
  3.304/3.305 and 14.63/14.64 are printed in swapped order.
- Iliad (6 lines fewer): 9.458–461, 11.543, 14.269 are absent. Iliad
  8.548–552 are marked as doubtful in the source but are kept here.

## How it was built

With a small Python script, `scripts/corpus/build.py`. It uses only the standard library
(`xml.etree.ElementTree`, `re`, `unicodedata`), reads the downloaded XML and text files,
and writes the files here.

## How to rebuild this folder

Three scripts in `scripts/corpus/` rebuild everything here from the original sources:

```
uv run python -I scripts/corpus/download.py   # get the sources into data/raw/ (not stored in git)
uv run python -I scripts/corpus/build.py      # clean them into one file per book, here
uv run python -I scripts/corpus/check.py      # check the files and print the counts
```

The Greek is pinned to Perseus commit `dacb049`, so you get the same result every time.
