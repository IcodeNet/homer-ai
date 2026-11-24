"""Search Homer's text by words: passages, BM25 from scratch, and search scores.

Chapter 9 builds every piece of this in its notebooks (09_01 and 09_02) and
imports the finished version afterwards. Only the standard library is used.

The pieces, in the order a question passes through them:

    load_passages   cut the books into passages that remember where they came from
    tokenize        text -> lowercase words without accents
    BM25            score every passage against the words of a question
    search          the three above, with the index built once and kept in memory
    retrieval_scores, first_hit_rank   measure the search on labelled questions
"""

import json
import math
import re
import unicodedata
from typing import NamedTuple

from homer_ai.train_homer import BOOKS_DIR

HOMER = BOOKS_DIR.parents[1]  # data/homer
QUESTIONS_FILE = HOMER.parent / "eval" / "questions.jsonl"

POEMS = ("odyssey", "iliad")
LANGUAGES = ("english", "greek")
# English: one line of a file is one paragraph of prose, so one line is a passage.
# Greek: one line is one verse, which is too short to search, so 5 verses are one.
LINES_PER_PASSAGE = {"english": 1, "greek": 5}

# Butler's translation uses the Roman names. A question usually uses the Greek
# ones, so a question word on the left also searches for the word on the right.
BUTLER_NAMES = {
    "odysseus": "ulysses",
    "zeus": "jove",
    "athena": "minerva",
    "athene": "minerva",
    "poseidon": "neptune",
    "hera": "juno",
    "aphrodite": "venus",
    "ares": "mars",
    "hermes": "mercury",
    "artemis": "diana",
    "hephaestus": "vulcan",
    "heracles": "hercules",
    "persephone": "proserpine",
    "demeter": "ceres",
    "cronos": "saturn",
    "dionysus": "bacchus",
}


class Passage(NamedTuple):
    """A piece of one book, and where it came from."""

    poem: str  # "odyssey" or "iliad"
    language: str  # "english" or "greek"
    book: int  # 1 to 24
    first: int  # its first line in the book's file, counted from 1
    last: int  # its last line (the same as `first` for a one-line passage)
    text: str

    @property
    def ref(self):
        """Where to find it, e.g. "odyssey 9:12" or "iliad 1:1-5" (book:lines)."""
        lines = (
            str(self.first) if self.first == self.last else f"{self.first}-{self.last}"
        )
        return f"{self.poem} {self.book}:{lines}"


def chunk_lines(lines, size):
    """Cut a list of lines into runs of `size`: (first, last, text), lines from 1."""
    chunks = []
    for start in range(0, len(lines), size):
        part = lines[start : start + size]
        chunks.append((start + 1, start + len(part), "\n".join(part)))
    return chunks


def load_passages(poem, language, size=None):
    """All passages of one poem in one language. No passage crosses a book."""
    size = size or LINES_PER_PASSAGE[language]
    passages = []
    for path in sorted((HOMER / poem / language).glob("book_*.txt")):
        book = int(path.stem.split("_")[1])
        lines = path.read_text(encoding="utf-8").splitlines()
        for first, last, text in chunk_lines(lines, size):
            passages.append(Passage(poem, language, book, first, last, text))
    return passages


def strip_accents(text):
    """Remove accents and breathings: `μῆνιν` -> `μηνιν`. Plain letters stay."""
    split = unicodedata.normalize("NFD", text)  # a letter, then its marks
    return "".join(c for c in split if unicodedata.category(c) != "Mn")


def tokenize(text):
    """Text -> a list of lowercase words without accents. Digits and marks go."""
    plain = strip_accents(text).casefold().replace("ʼ", " ")
    return re.findall(r"[^\W\d_]+", plain)


def expand_names(words):
    """Add Butler's Roman name after each Greek one: odysseus -> odysseus ulysses."""
    out = []
    for word in words:
        out.append(word)
        if word in BUTLER_NAMES:
            out.append(BUTLER_NAMES[word])
    return out


class BM25:
    """Score texts against the words of a question (Okapi BM25).

    A word of the question adds to a text's score:

        idf(word) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / average))

    tf is how often the word is in the text; idf is high for a rare word; the
    last part makes a long text earn a little less for the same count.
    """

    def __init__(self, texts, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.lengths = []
        self.postings = {}  # word -> [(text number, how often it is in that text)]
        for number, text in enumerate(texts):
            words = tokenize(text)
            self.lengths.append(len(words))
            counts = {}
            for word in words:
                counts[word] = counts.get(word, 0) + 1
            for word, count in counts.items():
                self.postings.setdefault(word, []).append((number, count))
        self.size = len(self.lengths)
        self.average = sum(self.lengths) / max(1, self.size)

    def idf(self, word):
        """How much a word is worth: high if few texts have it, near 0 if most do."""
        n = len(self.postings.get(word, ()))
        return math.log(1 + (self.size - n + 0.5) / (n + 0.5))

    def scores(self, words):
        """{text number: score} for every text that has at least one of the words."""
        totals = {}
        for word in dict.fromkeys(words):  # each different word once, in order
            idf = self.idf(word)
            for number, tf in self.postings.get(word, ()):
                long = 1 - self.b + self.b * self.lengths[number] / self.average
                part = idf * tf * (self.k1 + 1) / (tf + self.k1 * long)
                totals[number] = totals.get(number, 0.0) + part
        return totals

    def search(self, words, k=5):
        """The k best (text number, score), best first; ties go to the earlier text."""
        ranked = sorted(
            self.scores(words).items(), key=lambda item: (-item[1], item[0])
        )
        return ranked[:k]


_indexes = {}  # (poem, language) -> (passages, BM25); built on first use


def get_index(poem="both", language="english"):
    """The passages and their BM25 index, built once and then kept in memory."""
    if poem not in (*POEMS, "both"):
        raise ValueError(f'poem must be "odyssey", "iliad" or "both", not {poem!r}')
    if language not in LANGUAGES:
        raise ValueError(f'language must be "english" or "greek", not {language!r}')
    key = (poem, language)
    if key not in _indexes:
        poems = POEMS if poem == "both" else (poem,)
        passages = [p for name in poems for p in load_passages(name, language)]
        _indexes[key] = (passages, BM25([p.text for p in passages]))
    return _indexes[key]


def search(question, poem="both", language="english", k=5, names=True):
    """The k passages that best match the question's words: [(Passage, score)].

    This matches words, not meaning: a passage that says the same thing in
    other words is not found. With `names=True`, Greek names in the question
    also search for the Roman names Butler's English uses.
    """
    passages, index = get_index(poem, language)
    words = tokenize(question)
    if names and language == "english":
        words = expand_names(words)
    return [(passages[number], score) for number, score in index.search(words, k)]


def load_questions(path=QUESTIONS_FILE):
    """The labelled questions: one JSON object per line (see data/eval/README.md)."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_hit(passage, question):
    """Is this passage inside one of the places that answer the question?

    `question["refs"]` lists them as [book, first line, last line], in the
    English files of `question["poem"]`.
    """
    if passage.poem != question["poem"]:
        return False
    return any(
        passage.book == book and passage.first <= last and passage.last >= first
        for book, first, last in question["refs"]
    )


def first_hit_rank(passages, question):
    """The place (1, 2, 3 ...) of the first right passage in a result, or None."""
    for rank, passage in enumerate(passages, start=1):
        if is_hit(passage, question):
            return rank
    return None


def retrieval_scores(ranks, ks=(1, 5, 10)):
    """Recall@k and MRR from one rank per question (None = not found).

    recall@k: the share of questions whose right passage is in the first k.
    MRR (mean reciprocal rank): the average of 1/rank, with 0 for not found.
    """
    scores = {
        f"recall@{k}": sum(r is not None and r <= k for r in ranks) / len(ranks)
        for k in ks
    }
    scores["mrr"] = sum(1 / r for r in ranks if r is not None) / len(ranks)
    return scores
