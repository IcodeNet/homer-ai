"""Chapter 10: the course's pattern on invented company data (`data/domain/`).

Three small examples, each a set of data, a few functions that work on it, and
a way to score them. Everything in `data/domain/` is made up (see its README).

    savings       product sheets + a rates table: `search_terms`, `get_rate`,
                  `escalate`, and the guard `needs_person`        (notebook 10_01)
    emails        400 labelled emails: `keyword_flags` (a baseline), the labels
                  as 4 classes, and the scores `recall`, `macro_f1` (10_02)
    servicedesk   200 tickets + 10 articles: `suggest_articles`, `TicketRouter`
                                                                  (10_03)

The search is the `BM25` of chapter 9 (`retrieval.py`). No model is loaded
anywhere in this file; only `encode_emails` builds tensors, for the fine-tuning
of notebook 10_02.
"""

import csv
import hashlib
import json
import re
from datetime import UTC, date, datetime
from typing import NamedTuple

from homer_ai.retrieval import BM25, HOMER, tokenize

DOMAIN = HOMER.parent / "domain"  # data/domain
SAVINGS = DOMAIN / "savings"
EMAILS = DOMAIN / "emails"
SERVICEDESK = DOMAIN / "servicedesk"

SHEETS = (
    "easy_access.md",
    "fixed_1_year.md",
    "notice_95_day.md",
    "cash_isa.md",
    "faq.md",
)
PRODUCTS = ("easy-access", "fixed-1-year", "notice-95", "cash-isa")


# ---------------------------------------------------------------- savings: documents
class Section(NamedTuple):
    """One numbered section of a product sheet, and where it came from."""

    doc: str  # "easy-access", ... or "faq"
    number: int  # the section's number in its sheet
    doc_title: str  # "Easy Access Saver"
    title: str  # "Withdrawals"
    text: str
    version: int  # the version of the sheet
    in_force_from: str  # the day that version started to apply, "2026-09-01"

    @property
    def ref(self):
        """Where to find it: "easy-access#4" is section 4 of that sheet."""
        return f"{self.doc}#{self.number}"

    @property
    def stamp(self):
        """Which version this is: "version 3, in force from 2026-09-01"."""
        return f"version {self.version}, in force from {self.in_force_from}"


def read_sheet(text):
    """Cut one sheet into (title, {header field: value}, [(number, title, text)])."""
    head, *parts = re.split(r"^## (\d+)\. (.+)$", text, flags=re.MULTILINE)
    title = re.search(r"^# (.+)$", head, flags=re.MULTILINE).group(1)
    fields = dict(re.findall(r"^([A-Z][A-Za-z ]+): (.+)$", head, flags=re.MULTILINE))
    sections = [
        (int(parts[i]), parts[i + 1].strip(), parts[i + 2].strip())
        for i in range(0, len(parts), 3)
    ]
    return title, fields, sections


def load_sections():
    """Every section of the four product sheets and the FAQ, in file order."""
    sections = []
    for name in SHEETS:
        text = (SAVINGS / name).read_text(encoding="utf-8")
        doc_title, fields, parts = read_sheet(text)
        for number, title, body in parts:
            sections.append(
                Section(
                    fields["Product"],
                    number,
                    doc_title,
                    title,
                    body,
                    int(fields["Version"]),
                    fields["In force from"],
                )
            )
    return sections


_terms = {}  # with_titles -> (sections, BM25); built on first use


def terms_index(with_titles=True):
    """The sections and their BM25 index, built once and then kept in memory.

    With `with_titles=True` a section is indexed as "sheet title. section title.
    text", so a question that names the product can find the right sheet.
    """
    if with_titles not in _terms:
        sections = load_sections()
        texts = [
            f"{s.doc_title}. {s.title}. {s.text}" if with_titles else s.text
            for s in sections
        ]
        _terms[with_titles] = (sections, BM25(texts))
    return _terms[with_titles]


def search_terms(query, k=3, with_titles=True):
    """The k sections that best match the words of the query: [(Section, score)]."""
    sections, index = terms_index(with_titles)
    return [(sections[n], score) for n, score in index.search(tokenize(query), k)]


# ---------------------------------------------------------------- savings: rates
class Rate(NamedTuple):
    """One row of the rates table."""

    product: str
    in_force_from: date
    aer: float  # yearly interest, in percent

    @property
    def ref(self):
        """Where to find it: "rates#easy-access/2026-09-01"."""
        return f"rates#{self.product}/{self.in_force_from.isoformat()}"


def load_rates():
    """The rates table, oldest row first within each product."""
    with open(SAVINGS / "rates.csv", encoding="utf-8", newline="") as f:
        rows = [
            Rate(r["product"], date.fromisoformat(r["in_force_from"]), float(r["aer"]))
            for r in csv.DictReader(f)
        ]
    return sorted(rows, key=lambda r: (r.product, r.in_force_from))


def get_rate(product, on):
    """The rate of `product` that was in force on the day `on`.

    `on` is a `date` or a text like "2026-10-01". Raises `ValueError` for a
    product we don't have or a date that isn't one, and `LookupError` when the
    table has no rate that early. It never guesses.
    """
    if product not in PRODUCTS:
        raise ValueError(f"unknown product {product!r}; use one of {list(PRODUCTS)}")
    if not isinstance(on, date):
        try:
            on = date.fromisoformat(str(on))
        except ValueError:
            raise ValueError(f"not a date: {on!r}; write it as YYYY-MM-DD") from None
    earlier = [
        r for r in load_rates() if r.product == product and r.in_force_from <= on
    ]
    if not earlier:
        raise LookupError(f"no rate on file for {product} on {on.isoformat()}")
    return earlier[-1]


# ---------------------------------------------------------------- savings: handing over
# Why a message goes to a person, and which team picks it up.
QUEUES = {
    "complaint": "complaints team",
    "vulnerability": "customer support team",
    "advice": "customer service (we do not give financial advice)",
    "bereavement": "bereavement team",
}
# A first list of words, the kind you write on day one. It is a safety net under
# the model, and notebook 10_01 measures how much it misses.
GUARD_WORDS = {
    "bereavement": ("passed away", "died", "death", "bereave", "funeral", "my late "),
    "vulnerability": (
        "diagnosed",
        "illness",
        "hospital",
        "lost my job",
        "redundan",
        "disab",
        "dementia",
        "can't pay",
        "cannot pay",
    ),
    "complaint": (
        "complain",
        "unacceptable",
        "unhappy",
        "dissatisfied",
        "disappointed",
        "disgrace",
        "fed up",
    ),
    "advice": ("should i", "best for me", "recommend", "good idea", "advise me"),
}


def needs_person(text):
    """The first reason in `GUARD_WORDS` whose words are in the text, or None."""
    low = text.casefold().replace("’", "'")
    for reason, phrases in GUARD_WORDS.items():
        if any(phrase in low for phrase in phrases):
            return reason
    return None


def escalate(reason):
    """Hand the conversation to a person. Returns the handover as a dict.

    Nothing is sent anywhere: in this course the handover is only a record. A
    real system would open a case here.
    """
    if reason not in QUEUES:
        raise ValueError(f"unknown reason {reason!r}; use one of {list(QUEUES)}")
    return {
        "reason": reason,
        "queue": QUEUES[reason],
        "say": "I'm passing this to a colleague, who will reply to you in person.",
    }


def read_jsonl(path):
    """One JSON object per line -> a list of dicts."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_support_questions():
    """The 39 labelled support questions (see data/domain/README.md)."""
    return read_jsonl(SAVINGS / "questions.jsonl")


def answer_has_keys(answer, keys):
    """Does the answer hold every key? Each key is a list of accepted spellings.

    Commas inside numbers are ignored, so "£1,000" matches the key "1000".
    """
    low = re.sub(r"(?<=\d),(?=\d)", "", answer.casefold()) + " "
    return all(any(spelling in low for spelling in group) for group in keys)


def data_fingerprint():
    """12 hex digits that change when a product sheet or the rates table changes."""
    digest = hashlib.sha256()
    for name in (*SHEETS, "rates.csv"):
        digest.update(name.encode())
        digest.update((SAVINGS / name).read_bytes())
    return digest.hexdigest()[:12]


def audit_line(tool, **fields):
    """One line for the audit trail: when, which tool, which data, and `fields`."""
    record = {
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "tool": tool,
        "data": data_fingerprint(),
        **fields,
    }
    return json.dumps(record, ensure_ascii=False)


# ---------------------------------------------------------------- emails
EMAIL_CLASSES = ("neither", "complaint", "vulnerable", "both")

# The baseline: two lists of words a person might write down on day one.
COMPLAINT_WORDS = (
    "complain",
    "unhappy",
    "disappointed",
    "dissatisfied",
    "unacceptable",
    "appalling",
    "not happy",
)
VULNERABLE_WORDS = (
    "passed away",
    "died",
    "bereave",
    "funeral",
    "hospital",
    "diagnosed",
    "illness",
    "disabled",
    "lost my job",
    "redundan",
    "carer",
    "dementia",
)


def load_emails(name="emails", split=None):
    """The emails of `data/domain/emails/<name>.jsonl`, all or one split.

    `name` is "emails" (the 400 built from templates) or "handwritten" (24
    written freely). `split` is "train", "val" or "test".
    """
    emails = read_jsonl(EMAILS / f"{name}.jsonl")
    return [e for e in emails if split in (None, e["split"])]


def email_class(email):
    """Two labels as one class: 0 neither, 1 complaint, 2 vulnerable, 3 both."""
    return int(email["complaint"]) + 2 * int(email["vulnerable"])


def flags_of_class(number):
    """The class number back to (complaint, vulnerable)."""
    return bool(number % 2), bool(number // 2)


def keyword_flags(text):
    """The baseline's guess for one email: (complaint, vulnerable), from word lists."""
    low = text.casefold().replace("’", "'")
    return (
        any(word in low for word in COMPLAINT_WORDS),
        any(word in low for word in VULNERABLE_WORDS),
    )


def encode_emails(emails, encoding, length):
    """The emails as one table of GPT-2 token IDs, and their class numbers.

    The padding and cutting are `encode_messages` of chapter 8 (`finetune.py`).
    """
    import torch

    from homer_ai.finetune import encode_messages

    ids, _ = encode_messages([("ham", e["text"]) for e in emails], encoding, length)
    return ids, torch.tensor([email_class(e) for e in emails])


# ---------------------------------------------------------------- scores
def recall(truth, guess):
    """Of the cases that really are positive, the share we caught (0.0 to 1.0)."""
    caught = sum(t and g for t, g in zip(truth, guess, strict=True))
    return caught / max(1, sum(bool(t) for t in truth))


def precision(truth, guess):
    """Of the cases we flagged, the share that really are positive."""
    caught = sum(t and g for t, g in zip(truth, guess, strict=True))
    return caught / max(1, sum(bool(g) for g in guess))


def f1(truth, guess):
    """One number from precision and recall: 2pr / (p + r). 0 when both are 0."""
    p, r = precision(truth, guess), recall(truth, guess)
    return 2 * p * r / (p + r) if p + r else 0.0


def macro_f1(truth, guess, labels):
    """The plain average of the F1 of every label, each label counting the same."""
    scores = [f1([t == c for t in truth], [g == c for g in guess]) for c in labels]
    return sum(scores) / len(scores)


def confusion(truth, guess, labels):
    """A table of counts: `table[i][j]` = really `labels[i]`, guessed `labels[j]`."""
    place = {label: n for n, label in enumerate(labels)}
    table = [[0] * len(labels) for _ in labels]
    for t, g in zip(truth, guess, strict=True):
        table[place[t]][place[g]] += 1
    return table


# ---------------------------------------------------------------- service desk
class Article(NamedTuple):
    """One knowledge-base article."""

    id: str  # "kb-03"
    title: str
    category: str
    reviewed: str  # the day somebody last checked it, "2026-10-12"
    text: str


def load_kb():
    """The 10 knowledge-base articles, in order of their ID."""
    articles = []
    for path in sorted((SERVICEDESK / "kb").glob("kb-*.md")):
        title, head, text = path.read_text(encoding="utf-8").split("\n\n", 2)
        fields = dict(line.split(": ", 1) for line in head.splitlines())
        articles.append(
            Article(
                fields["ID"],
                title.removeprefix("# "),
                fields["Category"],
                fields["Last reviewed"],
                text.strip(),
            )
        )
    return articles


def load_tickets(split=None):
    """The 200 tickets, all or one split ("train" or "test")."""
    tickets = read_jsonl(SERVICEDESK / "tickets.jsonl")
    return [t for t in tickets if split in (None, t["split"])]


_kb = None


def suggest_articles(text, k=3):
    """The k articles that best match a ticket's words: [(Article, score)]."""
    global _kb
    if _kb is None:
        articles = load_kb()
        _kb = (articles, BM25([f"{a.title}. {a.text}" for a in articles]))
    articles, index = _kb
    return [(articles[n], score) for n, score in index.search(tokenize(text), k)]


class TicketRouter:
    """Route a ticket like its nearest solved tickets (k nearest neighbours).

    Nothing is trained. The solved tickets are put in a BM25 index; a new ticket
    is a query against it, and the categories of the `k` best matches vote.
    """

    def __init__(self, solved, k=5):
        self.solved = solved
        self.k = k
        self.index = BM25([t["text"] for t in solved])
        counts = {}
        for ticket in solved:
            counts[ticket["category"]] = counts.get(ticket["category"], 0) + 1
        self.most_common = max(counts, key=lambda c: (counts[c], c))

    def neighbours(self, text):
        """The k most similar solved tickets: [(ticket, score)]."""
        found = self.index.search(tokenize(text), self.k)
        return [(self.solved[n], score) for n, score in found]

    def route(self, text):
        """(category, votes). The votes are BM25 scores added up per category.

        A ticket that shares no word with any solved ticket gets the most
        common category and no votes, so the caller can see it was a guess.
        """
        votes = {}
        for ticket, score in self.neighbours(text):
            votes[ticket["category"]] = votes.get(ticket["category"], 0.0) + score
        if not votes:
            return self.most_common, {}
        return max(votes, key=lambda c: (votes[c], c)), votes


class ProfileRouter:
    """Route a ticket to the category whose words it shares most.

    Each category becomes ONE long text: all its solved tickets, and its
    knowledge-base articles too if `articles` is given. A word that every
    category uses ("thanks", "urgent", "since yesterday") is then in every
    text, so its idf is close to 0 and it stops counting.
    """

    def __init__(self, solved, articles=()):
        self.categories = sorted({t["category"] for t in solved})
        texts = []
        for category in self.categories:
            parts = [t["text"] for t in solved if t["category"] == category]
            parts += [
                f"{a.title}. {a.text}" for a in articles if a.category == category
            ]
            texts.append(" ".join(parts))
        self.index = BM25(texts)
        sizes = [sum(t["category"] == c for t in solved) for c in self.categories]
        self.most_common = self.categories[sizes.index(max(sizes))]

    def route(self, text):
        """(category, {category: score}); the most common category if no word fits."""
        found = self.index.search(tokenize(text), len(self.categories))
        if not found:
            return self.most_common, {}
        scores = {self.categories[n]: score for n, score in found}
        return self.categories[found[0][0]], scores
