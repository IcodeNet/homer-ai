# Invented company data for chapter 10

**Everything in this folder is made up.** There is no "Example Bank". No real customer, employee or company wrote any of these texts, and none of them is copied or adapted from a real firm's documents, emails or tickets.
The rates, limits and timescales are invented too, so don't read them as facts about any real product or as what any rule requires.

[Chapter 10](../../notebooks/10_advanced_your_domain/10_00_chapter_overview.ipynb) uses this data for three worked examples. It is small on purpose: every notebook reads it in well under a second.

## What is here?

| Folder | Files | Used in |
|---|---|---|
| `savings/` | 4 product sheets and a FAQ (39 numbered sections), `rates.csv` (8 rows), `questions.jsonl` (39 labelled questions) | [10_01](../../notebooks/10_advanced_your_domain/10_01_support_with_citations.ipynb) |
| `emails/` | `emails.jsonl` (400 emails built from templates), `handwritten.jsonl` (24 written freely) | [10_02](../../notebooks/10_advanced_your_domain/10_02_spotting_complaints.ipynb) |
| `servicedesk/` | `kb/` (10 knowledge-base articles), `tickets.jsonl` (200 tickets) | [10_03](../../notebooks/10_advanced_your_domain/10_03_service_desk_triage.ipynb) |

The code that reads it is `src/homer_ai/domain.py`.

## `savings/`: product sheets, rates and questions

Each sheet is a Markdown file. Its head says which product it is, which version, and from which day that version applies. Then come numbered sections:

```
# Easy Access Saver

Product: easy-access
Version: 3
In force from: 2026-09-01

## 4. Withdrawals

You can take money out at any time, ...
```

A section is cited as `easy-access#4`. The sheets hold no interest rates. Those are in `rates.csv`, one row per product and start date, and a row is cited as `rates#easy-access/2026-09-01`.

`questions.jsonl` has one JSON object per line:

| Field | Meaning |
|---|---|
| `id` | 1 to 39, in file order |
| `kind` | `terms` (22 questions answered by a section), `rate` (7 answered by the rates table) or `escalate` (10 that must go to a person) |
| `question` | what the customer writes |
| `answer` | a short answer, for people reading the file |
| `keys` | what a right answer must contain: a list of groups, and one spelling from every group has to be there (`answer_has_keys` in `domain.py`) |
| `refs` | the sections or rate rows that hold the answer |
| `product`, `date` | for `rate` questions: the arguments `get_rate` needs |
| `reason` | for `escalate` questions: `complaint`, `vulnerability`, `advice` or `bereavement` |

## `emails/`: emails labelled complaint and vulnerable

One JSON object per line: `id`, `split` (`train`, `val` or `test`), `text`, and two labels, `complaint` and `vulnerable` (each `true` or `false`). The 400 template emails also have `product` and `styles`, which says how each label shows in the words: `explicit` (with a word like *complaint* or *passed away*), `implicit` or `soft` (without one), and `tricky` (a trigger word in an email that is neither).

| | train | val | test | handwritten |
|---|---|---|---|---|
| neither | 107 | 16 | 29 | 6 |
| complaint | 91 | 6 | 29 | 6 |
| vulnerable | 38 | 9 | 12 | 6 |
| both | 44 | 9 | 10 | 6 |
| **all** | **280** | **40** | **80** | **24** |

## `servicedesk/`: tickets and articles

Each article in `kb/` has a title, an `ID`, a `Category` and a `Last reviewed` date. Each ticket in `tickets.jsonl` has `id`, `split` (`train` 150, `test` 50), `text`, `category` (one of `access`, `email`, `hardware`, `network`, `security`, `software`) and `kb`: the article that solves it, or `null` when no article does (78 of the 200).

## How was it made?

The product sheets, the FAQ, the rates, the 39 questions, the 10 articles and the 24 handwritten emails were written by hand for this course.

`emails.jsonl` and `tickets.jsonl` are written by a script, from lists of sentences and a seeded random number generator:

```
uv run python scripts/make_domain_data.py           # writes both files
uv run python scripts/make_domain_data.py --check   # exits 1 if the files differ from what it would write
```

The same seed always gives the same bytes, and `tests/test_domain.py` checks that the committed files are what the script writes.

The split into training and test is made **by wording**: about a quarter of the sentences in each list are kept for the test set only. So a model is tested on wordings it never trained on. Splitting the finished texts at random would put near-copies of the training texts into the test set, and every score would look better than it is.

## What are its limits?

- **Template text is easier than real text.** A few dozen sentence patterns with slots can't match the range of real mail: no long threads, no quoted replies, no signatures, no typing mistakes, one topic per message. The 24 handwritten emails are there to show the gap, and notebook 10_02 measures it. They were still written by one person in an afternoon.
- **The labels are mine.** In a real firm, the people who own the process decide what counts as a complaint or a sign of vulnerability, and they disagree on the hard cases. Here one author decided.
- **The sets are small.** One test email is 1.25 points of a score, one test ticket is 2 points, one support question about 2.6.

So the numbers in chapter 10 show a method. They say nothing about how well it would work on a real firm's mail.

## Licence

All files in this folder are our own work, under the project's [MIT licence](../../LICENSE).
