# Labelled questions about Homer

`questions.jsonl` holds 40 questions about the two poems, 20 each, with the places in the text that answer them. Chapter 9 uses them to score its search
([09_02](../../notebooks/09_advanced_grounded_answers/09_02_search_and_scores.ipynb)) and to compare Claude with and without the retrieved passages
([09_04](../../notebooks/09_advanced_grounded_answers/09_04_with_and_without.ipynb)).

## File format

One JSON object per line:

```
{"id": 4, "poem": "odyssey", "question": "What false name does Odysseus give to the Cyclops?", "answer": "Noman", "refs": [[9, 24, 25]]}
```

| Field | Meaning |
|---|---|
| `id` | 1 to 40, in file order |
| `poem` | `odyssey` or `iliad` |
| `question` | the question, in English, with the Greek names a reader would use (Odysseus, Athena, Zeus) |
| `answer` | a short answer, for people reading the file; the scores don't use it |
| `refs` | the places that answer it: a list of `[book, first line, last line]` |

The line numbers are lines of the English files in `data/homer/<poem>/english/book_NN.txt`, counted from 1. One line there is one paragraph of Butler's prose.
They are not the verse numbers of printed editions. A question can have several places, when the poem tells the thing more than once.

A search result counts as right when it is from that poem and book and overlaps one of the line ranges (`is_hit` in `src/homer_ai/retrieval.py`).

## How it was made

The questions were written first, the way a reader would ask them. Each answer was then looked up in the English text, and the book and lines were written down from the files.
`tests/test_retrieval.py` checks that every place points at lines that exist.

Two limits. The set is small: one question is 2.5 points of any score. And a passage that isn't listed can still hold an answer, so read the misses before trusting a number.

## Licence

The questions, answers and line references are our own work, under the project's [MIT licence](../../LICENSE). The texts they point into keep their own licences: see [data/homer/LICENSE.md](../homer/LICENSE.md).
