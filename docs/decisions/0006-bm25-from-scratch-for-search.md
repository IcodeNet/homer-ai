# 0006. BM25 from scratch for the search of chapter 9, embeddings as an optional comparison

Status: Accepted

## Context

Chapter 9 answers English questions about Homer from the real text: a search fetches passages, and a model (Claude, through the MCP tool `search_homer`) answers from them and cites book and line. The search has to be chosen.

The usual choice today is a pretrained embedding model plus a vector index. For this course that has costs: a new dependency of several hundred megabytes with its own model download, a search whose scores the reader can't work out by hand, and a CI run that either downloads a model or skips the chapter. The course's own embeddings are no alternative: they are one vector per character, and notebook 09_02 shows that their average can't tell a word from its scrambled letters.

Numbers from notebook 09_02, on the 40 labelled questions of `data/eval/questions.jsonl` (2,140 English passages, both poems):

| Search | recall@1 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|
| BM25, with Butler's Roman names added to the question | 0.350 | 0.550 | 0.575 | 0.420 |
| `all-MiniLM-L6-v2` sentence embeddings (measured 24 November 2026, `sentence-transformers` 6.3.0) | 0.275 | 0.475 | 0.525 | 0.336 |
| both, merged by reciprocal rank fusion | 0.275 | 0.575 | 0.650 | 0.415 |

## Decision

- The search is BM25, written from scratch in `src/homer_ai/retrieval.py` with the standard library only: passages, a tokenizer that lowercases and strips accents, an inverted index, and the scoring formula (`k1 = 1.5`, `b = 0.75`).
- English passages are single lines of the files (one paragraph each); Greek passages are 5 verses. No passage crosses a book boundary, and every passage carries `poem book:line`.
- A 16-entry dictionary adds Butler's Roman names to Greek names in the question (`odysseus` → `ulysses`).
- The index is built on the first search and kept in memory, so `homer-serve` starts as fast as it does without the search.
- A pretrained sentence-embedding model appears once, as an optional comparison cell in 09_02. It is behind the optional extra `rag` in `pyproject.toml`; without it the cell prints one line. CI does not install it.
- The search is scored on labelled questions, and `tests/test_retrieval.py` pins the BM25 scores, so a change to the search has to show its effect.

## Consequences

- No new dependency and no download: the chapter's notebooks run in seconds, in CI too.
- Every score can be checked by hand, which notebook 09_01 does on three real passages.
- The search matches words, not meaning. It misses a passage when the question and the text use different words, which happens for about half of the labelled questions. The tool's description says so, and tells the model to search again with other words.
- An English question finds nothing in the Greek text. Greek search works with Greek words only.
- On this small set the embedding model was not better than BM25, and the merged search was only a little better. With 40 questions, one question is 2.5 points, so this does not show that BM25 is the better method in general.
- A reader who wants hybrid search has the pieces: the optional cell shows the fusion. Making it the default would mean accepting the dependency and the download for everyone, and a new record.
