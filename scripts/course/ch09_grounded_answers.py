# ruff: noqa: E501
"""Chapter 9, Advanced: answers grounded in Homer's text (search, citations, MCP)."""

from .cells import (
    background,
    code,
    exercise,
    learn,
    md,
    q,
    questions,
    recap,
    save,
    saw,
    step_intro,
    title,
    where,
    words,
)
from .diagrams import DIAGRAM_BM25, DIAGRAM_RAG

FOLDER = "09_advanced_grounded_answers/"
HELPERS = "The finished code lives in `src/homer_ai/retrieval.py`. It uses only Python's standard library, so there's nothing new to install."

# The numbers below were measured on the 40 questions of data/eval/questions.jsonl.
# tests/test_retrieval.py checks the BM25 row, so a change to the search shows up there.
SCORE_TABLE = (
    "| Search | recall@1 | recall@5 | recall@10 | MRR |\n|---|---|---|---|---|\n"
    "| BM25, the question as it is | 0.300 | 0.525 | 0.600 | 0.384 |\n"
    "| BM25, plus Butler's names | 0.350 | 0.550 | 0.575 | 0.420 |\n"
)

TOY = r"""from homer_ai.retrieval import load_passages, tokenize

by_ref = {p.ref: p for p in load_passages("odyssey", "english")}
toy = [by_ref["odyssey 9:25"], by_ref["odyssey 9:28"], by_ref["odyssey 10:18"]]
toy_words = [tokenize(p.text) for p in toy]"""


def build():
    # ---------------------------------------------------------------- 09_00
    save(
        FOLDER + "09_00_chapter_overview.ipynb",
        [
            title("09_00 Advanced: answers grounded in Homer's text (overview)"),
            where(
                None,
                [
                    "src/homer_ai/retrieval.py",
                    "src/homer_ai/homer_mcp_server.py",
                    "data/eval/questions.jsonl",
                    "scripts/benchmark_grounding.py",
                ],
                "**after the main course**: real questions, answered from the real text",
            ),
            learn(
                [
                    "What a grounded answer is, and why it needs a search step before the model",
                    "Why our own Homer model can't answer a question, and what it can still add",
                    "The parts we build: passages, a word search called BM25, a way to score the search, and an MCP tool",
                    "What results to expect, including the ones that surprised me",
                ]
            ),
            words(
                [
                    (
                        "retrieval",
                        "finding the pieces of text that fit a question, before anybody answers it",
                        "the 5 paragraphs of Homer that best match your question",
                    ),
                    (
                        "grounded answer",
                        "an answer written from text that was put in front of the model, not from the model's memory",
                        "Claude answers from 5 passages we handed it",
                    ),
                    (
                        "citation",
                        "the place an answer came from, written so you can look it up",
                        "`odyssey 9:24`: the Odyssey, Book 9, line 24 of our file",
                    ),
                    (
                        "passage",
                        "a short piece of a text that is searched and shown as one unit; also called a chunk",
                        "one paragraph of the English, or 5 verses of the Greek",
                    ),
                    (
                        "index",
                        "the tables a search prepares once, so that each question is answered fast",
                        "for every word, the passages that contain it",
                    ),
                ]
            ),
            *background(),
            q(
                "What are we building?",
                "You ask a question about Homer in English, and you get an answer that comes from the real text, with the place written next to it: *Odyssey, Book 9, line 24*. So you can open the book and check.\n\n"
                "Here's the picture I use. Think of a closed-book exam and an open-book exam.\n\n"
                "In a closed-book exam you answer from memory. That's what a language model does when you just ask it. It may be right, but you can't see where the answer came from, and when its memory is wrong it sounds exactly as sure.\n\n"
                "In an open-book exam, a librarian first fetches the few pages that fit the question. You read them, you answer, and you write the page number beside your answer. "
                "That's this chapter. The librarian is a search program we write ourselves. The student is Claude. The page number is the **citation**.\n\n"
                "| In the picture | In this chapter | Notebook |\n|---|---|---|\n"
                "| the book, cut into pages that know their number | both poems as **passages**, each with its book and line | [09_01](09_01_chunking_and_bm25.ipynb) |\n"
                "| the librarian who fetches the right pages | a word search called **BM25**, built from scratch | [09_01](09_01_chunking_and_bm25.ipynb), [09_02](09_02_search_and_scores.ipynb) |\n"
                "| checking how often the librarian brings the right page | 40 labelled questions, **recall@k** and **MRR** | [09_02](09_02_search_and_scores.ipynb) |\n"
                "| a desk where any student can ask the librarian | the MCP tool `search_homer` | [09_03](09_03_the_search_tool.ipynb) |\n"
                "| the exam, with the book open and closed | Claude with and without the passages | [09_04](09_04_with_and_without.ipynb) |\n\n"
                "The picture has a limit. Our librarian can't read. It matches words: ask about *pigs* and it finds pages with the word *pigs*. A page that says the same thing in other words stays on the shelf. "
                "You'll see exactly how often that happens, because we measure it.\n\n"
                "The usual name for all of this is **RAG**, retrieval-augmented generation ([08_00](../08_advanced_fine_tuning/08_00_chapter_overview.ipynb) put it on step 2 of the ladder). The next cell draws it.",
            ),
            code(DIAGRAM_RAG),
            saw(
                "Two steps. Step 1, **retrieve**: the question goes to the search, which picks the 5 best of the 2,140 English passages. Each passage carries its place, like `odyssey 9:24`. "
                "Step 2, **answer**: Claude gets the question and those 5 passages, and writes an answer with the place it used. No model is trained anywhere in this chapter."
            ),
            q(
                "We built a Homer model. Why doesn't it answer the question?",
                "Because it can't, and it's better to see that than to take my word for it.\n\n"
                "Our GPT reads one character at a time, and it learned from Book 1 of the Odyssey in Greek. So it knows which letter tends to follow which. It has no idea what a question is, and it has never seen most English letters.\n\n"
                "👉 **Predict first:** we give it `Who blinded the Cyclops?`. How many of those 24 characters does it know?",
            ),
            code(
                r"""import torch

from homer_ai.generation import next_scores
from homer_ai.lessons import load_homer_gpt

tokenizer, model = load_homer_gpt()  # read-only
question = "Who blinded the Cyclops?"
ids = tokenizer.encode(question)
unknown = sum(i == tokenizer.unk_id for i in ids)
print(f"{len(ids)} characters, {unknown} of them unknown to the model")
print("what it reads:", tokenizer.decode(ids))

generated = list(ids)
with torch.no_grad():  # just guessing, not learning
    for _ in range(40):
        generated.append(torch.argmax(next_scores(model, generated)).item())
print("what it adds: ", repr(tokenizer.decode(generated[len(ids) :])))"""
            ),
            saw(
                "It knows 3 of the 24 characters: the three spaces. Every letter of the question, and the question mark, is `<|unk|>` to it, because Book 1 has no Latin letters. "
                "Then it does the only thing it can: it adds Greek letters that look like Homer. That isn't an answer to anything.\n\n"
                "A bigger model of the same kind would still have the first problem of the closed-book exam: no page number."
            ),
            q(
                "So is our model out of the picture?",
                "Almost. It keeps one small job. Our MCP server already has the tool `consult_homer_llm` from [06_03](../06_making_it_talk/06_03_the_mcp_server.ipynb), which continues a Greek seed in Homer's style. "
                "This chapter adds a second tool to the same server, `search_homer`. So Claude can fetch the real passage with one tool, and, if you ask for it, get a few made-up Greek letters in Homer's style from the other.\n\n"
                "The answer itself always comes from the real text. The model's Greek is decoration, and the tool's description says so.",
            ),
            q(
                "Does search mean embeddings and a vector database?",
                "Often, yes. Not here, and that's a choice.\n\n"
                "We start with a search that counts words: **BM25**. It's what search engines used for decades, it fits in about 40 lines of plain Python, and you can check every number by hand. "
                "In [09_02](09_02_search_and_scores.ipynb) we compare it, on the same questions, with a small pretrained model that turns sentences into vectors. I expected the vectors to win easily. They didn't.\n\n"
                "We also try the vectors we already have, the token embeddings of our own model. They fail completely, and it's worth seeing why.",
            ),
            q(
                "What do I need on my computer?",
                "Nothing new. No download, no training, and every notebook of this chapter runs in a few seconds.\n\n"
                "- The text is already in `data/homer/`: both poems, in Greek and in Samuel Butler's English.\n"
                "- " + HELPERS + "\n"
                "- One cell in 09_02 is optional. It needs the `sentence-transformers` library and a model of about 90 MB. Without them it prints one line and moves on. To try it: `uv sync --extra rag`.\n"
                "- [09_04](09_04_with_and_without.ipynb) describes a script that calls Claude. You don't have to run it. The notebook shows one recorded run.",
            ),
            q(
                "How good will the results be?",
                "Here's what I measured, on 40 questions I wrote and labelled by hand:\n\n"
                "| Notebook | What we measure | Result |\n|---|---|---|\n"
                "| 09_02 | is a right passage among the first 5 the search returns? | 22 of 40 questions (0.55) |\n"
                "| 09_02 | the same, with a pretrained sentence model in place of BM25 | 19 of 40 (0.475) |\n"
                "| 09_02 | the same, with our own model's letter vectors | useless: a word and its scrambled letters get the same vector |\n"
                "| 09_04 | Claude names the right book, without and with our passages (8 questions) | 8 of 8 without, 5 of 8 with |\n\n"
                "Read the last row twice. Claude has read Homer, so on a famous poem the closed-book exam goes very well, and our half-good search can even hold it back. "
                "What the open book adds is a line you can check, and an honest *the passages don't say* when the search missed. On text a model has never read, like your own documents, there is no closed-book option at all. 09_04 goes through it.",
            ),
            q(
                "Which notebook do I open for each idea?",
                "| Notebook | Question it answers |\n|---|---|\n"
                "| [`09_01_chunking_and_bm25.ipynb`](09_01_chunking_and_bm25.ipynb) | How do I cut a text into passages, and how does a word search score them? |\n"
                "| [`09_02_search_and_scores.ipynb`](09_02_search_and_scores.ipynb) | How good is the search, measured on real questions, and would vectors do better? |\n"
                "| [`09_03_the_search_tool.ipynb`](09_03_the_search_tool.ipynb) | How do Claude and other AI apps get to use the search? |\n"
                "| [`09_04_with_and_without.ipynb`](09_04_with_and_without.ipynb) | Does it help? And what would this look like on a company's own documents? |",
            ),
            recap(
                [
                    "A grounded answer is written from text that a search step fetched, and it says where the text is.",
                    "Our character-level model can't answer questions. It doesn't even know the letters of an English question.",
                    "The search is BM25: word counting, built from scratch, no new library.",
                    "We measure the search on 40 labelled questions before we trust it, and we report the numbers as they came.",
                ],
                "[09_01: Passages, and a word search from scratch](09_01_chunking_and_bm25.ipynb).",
            ),
        ],
    )

    # ---------------------------------------------------------------- 09_01
    save(
        FOLDER + "09_01_chunking_and_bm25.ipynb",
        [
            title("09_01 Passages, and a word search from scratch (BM25)"),
            where(
                None,
                ["src/homer_ai/retrieval.py", "data/homer/"],
                "**after the main course**: cutting the book into pages, and teaching the librarian to fetch",
            ),
            learn(
                [
                    "How to cut a text into passages that remember their book and line",
                    "How to turn text into a bag of words, and why counting words isn't enough",
                    "What tf, idf and BM25 are, worked out by hand on 3 real passages",
                    "How an index makes the search fast on all 2,140 passages",
                ]
            ),
            words(
                [
                    (
                        "chunking",
                        "cutting a long text into passages",
                        "Book 1 of the Odyssey: 444 verses → 89 passages of 5",
                    ),
                    (
                        "bag of words",
                        "a text seen only as its words and how often each appears; the order is thrown away",
                        "`noman: 3, comrades: 1, …`",
                    ),
                    (
                        "tf",
                        "term frequency: how often a word is in one passage",
                        "`noman` is 3 times in `odyssey 9:25`",
                    ),
                    (
                        "idf",
                        "inverse document frequency: how much a word is worth; high when few passages have it",
                        "`pigs` is worth more than `the`",
                    ),
                    (
                        "BM25",
                        "a formula that scores a passage against the words of a question, from tf, idf and the passage's length",
                        "what search engines used long before neural networks",
                    ),
                    (
                        "inverted index",
                        "a table that lists, for each word, the passages that contain it",
                        "`moly → [odyssey 10:24]`",
                    ),
                ]
            ),
            *background(),
            q(
                "What does the librarian need before the first question?",
                "Two things. Pages to fetch, and a way to decide which page fits a question.\n\n"
                "The pages are **passages**: short pieces of the poems, each one tagged with where it came from. Without the tag there's no citation later.\n\n"
                "The way to decide is a score. For a question and a passage, we want one number that is high when the passage fits. We build that number in three tries, each fixing what the one before got wrong:\n\n"
                "```\n"
                "1. count the question's words in the passage      (bag of words)\n"
                "2. make rare words worth more                     (idf)\n"
                "3. stop repeats and long passages from winning    (BM25)\n"
                "```\n\n"
                "In the picture of [09_00](09_00_chapter_overview.ipynb), this notebook cuts the book into pages and teaches the librarian to fetch.",
            ),
            step_intro(),
            q(
                "Step 1: what do the files look like?",
                "Each book is one text file. In the Greek files a line is one verse. In Butler's English, which is prose, a line is one whole paragraph. "
                "[07_01](../07_advanced_iliad_and_combining/07_01_the_corpus.ipynb) looked at these files; here we only need the line counts.\n\n"
                "👉 **Predict first:** Book 9 of the Odyssey (the Cyclops) has 566 Greek verses. Roughly how many English lines?",
            ),
            code(
                r"""from homer_ai.retrieval import HOMER

greek = (HOMER / "odyssey" / "greek" / "book_09.txt").read_text(encoding="utf-8")
english = (HOMER / "odyssey" / "english" / "book_09.txt").read_text(encoding="utf-8")
greek_lines, english_lines = greek.splitlines(), english.splitlines()

print(f"{'':8} {'lines':>6} {'characters in an average line':>30}")
for name, lines in (("greek", greek_lines), ("english", english_lines)):
    print(f"{name:8} {len(lines):>6} {sum(map(len, lines)) / len(lines):>30.0f}")
print()
print("greek   line 1:", greek_lines[0])
print("english line 1:", english_lines[0][:60] + "…")"""
            ),
            saw(
                "566 Greek lines of about 44 characters, and only 44 English lines of about 670. An English line is a paragraph, well over 100 words. A Greek line is one verse, around 7 words.\n\n"
                "So the two languages need different cuts. A paragraph is already a good passage. One verse is too little to search or to answer from."
            ),
            q(
                "Step 2: how do I cut lines into passages?",
                "Take the lines in runs of a fixed size, and write down the first and last line number of each run. For the Greek we use 5 verses.\n\n"
                "Here are the first 12 verses of the Odyssey, cut by 5:\n\n"
                "```\n"
                "lines   :  1  2  3  4  5 |  6  7  8  9 10 | 11 12\n"
                "passage :  odyssey 1:1-5 | odyssey 1:6-10 | odyssey 1:11-12\n"
                "```\n\n"
                "The last run is short, and that's fine. The reference `odyssey 1:6-10` reads: the Odyssey, Book 1, lines 6 to 10 of our file.\n\n"
                "👉 **Predict first:** all of Book 1 has 444 verses. How many passages of 5 is that?",
            ),
            code(
                r"""def chunk_lines(lines, size):
    chunks = []
    for start in range(0, len(lines), size):
        part = lines[start : start + size]
        chunks.append((start + 1, start + len(part), "\n".join(part)))
    return chunks


book1 = (HOMER / "odyssey" / "greek" / "book_01.txt").read_text(encoding="utf-8")
book1_lines = book1.splitlines()
for first, last, text in chunk_lines(book1_lines[:12], 5):
    print(f"{f'odyssey 1:{first}-{last}':16} {text.splitlines()[0]}")
print()
print("all of Book 1:", len(chunk_lines(book1_lines, 5)), "passages")"""
            ),
            saw(
                "Three passages from 12 lines, as in the picture, each shown by its first verse. All of Book 1 gives 89: that's 88 full ones (440 verses) and one of 4.\n\n"
                "`start + 1` is there because Python counts from 0 and people count lines from 1."
            ),
            q(
                "Step 3: what about the end of a book?",
                "A passage never crosses it. We cut each book's file by itself, so the last passage of Book 1 is short, and the first passage of Book 2 starts at line 1.\n\n"
                "If a passage ran from the end of one book into the next, its reference would be wrong, and it would mix two scenes that don't belong together. It's the same reason [02_02](../02_text_to_numbers/02_02_special_tokens.ipynb) put `<|endoftext|>` between books.\n\n"
                "`load_passages` in `retrieval.py` does this for a whole poem: one line per passage for the English, 5 for the Greek. Each passage is a small record with `poem`, `language`, `book`, `first`, `last`, `text`, and a `ref`.",
            ),
            code(
                r"""from homer_ai.retrieval import load_passages

print(f"{'':10} {'english':>8} {'greek':>8}")
for poem in ("odyssey", "iliad"):
    counts = [len(load_passages(poem, language)) for language in ("english", "greek")]
    print(f"{poem:10} {counts[0]:>8,} {counts[1]:>8,}")

passages = load_passages("odyssey", "english")
passage = passages[1]  # the second paragraph of Book 1
print()
print(passage.ref, "|", passage.text[:70] + "…")"""
            ),
            saw(
                "1,027 English passages for the Odyssey and 1,113 for the Iliad: 2,140 together, one per paragraph. The Greek gives 2,432 and 3,149.\n\n"
                "The record for `odyssey 1:2` is the paragraph that begins *So now all who escaped death in battle…*. That reference is what Claude will cite later."
            ),
            q(
                "Step 4: how does text become words?",
                "Lowercase it, drop the accents, and keep runs of letters. That's `tokenize` in `retrieval.py`. (It's a much simpler job than the tokenizers of chapter 2. Here a token is a whole word, and nothing has to be turned back into text.)\n\n"
                "Dropping accents matters for the Greek. `μῆνιν` and `μηνιν` should be the same word to a search, and nobody wants to type breathings into a question.\n\n"
                "👉 **Predict first:** what happens to the `’s` in `Noman’s`?",
            ),
            code(
                r"""from homer_ai.retrieval import tokenize

print(tokenize("‘Then I will eat all Noman’s comrades before Noman himself’"))
print(tokenize("μῆνιν ἄειδε θεὰ Πηληϊάδεω Ἀχιλῆος"))"""
            ),
            saw(
                "The quote marks are gone, everything is lowercase, and `Noman’s` became two words, `noman` and `s`. That lone `s` is noise, but harmless: it's in so many passages that it will count for almost nothing. You'll see why in step 6.\n\n"
                "The Greek lost its accents and breathings: `μηνιν`, `αειδε`, `θεα`."
            ),
            q(
                "Step 5: can't we just count the question's words?",
                "Let's try, on three real passages. Two are from the Cyclops story, where Odysseus has said his name is *Noman*. The third is from Circe's island.\n\n"
                "```\n"
                "odyssey 9:25   ‘Then I will eat all Noman’s comrades before Noman himself, and will keep Noman for the last. …’\n"
                "odyssey 9:28   ‘Noman is killing me by fraud; no man is killing me by force.’\n"
                "odyssey 10:18  … Circe threw them some acorns and beech masts such as pigs eat, but Eurylochus hurried back to tell me about the sad fate of our comrades. …\n"
                "```\n\n"
                "The question is *who is killing Noman?* We look at two of its words, `noman` and `killing`, and count them in each passage. A text seen this way, as words and counts with the order thrown away, is called a **bag of words**. The count of one word in one passage is its **tf**, term frequency.\n\n"
                "👉 **Predict first:** the passage that answers is `odyssey 9:28`. Will it get the highest count?",
            ),
            code(
                TOY
                + r"""

query = ["noman", "killing"]
print(f"{'':14} {'noman':>6} {'killing':>8} {'sum':>5} {'words':>6}")
for p, words in zip(toy, toy_words, strict=True):
    tf = [words.count(word) for word in query]
    print(f"{p.ref:14} {tf[0]:>6} {tf[1]:>8} {sum(tf):>5} {len(words):>6}")"""
            ),
            saw(
                "A tie. `odyssey 9:25` says `noman` three times and gets 3. `odyssey 9:28`, the one that answers, has `noman` once and `killing` twice, and also gets 3.\n\n"
                "Plain counting treats every word the same. But `killing` is the word that tells these passages apart, and `noman` is in both. The count should know that."
            ),
            q(
                "Step 6: how do I make a rare word worth more?",
                "Count in how many passages each word appears, and turn that into a weight. A word that's in few passages gets a high weight. A word that's in nearly all of them gets a weight near 0. This weight is the **idf**, inverse document frequency (a *document* is a passage here).\n\n"
                "With `N` passages in all, and `n` of them containing the word:\n\n"
                "```\n"
                "idf = log(1 + (N - n + 0.5) / (n + 0.5))\n"
                "```\n\n"
                "Read it from the inside. `(N - n) / n` is *passages without the word* divided by *passages with it*: big for a rare word. The two `0.5`s keep the division safe when `n` is 0. "
                "`log` (the natural logarithm, met in [05_01](../05_training_on_the_odyssey/05_01_loss_and_epochs.ipynb)) squeezes the result, so a word that's 100 times rarer isn't worth 100 times more. And the `1 +` keeps the weight above 0.\n\n"
                "With our 3 passages:\n\n"
                "| word | in how many of 3 | the maths | idf |\n|---|---|---|---|\n"
                "| `noman` | 2 | log(1 + 1.5 / 2.5) | 0.47 |\n"
                "| `killing` | 1 | log(1 + 2.5 / 1.5) | 0.98 |\n"
                "| `the` | 3 | log(1 + 0.5 / 3.5) | 0.13 |\n\n"
                "Then the score of a passage is, for each question word, `tf × idf`, added up. That's **TF-IDF**.",
            ),
            code(
                r"""import math

N = len(toy)


def idf(word):
    n = sum(word in words for words in toy_words)
    return math.log(1 + (N - n + 0.5) / (n + 0.5))


for word in ("noman", "killing", "the"):
    print(f"idf({word}) = {idf(word):.2f}")
print()
for p, words in zip(toy, toy_words, strict=True):
    parts = [words.count(word) * idf(word) for word in query]
    print(f"{p.ref:14} {parts[0]:.2f} + {parts[1]:.2f} = {sum(parts):.2f}")"""
            ),
            saw(
                "The same three idf values as in the table. And the tie is broken: `odyssey 9:28` scores 0.47 + 1.96 = 2.43, ahead of `odyssey 9:25` with 1.41. The passage with the telling word wins.\n\n"
                "Look at `the`: 0.13. Words that are everywhere fade out by themselves, so we never need a list of words to ignore. (That's also what happens to the stray `s` of step 4.)"
            ),
            *exercise(
                "can you score a passage for a new question?",
                "A new question, two words: `noman` and `comrades`. Work out the TF-IDF score of the first toy passage (`odyssey 9:25`) for it.\n\n"
                "👉 **Predict first:** `noman` is in this passage 3 times and `comrades` once. Both have an idf of 0.47. What's the score?",
                starter="""
            new_query = ["noman", "comrades"]

            # 👉 do: the TF-IDF score of toy_words[0] for new_query: tf × idf per word, added up
            # 💡 hint: toy_words[0] is the passage as a list of words, and a list can count() a word
            tfidf_score = None
            """,
                solution="""
            new_query = ["noman", "comrades"]

            tfidf_score = sum(toy_words[0].count(word) * idf(word) for word in new_query)
            """,
                check="""
            check(
                "TF-IDF score of odyssey 9:25",
                tfidf_score,
                3 * idf("noman") + 1 * idf("comrades"),
                coach="Got 0.94? That's the two idf values without the counts: noman is in "
                "this passage 3 times, so its idf counts 3 times. Got 4? Those are the "
                "counts without the idf.",
            )
            """,
                hint="The last loop of the cell above does it for every passage: `words.count(word) * idf(word)` for each word of the query. Here `words` is `toy_words[0]`.",
                answer="`3 × 0.47 + 1 × 0.47 = 1.88`. Notice that three `noman`s count exactly three times as much as one. "
                "Is the third mention really worth as much as the first? That's the first thing BM25 changes.",
            ),
            q(
                "Step 7: what's still wrong, and what does BM25 change?",
                "Two things are still wrong with `tf × idf`.\n\n"
                "**Repeats count too much.** A passage that says `noman` three times gets three times the points of one that says it once. But it isn't three times more about Noman. The first mention tells you a lot, and the third tells you little.\n\n"
                "**Long passages win too easily.** A paragraph of 400 words has more chances to contain any word than one of 20 words. That makes it longer. It doesn't make it a better fit.\n\n"
                "BM25 keeps the idf and swaps the plain `tf` for this:\n\n"
                "```\n"
                "            tf × (k1 + 1)\n"
                "idf  ×  ─────────────────────────────────────────────\n"
                "         tf + k1 × (1 - b + b × length / average)\n"
                "```\n\n"
                "| Piece | What it does |\n|---|---|\n"
                "| `tf` above and below the line | the fraction grows with `tf`, but ever more slowly; it can never pass `k1 + 1` |\n"
                "| `k1` (we use 1.5) | how fast repeats stop counting; a smaller `k1` flattens sooner |\n"
                "| `length / average` | the passage's word count against the average passage; 1 for an average one |\n"
                "| `b` (we use 0.75) | how much length matters; with `b = 0` length is ignored |\n\n"
                "For a passage of average length the bracket is exactly 1, and the fraction is `tf × 2.5 / (tf + 1.5)`: 1.0 for one mention, 1.43 for two, 1.67 for three. Let's draw it.",
            ),
            code(DIAGRAM_BM25),
            saw(
                "The dashed line is plain counting: every repeat adds the same. The three curves are BM25. They rise fast at first and then flatten under the dotted ceiling at 2.5.\n\n"
                "The green curve is a passage half the average length, the orange one is average, and the pink one is 3 times longer. For the same count, the long passage always earns less."
            ),
            q(
                "Step 8: does it give those numbers on our three passages?",
                "We write the formula as a function, score the three passages, and then let the finished `BM25` class from `retrieval.py` do the same. The two must agree.\n\n"
                "This time we score three question words: `noman`, `killing` and `comrades`. The word `comrades` is once in `odyssey 9:25` (32 words) and once in `odyssey 10:18` (86 words).\n\n"
                "👉 **Predict first:** same word, same count. Which of those two passages gets more points for `comrades`?",
            ),
            code(
                r"""from homer_ai.retrieval import BM25

K1, B = 1.5, 0.75
average = sum(len(words) for words in toy_words) / N


def bm25_part(word, words):
    tf = words.count(word)
    length = 1 - B + B * len(words) / average
    return idf(word) * tf * (K1 + 1) / (tf + K1 * length)


query = ["noman", "killing", "comrades"]
index = BM25([p.text for p in toy])  # the finished class
from_class = index.scores(query)

print(f"{'':14} {'noman':>6} {'killing':>8} {'comrades':>9} {'sum':>6} {'class':>6}")
for number, (p, words) in enumerate(zip(toy, toy_words, strict=True)):
    parts = [bm25_part(word, words) for word in query]
    print(
        f"{p.ref:14} {parts[0]:>6.2f} {parts[1]:>8.2f} {parts[2]:>9.2f} "
        f"{sum(parts):>6.2f} {from_class.get(number, 0):>6.2f}"
    )"""
            ),
            saw(
                "The last two columns, `sum` and `class`, match, so the class does what our hand-written function does.\n\n"
                "Read the `noman` column first. Three mentions in `odyssey 9:25` earn 0.85, and one mention in `odyssey 9:28` earns 0.62. Three times the count gives about a third more points. That's the flattening (the second passage is also a little shorter, which helps it).\n\n"
                "Now `comrades`: 0.55 in the short passage, 0.34 in the long one, for the same single mention. That's the length part.\n\n"
                "`odyssey 9:28` still wins overall with 2.30, thanks to `killing`."
            ),
            q(
                "Step 9: how does this stay fast on 2,140 passages?",
                "By not looking at most of them. When the index is built, we go through every passage once and fill a table: for each word, the passages that contain it and how often. It's called an **inverted index**, and it's the same idea as the index at the back of a book.\n\n"
                "```\n"
                "moly    → [(odyssey 10:24, 1 time)]\n"
                "sirens  → [(odyssey 12:5, 2 times), (odyssey 12:14, 1 time), …]\n"
                "the     → nearly every passage\n"
                "```\n\n"
                "For a question we only visit the rows of its own words, and add up the BM25 parts per passage. A rare word touches a handful of passages. That table is `BM25.postings`.",
            ),
            code(
                r"""import time

both = load_passages("odyssey", "english") + load_passages("iliad", "english")
started = time.perf_counter()
index = BM25([p.text for p in both])
print(f"{len(both):,} passages indexed in {time.perf_counter() - started:.2f} seconds")
print(f"{len(index.postings):,} different words")
print(f"an average passage has {index.average:.0f} words")
print()
for word in ("moly", "sirens", "noman", "ulysses", "the"):
    n = len(index.postings[word])
    print(f"{word:8} in {n:>5,} passages   idf {index.idf(word):.2f}")"""
            ),
            saw(
                "About a quarter of a second to index both poems, and that happens once. There are 9,646 different words, and an average passage has 127.\n\n"
                "`moly` is in a single passage, so its idf is high (7.26). `the` is in 2,049 of 2,140 and is worth 0.04. Look at `ulysses`: 477 passages, idf 1.50. The hero's name is so common in this text that it hardly helps a search. Remember that for the next notebook."
            ),
            q(
                "Step 10: does it find the right passage?",
                "Let's ask a real question: *What false name does Odysseus give to the Cyclops?* `index.search` returns the best passage numbers with their scores.\n\n"
                "👉 **Predict first:** which words of the question will do the work?",
            ),
            code(
                r"""question = "What false name does Odysseus give to the Cyclops?"
question_words = tokenize(question)
print(f"{'word':10} {'idf':>5}")
for word in question_words:
    print(f"{word:10} {index.idf(word):>5.2f}")
print()
for number, score in index.search(question_words, k=3):
    p = both[number]
    print(f"{score:5.1f}  {p.ref:14} {p.text[:66]}…")"""
            ),
            saw(
                "The first hit is `odyssey 9:24`, the paragraph where he says *my name is Noman*. That's the right one.\n\n"
                "The idf column shows why. `false`, `cyclops` and `name` carry the search, and `to` and `the` are worth next to nothing. (`does` gets 3.69, more than `name`. Butler rarely writes *does*. idf knows how rare a word is. It doesn't know whether the word means anything.)\n\n"
                "And `odysseus` has the highest idf of all, 8.36. That's a warning, not good news: it's what a word gets when **no passage contains it**. Butler's English never says Odysseus. [09_02](09_02_search_and_scores.ipynb) starts there."
            ),
            *questions(
                [
                    (
                        "Why 5 verses per Greek passage? Why not 3, or 20?",
                        "It's a trade. A small passage gives a precise citation but may not hold enough to answer from, and its few words make a thin target for a search. A big one is easier to find and blurrier to cite. Five verses is about one sentence or two of Homer. With labelled questions you can try other sizes and measure, which is what 09_02 is about.",
                    ),
                    (
                        "Real systems let passages overlap. Why don't we?",
                        "Overlap helps when an answer sits across a cut: with a shared verse or two, one of the passages holds all of it. It also makes the index bigger and returns near-duplicates. We leave it out to keep the line numbers simple. `chunk_lines` would need one more argument, the step between starts.",
                    ),
                    (
                        "Shouldn't `kills`, `killed` and `killing` count as the same word?",
                        "Cutting words down to a stem is called stemming, and many search engines do it. I tried a simple one here (strip `-s`, `-ed`, `-ing`) and the scores on our questions didn't improve, so it's not in the code. That's the useful habit: measure before you add a part.",
                    ),
                    (
                        "Where do `k1 = 1.5` and `b = 0.75` come from?",
                        "They're the usual starting values, found by people who tested BM25 on many collections. On our questions, other values in the normal range (`k1` from 0.9 to 2, `b` from 0.4 to 0.9) moved the scores by a question or two at most.",
                    ),
                ]
            ),
            recap(
                [
                    "A passage is a short run of lines that remembers its poem, book and lines. It never crosses a book boundary.",
                    "A bag of words keeps the words and their counts, and forgets the order.",
                    "idf makes a rare word worth more; words that are everywhere fade out by themselves.",
                    "BM25 adds two corrections: repeats count less and less, and a long passage earns less for the same count.",
                    "An inverted index lists the passages for each word, so a question only touches the passages that share a word with it.",
                ],
                "[09_02: Searching both poems, and scoring the search](09_02_search_and_scores.ipynb).",
                can=[
                    "cut a text into passages that know their book and lines",
                    "compute idf and a TF-IDF score by hand",
                    "say what BM25's two corrections do, and what `k1` and `b` set",
                ],
            ),
        ],
    )

    # ---------------------------------------------------------------- 09_02
    save(
        FOLDER + "09_02_search_and_scores.ipynb",
        [
            title("09_02 Searching both poems, and scoring the search"),
            where(
                None,
                [
                    "src/homer_ai/retrieval.py",
                    "data/eval/questions.jsonl",
                    "tests/test_retrieval.py",
                ],
                "**after the main course**: checking how often the librarian brings the right page",
            ),
            learn(
                [
                    "Why a question about Odysseus finds nothing in Butler's English, and a small fix",
                    "How to label questions with the lines that answer them",
                    "What recall@k and MRR are, worked out on 4 questions and then on 40",
                    "Why our own model's embeddings can't do this search",
                    "How a pretrained sentence model compares with BM25 (optional cell)",
                ]
            ),
            words(
                [
                    (
                        "labelled question",
                        "a question stored together with the places in the text that answer it",
                        "*Which suitor does Odysseus kill first?* → `odyssey 22:2`",
                    ),
                    (
                        "rank",
                        "the place of a passage in the search result: 1 for the first, 2 for the second",
                        "the right passage came third: rank 3",
                    ),
                    (
                        "recall@k",
                        "the share of questions for which a right passage is among the first k results",
                        "recall@5 = 0.55: for 22 of 40 questions it's in the first 5",
                    ),
                    (
                        "MRR",
                        "mean reciprocal rank: the average of 1/rank over all questions, with 0 when nothing right was found",
                        "ranks 1 and 2 → (1 + 0.5) / 2 = 0.75",
                    ),
                    (
                        "query expansion",
                        "adding words to a question before searching, so it matches the words the text uses",
                        "`odysseus` → `odysseus ulysses`",
                    ),
                    (
                        "sentence embedding",
                        "a list of numbers for a whole sentence or paragraph, made by a model trained so that texts with a similar meaning get similar lists",
                        "384 numbers per paragraph",
                    ),
                ]
            ),
            *background(),
            q(
                "The search found Noman. Are we done?",
                "No. One question that worked tells us very little. I picked it, and I picked it because it works.\n\n"
                "It's the lesson of [05_03](../05_training_on_the_odyssey/05_03_memorising_vs_understanding.ipynb) and [05_05](../05_training_on_the_odyssey/05_05_validation.ipynb) again. There, a low loss on the training text didn't prove the model had learned, so we measured on text it hadn't seen. "
                "Here, a demo that looks good doesn't prove the search is good. We need a set of questions with known answers, and a number.\n\n"
                "So this notebook does three things: it fixes one clear problem, builds the yardstick, and then uses the yardstick on our search and on two other ideas.",
            ),
            step_intro(),
            q(
                "Step 1: how do I search both poems with one call?",
                "`search` in `retrieval.py` wraps what 09_01 built. It builds the index the first time you ask and keeps it in memory. It returns passages with their scores.\n\n"
                "```\n"
                'search(question, poem="both", language="english", k=5, names=True)\n'
                "```\n\n"
                "Let's ask about the Iliad this time. (`names=False` switches off the fix of step 2, so we see the plain search first.)",
            ),
            code(
                r"""from homer_ai.retrieval import search

question = "Who is the Trojan spy caught at night by Odysseus and Diomedes?"
for passage, score in search(question, k=5, names=False):
    print(f"{score:5.1f}  {passage.ref:12} {passage.text[:70]}…")"""
            ),
            saw(
                "The first hit is `iliad 10:32`, where Ulysses tells the man they've just caught *Fear not*, and starts questioning him. The third, `iliad 10:29`, names him: Dolon. Right book, right scene. The last two are from other books and only share words like `night` and `Trojans`.\n\n"
                "Both poems are in one index, so the search also decides which poem to look in."
            ),
            q(
                "Step 2: why did `odysseus` have such a strange idf?",
                "Because Butler doesn't use that name. He translated in the 1890s, when English books called the Greek gods and heroes by their Roman names. Let's count.\n\n"
                "👉 **Predict first:** how many times does `Odysseus` appear in Butler's two translations? And `Zeus`?",
            ),
            code(
                r"""from homer_ai.retrieval import BUTLER_NAMES, get_index

passages, index = get_index("both", "english")  # built once, reused from here on
print(f"{'you say':12} {'passages':>8}    {'Butler says':12} {'passages':>8}")
for greek_name in ("odysseus", "zeus", "athena", "poseidon", "hera", "hermes"):
    roman = BUTLER_NAMES[greek_name]
    print(
        f"{greek_name:12} {len(index.postings.get(greek_name, [])):>8}    "
        f"{roman:12} {len(index.postings.get(roman, [])):>8}"
    )"""
            ),
            saw(
                "`Odysseus` is in no passage at all, and `Ulysses` is in 477. `Zeus` appears in one passage, `Jove` in 501. Athena is `Minerva`, Poseidon is `Neptune`, Hera is `Juno`, Hermes is `Mercury`.\n\n"
                "So when you type *Odysseus*, the search throws the word away. No passage has it, so it adds nothing to any score."
            ),
            q(
                "Step 3: how do I fix that?",
                "With a small dictionary. Before searching, for every Greek name in the question we add Butler's name next to it. Adding words to a question so that it matches the text is called **query expansion**.\n\n"
                "```\n"
                "question words:  who  helps  odysseus          among  the  phaeacians\n"
                "after expansion: who  helps  odysseus ulysses  among  the  phaeacians\n"
                "```\n\n"
                "`expand_names` does it, with the 16 names in `BUTLER_NAMES`. `search` uses it unless you say `names=False`.\n\n"
                "👉 **Predict first:** `ulysses` is in 477 passages, so its idf is low. Will adding it change much?",
            ),
            code(
                r"""from homer_ai.retrieval import expand_names, first_hit_rank, load_questions, tokenize

questions = load_questions()
nausicaa = questions[12]
print(nausicaa["question"])
print(expand_names(tokenize("who helps Odysseus among the Phaeacians")))
print()
for names in (False, True):
    found = [p for p, _ in search(nausicaa["question"], k=10, names=names)]
    print(f"names={names!s:5}  first three: {[p.ref for p in found[:3]]}")
    print(f"             right passage at rank: {first_hit_rank(found, nausicaa)}")"""
            ),
            saw(
                "For this question it changes a lot. Without the names, the right passage (Odysseus meeting Nausicaa in Book 6 of the Odyssey) is at rank 10. With them it's at rank 1.\n\n"
                "The word `ulysses` is weak, but here it was enough to lift Odyssey paragraphs above others that also mention a princess or the Phaeacians.\n\n"
                "Is that typical, or did I pick a lucky example again? One example can't tell us. Time for the yardstick."
            ),
            q(
                "Step 4: what is a labelled question?",
                "A question, plus the places in the text where the answer is. The file `data/eval/questions.jsonl` has 40 of them, 20 for each poem, one JSON object per line. I wrote the questions first, the way a reader would ask them, with the Greek names. Then I found each answer in Butler's text and wrote down its book and lines.\n\n"
                "A place is `[book, first line, last line]` in the English file. A question can have several: Penelope's weaving trick is told three times, in Books 2, 19 and 24.",
            ),
            code(
                r"""print(len(questions), "questions")
for q in (questions[3], questions[1]):
    print()
    for key in ("id", "poem", "question", "answer", "refs"):
        print(f"{key:9} {q[key]}")"""
            ),
            saw(
                "Question 4 has one place: `[9, 24, 25]`, Book 9, lines 24 to 25. Question 2 has three.\n\n"
                "The `answer` field is for people who read the file. The search is scored only on the places: a returned passage is **right** if it's from that poem and book and overlaps one of the line ranges. `first_hit_rank` gives the rank of the first right passage in a result, or `None`."
            ),
            q(
                "Step 5: how do I turn 40 results into one number?",
                "Two numbers are common, and both start from the rank of the first right passage for each question. Take four questions:\n\n"
                "```\n"
                "question :   A     B     C      D\n"
                "rank     :   1     3   none     2\n"
                "in top 1?:  yes    no    no     no     → recall@1 = 1/4 = 0.25\n"
                "in top 5?:  yes   yes    no    yes     → recall@5 = 3/4 = 0.75\n"
                "1 / rank :   1    0.33    0    0.5     → MRR = 1.83 / 4 = 0.46\n"
                "```\n\n"
                "**recall@k** asks: for what share of the questions is a right passage among the first `k`? It matters because of the next step of the pipeline. If we hand Claude 5 passages, recall@5 is how often the answer is on the table at all.\n\n"
                "**MRR**, mean reciprocal rank, also rewards a right passage for being near the top: rank 1 gives 1 point, rank 2 half a point, rank 3 a third, and not found gives 0. It's one number for *how far down do I have to read?*\n\n"
                "👉 **Predict first:** if question C were found at rank 10, which of the three numbers would change?",
            ),
            code(
                r"""ranks = [1, 3, None, 2]


def recall_at(k, ranks):
    return sum(rank is not None and rank <= k for rank in ranks) / len(ranks)


def mrr(ranks):
    return sum(1 / rank for rank in ranks if rank is not None) / len(ranks)


print("recall@1:", recall_at(1, ranks))
print("recall@5:", recall_at(5, ranks))
print(f"MRR:      {mrr(ranks):.3f}")
print()
ranks[2] = 10
print(f"with C at rank 10: recall@5 {recall_at(5, ranks)}, MRR {mrr(ranks):.3f}")"""
            ),
            saw(
                "0.25, 0.75 and 0.458, as in the picture. With C at rank 10, recall@1 and recall@5 stay where they were, and MRR creeps up from 0.458 to 0.483: a tenth of a point, shared over four questions. (recall@10 would go up too, from 0.75 to 1.0.)\n\n"
                "`retrieval_scores` in `retrieval.py` is these two functions together."
            ),
            q(
                "Step 6: so how good is our search?",
                "We run all 40 questions twice, without and with the names, and look at the first 10 results each time.\n\n"
                "It's the rule of [05_04](../05_training_on_the_odyssey/05_04_running_fair_experiments.ipynb): change one thing, keep everything else the same, and compare on the same questions.\n\n"
                "👉 **Predict first:** for how many of the 40 questions will a right passage be in the first 5?",
            ),
            code(
                r"""from homer_ai.retrieval import retrieval_scores


def all_ranks(find):
    # find(question text) -> passages, best first
    return [first_hit_rank(find(q["question"]), q) for q in questions]


def show(name, ranks):
    s = retrieval_scores(ranks)
    print(
        f"{name:28} {s['recall@1']:>8.3f} {s['recall@5']:>8.3f} "
        f"{s['recall@10']:>9.3f} {s['mrr']:>6.3f}"
    )


print(f"{'':28} {'recall@1':>8} {'recall@5':>8} {'recall@10':>9} {'MRR':>6}")
plain_ranks = all_ranks(lambda text: [p for p, _ in search(text, k=10, names=False)])
bm25_ranks = all_ranks(lambda text: [p for p, _ in search(text, k=10)])
show("BM25, question as it is", plain_ranks)
show("BM25, plus Butler's names", bm25_ranks)"""
            ),
            saw(
                SCORE_TABLE
                + "\nWith the names, a right passage is first for 14 of 40 questions and in the first 5 for 22 of 40. So for 18 questions, Claude would get five passages and none of them would hold the answer.\n\n"
                "That's the honest size of what we built: a search that works about half the time. And the names helped only a little: two more questions at rank 1, one more in the first 5, and one *fewer* in the first 10. The Nausicaa example of step 3 was the best case. This is why we count."
            ),
            q(
                "Step 7: which questions does it miss, and why?",
                "Numbers say how often. To learn why, you have to read the misses. Here are the questions with no right passage in the first 10.",
            ),
            code(
                r"""missed = [q for q, rank in zip(questions, bm25_ranks, strict=True) if rank is None]
print(len(missed), "questions without a right passage in the first 10")
for q in missed:
    print(f"  {q['id']:>2}  {q['question']}")
    print(f"      → {q['answer']}")"""
            ),
            saw(
                "17 misses, and most have the same shape. Take *What does Circe turn Odysseus' men into?* The answer is *pigs*, and that's the one word that would find the passage. But it's the answer, so it isn't in the question. "
                "What's left is `circe`, `turn`, `men` and `ulysses`, and `circe` alone is in 34 paragraphs.\n\n"
                "Or *Who kills Patroclus?*: the one name in it, `patroclus`, is in 124 passages. Or *How does the Iliad end?*: the only word it shares with the last paragraph is `the`.\n\n"
                "So the limit from [09_00](09_00_chapter_overview.ipynb) is real. Our librarian matches words. When the question and the passage use different words for the same thing, the search has nothing to hold on to."
            ),
            q(
                "Step 8: we have embeddings. Couldn't our own model's vectors match by meaning?",
                "It's a fair idea. In [02_05](../02_text_to_numbers/02_05_token_embeddings.ipynb) each token got a list of numbers that *can carry meaning*. So give every passage a vector, give the question a vector, and take the passages whose vectors point the same way (the dot product of [A_03](../A_toolbox/A_03_dot_product.ipynb), scaled so that length doesn't count: **cosine similarity**, 1 for the same direction).\n\n"
                "The trouble is what our tokens are. They're single characters. The model has a vector for `π` and one for `ο`, and none for `πολύτροπον`. The simplest way to get one vector for a text is to average the vectors of its characters.\n\n"
                "👉 **Predict first:** `πολύτροπον` (the *man of many turns* of the Odyssey's first line) and the same ten letters scrambled, `νοπορτύλοπ`. How similar will their vectors be?",
            ),
            code(
                r"""import torch

from homer_ai.lessons import load_homer_gpt

tokenizer, model = load_homer_gpt()  # read-only
table = model.token_embedding.weight.detach()  # one row of numbers per character


def text_vector(text):
    # the average of the rows of its characters
    return table[torch.tensor(tokenizer.encode(text))].mean(dim=0)


word, scrambled = "πολύτροπον", "νοπορτύλοπ"
same = torch.cosine_similarity(text_vector(word), text_vector(scrambled), dim=0)
print(f"{word} and {scrambled}: similarity {same:.3f}")"""
            ),
            saw(
                "Similarity 1.000: the two vectors are the same. An average doesn't care about order, so any text with those ten letters gets this vector. For this search, `πολύτροπον` is the same as a spelling mistake.\n\n"
                "A vector per character knows about letters. Meaning lives in words and sentences, and nothing in our model stores a vector for those."
            ),
            q(
                "Step 9: does it at least find the right passages?",
                "Let's search the Greek Odyssey for `Πηνελόπεια`, Penelope's name, both ways: with the letter vectors, and with BM25 (which works on Greek too, since it only needs words).",
            ),
            code(
                r"""from homer_ai.retrieval import load_passages

greek = load_passages("odyssey", "greek")
vectors = torch.stack([text_vector(p.text) for p in greek])


def nearest(query, k=3):
    scores = torch.cosine_similarity(vectors, text_vector(query).unsqueeze(0), dim=1)
    best = torch.topk(scores, k)
    places, values = best.indices.tolist(), best.values.tolist()
    return [(greek[i], value) for i, value in zip(places, values, strict=True)]


query = "Πηνελόπεια"
results = (
    ("letter vectors", nearest(query)),
    ("BM25", search(query, poem="odyssey", language="greek", k=3)),
)
for name, found in results:
    for passage, score in found:
        has_it = "πηνελοπεια" in tokenize(passage.text)
        note = "has the word" if has_it else "no such word"
        print(f"{name:15} {passage.ref:18} {score:6.2f}  {note}")"""
            ),
            saw(
                "Two of the three passages nearest by letter vectors don't contain the name at all. They're passages whose mix of letters happens to resemble the mix in `Πηνελόπεια`. The third has it, which isn't surprising: a passage with the word in it shares its letters. BM25's three all have the word.\n\n"
                "So our own embeddings are out. They were trained to help guess the next *letter*, and that's all they hold. To search by meaning you need a model trained on sentences. "
                "We won't train one, but we can borrow one, the way chapter 8 borrowed GPT-2."
            ),
            q(
                "Step 10 (optional): does a pretrained sentence model beat BM25?",
                "The model is `all-MiniLM-L6-v2`, a small and popular one: 22 million weights, about 90 MB, runs on a CPU. It turns a sentence or paragraph into a **sentence embedding** of 384 numbers, and it was trained on over a billion pairs of texts so that pairs with a similar meaning land close together. "
                "The library that loads it is `sentence-transformers`.\n\n"
                "It's an optional extra, so the course doesn't depend on it. To install it: `uv sync --extra rag` (then restart the notebook's kernel). Without it, the next cell prints one line and stops.\n\n"
                "The cell scores three searches on the same 40 questions: the sentence model alone, BM25, and the two together. *Together* uses **reciprocal rank fusion**: each passage gets `1 / (60 + rank)` from each search, the two are added, and the passages are sorted by the sum. A passage that both searches rank fairly high beats one that only one of them likes.\n\n"
                "👉 **Predict first:** the sentence model matches meaning, not words. By how much will it beat BM25?",
            ),
            code(
                r"""import os

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

if SentenceTransformer is None:
    print("sentence-transformers isn't installed: cell skipped (uv sync --extra rag)")
elif os.environ.get("HOMER_SKIP_DOWNLOADS") == "1":
    print("HOMER_SKIP_DOWNLOADS=1: cell skipped (it downloads about 90 MB)")
else:
    encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    def vectors_of(texts):  # one row of 384 numbers per text, each of length 1
        return encoder.encode(texts, convert_to_tensor=True, normalize_embeddings=True)

    passage_vectors = vectors_of([p.text for p in passages])

    def by_meaning(text, k=50):
        best = torch.topk(passage_vectors @ vectors_of(text), k).indices.tolist()
        return [passages[i] for i in best]

    def together(text, k=10):
        points = {}
        for found in (by_meaning(text), [p for p, _ in search(text, k=50)]):
            for rank, passage in enumerate(found, start=1):
                points[passage] = points.get(passage, 0) + 1 / (60 + rank)
        return sorted(points, key=lambda p: (-points[p], passages.index(p)))[:k]

    print(f"{'':28} {'recall@1':>8} {'recall@5':>8} {'recall@10':>9} {'MRR':>6}")
    show("BM25, plus Butler's names", bm25_ranks)
    show("sentence model", all_ranks(lambda text: by_meaning(text, 10)))
    show("both together", all_ranks(together))"""
            ),
            saw(
                "If the cell was skipped, here's what it printed on my laptop (24 November 2026, `sentence-transformers` 6.3.0, about 12 seconds to turn the 2,140 passages into vectors):\n\n"
                + SCORE_TABLE
                + "| sentence model (`all-MiniLM-L6-v2`) | 0.275 | 0.475 | 0.525 | 0.336 |\n"
                "| both together | 0.275 | 0.575 | 0.650 | 0.415 |\n\n"
                "The sentence model didn't beat BM25. It lost on every column. I didn't expect that.\n\n"
                "Looking at the ranks, the two fail on different questions. The sentence model finds *Why is Achilles angry with Agamemnon?* at rank 1, where BM25 finds nothing. But it loses the Sirens, Scylla and Nausicaa completely, and BM25 has all three in its first two results, on the strength of words like `wax` and `phaeacians`. "
                "And it has a handicap here: it reads at most 256 tokens, so it only sees the first part of Butler's long paragraphs.\n\n"
                "Together they get the best recall@5 and recall@10, 23 and 26 of 40, though fewer first places. That's why real systems often run both, which is called **hybrid search**.\n\n"
                "One caution. With 40 questions, one question is 2.5 points. Differences of one or two questions are noise. *The vectors are not clearly better here* is a safe reading. *BM25 beats embeddings* would be saying too much."
            ),
            *questions(
                [
                    (
                        "Is 40 questions enough?",
                        "Enough to see that the search works about half the time, and to catch a change that breaks it. Not enough to rank two methods that differ by a few points. For that you'd want a few hundred, and more than one person writing them. Mine all come from one head.",
                    ),
                    (
                        "Could the labels be wrong?",
                        "Yes, in one direction: a passage I didn't list can hold the answer too. For question 1, *where is Odysseus when the poem begins*, the file lists `odyssey 1:2`, which names Calypso, and `odyssey 1:5`, which describes her island without naming her. The second one is in the file because a model answered from it, correctly, and I checked. Read the misses before you trust a score.",
                    ),
                    (
                        "Why is recall@10 lower with the names than without?",
                        "Because expansion adds a word to every question that mentions a god or a hero, and that word pulls in other paragraphs about the same person. For one question that pushed the right passage from rank 7 out of the first 10. A fix that helps most questions can still hurt one. The table shows both.",
                    ),
                    (
                        "How would I make the search better?",
                        "The misses of step 7 point the way: the question lacks the passage's words. So people add a step that rewrites the question, or has a model guess an answer first and searches for that. Others fetch 50 passages cheaply and let a stronger model re-order them, which is called reranking. Whatever you try, run the 40 questions before and after.",
                    ),
                ]
            ),
            recap(
                [
                    "Butler's English uses Roman names, so `odysseus` matches nothing. Adding `ulysses` to the question (query expansion) is a small, measurable fix.",
                    "A labelled question stores the places that answer it. recall@k is how often a right passage is in the first k; MRR also rewards being near the top.",
                    "On 40 questions our BM25 puts a right passage in the first 5 for 22 of them. It misses when the question and the passage don't share words.",
                    "Our model's embeddings are per character. Averaged, they can't tell a word from its scrambled letters.",
                    "A small pretrained sentence model did no better than BM25 here; the two together did a little better than either.",
                ],
                "[09_03: The search as an MCP tool, for Claude and other apps](09_03_the_search_tool.ipynb).",
            ),
        ],
    )

    # ---------------------------------------------------------------- 09_03
    save(
        FOLDER + "09_03_the_search_tool.ipynb",
        [
            title("09_03 The search as an MCP tool, for Claude and other apps"),
            where(
                "MCP tool",
                [
                    "src/homer_ai/homer_mcp_server.py",
                    "src/homer_ai/retrieval.py",
                    "tests/test_mcp_server.py",
                ],
                "**after the main course**: a desk where any AI app can ask the librarian",
            ),
            learn(
                [
                    "How the search becomes a second tool, `search_homer`, on the server of 06_03",
                    "What the tool's description has to tell the model, including what the search can't do",
                    "How to talk to the server from Python, the way an AI app does",
                    "How to connect the server to Claude Code, Claude Desktop, Codex CLI, LM Studio and nanobot",
                ]
            ),
            words(
                [
                    (
                        "MCP client",
                        "the program that starts an MCP server and calls its tools on behalf of a model",
                        "Claude Code, Claude Desktop, Codex CLI",
                    ),
                    (
                        "lazy loading",
                        "doing a slow piece of work only when it's first needed, then keeping the result",
                        "the search index is built by the first search",
                    ),
                ]
            ),
            *background(),
            q(
                "The search works in a notebook. How does Claude get to use it?",
                "The same way it got to use our model in [06_03](../06_making_it_talk/06_03_the_mcp_server.ipynb): through the MCP server. A tool there is a Python function with `@mcp.tool()` above it, type hints for its arguments, and a docstring that the model reads as the description.\n\n"
                "So we add one function to `homer_mcp_server.py`. The server then offers two tools:\n\n"
                "| Tool | What it gives | Is the text real? |\n|---|---|---|\n"
                "| `consult_homer_llm` | our model's continuation of a Greek seed | no, it only looks like Greek |\n"
                "| `search_homer` | passages of the Odyssey and the Iliad, with `poem book:line` | yes, copied from `data/homer/` |\n\n"
                "The search tool doesn't answer anything itself. It fetches. The app's model reads the passages and writes the answer, and because every passage starts with its reference, it can cite. "
                "In the picture of [09_00](09_00_chapter_overview.ipynb), this is the desk where any student can ask the librarian.",
            ),
            step_intro(),
            q(
                "Step 1: what does the model see of the new tool?",
                "Its name, its arguments and its description. Let's ask the server, without starting it.\n\n"
                "```\n"
                "search_homer(question, poem='both', language='english', k=5)\n"
                "```",
            ),
            code(
                r"""from homer_ai.lessons import import_mcp_server

server = import_mcp_server()
tools = {tool.name: tool for tool in await server.mcp.list_tools()}
print("tools:", list(tools))
print()
tool = tools["search_homer"]
for name, details in tool.input_schema["properties"].items():
    print(f"{name:9} {details}")
print("must give:", tool.input_schema["required"])"""
            ),
            saw(
                "Two tools now. `search_homer` has four arguments, and only `question` is required.\n\n"
                'Look at `poem` and `language`. In the code their type is `Literal["odyssey", "iliad", "both"]`, and that became an `enum` in what the model sees: a list of the only values allowed. The model can\'t invent a third poem, and if it tries, the call is refused before our function runs.'
            ),
            q(
                "Step 2: what does the description have to say?",
                "Everything the model needs to use the tool well, and that includes what it can't do. A model that thinks the search understands meaning will ask it vague questions and trust weak results.\n\n"
                "👉 **Predict first:** from what you saw in 09_02, what should the description warn about?",
            ),
            code(r"""print(tool.description)"""),
            saw(
                "Three warnings, each from something we measured or saw:\n\n"
                "- **Roman names.** Butler says Ulysses, Jove, Minerva. The model should expect them in the passages.\n"
                "- **Line numbers are ours.** They're lines of this project's files. A scholar's *Odyssey 9.366* is a verse number, which is something else, and the model shouldn't mix them up.\n"
                "- **Words, not meaning.** The search misses a passage that says it in other words, so the model is told to try again with words the text would use, and to say so when the passages don't hold the answer.\n\n"
                "That last line matters most. Without it, a model handed five passages tends to squeeze an answer out of them."
            ),
            q(
                "Step 3: what does a call return?",
                "Plain text: each passage with its reference in square brackets and its score. We can call the function directly.\n\n"
                "👉 **Predict first:** we ask for `k=500` passages. How many come back?",
            ),
            code(
                r"""result = server.search_homer("Who recognises Odysseus by the scar on his leg?", k=2)
print(result[:420] + "…")
print()
many = server.search_homer("Cyclops", k=500)
print("asked for 500, got", many.count("] (score "))"""
            ),
            saw(
                "The first passage is `[odyssey 19:25]`, with a score of 15.7. It's the paragraph where the old nurse is about to wash his feet, and he remembers the scar. (We print only the start. The call returned two passages.)\n\n"
                "Asked for 500, we got 10. `k` is clamped to 1 to 10, like `chars_to_generate` in 06_03, so one call can't flood the model's context with text. A passage longer than 1,200 characters is cut there and ends in `…`.\n\n"
                "The lines that begin `homer: search …` are the server's log. Notebooks show them in red or in a separate box. It goes to stderr, because stdout belongs to the MCP messages."
            ),
            q(
                "Step 4: does the server start more slowly now?",
                "No, because the index is built by the first search and not at startup. Doing slow work only when it's first needed is called **lazy loading**.\n\n"
                "It matters for a server like ours. The client starts it for every session, often to call the other tool or none at all. Building the index for both poems in English takes about a quarter of a second, and it's kept in memory after that.",
            ),
            code(
                r"""import time

from homer_ai import retrieval

retrieval._indexes.clear()  # forget the indexes, as in a server that has just started
for attempt in ("first call", "second call"):
    started = time.perf_counter()
    server.search_homer("Sirens wax", poem="odyssey", k=1)
    print(f"{attempt}: {(time.perf_counter() - started) * 1000:.0f} ms")"""
            ),
            saw(
                "The first call takes around a tenth of a second, because it reads the 24 files of the Odyssey and builds their index. The second takes less than a millisecond."
            ),
            q(
                "Step 5: can I talk to the server the way Claude does?",
                "Yes. The `mcp` library has a client as well. The next cell starts our server as a separate program, connects to it over stdio (its standard input and output, see [06_03](../06_making_it_talk/06_03_the_mcp_server.ipynb)), asks for the tool list and calls the search. "
                "That's what every **MCP client** does, whether it's Claude Code or another app.",
            ),
            code(
                r"""import os
import sys

from mcp import ClientSession, StdioServerParameters, stdio_client

params = StdioServerParameters(
    command=sys.executable,  # the Python this notebook runs on
    args=["-m", "homer_ai.homer_mcp_server"],  # what `uv run homer-serve` runs
    env=dict(os.environ),  # the notebook's settings, e.g. $HOMER_MODEL
)
async with stdio_client(params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        listed = await session.list_tools()
        print("tools:", [tool.name for tool in listed.tools])
        arguments = {"question": "Sirens wax", "poem": "odyssey", "k": 1}
        reply = await session.call_tool("search_homer", arguments)
        print(reply.content[0].text[:150] + "…")"""
            ),
            saw(
                "A real server process answered: two tools, and for the search a passage from Book 12 of the Odyssey, the Sirens. Nothing in this exchange is special to Claude. Any program that speaks MCP can do the same."
            ),
            q(
                "Step 6: how do I connect it to my AI app?",
                "Every client needs the same three facts: a name for the server, the command that starts it, and the command's arguments. For us that's\n\n"
                "```\n"
                "name:      homer\n"
                "command:   uv\n"
                "arguments: --directory /path/to/homer-ai run homer-serve\n"
                "```\n\n"
                "`--directory` lets `uv` find the project from wherever the app happens to start. The clients only differ in where you write this down. The next cell prints it for five of them, with your own project path filled in.",
            ),
            code(
                r"""import json

from homer_ai.retrieval import HOMER

project = str(HOMER.parents[1])  # the folder that holds pyproject.toml
arguments = ["--directory", project, "run", "homer-serve"]
entry = {"command": "uv", "args": arguments}

print("Claude Code (run in a terminal):")
print("  claude mcp add homer -- uv", " ".join(arguments))
print()
print("Claude Desktop (claude_desktop_config.json) and LM Studio (mcp.json):")
print(json.dumps({"mcpServers": {"homer": entry}}, indent=2))
print()
print("Codex CLI (~/.codex/config.toml):")
print(f'[mcp_servers.homer]\ncommand = "uv"\nargs = {json.dumps(arguments)}')
print()
print("nanobot, the HKUDS project (~/.nanobot/config.json):")
print(json.dumps({"tools": {"mcpServers": {"homer": entry}}}, indent=2))"""
            ),
            saw(
                "Four spellings of the same three facts. JSON with `mcpServers` for Claude Desktop and LM Studio, a TOML table named `mcp_servers.homer` for Codex, and the same JSON one level deeper, under `tools`, for nanobot."
            ),
            q(
                "Where does each client keep its settings, and how do I check it worked?",
                "I read each client's own documentation on 24 November 2026. These tools change quickly, so the last column says how sure I am. If something differs on your machine, the linked page wins.\n\n"
                "| Client | Where it goes | How to check | Checked against the docs? |\n|---|---|---|---|\n"
                "| **Claude Code** | the command `claude mcp add homer -- uv --directory … run homer-serve` (add `--scope user` before the name for all your projects; `--scope project` writes a shareable `.mcp.json` with an `mcpServers` entry) | `claude mcp list`, or `/mcp` inside a session | yes: [code.claude.com/docs/en/mcp](https://code.claude.com/docs/en/mcp). This is the one I ran myself, with the `mcpServers` entry. |\n"
                "| **Claude Desktop** | Settings → Developer → Edit Config opens `claude_desktop_config.json` (macOS: `~/Library/Application Support/Claude/`, Windows: `%APPDATA%\\Claude\\`). Quit the app fully and start it again. | the server and its tools appear under the connectors menu of the message box | yes: [modelcontextprotocol.io/docs/develop/connect-local-servers](https://modelcontextprotocol.io/docs/develop/connect-local-servers). Not run by me. |\n"
                "| **Codex CLI** | `~/.codex/config.toml`, or the command `codex mcp add homer -- uv --directory … run homer-serve` | `codex mcp list`, or `/mcp` inside Codex | yes: OpenAI's Codex MCP page (it has moved between sites; search for *Codex MCP*). Not run by me. |\n"
                "| **LM Studio** | in the app: the Program tab in the right sidebar → Install → Edit mcp.json | the tools show up in the chat's tool list | partly: [lmstudio.ai/docs/app/mcp](https://lmstudio.ai/docs/app/mcp) says the file follows Cursor's `mcp.json` notation with `mcpServers`, but its only example is a remote server. The `command` and `args` form above is taken from that notation. **Unverified** for LM Studio itself. |\n"
                "| **nanobot** (HKUDS) | `~/.nanobot/config.json`, under `tools` → `mcpServers` | start `nanobot gateway --verbose` and read the startup log | partly: the project's `docs/configuration.md` shows `command` and `args` there, with `npx` examples. Using `uv` as the command is **unverified**. |\n\n"
                "Two things trip people up with any of them:\n\n"
                "- **The path must be absolute.** `~` and relative paths are often not expanded.\n"
                "- **Desktop apps may not find `uv`.** An app started from the dock doesn't always have your terminal's `PATH`. If the server doesn't start, put the full path of `uv` as the command (`which uv` in a terminal prints it; on Windows, `where uv`).\n\n"
                "There's a second, unrelated project called nanobot (`nanobot-ai/nanobot`, written in Go). Its README only shows servers reached by URL, so I give no snippet for it.",
            ),
            q(
                "And then? What do I type?",
                "A question, in plain words:\n\n"
                "> Using the homer tools: what does the ghost of Achilles say about being dead? Quote the passage and give its reference.\n\n"
                "The model reads the two tool descriptions, picks `search_homer`, gets passages back, and answers from them. Ask the same thing without *using the homer tools* and it may answer from memory. That's its choice, and often a reasonable one. [09_04](09_04_with_and_without.ipynb) is about what the difference is worth.\n\n"
                "You can bring the other tool in too:\n\n"
                "> …and then let the Homer model continue the Greek word μῆνιν for 40 letters.\n\n"
                "That calls `consult_homer_llm`. What comes back looks like Greek and mostly isn't, and its description says so, so a careful model will tell you.\n\n"
                "*A tool description is the cheapest thing you'll ever tune. When a model misuses one of my tools, I rewrite the description before I touch any code.*",
            ),
            *questions(
                [
                    (
                        "Why does the tool return text and not JSON?",
                        "Because its reader is a language model, and the references are what it has to copy. A line like `[odyssey 10:24]` right above the passage is hard to get wrong. JSON would work too, and is the better choice when a program reads the result.",
                    ),
                    (
                        "Does the language have to be English?",
                        'No. With `language="greek"` the tool searches the original, in passages of 5 verses, and accents may be left out: `μηνιν αειδε` finds `iliad 1:1-5`. But the question must then be Greek words. BM25 matches words, so an English question finds nothing in Greek text. A search across languages needs a model that maps both into the same vectors.',
                    ),
                    (
                        "Is it safe to let an AI app run this server?",
                        "It only reads files in `data/homer/` and the model file, writes nothing, and makes no network calls. The inputs are bounded: `poem` and `language` take only the listed values, `k` is clamped, and long passages are cut. `tests/test_mcp_server.py` checks each of those.",
                    ),
                ]
            ),
            recap(
                [
                    "`search_homer` is a second tool on the same MCP server: a typed function with an honest docstring.",
                    "Its description tells the model about Roman names, about our own line numbers, and that the search matches words and can miss.",
                    "`k` is clamped to 1 to 10, the allowed poems and languages are an `enum`, and the index is built by the first search, so the server starts fast.",
                    "Every client needs a name, a command and its arguments. Only the file and the spelling differ.",
                ],
                "[09_04: With and without the passages, and where this goes in real work](09_04_with_and_without.ipynb).",
            ),
        ],
    )

    # ---------------------------------------------------------------- 09_04
    save(
        FOLDER + "09_04_with_and_without.ipynb",
        [
            title(
                "09_04 With and without the passages, and where this goes in real work"
            ),
            where(
                None,
                [
                    "scripts/benchmark_grounding.py",
                    "src/homer_ai/retrieval.py",
                    "data/eval/questions.jsonl",
                ],
                "**after the main course**: the exam, with the book closed and open",
            ),
            learn(
                [
                    "How the passages get into the prompt, and what the model is told to do with them",
                    "How to compare a model with and without them, and what a small run showed",
                    "Why the answer step can't be better than the search step",
                    "What changes when the text is a company's own documents",
                ]
            ),
            words(
                [
                    (
                        "headless mode",
                        "running a chat program from a script, with no window: text in, text out",
                        '`claude -p "…"`',
                    ),
                    (
                        "hybrid search",
                        "running a word search and a vector search and merging their results",
                        "BM25 + a sentence model, as in 09_02",
                    ),
                    (
                        "reranking",
                        "fetching many passages cheaply, then letting a stronger model put them in a better order",
                        "fetch 50, keep the best 5",
                    ),
                ]
            ),
            *background(),
            q(
                "We have a search and a tool. Does any of it help?",
                "That's the question to ask of everything you build, and the only way to answer it is the one from chapter 5: run it both ways and compare.\n\n"
                "So we give the same questions to the same model twice. Once it gets only the question: the closed-book exam. Once it gets the question and the five passages our search found: the open-book exam. Then we score both.\n\n"
                "I'll tell you now that the result isn't flattering for our search, and that's what makes it worth looking at.",
            ),
            step_intro(),
            q(
                "Step 1: how do the passages reach the model?",
                "As text in the prompt. There's no special channel. We paste the passages under the question, each with its reference in front, and add instructions: answer only from these, say so if they don't hold the answer, and end with the reference you used.\n\n"
                "(With the MCP tool of [09_03](09_03_the_search_tool.ipynb) the passages arrive as a tool result. For the model it comes to the same thing: text that sits in front of it while it answers.)\n\n"
                "Here is the prompt for one question, with the passages shortened so it fits on a screen.",
            ),
            code(
                r"""from homer_ai.retrieval import load_questions, search

questions = load_questions()
question = questions[25]  # "Which goddess does Diomedes wound in the hand?"
found = search(question["question"], k=5)
passages = "\n\n".join(f"[{p.ref}] {p.text[:150]}…" for p, _score in found)

prompt = (
    f"Question: {question['question']}\n\n"
    "Here are passages from Samuel Butler's translation. Each starts with its "
    "reference in square brackets: poem book:line.\n\n"
    f"{passages}\n\n"
    "Answer only from these passages. If they do not hold the answer, say so. "
    "Then end with one line in exactly this form, giving the reference of the "
    "passage your answer comes from:\n"
    "SOURCE: <odyssey or iliad> <book>:<line>"
)
print(prompt)"""
            ),
            saw(
                "A question, five passages with their references, and three instructions. The third passage, `[iliad 5:29]`, is the one that holds the answer (Butler's Venus is Aphrodite).\n\n"
                "The last line, `SOURCE: …`, is there for scoring. A fixed form at the end is easy for a program to find, so we don't need a person, or another model, to read every answer."
            ),
            q(
                "Step 2: how do I run the comparison?",
                "With `scripts/benchmark_grounding.py`. For each question it calls Claude twice through the Claude Code command line in **headless mode** (`claude -p`: text in, text out, no window), with all tools switched off and a fixed system prompt, from an empty folder. So Claude sees only what we send.\n\n"
                "```\n"
                "uv run python scripts/benchmark_grounding.py --limit 8 --model sonnet\n"
                "```\n\n"
                "| Run | What Claude gets | What it must end with |\n|---|---|---|\n"
                "| plain | the question | `SOURCE: iliad 5` (poem and book) |\n"
                "| grounded | the question + the 5 passages `search` returns | `SOURCE: iliad 5:29` (poem, book and line) |\n\n"
                "And two scores:\n\n"
                "- **right book**: the poem and book are one of the labelled places.\n"
                "- **right passage** (grounded only): the cited line is inside a labelled place.\n\n"
                "The plain run is only asked for the book, and that's deliberate. Our line numbers are lines of our own files, which no model has seen. Asking for them from memory would be unfair.\n\n"
                "The script isn't part of the course's automatic checks: it needs Claude Code installed and uses your account. `--limit 8` takes 8 questions spread evenly over the 40. You don't have to run it to follow the rest.",
            ),
            q(
                "Step 3: what came out?",
                "This is one recorded run, not a number to quote:\n\n"
                "| | |\n|---|---|\n"
                "| Date | 24 November 2026 |\n"
                "| Model | `sonnet` (the alias, through Claude Code 2.1.316) |\n"
                "| Questions | 8 of the 40 (ids 1, 6, 11, 16, 21, 26, 31, 36) |\n"
                "| Passages per question | 5 |\n"
                "| Command | `uv run python scripts/benchmark_grounding.py --limit 8 --model sonnet` |\n\n"
                "| | plain | grounded |\n|---|---|---|\n"
                "| right book | **8 of 8** | 5 of 8 |\n"
                "| right passage | not asked | 5 of 8 |\n\n"
                "So the closed-book exam went perfectly, and the open-book exam did worse. Here's how each question went:\n\n"
                "| id | Question | plain | grounded |\n|---|---|---|---|\n"
                "| 1 | Where is Odysseus when the poem begins…? | right, Book 1 | right, cites `odyssey 1:5` |\n"
                "| 6 | What did Aeolus give Odysseus…? | right, Book 10 | right, cites `odyssey 10:2`, and notes that the passages stop before the bag is opened |\n"
                "| 11 | How does Odysseus … listen to the Sirens…? | right, Book 12 | right, cites `odyssey 12:5`, and notes it's Circe's advice, not the scene itself |\n"
                "| 16 | Who recognises Odysseus by the scar…? | right, Book 19 | right, cites `odyssey 19:29` |\n"
                "| 21 | Why is Achilles angry with Agamemnon? | right, Book 1 | *The passages don't say why* |\n"
                "| 26 | Which goddess does Diomedes wound…? | right, Book 5 | right, cites `iliad 5:29` |\n"
                "| 31 | How does Hera take Zeus's mind off the battle? | right, Book 14 | *The passages only hint at it* (cites a later mention in Book 15) |\n"
                "| 36 | How does Athena deceive Hector…? | right, Book 22 | *The passages don't say* |\n\n"
                "The plain answers weren't just the right book. I read all eight and they're correct, down to details like Athena taking the form of Hector's brother Deiphobus.",
            ),
            q(
                "Step 4: why did the grounded run lose three?",
                "The search lost them, before Claude read a word. We can check that without calling any model: for the same 8 questions, was a right passage among the 5 that `search` returned?\n\n"
                "👉 **Predict first:** for how many of the 8?",
            ),
            code(
                r"""from homer_ai.retrieval import first_hit_rank

# the 8 questions that --limit 8 picks: every fifth one
sample = [questions[int(i * len(questions) / 8)] for i in range(8)]
on_the_table = 0
for q in sample:
    rank = first_hit_rank([p for p, _ in search(q["question"], k=5)], q)
    on_the_table += rank is not None
    where = f"rank {rank}" if rank else "not in the 5"
    print(f"{q['id']:>2}  {where:14} {q['question']}")
print()
print(f"a right passage was on the table for {on_the_table} of {len(sample)}")"""
            ),
            saw(
                "5 of 8, and they're the same five that the grounded run got right. For questions 21, 31 and 36 the search brought five passages that don't hold the answer, and Claude, told to answer only from them, said so.\n\n"
                "So the answer step was as good as it could be: 5 of 5 when the passage was there. It can't do better than the search that feeds it. That's why 09_02 spent so long on recall@5. If you want better grounded answers, the search is usually the first place to look."
            ),
            q(
                "Step 5: so the open book made things worse. Was this chapter pointless?",
                "On this test, with this text, the closed book wins. I'd rather show you that than a benchmark picked to flatter the method. But look at what each side really gave us.\n\n"
                "**The plain run was right because Claude has read Homer.** The Iliad and the Odyssey are among the most written-about texts there are. A model trained on the internet knows them the way you know your own street. Our test asked it about its home ground.\n\n"
                "**The grounded run gave us things the plain run can't.**\n\n"
                "- A line you can check. `odyssey 19:29` is a paragraph in a file. *Book 19* from memory is a claim.\n"
                "- An honest *I don't know*. Three times the passages didn't hold the answer, and three times Claude said so. It didn't invent one. On a question the model really doesn't know, that's the difference between a gap and a confident mistake.\n"
                "- The exact wording of *our* copy: Butler's *Noman*, not the *Nobody* of other translations.\n\n"
                "**And most text isn't Homer.** Your team's handbook, last week's contract, the ticket history of your product: no model has read those. For them there is no closed-book option. The model either gets the page, or it guesses.\n\n"
                "*This is the question I get asked first on real projects: where did that answer come from? A system that can point at the paragraph gets trusted. One that can't gets switched off after its first wrong answer, however good it was on average.*\n\n"
                "One caution about the numbers. Eight questions is a tiny sample: one question is 12.5 points. This run shows the pattern. It measures nothing precisely.",
            ),
            q(
                "Step 6: what would I change first?",
                "The search, since it's what lost the three. In the order I'd try things:\n\n"
                "1. **Let the model search for itself.** In our benchmark the search ran once, with the question as typed. Through the MCP tool of 09_03, Claude can read the result, see that it's thin, and search again with other words. The tool's description tells it to. "
                "I tried that on question 10, *What does the ghost of Achilles say about being dead?*, which our one-shot search misses (it's in the list of 09_02). With the server connected to Claude Code (24 November 2026, `sonnet`), Claude called `search_homer` twice. "
                "The first call, with its own words for the scene, brought the paragraphs around the answer. The second asked for *rather be alive on earth as a paid servant to a poor man than rule over all the dead*, and got `odyssey 11:42`, which it quoted. "
                "It already knew roughly what the line says, and used that to find where it is. That's one question, so it's an example and not a score.\n"
                "2. **Don't force the book closed.** We told Claude to answer *only* from the passages. In a real assistant you might let it say: the passages don't cover this; from memory, the answer is such-and-such, unverified.\n"
                "3. **Better retrieval.** The hybrid of 09_02, a reranker, or smaller passages. Each one is a change you can score on the 40 questions before you believe in it.",
            ),
            q(
                "Where does this go in a real company?",
                "Swap Homer for the company's own text, and every part of this chapter has a counterpart. The parts are the same. What grows is the care each one needs.\n\n"
                "| In this chapter | In a company | What gets harder |\n|---|---|---|\n"
                "| 48 text files that never change | handbooks, contracts, tickets, wiki pages, PDFs | getting clean text out of them, and noticing when a document changes or is withdrawn |\n"
                "| one paragraph, or 5 verses | passages cut along headings, clauses or table rows | a bad cut hides the answer; tables and lists need their own rules |\n"
                "| `odyssey 9:24` | document, version, page or clause | the citation has to open the right version of the right file |\n"
                "| BM25 over 2,140 passages | **hybrid search** over millions, often with **reranking** | speed, cost, and keeping the index up to date |\n"
                "| everyone may read Homer | each person may read only some documents | the search must apply those permissions before anything reaches the model |\n"
                "| 40 questions I wrote | hundreds of real questions from the people who do the job, with answers they signed off | someone has to own that set and keep it current |\n"
                "| `search_homer` over MCP | the same kind of tool, behind a login | who called it, with what, and what came back: all of it logged |\n"
                "| *the passages don't say* | the same sentence, plus where to ask instead | deciding when the system should refuse to answer |\n\n"
                "If I had to pick the two rows that decide whether such a project works, they'd be the labelled questions and the permissions. The first tells you whether it's any good. The second decides whether you're allowed to switch it on.",
            ),
            *questions(
                [
                    (
                        "Could I run the benchmark with another model?",
                        "Yes: `--model` takes any name your Claude Code accepts, and `--limit 40` runs every question (80 calls). A weaker model will likely do worse on the plain run, and that's where grounding starts to pay. To use a different program altogether, `ask_claude` in the script is the one function to replace.",
                    ),
                    (
                        "Why not let a model judge whether each answer is right?",
                        "You can, and people do; it's called an LLM judge. It reads wording, which our two scores don't. But then the judge needs checking too. A fixed `SOURCE:` line and a labelled range are crude, and you can see exactly what they measure. I start with the crude score and read the answers myself.",
                    ),
                    (
                        "Does grounding stop a model from making things up?",
                        "It makes it less likely and easier to catch. A model can still misread a passage, or blend two. What changes is that you can check: the reference is right there. In the run above it also turned three would-be guesses into three plain *the passages don't say*.",
                    ),
                    (
                        "Is this what people mean by RAG?",
                        "Yes. Retrieve (the search), augment (paste the passages into the prompt), generate (the model writes). The name makes it sound like one technique. As you've seen, it's a pipeline of ordinary parts, and each part can be measured by itself.",
                    ),
                ]
            ),
            md(
                "## 6. Recap\n\n"
                "- The passages reach the model as text in the prompt, each with its reference, plus an instruction to answer only from them and to say when they don't hold the answer.\n"
                "- In one small run (8 questions), Claude named the right book 8 times from memory and 5 times from our passages. Homer is home ground for a big model.\n"
                "- The 3 losses were the search's: no right passage among the 5. When the passage was there, the answer and its citation were right every time.\n"
                "- What grounding adds is a line you can check and an honest *I don't know*. On text no model has read, it's the only way to get an answer at all.\n"
                "- In a company the same parts return, with more care: clean text, good cuts, permissions, and above all a set of labelled questions.\n\n"
                "### ❓ What did this chapter build?\n\n"
                "- Passages of both poems that know their book and line.\n"
                "- BM25 from scratch, with every number checked by hand on three passages.\n"
                "- A set of 40 labelled questions, and two scores to read it with.\n"
                "- A second MCP tool, and the settings to use it from five AI apps.\n"
                "- A with-and-without test that told us something we didn't expect.\n\n"
                "**Next:** [10_00: Your own domain](../10_advanced_your_domain/10_00_chapter_overview.ipynb), where the text is a company's own. Or go [back to the course map](../00_start_here/00_00_course_map.ipynb)."
            ),
        ],
    )
