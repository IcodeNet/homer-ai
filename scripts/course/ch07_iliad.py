# ruff: noqa: E501
"""Chapter 7, Advanced: the Iliad, and combining both."""

from .cells import (
    background,
    code,
    exercise,
    learn,
    md,
    q,
    recap,
    save,
    saw,
    step_intro,
    title,
    where,
    words,
)
from .setup_code import CH7_FIND, CH7_TOOLS

# ---------------------------------------------------------------- 07_02
FULL_ILIAD_CODE = r"""# train_iliad.py: a full Iliad model. Save it in the project folder, then run:
#     uv run python train_iliad.py
from pathlib import Path

import torch

from homer_ai.model import HomerGreekLLM
from homer_ai.tokenizer import CharTokenizer
from homer_ai.train_homer import save_checkpoint, train

ILIAD = Path("data/homer/iliad/greek")
files = sorted(ILIAD.glob("book_*.txt"))
books = [f.read_text(encoding="utf-8") for f in files]
text = "\n".join(books)  # all 24 books: to build the vocabulary
tokenizer = CharTokenizer(text)

ids = []  # the numbers, with <|endoftext|> between books (the loop of load_data)
for book in books:
    if ids:
        ids.append(tokenizer.eot_id)
    ids.extend(tokenizer.encode(book))
data = torch.tensor(ids, dtype=torch.long)

torch.manual_seed(42)
model = HomerGreekLLM(vocab_size=tokenizer.vocab_size, embed_dim=32)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
train(model, optimizer, data, epochs=50)  # prints one line, at epoch 50

# the notes saved with the model: "poem" and "books" tell the MCP server which
# text to search for real lines (without them it would search Odyssey Book 1)
notes = {
    "seed": 42,
    "epochs": 50,
    "lr": 0.01,
    "chunk": 24,
    "train_chars": len(data),
    "val_chars": 0,  # trained on all 24 books: no held-out text
    "books": [f.name for f in files],
    "poem": "iliad",
}
save_checkpoint(model, text, Path("iliad_model.pt"), notes)  # NOT homer_model.pt"""


def build():
    # ---------------------------------------------------------------- 07_00
    save(
        "07_advanced_iliad_and_combining/07_00_chapter_overview.ipynb",
        [
            title("07_00 Advanced: the Iliad, and combining both (overview)"),
            where(
                None,
                ["data/homer/", "src/homer_ai/train_homer.py"],
                "**every stage again**, on a second poem",
            ),
            learn(
                [
                    "Why we repeat every stage on the Iliad, and what that proves",
                    "What it means to compare two models fairly",
                    "Three ways to combine two poems, and which ones this chapter builds",
                    "Which notebook to open for each idea",
                ]
            ),
            words(
                [
                    (
                        "Iliad",
                        "Homer's other poem: 24 books about the anger of Achilles in the last year of the Trojan War",
                        'its first word, `μῆνιν`, means "anger"',
                    ),
                    (
                        "shared vocabulary",
                        "one tokenizer, built from all the texts a comparison uses, so every model uses the same numbers for the same letters",
                        "`α` is number 32 for every model",
                    ),
                    (
                        "fine-tuning",
                        "taking a model already trained on one text and training it a little more on another",
                        "an Odyssey model, then more training on the Iliad",
                    ),
                ]
            ),
            *background(),
            q(
                "We have an Odyssey model. Why build a second one?",
                "Say you've trained a tour guide for one city, and now you need one for a second city. "
                "If the same lessons, unchanged, turn out a second good guide, then the lessons work and the first guide wasn't a lucky accident. "
                "With two guides you can also be fair about who knows what: ask both the same questions about both cities and compare. "
                "And then there's the question this chapter ends on: would one guide who studied both cities do as well as the two specialists, or do you keep both and send each visitor to the right one? "
                'The picture promises too much in one way: a guide knows facts about a city, and our models know only which letter tends to follow which, so "knowing a poem" here means a slightly lower loss on it.\n\n'
                "Now in our own terms, where the lessons are our code and the cities are the two poems. First, as proof. If the same code, unchanged, learns a second poem, then the code is general and not a trick that only works on Book 1. "
                'Second, to compare. With two models we can ask real questions, such as "does a model know its own poem better than the other one?". '
                "And then we can combine them, into one tool that knows both poems.\n\n"
                "```\n"
                "Odyssey Greek ──▶ [ the same steps as chapters 2-5 ] ──▶ Odyssey model\n"
                "Iliad Greek   ──▶ [ the same steps as chapters 2-5 ] ──▶ Iliad model\n"
                "```",
            ),
            q(
                "If the code is the same, what actually changes?",
                "Only the input: **which folder of text we read**. Everything after it (the tokenizer, the 90/10 split, the model, the training loop) is the code you already know. "
                "That's why `load_books` takes a folder as an argument. "
                "Here's that one change on the real texts, the same few lines run on two folders.\n\n"
                "👉 **Predict first:** the two poems have different lengths. Will their vocabularies (the number of different characters, plus our 2 special tokens) have the same size?",
            ),
            code(CH7_FIND),
            code(
                r"""from homer_ai.tokenizer import CharTokenizer

for poem in ("odyssey", "iliad"):
    books = load_book_files(HOMER / poem / "greek")  # the same code for both poems
    text = "\n".join(books)
    tokenizer = CharTokenizer(text)
    print(
        f"{poem:8} {len(books)} books, {sum(len(b) for b in books):,} characters, "
        f"vocabulary {tokenizer.vocab_size}"
    )"""
            ),
            saw(
                "Two poems, one set of lines. The Iliad is longer, and its vocabulary is slightly bigger (171, against 169), because it uses a few characters the Odyssey never does. "
                "That small difference matters more than it looks, and [07_03](07_03_comparing_fairly.ipynb) shows why."
            ),
            q(
                "Earlier our vocabulary was 137. Why are these numbers different?",
                "137 was the tokenizer of chapters 1-6, built from Odyssey **Book 1 alone** (135 characters plus our 2 special tokens). In this chapter we read more text, so we meet more different characters. "
                "That gives 169 for the whole Greek Odyssey, 171 for the Iliad, 175 when one shared vocabulary is built from both, and 66 for Butler's English Book 1 (69 for the whole English poem). "
                "A token's number is its position in the sorted list, so a new character moves every letter after it. `α` is 26 in Book 1's tokenizer ([02_05](../02_text_to_numbers/02_05_token_embeddings.ipynb)), 31 in the Odyssey's and the Iliad's, and 32 in the shared one. "
                "Same letter, different number. That's the trap [07_03](07_03_comparing_fairly.ipynb) deals with.",
            ),
            q(
                'What does "compare two models fairly" mean?',
                "Two models are only comparable if the test is the same for both. We use the habits of [05_04](../05_training_on_the_odyssey/05_04_running_fair_experiments.ipynb) (fix the seed, change one thing) "
                "and the held-out text of [05_05](../05_training_on_the_odyssey/05_05_validation.ipynb) (a part the model never trained on). "
                "One thing is new here: both models must use the same tokenizer, a **shared vocabulary**. Then we test each model on both poems:\n\n"
                "```\n"
                "                       tested on          tested on\n"
                "                       Odyssey (held out) Iliad (held out)\n"
                "trained on Odyssey          ?                  ?\n"
                "trained on Iliad            ?                  ?\n"
                "```\n\n"
                "So the question is, does the model trained on a poem do better on that poem than the other model does? [07_03](07_03_comparing_fairly.ipynb) shows how to read the table.",
            ),
            q(
                "How can I combine two poems?",
                "There are three ways. This chapter builds the first one.\n\n"
                "```\n"
                "1. ONE model, both texts     Odyssey ⏎ <|endoftext|> ⏎ Iliad  ──▶ one model   (07_04)\n"
                "2. TWO models, side by side  Odyssey ──▶ model A   Iliad ──▶ model B\n"
                "                             a program (like the MCP server) asks the one you need\n"
                "3. FINE-TUNING               train on the Odyssey, then train a little more on the Iliad\n"
                "```\n\n"
                "Way 1 uses the `<|endoftext|>` marker of [02_02](../02_text_to_numbers/02_02_special_tokens.ipynb) between the poems. "
                "Way 2 needs no new maths. It's two copies of what you built, and [06_03](../06_making_it_talk/06_03_the_mcp_server.ipynb) shows how a server offers a model. "
                "Way 3, fine-tuning, is how big models are often adapted. This chapter doesn't build it for the two poems; [chapter 8](../08_advanced_fine_tuning/08_00_chapter_overview.ipynb) fine-tunes a much bigger model, GPT-2.",
            ),
            q(
                "Will the Iliad model write good Greek?",
                "No, and I'd rather say so up front. In these notebooks our model is tiny (a few thousand characters of practice, a few seconds of training), "
                "so its writing is word-shaped pieces, like the Odyssey model in [06_02](../06_making_it_talk/06_02_less_repetition.ipynb). "
                "What we measure are **small differences between losses**, and we report them as they come, even when they're mixed.\n\n"
                "Every cell here runs in seconds, so the training uses small slices of text and a few epochs. "
                "Full runs take much longer. [07_02](07_02_an_iliad_model.ipynb) gives the exact code to run them yourself.",
            ),
            q(
                "Which notebook do I open for each idea?",
                "| Notebook | Question it answers |\n|---|---|\n"
                "| [`07_01_the_corpus.ipynb`](07_01_the_corpus.ipynb) | What's in `data/homer/`, in real numbers? |\n"
                "| [`07_02_an_iliad_model.ipynb`](07_02_an_iliad_model.ipynb) | How do I train an Iliad model with the same code? |\n"
                "| [`07_03_comparing_fairly.ipynb`](07_03_comparing_fairly.ipynb) | Why must two models share one vocabulary, and how do they compare? |\n"
                "| [`07_04_combining_both.ipynb`](07_04_combining_both.ipynb) | What do I get, and lose, from one model on both poems? |\n"
                "| [`07_05_english_and_greek.ipynb`](07_05_english_and_greek.ipynb) | Does the same code learn Butler's English? How does it differ from Greek? |",
            ),
            recap(
                [
                    "The same steps on a new poem only change the folder we read. That shows the code is general.",
                    "A fair comparison needs the same tokenizer (a shared vocabulary), the same seed, one change at a time, and held-out text.",
                    "To combine poems you can train one model with `<|endoftext|>` between the works (built here), keep two models side by side, or fine-tune (chapter 8 does that, on GPT-2).",
                ],
                "[07_01: The corpus](07_01_the_corpus.ipynb).",
            ),
        ],
    )

    # ---------------------------------------------------------------- 07_01
    save(
        "07_advanced_iliad_and_combining/07_01_the_corpus.ipynb",
        [
            title("07_01 The corpus: what is in data/homer?"),
            where(
                "Odyssey text",
                ["data/homer/README.md", "data/homer/iliad/", "data/homer/odyssey/"],
            ),
            learn(
                [
                    "How big each poem is, in books, lines and characters",
                    "How many different characters each text uses, and why Greek needs so many more",
                    "Which characters only one Greek poem has",
                    "How the two poems begin, word by word",
                ]
            ),
            words(
                [
                    (
                        "corpus",
                        "a collection of texts that we study or train on",
                        "the 4 texts in `data/homer/`",
                    ),
                    (
                        "edition",
                        "one scholar's prepared version of an ancient text: the old manuscripts disagree in places, so editors choose",
                        "Murray's Odyssey, Monro and Allen's Iliad",
                    ),
                    (
                        "unicodedata.name",
                        "a built-in lookup that gives a character its official name",
                        '`unicodedata.name("ἄ")` is `GREEK SMALL LETTER ALPHA WITH PSILI AND OXIA`',
                    ),
                ]
            ),
            *background(),
            q(
                "What's in the `data/homer/` folder?",
                "Four texts: two poems, each in Ancient Greek and in English, 24 books each, one file per book ([data/homer/README.md](../../data/homer/README.md) has the sources and licences).\n\n"
                "```\n"
                "data/homer/\n"
                "  odyssey/greek/    book_01.txt … book_24.txt     one verse per line\n"
                "  odyssey/english/  book_01.txt … book_24.txt     one paragraph per line\n"
                "  iliad/greek/      book_01.txt … book_24.txt     one verse per line\n"
                "  iliad/english/    book_01.txt … book_24.txt     one paragraph per line\n"
                "```\n\n"
                'The Greek is the original verse. The English is Samuel Butler\'s prose translation, so its "lines" are whole paragraphs. '
                "Training reads this folder too, but by default only `odyssey/greek/book_01.txt`. "
                "This chapter is the first to use the other books and texts.\n\n"
                "👉 **Predict first:** which text is the biggest, in characters? Does the English or the Greek have more?",
            ),
            step_intro(),
            q(
                "How big is each text?",
                "We read all 24 files of each folder and count the books, the lines (a line ends with `\\n`) and the characters (what `len()` counts, including the `\\n`).",
            ),
            code(CH7_FIND),
            code(
                r"""corpus = {}  # (poem, language) -> the list of 24 book texts
for poem in ("odyssey", "iliad"):
    for language in ("greek", "english"):
        corpus[(poem, language)] = load_book_files(HOMER / poem / language)

print(f"{'text':16} {'books':>5} {'lines':>7} {'characters':>11} {'per line':>9}")
for (poem, language), books in corpus.items():
    lines = sum(book.count("\n") for book in books)
    characters = sum(len(book) for book in books)
    print(
        f"{poem + ' ' + language:16} {len(books):>5} {lines:>7,} "
        f"{characters:>11,} {characters / lines:>9.1f}"
    )"""
            ),
            code(
                r"""import matplotlib.pyplot as plt

from homer_ai.studio_style import ACCENT, GOOD, INK, studio_axes, studio_figure

title = "How much text is there? (thousands of characters)"
fig = studio_figure("the corpus", title, (10, 5.5))
ax = studio_axes(fig, (0.1, 0.12, 0.8, 0.64))
width = 0.35
for n, (poem, colour) in enumerate([("odyssey", ACCENT), ("iliad", GOOD)]):
    sizes = [
        sum(len(book) for book in corpus[(poem, language)]) / 1000
        for language in ("greek", "english")
    ]
    positions = [i + (n - 0.5) * width for i in range(2)]
    ax.bar(positions, sizes, width, color=colour, label=poem.capitalize())
    for x, size in zip(positions, sizes, strict=True):
        ax.text(x, size + 10, f"{size:,.0f}", ha="center", color=INK,
                family="monospace", fontsize=11, fontweight="bold")
ax.set_xticks([0, 1], ["Greek", "English (Butler)"])
ax.tick_params(axis="x", labelsize=13, labelcolor=INK)
ax.set_ylabel("thousands of characters")
ax.set_ylim(0, 950)
ax.grid(axis="x", visible=False)
ax.set_axisbelow(True)
ax.legend(frameon=False, labelcolor=INK)
plt.show()"""
            ),
            saw(
                "In the table and the picture, the Iliad is bigger than the Odyssey (15,687 Greek verses against 12,107). "
                "The English has **more** characters than the Greek (606,758 against 538,303 for the Odyssey) but far fewer lines. "
                "That's about 590 characters per line, against about 44 for Greek verse. A Greek line is one verse, and an English line is a whole paragraph. "
                "[07_05](07_05_english_and_greek.ipynb) comes back to this."
            ),
            q(
                "How many different characters does each text use?",
                "This is the size of the character vocabulary ([02_01](../02_text_to_numbers/02_01_tokens_and_vocabulary.ipynb)), before the two special tokens. "
                "We keep each character once, as in 02_01. A line break counts as a character too.\n\n"
                "👉 **Predict first:** English has 26 letters, capitals, digits and punctuation. Greek has 24 letters. Which will have more different characters, and by how many times?",
            ),
            code(
                r"""distinct = {key: set("".join(books)) for key, books in corpus.items()}
for (poem, language), characters in distinct.items():
    print(f"{poem + ' ' + language:16} {len(characters):>4} different characters")"""
            ),
            saw(
                "167 (Odyssey) and 169 (Iliad) for Greek, against 67 and 68 for English. Greek needs more than **twice** as many. "
                "The reason is the accents. A Greek letter can carry a breathing mark, an accent, or both, and each combination is its own character. Here's `α`:"
            ),
            code(
                r"""import unicodedata

odyssey_greek = distinct[("odyssey", "greek")]
alphas = sorted(c for c in odyssey_greek if "ALPHA" in unicodedata.name(c, ""))
print(len(alphas), "different alpha characters in the Odyssey:", " ".join(alphas))
print()
print("ἄ is called:", unicodedata.name("ἄ"))"""
            ),
            saw(
                "That's 18 different alphas in the Odyssey alone, each its own character. The name spells out the parts (`PSILI` is the gentle breathing mark, `OXIA` the sharp accent). "
                "English has one `a` and one `A`. That's why a Greek model needs a bigger vocabulary, and why the numbers of tokens differ later in [07_05](07_05_english_and_greek.ipynb)."
            ),
            q(
                "Which characters are in one Greek poem but not the other?",
                "We compare the two character sets and look for what's in one poem and not in the other. We also count how often each one occurs, and print its name.\n\n"
                "👉 **Predict first:** the two poems are in the same language, so the sets should be nearly equal. Will `Iliad − Odyssey` be empty?",
            ),
            code(
                r"""odyssey_chars = distinct[("odyssey", "greek")]
iliad_chars = distinct[("iliad", "greek")]
iliad_text = "".join(corpus[("iliad", "greek")])
odyssey_text = "".join(corpus[("odyssey", "greek")])

print("only in the Iliad:")
for c in sorted(iliad_chars - odyssey_chars):
    print(f"  {c}  {iliad_text.count(c):>3} times  {unicodedata.name(c)}")
print("only in the Odyssey:")
for c in sorted(odyssey_chars - iliad_chars):
    print(f"  {c}  {odyssey_text.count(c):>3} times  {unicodedata.name(c)}")"""
            ),
            saw(
                "Six characters appear only in the Iliad, and four only in the Odyssey. They're all rare (some occur once or twice in a whole poem), "
                "but a tokenizer built from one poem has **no number** for them. That's the trap of [07_03](07_03_comparing_fairly.ipynb). "
                "And one of the four Odyssey-only characters isn't Greek at all. It's the long dash `—`."
            ),
            q(
                "How do the two poems begin?",
                "Each poem starts with a request to the Muse. Here's the first line of each, word by word, from the real files. "
                "The English is the start of Butler's paragraph, and the glosses are our own.\n\n"
                "```\n"
                "μῆνιν    ἄειδε   θεὰ      Πηληϊάδεω     Ἀχιλῆος\n"
                "anger    sing    goddess  of-Peleus'-son  of-Achilles\n"
                "\n"
                "ἄνδρα    μοι     ἔννεπε,  μοῦσα,   πολύτροπον,\n"
                "man      me      tell     Muse     of-many-turns\n"
                "```\n\n"
                "👉 **Predict first:** the code prints the same two rows. Which poem starts with a man, and which with an anger?",
            ),
            code(
                r"""GLOSSES = {
    "iliad": ["anger", "sing", "goddess", "of-Peleus'-son", "of-Achilles"],
    "odyssey": ["man", "me", "tell", "Muse", "of-many-turns"],
}
for poem in ("iliad", "odyssey"):
    first_line = corpus[(poem, "greek")][0].splitlines()[0]
    greek_words = first_line.split()[:5]
    print(f"{poem.upper()}  (verse 1 of book 1)")
    print("   ".join(f"{w:<14}" for w in greek_words))
    print("   ".join(f"{g:<14}" for g in GLOSSES[poem]))
    english = corpus[(poem, "english")][0].splitlines()[0]
    print("Butler:", english[:80], "…\n")"""
            ),
            saw(
                'The Iliad starts with `μῆνιν` ("anger"), the Odyssey with `ἄνδρα` ("man"). So each poem names its subject in its first word. '
                "We use these two words as prompts in [07_04](07_04_combining_both.ipynb), to see whether a model that knows both poems answers differently."
            ),
            q(
                "Why do the extra characters matter for a model?",
                "A model has one row of numbers for every token in its vocabulary (the embedding table of [02_05](../02_text_to_numbers/02_05_token_embeddings.ipynb), and the output head). "
                "A different vocabulary means a different-shaped model, and the same letter can get a different number. Two models built that way can't be compared directly. "
                "[07_03](07_03_comparing_fairly.ipynb) shows it with real numbers.",
            ),
            recap(
                [
                    "`data/homer/` has four texts: 24 books each of Greek verse and Butler's English prose, for the Odyssey and the Iliad.",
                    "Greek uses about 167 different characters against English's 67, because every accent combination is its own character.",
                    "Six characters appear only in the Iliad and four only in the Odyssey. It's a small thing, but it matters for fair comparisons.",
                ],
                "[07_02: An Iliad model](07_02_an_iliad_model.ipynb).",
            ),
        ],
    )

    save(
        "07_advanced_iliad_and_combining/07_02_an_iliad_model.ipynb",
        [
            title("07_02 An Iliad model: the same code, a new poem"),
            where(
                "model learns",
                [
                    "data/homer/iliad/greek/",
                    "src/homer_ai/train_homer.py",
                    "src/homer_ai/model.py",
                ],
            ),
            learn(
                [
                    "How to point the same code at a new poem: two settings, `BOOKS_DIR` and `BOOKS`",
                    "How to build the Iliad's tokenizer, split 90/10 and train briefly",
                    "How to read the train and validation losses, and generate lines",
                    "The exact code to train a full Iliad model yourself, and how long it takes",
                ]
            ),
            words(
                [
                    (
                        "glob pattern",
                        "a file-name template where `*` stands for anything: `book_*.txt` matches `book_01.txt`, `book_02.txt`, …",
                        '`folder.glob("book_*.txt")`',
                    ),
                    (
                        "slice (of text)",
                        "only the first part of a text, so a training run takes seconds instead of hours",
                        "the first 10,000 characters of Book 1",
                    ),
                ]
            ),
            *background(),
            q(
                "Do I need new code for the Iliad?",
                "Almost none. The steps are the ones you know, and only the folder changes:\n\n"
                "```\n"
                "folder ─▶ books ─▶ tokenizer ─▶ numbers ─▶ split 90/10 ─▶ train ─▶ check ─▶ write\n"
                "(new)     02_02     02_01       02_04      05_05          05_01    05_05    06_02\n"
                "```\n\n"
                "Two settings in `train_homer.py` pick the text. `BOOKS_DIR` says which poem (a folder in `data/homer/`), and `BOOKS` says which books (file names). "
                "`load_books(folder)` takes the folder as an argument, which is why a new folder works at once ([A_07](../A_toolbox/A_07_organising_code.ipynb)). "
                "Let's point it at the Iliad's Greek folder.\n\n"
                "👉 **Predict first:** `BOOKS` names only `book_01.txt`. How many books will `load_books` return for the Iliad folder? And for the Odyssey folder?",
            ),
            code(
                r"""from homer_ai.lessons import HOMER  # the project's data/homer folder
from homer_ai.train_homer import BOOKS, BOOKS_DIR, load_books

iliad_folder = HOMER / "iliad" / "greek"

print("BOOKS setting: ", BOOKS)
print("Odyssey folder:", len(load_books(BOOKS_DIR)), "book loaded")
print("Iliad folder:  ", len(load_books(iliad_folder)), "book loaded")
iliad_files = sorted(iliad_folder.glob("book_*.txt"))
print("Iliad book files:", len(iliad_files), [f.name for f in iliad_files[:3]])"""
            ),
            saw(
                'You get 1 book from each folder, because `BOOKS = ["book_01.txt"]` asks for Book 1 only. '
                "The Iliad folder has 24 book files, found with a **glob pattern**, a file-name template where `*` stands for anything. "
                "It's the same code and the same file names (`book_01.txt` …), so changing the poem really is just a change of folder."
            ),
            q(
                "How do I load all 24 books?",
                "Pass the file names yourself: `load_books(folder, books=[...])`. In `train_homer.py` you'd widen the `BOOKS` setting the same way.\n\n"
                "👉 **Predict first:** how many books, and how long is Book 1?",
            ),
            code(
                r"""all_names = [f.name for f in iliad_files]  # book_01.txt ... book_24.txt
iliad = load_books(iliad_folder, books=all_names)
print(len(iliad), "books; book 1 has", len(iliad[0]), "characters")
print("first verse:", iliad[0].splitlines()[0])"""
            ),
            saw(
                "That's 24 books, and the first verse is the opening of the Iliad that we met in [07_01](07_01_the_corpus.ipynb). "
                "From here on, the steps are exactly the ones from the earlier chapters."
            ),
            step_intro(),
            q(
                "Step 1: how do I build the Iliad's tokenizer?",
                "Glue the 24 books together and give the text to `CharTokenizer` ([02_01](../02_text_to_numbers/02_01_tokens_and_vocabulary.ipynb), [02_02](../02_text_to_numbers/02_02_special_tokens.ipynb)). "
                "Then we do the usual check: encode a verse, decode it again, and see if we get the same text back.\n\n"
                "👉 **Predict first:** the Iliad's vocabulary was 171 in [07_00](07_00_chapter_overview.ipynb). What number will `μῆνιν` (5 characters) become: 5 numbers, or fewer?",
            ),
            code(
                r"""import torch

from homer_ai.tokenizer import CharTokenizer

text = "\n".join(iliad)
tokenizer = CharTokenizer(text)
print("vocabulary size:", tokenizer.vocab_size)

first_verse = iliad[0].splitlines()[0]
ids = tokenizer.encode(first_verse)
print("μῆνιν →", tokenizer.encode("μῆνιν"))
print("round trip works:", tokenizer.decode(ids) == first_verse)
print("<|unk|> =", tokenizer.unk_id, "  <|endoftext|> =", tokenizer.eot_id)"""
            ),
            saw(
                "171 tokens (169 characters and the two special ones), `μῆνιν` becomes 5 numbers, and the round trip is exact. "
                "The special tokens sit at the end, 169 and 170, as always."
            ),
            *exercise(
                "how many characters are new in the Iliad?",
                "The Iliad's vocabulary is bigger than the 137 of Odyssey Book 1. Count the different characters the Iliad uses that Book 1 doesn't have. (`text` is the whole Iliad.)\n\n"
                "👉 **Predict first:** what would Book 1's tokenizer make of such a character?",
                starter="""
            odyssey_book1 = load_books(BOOKS_DIR)[0]

            # 👉 do: the number of different characters that are in `text` but not in odyssey_book1
            # 💡 hint: set(a) - set(b) keeps what is in a and not in b
            new_characters = None
            """,
                solution="""
            odyssey_book1 = load_books(BOOKS_DIR)[0]

            new_characters = len(set(text) - set(odyssey_book1))
            """,
                check="""
            check(
                "characters only the Iliad has",
                new_characters,
                len({ch for ch in text if ch not in odyssey_book1}),
                coach="Got 1? That's the other way round: what Book 1 has and the Iliad "
                "doesn't. Got 34? Subtracting the two sizes isn't enough, because Book 1 has "
                "a character of its own. A set and not a number? Wrap it in len().",
            )
            """,
                hint="Make a set of each text's characters, take the difference, and count it with `len`.",
                answer="35. Book 1's tokenizer would turn every one of them into `<|unk|>`, so a model trained with that vocabulary couldn't even read the Iliad properly. "
                "That's why this notebook builds a new tokenizer, and why [07_03](07_03_comparing_fairly.ipynb) gives both poems one shared vocabulary before it compares anything.",
            ),
            q(
                "Step 2: which text do I train on, and how do I split it?",
                "The whole Iliad is 690,573 characters, which is too long for a notebook. So we use a **slice**, the first 10,000 characters of Book 1 (Book 1 alone is 27,201). "
                "Then comes the split of [05_05](../05_training_on_the_odyssey/05_05_validation.ipynb): the first 90% to train on, the last 10% hidden for the check.\n\n"
                "👉 **Predict first:** how many numbers go to training and how many to validation?",
            ),
            code(
                r"""SLICE = 10000
data = torch.tensor(tokenizer.encode(iliad[0][:SLICE]))
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
print("training:  ", len(train_data))
print("validation:", len(val_data))
print("validation text starts:", repr(tokenizer.decode(val_data[:30].tolist())))"""
            ),
            saw(
                "You get 9,000 and 1,000, and the held-out text begins in the middle of a verse. "
                "The model never trains on those 1,000 characters."
            ),
            q(
                "Step 3: how good is the untrained model, and how do I train it?",
                "The measure is `evaluate` from [05_05](../05_training_on_the_odyssey/05_05_validation.ipynb) (the loss without learning). "
                "An untrained model guesses among 171 tokens, so its loss should be near `ln(171)`.\n\n"
                "Then we run the training loop of [05_01](../05_training_on_the_odyssey/05_01_loss_and_epochs.ipynb) on `train_data` only, with a fixed seed ([05_04](../05_training_on_the_odyssey/05_04_running_fair_experiments.ipynb)), "
                "for 6 epochs. It prints the train and validation loss after each one. We also time it.\n\n"
                "👉 **Predict first:** will the validation loss end near the training loss, or far above?",
            ),
            code(
                r"""import math
import time

import torch.nn.functional as F

from homer_ai.lessons import CHUNK, evaluate  # evaluate: the function of 05_05
from homer_ai.model import HomerGreekLLM

torch.manual_seed(42)
model = HomerGreekLLM(vocab_size=tokenizer.vocab_size, embed_dim=32)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
untrained = evaluate(model, val_data)
print(f"untrained validation loss: {untrained:.3f}   (ln(171) = {math.log(171):.3f})")

train_curve, val_curve, steps = [], [], 0
started = time.perf_counter()
for epoch in range(6):
    losses = []
    for i in range(0, len(train_data) - CHUNK, CHUNK):
        inputs = train_data[i : i + CHUNK]
        targets = train_data[i + 1 : i + CHUNK + 1]
        loss = F.cross_entropy(model(inputs), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        steps += 1
    train_curve.append(sum(losses) / len(losses))
    val_curve.append(evaluate(model, val_data))
    print(
        f"epoch {epoch + 1} | train {train_curve[-1]:.3f} "
        f"| validation {val_curve[-1]:.3f}"
    )
seconds = time.perf_counter() - started
print(f"{steps} steps in {seconds:.1f} seconds")"""
            ),
            saw(
                "The untrained loss is 5.165, right at `ln(171) = 5.142` (pure guessing). After six epochs the training loss is 2.805 and the validation loss 2.861. They're close together, so the model learned something real about Greek letters and didn't just memorise. "
                "The validation loss stopped improving at epoch 5 (2.855, then 2.861), a small sign that this slice has little left to teach. The graph shows it."
            ),
            code(
                r"""import matplotlib.pyplot as plt

from homer_ai.studio_style import ACCENT, GOOD, studio_axes, studio_figure

epochs = list(range(1, 7))
title = "Training and validation loss, 6 epochs"
fig = studio_figure("the Iliad model", title, (9, 5.5))
ax = studio_axes(fig, (0.1, 0.14, 0.8, 0.64))
style = {"marker": "o", "linewidth": 2.5}
ax.plot(epochs, train_curve, color=GOOD, label="training loss", **style)
ax.plot(epochs, val_curve, color=ACCENT, label="validation loss", **style)
for curve, colour, shift in ((val_curve, ACCENT, 6), (train_curve, GOOD, -16)):
    ax.annotate(
        f"{curve[-1]:.2f}", (6, curve[-1]), textcoords="offset points",
        xytext=(8, shift), color=colour, fontsize=11,
        fontfamily="monospace", fontweight="bold",
    )
ax.set_xlim(0.7, 6.9)
ax.set_xlabel("epoch")
ax.set_ylabel("average loss")
ax.legend(frameon=False, labelcolor="white")
plt.show()"""
            ),
            saw(
                "Both lines fall together and stay close. That's learning, not memorising ([05_05](../05_training_on_the_odyssey/05_05_validation.ipynb)). And this was only six epochs on a small slice, so there's more to gain with more text and more epochs."
            ),
            q(
                "Step 4: what does the Iliad model write?",
                "We try the same two ways as [06_02](../06_making_it_talk/06_02_less_repetition.ipynb): greedy (always the best letter, `top_k=1`) and top-k with a little temperature. "
                "The prompt is the Iliad's first word, `μῆνιν`. A seed makes it repeatable, and `⏎` is a line break.\n\n"
                "👉 **Predict first:** which will loop, and will either write real Greek sentences?",
            ),
            code(
                r"""from homer_ai.lessons import write  # the function of 06_02; top_k=1 is greedy

print("greedy:", write(model, tokenizer, "μῆνιν", 70, top_k=1))
print("top-5: ", write(model, tokenizer, "μῆνιν", 70, temperature=0.8, top_k=5))"""
            ),
            saw(
                'Greedy loops at once, on `δʼ` (that\'s `δέ`, "but", with its last vowel dropped, and the mark `ʼ` shows the drop). '
                'Top-5 writes short Greek-looking pieces and a few real small words (`μέν`, `ἐγὼ` "I", `δὲ`), in no sensible order. '
                "That's what a few seconds of training on 9,000 characters buys you: the shape of Greek words, but no Greek sentences."
            ),
            q(
                "How do I train a FULL Iliad model myself?",
                "With the same loop, on all 24 books of the Iliad folder. It saves to a **new** file, so the Odyssey's `homer_model.pt` is never overwritten. "
                "Save this as `train_iliad.py` in the project folder. It needs only functions from `train_homer.py`. `train` prints a line at every 50th epoch, so this short run prints just one, at the end:\n\n"
                "```python\n" + FULL_ILIAD_CODE + "\n```\n\n"
                "How long will it take? The code below estimates it from the speed we just measured in this notebook.\n\n"
                "👉 **Predict first:** one epoch of the full Iliad is about 28,800 chunks of 24 characters, and we just did a few thousand steps in a few seconds. About how long is one epoch?",
            ),
            code(
                r"""seconds_per_step = seconds / steps
steps_per_epoch = len(text) // CHUNK  # about: one step per 24-character chunk
one_epoch = seconds_per_step * steps_per_epoch
print(f"one step:   {seconds_per_step * 1000:.1f} milliseconds")
print(f"one epoch:  about {one_epoch:.0f} seconds ({steps_per_epoch:,} steps)")
print(f"50 epochs:  about {50 * one_epoch / 60:.0f} minutes")
print(f"800 epochs: about {800 * one_epoch / 3600:.0f} hours (the Odyssey's setting)")"""
            ),
            saw(
                "About a millisecond per step, so one epoch of the full Iliad takes about half a minute on this machine (your number will differ), 50 epochs about 25 minutes, and the 800 epochs of the Odyssey model about 6 hours. "
                "Start with `epochs=50`, and let it run while you do something else. The file `iliad_model.pt` can be loaded just like in [05_06](../05_training_on_the_odyssey/05_06_saving_and_loading.ipynb). "
                "The MCP server doesn't look for `iliad_model.pt` by itself, so it keeps offering the Odyssey model unless you point `HOMER_MODEL` at the new file ([06_03](../06_making_it_talk/06_03_the_mcp_server.ipynb)).\n\n"
                "And the `notes` at the end of the script matter there. The server shows real lines that contain the seed, and it finds them by reading the books named in the checkpoint: `poem` picks the folder in `data/homer/`, `books` the files (`training_text` in `checkpoint.py`). "
                "Leave the notes out and it falls back to Odyssey Book 1, so an Iliad model would answer with Odyssey lines. "
                "One thing the notes can't fix: the tool's description still says \"Homer's Odyssey, Book 1\". If you serve an Iliad model for real, change that docstring too, or it won't be true."
            ),
            q(
                "Why didn't this notebook save a model?",
                "Because training here is a demo. Saving would make a file nobody asked for, and we never write `homer_model.pt` from a notebook. "
                "The script above is the real thing, and you run it on purpose.",
            ),
            recap(
                [
                    "The Iliad needs the same code on a different folder. `BOOKS_DIR` picks the poem and `BOOKS` picks the books (`load_books(folder, books=[...])`).",
                    "A short run on a slice gives a loss near 3 and no overfitting. The writing is word-shaped pieces, not Greek sentences.",
                    "A full run is `train_iliad.py` above. It takes about half a minute per epoch on this machine and saves to `iliad_model.pt`.",
                ],
                "[07_03: Comparing fairly](07_03_comparing_fairly.ipynb).",
                can=[
                    "load any poem's books with `load_books(folder, books=[...])`",
                    "find the characters one text has and another lacks",
                    "train and check a small model on a new text with the code you already have",
                ],
            ),
        ],
    )

    # ---------------------------------------------------------------- 07_03
    save(
        "07_advanced_iliad_and_combining/07_03_comparing_fairly.ipynb",
        [
            title("07_03 Comparing fairly: one vocabulary for both poems"),
            where(
                "model learns",
                ["src/homer_ai/tokenizer.py", "src/homer_ai/train_homer.py"],
            ),
            learn(
                [
                    "Why a tokenizer built from one poem can't read the other",
                    "How a shared vocabulary fixes it",
                    "How to run a fair 2×2 experiment: two models, each tested on both poems",
                    "How to read the table without claiming too much",
                ]
            ),
            words(
                [
                    (
                        "held-out text",
                        "text a model never trained on, used only to test it (the validation set of 05_05)",
                        "the last 10% of each slice",
                    ),
                    (
                        "heatmap",
                        "a grid of numbers drawn as colours, so you see the pattern at once",
                        "green = low loss, pink-red = high loss",
                    ),
                ]
            ),
            *background(),
            q(
                "Why can't I just train an Odyssey model and an Iliad model and compare their losses?",
                "Because the text, the tokenizer and the test would all differ at once. Then a difference in the loss could come from any of them (remember the rule of [05_04](../05_training_on_the_odyssey/05_04_running_fair_experiments.ipynb): change **one** thing). "
                "We want the text to be the only difference. So the tokenizer must be the same, and the test text must be the same for both models.\n\n"
                "Let's see what goes wrong when each poem has its own tokenizer.",
            ),
            code(CH7_TOOLS),
            step_intro(),
            q(
                "Step 1: what happens if I read the Iliad with the Odyssey's tokenizer?",
                "We build a tokenizer from the Odyssey alone and use it on an Iliad verse. We saw in [07_01](07_01_the_corpus.ipynb) that the Iliad has six characters the Odyssey lacks. "
                "Where one appears, `<|unk|>` ([02_02](../02_text_to_numbers/02_02_special_tokens.ipynb)) takes its place.\n\n"
                "👉 **Predict first:** the real verse below has `Ὕ` (capital upsilon with a mark). What will the Odyssey's tokenizer give back?",
            ),
            code(
                r"""odyssey_tok = CharTokenizer("\n".join(odyssey))
iliad_tok = CharTokenizer("\n".join(iliad))

verse = "οἵ τʼ Ἐλεῶνʼ εἶχον ἠδʼ Ὕλην καὶ Πετεῶνα,"  # Iliad, book 2
print("original:        ", verse)
print("Odyssey's tokens:", odyssey_tok.decode(odyssey_tok.encode(verse)))
print("Iliad's tokens:  ", iliad_tok.decode(iliad_tok.encode(verse)))"""
            ),
            saw(
                "The Odyssey's tokenizer turns `Ὕ` into `<|unk|>`, while the Iliad's tokenizer reads the verse perfectly. "
                "One unknown letter is like a hole in a word. So how often does it happen over the whole Iliad?"
            ),
            code(
                r"""iliad_text = "".join(iliad)
unknown = [c for c in iliad_text if c not in odyssey_tok.char_to_id]
lines_hit = sum(
    any(c not in odyssey_tok.char_to_id for c in line)
    for line in iliad_text.splitlines()
)
print(f"unknown characters in the whole Iliad: {len(unknown)} of {len(iliad_text):,}")
all_verses = iliad_text.count("\n")
print(f"verses with at least one:             {lines_hit} of {all_verses:,}")
print("which ones:", " ".join(sorted(set(unknown))))"""
            ),
            saw(
                "That's 42 unknown characters out of 690,573, in 42 verses. Very rare, about 6 in 100,000. "
                "Small, but it isn't zero. The Odyssey's model can never read, or write, `Ξ` or `Ὕ`. "
                "The next problem is worse, because it's silent."
            ),
            q(
                "Step 2: does the same letter get the same number in both tokenizers?",
                "A token's number is its position in the sorted list of characters. If one tokenizer has an extra character in the list, every letter after it moves one place. "
                "A model trained with one tokenizer would then read the other's numbers as different letters.\n\n"
                "👉 **Predict first:** the vocabularies are 169 and 171. Will most letters keep their number?",
            ),
            code(
                r"""moved = [
    c
    for c in odyssey_tok.chars
    if c in iliad_tok.char_to_id
    and odyssey_tok.char_to_id[c] != iliad_tok.char_to_id[c]
]
sizes = (odyssey_tok.vocab_size, iliad_tok.vocab_size)
print("vocabulary sizes:", sizes[0], "(Odyssey),", sizes[1], "(Iliad)")
print("characters in both, with a different number:", len(moved))
for c in moved[:3]:
    ids = (odyssey_tok.char_to_id[c], iliad_tok.char_to_id[c])
    print(f"  {c}  Odyssey {ids[0]}  Iliad {ids[1]}")
print("<|unk|>:       ", odyssey_tok.unk_id, "(Odyssey)", iliad_tok.unk_id, "(Iliad)")
print("<|endoftext|>: ", odyssey_tok.eot_id, "(Odyssey)", iliad_tok.eot_id, "(Iliad)")"""
            ),
            saw(
                "54 characters that both poems use have a **different number** in the two tokenizers. `Ο` is 19 in the Odyssey's and 20 in the Iliad's, because the Iliad has an extra character (`Ξ`) earlier in the sorted list, and every letter after it moves one place. The special tokens move too (`<|unk|>` is 167 against 169). So a model trained with one tokenizer would read the other's numbers as the wrong letters. And the two vocabularies have different sizes (169 and 171), which means the two models wouldn't even have the same shape. A comparison built on them would be unfair twice over."
            ),
            q(
                "What's the fix?",
                "One **shared vocabulary**. We build a single tokenizer from both poems, and give it to both models. Every letter gets one number, for every model. "
                "The special tokens still sit at the end (`<|unk|>`, `<|endoftext|>`), and this time in the same place for everyone.\n\n"
                "```\n"
                "Odyssey text ┐\n"
                "              ├──▶ ONE tokenizer ──▶ Odyssey model\n"
                "Iliad text   ┘                  └──▶ Iliad model\n"
                "```\n\n"
                "👉 **Predict first:** how many unknown characters will the shared tokenizer find in each poem?",
            ),
            code(
                r"""import math

shared = CharTokenizer("\n".join(odyssey + iliad))
print("shared vocabulary size:", shared.vocab_size)

for name, books in (("Odyssey", odyssey), ("Iliad", iliad)):
    text = "".join(books)
    unknown = sum(c not in shared.char_to_id for c in text)
    print(f"{name}: {unknown} unknown characters of {len(text):,}")
guess = math.log(shared.vocab_size)
print(f"random guessing now costs ln(175) = {guess:.3f} for every model")"""
            ),
            saw(
                "A vocabulary of 175 (173 characters and the two special ones), and **0** unknown characters in either poem. "
                "There's one cost. The Odyssey model gets table rows for six Iliad-only characters it never sees. Those rows stay at their random start, "
                "just like `<|unk|>` in [02_02](../02_text_to_numbers/02_02_special_tokens.ipynb), and the model learns never to guess them."
            ),
            q(
                "Step 3: how do I set up the fair experiment?",
                "We write the rules down first, like the results log of [05_04](../05_training_on_the_odyssey/05_04_running_fair_experiments.ipynb):\n\n"
                "| | Same for every run | The ONE thing that differs |\n|---|---|---|\n"
                "| tokenizer | the shared one (175) | |\n"
                "| text size | the first 10,000 characters of Book 1 of each poem | |\n"
                "| split | first 90% train, last 10% held out (05_05) | |\n"
                "| model, epochs, lr | `HomerGreekLLM`, 4 epochs, 0.01 | |\n"
                "| seed | 0 and 1 (two runs, then the average) | |\n"
                "| **training poem** | | **Odyssey for model A, Iliad for model B** |\n\n"
                "Each model is then tested on both held-out parts (1,000 characters each). "
                "Short slices and 4 epochs keep it to a few seconds per model.\n\n"
                "👉 **Predict first:** on the Odyssey's held-out part, which model will have the lower loss? And on the Iliad's?",
            ),
            code(
                r"""def book_1_split(books):
    data = torch.tensor(shared.encode(books[0][:SLICE]))
    return split_90_10(data)  # (train, held-out)


odyssey_train, odyssey_val = book_1_split(odyssey)
iliad_train, iliad_val = book_1_split(iliad)
print("Odyssey: train", len(odyssey_train), "held-out", len(odyssey_val))
print("Iliad:   train", len(iliad_train), "held-out", len(iliad_val))

SEEDS = (0, 1)
tests = {"Odyssey held-out": odyssey_val, "Iliad held-out": iliad_val}
trained_on = {"Odyssey": odyssey_train, "Iliad": iliad_train}

table = {}  # (trained on, tested on) -> average loss over the seeds
for poem, train_data in trained_on.items():
    runs = [fit(train_data, shared.vocab_size, seed) for seed in SEEDS]
    for test, val_data in tests.items():
        table[(poem, test)] = sum(evaluate(m, val_data) for m in runs) / len(runs)

print()
print(f"{'':18} {'tested on Odyssey':>18} {'tested on Iliad':>16}")
for poem in trained_on:
    print(
        f"trained on {poem:7} {table[(poem, 'Odyssey held-out')]:>18.3f} "
        f"{table[(poem, 'Iliad held-out')]:>16.3f}"
    )"""
            ),
            saw(
                "The Odyssey model scores 2.811 on the Odyssey's held-out part and 3.026 on the Iliad's. The Iliad model scores 2.880 and 2.873. Each number is the average of two runs (seeds 0 and 1). Random guessing would give 5.165, so both models learned something about Greek. The picture shows the pattern."
            ),
            code(
                r"""import matplotlib.pyplot as plt

from homer_ai.studio_style import GOOD_BAD, INK, studio_axes, studio_figure, text_on

values = [[table[(p, t)] for t in tests] for p in trained_on]
low = min(min(row) for row in values) - 0.05
high = max(max(row) for row in values) + 0.05

fig = studio_figure("fair comparison", "Loss: own poem against the other poem", (8, 6))
ax = studio_axes(fig, (0.2, 0.12, 0.7, 0.62))
ax.imshow(values, cmap=GOOD_BAD.reversed(), vmin=low, vmax=high, aspect="auto")
for r, row in enumerate(values):
    for c, value in enumerate(row):
        middle = abs(2 * (value - low) / (high - low) - 1)  # far from the middle
        ax.text(c, r, f"{value:.2f}", ha="center", va="center", fontsize=20,
                fontweight="bold", family="monospace", color=text_on(middle, 1))
ax.set_xticks([0, 1], ["Odyssey\nheld-out", "Iliad\nheld-out"])
ax.set_yticks([0, 1], ["trained on\nOdyssey", "trained on\nIliad"])
ax.tick_params(axis="both", labelsize=12, labelcolor=INK, length=0)
ax.grid(visible=False)
plt.show()"""
            ),
            saw(
                "Green is the lower (better) loss and pink-red the higher. In the left column the Odyssey model is greener than the Iliad model (2.81 against 2.88). In the right column the Iliad model is much greener (2.87 against 3.03)."
            ),
            q(
                "What does the table really say?",
                "Read **down each column** (the same test text):\n\n- On the Odyssey's held-out part, the Odyssey model is better: 2.811 against 2.880 (a gap of 0.07).\n- On the Iliad's held-out part, the Iliad model is better: 2.873 against 3.026 (a gap of 0.15).\n\nSo each model does a little better on its own poem, as we expected. But a few warnings before you trust that:\n\n- The gaps are small, the model is tiny, and we have two seeds and only 1,000 held-out characters per poem. So treat the 0.07 as fragile. Set `SLICE = 8000` in the first cell and see whether the order in the Odyssey column holds (that's a fair experiment too, because you change only one thing).\n- Don't read across a row. The Iliad's held-out part is a different text from the Odyssey's, and it may simply be harder or easier. Only differences down a column are fair.\n- Our model has a window of 24 characters and one attention head, so it can only learn short letter patterns. Both poems share such patterns, which is why the models aren't far apart.",
            ),
            recap(
                [
                    "Separate tokenizers have different sizes and move letters to other numbers, so models built on them can't be compared. One shared vocabulary fixes it.",
                    "A fair 2×2 table keeps the same slices, seed, epochs and tokenizer. The training poem is the only change, and each model is tested on both held-out parts.",
                    "Report small, noisy gaps as small and noisy.",
                ],
                "[07_04: Combining both](07_04_combining_both.ipynb).",
            ),
        ],
    )

    # ---------------------------------------------------------------- 07_04
    save(
        "07_advanced_iliad_and_combining/07_04_combining_both.ipynb",
        [
            title("07_04 Combining both: one model, two poems"),
            where(
                "model learns",
                [
                    "src/homer_ai/train_homer.py",
                    "src/homer_ai/tokenizer.py",
                    "src/homer_ai/model.py",
                ],
            ),
            learn(
                [
                    "How to put both poems into one training text, with `<|endoftext|>` between them",
                    "How to compare a combined model fairly with the two single-poem models",
                    "How to read the result, and what a second change hidden in the test does",
                    "What combining gives and what it costs",
                ]
            ),
            words(
                [
                    (
                        "confound",
                        "a second change hidden inside the one you meant to test, so you can't tell which one caused the result",
                        "the combined model also saw twice as much text",
                    ),
                    (
                        "specialist",
                        "a model trained on only one poem, as opposed to one trained on both",
                        "the Odyssey-only model is the specialist on the Odyssey",
                    ),
                ]
            ),
            *background(),
            q(
                "How do I make ONE training text from two poems?",
                "With the same loop as `load_data` in `train_homer.py` ([02_02](../02_text_to_numbers/02_02_special_tokens.ipynb)). We encode each text on its own, and put the number of `<|endoftext|>` between them. "
                "The tokenizer is the shared vocabulary of [07_03](07_03_comparing_fairly.ipynb). The marker is how the model learns that one work has ended and another begins.\n\n"
                "```\n"
                "Odyssey slice → [ … ]  +  [<|endoftext|>]  +  Iliad slice → [ … ]\n"
                "```\n\n"
                "👉 **Predict first:** the code prints the end of the Odyssey slice, the marker, and the start of the Iliad slice. What will it show?",
            ),
            code(CH7_TOOLS),
            code(
                r"""shared = CharTokenizer("\n".join(odyssey + iliad))  # the shared vocabulary (07_03)


def join_with_eot(parts):  # the loop of load_data (02_02); parts are tensors
    ids = []
    for part in parts:
        if ids:  # not the first part: mark where the previous one ended
            ids.append(shared.eot_id)
        ids.extend(part.tolist())
    return torch.tensor(ids, dtype=torch.long)


odyssey_1 = torch.tensor(shared.encode(odyssey[0][:SLICE]))  # slice of Book 1
iliad_1 = torch.tensor(shared.encode(iliad[0][:SLICE]))
odyssey_train, odyssey_val = split_90_10(odyssey_1)
iliad_train, iliad_val = split_90_10(iliad_1)

both = join_with_eot([odyssey_train, iliad_train])
at = len(odyssey_train)  # where the Odyssey part ends
print("length:", len(both), "=", len(odyssey_train), "+ 1 marker +", len(iliad_train))
print("ends of the Odyssey part:", repr(shared.decode(both[at - 8 : at].tolist())))
marker = repr(shared.decode(both[at : at + 1].tolist()))
print("the marker:              ", marker, "= number", both[at].item())
print("start of the Iliad part: ", repr(shared.decode(both[at + 1 : at + 9].tolist())))"""
            ),
            saw(
                "The combined text is the two training parts plus one marker (number 174), with the Odyssey's last 8 characters on one side and the Iliad's first 8 on the other. "
                "The marker is one token, not 13 characters, just as in [02_02](../02_text_to_numbers/02_02_special_tokens.ipynb)."
            ),
            step_intro(),
            q(
                "Step 1: what exactly do I compare?",
                "Four models. They all get the shared vocabulary, 4 epochs and seeds 0 and 1, and they're all tested on both held-out parts:\n\n"
                "| Model | Trained on | Training characters |\n|---|---|---|\n"
                "| A | Odyssey slice | 9,000 |\n"
                "| B | Iliad slice | 9,000 |\n"
                "| C | both slices, in full | 18,001 |\n"
                "| D | the first half of each slice | 9,001 |\n\n"
                'C is the natural "combined" model, but it sees twice as much text as A and B, so it changes two things at once: the mix and the amount. That\'s a **confound**, and it breaks the rule of [05_04](../05_training_on_the_odyssey/05_04_running_fair_experiments.ipynb) (change one thing). '
                "D fixes it. It gets the same amount of text as A and B (9,001 characters), half from each poem. "
                "So comparing D with A and B changes only the mix.\n\n"
                "👉 **Predict first:** which will be best on the Odyssey's held-out part, A or C or D? And what will D look like next to A and B?",
            ),
            code(
                r"""SEEDS = (0, 1)
half = len(odyssey_train) // 2
training = {
    "A  Odyssey": odyssey_train,
    "B  Iliad": iliad_train,
    "C  both, in full": join_with_eot([odyssey_train, iliad_train]),
    "D  both, half each": join_with_eot([odyssey_train[:half], iliad_train[:half]]),
}

results = {}  # name -> (loss on Odyssey held-out, loss on Iliad held-out)
first_runs = {}  # name -> the seed-0 model, kept for writing below
for name, data in training.items():
    runs = [fit(data, shared.vocab_size, seed) for seed in SEEDS]
    first_runs[name] = runs[0]
    results[name] = (
        sum(evaluate(m, odyssey_val) for m in runs) / len(runs),
        sum(evaluate(m, iliad_val) for m in runs) / len(runs),
    )

print(f"{'model':20} {'chars':>6} {'Odyssey held-out':>17} {'Iliad held-out':>15}")
for name, (on_odyssey, on_iliad) in results.items():
    print(f"{name:20} {len(training[name]):>6,} {on_odyssey:>17.3f} {on_iliad:>15.3f}")"""
            ),
            saw(
                "C (both poems, in full) matches A on the Odyssey (2.800 against 2.811) and beats B on the Iliad (2.809 against 2.873). D (the same amount of text as A and B, half from each poem) is **worse than the specialist on each poem**. On the Odyssey it gets 2.930 against A's 2.811, and on the Iliad 3.024 against B's 2.873. In fact on the Iliad, D (3.024) is no better than A (3.026), which never saw the Iliad. The graph shows it. Just note that its axis starts at 2.5, which makes small gaps look big."
            ),
            code(
                r"""import matplotlib.pyplot as plt

from homer_ai.studio_style import (
    ACCENT,
    BAD,
    GOOD,
    INK,
    MUTED,
    studio_axes,
    studio_figure,
)

title = "Held-out loss of the four models (lower is better)"
fig = studio_figure("combining", title, (10, 6))
ax = studio_axes(fig, (0.1, 0.12, 0.8, 0.64))
colours = [ACCENT, GOOD, BAD, MUTED]
width = 0.2
for n, (name, colour) in enumerate(zip(results, colours, strict=True)):
    positions = [i + (n - 1.5) * width for i in range(2)]
    ax.bar(positions, results[name], width, color=colour, label=name)
    for x, value in zip(positions, results[name], strict=True):
        ax.text(x, value + 0.01, f"{value:.2f}", ha="center", color=INK,
                family="monospace", fontsize=10, fontweight="bold")
ax.set_xticks([0, 1], ["Odyssey held-out", "Iliad held-out"])
ax.tick_params(axis="x", labelsize=13, labelcolor=INK)
ax.set_ylim(2.5, 3.25)
ax.set_ylabel("loss (this axis starts at 2.5, not 0)")
ax.grid(axis="x", visible=False)
ax.set_axisbelow(True)
ax.legend(facecolor="none", edgecolor=MUTED, labelcolor=INK, loc="upper center",
          ncols=2)
plt.show()"""
            ),
            saw(
                "C looks best, but it isn't a fair win. It saw twice as much text, so it also took twice as many training steps. D changes **only the mix**, and it's worse than each specialist. So in this experiment, mixing the poems didn't help by itself, and C's gain came from more text. Here's a guess, which we didn't test: the model is small, with only 14,272 weights (`175 × 32 × 2` for the embedding and the head, plus `3 × 32 × 32` for query, key and value), and they have to serve two sets of habits. We have two seeds and tiny slices, so a bigger model, more text or more epochs could change this."
            ),
            q(
                "Step 2: does the combined model answer differently to different prompts?",
                'We can ask a model that knows both poems with the first word of each: `μῆνιν` (the Iliad\'s "anger") and `ἄνδρα` (the Odyssey\'s "man"). '
                "We use model C with top-k 5 and temperature 0.8 ([06_02](../06_making_it_talk/06_02_less_repetition.ipynb)), the same seed, and 70 new characters. "
                "Model A (Odyssey only) and B (Iliad only) get the same prompts for comparison.\n\n"
                "👉 **Predict first:** will the combined model's text start to look like the poem the prompt came from?",
            ),
            code(
                r"""from homer_ai.lessons import write  # the function of 06_02

for model_name in ("A  Odyssey", "B  Iliad", "C  both, in full"):
    for prompt in ("μῆνιν", "ἄνδρα"):
        text = write(first_runs[model_name], shared, prompt, 70)
        print(f"{model_name[0]} {prompt:7}", text)
    print()"""
            ),
            saw(
                "The prompt only changes the first few characters. After `μῆνιν` or `ἄνδρα`, each model goes on with almost the **same text** for both prompts (model C writes `…τενα με τὰ κει τεν τʼ ἐγο καν καν δʼ …` both times). The prompt is only 5 characters, and it stays inside the model's 24-character window until 19 more characters are written. Even so, the two texts are almost the same. For this small model the prompt hardly steers the writing. Models B and C, which saw the Iliad, begin alike (`μῆνινα τε…`), while A, the Odyssey-only model, writes something else (`μῆνιντουν ἀλοὐ…`). So we see no sign that the combined model writes Iliad-like Greek after `μῆνιν` and Odyssey-like Greek after `ἄνδρα`. To be fair, expecting a model this small to separate the two poems would be asking too much."
            ),
            q(
                "So what does combining give, and what does it cost?",
                "Going by what we measured:\n\n**What it gives**\n\n- One model, one file and one tool for both poems.\n- More text to learn from. C beat B on the Iliad's held-out part (2.809 against 2.873) and matched A on the Odyssey's.\n- One vocabulary, so every character of either poem is known.\n\n**What it costs**\n\n- With the same amount of text, the mixed model D was worse than each specialist on its own poem (2.930 against 2.811, and 3.024 against 2.873).\n- The model's few weights must cover two poems, and two editors' habits ([07_01](07_01_the_corpus.ipynb)).\n- It can't say which poem it's writing. The prompt didn't steer it (above), so you can't simply ask for \"Iliad Greek\".\n- Twice the text means twice the training time.\n\nSo for a model this small, combining is a convenience. It doesn't make the model better. It gets interesting with a bigger model (the parts of chapter 4) and the whole texts.",
            ),
            q(
                "What about the other two ways to combine?",
                "**Two models side by side** needs no new maths. You train A and B as before, and a program (like the MCP server of [06_03](../06_making_it_talk/06_03_the_mcp_server.ipynb)) offers one tool for each, or one tool with a `poem` argument that picks the model. "
                "Each model keeps its own poem's habits, and you pay with two files and two training runs.\n\n"
                "**Fine-tuning** (training on the Odyssey, then a little more on the Iliad) isn't built here. It needs care with the learning rate so the model doesn't forget the first poem, and our tiny model would show little. "
                "It's the idea behind adapting big models to a new job, and [chapter 8](../08_advanced_fine_tuning/08_00_chapter_overview.ipynb) does exactly that with GPT-2.",
            ),
            recap(
                [
                    "For one model on both poems, encode each text, put `<|endoftext|>` between them, and use the shared vocabulary.",
                    "On the same amount of text, the mixed model was worse than each specialist on its own poem (2.930 against 2.811, and 3.024 against 2.873). The combined model's gain came from seeing more text.",
                    "Combining means one file and one tool, but the model's few weights now serve two styles, and the prompt didn't steer it towards one poem or the other.",
                ],
                "[07_05: English and Greek](07_05_english_and_greek.ipynb).",
            ),
        ],
    )

    # ---------------------------------------------------------------- 07_05
    save(
        "07_advanced_iliad_and_combining/07_05_english_and_greek.ipynb",
        [
            title("07_05 English and Greek: the same steps in Butler's English"),
            where(
                "numbers",
                ["data/homer/odyssey/english/", "src/homer_ai/tokenizer.py"],
            ),
            learn(
                [
                    "How the same code handles Butler's English, and how its vocabulary compares with Greek's",
                    "Why prose (long paragraph lines) differs from verse",
                    "How many tokens characters and GPT-2's tokenizer use for English and for Greek",
                    "How a tiny English model trains and writes",
                ]
            ),
            words(
                [
                    (
                        "verse",
                        "writing in short lines with a rhythm: the Greek poems have one verse per line",
                        "`μῆνιν ἄειδε θεὰ Πηληϊάδεω Ἀχιλῆος`",
                    ),
                    (
                        "prose",
                        "ordinary writing in sentences and paragraphs: Butler's translation",
                        "one paragraph per line in `data/homer/*/english/`",
                    ),
                ]
            ),
            *background(),
            q(
                "Can the same code learn English?",
                "Yes. The code never knew the text was Greek. `CharTokenizer` takes whatever characters it finds. "
                "Here's Butler's Book 1 of the Odyssey, next to the Greek Book 1, with the vocabulary of each (the number of different characters, plus our 2 special tokens).\n\n"
                "👉 **Predict first:** English has far fewer different characters than Greek (about 67 against 167, see [07_01](07_01_the_corpus.ipynb)). Will Book 1 alone follow the same pattern?",
            ),
            code(CH7_FIND),
            code(
                r"""import math

import torch
import torch.nn.functional as F

from homer_ai.model import HomerGreekLLM
from homer_ai.tokenizer import CharTokenizer

greek = load_book_files(HOMER / "odyssey" / "greek")
english = load_book_files(HOMER / "odyssey" / "english")

for name, books in (("Greek", greek), ("English", english)):
    book_1 = books[0]
    whole = CharTokenizer("\n".join(books))
    in_poem = len(set("".join(books)))
    print(
        f"{name:8} book 1: {len(book_1):,} characters, {len(set(book_1))} different; "
        f"whole poem: {in_poem} different, vocabulary {whole.vocab_size}"
    )"""
            ),
            saw(
                "Book 1 uses 135 different characters in Greek and 64 in English. The whole poem uses 167 and 67. "
                "Few new letters appear after Book 1, because the whole alphabet and punctuation show up early. The English vocabulary (69 with the special tokens) is less than half the Greek (169)."
            ),
            step_intro(),
            q(
                "Step 1: what's different about prose? (lines are paragraphs)",
                'Greek verse has one verse per line, and English prose has one **paragraph** per line. So in English, line breaks are much rarer, and a "line" is much longer. '
                "What does that mean for a model that learns which letter comes next?\n\n"
                "👉 **Predict first:** the model reads 24 characters at a time. In Greek a verse is about 44 characters. Roughly how often does a line break (`⏎`) come up in each text?",
            ),
            code(
                r"""for name, books in (("Greek", greek), ("English", english)):
    book_1 = books[0]
    lines = book_1.splitlines()
    breaks = book_1.count("\n")
    every = len(book_1) / breaks
    print(f"{name:8} {len(lines):>3} lines; a break every {every:.0f} characters")

print()
print("Greek line 1:  ", greek[0].splitlines()[0])
paragraph = english[0].splitlines()[0]
print("English line 1:", paragraph[:60], "…", f"({len(paragraph)} characters)")"""
            ),
            saw(
                "One break every 44 characters in Greek, and one every 665 in English. So a model with a 24-character window almost never sees a break in English, while the Greek model sees one about every other window. "
                "The first English line is a whole paragraph of 605 characters, not a verse. Butler's paragraphs don't match Greek verses one to one, which is why we can't pair them line by line."
            ),
            q(
                "Step 2: how many tokens do characters and GPT-2's tokenizer use for each language?",
                "Our tokenizer gives one token per character, always. GPT-2's tokenizer (via `tiktoken`, [A_05](../A_toolbox/A_05_tiktoken_bpe.ipynb)) is a BPE ([02_03](../02_text_to_numbers/02_03_bpe.ipynb)) that learned its merges from mostly English text. "
                "We count tokens for the first 1,000 characters of Book 1 in each language, and for the Greek verse and the English paragraph. "
                "(The first time, `tiktoken` downloads GPT-2's vocabulary file, so this needs an internet connection once.)\n\n"
                "👉 **Predict first:** for English, will GPT-2 use more or fewer tokens than characters? And for Greek? ([A_05](../A_toolbox/A_05_tiktoken_bpe.ipynb) found Greek needing more tokens than letters.)",
            ),
            code(
                r"""import tiktoken

# the first time, this downloads GPT-2's vocabulary file (needs internet once)
gpt2 = tiktoken.get_encoding("gpt2")

samples = {
    "Greek, first 1,000 characters": greek[0][:1000],
    "English, first 1,000 characters": english[0][:1000],
    "Greek, verse 1": greek[0].splitlines()[0],
    "English, paragraph 1": english[0].splitlines()[0],
}
print(f"{'text':33} {'characters':>10} {'GPT-2 tokens':>13} {'chars per token':>16}")
counts = {}
for name, text in samples.items():
    tokens = len(gpt2.encode(text))
    counts[name] = tokens
    print(f"{name:33} {len(text):>10,} {tokens:>13,} {len(text) / tokens:>16.2f}")"""
            ),
            saw(
                "1,000 English characters become 227 GPT-2 tokens (4.41 characters per token), but 1,000 Greek characters become **1,284** tokens (0.78 characters per token). That's more tokens than letters. The single Greek verse is 50 characters and 57 tokens. GPT-2's merges were learned from mostly English text, so for Greek it often falls back to pieces of single letters ([A_05](../A_toolbox/A_05_tiktoken_bpe.ipynb))."
            ),
            code(
                r"""import matplotlib.pyplot as plt

from homer_ai.studio_style import BAD, GOOD, INK, MUTED, studio_axes, studio_figure

names = ["our tokenizer\n(any text)", "GPT-2 on\nEnglish", "GPT-2 on\nGreek"]
values = [
    1000,
    counts["English, first 1,000 characters"],
    counts["Greek, first 1,000 characters"],
]
fig = studio_figure("tokens", "Tokens needed for 1,000 characters", (9, 5.5))
ax = studio_axes(fig, (0.1, 0.12, 0.8, 0.64))
ax.bar(names, values, color=[MUTED, GOOD, BAD], width=0.55)
for x, value in enumerate(values):
    ax.text(x, value + 25, f"{value:,}", ha="center", color=INK,
            family="monospace", fontsize=14, fontweight="bold")
ax.tick_params(axis="x", labelsize=12, labelcolor=INK)
ax.set_ylabel("tokens")
ax.set_ylim(0, 1500)
ax.grid(axis="x", visible=False)
ax.set_axisbelow(True)
plt.show()"""
            ),
            saw(
                "Our tokenizer is the grey bar, always 1,000 for 1,000 characters. GPT-2 is much shorter on English (green) and much longer on Greek (pink-red). So which tokenizer is better depends on the language. A BPE saves a lot on English, but GPT-2's BPE is worse than plain characters for Greek. A BPE trained on Greek is one of the ideas at the end of this notebook. (Mind that this counts tokens only. It doesn't measure quality.)"
            ),
            q(
                "Step 3: can a tiny model learn English?",
                "We follow the same steps as for the Iliad in [07_02](07_02_an_iliad_model.ipynb): a tokenizer from Butler's Book 1, the first 10,000 characters, a 90/10 split ([05_05](../05_training_on_the_odyssey/05_05_validation.ipynb)), the model, 6 epochs, a fixed seed. "
                "Nothing in `HomerGreekLLM` changes (the name says Greek, but nothing inside it is Greek-specific).\n\n"
                "👉 **Predict first:** the vocabulary is 66, so a random guess costs `ln(66)`. Will the trained loss be higher or lower than Greek's loss near 3 in [07_02](07_02_an_iliad_model.ipynb)?",
            ),
            code(
                r"""from homer_ai.lessons import CHUNK, evaluate  # 05_05

tokenizer = CharTokenizer(english[0])  # only Book 1: a small, quick vocabulary
data = torch.tensor(tokenizer.encode(english[0][:10000]))
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


torch.manual_seed(42)
model = HomerGreekLLM(vocab_size=tokenizer.vocab_size, embed_dim=32)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
guess = math.log(tokenizer.vocab_size)
print(f"vocabulary {tokenizer.vocab_size}; random guessing = ln(66) = {guess:.3f}")
print(f"untrained validation loss: {evaluate(model, val_data):.3f}")

for epoch in range(6):
    for i in range(0, len(train_data) - CHUNK, CHUNK):
        inputs = train_data[i : i + CHUNK]
        targets = train_data[i + 1 : i + CHUNK + 1]
        loss = F.cross_entropy(model(inputs), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss = evaluate(model, train_data)
    val_loss = evaluate(model, val_data)
    print(f"epoch {epoch + 1} | train {train_loss:.3f} | validation {val_loss:.3f}")"""
            ),
            saw(
                "The untrained loss is 4.208, near `ln(66) = 4.190` (pure guessing). After six epochs the training loss is 2.411 and the validation loss 2.404. They're close together and still falling, so it learned real letter patterns and didn't memorise. It ends lower than Greek's (about 2.8 in 07_02), but see the next question before you conclude anything from that."
            ),
            q(
                "What does the English model write?",
                "We use the same top-k writing as [06_02](../06_making_it_talk/06_02_less_repetition.ipynb), from `Tell me, O Muse` (Butler's first words): 120 new characters, top-k 5 and temperature 0.8, with greedy for comparison.\n\n"
                "👉 **Predict first:** will it write English words, or word-shaped pieces?",
            ),
            code(
                r"""from homer_ai.lessons import write  # the function of 06_02; top_k=1 is greedy

print("greedy:", write(model, tokenizer, "Tell me, O Muse", 120, top_k=1))
print("top-5: ", write(model, tokenizer, "Tell me, O Muse", 120))"""
            ),
            saw(
                "Greedy loops at once (`an an an …`, like the Greek `και και και`). Top-5 writes English-shaped pieces with a few real small words (`and`, `an`, `or`): `…Musend therisof angin ou an ther illo and,ofe athar har and tha …`. So it learned letter patterns like `th`, `an`, `er`, but no sentences. Same as for Greek."
            ),
            q(
                "Can I compare this English loss with the Greek loss?",
                "No. The vocabularies differ (66 against 175 or 171), the texts differ, and the held-out parts are different texts. That's exactly the unfairness of [07_03](07_03_comparing_fairly.ipynb). "
                'A lower English loss only says that guessing the next letter among fewer, simpler characters is easier. It doesn\'t say English is "better".',
            ),
            q(
                "Why does Butler write `Ulysses` and `Minerva`?",
                "Butler used the Roman (Latin) names: `Ulysses` is the Roman name for Odysseus, and `Minerva` the Roman name for Athena. "
                "A character-level model trained on his English never sees the Greek names, and the Greek model never sees the Roman ones.",
            ),
            md(
                "## 6. Recap\n\n"
                "- The same code reads English. The vocabulary is much smaller (67 characters against 167), and prose has long lines, so line breaks are rare.\n"
                "- GPT-2's tokenizer needs far fewer tokens for English (about 4.4 characters per token), but more tokens than letters for Greek.\n"
                "- A tiny model learns English letter patterns, not sentences. Its loss can't be compared with Greek's (different vocabulary and text)."
            ),
            md(
                "## You finished the course\n\n"
                "### ❓ What did I build?\n\n"
                "Starting from nothing but Homer's text, you built every part of a small language model yourself:\n\n"
                "- A tokenizer that turns Greek into numbers and back (chapter 2).\n"
                "- Attention and the parts of a GPT-style model (chapters 3 and 4).\n"
                "- A training loop, with loss, learning rate and a held-out check (chapter 5).\n"
                "- A model that writes Greek, and an MCP tool that lets Claude call it (chapter 6).\n"
                "- An Iliad model, a fair comparison of two models, and one model for both poems (this chapter).\n\n"
                "### ❓ What can I do now?\n\n"
                "- Explain, step by step, how an LLM turns text into a next-token guess, and why it needs training.\n"
                "- Read a training run: a loss that falls, one that jumps, and one that only memorises.\n"
                "- Run a fair experiment: change one thing, fix the seed, test on text the model hasn't seen.\n"
                "- Offer your own model to an AI app as a tool.\n\n"
                "Yours is a toy (835,072 weights in the shipped GPT, 11,840 in the small model, against GPT-2 small's 124 million). But it's built the same way, and you know where every part sits.\n\n"
                "*When I build chatbots for regulated teams, the first question is always: where did that answer come from? Keep asking it of everything you build.*\n\n"
                "### ❓ What could I try myself?\n\n"
                "Here are three things you can do with the code you have:\n\n"
                "1. **A BPE tokenizer trained on Greek.** One character per token makes lines long and tokens weak. You built a small BPE in [02_03](../02_text_to_numbers/02_03_bpe.ipynb); training one on `data/homer/` (instead of using GPT-2's) would give Greek-sized tokens.\n"
                "2. **Make the GPT bigger.** The shipped GPT ([04_05](../04_building_the_model/04_05_the_full_model.ipynb), `src/homer_ai/gpt.py`) is small on purpose, to train in minutes. Change `GPT_SETTINGS` in `train_homer.py` (wider, more blocks) and compare held-out loss fairly, as in [05_04](../05_training_on_the_odyssey/05_04_running_fair_experiments.ipynb). With more text, a bigger model can pay off.\n"
                "3. **Train on the whole Odyssey (and the Iliad).** We used one book at a time. All 24 books give about 540,000 characters, which needs longer runs (see [07_02](07_02_an_iliad_model.ipynb)) and, probably, the bigger GPT of idea 2.\n\n"
                "And there are three more advanced chapters if you want them. [Chapter 8](../08_advanced_fine_tuning/08_00_chapter_overview.ipynb) loads GPT-2's 124 million weights into the `GPTModel` you built, and fine-tunes it for a new job. [Chapter 9](../09_advanced_grounded_answers/09_00_chapter_overview.ipynb) answers questions about Homer from the real text, with the book and line beside each answer. [Chapter 10](../10_advanced_your_domain/10_00_chapter_overview.ipynb) takes the same pattern to a company's own documents, emails and tickets, on invented data.\n\n"
                "Stuck on a tool? The [toolbox](../A_toolbox/A_00_overview.ipynb) has it. Want the whole map again? [The course map](../00_start_here/00_00_course_map.ipynb). "
                "And if you haven't read it: Sebastian Raschka's book (see the [README](../../README.md)) goes much further than we did."
            ),
        ],
    )
