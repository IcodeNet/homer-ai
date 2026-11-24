# ruff: noqa: E501
"""Chapter 2, Text to numbers: tokens, BPE, the sliding window, embeddings, positions."""

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
from .diagrams import DIAGRAM_LOOKUP, DIAGRAM_WINDOW


def build():
    # =====================================================================
    # 02_text_to_numbers/02_00_chapter_overview.ipynb
    # =====================================================================
    save(
        "02_text_to_numbers/02_00_chapter_overview.ipynb",
        [
            title("02_00 Text to numbers (overview)", "chapter 2"),
            where(
                "numbers", ["src/homer_ai/tokenizer.py", "src/homer_ai/train_homer.py"]
            ),
            learn(
                [
                    "Why a model can't read letters, and needs numbers",
                    "What tokens, a vocabulary and token IDs are",
                    "How text becomes numbers, and numbers become text again",
                    "Why we need special tokens, and what BPE is",
                    "Which notebook to open for each idea",
                ]
            ),
            words(
                [
                    (
                        "token",
                        "one piece of text the model reads as a single unit",
                        "`α`, or `λλα`",
                    ),
                    (
                        "vocabulary",
                        "the list of all the tokens the model knows",
                        "our 137 tokens",
                    ),
                    (
                        "token ID",
                        "the number a token gets in the vocabulary",
                        "`π` → 41 in the real Book 1 vocabulary (in 02_01's 5-character toy it's 4)",
                    ),
                    (
                        "encode",
                        "turn text into token IDs",
                        "`πολλὰ` → `[41, 40, 36, 36, 110]`",
                    ),
                    (
                        "decode",
                        "turn token IDs back into text",
                        "`[41, 40, 36, 36, 110]` → `πολλὰ`",
                    ),
                    (
                        "tokenizer",
                        "the tool that does the encoding and decoding",
                        "`CharTokenizer`",
                    ),
                    (
                        "special token",
                        "an extra token that is a signal, not real text",
                        "`<|unk|>`, `<|endoftext|>`",
                    ),
                    (
                        "BPE",
                        "byte pair encoding: a way to make tokens out of common groups of letters",
                        "`ν` + space → one token",
                    ),
                ]
            ),
            *background(
                "At the very first stage, **Odyssey text**. We have Book 1 of the Odyssey as a text file: "
                "444 lines of Greek, 19,701 characters. Nothing has been turned into numbers yet.",
                "A model is a big calculation. It can only do maths, so it can't read `ἄνδρα` the way we do. "
                "This chapter turns the Greek text into numbers. Those numbers are **all the model will ever see**, "
                "so if a letter is lost or mixed up here, no amount of training can fix it later.",
            ),
            q(
                "So how do we turn Greek into numbers?",
                "Remember the secret code from school, where A is 1, B is 2, and a message becomes a row of numbers? "
                "This chapter builds that code for Homer's Greek: a key sheet with one number for every character, a way to write the poem as numbers, and a way to read it back without losing a letter. "
                "Then we cut the numbered poem into flashcards, so the model has something to practise on ([02_04](02_04_sliding_window.ipynb)). "
                'The code picture stops working at one point: in a secret code the numbers are only labels, but a model does maths with them, and then 41 would count as "more" than 40. '
                "That's why the chapter doesn't end with the code. It ends by swapping each number for a small list of numbers the model can adjust ([02_05](02_05_token_embeddings.ipynb)).\n\n"
                "Building the code takes two steps:\n\n1. Cut the text into small pieces, called **tokens**. The simplest choice is one letter = one token.\n"
                "2. Give every different token its own number, its **token ID**. The list of all the tokens is the **vocabulary**.\n\n"
                "```\nπολλὰ  →  π  ο  λ  λ  ὰ  →  41  40  36  36  110\n```",
            ),
            q(
                "Can we get the text back from the numbers?",
                "Yes. Turning text into IDs is called **encoding**, and turning IDs back into text is decoding. "
                "If you encode and then decode, you should get back exactly what you started with. That check is called a round trip. "
                "And the tool that does both jobs is the tokenizer.",
            ),
            q(
                "What happens to a letter that isn't in the vocabulary?",
                "The lazy option is to skip it. But then `Ζεὺς!` quietly becomes `Ζεὺς`, and nobody can tell that anything went missing. "
                'So we add a **special token**, `<|unk|>` ("unknown"), which stands in for any letter the tokenizer has never seen.',
            ),
            q(
                "How does the model know where one book ends and the next begins?",
                "A line break won't do, because every verse already ends with one. "
                "So we put another special token, `<|endoftext|>`, between books.",
            ),
            q(
                "Do tokens have to be single letters?",
                'No. **BPE** (byte pair encoding) builds bigger tokens out of groups of letters that turn up a lot, like `πολλ` ("many"). '
                "And bigger tokens mean the model sees more of the poem at once.",
            ),
            q(
                "Which notebook do I open for each idea?",
                "| Notebook | Question it answers |\n|---|---|\n"
                "| [`02_01_tokens_and_vocabulary.ipynb`](02_01_tokens_and_vocabulary.ipynb) | How do letters become numbers, and back? |\n"
                "| [`02_02_special_tokens.ipynb`](02_02_special_tokens.ipynb) | What about unknown letters, and where a book ends? |\n"
                "| [`02_03_bpe.ipynb`](02_03_bpe.ipynb) | Can tokens be bigger than one letter? |\n"
                "| [`02_04_sliding_window.ipynb`](02_04_sliding_window.ipynb) | Where do the practice questions come from? |\n"
                "| [`02_05_token_embeddings.ipynb`](02_05_token_embeddings.ipynb) | How does a token ID become numbers that can carry meaning? |\n"
                "| [`02_06_positions.ipynb`](02_06_positions.ipynb) | How does the model know where each letter sits? |",
            ),
            recap(
                [
                    "A model only sees numbers: text is cut into tokens, and each token gets an ID from the vocabulary.",
                    "Encode (text → IDs) and decode (IDs → text) must give back exactly the same text.",
                    "Special tokens mark unknown characters and book ends; BPE makes bigger tokens from common groups.",
                ],
                "[02_01: Tokens and vocabulary](02_01_tokens_and_vocabulary.ipynb).",
            ),
        ],
    )

    # =====================================================================
    # 02_text_to_numbers/02_01_tokens_and_vocabulary.ipynb
    # =====================================================================
    save(
        "02_text_to_numbers/02_01_tokens_and_vocabulary.ipynb",
        [
            title("02_01 Tokens and vocabulary", "sections 2.2 and 2.3"),
            where("numbers", ["src/homer_ai/tokenizer.py"]),
            learn(
                [
                    "How to cut text into one-character tokens",
                    "How to build a vocabulary, and give each token a number",
                    "How to encode text into numbers, and decode numbers back into text",
                    "How our real `CharTokenizer` does all of this on Book 1",
                ]
            ),
            words(
                [
                    (
                        "token",
                        "one piece of text the model reads as a single unit; here, one character",
                        "`π`",
                    ),
                    (
                        "vocabulary",
                        "the list of all the different tokens",
                        "`[' ', 'α', 'λ', 'ο', 'π']`",
                    ),
                    (
                        "token ID",
                        "the number a token gets: its position in the vocabulary",
                        "`π` → 4 in our 5-character toy (41 in the real Book 1 vocabulary)",
                    ),
                    (
                        "encode / decode",
                        "text → IDs, and IDs → text",
                        "`πολα` → `[4, 3, 2, 1]`",
                    ),
                    (
                        "round trip",
                        "encode, then decode; you should get back exactly what you started with",
                        "`πολα` → numbers → `πολα`",
                    ),
                ]
            ),
            *background(
                "At the stage **Odyssey text → numbers**. We have Book 1 as plain Greek text. "
                "The model can't use it yet, because it only works with numbers.",
                "We need a fixed way to turn each character into a number, and back again. "
                '"Fixed" matters: if `π` were 41 today and 12 tomorrow, the trained model would read nonsense.',
            ),
            q(
                "What's the simplest way to cut text into tokens?",
                "Make every character a token. A **character** is any single symbol: a letter, a mark, a space. So yes, spaces and punctuation count too.\n\n```\nπολλα πολλα  →  π ο λ λ α ␣ π ο λ λ α\n```\n\n"
                "(␣ stands for a space.) It's simple, and every Greek word can be written this way.",
            ),
            q(
                "How do we decide which number each character gets?",
                "1. Collect every different character in the text.\n2. Put them in a fixed order (alphabetical).\n3. Number them from 0.\n\n"
                "That ordered list is the **vocabulary**, and a character's position in it is its token ID. "
                "Because we sort, the same text always gives the same numbers.",
            ),
            step_intro(),
            q(
                "How do I find all the different characters in a text?",
                "A `set` keeps only one copy of each thing, so the set of a text holds each character once.\n\n"
                '👉 **Predict first:** `"πολλα πολλα"` has 11 characters. How many will the set have?',
            ),
            code('text = "πολλα πολλα"\nset(text)'),
            saw(
                "Five different characters: `π`, `ο`, `λ`, `α` and the space. A set has no fixed order, so yours may come out in a different order."
            ),
            q(
                "Why sort them?",
                "Because a set has no fixed order, and the numbering has to be the same every time. So we sort it.\n\n"
                "👉 **Predict first:** which character will come first?",
            ),
            code("chars = sorted(set(text))\nchars"),
            saw(
                "The space comes first, because computers sort the space before the Greek characters. After that come the Greek characters, in alphabetical order."
            ),
            q(
                "How do I give each character a number?",
                "With a lookup table (a dictionary) from character to number. The first character gets 0, the next gets 1, and so on.\n\n"
                "👉 **Predict first:** what number will `π` get?",
            ),
            code("char_to_id = {ch: i for i, ch in enumerate(chars)}\nchar_to_id"),
            saw("Each character now has a number, from 0 to 4. So `π` got 4."),
            q(
                "How do I go back from numbers to characters?",
                "Build the same dictionary the other way round, number → character.",
            ),
            code("id_to_char = {i: ch for i, ch in enumerate(chars)}\nid_to_char"),
            saw(
                "The same pairs, flipped. Now we can look things up in both directions."
            ),
            q(
                "So how do I encode and decode a word?",
                "To encode, look up each character's number. To decode, look up each number's character and join them back into text.\n\n"
                "👉 **Predict first:** what numbers will `πολα` become, and will decoding give back `πολα`?",
            ),
            code(
                'ids = [char_to_id[c] for c in "πολα"]\nprint(ids)\nprint("".join(id_to_char[i] for i in ids))'
            ),
            saw(
                "Four numbers, `[4, 3, 2, 1]`, and the same word back. That's a successful **round trip**."
            ),
            md(
                "## 5. Full example\n\nThe project has a real tokenizer, `CharTokenizer`, in `src/homer_ai/tokenizer.py`. "
                "It does exactly the steps above, plus the special tokens from the next notebook. Here it is on all of Book 1.\n\n"
                "👉 **Predict first:** Book 1 has 135 different characters. Our vocabulary also has 2 special tokens. How big is the vocabulary?"
            ),
            code(
                r"""from homer_ai.tokenizer import CharTokenizer
from homer_ai.train_homer import BOOKS_DIR, load_books

book1 = load_books(BOOKS_DIR)[0]
tokenizer = CharTokenizer(book1)

print("characters in Book 1:", len(book1))
print("vocabulary size:", tokenizer.vocab_size)

first_line = book1.splitlines()[0]
ids = tokenizer.encode(first_line)
print(first_line)
print(ids[:10], "...")
print("round trip ok:", tokenizer.decode(tokenizer.encode(book1)) == book1)"""
            ),
            saw(
                "So, 19,701 characters and a vocabulary of 137 (135 characters plus the 2 special tokens). You also get the first line's numbers, and a perfect round trip of the whole book."
            ),
            q(
                "What numbers does `μοῦσα` get in the real vocabulary?",
                'Same idea, real size. `μοῦσα` ("Muse", in the first line) becomes one number per character:\n\n'
                "```\nμ    ο    ῦ    σ    α\n37   40   128  44   26\n```\n\n"
                "Notice that `ο` is 40 here, and it's 40 in `πολλὰ` too. A character always gets the **same** number.\n\n"
                "👉 **Predict first:** which number is the biggest, and why might `ῦ` be so far down the list?",
            ),
            code(
                """word = "μοῦσα"
ids = tokenizer.encode(word)
print("".join(f"{ch:<5}" for ch in word))
print("".join(f"{i:<5}" for i in ids))"""
            ),
            saw(
                "`37 40 128 44 26`. The vocabulary is sorted, and characters with accents and breathing marks "
                "(like `ῦ`) sort after the plain letters. That's why they get the bigger numbers."
            ),
            *exercise(
                "which numbers does `ἄνδρα` get?",
                "`ἄνδρα` is the first word of the *Odyssey*, and you'll meet its five numbers again and again in the next chapters. Encode it with Book 1's real tokenizer.\n\n"
                "👉 **Predict first:** `α` is ID 26. Will `ἄ`, the same letter with marks on it, be close to 26?",
                starter="""
            # 👉 do: turn the word "ἄνδρα" into its list of IDs, with the real tokenizer
            andra_ids = None
            """,
                solution="""
            andra_ids = tokenizer.encode("ἄνδρα")
            """,
                check="""
            check(
                "the IDs of ἄνδρα",
                andra_ids,
                [tokenizer.char_to_id[ch] for ch in "ἄνδρα"],
                coach="There should be 5 numbers, one for each character, from the real "
                "tokenizer: tokenizer.encode(...). The IDs of another word? Then check the "
                "spelling. ἄ carries two marks, so copy the word from the task.",
            )
            """,
                hint="The cell above did it for `μοῦσα`: `tokenizer.encode(word)`.",
                answer="`[58, 38, 29, 42, 26]`. `ἄ` is 58 and plain `α` is 26: to the tokenizer they're two different characters, as far apart as any other two. "
                "The sorting decides the numbers, and `ἄ` sorts long after the plain letters.",
            ),
            q(
                "Which characters does Book 1 use most?",
                "Count every character, then draw the 15 most common. Each label shows the character and its ID. "
                "The space and the line break are characters too, so they get a bar as well (`' '` and `'\\n'`).\n\n"
                "👉 **Predict first:** which will be the most common: a letter, the space, or the line break?",
            ),
            code(
                """from collections import Counter

import matplotlib.pyplot as plt

from homer_ai.studio_style import ACCENT, INK, MUTED, studio_axes, studio_figure

top = Counter(book1).most_common(15)
labels = [f"{ch!r}  ID {tokenizer.encode(ch)[0]}" for ch, _ in top]
counts = [n for _, n in top]

fig = studio_figure(
    "Book 1, character by character",
    f"The space leads with {counts[0]:,} of {len(book1):,} characters",
    size=(9, 7),
)
ax = studio_axes(fig, (0.2, 0.08, 0.72, 0.76))
ax.set_axisbelow(True)  # grid lines behind the bars
ax.set_xlim(0, counts[0] * 1.12)
bars = ax.barh(labels, counts, color=MUTED)
bars[0].set_color(ACCENT)
for bar, n in zip(bars, counts, strict=True):
    ax.text(
        n + 8, bar.get_y() + bar.get_height() / 2, f"{n:,}",
        va="center", color=INK, family="monospace", fontsize=10,
    )
ax.invert_yaxis()
ax.tick_params(axis="y", labelsize=11, labelcolor=INK)
ax.grid(axis="y", visible=False)
ax.set_xlabel("times it appears")
plt.show()"""
            ),
            saw(
                "The space wins by a long way (2,731), then come `ν`, `ο` and `ε`. The line break (ID 0) makes the top 15 too. "
                "And notice the IDs have nothing to do with how common a character is. They just follow the sorted vocabulary."
            ),
            q(
                "Greek has 24 letters. Why does Book 1 have 135 characters?",
                "Because every accent, breathing mark and capital makes a different character. Plain `α` alone turns up in Book 1 in **10** forms: "
                "`ά α ἀ ἁ ἂ ἄ ἅ ὰ ᾶ ᾷ`. Add the capitals, the space, the punctuation marks and the line break, and you get to 135.\n\n"
                "The line break (the end of each verse) is a character too. In code it's written `\\n`. That's one character, even though you type it with two symbols.",
            ),
            *questions(
                [
                    (
                        "Why does the tokenizer have exactly `encode`, `decode` and `vocab_size`?",
                        "Because that's all the rest of the model needs from it. Any tokenizer with those three can be swapped in, "
                        "a BPE one for example (02_03), without touching the model or the training code.",
                    ),
                ]
            ),
            recap(
                [
                    "Simplest tokens: one character each.",
                    "Vocabulary = every different character, sorted, numbered from 0.",
                    "Two dictionaries encode and decode; a round trip gives back exactly the same text.",
                ],
                "[02_02: Special tokens](02_02_special_tokens.ipynb): what about characters the vocabulary doesn't have?",
                can=[
                    "build a vocabulary from any text: `sorted(set(text))`, numbered from 0",
                    "encode a word into IDs and decode it back with the real `CharTokenizer`",
                    "check a tokenizer with a round trip",
                ],
            ),
        ],
    )

    # =====================================================================
    # 02_text_to_numbers/02_02_special_tokens.ipynb
    # =====================================================================
    save(
        "02_text_to_numbers/02_02_special_tokens.ipynb",
        [
            title("02_02 Special tokens", "section 2.4"),
            where(
                "numbers", ["src/homer_ai/tokenizer.py", "src/homer_ai/train_homer.py"]
            ),
            learn(
                [
                    "Why a tokenizer that skips unknown letters is a problem",
                    "How `<|unk|>` stands in for letters we've never seen",
                    "Why a line break isn't enough to show where a book ends",
                    "How `<|endoftext|>` is put between books",
                ]
            ),
            words(
                [
                    (
                        "special token",
                        "an extra token that isn't real text: a signal to the model, like a road sign",
                        "`<|unk|>`",
                    ),
                    (
                        "`<|unk|>`",
                        '"unknown": stands in for any character the tokenizer has never seen',
                        "`!` → `<|unk|>`",
                    ),
                    (
                        "`<|endoftext|>`",
                        '"one text ends here, a new one starts"',
                        "between Book 1 and Book 2",
                    ),
                ]
            ),
            *background(
                "At the stage **numbers**. In [02_01](02_01_tokens_and_vocabulary.ipynb) we built a vocabulary and turned letters into numbers. "
                "It works, as long as every letter is in the vocabulary.",
                "Two problems. **1.** Someone can type a letter the vocabulary has never seen, like `!` (Homer never uses it). "
                "A tokenizer that skips such letters loses them without a word. **2.** When we add more books, the model can't tell where one book ends. "
                "Both mean the model learns from text that isn't what we gave it.",
            ),
            q(
                "What happens to a letter the vocabulary doesn't have?",
                'Say the tokenizer just skips what it doesn\'t know. Encode `Ζεὺς!` ("Zeus!") and decode it again:\n\n```\nΖεὺς!  →  [11, 30, 115, 43]  →  Ζεὺς\n```\n\n'
                "Five characters went in, but only four numbers came out. The `!` is gone, and there was no warning.",
            ),
            q(
                "Why is that so bad?",
                "The model gets a different sentence from the one you typed, and nobody can tell. "
                "It's like a translator who skips every word they don't know and doesn't mention it.",
            ),
            q(
                "So what's the fix?",
                "Add one extra token to the vocabulary: `<|unk|>`, short for **unknown**. Any letter we don't know becomes `<|unk|>` and doesn't vanish:\n\n"
                "```\nΖεὺς!  →  [11, 30, 115, 43, 135]  →  Ζεὺς<|unk|>\n```\n\n"
                "The `!` itself is still lost, but at least you can see that something was there. "
                "In code: look each character up, and if it isn't in the vocabulary, use `<|unk|>`'s number. "
                "The `<| |>` brackets are there so it can't be mistaken for real Greek.\n\n"
                "*Checking input is a big part of fintech work: an ID that's one letter off has to be caught, not quietly dropped. Same idea here with `<|unk|>`.*",
            ),
            q(
                "Why does `<|unk|>` get the number 135?",
                "Book 1 has 135 different characters, numbered 0 to 134. We add `<|unk|>` at the end, so it gets the next free number, 135. "
                "Putting it at the end means no letter's number has to change.",
            ),
            q(
                "And the second problem: where does a book end?",
                "When books are glued together, the last line of Book 1 runs straight into the first line of Book 2, "
                "and the model thinks it's all one story. A line break can't mark the join, because Book 1 alone already has 444 of them, one per verse. "
                'So we add a second special token, `<|endoftext|>`, meaning "one text ends here, a new one starts". It gets the number 136.',
            ),
            q(
                "Why not just type `<|endoftext|>` into the text between books?",
                "Because our tokenizer cuts text into single characters. It would see `<`, `|`, `e`, `n`, `d`… 13 separate characters, not one signal. "
                "So we encode each book on its own and put the marker's **number** between them:\n\n"
                "```\nBook 1 → [ … ]   +   [136]   +   Book 2 → [ … ]\n```",
            ),
            step_intro(),
            q(
                "Which numbers do the special tokens get in a tiny vocabulary?",
                'Here\'s our real `CharTokenizer`, built from just the text `"abc"`.\n\n👉 **Predict first:** `a`, `b`, `c` take 0, 1, 2. What will `<|unk|>` get?',
            ),
            code(
                'from homer_ai.tokenizer import CharTokenizer\n\nt = CharTokenizer("abc")\nprint(t.char_to_id)\nprint("unk_id:", t.unk_id)'
            ),
            saw(
                "`<|unk|>` comes straight after the real letters, so it gets 3. And `<|endoftext|>` gets 4."
            ),
            q(
                "What happens to an unknown letter now?",
                '👉 **Predict first:** `z` isn\'t in `"abc"`. What will `"abz"` encode to?',
            ),
            code('ids = t.encode("abz")\nprint(ids)\nprint(t.decode(ids))'),
            saw("`[0, 1, 3]`, which decodes to `ab<|unk|>`. Nothing vanished."),
            q(
                "How many pieces would the marker break into if we typed it?",
                "Our tokenizer cuts text into single characters, so it would see the marker as a string of characters.\n\n👉 **Predict first:** how many?",
            ),
            code('len(list("<|endoftext|>"))'),
            saw("13. That's why we insert the marker's number and not its text."),
            q(
                "How do we put the marker between two books?",
                'Keep one flat list of numbers. For each book except the first, add the marker first. Then add the book\'s numbers.\n\n```\nbook 1: a b          book 2: b a\n   ↓                     ↓\n[ a b ]  +  [<|endoftext|>]  +  [ b a ]\n```\n\n👉 **Predict first:** with the books `"ab"` and `"ba"`, what numbers come out?',
            ),
            code(
                r"""books = ["ab", "ba"]
t2 = CharTokenizer("\n".join(books))

ids = []
for book in books:
    if ids:  # not the first book: mark where the previous one ended
        ids.append(t2.eot_id)
    ids.extend(t2.encode(book))

print(ids)
print(t2.decode(ids))"""
            ),
            saw(
                "`[1, 2, 4, 2, 1]`: book 1, the marker (4), book 2. The marker only goes **between** books, because nothing comes after the last one.\n\n"
                'But why is `a` number 1 and not 0? The vocabulary was built from `"\\n".join(books)`, which put a line break (`\\n`) between the books. '
                "The line break sorts first, so it took ID 0."
            ),
            md(
                "## 5. Full example\n\nThe real `load_data` (in `src/homer_ai/train_homer.py`) does all of this on the Odyssey.\n\n"
                "👉 **Predict first:** we only have Book 1 so far. How many `<|endoftext|>` markers will the data have?"
            ),
            code(
                r"""from homer_ai.train_homer import BOOKS_DIR, load_data

data, tokenizer, text = load_data(BOOKS_DIR)

print("vocabulary size:", tokenizer.vocab_size)
print("<|unk|> ID:", tokenizer.unk_id)
print("<|endoftext|> ID:", tokenizer.eot_id)
print("markers in the data:", (data == tokenizer.eot_id).sum().item())

ids = tokenizer.encode("Ζεὺς!")
print(ids, "->", tokenizer.decode(ids))"""
            ),
            saw(
                "137 tokens, with `<|unk|>` = 135, `<|endoftext|>` = 136, and **0** markers (there's only one book, so nothing to separate). And `Ζεὺς!` keeps a visible `<|unk|>`."
            ),
            *exercise(
                "how much of `ὦ Homer!` survives?",
                "Book 1 is Greek, so the tokenizer has never seen a Latin letter. Here's a short text that mixes both. Count how many of its characters come out as `<|unk|>`.\n\n"
                "👉 **Predict first:** the text has 8 characters. How many of them does Book 1 know?",
                starter="""
            mixed_ids = tokenizer.encode("ὦ Homer!")
            print(mixed_ids)

            # 👉 do: count how many of these IDs are the unknown token
            unknown_count = None
            """,
                solution="""
            mixed_ids = tokenizer.encode("ὦ Homer!")
            print(mixed_ids)

            unknown_count = mixed_ids.count(tokenizer.unk_id)
            """,
                check="""
            check(
                "unknown characters in 'ὦ Homer!'",
                unknown_count,
                sum(ch not in tokenizer.char_to_id for ch in "ὦ Homer!"),
                coach="Got 8? That's every character. Count only the IDs that are equal to "
                "tokenizer.unk_id (135): a list has a count() method for that.",
            )
            """,
                hint="The unknown token's number is `tokenizer.unk_id`. A list can count how often a value is in it: `[1, 5, 5].count(5)` gives 2.",
                answer="6 of the 8: `H`, `o`, `m`, `e`, `r` and `!`. Only `ὦ` and the space are known. "
                "That Latin `o` looks just like the Greek `ο`, but it's a different character, so it's unknown too. "
                "Decode the IDs and you get `ὦ <|unk|><|unk|><|unk|><|unk|><|unk|><|unk|>`: we can see that six characters were there, but not which ones.",
            ),
            *questions(
                [
                    (
                        "Homer never uses `<|unk|>`. So what does the model learn about it?",
                        "When reading, nothing. It never sees `<|unk|>` in training, so the numbers it has for it stay random. "
                        "When guessing, it learns to **never** guess `<|unk|>`, because the right answer is always a real letter. Which is what we want.",
                    ),
                    (
                        "If I add a token to the vocabulary, can I keep my trained model?",
                        "No, because the model's size depends on the vocabulary size. Imagine a model trained on just the 135 characters, with no special tokens. Its tables have 135 rows. "
                        "Add `<|unk|>` and `<|endoftext|>` and the vocabulary has 137, so the saved weights don't fit any more, and the model has to be trained again from scratch (about 8 minutes for our small one).",
                    ),
                    (
                        "Could I just add `!` to the vocabulary instead?",
                        "Only if the text contains it. The vocabulary is built from the text, and Homer has no `!`, so it never gets a number. "
                        "`<|unk|>` is for exactly these cases: anything the text never showed us.",
                    ),
                    (
                        "Why put the marker between books and not after the last one?",
                        "It separates two books, so it only makes sense where a new book follows.",
                    ),
                ]
            ),
            recap(
                [
                    "A tokenizer that skips unknown characters loses them silently, like the `!` in `Ζεὺς!`.",
                    "`<|unk|>` (135) stands in for any unknown character, so nothing vanishes.",
                    "`<|endoftext|>` (136) goes between books as a number, because a line break can't mark a book's end.",
                ],
                "[02_03: BPE](02_03_bpe.ipynb): can tokens be bigger than one letter?",
                can=[
                    "say what happens to a character the vocabulary doesn't have",
                    "count the `<|unk|>` tokens in an encoded text",
                    "join several books into one list of IDs, with `<|endoftext|>` between them",
                ],
            ),
        ],
    )

    # =====================================================================
    # 02_text_to_numbers/02_03_bpe.ipynb
    # =====================================================================
    save(
        "02_text_to_numbers/02_03_bpe.ipynb",
        [
            title("02_03 Byte pair encoding (BPE)", "section 2.5"),
            where("numbers", ["src/homer_ai/tokenizer.py"]),
            learn(
                [
                    "Why one-letter tokens are a problem, and why whole words are too",
                    "What BPE is, and how it builds tokens out of common letter groups",
                    "How it chooses those groups automatically, by counting",
                    "How a small BPE works from start to finish on Book 1",
                ]
            ),
            words(
                [
                    (
                        "token",
                        "one piece of text the model reads as a single unit",
                        "`π`, or `λλα`",
                    ),
                    (
                        "vocabulary",
                        "the list of all the tokens the tokenizer knows",
                        "`π ο λ α λα λλα`",
                    ),
                    (
                        "window",
                        "how many tokens the model can look at at one time",
                        "our model: 24 tokens",
                    ),
                    (
                        "subword",
                        "a piece of a word: bigger than a letter, smaller than a word",
                        "`πολλ` + `ῶν`",
                    ),
                    (
                        "pair",
                        "two tokens that sit next to each other in the text",
                        "in `πολ`: `πο` and `ολ`",
                    ),
                    ("merge", "joining a pair into one new token", "`λ` + `α` → `λα`"),
                    (
                        "BPE",
                        "byte pair encoding: building a vocabulary by merging the most common pair, again and again",
                        "",
                    ),
                ]
            ),
            *background(
                "At the stage **numbers**. Our tokenizer works: it cuts the text into **single letters** and gives each one a number "
                "([02_01](02_01_tokens_and_vocabulary.ipynb)), with special tokens for unknown letters and book ends "
                "([02_02](02_02_special_tokens.ipynb)). So `πολλὰ` becomes 5 tokens: `π ο λ λ ὰ`.",
                "Single letters work, but they make the model's job harder than it needs to be (the next two questions show why). "
                "This notebook asks: can tokens be **bigger** than one letter? BPE is the standard answer, and it's how "
                "real language models cut up their text.",
            ),
            q(
                "What's wrong with one letter per token?",
                "It makes the text very long. The model can only look at a fixed number of tokens at once. That's its **window**, and ours is 24 tokens. (In the code the number is called `chunk_size`, and each 24-token piece of text is a chunk.) "
                "With one token per letter, 24 tokens is only about 4 words of Homer. "
                "It's like reading a book through a tiny hole in a card.",
            ),
            q(
                "Is there a second problem?",
                'Yes: single letters carry almost no meaning. `λ` on its own tells the model very little, while `πολλ` ("many") tells it a lot. '
                "With single letters, the model has to work out from scratch, every single time, that `π` + `ο` + `λ` + `λ` belong together.",
            ),
            q(
                "Then why not use whole words as tokens?",
                "That would fix both problems: fewer tokens, and each one means something. But Greek changes the ending of a word depending on how it's used. "
                '"Many" turns up in Book 1 as `πολλά`, `πολλὰ`, `πολλοὶ`, `πολλὸν`, `πολλῶν` and more, and each form would be a separate token.\n\n'
                "We counted Book 1:\n\n| | Count |\n|---|---|\n| words in Book 1 | 3,175 |\n| **different** words | 1,697 |\n| words that appear **only once** | 1,262 |\n\n"
                "So most words appear only once, and the model can't learn much from a word it sees one time. "
                "Worse, a word that isn't in the list can't be written at all.",
            ),
            q(
                "So what's in between letters and words?",
                "Pieces of words, called **subwords**:\n\n"
                "```\nletters:   π | ο | λ | λ | ῶ | ν      6 tokens, little meaning\n"
                "words:     πολλῶν                     1 token, but only if this exact form is in the list\n"
                'pieces:    πολλ | ῶν                  2 tokens: "many" + a common ending\n```\n\n'
                '`πολλ` is shared by every form of "many", and the ending `ῶν` is shared by lots of other words. '
                "So the model learns each piece once and reuses it everywhere.",
            ),
            q(
                "So what is BPE?",
                "A method that finds good pieces automatically. Picking them by hand would take forever, and it would be guesswork anyway.\n\n"
                "About the name: **byte pair encoding** was invented in 1994 to make computer files smaller. A byte is the small basic unit computers store data in. "
                "The method found the pair of bytes that appeared most often and replaced it with one new symbol. That's the encoding part. "
                "Language models use the same trick on text.",
            ),
            q(
                "How does it choose the pieces? Isn't that random?",
                "No, there's nothing random about it. A program counts, and the counts decide.\n\n"
                "1. Start with a vocabulary of single letters.\n"
                "2. Count every **pair** of neighbouring tokens in the whole text.\n"
                "3. **Merge** the pair that appears most often into one new token, and add it to the vocabulary.\n"
                "4. Go back to step 2. Stop after a number of merges you choose in advance.\n\n"
                "In the Odyssey, the very first merge is `ν` followed by a space, because so many Greek words end in -ν.",
            ),
            q(
                "Can BPE still write a word it never merged, like a rare name?",
                "Yes. The single letters are never removed from the vocabulary. Merges only add new tokens. "
                "It's like LEGO: you build bigger blocks, but all the small bricks stay in the box. "
                "So any word can still be written, in the worst case one letter at a time.",
            ),
            q(
                "Why does this matter for our Odyssey model?",
                "With BPE, the model could:\n\n- see more of the poem at once in the same 24-token window\n"
                "- learn pieces that mean something, like `πολλ` and common endings, and not rebuild them from letters every time\n"
                "- still write any Greek word, even one it has never seen",
            ),
            step_intro(),
            q(
                "What does a BPE vocabulary look like after a couple of merges?",
                "Here's a tiny made-up one, after two merges (`λα`, then `λλα`).\n\n👉 **Predict first:** are the single letters still in it?",
            ),
            code('vocab = ["π", "ο", "λ", "α", "λα", "λλα"]\nvocab'),
            saw(
                "The single letters are all still there. The merges only added new tokens at the end."
            ),
            q(
                "How is a word cut into tokens with that vocabulary?",
                "Use the **biggest block** that fits. If none fits, fall back to single letters. "
                "You don't need to follow every line of this code. Just run it.\n\n👉 **Predict first:** how will `πολα` be cut?",
            ),
            code(
                """def split_with_blocks(word, vocab):
    tokens = []
    i = 0
    while i < len(word):
        # try the longest block first, then shorter ones
        for size in range(len(word) - i, 0, -1):
            piece = word[i : i + size]
            if piece in vocab:
                tokens.append(piece)
                i += size
                break
    return tokens


split_with_blocks("πολα", vocab)"""
            ),
            saw(
                "`['π', 'ο', 'λα']`. There's no block for `πο`, so those two go in as single letters. `λα` is a block, so it's one token."
            ),
            q(
                "Your turn: how will `αλλα` be cut?",
                "👉 **Predict first**, and write your guess here: ...",
            ),
            code('split_with_blocks("αλλα", vocab)'),
            saw("`['α', 'λλα']`: one single letter, then the biggest block."),
            q(
                "How do I find all the neighbouring pairs?",
                '`pairwise` walks along a word and gives you every pair of neighbours: the 1st and 2nd letter, then the 2nd and 3rd, and so on.\n\n👉 **Predict first:** what pairs will `"πολλα"` give?',
            ),
            code(
                'from itertools import pairwise\n\nword = "πολλα"\nlist(pairwise(word))'
            ),
            saw(
                "That's 4 pairs from 5 letters. Each letter is paired with the one after it."
            ),
            q(
                "How do I count which pair appears most?",
                'A `Counter` counts how many times each thing appears, and `.most_common(2)` gives the 2 most frequent.\n\nWe use the tiny text `"λλα λλα λα"`.\n\n'
                "👉 **Predict first:** which pair appears most often, and how many times?",
            ),
            code(
                'from collections import Counter\n\nsmall_text = "λλα λλα λα"\npairs = Counter(pairwise(small_text))\npairs.most_common(2)'
            ),
            saw("`λ` + `α` appears 3 times, so it would be the **first merge**."),
            q(
                "Which pairs are most common in the real Odyssey?",
                "`load_books` is our own function from `train_homer.py`. It gives one text per book, and we take Book 1.\n\n"
                "👉 **Predict first:** which pair of characters do you think is the most common in Homer?",
            ),
            code(
                "from homer_ai.train_homer import BOOKS_DIR, load_books\n\ntext = load_books(BOOKS_DIR)[0]\npairs = Counter(pairwise(text))\npairs.most_common(5)"
            ),
            saw(
                "Each line is `((first character, second character), how many times)`, and `' '` is a space. "
                "`ν` followed by a space wins with 529, because so many Greek words end in **-ν**. "
                "Nobody picked it. The counting did."
            ),
            md(
                "## 5. Full example: a small BPE on Book 1\n\nThis puts all the steps together and learns **20 merges** from Book 1. Two things here are new:\n\n"
                "- `merge_pair` goes through the token list and joins every place where the chosen pair appears.\n"
                "- To cut new text into tokens, real BPE replays the merges in the same order it learned them.\n\n"
                "You don't need to follow every line yet. Run it, and look at which pieces it learns.\n\n"
                "👉 **Predict first:** after 20 merges, will Book 1 need many fewer tokens, or only a few fewer?"
            ),
            code(
                '''def merge_pair(tokens, pair):
    """Join every place where `pair` appears next to each other into one token."""
    merged = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
            merged.append(tokens[i] + tokens[i + 1])
            i += 2  # skip both parts of the pair
        else:
            merged.append(tokens[i])
            i += 1
    return merged


# 1. Start: every character is a token
tokens = list(text)
merges = []
sizes = [len(tokens)]  # how many tokens are left, after 0, 1, 2 ... merges

# 2-4. Count pairs, merge the most common one, repeat
for step in range(20):
    best_pair = Counter(pairwise(tokens)).most_common(1)[0][0]
    tokens = merge_pair(tokens, best_pair)
    merges.append(best_pair)
    sizes.append(len(tokens))
    left, right = best_pair
    print(f"merge {step + 1:2}: {left!r} + {right!r} -> {left + right!r}")

print()
print(f"Book 1: {len(text)} characters -> {len(tokens)} BPE tokens")'''
            ),
            saw(
                "The first merges join common word endings with a space (`ν `, `ς `). After 20 merges, Book 1 is down from 19,701 characters to 16,288 tokens."
            ),
            q(
                "Can I see how the token count fell?",
                "We wrote down the token count after every merge (`sizes`). If we draw it, we can see what each merge was worth.\n\n"
                "👉 **Predict first:** will every merge save about the same, or will the first merges save the most?",
            ),
            code(
                """import matplotlib.pyplot as plt

from homer_ai.studio_style import ACCENT, GOOD, MUTED, studio_axes, studio_figure

saved = sizes[0] - sizes[-1]
fig = studio_figure(
    "Tokens left after each merge",
    f"20 merges saved {saved:,} tokens ({saved / sizes[0]:.0%})",
)
ax = studio_axes(fig)
ax.plot(range(len(sizes)), sizes, color=ACCENT, linewidth=3, marker="o", markersize=6)
ax.set_ylim(sizes[-1] - 300, sizes[0] + 300)
ax.text(0.3, sizes[0] - 120, f"{sizes[0]:,}", color=MUTED, family="monospace")
ax.text(
    len(sizes) - 1, sizes[-1] - 170, f"{sizes[-1]:,}",
    color=GOOD, family="monospace", ha="right",
)
ax.set_xticks(range(0, 21, 2))
ax.set_xlabel("merges done")
ax.set_ylabel("tokens in Book 1")
plt.show()"""
            ),
            saw(
                "The line falls from 19,701 to 16,288, steeply at first and then more gently. "
                "The first merges join the most common pairs, so each one removes the most tokens. Later pairs are rarer, so each one saves less."
            ),
            md(
                "Now let's use the learned merges to cut the first line of the Odyssey into tokens."
            ),
            code(
                '''def bpe_encode(line, merges):
    """Cut new text into tokens: replay the merges in the order they were learned."""
    tokens = list(line)
    for pair in merges:
        tokens = merge_pair(tokens, pair)
    return tokens


first_line = text.splitlines()[0]
bpe_tokens = bpe_encode(first_line, merges)
print(first_line)
print(bpe_tokens)
print(f"{len(first_line)} characters -> {len(bpe_tokens)} BPE tokens")'''
            ),
            saw(
                "The first line goes from 50 characters to 42 tokens. With only 20 merges the pieces are still small. Real models do tens of thousands."
            ),
            *questions(
                [
                    (
                        "Why does the first merge contain a space?",
                        'The space is a character like any other. `ν` + space means "a word ended in ν", and so many Greek words end in -ν that it\'s the most common pair of all.',
                    ),
                    (
                        "How many merges should I choose?",
                        "Each merge adds one token to the vocabulary, so the model gets a little bigger each time. And too many merges on a small text just turns whole words into tokens. "
                        "In our full example, 20 merges took Book 1 from 19,701 characters to 16,288 tokens. Real models do tens of thousands, on far more text.",
                    ),
                    (
                        "Does our model use BPE?",
                        "No. Our tokenizer works with single letters. This notebook shows how BPE works, and switching our model over to it would be a separate step.",
                    ),
                ]
            ),
            recap(
                [
                    "Single-character tokens are long and nearly meaningless; whole words don't suit Greek's many word forms.",
                    "BPE finds the pieces in between automatically: count pairs, merge the most common, repeat.",
                    "Single characters stay in the vocabulary, so any text can still be written.",
                ],
                "[02_04: the sliding window](02_04_sliding_window.ipynb): cutting the text into practice pieces for the model.",
                can=[
                    "do one BPE merge by hand: count the pairs, merge the most common one",
                    "say why GPT-style models use pieces between a character and a word",
                ],
            ),
        ],
    )

    # =====================================================================
    # 02_text_to_numbers/02_04_sliding_window.ipynb  (built live with the learner)
    # =====================================================================
    save(
        "02_text_to_numbers/02_04_sliding_window.ipynb",
        [
            title("02_04 The sliding window", "section 2.6"),
            where("numbers", ["src/homer_ai/train_homer.py"]),
            learn(
                [
                    "What the model practises on: flashcards made from the text itself",
                    "Why nobody has to write the answers by hand",
                    "How one chunk of text becomes many flashcards at once",
                ]
            ),
            words(
                [
                    (
                        "input",
                        "the text the model is shown: the question on the flashcard",
                        "`ἄν`",
                    ),
                    (
                        "target",
                        "the character that really comes next: the answer on the flashcard",
                        "`δ`",
                    ),
                    (
                        "self-supervised learning",
                        "learning from examples whose answers come from the data itself, so no person has to label them",
                        "the next character is already in the text",
                    ),
                    (
                        "chunk",
                        "a short piece of the text, cut out to practise on; `chunk_size` is its length",
                        "24 characters",
                    ),
                ]
            ),
            *background(),
            q(
                "What does the model actually practise on?",
                "Practice questions with answers, like **flashcards**. For a language model every question is the same kind: "
                '"here is some text, what comes next?"',
            ),
            q(
                "Who writes all those flashcards?",
                "Nobody. We make them from the Odyssey itself. Take the first word, `ἄνδρα`:\n\n"
                "```\nquestion (input)  →  answer (target)\nἄ                 →  ν\nἄν                →  δ\nἄνδ               →  ρ\nἄνδρ              →  α\n```\n\n"
                "The answer is always just the next character, and the text already has it. "
                "That's why it's called **self-supervised** learning.",
            ),
            q(
                "How many flashcards does one word give?",
                "`ἄνδρα` has 5 characters and gives **4** flashcards. The last character has nothing after it, "
                "so it can only ever be an answer, never a question.",
            ),
            step_intro(),
            md(
                "### ❓ How do I make the flashcards in code?\n\n"
                "With a loop that cuts the word at every position. The first `n` characters are the question, and the character right after them is the answer.\n\n"
                "👉 **Predict first:** how many lines will it print?"
            ),
            code(
                """word = "ἄνδρα"
for n in range(1, len(word)):
    print(word[:n], "→", word[n])"""
            ),
            saw("So, 4 lines, one flashcard each, ending with `ἄνδρ → α`."),
            md(
                "### ❓ How does the training code make them all at once?\n\n"
                "It doesn't loop over flashcards at all. It takes a **chunk** of the text, plus a second copy shifted one place to the right "
                "(`src/homer_ai/train_homer.py`, in `train()`):\n\n"
                "```python\ninputs  = data[i     : i + chunk_size]\ntargets = data[i + 1 : i + chunk_size + 1]\n```\n\n"
                "Read the two copies top to bottom, one column at a time. Each column is one flashcard.\n\n"
                "👉 **Predict first:** with a chunk of 4 from the start of Book 1, what are the 4 columns?"
            ),
            code(
                """from homer_ai.tokenizer import CharTokenizer
from homer_ai.train_homer import BOOKS_DIR, load_books

text = load_books(BOOKS_DIR)[0]
tokenizer = CharTokenizer(text)
data = tokenizer.encode(text)

chunk_size = 4
inputs = data[0:chunk_size]
targets = data[1 : chunk_size + 1]

print("inputs: ", tokenizer.decode(inputs))
print("targets:", tokenizer.decode(targets))
for k in range(chunk_size):
    question = tokenizer.decode(inputs[: k + 1])
    answer = tokenizer.decode([targets[k]])
    print(f"{question:>5} → {answer}")"""
            ),
            saw(
                "`inputs` is `ἄνδρ` and `targets` is `νδρα`: the same text, moved one place. "
                "Column by column, they give the same 4 flashcards as the loop did."
            ),
            md(
                "Here are the same two rows as aligned columns, one letter per column, made from the real slices "
                "(`chunk_size = 4`, `i = 0`):"
            ),
            code(
                """i = 0
inputs = data[i : i + chunk_size]
targets = data[i + 1 : i + chunk_size + 1]

print("inputs :", "  ".join(tokenizer.decode([t]) for t in inputs))
print("targets:", "  ".join(tokenizer.decode([t]) for t in targets))"""
            ),
            md(
                "Read each column top to bottom. Each one is a flashcard, and that shift by one letter is the whole idea."
            ),
            md(
                "**Look at:** the top row is the text. The orange chunk is the input and the green one is the target, one place later. Below it, each flashcard is a growing orange window with its green answer."
            ),
            code(DIAGRAM_WINDOW),
            *exercise(
                "which flashcards start at `i = 6`?",
                "So far the window sat at the start of the text, `i = 0`. Training slides it along. Move it to `i = 6`, still with `chunk_size = 4`, and cut the inputs and the targets yourself.\n\n"
                "Book 1 starts `ἄνδρα μοι ἔννεπε`. Counting from 0, character 6 is the `μ`.\n\n"
                "👉 **Predict first:** write down the 4 input characters and the 4 target characters. Careful, one of them is a space.",
                starter="""
            start = 6

            # 👉 do: cut the 4 inputs that start at `start`, and the 4 targets one place later
            # 💡 hint: the `i = 0` cell did the same; data is the whole text as IDs
            window_inputs = None
            window_targets = None
            """,
                solution="""
            start = 6

            window_inputs = data[start : start + chunk_size]
            window_targets = data[start + 1 : start + chunk_size + 1]
            """,
                check="""
            check(
                "inputs at i = 6",
                window_inputs,
                tokenizer.encode("μοι "),
                show=tokenizer.decode,
                coach="The inputs are 4 IDs: from `start` up to (not including) "
                "`start + chunk_size`.",
            )
            check(
                "targets at i = 6",
                window_targets,
                tokenizer.encode("οι ἔ"),
                show=tokenizer.decode,
                coach="The targets are the inputs moved one place to the right, so BOTH ends "
                "of the slice get + 1. With only the start moved you get 3 characters, 'οι '.",
            )
            """,
                hint="Take the two slices of the `i = 0` cell and write `start` where it has `i`: `data[start : start + chunk_size]`, and the same with `+ 1` on both ends for the targets.",
                answer="```\ninputs : μ  ο  ι  ␣\ntargets: ο  ι  ␣  ἔ\n```\n\n"
                "(`␣` stands for the space.) Read the columns: after `μ` comes `ο`, after `μο` comes `ι`, after `μοι` a space, and after `μοι ` the `ἔ` of the next word. "
                "So the model also has to learn where words end.",
            ),
            md(
                "### ❓ Our `chunk_size` is 24. How many flashcards is that?\n\n"
                "👉 **Predict first**, then run."
            ),
            code(
                """chunk_size = 24
inputs = data[0:chunk_size]
targets = data[1 : chunk_size + 1]
print(len(inputs), "inputs,", len(targets), "targets")
print(repr(tokenizer.decode(inputs)))
print(repr(tokenizer.decode(targets)))"""
            ),
            saw(
                "**24 flashcards** from one chunk. Each target is the input moved one place to the right. `repr()` shows the text inside quote marks, so you can see that the last target is a space (the character after `μοῦσα,`)."
            ),
            md(
                "### ❓ Should the stop be `len(data) - chunk_size`, or one less?\n\n"
                "Training walks along the text in steps of `chunk_size`: `for i in range(0, len(data) - chunk_size, chunk_size)`. "
                "The targets reach one letter further than the inputs, so it's tempting to think the stop needs an extra `- 1`. Let's check on a text small enough to count. "
                "Take 9 letters, `ἄνδρα μοι`, with `chunk_size = 4` (`␣` is the space):\n\n"
                "```\n"
                "position:  0  1  2  3  4  5  6  7  8\n"
                "letter:    ἄ  ν  δ  ρ  α  ␣  μ  ο  ι\n"
                "i = 0:     inputs 0-3 (ἄ ν δ ρ)   targets 1-4 (ν δ ρ α)\n"
                "i = 4:     inputs 4-7 (α ␣ μ ο)   targets 5-8 (␣ μ ο ι)   <- fits: the last target is position 8\n"
                "```\n\n"
                "`range` leaves out its stop value. With the extra `- 1`, the stop `len(data) - chunk_size - 1` would give `i = 0` only, and `␣ μ ο ι` would be wasted. "
                "With `len(data) - chunk_size` it gives `i = 0` and `i = 4`.\n\n"
                "👉 **Predict first:** how many values of `i` does each stop give for 9 letters?"
            ),
            code(
                """letters = "ἄνδρα μοι"
chunk_size = 4
print("length:", len(letters))
print("with - 1   :", list(range(0, len(letters) - chunk_size - 1, chunk_size)))
print("without - 1:", list(range(0, len(letters) - chunk_size, chunk_size)))
for i in range(0, len(letters) - chunk_size, chunk_size):
    print(i, letters[i : i + chunk_size], "→", letters[i + 1 : i + chunk_size + 1])"""
            ),
            saw(
                "With `- 1` the stop gives `[0]`; without it, `[0, 4]`. And the second chunk (`α μο` → ` μοι`) fits exactly: its targets end at position 8, the last letter. "
                "On Book 1 (19,701 characters) both stops happen to give the same 820 chunks, so the trained model comes out the same either way."
            ),
            recap(
                [
                    "The model practises on flashcards: some text, and the character that comes next.",
                    "The text itself provides the answers, so this is self-supervised learning.",
                    "One chunk of 24 characters, plus a copy shifted by one, gives 24 flashcards at once.",
                ],
                "[02_05: Token embeddings](02_05_token_embeddings.ipynb): turning each token number into a list of numbers that can carry meaning.",
                can=[
                    "cut the inputs and the targets for any start `i`",
                    "read two aligned rows as flashcards, column by column",
                    "say why the loop stops at `len(data) - chunk_size`",
                ],
            ),
        ],
    )

    # =====================================================================
    # 02_text_to_numbers/02_05_token_embeddings.ipynb
    # =====================================================================
    save(
        "02_text_to_numbers/02_05_token_embeddings.ipynb",
        [
            title("02_05 Token embeddings", "section 2.7"),
            where("numbers", ["src/homer_ai/model.py", "src/homer_ai/tokenizer.py"]),
            learn(
                [
                    "Why a token ID is only a label, not something to compute with",
                    "What an embedding is: a list of numbers for each token, learned in training",
                    "What the lookup table looks like, and why its shape is 137 × 32",
                    "What our trained model's embeddings really show (and what they don't)",
                ]
            ),
            words(
                [
                    (
                        "vector",
                        "a list of numbers, used as a point or an arrow",
                        "`[9, 1]`",
                    ),
                    (
                        "embedding",
                        "a vector that stands for one token; the model learns its numbers during training",
                        "32 numbers for `α`",
                    ),
                    (
                        "embedding table",
                        "the lookup table of all embeddings: one row per token",
                        "137 rows",
                    ),
                    (
                        "embed_dim",
                        "how many numbers each embedding has: the table's number of columns",
                        "32",
                    ),
                    (
                        "cosine similarity",
                        "how much two vectors point the same way: `1` same, `0` unrelated, `-1` opposite (see [A_03](../A_toolbox/A_03_dot_product.ipynb))",
                        "`[9, 1]` and `[8, 1]` → `1.0`",
                    ),
                ]
            ),
            *background(),
            q(
                "A token ID like 41 is just a label. Why can't the model compute with it?",
                "Because the number was handed out for a boring reason: the tokenizer sorted the characters, and `π` landed in place 41. "
                'Nothing about `π` is "41-ish".\n\n'
                "```\nID:    38   39   40   41   42   43   44\nchar:   ν    ξ    ο    π    ρ    ς    σ\n```\n\n"
                '`ο` (40) isn\'t "smaller" than `π` (41) in any way that helps, and `ν` (38) isn\'t "closer" to `ο` than `ξ` is. '
                "If the model did maths on 41, it would find patterns in alphabet order, not in Greek.\n\n"
                "👉 **Predict first:** which characters have the IDs 38 to 44?",
            ),
            code(
                """from homer_ai.tokenizer import CharTokenizer
from homer_ai.train_homer import BOOKS_DIR, load_books

tokenizer = CharTokenizer(load_books(BOOKS_DIR)[0])
for i in range(38, 45):
    print(i, tokenizer.id_to_char[i])"""
            ),
            saw(
                "The IDs are just positions in the sorted list. We need something richer than one number."
            ),
            q(
                "What could we use instead?",
                "We describe each character with several numbers, one for each thing we care about. It's like rating films by `[Action, Romance]` in [A_03](../A_toolbox/A_03_dot_product.ipynb). "
                "A list of numbers like this is called a **vector**.\n\n"
                "**If the dot product is new to you, read [A_03](../A_toolbox/A_03_dot_product.ipynb) now** (about 10 minutes). The rest of this notebook measures how alike two vectors are, and all of chapter 3 is built on it.\n\n"
                "Let's try it by hand for four letters, with two features I made up: `[vowel, consonant]`, each from 0 to 10.\n\n"
                "| letter | vector | meaning |\n|---|---|---|\n"
                "| `α` | `[9, 1]` | very vowel-like |\n"
                "| `ά` | `[8, 1]` | `α` with an accent: still a vowel |\n"
                "| `ν` | `[1, 9]` | very consonant-like |\n"
                "| `ρ` | `[1, 8]` | a consonant too |\n\n"
                "Now similar letters have similar vectors, and we can measure that with the **cosine similarity** from A_03: "
                "`1` means the same direction, near `0` means unrelated.\n\n"
                "👉 **Predict first:** will `α` and `ά` score high or low? And `α` and `ν`?",
            ),
            code(
                """from math import sqrt


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (sqrt(sum(x * x for x in a)) * sqrt(sum(y * y for y in b)))


by_hand = {"α": [9, 1], "ά": [8, 1], "ν": [1, 9], "ρ": [1, 8]}
for first, second in [("α", "ά"), ("ν", "ρ"), ("α", "ν"), ("ά", "ρ")]:
    score = cosine(by_hand[first], by_hand[second])
    print(f"{first} {second}  {score:.2f}")"""
            ),
            saw(
                "`α`/`ά` and `ν`/`ρ` score `1.00`, so they point the same way. A vowel against a consonant scores about `0.2`, which is far apart. "
                "Similar tokens, similar vectors."
            ),
            q(
                "So what is an embedding?",
                "The same idea, with one big change: **we don't choose the numbers, the model learns them.** "
                "An embedding is a vector for each token. At the start of training its numbers are random. "
                "Then, while the model practises on the flashcards from [02_04](02_04_sliding_window.ipynb), it nudges every embedding, "
                "so that tokens which behave alike (and so help predict the same next letters) drift towards similar vectors.\n\n"
                'Nobody tells it "vowel" or "consonant". Our hand-made features had 2 numbers. The model uses 32, '
                "and no single one of them has a name. That count (32) is called `embed_dim`.",
            ),
            q(
                "Where are all these vectors kept, and what is its shape?",
                "In a lookup table called the **embedding table**, with one row for each token. "
                "Have a guess at its shape:\n\n"
                "```\nrows    = how many different tokens we have   (the vocabulary)\ncolumns = how many numbers each token gets   (embed_dim)\n```\n\n"
                "👉 **Predict first:** with 137 tokens and `embed_dim=32`, what is the table's shape, and how many weights is that?",
            ),
            code(
                """import torch
from torch import nn

table = nn.Embedding(tokenizer.vocab_size, 32)
print("shape:  ", tuple(table.weight.shape))
print("weights:", table.weight.numel(), "=", tokenizer.vocab_size, "×", 32)"""
            ),
            saw(
                "`(137, 32)`, which is 4,384 weights. Back in 01_02 that was one of the two biggest parts of our 11,840 weights. "
                "[A_04](../A_toolbox/A_04_pytorch_in_this_repo.ipynb) goes through `nn.Embedding` line by line. Here we only need the idea."
            ),
            q(
                'What does "look up" really do?',
                "It picks rows, and that's all. Here's a tiny table: 5 tokens, 3 numbers each (`seed` fixes the random start, so you get the same numbers as this page).\n\n"
                "👉 **Predict first:** which rows of the table will come back for the IDs `[1, 3]`?",
            ),
            code(
                """torch.manual_seed(0)
tiny = nn.Embedding(5, 3)
print(tiny.weight)
print(tiny(torch.tensor([1, 3])))"""
            ),
            saw(
                "The two rows it printed are exactly row 1 and row 3 of the table. Looking up an ID just means taking that row."
            ),
            q(
                "What does that look like for a real word?",
                'Take `ἄνδρα` ("man", the first word of the *Odyssey*). Five characters, five IDs, five rows.\n\n'
                "The text is `ἄ ν δ ρ α`, the IDs are `58 38 29 42 26`, and each ID picks one row of 32 numbers. So the result is a 5 × 32 table.\n\n"
                "The model never sees `ἄνδρα`. It sees this 5 × 32 table of numbers.\n\n"
                "👉 **Predict first:** what shape will the result have?",
            ),
            code(
                """ids = torch.tensor(tokenizer.encode("ἄνδρα"))
print("IDs:  ", ids.tolist())
print("shape:", tuple(table(ids).shape))"""
            ),
            saw(
                "`[58, 38, 29, 42, 26]` becomes shape `(5, 32)`: one row per character. "
                "(This table is untrained, so its numbers are random. Next we look at the trained one.)"
            ),
            md(
                "**Look at:** the arrows. Each ID on the left picks one row of the table and puts it in the stack on the right, in the order of the word. Row 58, for `ἄ`, is orange."
            ),
            code(DIAGRAM_LOOKUP),
            *exercise(
                "can you look up `μοι`?",
                "Same lookup, another word: `μοι` (to me), the second word of the *Odyssey*. Get its rows from `table`.\n\n"
                "👉 **Predict first:** what shape will come back?",
                starter="""
            # 👉 do: encode "μοι", make the IDs a tensor, and look them up in `table`
            moi_rows = None
            """,
                solution="""
            moi_rows = table(torch.tensor(tokenizer.encode("μοι")))
            """,
                check="""
            check(
                "the shape of the rows of μοι",
                moi_rows,
                table(torch.tensor(tokenizer.encode("μοι"))),
                show=lambda rows: tuple(rows.shape),
                coach="Shape (3,) means you still have the 3 IDs: pass them through table(...). "
                "A list instead of a shape means the IDs aren't a tensor yet: wrap them in "
                "torch.tensor(...). And tiny only has 5 rows, so use table.",
            )
            """,
                hint="It's the `ἄνδρα` cell with another word: `ids = torch.tensor(tokenizer.encode(...))`, then `table(ids)`.",
                answer="Shape `(3, 32)`: three characters, so three rows of 32 numbers. A longer text just means more rows. The 32 never changes.",
            ),
            step_intro(),
            q(
                "Did our trained model really give similar letters similar rows?",
                "Let's measure it and not guess. We load the saved model (`homer_model.pt`; we only look, we don't change it) "
                "and take the cosine similarity between rows. Some pairs to try:\n\n"
                "- `α` and its accented forms `ά`, `ὰ`, `ᾶ` (the same letter, with different accent marks)\n"
                "- `α` and the consonants `ν`, `ρ`\n"
                "- two vowels, `ε` and `ο`\n\n"
                "👉 **Predict first:** which pair do you think scores highest?",
            ),
            code(
                """import torch.nn.functional as F

from homer_ai.lessons import load_homer_model

trained_tokenizer, trained_model = load_homer_model()  # read-only
have_model = trained_model is not None
if have_model:
    learned = trained_model.embedding.weight.detach()  # the learned table
    print("trained table shape:", tuple(learned.shape))

    def similarity(a, b):
        row_a = learned[trained_tokenizer.encode(a)[0]]
        row_b = learned[trained_tokenizer.encode(b)[0]]
        return F.cosine_similarity(row_a, row_b, dim=0).item()

    pairs = [("α", "ά"), ("α", "ᾶ"), ("α", "ὰ"), ("α", "ν"), ("α", "ρ"), ("ε", "ο")]
    for first, second in [*pairs, ("α", "ε")]:
        print(f"{first} {second}  {similarity(first, second):+.2f}")
else:
    print("no trained model yet: make one with `uv run homer-train`")"""
            ),
            saw(
                "Small numbers, mostly. `α` with `ά` and `ᾶ` is about `+0.25`, and with `ὰ` only `+0.05`. "
                "`α` with the consonants `ν` and `ρ` is about `-0.05`. And `α` with the vowel `ε` is the highest, at `+0.40`. "
                "So there's a hint of \"vowels together, consonants apart\", but it's weak. A picture with more pairs comes next."
            ),
            q(
                "Can I see all the pairs at once?",
                "Yes, with a heatmap. Each square is the cosine similarity of two characters. Green means the rows point the same way, "
                "pink-red means the opposite way, and dark means unrelated. The diagonal is each character against itself, so it's always `1`.\n\n"
                "👉 **Predict first:** will the block of vowels (`α ά ὰ ᾶ ε ο ι`) look clearly greener than the vowel-and-consonant squares?",
            ),
            code(
                """import matplotlib.pyplot as plt

from homer_ai.studio_style import GOOD_BAD, INK, MUTED, studio_figure, text_on

if have_model:
    chars = list("αάὰᾶεοινρς")
    rows = learned[[trained_tokenizer.encode(ch)[0] for ch in chars]]
    unit = F.normalize(rows, dim=1)  # each row scaled to length 1
    sims = (unit @ unit.T).detach().numpy()  # all the cosine similarities at once

    fig = studio_figure("Trained embeddings", "How alike are two characters' rows?")
    ax = fig.add_axes((0.13, 0.08, 0.72, 0.68))
    ax.set_facecolor("none")
    ax.imshow(sims, cmap=GOOD_BAD, vmin=-1, vmax=1)
    ax.set_xticks(range(len(chars)), chars, fontsize=14, color=INK)
    ax.set_yticks(range(len(chars)), chars, fontsize=14, color=INK)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    for r in range(len(chars)):
        for c in range(len(chars)):
            value = sims[r, c]
            ax.text(c, r, f"{value:.2f}", ha="center", va="center", fontsize=8,
                    color=text_on(value, 1))
    note = "green = alike, pink-red = opposite, dark = unrelated"
    fig.text(0.13, 0.03, note, color=MUTED, fontsize=9)
    plt.show()
else:
    print("no trained model yet: make one with `uv run homer-train`")"""
            ),
            saw(
                "A mix. The vowels aren't one clean green block, and a pair of consonants can be just as green as the vowel squares (`ν`/`ς` is `+0.34`). "
                "The strongest pair is `α`/`ο` at `+0.53`. Most squares are dark, meaning unrelated."
            ),
            q(
                "Is that a failure? What would we expect?",
                "Not quite, but let's look at what the numbers really say. With 32 random numbers, two unrelated rows usually score about `±0.14` just by chance. "
                "Most of our scores sit inside that noise, so what they show is at most a faint hint of letter families. It's not a clear map.\n\n"
                "Why? Our model is tiny (11,840 weights, one attention layer), it practised on a small amount of text, and rare letters like `ὰ` "
                "appear so few times that their rows barely moved from the random start. "
                "Real models, with far more text and far more weights, make much clearer maps.\n\n"
                "👉 **Predict first:** what is the typical size of the cosine similarity between two *random* 32-number rows?",
            ),
            code(
                """torch.manual_seed(1)
random_rows = F.normalize(torch.randn(2000, 32), dim=1)
random_sims = random_rows @ random_rows.T
off_diagonal = random_sims[~torch.eye(2000, dtype=torch.bool)]
print(f"typical size of a random pair: {off_diagonal.abs().mean():.2f}")"""
            ),
            saw(
                "About `0.14`. So `α`/`ε` at `+0.40` stands out a little, while `+0.05` or `-0.06` is just noise. "
                "The thing to take away is the idea: embeddings are numbers the model learns, so that similar tokens *can* end up similar. "
                "How well that works depends on the data and on the model."
            ),
            *questions(
                [
                    (
                        "Does `α` get a different row each time it appears?",
                        "No. It's one row per token, the same wherever `α` sits. And that's a problem, because then the model can't tell `ἄνδρα` from the same letters in another order. "
                        "[02_06](02_06_positions.ipynb) deals with that.",
                    ),
                    (
                        "Can I choose `embed_dim`?",
                        "Yes, it's a setting. Bigger means a richer description of each token, but also more weights to learn. We use 32. Big models use thousands.",
                    ),
                ]
            ),
            recap(
                [
                    "A token ID is only a label; an embedding gives each token a vector the model can compute with.",
                    "The embedding table has one row per token: 137 × 32 = 4,384 weights, and lookup just picks rows.",
                    "The numbers are learned; in our tiny model they show only a faint hint of similar letters being alike.",
                ],
                "[02_06: Positions](02_06_positions.ipynb): the same letter gets the same row everywhere, so how does the model know where it is?",
                can=[
                    "make an embedding table with `nn.Embedding` and say how many weights it has",
                    "look up the rows of a word and predict the shape you get",
                    "compare two rows with cosine similarity",
                ],
            ),
        ],
    )

    # =====================================================================
    # 02_text_to_numbers/02_06_positions.ipynb
    # =====================================================================
    save(
        "02_text_to_numbers/02_06_positions.ipynb",
        [
            title("02_06 Positions", "section 2.8"),
            where("numbers", ["src/homer_ai/model.py"]),
            learn(
                [
                    "Why the same row for `α` everywhere hides the order of the letters",
                    "What a positional embedding is, and how it's added to the token embedding",
                    "That our `model.py` has no positional embedding yet",
                ]
            ),
            words(
                [
                    (
                        "position",
                        "where a token sits in the chunk: 0 for the first, 1 for the second, and so on",
                        "in `ἄνδρα`, `ρ` is at position 3",
                    ),
                    (
                        "positional embedding",
                        "a second embedding table with one row per position; its row is added to the token's row",
                        "row 3 for the 4th letter",
                    ),
                    (
                        "context length",
                        "how many tokens the model reads at once: the number of positions",
                        "24 in our training",
                    ),
                ]
            ),
            *background(),
            q(
                "What is the problem?",
                "In [02_05](02_05_token_embeddings.ipynb) each token gets **one** row, wherever it appears. "
                'So `ἄνδρα` ("man") and the scrambled `ρνδἄα` use exactly the same five rows, just in a different order:\n\n'
                "```\nἄνδρα   IDs [58, 38, 29, 42, 26]     rows: ἄ ν δ ρ α\nρνδἄα   IDs [42, 38, 29, 58, 26]     rows: ρ ν δ ἄ α   (same five rows)\n```\n\n"
                "👉 **Predict first:** if we add up the five rows of each word, will the two totals be the same?",
            ),
            code(
                """import torch
from torch import nn

from homer_ai.tokenizer import CharTokenizer
from homer_ai.train_homer import BOOKS_DIR, load_books

tokenizer = CharTokenizer(load_books(BOOKS_DIR)[0])
torch.manual_seed(0)
table = nn.Embedding(tokenizer.vocab_size, 32)

word = torch.tensor(tokenizer.encode("ἄνδρα"))
scrambled = torch.tensor(tokenizer.encode("ρνδἄα"))
print(word.tolist(), scrambled.tolist())
total_word = table(word).sum(dim=0)
total_scrambled = table(scrambled).sum(dim=0)
print("same total?", torch.allclose(total_word, total_scrambled))"""
            ),
            saw("`True`. Nothing in the rows themselves says which letter came first."),
            q(
                "Surely the model can still see the order?",
                "Not really. The part of the model that reads the earlier letters is **attention** (chapter 3). It mixes in the rows of the letters before, "
                "but it only sees *which rows* are there, not *where* they sit. So the model couldn't tell `ἄνδρα` from `νἄδρα` by their earlier letters. "
                "We can only test that once we have attention, and [03_03](../03_attention/03_03_hiding_the_future.ipynb) runs the test on our real `HomerGreekLLM`.",
            ),
            q(
                "What is the fix?",
                "Give every position its own row too, in a second table called the **positional embedding**. "
                "Row 0 is for the 1st place, row 1 for the 2nd, and so on. Then the model adds the two rows together, number by number:\n\n"
                "```\nposition        0      1      2      3      4\ntoken row       ἄ      ν      δ      ρ      α\n+ position row  p0     p1     p2     p3     p4\n= model input   ἄ+p0   ν+p1   δ+p2   ρ+p3   α+p4\n```\n\n"
                "Now `ρ` at position 3 and `ρ` at position 0 give different inputs. Like the token table, the position table starts out random and is learned in training.\n\n"
                "Here's a tiny example with 3 numbers per row, so you can check the sums by eye (the rows are small hand-made numbers):\n\n"
                "👉 **Predict first:** the token rows are `[1, 0, 2]` for `ν` and `[0, 3, 1]` for `ο`; the position rows are `[10, 10, 10]` and `[20, 20, 20]`. What is `νο`'s input? And `ον`'s?",
            ),
            code(
                """token_rows = {"ν": torch.tensor([1, 0, 2]), "ο": torch.tensor([0, 3, 1])}
position_rows = torch.tensor([[10, 10, 10], [20, 20, 20]])

for text in ["νο", "ον"]:
    tokens = torch.stack([token_rows[ch] for ch in text])
    print(text, "→", (tokens + position_rows).tolist())"""
            ),
            saw(
                "`νο` gives `[[11, 10, 12], [20, 23, 21]]` and `ον` gives `[[10, 13, 11], [21, 20, 22]]`. "
                "Same two rows, but now the inputs differ. The order is in the numbers."
            ),
            q(
                "How big is the position table?",
                "One row per position, so as many rows as the **context length**, which is how many tokens the model reads at once. "
                "Our training chunk is 24 characters (see [02_04](02_04_sliding_window.ipynb)), so that's 24 rows, with the same 32 columns as the token table.\n\n"
                "👉 **Predict first:** how many weights is that?",
            ),
            code(
                """context_length = 24
positions = nn.Embedding(context_length, 32)
print("shape:  ", tuple(positions.weight.shape))
print("weights:", positions.weight.numel())
grown = 11840 + positions.weight.numel()
print(f"our model's weights would grow from 11,840 to {grown:,}")"""
            ),
            saw(
                "A `(24, 32)` table with 768 weights. That's a small addition to 11,840."
            ),
            q(
                "Does our `model.py` have positions?",
                "No. Here are the real lines of `HomerGreekLLM` that touch the embedding, printed straight from the code:\n\n"
                "👉 **Predict first:** how many lines will mention a position table?",
            ),
            code(
                """import inspect

from homer_ai.model import HomerGreekLLM

for line in inspect.getsource(HomerGreekLLM).splitlines():
    if "embedding" in line:
        print(line.strip())"""
            ),
            saw(
                "Only the token embedding: `self.embedding = nn.Embedding(vocab_size, embed_dim)` and `x = self.embedding(idx)`. "
                "Nothing about positions. That's one reason our model is limited: it knows which letters came before, but not in what order. "
                "The GPT of chapter 4 does have a position table (`position_embedding` in `src/homer_ai/gpt.py`, built in [04_05](../04_building_the_model/04_05_the_full_model.ipynb)). We don't change `model.py` here."
            ),
            q(
                "Can I see token + position = input?",
                "Yes. Here are three stacked pictures for `ἄνδρα`: the five token rows, the five position rows, and their sum. "
                "These are untrained random tables, so it's the shape of the idea that counts. The last picture isn't the same as the first one, "
                "because each row has been shifted by its position.\n\n"
                "👉 **Predict first:** will the bottom picture look more like the top one or the middle one?",
            ),
            code(
                """import matplotlib.pyplot as plt

from homer_ai.studio_style import GOOD_BAD, INK, MUTED, studio_figure

torch.manual_seed(0)
token_table = nn.Embedding(tokenizer.vocab_size, 32)
position_table = nn.Embedding(context_length, 32)

ids = torch.tensor(tokenizer.encode("ἄνδρα"))
token_part = token_table(ids).detach()
position_part = position_table(torch.arange(len(ids))).detach()
model_input = token_part + position_part

fig = studio_figure("Token + position = input", "ἄνδρα as numbers", size=(9, 7.5))
limit = max(t.abs().max().item() for t in (token_part, position_part, model_input))
panels = [
    ("token rows", token_part),
    ("+ position rows", position_part),
    ("= model input", model_input),
]
for k, (name, grid) in enumerate(panels):
    ax = fig.add_axes((0.13, 0.60 - k * 0.27, 0.8, 0.19))
    ax.set_facecolor("none")
    ax.imshow(grid, cmap=GOOD_BAD, vmin=-limit, vmax=limit, aspect="auto")
    labels = [f"{ch}  pos {p}" for p, ch in enumerate("ἄνδρα")]
    ax.set_yticks(range(5), labels, fontsize=9, color=INK)
    ax.set_xticks([])
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(name, loc="left", color=MUTED, fontsize=10)
plt.show()"""
            ),
            saw(
                "The bottom picture is a blend of the two above it. "
                "Each row is its letter's pattern plus its position's pattern, so `ἄ` at position 0 and `ἄ` at position 3 would look different."
            ),
            q(
                "Are there other ways to give positions?",
                "Yes. What we described is **absolute** positions: each place (0, 1, 2 …) has its own learned row. "
                'Relative positions tell the model how far apart two letters are ("3 places before"), wherever they sit in the chunk. '
                "GPT-2 uses the absolute kind, and so does the GPT we build in chapter 4. Many newer models use a relative kind. We stay with absolute positions: one extra table, and nothing else changes.",
            ),
            recap(
                [
                    "One row per token hides the order: `ἄνδρα` and `ρνδἄα` use the same five rows (03_03 shows our model really can't tell).",
                    "A positional embedding is a second learned table, one row per position (24 × 32 for us), added to the token row.",
                    "Our `model.py` has no positional embedding. The GPT of chapter 4 (`gpt.py`) has one.",
                ],
                "[chapter 3: Attention](../03_attention/03_00_chapter_overview.ipynb): how each letter looks back at the letters before it.",
                can=[
                    "say why token rows alone can't tell `ἄνδρα` from `ρνδἄα`",
                    "add a positional row to each token row",
                ],
            ),
        ],
    )
