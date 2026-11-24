# ruff: noqa: E501
"""The short README of each chapter folder: its facts, and the code that writes it."""

from .cells import JOURNEY, ROOT
from .ch00_start_here import TOOL_CHAIN
from .diagrams import DIAGRAMS

# =====================================================================
# chapter READMEs
# =====================================================================
CHAPTERS = {
    "00_start_here": {
        "title": "Start here",
        "stage": None,
        "book": None,
        "why": (
            "Start here. Three short pages get you ready: a map of the whole course, "
            "setting up your computer, and reading an error message.\n\n"
            "Where we're heading is three models, one after the other:\n\n"
            "```\n"
            " Odyssey model  →  Iliad model  →  combine both\n"
            " (chapters 1-6)    (same steps)    (chapter 7)\n"
            "```"
        ),
        "learn": [
            "What we're building, and the order of the chapters",
            "How to set up the project and run a notebook",
            "How to read an error message",
        ],
        "words": [
            (
                "notebook",
                "a file of small boxes (cells) with code or notes; you run one cell and see the result under it",
                "`00_00_course_map.ipynb`",
            ),
            (
                "kernel",
                "the Python that runs your notebook's cells",
                "the one in `.venv`",
            ),
            (
                "traceback",
                "the message Python prints when it stops on an error; read its last line first",
                "`NameError: name 'epochs' is not defined`",
            ),
        ],
        "notebooks": [
            (
                "00_00_course_map.ipynb",
                "The whole course on one page: what we build, the chapters, and how an LLM can use a tool",
            ),
            (
                "00_01_setup.ipynb",
                "Terminal, VS Code, git, uv, the kernel, and a cell that checks your setup",
            ),
            (
                "00_02_reading_errors.ipynb",
                "How to read an error, using the real ones we hit while building this project",
            ),
        ],
        "before": "Nothing at all. This is where it starts.",
        "files": [
            ("pyproject.toml", "the shopping list of everything `uv sync` installs"),
        ],
        "questions": [
            (
                "Do I need to know Python?",
                "Not really, but you do need to be a developer in some language. We explain the AI and the PyTorch, not general Python. "
                "If Python is new to you, [A_01](../A_toolbox/A_01_python_essentials.ipynb) is an optional primer.",
            ),
            (
                "Can I break anything?",
                "No. The notebooks only read the text and run short demos. They never change the code or the trained models.",
            ),
        ],
    },
    "01_foundations": {
        "title": "Chapter 1: Foundations",
        "stage": None,
        "book": "Ch 1 and Appendix A (Introduction to PyTorch)",
        "why": (
            "Before we build a language model, there are four ideas you need, because every later chapter uses them: "
            "what a language model is, what a neural network is, how a program learns, and what our own model does when you look at it from the outside.\n\n"
            "It's a bit like learning what bricks, floors and stairs are before you build a house. "
            "There's a picture of this in [docs/neural-network-building.html](../../docs/neural-network-building.html) (GitHub shows HTML as text: open the file locally in your browser)."
        ),
        "learn": [
            "What a language model is, and how small ours is",
            "A language model made only of counting, and why it isn't enough",
            "What a neural network is, and how to count its weights",
            "How a program learns: loss, gradient, backpropagation",
            "What our model does as a black box, untrained and trained",
        ],
        "words": [],
        "notebooks": [
            (
                "01_00_chapter_overview.ipynb",
                "The big ideas of the chapter, in plain words",
            ),
            (
                "01_01_a_language_model_without_a_brain.ipynb",
                "Counting which character follows which, and writing Greek with it",
            ),
            (
                "01_02_neural_networks.ipynb",
                "Layers, weights, and the floors of our model",
            ),
            (
                "01_03_how_a_program_learns.ipynb",
                "Guess, mistake score, adjust, repeat: one weight by hand, then with PyTorch",
            ),
            (
                "01_04_our_model_as_a_black_box.ipynb",
                "What our model writes, untrained and trained",
            ),
        ],
        "before": "[00_start_here](../00_start_here/): set up first with [00_01](../00_start_here/00_01_setup.ipynb). "
        "New to Python? See [A_01](../A_toolbox/A_01_python_essentials.ipynb). New to tensors? See [A_02](../A_toolbox/A_02_pytorch_tensors.ipynb).",
        "files": [
            ("src/homer_ai/model.py", "the model's design: its layers and weights"),
            ("src/homer_ai/train_homer.py", "`train()`: the learning loop"),
            (
                "src/homer_ai/homer_model.pt",
                "the trained model (made by training; not stored in git)",
            ),
        ],
        "questions": [],
    },
    "02_text_to_numbers": {
        "title": "Chapter 2: Text to numbers",
        "stage": "numbers",
        "book": "Ch 2 (Working with text data)",
        "why": (
            "A model can only do maths, so it can't read Greek letters. In this chapter we turn the "
            "Odyssey into numbers, and those numbers are all the model will ever see.\n\n"
            "Get this step wrong (say, letters quietly thrown away) and the model learns from "
            "the wrong text. No amount of training fixes that."
        ),
        "learn": [
            "Why text has to become numbers",
            "How to cut text into tokens, and give each token a number",
            "Why special tokens are needed: unknown letters, and where one book ends",
            "How BPE joins common letter groups into single tokens",
            "How the text is cut into practice pieces, and how token numbers become embeddings with positions",
        ],
        "words": [
            (
                "token",
                "one piece of text the model reads as a single unit",
                "`α`, or `λλα`",
            ),
            (
                "vocabulary",
                "the list of all the tokens a tokenizer knows",
                "our 137 tokens",
            ),
            (
                "token ID",
                "the number a token gets in the vocabulary",
                "`π` → 41 in the real Book 1 vocabulary (in 02_01's 5-character toy it's 4)",
            ),
            (
                "tokenizer",
                "the tool that turns text into token IDs (encode) and back (decode)",
                "`CharTokenizer`",
            ),
            (
                "special token",
                "an extra token that is a signal, not real text",
                "`<|unk|>`, `<|endoftext|>`",
            ),
            (
                "BPE",
                "byte pair encoding: building a vocabulary by joining the most common neighbouring pairs, again and again",
                "`ν` + space → `ν `",
            ),
        ],
        "notebooks": [
            (
                "02_00_chapter_overview.ipynb",
                "The ideas of the whole chapter, in plain words",
            ),
            (
                "02_01_tokens_and_vocabulary.ipynb",
                "Character tokens, the vocabulary, encode and decode (book 2.2 + 2.3)",
            ),
            (
                "02_02_special_tokens.ipynb",
                "`<|unk|>` for unknown letters, `<|endoftext|>` between books",
            ),
            ("02_03_bpe.ipynb", "How BPE builds its vocabulary by counting"),
            (
                "02_04_sliding_window.ipynb",
                "Cutting the text into practice pieces for the model",
            ),
            (
                "02_05_token_embeddings.ipynb",
                "Turning each token number into a list of numbers that can carry meaning",
            ),
            (
                "02_06_positions.ipynb",
                "Telling the model where each token sits in the line",
            ),
        ],
        "before": "[01_foundations](../01_foundations/). New to tensors? See [A_02](../A_toolbox/A_02_pytorch_tensors.ipynb). New to Python? See [A_01](../A_toolbox/A_01_python_essentials.ipynb).",
        "files": [
            (
                "src/homer_ai/tokenizer.py",
                "`CharTokenizer`: text → numbers and back, with the special tokens",
            ),
            (
                "src/homer_ai/train_homer.py",
                "`load_books` and `load_data`: read the books and put `<|endoftext|>` between them",
            ),
        ],
        "questions": [
            (
                "What happens to a letter the model has never seen?",
                "It becomes `<|unk|>`. Without that token an unknown letter would just be dropped, and `Ζεὺς!` would quietly turn into `Ζεὺς`.",
            ),
            (
                "Why isn't a line break enough between books?",
                "Book 1 already has 444 line breaks, one per verse. One more looks just like the others.",
            ),
            (
                "Isn't the BPE choice of pairs random?",
                "No. A program counts every pair, and the most common one wins. In the Odyssey that's `ν` followed by a space.",
            ),
        ],
    },
    "03_attention": {
        "title": "Chapter 3: Attention",
        "stage": "model reads",
        "book": "Ch 3 (Coding attention mechanisms)",
        "why": (
            "To guess the next letter well, the model has to know which earlier letters matter "
            "most. In `πολλὰ`, the letters just before matter a lot, and a word ten verses back "
            "matters much less. **Attention** is how the model decides.\n\n"
            "You'll find it at the centre of every modern language model."
        ),
        "learn": [
            "What attention is, with an everyday example",
            "How the model scores how much each earlier token matters",
            "Why the model must not peek at the letters that come after",
            'Why several attention "heads" work better than one',
        ],
        "words": [
            (
                "attention",
                "the model's way of deciding which earlier tokens matter most for the next guess",
                "like re-reading the key words of a sentence",
            ),
            (
                "mask",
                "a cover that hides the future tokens, so the model can't cheat",
                "while guessing letter 5, letters 6+ are hidden",
            ),
        ],
        "notebooks": [
            (
                "03_00_chapter_overview.ipynb",
                "The theory of attention: why the model looks back, and which letters matter",
            ),
            (
                "03_01_simple_attention.ipynb",
                "Scores, softmax and a weighted mix, with nothing to learn",
            ),
            (
                "03_02_trainable_attention.ipynb",
                "Query, key and value: attention the model can learn, rebuilt from `model.py`",
            ),
            (
                "03_03_hiding_the_future.ipynb",
                "The causal mask: no peeking at later letters",
            ),
            (
                "03_04_many_heads.ipynb",
                "Several attention heads at once (our `model.py` has one)",
            ),
        ],
        "before": "[02_text_to_numbers](../02_text_to_numbers/), up to token embeddings. The maths of attention is the dot product: see [A_03](../A_toolbox/A_03_dot_product.ipynb).",
        "files": [
            (
                "src/homer_ai/model.py",
                "the attention part of `HomerGreekLLM` (one simple head)",
            )
        ],
        "questions": [],
    },
    "04_building_the_model": {
        "title": "Chapter 4: Building the model",
        "stage": "model reads",
        "book": "Ch 4 (Implementing a GPT model from scratch)",
        "why": (
            "Attention alone isn't a full model. This chapter adds the other parts and stacks "
            "them into floors, like a building, to make a GPT-style model.\n\n"
            "Our small model (`model.py`) has only one attention part and none of these extras. "
            "That's one reason its loss stops at about 2.4."
        ),
        "learn": [
            "The extra parts a full model needs, and what each one is for",
            "How the parts stack into a repeating block",
            "How to build the whole model in code",
            "How the model turns its numbers back into a guessed letter",
        ],
        "words": [
            (
                "GPT",
                "a well-known design for language models; the book builds a small one",
                "GPT-2",
            ),
            (
                "block",
                "one floor of the model: attention plus a few helper parts; a model stacks many",
                "our small model has none, our GPT has 4, GPT-2 small has 12",
            ),
        ],
        "notebooks": [
            (
                "04_00_chapter_overview.ipynb",
                "The theory of the full model, and the config",
            ),
            (
                "04_01_layer_norm.ipynb",
                "Layer norm: keeping the numbers from drifting",
            ),
            (
                "04_02_feed_forward.ipynb",
                "The feed-forward step and the GELU curve",
            ),
            (
                "04_03_shortcuts.ipynb",
                "Shortcut connections: keeping the learning signal alive",
            ),
            (
                "04_04_one_block.ipynb",
                "Putting the parts into one block",
            ),
            (
                "04_05_the_full_model.ipynb",
                "Stacking blocks into the full model, and writing (untrained) letters",
            ),
        ],
        "before": "[03_attention](../03_attention/)",
        "files": [
            (
                "src/homer_ai/model.py",
                "`HomerGreekLLM`, the small one-head model we compare against",
            ),
            (
                "src/homer_ai/gpt.py",
                "`GPTModel`, the full model this chapter builds",
            ),
        ],
        "questions": [],
    },
    "05_training_on_the_odyssey": {
        "title": "Chapter 5: Training on the Odyssey",
        "stage": "model learns",
        "book": "Ch 5 (Pretraining on unlabeled data)",
        "why": (
            "A new model only makes random guesses. Training is how it gets better: it guesses "
            "the next letter, measures how wrong it was, and adjusts its weights, many thousands "
            "of times.\n\n"
            "And how do you know it really **learned** patterns, and didn't just memorise the text? "
            "This chapter shows you how to check."
        ),
        "learn": [
            "What the loss is, and how to read a loss curve",
            "How the learning rate (step size) changes training, and how to change it over time",
            "The difference between memorising and understanding",
            "How to check the model on hidden text, and how to save and load it",
        ],
        "words": [
            (
                "loss",
                "a mistake score: how wrong the guesses were; lower is better",
                "`2.38`",
            ),
            (
                "epoch",
                "one full pass through all the training text",
                "we train for 800",
            ),
            ("learning rate", "how big each adjustment step is", "`lr=0.01`"),
            (
                "learning-rate schedule",
                "a plan to change the step size during training: big steps first, small steps later",
                "cosine",
            ),
            (
                "overfitting",
                "memorising the training text instead of learning its patterns",
                "a student who memorised one exam",
            ),
            (
                "validation set",
                "text kept hidden from training, used only to test the model",
                "the last 10% of the text",
            ),
        ],
        "notebooks": [
            ("05_00_chapter_overview.ipynb", "The theory of training"),
            (
                "05_01_loss_and_epochs.ipynb",
                "Loss, epochs, and reading the loss curve",
            ),
            (
                "05_02_learning_rate.ipynb",
                "Step size, the trade-off, and the cosine schedule",
            ),
            (
                "05_03_memorising_vs_understanding.ipynb",
                "Overfitting, and why we need hidden test text",
            ),
            (
                "05_04_running_fair_experiments.ipynb",
                "Baseline, diff, fixed seed, one change at a time, timing, a results log",
            ),
            (
                "05_05_validation.ipynb",
                "Splitting the text and measuring the validation loss",
            ),
            (
                "05_06_saving_and_loading.ipynb",
                "Saving the trained model to a file, and loading it again",
            ),
            (
                "05_07_train_the_gpt.ipynb",
                "The GPT's own training loop: random batches, weight decay, dropout, and the command to train yours",
            ),
        ],
        "before": "[04_building_the_model](../04_building_the_model/). Most notebooks here use our small model (`model.py`), so they run in seconds. "
        "How PyTorch measures the loss and adjusts the weights: see [A_04](../A_toolbox/A_04_pytorch_in_this_repo.ipynb).",
        "files": [
            (
                "src/homer_ai/train_homer.py",
                "`train()`: the small model's training loop, the learning rate and the schedule; `train_gpt()`: the GPT's loop (05_07)",
            ),
            (
                "src/homer_ai/homer_model.pt",
                "the trained model, saved after training (not stored in git)",
            ),
        ],
        "questions": [
            (
                "Why did the loss suddenly jump up?",
                "The steps were too big: near the bottom, a big step overshoots and lands higher up the other side.",
            ),
            (
                "Can we change the learning rate during training?",
                "Yes. That's a learning-rate schedule: big steps first, small steps near the end.",
            ),
            (
                "Is a very low loss always good?",
                "No. On 5 lines of text, a low loss meant the model had memorised them. Hidden test text shows the truth.",
            ),
        ],
    },
    "06_making_it_talk": {
        "title": "Chapter 6: Making it talk",
        "stage": "model writes",
        "book": "Sections 4.7, 5.1.1 and 5.3 (text generation) and our own (MCP)",
        "why": (
            "A trained model is only useful if it can write. In this chapter we turn its guesses into "
            "new Greek text, one letter at a time. Then we let other AI apps, like Claude, "
            "ask it for text through MCP."
        ),
        "learn": [
            "How the model writes: guess one letter, add it, guess again",
            "How to make the writing less repetitive (greedy writing loops: `και και και`)",
            "What MCP is, and how our server offers the model as a tool",
        ],
        "words": [
            (
                "generation",
                "making new text by guessing one token at a time and adding it to the end",
                "`Ἄνδρα` → `Ἄνδρατος ἀντος…`",
            ),
            (
                "MCP",
                "Model Context Protocol: the standard way for AI apps like Claude to call outside tools",
                "`homer_mcp_server.py`",
            ),
        ],
        "notebooks": [
            (
                "06_00_chapter_overview.ipynb",
                "The theory: writing letter by letter, and MCP",
            ),
            (
                "06_01_writing_greek.ipynb",
                "Building `write()`, and why greedy writing loops",
            ),
            (
                "06_02_less_repetition.ipynb",
                "Temperature and top-k: choosing the next letter with some chance",
            ),
            (
                "06_03_the_mcp_server.ipynb",
                "How our MCP server works, and how to connect it to Claude Code",
            ),
        ],
        "before": "[05_training_on_the_odyssey](../05_training_on_the_odyssey/), because you need a trained model. For uv, git and MCP, see [A_06](../A_toolbox/A_06_other_tools.ipynb).",
        "files": [
            (
                "src/homer_ai/homer_mcp_server.py",
                "the MCP server: loads the model and offers it as a tool",
            )
        ],
        "questions": TOOL_CHAIN,
    },
    "07_advanced_iliad_and_combining": {
        "title": "Chapter 7: Advanced: the Iliad, and combining both",
        "stage": None,
        "book": None,
        "why": (
            "We have an Odyssey model. Here we repeat the same steps for the **Iliad**, Homer's other great poem, and compare the two models fairly. "
            "Then we combine them into one model trained on both poems, with `<|endoftext|>` between the works.\n\n"
            "So the code works for any text, not just the Odyssey. "
            "It runs on the texts in `data/homer/` and uses small slices and a few epochs, so every cell takes seconds. "
            "Full runs take longer, and the notebooks give you the code to run them yourself."
        ),
        "learn": [
            "How to reuse every step of the course on a new text",
            "How to compare two models fairly",
            "Ways to combine two texts or two models, and their trade-offs",
        ],
        "words": [],
        "notebooks": [
            (
                "07_00_chapter_overview.ipynb",
                "The theory and plan: the same steps on a new poem, fair comparison, three ways to combine",
            ),
            (
                "07_01_the_corpus.ipynb",
                "What is in data/homer/, in real numbers: sizes, vocabularies, the first lines",
            ),
            (
                "07_02_an_iliad_model.ipynb",
                "The same code on the Iliad: tokenizer, split, training, writing, and the code for a full run",
            ),
            (
                "07_03_comparing_fairly.ipynb",
                "Why two models need one shared vocabulary, and a fair 2×2 experiment",
            ),
            (
                "07_04_combining_both.ipynb",
                "One model on both poems with <|endoftext|> between them, and what it gives and costs",
            ),
            (
                "07_05_english_and_greek.ipynb",
                "The same steps in Butler's English: vocabulary, prose, tokens, a tiny model",
            ),
        ],
        "before": "All of chapters 1-6. Chapter 5's [05_04](../05_training_on_the_odyssey/05_04_running_fair_experiments.ipynb) and [05_05](../05_training_on_the_odyssey/05_05_validation.ipynb) matter most.",
        "files": [
            ("data/homer/", "the Greek and English texts of the Odyssey and the Iliad"),
            (
                "src/homer_ai/train_homer.py",
                "`load_books`, `train` and `save_checkpoint`",
            ),
        ],
        "questions": [],
    },
    "08_advanced_fine_tuning": {
        "title": "Chapter 8: Advanced: fine-tuning a pretrained model",
        "stage": None,
        "book": "section 5.5 (loading OpenAI's weights), chapter 6 (classification), chapter 7 (instructions) and appendix E (LoRA)",
        "why": (
            "Chapters 1 to 7 built a model from nothing. In real work you rarely do that: you take a model somebody already trained and give it a short course for your job. That's **fine-tuning**.\n\n"
            "Our Homer model is like a child we taught to babble Greek from scratch. GPT-2 is an adult who already speaks English. "
            "So here we borrow GPT-2 small, load its weights into the `GPTModel` you built in chapter 4, and teach it two jobs: sorting text messages into spam and not spam, and answering simple requests.\n\n"
            "This chapter downloads GPT-2's weights once (548 MB) to `~/.cache/homer-ai/`, outside the project. Each notebook runs in a few minutes on a laptop CPU. "
            "The results are reported as they came: the spam filter works well, and instruction-following only partly."
        ),
        "learn": [
            "What is inside a weights file, and how to load GPT-2 into our own model",
            "How to fine-tune for classification with most layers frozen",
            "What instruction fine-tuning is, and what a small model can learn from it",
            "How LoRA fine-tunes by training far fewer numbers",
        ],
        "words": [],
        "notebooks": [
            (
                "08_00_chapter_overview.ipynb",
                "The plan: prompt, RAG, fine-tune or from scratch; why we borrow GPT-2; what to expect",
            ),
            (
                "08_01_loading_gpt2.ipynb",
                "Read GPT-2's weights file by hand and pour it into our GPTModel, until it writes English",
            ),
            (
                "08_02_spam_classifier.ipynb",
                "A new 2-score head, frozen layers, and a spam filter trained in about a minute",
            ),
            (
                "08_03_instructions.ipynb",
                "A small instruction set from Butler's Homer, and an honest look at what GPT-2 small learns",
            ),
            (
                "08_04_lora.ipynb",
                "LoRA in 20 lines: the same spam filter with 5 times fewer trained numbers",
            ),
        ],
        "before": "Chapters 4 to 6: the GPT ([04_05](../04_building_the_model/04_05_the_full_model.ipynb)), the training loop ([05_07](../05_training_on_the_odyssey/05_07_train_the_gpt.ipynb)) and writing text ([06_02](../06_making_it_talk/06_02_less_repetition.ipynb)). BPE tokens ([02_03](../02_text_to_numbers/02_03_bpe.ipynb)) help too. You need an internet connection for the first run, and about 2 GB of free memory.",
        "files": [
            (
                "src/homer_ai/gpt.py",
                "`GPTModel`, the class GPT-2's weights are loaded into",
            ),
            (
                "src/homer_ai/pretrained.py",
                "download and check GPT-2, read the safetensors file, map the names (`load_gpt2`)",
            ),
            (
                "src/homer_ai/finetune.py",
                "the spam data, the classification head and the training loop",
            ),
            (
                "src/homer_ai/lora.py",
                "`LoRALinear` and `apply_lora`",
            ),
            (
                "tests/test_fine_tuning.py",
                "tests of those three files; they never download",
            ),
        ],
        "questions": [],
    },
    "09_advanced_grounded_answers": {
        "title": "Chapter 9: Advanced: answers grounded in Homer's text",
        "stage": None,
        "book": None,
        "why": (
            "Ask a language model about Homer and it answers from memory. It may well be right, but you can't see where the answer came from. "
            "This chapter builds the other way: a search fetches the passages that fit your question, the model answers from them, and the answer comes with the book and line, so you can check it. That's **RAG**, retrieval-augmented generation.\n\n"
            "Think of a closed-book exam and an open-book one. Here a librarian fetches the right pages first, and you write the page number beside your answer. "
            "The librarian is a word search called BM25, which we build from scratch. The student is Claude, who reaches the search through a second tool on our MCP server.\n\n"
            "Nothing is trained and nothing is downloaded. We measure the search on 40 labelled questions and report the numbers as they came: it finds a right passage in its first 5 for about half of them, "
            "and on a poem as famous as Homer's, Claude does very well from memory. What grounding adds is a line you can check."
        ),
        "learn": [
            "How to cut a text into passages that remember their book and line",
            "How BM25 scores a passage against a question: tf, idf and length",
            "How to score a search with labelled questions: recall@k and MRR",
            "How to offer the search as an MCP tool, and connect it to several AI apps",
            "How to test whether grounding helps, and what changes on a company's own documents",
        ],
        "words": [],
        "notebooks": [
            (
                "09_00_chapter_overview.ipynb",
                "The plan: an open-book exam for Claude, why our own model can't answer, what to expect",
            ),
            (
                "09_01_chunking_and_bm25.ipynb",
                "Passages with book and line, then bag of words, idf and BM25 by hand on three real passages",
            ),
            (
                "09_02_search_and_scores.ipynb",
                "Both poems in one search, 40 labelled questions, recall@k and MRR, and two kinds of embeddings",
            ),
            (
                "09_03_the_search_tool.ipynb",
                "`search_homer` on the MCP server, and the settings for Claude Code, Claude Desktop, Codex CLI, LM Studio and nanobot",
            ),
            (
                "09_04_with_and_without.ipynb",
                "Claude with and without the passages, an honest reading of the result, and the same parts in a company",
            ),
        ],
        "before": "The MCP server of [06_03](../06_making_it_talk/06_03_the_mcp_server.ipynb), and the habit of measuring from [05_04](../05_training_on_the_odyssey/05_04_running_fair_experiments.ipynb). "
        "Embeddings ([02_05](../02_text_to_numbers/02_05_token_embeddings.ipynb)) and the dot product ([A_03](../A_toolbox/A_03_dot_product.ipynb)) help for one part of 09_02. Chapter 8 isn't needed.",
        "files": [
            (
                "src/homer_ai/retrieval.py",
                "passages, `tokenize`, `BM25`, `search`, and the scores `recall@k` and MRR",
            ),
            (
                "src/homer_ai/homer_mcp_server.py",
                "the MCP server; `search_homer` is its second tool",
            ),
            (
                "data/eval/questions.jsonl",
                "40 questions about the two poems, each with the book and lines that answer it",
            ),
            (
                "scripts/benchmark_grounding.py",
                "asks Claude the questions with and without the passages (run by hand, not by CI)",
            ),
            (
                "tests/test_retrieval.py",
                "tests of the search, the labelled questions and the scores",
            ),
        ],
        "questions": [],
    },
    "10_advanced_your_domain": {
        "title": "Chapter 10: Advanced: your own domain",
        "stage": None,
        "book": None,
        "why": (
            "Why would a company want its own LLM? Mostly it doesn't mean training one. It means a model that answers from the firm's own documents, or a small one that runs on the firm's own machines. "
            "This chapter takes the course's pattern (your data, a tool, an MCP server, a score with and without) to that kind of work.\n\n"
            "Think of four ways to get help with your paperwork: phone an outside expert, give that expert your binder and a few extensions to call, train a junior clerk for one job, or raise a child to do it. "
            "Those are the four rungs: prompt a big model, RAG and tools over MCP, fine-tune a small open model, train from scratch. Firms climb to the second and third for exact answers with a source, for data that can't leave, for cost at volume, and for a model that stays frozen. The fourth is rare.\n\n"
            "There are three worked examples: a support assistant for a savings firm, a classifier that flags complaints and vulnerable customers in email, and triage for an IT service desk. "
            "**All the data is invented** ([data/domain/README.md](../../data/domain/README.md)), so the scores show how to measure, and say nothing about real mail. The last notebook lists what would make such demos look naive to the people who'd have to run them."
        ),
        "learn": [
            "What a firm usually means by its own LLM, and the practical reasons to climb each rung",
            "How to answer from versioned documents, get an exact number from a lookup tool, and hand over to a person",
            "How to fine-tune a small model to flag complaints, and score it by recall first",
            "How to route tickets and suggest articles with no training, and how a split changes a score",
            "Nine pitfalls of AI demos in a regulated setting, and what a real organisation adds",
        ],
        "words": [],
        "notebooks": [
            (
                "10_00_chapter_overview.ipynb",
                "The ladder with the reasons a firm climbs it, the three examples, and the invented data",
            ),
            (
                "10_01_support_with_citations.ipynb",
                "A savings firm's support desk: `search_terms`, `get_rate`, `escalate`, a second MCP server, and Claude with and without the tools",
            ),
            (
                "10_02_spotting_complaints.ipynb",
                "400 emails, a keyword baseline, GPT-2 fine-tuned in under a minute, recall first, macro-F1, a threshold, and a reality check (needs GPT-2's 548 MB file)",
            ),
            (
                "10_03_service_desk_triage.ipynb",
                "200 tickets and 10 articles: suggest the article, route the ticket, and see what boilerplate and a random split do to the score",
            ),
            (
                "10_04_pitfalls_and_next_steps.ipynb",
                "Nine pitfalls, each tied to a place in the course, and where this goes next at work",
            ),
        ],
        "before": "The search of [chapter 9](../09_advanced_grounded_answers/) (BM25, recall@k, the MCP tool) and the fine-tuning of [08_02](../08_advanced_fine_tuning/08_02_spam_classifier.ipynb). "
        "10_01, 10_03 and 10_04 need no download. 10_02 uses GPT-2's weights from chapter 8.",
        "files": [
            (
                "src/homer_ai/domain.py",
                "the documents, the tools (`search_terms`, `get_rate`, `escalate`), the guard, the scores (`recall`, `macro_f1`) and the ticket routers",
            ),
            (
                "src/homer_ai/domain_server.py",
                "a second MCP server with the three support tools and an audit trail (`uv run python -m homer_ai.domain_server`)",
            ),
            (
                "data/domain/",
                "the invented product sheets, rates, questions, emails, tickets and articles, with a README on how they were made",
            ),
            (
                "scripts/make_domain_data.py",
                "writes the emails and tickets from sentence lists and a seed; the same seed gives the same bytes",
            ),
            (
                "scripts/benchmark_domain.py",
                "asks Claude the support questions with and without the MCP tools (run by hand, not by CI)",
            ),
            (
                "tests/test_domain.py",
                "tests of the tools, the data, the generator, the scores and the server",
            ),
        ],
        "questions": [],
    },
    "A_toolbox": {
        "title": "Appendix A: The toolbox",
        "stage": None,
        "book": "Appendix A (Introduction to PyTorch) covers part of this",
        "why": (
            "Building an LLM uses a handful of tools again and again: Python, PyTorch, a bit of maths (the dot product), "
            "tiktoken, and the everyday tools uv, Ruff, Jupyter, git and MCP. This appendix explains each one, and shows where this repo uses it.\n\n"
            "You don't read it first. **Dip in whenever you meet a tool you don't know yet.** New to tensors? See A_02. Meeting `@` in the model? See A_03."
        ),
        "learn": [
            "What a library is, and which ones we use",
            "Just the Python the course needs",
            "What PyTorch and tensors are",
            "The dot product, and why it's the heart of attention",
            "How `model.py` and the training loop use PyTorch",
            "Why GPT-2's tokenizer (via tiktoken) is a poor fit for Homer's Greek",
            "uv, Ruff, Jupyter, git and MCP",
        ],
        "words": [
            (
                "library",
                "code someone else wrote and shared, so you can use it",
                "PyTorch",
            ),
            (
                "tensor",
                "PyTorch's list or table of numbers",
                "`torch.tensor([1, 2, 3])`",
            ),
            (
                "dot product",
                "multiply two lists of numbers pair by pair, then add up",
                "`[1, 2]·[3, 4] = 11`",
            ),
            (
                "tiktoken",
                "OpenAI's fast BPE tokenizer library",
                '`tiktoken.get_encoding("gpt2")`',
            ),
        ],
        "notebooks": [
            (
                "A_00_overview.ipynb",
                "Every tool in one line, and which chapter needs it",
            ),
            (
                "A_01_python_essentials.ipynb",
                "Variables, lists, dictionaries, loops, functions, classes",
            ),
            (
                "A_02_pytorch_tensors.ipynb",
                "What PyTorch is, then tensors, shape, dtype, slicing and the seed",
            ),
            (
                "A_03_dot_product.ipynb",
                "The dot product, matrix multiplication, and `Q @ K.T`",
            ),
            (
                "A_04_pytorch_in_this_repo.ipynb",
                "Embedding, Linear, softmax, the loss, backward and the optimizer",
            ),
            (
                "A_05_tiktoken_bpe.ipynb",
                "GPT-2's tokenizer on English and on Greek",
            ),
            ("A_06_other_tools.ipynb", "uv, Ruff, Jupyter, git and MCP"),
            (
                "A_07_organising_code.ipynb",
                "Why the model lives in one shared file, `model.py`",
            ),
        ],
        "before": "Nothing. Open any notebook here whenever you need it.",
        "files": [
            ("pyproject.toml", "the shopping list of libraries"),
            ("src/homer_ai/model.py", "the model, built from PyTorch parts (A_04)"),
            (
                "src/homer_ai/train_homer.py",
                "the training loop (A_04) and the data as a tensor (A_02)",
            ),
        ],
        "questions": [
            (
                "I see 'Failed to initialize NumPy'. Is something broken?",
                "No, it's harmless. PyTorch prints it when it can't find numpy. This project installs numpy, so run `uv sync` again if you see it.",
            ),
            (
                "Why does GPT-2's tokenizer need more tokens than letters for Greek?",
                "It learned its merges from mostly English text, so Greek falls back to bytes: up to 3 tokens for one accented letter.",
            ),
        ],
    },
}


def cell(text):
    """Escape | so it does not break a markdown table (e.g. in `<|unk|>`)."""
    return text.replace("|", "\\|")


def chapter_readme(folder, ch):
    # Say it once: if the chapter has an overview notebook, the theory, the words and
    # the learner questions live there. The README stays a short map of the chapter.
    overview = next(
        (
            n
            for n, _ in ch["notebooks"]
            if n.endswith(("_00_chapter_overview.ipynb", "_00_overview.ipynb"))
            and (ROOT / folder / n).exists()
        ),
        None,
    )
    out = [f"# {ch['title']}", ""]
    if ch["stage"]:
        marked = JOURNEY.replace(ch["stage"], f"[{ch['stage']}]")
        out += ["## ❓ Where does this chapter fit?", "", "```", marked, "```", ""]
    out += ["## ❓ Why does this chapter matter?", "", ch["why"], ""]
    if overview:
        out += [
            f"The theory, the new words and other learners' questions live in the overview notebook, [{overview}]({overview}).",
            "",
        ]
    else:
        out += ["## ❓ What will I learn?", ""] + [f"- {x}" for x in ch["learn"]] + [""]
    if ch["words"] and not overview:
        out += [
            "## ❓ Which new words will I meet?",
            "",
            "| Word | Meaning in plain English | Example |",
            "|---|---|---|",
        ]
        out += [
            f"| **{cell(w)}** | {cell(m)} | {cell(e)} |" for w, m, e in ch["words"]
        ] + [""]
    pictures = [d for d in DIAGRAMS if d[3].split("/")[1] == folder]
    if pictures:
        out += ["## ❓ What will I see?", ""]
        for file_name, _, _, notebook, alt in pictures:
            note = notebook.rsplit("/", 1)[1]
            out += [
                f"![{alt}](../../docs/images/{file_name}.png)",
                "",
                f"*Drawn in [{note}]({note}).*",
                "",
            ]
    out += ["## ❓ Which notebook do I open first?", ""]
    if ch["notebooks"]:
        out += [
            "Open them in this order.",
            "",
            "| Notebook | What it teaches |",
            "|---|---|",
        ]
        for name, desc in ch["notebooks"]:
            out.append(f"| [{name}]({name}) | {cell(desc)} |")
        out += [""]
    out += ["## ❓ What should I know before I start?", "", ch["before"], ""]
    if ch["files"]:
        out += (
            ["## ❓ Where is this in the project code?", ""]
            + [f"- `{f}`: {d}" for f, d in ch["files"]]
            + [""]
        )
    if ch["questions"] and not overview:
        out += ["## ❓ What do other learners ask?", ""]
        for qq, a in ch["questions"]:
            out += [f"**{qq}**  ", a, ""]
    out += ["## ❓ Do I need the book?", ""]
    if ch["book"]:
        out += [
            f"No, everything is explained here. If you do have it, this chapter matches {ch['book']}.",
            "",
        ]
    else:
        out += ["No. This chapter is our own; it isn't in the book.", ""]
    return "\n".join(out)


def build():
    for folder, ch in CHAPTERS.items():
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
        (ROOT / folder / "README.md").write_text(
            chapter_readme(folder, ch), encoding="utf-8"
        )
