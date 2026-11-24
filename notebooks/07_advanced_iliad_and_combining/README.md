# Chapter 7: Advanced: the Iliad, and combining both

## ❓ Why does this chapter matter?

We have an Odyssey model. Here we repeat the same steps for the **Iliad**, Homer's other great poem, and compare the two models fairly. Then we combine them into one model trained on both poems, with `<|endoftext|>` between the works.

So the code works for any text, not just the Odyssey. It runs on the texts in `data/homer/` and uses small slices and a few epochs, so every cell takes seconds. Full runs take longer, and the notebooks give you the code to run them yourself.

The theory, the new words and other learners' questions live in the overview notebook, [07_00_chapter_overview.ipynb](07_00_chapter_overview.ipynb).

## ❓ Which notebook do I open first?

Open them in this order.

| Notebook | What it teaches |
|---|---|
| [07_00_chapter_overview.ipynb](07_00_chapter_overview.ipynb) | The theory and plan: the same steps on a new poem, fair comparison, three ways to combine |
| [07_01_the_corpus.ipynb](07_01_the_corpus.ipynb) | What is in data/homer/, in real numbers: sizes, vocabularies, the first lines |
| [07_02_an_iliad_model.ipynb](07_02_an_iliad_model.ipynb) | The same code on the Iliad: tokenizer, split, training, writing, and the code for a full run |
| [07_03_comparing_fairly.ipynb](07_03_comparing_fairly.ipynb) | Why two models need one shared vocabulary, and a fair 2×2 experiment |
| [07_04_combining_both.ipynb](07_04_combining_both.ipynb) | One model on both poems with <\|endoftext\|> between them, and what it gives and costs |
| [07_05_english_and_greek.ipynb](07_05_english_and_greek.ipynb) | The same steps in Butler's English: vocabulary, prose, tokens, a tiny model |

## ❓ What should I know before I start?

All of chapters 1-6. Chapter 5's [05_04](../05_training_on_the_odyssey/05_04_running_fair_experiments.ipynb) and [05_05](../05_training_on_the_odyssey/05_05_validation.ipynb) matter most.

## ❓ Where is this in the project code?

- `data/homer/`: the Greek and English texts of the Odyssey and the Iliad
- `src/homer_ai/train_homer.py`: `load_books`, `train` and `save_checkpoint`

## ❓ Do I need the book?

No. This chapter is our own; it isn't in the book.
