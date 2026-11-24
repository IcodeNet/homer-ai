# ruff: noqa: E501
"""Chapter 5, Training on the Odyssey."""

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
from .diagrams import DIAGRAM_TRAINING
from .setup_code import TINY_SETUP

TRAINING_WORDS = [
    (
        "weight",
        "one adjustable number inside the model; learning means changing the weights",
        "the model has thousands",
    ),
    (
        "loss",
        "a mistake score: how wrong the model's guesses were; lower is better",
        "`2.38`",
    ),
    (
        "training step",
        "one round of: guess → measure the loss → work out which way to nudge each weight → nudge them",
        "`loss.backward()` then `optimizer.step()`",
    ),
    (
        "optimizer",
        "the part of the code that nudges the weights after each guess; `AdamW` is the name of the one we use",
        "`torch.optim.AdamW`",
    ),
    ("epoch", "one full pass through all the training text", "we train for 800"),
    (
        "chunk",
        "one short piece of the text that the model practises on at a time",
        "24 characters",
    ),
]

# =====================================================================
# 05_training_on_the_odyssey/05_02_learning_rate.ipynb
# =====================================================================
LR_RUNS = r"""# tiny_train is the loop of 05_01; cosine=True adds the schedule
from homer_ai.lessons import tiny_train

RUNS = {
    "constant 0.01": {"epochs": 400},
    "constant 0.003": {"epochs": 400, "lr": 0.003},
    "cosine, 400": {"epochs": 400, "cosine": True},
    "cosine, 800": {"epochs": 800, "cosine": True},
}

curves = {}
head = [f"{'ep ' + str(e):>7}" for e in (100, 200, 300)]
print(f"{'run':15} {' '.join(head)} {'final':>7} {'jumps':>6}  last jump")
for name, settings in RUNS.items():
    losses = tiny_train(seed=42, **settings)
    curves[name] = losses
    # a jump: an epoch whose loss is more than 0.1 above the epoch before
    jumps = [e + 1 for e in range(1, len(losses)) if losses[e] - losses[e - 1] > 0.1]
    at = " ".join(f"{losses[e - 1]:7.3f}" for e in (100, 200, 300))
    print(f"{name:15} {at} {losses[-1]:7.3f} {len(jumps):>6}  epoch {jumps[-1]}")

steady = curves["constant 0.01"]
print()
print("constant 0.01, epochs 168-171:", [round(x, 2) for x in steady[167:171]])
print("constant 0.01, epochs 395-400:", [round(x, 3) for x in steady[394:400]])
late = curves["cosine, 400"]
print(f"cosine, 400: epoch 350 {late[349]:.3f}, epoch 400 {late[399]:.3f}")"""

# =====================================================================
# 05_training_on_the_odyssey/05_04_running_fair_experiments.ipynb
# =====================================================================
BUG_CODE = r"""def tiny_train_bug(seed, epochs, lr=0.01, chunk_size=24):
    # The same as tiny_train, but the three training lines are indented one
    # level too little: they sit AFTER the for loop, not inside it.
    torch.manual_seed(seed)
    model = HomerGreekLLM(vocab_size=tokenizer.vocab_size, embed_dim=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    average_losses = []
    for _epoch in range(epochs):
        losses = []
        for i in range(0, len(data) - chunk_size, chunk_size):
            inputs = data[i : i + chunk_size]
            targets = data[i + 1 : i + chunk_size + 1]
            loss = F.cross_entropy(model(inputs), targets)
            losses.append(loss.item())
        optimizer.zero_grad()  # runs once per epoch: only the LAST chunk teaches
        loss.backward()
        optimizer.step()
        average_losses.append(sum(losses) / len(losses))
    return average_losses


buggy = to_log(tiny_train_bug(42, 100))
for line in difflib.unified_diff(baseline, buggy, "baseline", "buggy", lineterm=""):
    print(line)"""

# =====================================================================
# 05_training_on_the_odyssey/05_05_validation.ipynb
# =====================================================================
EVALUATE_CODE = '''def evaluate(model, data, chunk_size=24):
    """Average loss on `data`. Measure only: the model does not learn."""
    losses = []
    with torch.no_grad():  # "measure only, don't learn"
        for i in range(0, len(data) - chunk_size, chunk_size):
            inputs = data[i : i + chunk_size]
            targets = data[i + 1 : i + chunk_size + 1]
            loss = F.cross_entropy(model(inputs), targets)
            losses.append(loss.item())
    return sum(losses) / len(losses)'''


def build():
    # =====================================================================
    # 05_training_on_the_odyssey/05_00_chapter_overview.ipynb
    # =====================================================================
    save(
        "05_training_on_the_odyssey/05_00_chapter_overview.ipynb",
        [
            title("05_00 Training on the Odyssey (overview)", "chapter 5"),
            where("model learns", ["src/homer_ai/train_homer.py"]),
            learn(
                [
                    'What "training" means for a model, in plain words',
                    "What the loss, a training step and an epoch are",
                    "What the learning rate is, and why its size matters",
                    "How to tell real learning from memorising",
                    "Which notebook to open for each idea",
                ]
            ),
            words(
                TRAINING_WORDS
                + [
                    (
                        "learning rate",
                        "how big each nudge to the weights is",
                        "`lr=0.01`",
                    ),
                    (
                        "validation set",
                        "text kept hidden from training, used only to test the model",
                        "the last 10% of the text",
                    ),
                ]
            ),
            *background(
                "At the stage **model learns**. The text is already numbers (chapter 2), and we have a model, `HomerGreekLLM` in `src/homer_ai/model.py`. "
                "But a brand-new model is full of random numbers, so its guesses are random too.",
                "Training is what turns random guesses into Homer-like Greek. And if training goes wrong (say the steps are too big, "
                "or the model just memorises the text), the model we serve to other AI apps writes nonsense.",
            ),
            q(
                "How does a model get better?",
                "Training is revising for an exam with a pile of flashcards. "
                "The model guesses the back of a card, we turn the card over, and it adjusts itself a little. Then the next card, thousands of times. "
                "You know the catch from school: go through the same pile often enough and you know the cards by heart without having understood the subject. "
                "So we lock a few cards in a drawer where the model never sees them, and those are the exam. "
                "Where the picture is wrong: a student gets the idea after a handful of cards and knows why an answer is right, and our model needs thousands of tiny nudges and only ever shifts numbers.\n\n"
                "In the model, one round of this is called a **training step**. It has four parts:\n\n"
                "1. **Guess** the next letter for every position in a short piece of text (a chunk).\n"
                "2. **Measure** how wrong the guesses were. That mistake score is the loss.\n"
                "3. **Work out** which way to nudge each adjustable number (each weight) so the loss goes down.\n"
                "4. **Nudge** the weights a little. The part of the code that does this is the optimizer.\n\n"
                "Here's one real flashcard from Book 1, `ἄνδρ → α`, before and after training:\n\n"
                "```\n"
                "flashcard:   ἄνδρ → α\n"
                "untrained:   chance for α 0.0056   →  loss 5.18   (a wild guess among 137)\n"
                "trained:     chance for α 0.1194   →  loss 2.125  (its top 3: ο 0.218, ε 0.201, α 0.119)\n"
                "```\n\n"
                "So training raised the right answer's chance about 20 times, and the loss fell. "
                "You'll compute these numbers yourself in [05_01](05_01_loss_and_epochs.ipynb).",
            ),
            q(
                "What do those steps look like in our code?",
                "They're in `train()` in `src/homer_ai/train_homer.py`. One word first: the model's scores for each possible next character are called **logits**.\n\n```python\nlogits = model(inputs)                   # 1. guess\n"
                "loss = F.cross_entropy(logits, targets) # 2. measure\noptimizer.zero_grad()                    # wipe the last chunk's nudges\n"
                "loss.backward()                          # 3. work out the nudges\noptimizer.step()                         # 4. nudge\n```\n\n"
                "`optimizer.zero_grad()` clears the nudges left over from the last chunk. Without it they'd pile up, like not clearing a calculator between sums.\n\n"
                "One pass through all the chunks of the text is an **epoch**. We train for 800 epochs.",
            ),
            md(
                "**Look at:** the ring from the top, clockwise. Each box shows its step and the line of code that does it. The orange box, the loss, is the number training tries to lower."
            ),
            code(DIAGRAM_TRAINING),
            q(
                "How big should each nudge be?",
                "That's the **learning rate**. Too big, and the model overshoots and the loss jumps up. Too small, and it learns very slowly. "
                "You can also change the size during training, big first and small later. That plan is called a learning-rate schedule.",
            ),
            q(
                "If the loss is low, has the model learned?",
                "Not necessarily. It may just have memorised the text. To find out, keep some text hidden from training and test on that. "
                "The hidden text is the **validation set**.",
            ),
            q(
                "Which notebook do I open for each idea?",
                "| Notebook | Question it answers |\n|---|---|\n"
                "| [`05_01_loss_and_epochs.ipynb`](05_01_loss_and_epochs.ipynb) | What's the loss, and how do I read a loss curve? |\n"
                "| [`05_02_learning_rate.ipynb`](05_02_learning_rate.ipynb) | Why did the loss jump up, and how big should the steps be? |\n"
                "| [`05_03_memorising_vs_understanding.ipynb`](05_03_memorising_vs_understanding.ipynb) | Is a low loss really good? |\n"
                "| [`05_04_running_fair_experiments.ipynb`](05_04_running_fair_experiments.ipynb) | How do I change something and know what the change did? |\n"
                "| [`05_05_validation.ipynb`](05_05_validation.ipynb) | Did the model learn, or memorise? |\n"
                "| [`05_06_saving_and_loading.ipynb`](05_06_saving_and_loading.ipynb) | How do I keep the trained model, and use it later? |\n"
                "| [`05_07_train_the_gpt.ipynb`](05_07_train_the_gpt.ipynb) | How is the GPT of chapter 4 trained, and how do I train my own? |",
            ),
            recap(
                [
                    "Training repeats one step: guess, measure the loss, work out the nudges, nudge the weights.",
                    "One pass through all the text is an epoch. The learning rate is the size of each nudge.",
                    "A low loss can mean memorising. Hidden text (the validation set) shows the truth.",
                ],
                "[05_01: Loss and epochs](05_01_loss_and_epochs.ipynb).",
            ),
        ],
    )

    # =====================================================================
    # 05_training_on_the_odyssey/05_01_loss_and_epochs.ipynb
    # =====================================================================
    save(
        "05_training_on_the_odyssey/05_01_loss_and_epochs.ipynb",
        [
            title("05_01 Loss and epochs", "sections 5.1.2 (loss) and 5.2 (training)"),
            where("model learns", ["src/homer_ai/train_homer.py"]),
            learn(
                [
                    "What a loss number means, for a good guess and a bad one",
                    "Why we report the average loss, not the total",
                    "How to read a loss curve, and why a few snapshots can hide spikes",
                    "Why the same seed gives the same results every run",
                ]
            ),
            words(
                TRAINING_WORDS
                + [
                    (
                        "score",
                        "how much the model likes each possible next token; the highest score is its guess",
                        "`[3.0, 0.0, 0.0]`",
                    ),
                    (
                        "loss curve",
                        "the loss after each epoch, one after another; it shows how training went",
                        "`3.88 → … → 0.71`",
                    ),
                    (
                        "seed",
                        'a starting number for randomness; the same seed gives the same "random" numbers',
                        "`torch.manual_seed(42)`",
                    ),
                ]
            ),
            *background(
                "At the stage **model learns**. We have the text as numbers and a model that makes guesses. "
                "Our training code prints a loss number every so often.",
                "We need to read those numbers correctly, because a wrong reading leads to wrong decisions. Print totals where you meant averages, "
                "or print too rarely, and big jumps in the loss stay hidden.",
            ),
            q(
                "What's the loss, exactly?",
                "A **mistake score**. For each position, the model gives every possible next token a score, and the highest score is its guess. "
                "The loss is small when the model gave the right answer a high score, and large when it didn't. Lower is better.",
            ),
            q(
                "What number should I hope for?",
                "Random guessing among our 137 tokens gives a loss of about **4.9**. Our model, trained on Book 1, reaches an average of about 2.38. "
                "So it has clearly learned something, but it's far from perfect.",
            ),
            q(
                "What does a loss of 2.38 mean?",
                "Here's a rough picture: e^2.38 ≈ 10.8, so the model is about as unsure as if it were choosing between **11** characters. "
                "At the start it was choosing between all 137. (`e` is a special number in maths, about 2.718. The loss is measured on that scale.)",
            ),
            q(
                "Why are the targets the inputs shifted by one?",
                "Because the job is always the same: guess the next character. If the input is `ἄνδρ`, the targets are `νδρα`. "
                "After `ἄ` the answer is `ν`, after `ἄν` it's `δ`, and so on. So one chunk of 24 characters gives the model **24 practice questions** at once.",
            ),
            q(
                "What's an epoch?",
                "One full pass through all the training text, chunk by chunk. Training repeats many epochs, so the model sees the same text again and again.",
            ),
            step_intro(),
            q(
                "What's the loss for one real flashcard?",
                "Take the flashcard `ἄνδρ → α` from [02_04](../02_text_to_numbers/02_04_sliding_window.ipynb). "
                "The model gives each of the 137 possible next characters a chance. The loss only looks at the chance it gave the **right** answer, `α`:\n\n"
                "```\n"
                "flashcard: ἄνδρ → α\n"
                "untrained: chance for α = 0.0056, loss = 5.180\n"
                "trained:   chance for α = 0.1194, loss = 2.125\n"
                "```\n\n"
                "👉 **Predict first:** which model gets the lower loss?",
            ),
            code(
                """import torch
import torch.nn.functional as F

from homer_ai.lessons import load_homer_model
from homer_ai.model import HomerGreekLLM

tok, trained = load_homer_model()  # read-only: the saved model, if there is one (05_06)
torch.manual_seed(42)
models = [("untrained", HomerGreekLLM(tok.vocab_size, 32))]
if trained is not None:
    models.append(("trained", trained))

inputs = torch.tensor(tok.encode("ἄνδρ"), dtype=torch.long)
target = torch.tensor(tok.encode("α"), dtype=torch.long)

print("flashcard: ἄνδρ → α")
for name, model in models:
    with torch.no_grad():
        scores = model(inputs)[-1:]  # the scores after the last character
    chance = F.softmax(scores, dim=-1)[0, target].item()
    loss = F.cross_entropy(scores, target).item()
    print(f"{name + ':':10} chance for α = {chance:.4f}, loss = {loss:.3f}")"""
            ),
            saw(
                "The untrained model gave `α` a tiny chance (about 1 in 180), so its loss is high. "
                "The trained one gave it about 0.12, so its loss is lower. `[-1:]` keeps only the last row of scores, which is the guess after `ἄνδρ`."
            ),
            q(
                "How does the chance turn into the loss?",
                "The bigger the chance for the right answer, the smaller the loss. Some real pairs:\n\n"
                "```\nchance 0.90  →  loss 0.105\nchance 0.50  →  loss 0.693\nchance 0.12  →  loss 2.120\nchance 0.01  →  loss 4.605\n```\n\n"
                "A chance of 1 (completely sure, and right) would give a loss of 0. "
                "In maths this is the **negative logarithm** of the chance. You don't need the maths here, only the pattern.\n\n"
                "👉 **Predict first:** is a chance of 0.5 closer to loss 0 or to loss 5?",
            ),
            code(
                """import math

for chance in [0.90, 0.50, 0.12, 0.01]:
    print(f"chance {chance:.2f}  →  loss {-math.log(chance):.3f}")"""
            ),
            saw(
                "The loss grows slowly at first, then fast as the chance gets small. So being confidently wrong is what costs the most."
            ),
            q(
                "Can I see how the chance turns into the loss?",
                "Draw the loss for every chance from 0 to 1, and mark the four real pairs from above.\n\n"
                "👉 **Predict first:** where will the curve be steepest: near chance 0, or near chance 1?",
            ),
            code(
                """import matplotlib.pyplot as plt
import numpy as np

from homer_ai.studio_style import ACCENT, INK, MUTED, studio_axes, studio_figure

chances = np.linspace(0.005, 1, 200)
fig = studio_figure("From chance to loss", "The less sure it is, the more it costs")
ax = studio_axes(fig)
ax.plot(chances, -np.log(chances), color=MUTED, linewidth=3)
for chance in [0.90, 0.50, 0.12, 0.01]:
    loss = -math.log(chance)
    ax.plot(chance, loss, "o", color=ACCENT, markersize=10)
    ax.text(
        chance + 0.025, loss + 0.1, f"{chance:.2f} → {loss:.2f}",
        color=INK, family="monospace", fontsize=11,
    )
ax.set_xlabel("chance the model gave the right answer")
ax.set_ylabel("loss")
plt.show()"""
            ),
            saw(
                "The curve is flat and low on the right (being sure and right costs little). On the left it climbs steeply, "
                "and that's where the model gave the right answer almost no chance."
            ),
            q(
                "What does the loss look like for a good guess and a bad guess?",
                "`F.cross_entropy` is PyTorch's function for the loss. Here there are 3 possible tokens, and the right answer is token 0. "
                "`torch.tensor(...)` makes a **tensor**, PyTorch's list of numbers.\n\n👉 **Predict first:** which guess gets the lower loss?",
            ),
            code(
                """import torch
import torch.nn.functional as F

right_answer = torch.tensor([0])
good_guess = torch.tensor([[3.0, 0.0, 0.0]])  # high score for token 0
bad_guess = torch.tensor([[0.0, 0.0, 3.0]])  # high score for token 2

print("good guess loss:", F.cross_entropy(good_guess, right_answer).item())
print("bad guess loss:", F.cross_entropy(bad_guess, right_answer).item())"""
            ),
            saw(
                "About 0.09 for the good guess and about 3.09 for the bad one. Lower loss, better guess."
            ),
            q(
                "Should we add up the losses of all chunks, or average them?",
                'Average them. A total grows with the amount of text, so a longer book would look "worse" even if every guess were just as good.\n\n'
                "👉 **Predict first:** what are the total and the average of these 4 chunk losses?",
            ),
            code(
                'chunk_losses = [2.0, 1.5, 1.8, 1.7]\nprint("total:", sum(chunk_losses))\nprint("average:", sum(chunk_losses) / len(chunk_losses))'
            ),
            saw(
                "Total 7.0, average 1.75. With 10× the chunks the total would be 10× bigger, but the average would stay at 1.75. That's why our training code prints the **average**."
            ),
            q(
                "What does training look like on a small text?",
                "This cell sets up a tiny training run on the first 5 lines of Book 1. It finishes in a second or two, and saves nothing. "
                "It does the four training steps from the overview for every chunk, and records the average loss of each epoch. "
                "Read `tiny_train` once, slowly: the next notebooks use this same function (a copy is kept in `src/homer_ai/lessons.py`, so they can import it).\n\nJust run it. It only sets things up, so nothing is printed.",
            ),
            code(TINY_SETUP),
            *exercise(
                "can you write the three training lines?",
                "Inside `tiny_train`, three lines do the learning: `zero_grad`, `backward`, `step`. Here's one model and one chunk of 24 characters, with the loss already measured. "
                "You write the three lines, then measure the loss on the same chunk again.\n\n"
                "👉 **Predict first:** after one single step, will the loss on this chunk be lower? By a lot, or by a little?",
                starter="""
            torch.manual_seed(0)
            step_model = HomerGreekLLM(vocab_size=tokenizer.vocab_size, embed_dim=32)
            step_optimizer = torch.optim.AdamW(step_model.parameters(), lr=0.01)
            step_inputs, step_targets = data[0:24], data[1:25]

            loss_before = F.cross_entropy(step_model(step_inputs), step_targets)
            print("loss before:", round(loss_before.item(), 3))

            # 👉 do: write the three training lines here, with step_optimizer and loss_before
            # 💡 hint: forget the old gradients, work out the new ones, take one step

            # 👉 do: measure the loss on the same chunk again, the way loss_before was measured
            loss_after = None
            """,
                solution="""
            torch.manual_seed(0)
            step_model = HomerGreekLLM(vocab_size=tokenizer.vocab_size, embed_dim=32)
            step_optimizer = torch.optim.AdamW(step_model.parameters(), lr=0.01)
            step_inputs, step_targets = data[0:24], data[1:25]

            loss_before = F.cross_entropy(step_model(step_inputs), step_targets)
            print("loss before:", round(loss_before.item(), 3))

            step_optimizer.zero_grad()
            loss_before.backward()
            step_optimizer.step()

            loss_after = F.cross_entropy(step_model(step_inputs), step_targets)
            print("loss after: ", round(loss_after.item(), 3))
            """,
                check="""
            check(
                "the loss after one training step",
                loss_after,
                test=lambda after: after < loss_before,
                expected=f"a loss below {loss_before.item():.3f}",
                coach="The same loss as before means the weights didn't change. Is one of "
                "the three lines missing? backward() only works out the gradients. It's "
                "step() that changes the weights.",
            )
            """,
                hint="Look inside `tiny_train`, right after the line that computes `loss`. The same three lines, with `step_optimizer` as the optimizer and `loss_before` as the loss.",
                answer="The loss drops from about 3.99 to about 3.81. One step is one small nudge to every weight, like the single nudge of `w` in [01_03](../01_foundations/01_03_how_a_program_learns.ipynb). "
                "Training is this step, thousands of times.",
            ),
            q(
                "Does the loss go down?",
                "👉 **Predict first:** after 60 epochs, will the last loss be lower than the first?",
            ),
            code(
                'losses = tiny_train(seed=42, epochs=60)\nprint("first epoch:", round(losses[0], 3))\nprint("last epoch:", round(losses[-1], 3))'
            ),
            saw(
                "About 3.88 at the start and 0.71 at the end. So the model is learning the 5 lines.\n\n"
                "But why 3.88 and not 4.9? The 5-line text has only **52** different characters, not 137, and random guessing among 52 scores about 3.95. "
                "And the first number is already the average over the whole first epoch, so the model has learned a little by then."
            ),
            q(
                "Is it enough to look at a few snapshots?",
                "No. Printing only every so often is like judging a whole football match from 4 photos. "
                "Let's keep only every 15th loss and compare that with the full list.\n\n👉 **Predict first:** will the full list look as smooth as the snapshots?",
            ),
            code(
                'print("every 15th epoch:", [round(x, 2) for x in losses[14::15]])\nprint()\nprint("every epoch:", [round(x, 2) for x in losses])'
            ),
            saw(
                "The snapshots go smoothly down, but the full list wobbles up and down along the way.\n\n"
                "On a longer run the snapshots hide even more. In [05_02](05_02_learning_rate.ipynb) you'll train for 400 epochs. Printed every 100 epochs, that run reads `0.46 → 0.15 → 0.21 → 0.48`: four calm-looking numbers. "
                "Count every epoch, and the loss jumps back up 14 times in between."
            ),
            q(
                "Why do I get exactly the same numbers every time I run it?",
                'The model starts with random weights. The **seed** fixes the starting point of that randomness, so the "random" numbers are the same every run.\n\n'
                "👉 **Predict first:** two runs with seed 42, and one with seed 7. Which ones match?",
            ),
            code(
                'run_a = tiny_train(seed=42, epochs=20)\nrun_b = tiny_train(seed=42, epochs=20)\nrun_c = tiny_train(seed=7, epochs=20)\n\nprint("seed 42 twice, same?", run_a == run_b)\nprint("seed 42 vs seed 7, same?", run_a == run_c)'
            ),
            saw(
                "The same seed gives exactly the same numbers, and a different seed doesn't. That's handy when you change the code: same seed, same numbers, nothing broken."
            ),
            md(
                "## 5. Full example: a loss curve\n\nA 100-epoch tiny run, printed every 10 epochs.\n\n👉 **Predict first:** will every printed loss be lower than the one before?"
            ),
            code(
                'curve = tiny_train(seed=42, epochs=100)\nfor epoch in range(9, 100, 10):\n    print(f"epoch {epoch + 1:3} | average loss {curve[epoch]:.3f}")'
            ),
            saw(
                "The trend goes down (about 1.60 → 0.46), but not at every step. Epoch 60 and epoch 90 are a little higher than the epoch before."
            ),
            q(
                "Can I see the whole curve?",
                "The numbers above show every 10th epoch. Here's every epoch, with the printed ones marked.\n\n"
                "👉 **Predict first:** will the line fall smoothly, or wobble?",
            ),
            code(
                """import matplotlib.pyplot as plt

from homer_ai.studio_style import ACCENT, MUTED, studio_axes, studio_figure

fig = studio_figure(
    "A tiny training run, all 100 epochs",
    f"The loss falls from {curve[0]:.2f} to {curve[-1]:.2f}, with wobbles",
)
ax = studio_axes(fig)
ax.plot(range(1, 101), curve, color=MUTED, linewidth=2)
printed = list(range(10, 101, 10))
ax.plot(printed, [curve[e - 1] for e in printed], "o", color=ACCENT, markersize=8)
ax.set_xlabel("epoch")
ax.set_ylabel("average loss")
plt.show()"""
            ),
            saw(
                "The grey line wobbles on its way down, and the orange dots (the epochs we printed) hide those wobbles. "
                "Same warning as before: look at every epoch, not just a few."
            ),
            *questions(
                [
                    (
                        "The loss suddenly jumped up during training. Is something broken?",
                        "Not necessarily. Usually the steps are too big, and the model overshoots. That's the next notebook, [05_02](05_02_learning_rate.ipynb).",
                    ),
                    (
                        "A log says the loss is 1966. Is that terrible?",
                        "It depends what was printed. As a **total** over all the chunks of Book 1, 1966 is fine: it's an average of about 2.40 per chunk. So always check whether a number is a total or an average.",
                    ),
                    (
                        "Why print the loss at all?",
                        "It's the only way to see whether training is working. A curve that goes down means the model is learning.",
                    ),
                ]
            ),
            recap(
                [
                    "The loss is a mistake score, and lower is better (random guessing on our 137 tokens: about 4.9).",
                    "Report the average loss, and look at it often. Rare snapshots hide spikes.",
                    "The same seed gives the same numbers, so runs can be compared.",
                ],
                "[05_02: Learning rate](05_02_learning_rate.ipynb): why does the loss jump up?",
                can=[
                    "turn a chance into a loss, and say what loss random guessing gives",
                    "write the three lines of a training step: `zero_grad`, `backward`, `step`",
                    "read a loss curve, epoch by epoch",
                ],
            ),
        ],
    )

    save(
        "05_training_on_the_odyssey/05_02_learning_rate.ipynb",
        [
            title(
                "05_02 Learning rate",
                "section 5.2 (training); the learning-rate schedule is in appendix D",
            ),
            where("model learns", ["src/homer_ai/train_homer.py"]),
            learn(
                [
                    "What the learning rate is: the size of each step",
                    "Why steps that are too big make the loss jump up",
                    "The trade-off between fast and smooth",
                    "How a learning-rate schedule gives you both",
                ]
            ),
            words(
                [
                    (
                        "learning rate",
                        "how big each nudge to the weights is",
                        "`lr=0.01`",
                    ),
                    (
                        "overshoot",
                        "a step so big that it goes past the lowest point and lands higher up the other side",
                        "the average loss goes 0.25 → 0.37 → 0.80 → 1.14 in three epochs",
                    ),
                    (
                        "trade-off",
                        "you gain one thing but lose another",
                        "smooth, but slow",
                    ),
                    (
                        "learning-rate schedule",
                        "a plan that changes the learning rate during training: big steps first, small steps later",
                        "cosine",
                    ),
                    (
                        "cosine schedule",
                        "a schedule that slides the learning rate smoothly from its start value down to (almost) 0: slow at first, then faster, then slow again near the end",
                        "0.01 → 0.005 → 0",
                    ),
                    (
                        "`eta_min`",
                        "the lowest the learning rate is allowed to go; η (eta) is the Greek letter maths papers use for the learning rate",
                        "`eta_min=0.001`",
                    ),
                ]
            ),
            *background(
                "At the stage **model learns**. In [05_01](05_01_loss_and_epochs.ipynb) we learned to read the loss, and saw that it wobbles on its way down.",
                "We want to know why the loss jumps, and how to stop it. The answer is the size of the training steps, the **learning rate**. "
                "Choose it well and the model learns smoothly. Choose it badly and it keeps falling back.",
            ),
            q(
                "What does the learning rate actually do?",
                "Imagine walking down into a valley in fog. You can't see the bottom, but you can feel which way is downhill. "
                "How far each step goes is set by the **learning rate**, together with how steep the ground is where you stand. "
                "That steepness is called the slope: steep ground means a bigger step, flat ground a smaller one. "
                "In our code it's `lr=0.01`, where the optimizer is created in `main_char()` (`train_homer.py`).",
            ),
            q(
                "So why does the loss jump back up?",
                "Near the bottom of the valley, a step that's too long carries you past the lowest point and up the other side. "
                "That's an **overshoot**. Then you have to walk back down. In the loss curve it looks like a sudden spike, and then a slower recovery. You'll count the spikes of a real run in section 5.",
            ),
            q(
                "Then why not just take small steps?",
                "Small steps don't overshoot, but they're slow. In the same number of epochs you get less far down. "
                "That's a **trade-off**, and the runs in section 5 put numbers on it.",
            ),
            q(
                "Can we change the learning rate during training?",
                "Yes. Take big steps at the start to learn fast, and small steps near the end so you don't overshoot. "
                "That plan is a **learning-rate schedule**. A popular one, the cosine schedule, slides the learning rate from its start value down to 0: slow at first, then faster, then slow again near the end. Our training code uses it, as `CosineAnnealingLR` in `train()`.",
            ),
            step_intro(),
            q(
                "What happens with different step sizes, in the simplest example?",
                "Picture a valley with its bottom at 0. Each step moves `x` downhill by `learning_rate × slope`, where the slope here is `2 × x`. We start at `x = 1` and take 5 steps.\n\n"
                "```\n"
                "  \\                  /\n"
                "   \\                ●      start: x = 1\n"
                "    \\              /\n"
                "     \\_____0______/        bottom: x = 0\n"
                "```\n\n"
                "One step with learning rate 0.1: slope = 2 × 1 = 2, step = 0.1 × 2 = 0.2, so `x` goes from 1 to **0.8**.\n\n"
                "👉 **Predict first:** with learning rates 0.1, 0.9 and 1.1, which one reaches the bottom smoothly, which one jumps from side to side, and which one climbs out?",
            ),
            code(
                """for learning_rate in [0.1, 0.9, 1.1]:
    x = 1.0
    path = []
    for _step in range(5):
        x = x - learning_rate * 2 * x
        path.append(round(x, 3))
    print(f"lr {learning_rate}: {path}")"""
            ),
            saw(
                "Three very different walks:\n\n- **0.1** walks steadily towards 0, but slowly.\n"
                "- **0.9** jumps from side to side (+, −, +, −) across the bottom. It's overshooting, but still getting closer.\n"
                "- **1.1** lands further away each time. The steps are so big that it climbs out of the valley."
            ),
            q(
                "Can I see the steps in the valley?",
                "Draw the valley as a curve, with the lowest point in the middle, and mark where each step lands. "
                "We draw the same three learning rates as above (0.1, 0.9 and 1.1), plus a good one, 0.4.\n\n"
                "👉 **Predict first:** which one will still be far from the bottom after 5 steps, and which one gets there fastest?",
            ),
            code(
                """import matplotlib.pyplot as plt
import numpy as np

from homer_ai.studio_style import (
    ACCENT,
    BAD,
    GOOD,
    INK,
    MUTED,
    studio_axes,
    studio_figure,
)

runs = [
    ("small: 0.1", 0.1, ACCENT),
    ("good: 0.4", 0.4, GOOD),
    ("jumpy: 0.9", 0.9, MUTED),
    ("too big: 1.1", 1.1, BAD),
]
valley = np.linspace(-2.7, 2.7, 200)

fig = studio_figure("Steps in the valley", "Same start, four step sizes", size=(12, 5))
for i, (label, learning_rate, colour) in enumerate(runs):
    ax = studio_axes(fig, (0.06 + i * 0.235, 0.12, 0.2, 0.65))
    ax.plot(valley, valley**2, color=INK, linewidth=1.5, alpha=0.5)
    x, xs = 1.0, [1.0]
    for _step in range(5):
        x = x - learning_rate * 2 * x
        xs.append(x)
    ax.plot(xs, [v**2 for v in xs], "o-", color=colour, linewidth=2, markersize=6)
    ax.set_title(label, color=colour, fontsize=13, fontweight="bold")
    ax.set_xlim(-2.7, 2.7)
    ax.set_ylim(-0.3, 7.5)
    ax.set_xlabel("x")
    if i == 0:
        ax.set_ylabel("loss (height of the valley)")
    else:
        ax.set_yticklabels([])
plt.show()"""
            ),
            saw(
                "The small steps (0.1) creep down and are still far from the bottom after 5 steps. "
                "The good ones (0.4) are almost there. The jumpy ones (0.9) cross from side to side but still get closer. And the too-big ones (1.1) leap from side to side and land **higher** each time."
            ),
            q(
                "How does the cosine schedule change the learning rate?",
                "This little function gives the learning rate at any epoch (`math.cos` is the cosine, `math.pi` is π).\n\n👉 **Predict first:** what will the learning rate be halfway through, at epoch 200 of 400?",
            ),
            code(
                """import math


def cosine_lr(epoch, epochs, start_lr=0.01, eta_min=0.0):
    progress = epoch / epochs  # 0.0 at the start, 1.0 at the end
    return eta_min + (start_lr - eta_min) * (1 + math.cos(math.pi * progress)) / 2


for epoch in [0, 100, 200, 300, 400]:
    print(f"epoch {epoch:3}: lr {cosine_lr(epoch, 400):.5f}")"""
            ),
            saw(
                "It starts at 0.01, is exactly half (0.005) in the middle, and reaches 0 at the end.\n\n"
                "There's a catch, though. Near the end the steps are almost zero, so the model just shuffles its feet. A training run can spend its last epochs learning nearly nothing. You'll see that too in section 5."
            ),
            q(
                "Can I see the cosine schedule?",
                "Draw `cosine_lr` for every epoch of a 400-epoch run, next to a constant learning rate.\n\n"
                "👉 **Predict first:** will the cosine line drop straight down, or bend?",
            ),
            code(
                """import matplotlib.pyplot as plt

from homer_ai.studio_style import ACCENT, INK, MUTED, studio_axes, studio_figure

epochs = list(range(401))
rates = [cosine_lr(e, 400) for e in epochs]

fig = studio_figure("The cosine schedule", "Big steps first, tiny steps at the end")
ax = studio_axes(fig)
ax.plot(epochs, [0.01] * 401, color=MUTED, linestyle="--", linewidth=1.5)
ax.text(395, 0.0103, "constant 0.01", color=MUTED, ha="right", family="monospace")
ax.plot(epochs, rates, color=ACCENT, linewidth=3)
ax.plot(200, rates[200], "o", color=INK, markersize=9)
ax.text(
    208, rates[200] + 0.0004, f"epoch 200: {rates[200]:.3f}",
    color=INK, family="monospace",
)
ax.set_xlabel("epoch")
ax.set_ylabel("learning rate")
plt.show()"""
            ),
            saw(
                "The cosine line starts at 0.01, bends down through exactly half (0.005) at epoch 200, and flattens out near 0. "
                "It's gentle at both ends and steepest in the middle."
            ),
            q(
                "How do we stop the steps from shrinking to nothing too early?",
                "Two ways. Either set a **floor** (`eta_min`), so the learning rate never goes below it, or use more epochs, so the slide down takes longer.\n\n"
                "👉 **Predict first:** with 800 epochs, what's the learning rate at epoch 400?",
            ),
            code(
                'print("with a floor, end:", cosine_lr(400, 400, eta_min=0.001))\nprint("800 epochs, epoch 400:", round(cosine_lr(400, 800), 5))'
            ),
            saw(
                "The floor keeps it at 0.001. And with 800 epochs, the learning rate is still 0.005 at epoch 400. Our `train()` takes the second road: 800 epochs, and no floor."
            ),
            md(
                "## 5. Full example: four real runs\n\nNow the real thing, on the 5-line practice text of [05_01](05_01_loss_and_epochs.ipynb). "
                "The loop is the `tiny_train` you built there. It's kept in `src/homer_ai/lessons.py`, so we import it here. Asking for `cosine=True` adds two lines to it. "
                "`CosineAnnealingLR` is PyTorch's ready-made cosine schedule, and `scheduler.step()` moves it one notch along:\n\n"
                "```python\n"
                "optimizer = torch.optim.AdamW(model.parameters(), lr=lr)\n"
                "scheduler = CosineAnnealingLR(optimizer, T_max=epochs)  # new: made once\n"
                "for epoch in range(epochs):\n"
                "    for i in range(...):   # every chunk: guess, loss, nudge (as before)\n"
                "        ...\n"
                "    scheduler.step()       # new: once per epoch, one notch down the curve\n"
                "```\n\n"
                "`train()` in `train_homer.py` has the same two lines.\n\n"
                "We change one thing at a time. All four runs use seed 42:\n\n"
                "```\n"
                "constant 0.01     the starting point: 400 epochs, always lr 0.01\n"
                "constant 0.003    only the learning rate changes: smaller steps\n"
                "cosine, 400       only the schedule changes: lr slides from 0.01 to 0\n"
                "cosine, 800       the schedule, and twice the epochs\n"
                "```\n\n"
                "A **jump** is an epoch whose average loss is more than 0.1 above the epoch before. The cell takes about 20 seconds and saves nothing.\n\n"
                "👉 **Predict first:** which run will have the most jumps? And which one ends lowest?"
            ),
            code(LR_RUNS),
            saw(
                "| Run | Epoch 100 | Epoch 200 | Final | Jumps | Last jump |\n|---|---|---|---|---|---|\n"
                "| constant 0.01 | 0.463 | 0.151 | 0.475 | 14 | epoch 400 |\n"
                "| constant 0.003 | 0.538 | 0.282 | 0.135 | 1 | epoch 259 |\n"
                "| cosine, 400 | 0.384 | 0.136 | 0.067 | 3 | epoch 154 |\n"
                "| cosine, 800 | 0.399 | 0.138 | **0.008** | 10 | epoch 320 |\n\n"
                "- **Constant 0.01** learns fast, then keeps overshooting: 14 jumps. Look at epochs 168 to 171, where the loss goes 0.25 → 0.37 → 0.80 → 1.14. "
                "And the last jump lands on the very last epoch (0.128 → 0.236 → 0.475), so the run ends three times worse than it was at epoch 200. Where it stops is luck.\n"
                "- **Constant 0.003** is calm (1 jump) but slow. At epoch 200 it's at 0.28, where the bigger steps had reached 0.15.\n"
                "- **Cosine, 400** is fast early and calm late: no jump after epoch 154, and it ends at 0.067. That's the schedule doing its job.\n"
                "- **Cosine, 800** ends lowest of all, at 0.008. It still jumps while the learning rate is high (the last one is at epoch 320), then settles.\n\n"
                "And there's the catch from above: in the 400-epoch cosine run, the last 50 epochs only move the loss from 0.069 to 0.067. The steps are too small to matter by then.\n\n"
                "One warning before you celebrate that 0.008. It's measured on the same 5 lines the model read 800 times. Whether that's learning is the question of [05_03](05_03_memorising_vs_understanding.ipynb)."
            ),
            *questions(
                [
                    (
                        "The small constant rate ended lower than the big one (0.135 against 0.475). Is smaller better, then?",
                        "No, and this is why you look at more than the last number. Smaller steps get less far in the same time: at epoch 200 the 0.003 run is at 0.28 and the 0.01 run at 0.15. "
                        "The 0.01 run only loses at the end because its last jump happened to land on epoch 400. Stop it at epoch 395 and it wins (0.085).",
                    ),
                    (
                        "Is there a correct learning rate?",
                        "No single one. You find a good value by trying a few and comparing them, as in the four runs above. That's called **tuning**.",
                    ),
                ]
            ),
            recap(
                [
                    "The learning rate sets the step size: too big overshoots, too small is slow.",
                    "A schedule gives you both: big steps first, small steps later.",
                    "Don't let the steps shrink to nothing too early. Use a floor (`eta_min`) or more epochs.",
                ],
                "[05_03: Memorising vs understanding](05_03_memorising_vs_understanding.ipynb): is a low loss really good?",
                can=[
                    "spot a learning rate that's too big from the loss curve",
                    "add a cosine schedule to a training loop",
                ],
            ),
        ],
    )

    # =====================================================================
    # 05_training_on_the_odyssey/05_03_memorising_vs_understanding.ipynb
    # =====================================================================
    save(
        "05_training_on_the_odyssey/05_03_memorising_vs_understanding.ipynb",
        [
            title(
                "05_03 Memorising vs understanding",
                "sections 5.1.3 (training and validation losses) and 5.2",
            ),
            where("model learns", ["src/homer_ai/train_homer.py"]),
            learn(
                [
                    "Why a low loss can mean the model only memorised the text",
                    "What happens when the model meets text it has never seen",
                    "How to test a model on hidden text: the validation set",
                    "How to cut the data into a training part and a validation part",
                ]
            ),
            words(
                [
                    (
                        "training loss",
                        "the loss on the text the model learns from",
                        "0.008 on the 5 lines, after 800 epochs",
                    ),
                    (
                        "overfitting",
                        "memorising the training text instead of learning its patterns",
                        "a student who memorised one exam",
                    ),
                    (
                        "generalisation",
                        "doing well on text the model has never seen",
                        "writing good Greek after a new word",
                    ),
                    (
                        "OOV",
                        "out of vocabulary: a character the tokenizer has no number for",
                        "`Ὀ` for the 5-line model",
                    ),
                    (
                        "training set",
                        "the part of the text the model learns from",
                        "the first 90%",
                    ),
                    (
                        "validation set",
                        "text kept hidden from training, used only to test the model",
                        "the last 10%",
                    ),
                ]
            ),
            *background(
                "At the stage **model learns**. In [05_02](05_02_learning_rate.ipynb) we tuned the learning rate and got the average loss down to **0.008**. That was the best run.",
                "But that 0.008 was measured on the same 5 lines the model trained on, 800 times. A low loss there may only mean the model memorised them. "
                "And if our Odyssey model only memorises, it can't write anything new, which is the whole point of an LLM.",
            ),
            q(
                "How small is the practice text?",
                "Five lines, 234 characters. The best run of [05_02](05_02_learning_rate.ipynb) (cosine schedule, 800 epochs) ended with an average loss of 0.008. To get there, the model read those same 5 lines 800 times.",
            ),
            q(
                "Is a low loss on 234 characters a good thing?",
                'No. Imagine a student who gets the same 5-question test 800 times. They\'ll score 100%, but they may just have memorised "answer 3 is B" '
                "without understanding anything. When a model memorises like that, it's called **overfitting**.",
            ),
            q(
                "What does that model do with text it has never seen?",
                "Let's train it again and ask. We give it two starting words. A starting text like that is called a **prompt**: the text you give a model to continue. "
                "It then writes by guessing one character, adding it to the end, and guessing again (chapter 6 explains this in full; `write` with `top_k=1` always takes the most likely character).\n\n"
                "```\nthe 5 lines start:   ἄνδρα μοι ἔννεπε, μοῦσα, πολύτροπον, ὃς μάλα πολλὰ\n"
                "prompt 1:            ἄνδρα      <- the first word of the 5 lines\n"
                "prompt 2:            Ὀδυσσεύς   <- Odysseus: not in the 5 lines at all\n```\n\n"
                "`tiny_model` is the training run of 05_02 (the same seed and settings), and it hands back the model too. It takes about 7 seconds and saves nothing. "
                "A line break in the output is shown as `⏎`.\n\n"
                "👉 **Predict first:** what will the model write after `ἄνδρα`? And after a word it has never read?",
            ),
            code(
                """from homer_ai.lessons import tiny_model, write

# the best run of 05_02: 5 lines, cosine schedule, 800 epochs
model, five_line_tokenizer, losses = tiny_model(seed=42, epochs=800, cosine=True)
print("final training loss:", round(losses[-1], 3))
for prompt in ["ἄνδρα", "Ὀδυσσεύς"]:
    print(write(model, five_line_tokenizer, prompt, 40, top_k=1))"""
            ),
            saw(
                "```\nἄνδρα μοι ἔννεπε, μοῦσα, α, μοῦσαμοῦσαῦσαῦσαῦ\n"
                "<|unk|>δυσσεύς,ολα, α, μυ, μλμλμλμλμλμοῦσα α α α α ν γ\n```\n\n"
                "- After `ἄνδρα` the model **recites** line 1 word for word (`μοι ἔννεπε, μοῦσα,`). Then it loses its place and goes round in circles (`μοῦσαμοῦσαῦσαῦσα`).\n"
                "- `Ὀδυσσεύς` isn't in the 5 lines, and the model writes gibberish. Its first letter, `Ὀ`, isn't even in the vocabulary. It's **OOV**, so the tokenizer turned it into `<|unk|>` ([02_02](../02_text_to_numbers/02_02_special_tokens.ipynb)).\n\n"
                "So the loss of 0.008 didn't mean the model knows Greek. It means it knows these 5 lines by heart."
            ),
            q(
                "So how do we measure real learning?",
                "Keep some text hidden. The model trains on one part, the **training set**, and is only tested on the other part, the validation set. "
                "Think of an exam with questions it has never seen.\n\n"
                "```\ntraining loss ↓   validation loss ↓   → it's learning real patterns\n"
                "training loss ↓   validation loss ↑   → it's memorising (overfitting)\n```",
            ),
            q(
                "Should we train on the whole book, not just 5 lines?",
                "Yes. More text is the best cure for memorising. With 5 lines, memorising is all the model can do. With the whole of Book 1 (444 lines, 19,701 characters), "
                "the same patterns show up again and again in different places, so learning the pattern becomes the easiest way to lower the loss.",
            ),
            step_intro(),
            q(
                "How small is the 5-line text, and is `Ὀ` in it?",
                "`in` checks whether something is inside a text. 👉 **Predict first:** is `Ὀ` in the 5 lines? In Book 1?",
            ),
            code(
                r"""from homer_ai.train_homer import BOOKS_DIR, load_books

book1 = load_books(BOOKS_DIR)[0]
five_lines = "\n".join(book1.splitlines()[:5])

print("characters in 5 lines:", len(five_lines))
print("Ὀ in the 5 lines?", "Ὀ" in five_lines)
print("Ὀ in Book 1?", "Ὀ" in book1)"""
            ),
            saw(
                "So, 234 characters, and no `Ὀ` among them. That's why the 5-line model had never heard of it: it was out of vocabulary, and the tokenizer could only give it `<|unk|>`. Give the model all of Book 1 and the problem goes away."
            ),
            q(
                "How do I cut the data into two parts?",
                "Pick a cut position `n` at the 80% mark. Everything before `n` is the first part, and everything from `n` on is the second.\n\n👉 **Predict first:** with 10 numbers and an 80% split, what are the two parts?",
            ),
            code(
                'numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]\nn = int(0.8 * len(numbers))  # the 80% mark\nprint("n:", n)\nprint("first part:", numbers[:n])\nprint("second part:", numbers[n:])'
            ),
            saw(
                "`n` is 8, so the first part has 8 numbers and the second part has 2. Nothing is lost and nothing is repeated."
            ),
            q(
                "Why take the last block as the test, and not random letters?",
                'Because text has an order. If the test letters were picked at random, each one would sit right next to training letters, so the "unseen" text would hardly be unseen. '
                "A whole block at the end is a real passage the model has never read.\n\n👉 **Predict first:** what's the very last line of Book 1?",
            ),
            code("print(book1.splitlines()[-1])"),
            saw(
                "That's the last line of Book 1. With the split, the model never trains on it. It's only tested on it."
            ),
            md(
                "## 5. Full example: splitting Book 1\n\n`load_data` gives us all of Book 1 as numbers. We keep the first 90% for training and the last 10% for validation. "
                "`.tolist()` turns a tensor into a plain Python list, so we can decode it.\n\n"
                "```\nBook 1:  ἄνδρα μοι ἔννεπε … … … … … … … … … … … … … | … ὁδὸν τὴν πέφραδʼ Ἀθήνη.\n"
                "         └────────────── training (first 90%) ──────────┘ └── validation (last 10%) ──┘\n```\n\n"
                "👉 **Predict first:** Book 1 has 19,701 characters. How many go to validation?"
            ),
            code(
                """from homer_ai.train_homer import load_data

data, tokenizer, text = load_data(BOOKS_DIR)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print("all data:", len(data))
print("training:", len(train_data))
print("validation:", len(val_data))
print("together:", len(train_data) + len(val_data))
print()
print("start of the validation text:")
print(tokenizer.decode(val_data[:80].tolist()))"""
            ),
            saw(
                "17,730 characters for training and 1,971 for validation, which add up to all 19,701. "
                "The validation text is about the last 44 lines of Book 1.\n\n"
                "The small model's `train()` doesn't make this split: it trains on all of Book 1. Doing the split for real, with `evaluate()` and a validation loss, is [05_05](05_05_validation.ipynb)."
            ),
            q(
                "Can I see the split?",
                "One long bar is Book 1. The orange end is the part the model never trains on.\n\n"
                "👉 **Predict first:** will the orange part look much smaller than the grey part?",
            ),
            code(
                """import matplotlib.pyplot as plt

from homer_ai.studio_style import ACCENT, MUTED, studio_axes, studio_figure, text_on

fig = studio_figure(
    "Book 1, cut in two",
    f"{len(train_data):,} characters to learn from, {len(val_data):,} kept hidden",
    size=(9, 5),
)
ax = studio_axes(fig, (0.07, 0.2, 0.86, 0.5))
ax.barh([0], [len(train_data)], color=MUTED, height=0.6)
ax.barh([0], [len(val_data)], left=len(train_data), color=ACCENT, height=0.6)
ax.text(
    len(train_data) / 2, 0, f"training: first {len(train_data) / len(data):.0%}",
    ha="center", va="center", color=text_on(1, 1), fontsize=14, fontweight="bold",
)
ax.text(
    len(data), 0.38, f"validation: last {len(val_data) / len(data):.0%}",
    ha="right", va="bottom", color=ACCENT, fontsize=13, fontweight="bold",
)
ax.set_xlim(0, len(data))
ax.set_ylim(-0.4, 0.8)
ax.set_yticks([])
ax.grid(False)
ax.set_xlabel("position in the text (characters)")
plt.show()"""
            ),
            saw(
                "The orange validation part is a small slice at the end, about one character in ten. "
                "It's still big enough to be a fair test, and it's text the model has never read."
            ),
            *questions(
                [
                    (
                        "Is 10% enough for the validation set?",
                        "For Book 1, 10% is 1,971 characters. That's enough to see whether the model learned, but it's small, so the number wobbles a little from run to run. "
                        "With more books, the validation set grows and the number gets steadier.",
                    ),
                    (
                        "Why did the model go round in circles (`μοῦσαμοῦσαῦσαῦσα`)?",
                        'It learned "what comes next" only for paths it had seen. Once it fell off the memorised path, it went round in circles.',
                    ),
                ]
            ),
            recap(
                [
                    "A low training loss can mean memorising (overfitting), not understanding.",
                    "Text the model has never seen shows the truth: our 5-line model recited, looped, or wrote gibberish.",
                    "So keep a block at the end hidden (the validation set), and measure on it.",
                ],
                "[05_04: Running fair experiments](05_04_running_fair_experiments.ipynb): how to change one thing and know what it did.",
                can=[
                    "test a model on text it has never seen",
                    "tell memorising from learning by what the model writes",
                ],
            ),
        ],
    )

    save(
        "05_training_on_the_odyssey/05_04_running_fair_experiments.ipynb",
        [
            title("05_04 Running fair experiments", None),
            where("model learns", ["src/homer_ai/train_homer.py"]),
            learn(
                [
                    'How to save a baseline, a saved "before" result',
                    "How to compare before and after, and spot a bug or a refactor",
                    "Why the seed stays fixed, and why you change one thing at a time",
                    "How to time a run, and how to keep a log of results",
                ]
            ),
            words(
                [
                    (
                        "baseline",
                        'a saved "before" result, kept so every change can be compared with it',
                        "`baseline.txt`",
                    ),
                    (
                        "standard output (stdout)",
                        "the normal channel a program prints to; `print()` uses it",
                        '`print("Epoch 100/400 …")`',
                    ),
                    (
                        "standard error (stderr)",
                        "a second channel, used for warnings and errors",
                        "a NumPy warning",
                    ),
                    (
                        "redirect (`>`)",
                        "in the terminal, `>` sends stdout into a file instead of the screen",
                        "`… > baseline.txt`",
                    ),
                    (
                        "`diff`",
                        "a terminal command that compares two files and prints only the lines that differ; no output means identical",
                        "`diff baseline.txt after.txt`",
                    ),
                    (
                        "refactor",
                        "reorganising code without changing what it does",
                        "moving the model into `model.py`",
                    ),
                    (
                        "results log",
                        "a table where you write down each run's settings and result",
                        "run, what changed, final loss",
                    ),
                ]
            ),
            *background(),
            q(
                "I changed something and the loss moved. How do I know my change did it?",
                "Only by comparing it with a run where nothing else changed. That's a small experiment, and it comes down to four habits:\n\n"
                "```\n"
                '1. save a BASELINE        the "before": one run, written to a file\n'
                "2. fix the SEED           so luck is the same in both runs\n"
                "3. change ONE thing       so you know what to blame\n"
                "4. COMPARE and LOG        diff the files; write the result in a table\n"
                "```\n\n"
                "You've already done it once. The four runs of [05_02](05_02_learning_rate.ipynb) jumped around, and they only made sense because each one changed one thing, with the same seed, and we kept every number.\n\n"
                "*In my day job I build [evaluation dashboards](https://icodenet.github.io/eval-dashboards/docs/) for AI agents. The habit starts right here: one baseline, change one thing, compare.*",
            ),
            step_intro(),
            q(
                "What's a baseline, and how do I save one?",
                'A **baseline** is a saved "before". Before you change anything, run the training once and keep what it printed in a file:\n\n'
                "```\nuv run python src/homer_ai/train_homer.py > baseline.txt\n```\n\n"
                "(Don't run it yet. It takes minutes and it writes a model file. The next question says what to do first.)\n\n"
                "`>` is a **redirect**. It sends the normal output (stdout) into the file and not to the screen. "
                "Warnings travel on a second channel, stderr, which `>` doesn't catch, so they still show up on the screen. "
                "That's why the file holds only our clean lines.\n\n"
                "👉 **Predict first:** if a program prints one normal line and one warning, which one will `result.stdout` hold?",
            ),
            code(
                """import subprocess
import sys

script = 'import sys\\nprint("normal output")\\nprint("a warning", file=sys.stderr)'
result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
print("stdout (what > catches):", repr(result.stdout))
print("stderr (what > misses): ", repr(result.stderr))"""
            ),
            saw(
                "The two lines went to two different channels. `subprocess.run` starts the small script, and `capture_output=True` catches both channels so we can look at them. "
                "(`repr()` shows the hidden `\\n` at the end of each line, as in [02_04](../02_text_to_numbers/02_04_sliding_window.ipynb).)"
            ),
            q(
                "What does a baseline file look like?",
                'Eighteen lines: one "Training on…" line, sixteen loss lines (one every 50 epochs, for 800 epochs), and one "Saved…" line. '
                "The loss is the average per chunk, and `lr` is the learning rate sliding down its cosine curve ([05_02](05_02_learning_rate.ipynb)):\n\n"
                "```\n"
                "baseline.txt\n"
                " 1  Training on 19701 Greek characters (Vocab size: 137)...\n"
                " 2  Epoch 50/800 | Avg loss: … | lr: 0.00990\n"
                " 3  Epoch 100/800 | Avg loss: … | lr: 0.00962\n"
                "    …\n"
                "17  Epoch 800/800 | Avg loss: 2.3817 | lr: 0.00000\n"
                "18  Saved trained model to …/src/homer_ai/homer_model.pt\n"
                "```\n\n"
                "(The `…` are losses we haven't copied here. The last one, 2.3817, is what our run printed. Yours should match on the same machine and PyTorch version, and may differ in the last digits on another.)\n\n"
                "**Two things to know before you run that command.** It takes about 8 to 10 minutes on a laptop. "
                "And line 18 is real: the run **writes `src/homer_ai/homer_model.pt`, and replaces the one that's there** if you trained before. "
                "The script has no option for another file name. So if you want to keep your current model, copy it first:\n\n"
                "```\ncp src/homer_ai/homer_model.pt my_model_before.pt\n```\n\n"
                "(The ready-made files in `models/` are never touched. After the run, the notebooks say `Using your own model`.)\n\n"
                "You don't need to run it now. Here's the same idea in a few seconds. `tiny_train` is the small training run you built in [05_01](05_01_loss_and_epochs.ipynb) (5 lines of Book 1, 24-character chunks). It's kept in `src/homer_ai/lessons.py`, so we import it, "
                'and `to_log` turns its losses into lines shaped like the real printout. It saves nothing. The "Saved" line is just text.\n\n'
                "👉 **Predict first:** with 100 epochs, printing every 25, how many lines will the log have?",
            ),
            code(
                '''import torch
import torch.nn.functional as F

from homer_ai.lessons import practice_data, tiny_train
from homer_ai.model import HomerGreekLLM

# the practice text of 05_01: the first 5 lines of Book 1, as text and as numbers
practice_text, tokenizer, data = practice_data()


def to_log(losses, every=25):
    """Turn a list of average losses into lines like the real printout."""
    lines = [f"Training on {len(data)} Greek characters..."]
    for epoch in range(every, len(losses) + 1, every):
        lines.append(f"Epoch {epoch}/{len(losses)} | Avg loss: {losses[epoch - 1]:.4f}")
    lines.append("Saved trained model to homer_model.pt")
    return lines


baseline_losses = tiny_train(seed=42, epochs=100)
baseline = to_log(baseline_losses)
print(len(baseline), "lines")
for line in baseline:
    print(line)'''
            ),
            saw(
                "6 lines: 1 + 4 + 1, the same shape as the real baseline, only shorter. "
                "Here a **list of lines** in Python plays the role of `baseline.txt`."
            ),
            q(
                "How do I compare the new run with the baseline?",
                "In the terminal, with **`diff`**:\n\n"
                "```\nuv run python src/homer_ai/train_homer.py > after.txt\ndiff baseline.txt after.txt\n```\n\n"
                "`diff` prints only the lines that differ. So no output means the two files are identical. "
                "That's exactly what you want after a **refactor** (reorganising code without changing what it does). "
                "Say you move the model into its own file, `model.py`, and `diff` prints nothing: then nothing broke. "
                "It only works because of the fixed seed. Without it, even identical code would give different numbers.\n\n"
                "In Python, you compare two lists of lines with `==`, and the standard-library module `difflib` prints a readable diff (`unified_diff` marks removed lines with `-` and added lines with `+`).\n\n"
                "👉 **Predict first:** we run the same code again, with the same seed. What will `==` say, and what will `unified_diff` print?",
            ),
            code(
                """import difflib

after = to_log(tiny_train(seed=42, epochs=100))
print("identical?", after == baseline)
print("diff lines:", list(difflib.unified_diff(baseline, after, lineterm="")))"""
            ),
            saw(
                "`True`, and an empty list. No differences. In the terminal, this is `diff` printing nothing."
            ),
            q(
                "And what does a real bug look like in a diff?",
                "Here's one we made ourselves. We indented the three training lines one level too little. Indentation decides which lines belong to the loop, so those lines fell out of the `for` loop. "
                "They ran once per epoch, after the loop, and the model learned from only **one chunk of nine**. "
                "The code still ran without errors. Only the diff showed that something had changed.\n\n"
                "```\n"
                "correct                                  one level too little (bug)\n"
                "for i in range(...):                     for i in range(...):\n"
                "    loss = ...                               loss = ...\n"
                "    optimizer.zero_grad()                optimizer.zero_grad()   <- outside the loop\n"
                "    loss.backward()                      loss.backward()\n"
                "    optimizer.step()                     optimizer.step()\n"
                "```\n\n"
                "Here's the same mistake, run on the tiny text.\n\n"
                "👉 **Predict first:** which lines of the log will differ: the first, the middle, the last?",
            ),
            code(BUG_CODE),
            saw(
                "The first and last lines are the same (`Training on…`, `Saved…`), and only the loss lines changed: `-` is the baseline, `+` is the buggy run. "
                "And look at the buggy numbers: they **climb**, from 7.7 to 14.2. That's far worse than plain guessing, which would score about 3.95 here (52 characters, see [05_01](05_01_loss_and_epochs.ipynb)). "
                "Why? The model now learns from the last chunk only, epoch after epoch. So it gets very sure about that one chunk, and takes the same sureness to the other eight, where it's wrong. "
                'Being confidently wrong is what costs the most (05_01 again), so the average goes up while the model "learns". '
                "So a change you didn't mean to make leaves a fingerprint. **A diff turns a silent bug into a visible one.**"
            ),
            q(
                "Why must the seed stay fixed?",
                'The **seed** (see [05_01](05_01_loss_and_epochs.ipynb)) fixes the starting point of the "random" numbers, like shuffling a deck of cards the same way every time. '
                "`torch.manual_seed(42)` comes before the model is made, so the model starts with the same weights. "
                "If the seed changes too, you can't tell a real effect from plain luck.\n\n"
                "One more rule: you only get the same numbers from the same seed plus the **same order of random steps**. "
                "If you add code that draws one extra random number before the model is made, the whole shuffled deck shifts.\n\n"
                "👉 **Predict first:** which of these three will match the first one?",
            ),
            code(
                """torch.manual_seed(42)
first = torch.rand(3)

torch.manual_seed(42)
again = torch.rand(3)

torch.manual_seed(42)
_ = torch.rand(1)  # one extra random step
shifted = torch.rand(3)

print("first:  ", first)
print("again:  ", again)
print("shifted:", shifted)
print("first == again?  ", bool((first == again).all()))
print("first == shifted?", bool((first == shifted).all()))"""
            ),
            saw(
                "The same seed and the same steps give exactly the same numbers. One extra random step in between, and everything shifted. "
                "(`torch.rand(3)` draws 3 random numbers between 0 and 1.)"
            ),
            q(
                "Why change only one thing at a time?",
                "Because if two things change and the result improves, you can't say which one helped. That's how the runs of [05_02](05_02_learning_rate.ipynb) were set up: first only the **learning rate** changed (0.01 → 0.003), "
                "then only the schedule (constant → cosine), then only the epochs (400 → 800). Each change gave its own answer.\n\n"
                "Here's a tiny version, using the picture from the start: the same seed, one change. "
                "`baseline_losses` is the lr 0.01 run from above. The next run changes only the learning rate.\n\n"
                "👉 **Predict first:** with 100 epochs, will the smaller learning rate end with a lower or a higher loss?",
            ),
            code(
                """slow_losses = tiny_train(seed=42, epochs=100, lr=0.003)  # the ONE change

print(f"lr 0.01  (baseline): final loss {baseline_losses[-1]:.3f}")
print(f"lr 0.003 (one change): final loss {slow_losses[-1]:.3f}")"""
            ),
            saw(
                "The smaller learning rate ended higher. Smaller steps get less far in the same number of epochs, which is the trade-off from [05_02](05_02_learning_rate.ipynb). "
                "And we know the learning rate is to blame, because nothing else changed."
            ),
            q(
                "Can I see the two runs together?",
                "Let's draw both loss curves. The orange line is the baseline. We also draw a second baseline run (same seed) as white dots on top of it. "
                "If the dots sit exactly on the line, the two runs were identical, so any gap to the grey line comes from the one change.\n\n"
                "👉 **Predict first:** will you see the white dots as a separate line?",
            ),
            code(
                """import matplotlib.pyplot as plt

from homer_ai.studio_style import ACCENT, INK, MUTED, studio_axes, studio_figure

again_losses = tiny_train(seed=42, epochs=100)  # the baseline, run a second time
epochs = list(range(1, 101))

fig = studio_figure("Same seed, one change", "Only the learning rate is different")
ax = studio_axes(fig)
ax.plot(epochs, baseline_losses, color=ACCENT, linewidth=3)
ax.plot(epochs[::4], again_losses[::4], "o", color=INK, markersize=5)
ax.plot(epochs, slow_losses, color=MUTED, linewidth=3)
ax.text(
    40, 0.4, "lr 0.01 (baseline)",
    color=ACCENT, family="monospace", fontweight="bold",
)
ax.text(
    30, 2.4, "lr 0.003 (the one change)",
    color=MUTED, family="monospace", fontweight="bold",
)
ax.set_ylim(0.25, 4.2)
ax.text(
    98, 0.97 * max(baseline_losses[0], slow_losses[0]),
    "white dots: the baseline run again,\\nsame seed: exactly on the line",
    color=INK, ha="right", va="top", family="monospace",
)
ax.set_xlabel("epoch")
ax.set_ylabel("average loss per chunk")
plt.show()"""
            ),
            saw(
                "The white dots sit exactly on the orange line (the same seed repeats a run exactly), "
                "and the grey line is the one change. So that gap is the effect of the learning rate, and nothing else."
            ),
            q(
                "What goes wrong if I change two things at once?",
                "Let's try it. We change the learning rate **and** double the epochs in one run, and compare it with two runs that each change one thing.\n\n"
                '👉 **Predict first:** from the "both" number alone, will you be able to tell which change helped?',
            ),
            code(
                """more_epochs = tiny_train(seed=42, epochs=200)  # ONE change: epochs
both = tiny_train(seed=42, epochs=200, lr=0.003)  # TWO changes at once

print(f"baseline  (lr 0.01,  100 epochs):  {baseline_losses[-1]:.3f}")
print(f"lr only   (lr 0.003, 100 epochs):  {slow_losses[-1]:.3f}")
print(f"epochs only (lr 0.01,  200 epochs): {more_epochs[-1]:.3f}")
print(f"both      (lr 0.003, 200 epochs):  {both[-1]:.3f}")"""
            ),
            saw(
                'The "both" run ends somewhere in between, and that number alone can\'t tell you why. More epochs help, a smaller learning rate hurts, and the two effects get mixed into one number. '
                "Only the two single-change runs, each compared with the baseline, can untangle them."
            ),
            q(
                "How do I time a run?",
                'Put `time` in front of the command, and throw the output away with `/dev/null` (a special "file" that swallows everything), so that printing doesn\'t slow the clock:\n\n'
                "```\ntime uv run python src/homer_ai/train_homer.py > /dev/null\n```\n\n"
                "Read the line that starts with **`real`**. That's the time on a wall clock, start to finish. A full Book 1 run takes about 8 to 10 minutes on a laptop (one of ours: `real 7m47.27s`). And remember, it writes `homer_model.pt`.\n\n"
                "In Python, `time.perf_counter()` gives a clock reading in seconds. Take one before and one after, and subtract.\n\n"
                "👉 **Predict first:** 40 epochs instead of 20. About how long compared with 20?",
            ),
            code(
                """import time

for epochs in [20, 40]:
    start = time.perf_counter()
    tiny_train(seed=42, epochs=epochs)
    seconds = time.perf_counter() - start
    print(f"{epochs} epochs: {seconds:.2f} seconds")"""
            ),
            saw(
                "Twice the epochs takes about twice as long. That's why timing matters: you know what an idea will cost before you commit to the 800-epoch run."
            ),
            q(
                "Where do I write all this down?",
                "In a **results log**: one row per run, with what you changed and what came out. "
                "This is the log of the four runs of [05_02](05_02_learning_rate.ipynb) (the final loss is the average per chunk, as printed there):\n\n"
                "| # | The one thing that changed | Learning rate | Epochs | Final loss |\n|---|---|---|---|---|\n"
                "| 1 | nothing: the baseline | 0.01 | 400 | 0.475 |\n"
                "| 2 | only the learning rate (against 1) | 0.003 | 400 | 0.135 |\n"
                "| 3 | only the schedule: cosine (against 1) | 0.01 → 0 | 400 | 0.067 |\n"
                "| 4 | only the epochs (against 3) | 0.01 → 0 | 800 | 0.008 |\n\n"
                "And here's the log for our tiny runs, built from the numbers computed above.\n\n"
                "👉 **Predict first:** which row has the lowest final loss?",
            ),
            code(
                """log = [
    ("baseline", 0.01, 100, baseline_losses[-1]),
    ("lr only", 0.003, 100, slow_losses[-1]),
    ("epochs only", 0.01, 200, more_epochs[-1]),
    ("both (unfair)", 0.003, 200, both[-1]),
]

print(f"{'run':<15}{'lr':>8}{'epochs':>8}{'final loss':>12}")
for name, lr, epochs, final in log:
    print(f"{name:<15}{lr:>8}{epochs:>8}{final:>12.3f}")"""
            ),
            saw(
                'A log you can still read in ten seconds a month from today. Each row says what changed, so you can trust it. And the "both" row is marked unfair on purpose.'
            ),
            *questions(
                [
                    (
                        "Is one run enough to trust?",
                        "For comparing two versions of the same code with a fixed seed, yes: any difference comes from your change. "
                        "But a different seed might rank two ideas differently. So for a big decision, repeat the comparison with a few seeds.",
                    ),
                    (
                        "Do I have to use `diff` and `time` in the terminal?",
                        "They're the quickest way, and you'll see them everywhere in AI engineering. The notebook versions here (`==`, `difflib`, `time.perf_counter()`) give you the same answers inside Python.",
                    ),
                ]
            ),
            recap(
                [
                    "Save a baseline, change one thing, keep the seed fixed, and compare (`diff`). Then you know what a change did.",
                    "A refactor leaves the diff empty. A bug leaves a fingerprint.",
                    "Time your runs, and write every run in a results log.",
                ],
                "[05_05: Validation](05_05_validation.ipynb): the real split, and the validation loss.",
                can=[
                    "run an experiment that changes one thing, with a fixed seed and a saved baseline",
                    "tell a refactor from a bug by comparing the losses",
                ],
            ),
        ],
    )

    save(
        "05_training_on_the_odyssey/05_05_validation.ipynb",
        [
            title(
                "05_05 Validation: did it learn, or memorise?",
                "section 5.1.3 (training and validation losses)",
            ),
            where(
                "model learns",
                ["src/homer_ai/train_homer.py", "src/homer_ai/homer_model.pt"],
            ),
            learn(
                [
                    "How to split Book 1 for real, and check the sizes",
                    "How to write `evaluate()`: the loss, measured without learning",
                    "How to read a training-vs-validation graph",
                    "Why the trained model's score on the last 10% isn't a fair test",
                ]
            ),
            words(
                [
                    (
                        "validation loss",
                        "the loss on the validation set: text the model never trained on",
                        "`2.87`",
                    ),
                    (
                        "evaluate",
                        "measure the average loss on some text, without changing the model",
                        "`evaluate(model, val_data)`",
                    ),
                    (
                        "torch.no_grad()",
                        "a block of code where PyTorch doesn't prepare for learning: measure only",
                        "`with torch.no_grad():`",
                    ),
                    (
                        "perplexity",
                        "`e` to the power of the loss: how many characters the model is choosing between, roughly",
                        "loss 2.38 → about 11",
                    ),
                    (
                        "data leak",
                        "test text that the model has already seen in training, so the test isn't fair",
                        "the last 10% inside `homer_model.pt`'s training",
                    ),
                ]
            ),
            *background(),
            q(
                "The training loss went down. Did the model LEARN, or did it memorise?",
                "The training loss can't tell us. It's measured on text the model practised on, like a student marking their own homework. "
                "In [05_03](05_03_memorising_vs_understanding.ipynb) we saw why (overfitting) and the cure: hide a part of the text, the **validation set**, and only ever test on it. "
                "This notebook does it for real, on Book 1, and measures the validation loss.",
            ),
            step_intro(),
            q(
                "How do I make the real split?",
                "With the same two slices as in [05_03](05_03_memorising_vs_understanding.ipynb), on the real data. "
                "`load_data` gives all of Book 1 as numbers, and we cut at 90%.\n\n"
                "```\nBook 1 (19,701 characters)\n"
                "[ ───────── train_data: data[:n] ─────────── ][ val_data: data[n:] ]\n```\n\n"
                "👉 **Predict first:** 90% of 19,701 is 17,730.9. What does `int(...)` make of it? How many characters are left for validation?",
            ),
            code(
                """import math

import torch
import torch.nn.functional as F

from homer_ai.model import HomerGreekLLM
from homer_ai.train_homer import BOOKS_DIR, load_data

data, tokenizer, text = load_data(BOOKS_DIR)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print("all data:  ", len(data))
print("training:  ", len(train_data))
print("validation:", len(val_data))"""
            ),
            saw(
                "17,730 characters to train on and 1,971 to validate on, the numbers we expected. `int` chopped off the `.9`. "
                "One thing to hold on to: **the split happens before training, once, and the model never sees `val_data` while it learns.**"
            ),
            *exercise(
                "can you make an 80/20 split?",
                "Some people hold back more than 10%. Cut the same `data` at 80%: the first part to train on, the rest held out.\n\n"
                "👉 **Predict first:** 80% of 19,701 is 15,760.8. How long will each part be?",
                starter="""
            # 👉 do: find the cut point at 80% of the data (a whole number), then cut twice
            # 💡 hint: use the same cut point in both slices
            cut = None
            train_80 = None
            held_out_20 = None
            """,
                solution="""
            cut = int(0.8 * len(data))
            train_80 = data[:cut]
            held_out_20 = data[cut:]
            """,
                check="""
            check(
                "length of the training part",
                train_80,
                data[: int(0.8 * len(data))],
                show=len,
                coach="The training part is everything before the cut: data[:cut], with "
                "cut = int(0.8 * len(data)). A slice needs a whole number, and int() "
                "makes one.",
            )
            check(
                "length of the held-out part",
                held_out_20,
                data[int(0.8 * len(data)) :],
                show=len,
                coach="The held-out part starts exactly where the training part stops: "
                "data[cut:]. If the two overlap, the model is tested on text it trained on. "
                "If there's a gap between them, text is lost.",
            )
            """,
                hint="The cell above does it with `0.9`: `n = int(0.9 * len(data))`, then `data[:n]` and `data[n:]`.",
                answer="15,760 characters to train on and 3,941 held out. Add them up and you get 19,701 again: nothing lost, nothing used twice. "
                "A bigger held-out part gives a steadier score, but the model has less text to learn from. With a text this small, we keep 10%.",
            ),
            q(
                "How do I measure the loss without teaching the model?",
                "We write a function, `evaluate`, that does what the training loop does, **minus the learning**. Compare the two:\n\n"
                "```\ntrain() loop                    evaluate()\n"
                "-----------                     ----------\n"
                "logits = model(inputs)          logits = model(inputs)\n"
                "loss = cross_entropy(...)       loss = cross_entropy(...)\n"
                "optimizer.zero_grad()    ✗      (not here)\n"
                "loss.backward()          ✗      (not here)\n"
                "optimizer.step()         ✗      (not here)\n```\n\n"
                "Those three lines are the learning: `zero_grad` clears the previous nudges, `backward` works out fresh ones, `step` applies them. "
                "If `evaluate` had them, testing on the validation set would also teach the model the validation set, and the test wouldn't be fair any more.\n\n"
                "`with torch.no_grad():` is the second safeguard. It says: measure only, don't learn. Inside it, PyTorch doesn't keep the notes it would need for `backward()`, so it also runs a little faster and uses less memory.\n\n"
                "One bit of maths before we run it. Random guessing among 137 characters gives each a chance of 1/137, so the loss ([05_01](05_01_loss_and_epochs.ipynb): minus the log of the chance) is `ln(137)`. "
                "`ln` is the **natural logarithm**: `ln(x)` is the power you raise `e` (about 2.718) to, to get `x`. So `ln(137)` is about 4.92, because `e` to the power 4.92 is about 137. "
                "(It's `math.log` in Python.)\n\n"
                "👉 **Predict first:** an untrained model guesses at random among 137 characters. About what loss should `evaluate` give on the validation text?",
            ),
            code(
                EVALUATE_CODE
                + """


torch.manual_seed(42)
model = HomerGreekLLM(vocab_size=tokenizer.vocab_size, embed_dim=32)

print("untrained, on validation:", round(evaluate(model, val_data), 3))
print("random guessing, ln(137):", round(math.log(tokenizer.vocab_size), 3))"""
            ),
            saw(
                "About 4.93 for the untrained model, close to `ln(137) = 4.92`. That's pure guessing, and it's our starting line. "
                "Notice that `evaluate` never touched the optimizer. There's no `optimizer` in it at all."
            ),
            q(
                "What does a real run look like, training loss next to validation loss?",
                "We train our small `HomerGreekLLM` on `train_data` only, for 8 epochs (about 6 seconds). After every epoch we print the training loss (averaged on the way) "
                "and the validation loss (from `evaluate`). The seed (see [05_01](05_01_loss_and_epochs.ipynb)) makes it repeatable.\n\n"
                "👉 **Predict first:** will the validation loss be lower or higher than the training loss? Will it keep falling?",
            ),
            code(
                """torch.manual_seed(42)
model = HomerGreekLLM(vocab_size=tokenizer.vocab_size, embed_dim=32)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

train_curve, val_curve = [], []
for epoch in range(8):
    losses = []
    for i in range(0, len(train_data) - 24, 24):
        inputs = train_data[i : i + 24]
        targets = train_data[i + 1 : i + 24 + 1]
        loss = F.cross_entropy(model(inputs), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    train_curve.append(sum(losses) / len(losses))
    val_curve.append(evaluate(model, val_data))
    print(
        f"epoch {epoch + 1} | train {train_curve[-1]:.3f} "
        f"| validation {val_curve[-1]:.3f}"
    )"""
            ),
            saw(
                "Both losses fall a lot in the first epoch (from about 4.9) and then slowly. The validation loss stays a little **above** the training loss. "
                "That's normal, because the model has practised on one and not the other. (In epoch 1 the training number is higher, but only because it's averaged *while* the model is still learning. The validation loss is measured at the end of the epoch.) The graph shows it better."
            ),
            q(
                "How do I read the two curves?",
                "```\ntrain ↓  validation ↓   → learning real patterns\n"
                "train ↓  validation ↑   → memorising (overfitting): the curves split apart\n```\n\n"
                "So look for two things: the gap between the lines, and which way the validation line goes.",
            ),
            code(
                """import matplotlib.pyplot as plt

from homer_ai.studio_style import ACCENT, GOOD, studio_axes, studio_figure

epochs = list(range(1, 9))
fig = studio_figure(
    "training vs validation",
    "Both fall, close together: learning, not memorising",
    size=(9, 5.5),
)
ax = studio_axes(fig, (0.1, 0.14, 0.8, 0.64))
style = {"marker": "o", "linewidth": 2.5}
ax.plot(epochs, train_curve, color=GOOD, label="training loss", **style)
ax.plot(epochs, val_curve, color=ACCENT, label="validation loss", **style)
ax.annotate(
    f"{val_curve[-1]:.2f}", (8, val_curve[-1]), textcoords="offset points",
    xytext=(8, 6), color=ACCENT, fontsize=11, fontfamily="monospace", fontweight="bold",
)
ax.annotate(
    f"{train_curve[-1]:.2f}", (8, train_curve[-1]), textcoords="offset points",
    xytext=(8, -16), color=GOOD, fontsize=11, fontfamily="monospace", fontweight="bold",
)
ax.set_xlim(0.7, 8.9)
ax.set_xlabel("epoch")
ax.set_ylabel("average loss")
ax.legend(frameon=False, labelcolor="white")
plt.show()"""
            ),
            saw(
                "What can we really say? In this run both lines fall and still sit close together, with validation a little above training. "
                "That's **learning**, not memorising. We don't claim more than that. 8 epochs is short, so this run can't tell us whether a longer one would start to overfit. "
                "(The 5-line model of [05_03](05_03_memorising_vs_understanding.ipynb) had a tiny text to memorise, and it did.) "
                "Overfitting would show as the orange line turning upwards while the green one keeps falling, and we don't see that here. "
                "That's what the validation loss is for: we can check, and we don't have to hope."
            ),
            q(
                "What does the trained `homer_model.pt` score on the last 10%?",
                "We have a real model, trained for 800 epochs: yours (`src/homer_ai/homer_model.pt`) if you trained one, otherwise the ready-made copy in [models/](../../models/README.md). "
                "We can load it (read-only, and the next notebook explains how) and run `evaluate` on `val_data`.\n\n"
                "👉 **Predict first:** will it be better (lower) than the 8-epoch run above? Is it a fair test?",
            ),
            code(
                """from homer_ai.lessons import load_homer_model

# your own homer_model.pt if you made one, otherwise the ready-made model in models/
saved_tokenizer, trained = load_homer_model()

print("trained model, on the first 90%:", round(evaluate(trained, train_data), 3))
print("trained model, on the last 10%: ", round(evaluate(trained, val_data), 3))
print("8-epoch run, on the last 10%:   ", round(val_curve[-1], 3))"""
            ),
            saw(
                "About 2.38 on the first 90% and about 2.53 on the last 10%, against 2.88 for our 8-epoch run. "
                "**But this 2.53 isn't a fair test.** `homer_model.pt` was trained on all of Book 1, including the last 10%, 800 times. "
                "That's a data leak. It's like a student sitting an exam they've already seen. "
                "How much does the leak flatter the number? We can't say. Maybe only a little, because our model is small. "
                "And this is exactly why the split has to happen before training. You can't split afterwards and trust the result.\n\n"
                "In short: our 8-epoch run is a fair test of a weaker model. The saved model is a better model, but its score isn't honest."
            ),
            q(
                "What does a loss like 2.53 mean in everyday words?",
                "**Perplexity** is `e` to the power of the loss. It says the model is about as unsure as if it were choosing between that many characters. "
                "In [05_01](05_01_loss_and_epochs.ipynb), `e^2.38 ≈ 11`. Here:",
            ),
            code(
                """perplexity = math.exp(evaluate(trained, val_data))
print("perplexity on the last 10%:", round(perplexity, 1))
print("random guessing among 137:  ", tokenizer.vocab_size)"""
            ),
            saw(
                "About 12.6. So the model isn't choosing among 137 characters any more. It's about as unsure as if it were choosing between 13. Lower is better."
            ),
            q(
                "Is there a model whose held-out score IS fair?",
                "Yes: the **GPT** of [chapter 4](../04_building_the_model/04_05_the_full_model.ipynb). `uv run homer-train --model gpt` does what this notebook describes. "
                "It trains on the first 90% of Book 1 only, and prints the loss on the last 10% every 20 epochs. "
                "The ready-made copy, `models/odyssey_book1_gpt.pt`, was trained that way (or yours, `src/homer_ai/homer_gpt.pt`, if you ran it). "
                "We only load it here (no training), and measure it on `val_data`, which it never saw.\n\n"
                "👉 **Predict first:** the GPT has 835,072 weights, about 70 times more than our small model. Will its held-out loss beat the unfair 2.53? And the bigram's 2.99 (a table counting which letter follows which, [README](../../README.md))?",
            ),
            code(
                """from homer_ai.lessons import load_homer_gpt
from homer_ai.train_homer import evaluate_gpt

gpt_tokenizer, gpt = load_homer_gpt()  # read-only
assert gpt_tokenizer.chars == tokenizer.chars  # the same numbers for the same letters

gpt_train = evaluate_gpt(gpt, train_data)
gpt_val = evaluate_gpt(gpt, val_data)
print("GPT, on the first 90% (it trained on it):", round(gpt_train, 3))
print("GPT, on the last 10%  (it never saw it): ", round(gpt_val, 3))
print("perplexity on the last 10%:", round(math.exp(gpt_val), 1))"""
            ),
            saw(
                "About 1.51 on the text it trained on and **2.27 on the last 10%**, a perplexity of about 9.7. This one is a fair, held-out number. "
                "It's better than the bigram's 2.99, much better than random guessing (4.92), and even lower than the small model's leaked 2.53. "
                "The gap between 1.51 and 2.27 is the price of never having seen the text. Some memorising is always left. "
                "It's still a long way from a model that writes Greek sentences, but the numbers are real. The samples are in [06_01](../06_making_it_talk/06_01_writing_greek.ipynb) and [06_02](../06_making_it_talk/06_02_less_repetition.ipynb)."
            ),
            q(
                "Why didn't the GPT's training overfit badly?",
                "It could have. With 835,072 weights and only 17,730 letters, a model can learn the text by heart (the 5-line model of [05_03](05_03_memorising_vs_understanding.ipynb) did). "
                "Two things held it back: **dropout** (0.3: while training, 3 numbers in 10 are switched off at random, so it can't rely on any single one), and a short run (200 epochs, with the learning rate sliding down to 0). "
                "The held-out line printed during training is how you'd know when to stop. [05_07](05_07_train_the_gpt.ipynb) walks through that training loop, and shows on a small slice what happens to the held-out loss without dropout.",
            ),
            *questions(
                [
                    (
                        "Should `train()` print the validation loss too?",
                        "Yes. That's how you'd spot overfitting while it happens, and stop early. The small model's `train()` doesn't do it, to keep it simple. "
                        "The GPT's `train_gpt()` in `train_homer.py` does, every 20 epochs.",
                    ),
                    (
                        "Does `torch.no_grad()` change the model?",
                        "No. `no_grad` only stops PyTorch from taking notes for learning. The weights are the same before and after.",
                    ),
                ]
            ),
            recap(
                [
                    "Split first (`data[:n]` and `data[n:]`: 17,730 / 1,971), train on the first part only, test on the second.",
                    "`evaluate` = the training loop without `zero_grad`, `backward` and `step`, inside `torch.no_grad()`.",
                    "Train and validation both falling = learning. A saved model that saw the validation text gives an unfair score.",
                ],
                "[05_06: Saving and loading](05_06_saving_and_loading.ipynb): how to keep the trained model, and use it later.",
                can=[
                    "split data into a training part and a held-out part, with no overlap",
                    "measure a loss without learning from it: `torch.no_grad()`, and no `step`",
                    "read a training curve next to a validation curve",
                ],
            ),
        ],
    )

    # =====================================================================
    # 05_training_on_the_odyssey/05_06_saving_and_loading.ipynb
    # =====================================================================
    save(
        "05_training_on_the_odyssey/05_06_saving_and_loading.ipynb",
        [
            title("05_06 Saving and loading the model", "section 5.4"),
            where(
                "model learns",
                [
                    "src/homer_ai/checkpoint.py",
                    "src/homer_ai/train_homer.py",
                    "src/homer_ai/homer_mcp_server.py",
                ],
            ),
            learn(
                [
                    "What a checkpoint is, and what ours holds (weights, sizes, vocabulary, training notes)",
                    "How to save the weights and load them into a fresh model",
                    "What goes wrong when the shapes differ",
                    "Why `weights_only=True` is safer",
                    "How the MCP server loads our model",
                ]
            ),
            words(
                [
                    (
                        "checkpoint",
                        "a file with everything needed to use (or continue) a trained model",
                        "`homer_gpt.pt`",
                    ),
                    (
                        "state_dict",
                        "a Python dictionary: the name of each weight group and its numbers",
                        "`model.state_dict()`",
                    ),
                    (
                        "pickle",
                        "Python's way of turning objects into bytes; loading a pickle can run code hidden inside it",
                        "why `.pt` files from strangers can be unsafe",
                    ),
                    (
                        "weights_only",
                        "a `torch.load` setting that only allows plain numbers and text, no hidden code",
                        "`weights_only=True`",
                    ),
                    (
                        "temporary folder",
                        "a folder Python makes for you and deletes afterwards, so a demo can't touch real files",
                        "`tempfile.TemporaryDirectory()`",
                    ),
                ]
            ),
            *background(),
            q(
                "Training took minutes. How do I keep the result, and use it later or somewhere else?",
                "When the notebook or script ends, everything in memory is gone, including the trained weights. "
                "So at the end of training we write them to a file. That file is the **checkpoint**. "
                "Later, another program (like our MCP server in [chapter 6](../06_making_it_talk/06_00_chapter_overview.ipynb)) reads it and gets the trained model back, with no training needed.",
            ),
            step_intro(),
            q(
                "What's in our checkpoint?",
                "Everything you need to use the model without asking anyone. Our files (**format 2**) hold six things:\n\n"
                "```\nhomer_gpt.pt\n"
                '├─ "format":      2\n'
                '├─ "kind":        "gpt"  (or "char-1head", the small model)\n'
                '├─ "config":      the sizes: vocab_size, context_length, emb_dim, n_heads, n_layers, …\n'
                '├─ "chars":       the vocabulary, in order (the tokenizer\'s table)\n'
                '├─ "model_state": the weights (a state_dict)\n'
                '└─ "train_meta":  how it was trained: seed, epochs, learning rate, losses, books, versions\n```\n\n'
                "Why the sizes? A bare `state_dict` only fits a model of the same shape (we break that on purpose below). "
                "Without the sizes in the file, the server and every notebook would each have to *know* `embed_dim=32`. With them, the file speaks for itself, and one loader serves any model. "
                "Here's the real code that writes it (printed with `inspect.getsource`, which shows a function's code).\n\n"
                "👉 **Predict first:** which item is biggest on disk?",
            ),
            code(
                """import inspect

from homer_ai.checkpoint import save_checkpoint

print(inspect.getsource(save_checkpoint))"""
            ),
            saw(
                "One dictionary, written by `torch.save`. The `state_dict` is by far the biggest part: the whole GPT file is 3.4 MB (835,072 weights), and the small model's is 88 KB. "
                "The config, the vocabulary and the notes come to a few kilobytes. `git_sha` is the version of the code (if git is there), and `torch_version` is the PyTorch that wrote it."
            ),
            q(
                "Can I look inside a real one?",
                "Yes. We open the ready-made GPT of [models/](../../models/README.md) with `weights_only=True` (more on that below) and print everything except the weights. "
                "We also look at the ready-made small model. Its file is in **format 1**, the simpler of our two checkpoint formats, and has only two keys.",
            ),
            code(
                """import torch

from homer_ai.lessons import READY_MADE_GPT, READY_MADE_MODEL

gpt_file = torch.load(READY_MADE_GPT, weights_only=True)  # read only
print("keys:", list(gpt_file))
print("kind:", gpt_file["kind"], "| config:", gpt_file["config"])
chars = gpt_file["chars"]
print("vocabulary:", len(chars), "tokens, the last three:", chars[-3:])
for name, value in gpt_file["train_meta"].items():
    print(f"  {name:18}{value}")

small_file = torch.load(READY_MADE_MODEL, weights_only=True)
print("\\nsmall model (format 1), keys:", list(small_file))"""
            ),
            saw(
                "Six keys. The config is a plain dictionary (context 64, 128 numbers wide, 4 heads, 4 blocks, dropout 0.3). The vocabulary has 137 tokens, ending with `<|unk|>` and `<|endoftext|>`. "
                "`train_meta` is the lab notebook: seed 42, 200 epochs, 17,730 letters to train on and 1,971 held out, the held-out loss 2.27, and how long it took. "
                'The format-1 file has only `model_state` and `rhapsody_text` (the whole of Book 1 as text). A **rhapsody** is the old Greek name for one book of a Homeric poem, a piece a singer (a *rhapsode*) performed in one go. So `rhapsody_text` just means "the text of the book it trained on". So that\'s the two formats: format 1 is weights plus text, and format 2 adds the settings, the vocabulary and the notes. `load_checkpoint` reads both.'
            ),
            q(
                "Why save the vocabulary (the `chars` list)?",
                "Our tokenizer ([02_01](../02_text_to_numbers/02_01_tokens_and_vocabulary.ipynb)) is a table of characters in a fixed order, and the ID of a character is its place in that table. "
                "So the same table gives the same IDs. The weights are matched to it: token ID 41 still means `π`, because the model learned with that table. "
                "A different table would give a different numbering, and the trained weights would read nonsense. So we save the table itself, and `CharTokenizer.from_chars` rebuilds it. "
                "(A format-1 file carries the whole book, and the table is rebuilt from that text. Same result, but a lot more to carry.)\n\n"
                "👉 **Predict first:** a tokenizer rebuilt from the saved table: does it give the same IDs as the original?",
            ),
            code(
                """from homer_ai.tokenizer import CharTokenizer

sample = "ἄνδρα μοι ἔννεπε"
original = CharTokenizer(sample)
rebuilt = CharTokenizer.from_chars(original.chars)
print(original.encode("ἄνδρα"))
print(rebuilt.encode("ἄνδρα"))
print("same IDs?", original.encode("ἄνδρα") == rebuilt.encode("ἄνδρα"))"""
            ),
            saw(
                "The same list of numbers twice, so `True`. That's why the table is a safe thing to carry around."
            ),
            q(
                "How do I save weights and load them into a fresh model?",
                "`torch.save(object, path)` writes, `torch.load(path)` reads, and `load_state_dict` copies the numbers into a model of the same shape. "
                "We do it in a **temporary folder**, so we never touch the real `homer_model.pt`. The test is simple: the original and the copy must give identical outputs. "
                "`torch.equal` says whether two tensors are exactly the same.\n\n"
                "👉 **Predict first:** a brand-new model has random weights. Does it give the same output as the saved one before loading? After?",
            ),
            code(
                """import tempfile
from pathlib import Path

import torch

from homer_ai.model import HomerGreekLLM

ids = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6])  # a few made-up character IDs

torch.manual_seed(1)
original = HomerGreekLLM(vocab_size=10, embed_dim=32)

with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "small.pt"
    torch.save({"model_state": original.state_dict()}, path)
    print("file size:", path.stat().st_size, "bytes")

    torch.manual_seed(2)  # different random start
    fresh = HomerGreekLLM(vocab_size=10, embed_dim=32)
    with torch.no_grad():
        print("before loading, same output?", torch.equal(original(ids), fresh(ids)))
        fresh.load_state_dict(torch.load(path, weights_only=True)["model_state"])
        print("after loading, same output? ", torch.equal(original(ids), fresh(ids)))"""
            ),
            saw(
                "`False` before, `True` after. Once loaded, the fresh model **is** the saved one, down to the last digit. "
                "The temporary folder, and the file in it, are gone after the `with` block."
            ),
            q(
                "What goes wrong if the new model has a different shape?",
                "`load_state_dict` refuses, because the saved numbers don't fit. [A_07](../A_toolbox/A_07_organising_code.ipynb) shows the same error, and it's why `model.py` is shared.\n\n"
                "👉 **Predict first:** the file holds weights for `embed_dim=32`. Load them into a model with `embed_dim=16`.",
            ),
            code(
                """small = HomerGreekLLM(vocab_size=10, embed_dim=16)
try:
    small.load_state_dict(original.state_dict())
except RuntimeError as error:
    print("Could not load:", str(error).splitlines()[1].strip())"""
            ),
            saw(
                "A size-mismatch error that names the first layer that doesn't fit. A loud error is a good thing here. PyTorch never quietly accepts wrong shapes. "
                "So the model you build must have the **same settings** (`vocab_size`, `embed_dim`) as the one you saved. That's why a format-2 checkpoint carries its `config`, and `load_checkpoint` builds the model from it."
            ),
            q(
                "What's `weights_only=True`, and why does the server use it?",
                "A `.pt` file is a **pickle**, and loading a pickle can run any code hidden inside it. So a `.pt` file from a stranger could harm your computer. "
                "`weights_only=True` tells `torch.load` to accept only plain numbers, text and dictionaries, and to refuse anything else.\n\n"
                "Our checkpoint is just a dictionary of tensors and text, so it loads fine with the flag on. A rule worth keeping: always set it, unless you trust the file completely.",
            ),
            q(
                "How does the MCP server load our model?",
                "It calls `load_checkpoint` (in `checkpoint.py`), which does what we did above, in a few lines. Let's print them from the file (we read it as text, so there's no need to import it).\n\n"
                "👉 **Predict first:** which lines match what we just did by hand?",
            ),
            code(
                """from homer_ai.train_homer import HERE

checkpoint_file = HERE / "checkpoint.py"  # we only read it as text, never import it
keys = ("torch.load", "from_chars(", "= build_model", "load_state_dict", ".eval()")
for line in checkpoint_file.read_text(encoding="utf-8").splitlines():
    if any(key in line for key in keys):
        print(line.strip())"""
            ),
            saw(
                "`torch.load(..., weights_only=True)`, the tokenizer rebuilt from the saved table (`from_chars`; for a format-1 file, from its text), a fresh model of the right kind and size (`build_model`), then `load_state_dict`. "
                "Then comes `model.eval()`, which [chapter 6](../06_making_it_talk/06_00_chapter_overview.ipynb) explains. "
                "So the server needs no training and no numbers of its own. Just this file."
            ),
            q(
                "Can I save the optimizer too, to continue training later?",
                "Yes. The optimizer (`AdamW`) keeps its own memory of recent steps, and that memory isn't in the weights. "
                "Add its `state_dict()` to the checkpoint, and training can carry on where it stopped. Our checkpoints don't have it, because we only need to use the model, not train it further.",
            ),
            code(
                """import torch.nn.functional as F

learner = HomerGreekLLM(vocab_size=10, embed_dim=32)  # separate: `original` stays
optimizer = torch.optim.AdamW(learner.parameters(), lr=0.01)
loss = F.cross_entropy(learner(ids[:-1]), ids[1:])  # guess each next ID
loss.backward()
optimizer.step()  # one step, so the optimizer has something to remember

with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "resume.pt"
    torch.save(
        {
            "model_state": learner.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )
    saved = torch.load(path, weights_only=True)
    print("keys in the file:", list(saved))
    print("optimizer remembers:", list(saved["optimizer_state"]["state"][0]))"""
            ),
            saw(
                "A second key, `optimizer_state`, which holds the optimizer's memory (`exp_avg` is a running average of each weight's recent nudges, `exp_avg_sq` a running average of their squares, and AdamW uses both to size the next nudge). It loads with `weights_only=True` too."
            ),
            *questions(
                [
                    (
                        "Why not save the whole model object with `torch.save(model, path)`?",
                        "It works, but it ties the file to the exact code layout that made it, and loading it needs full pickle trust. "
                        "Saving the `state_dict` (plain numbers) is safer and more flexible.",
                    ),
                ]
            ),
            recap(
                [
                    "A checkpoint (format 2) = the weights (`state_dict`) + the config (sizes) + the vocabulary + notes on how it was trained. The same vocabulary gives the same IDs.",
                    "Save with `torch.save`, load with `load_state_dict` into a model of the same shape. `torch.equal` proves the copy is exact.",
                    "Load with `weights_only=True`. The MCP server does just that.",
                ],
                "[05_07: Train the GPT](05_07_train_the_gpt.ipynb): how the GPT in that checkpoint was trained.",
                can=[
                    "save a model with its vocabulary and settings, and load it back",
                    "prove the loaded copy is exact with `torch.equal`",
                ],
            ),
        ],
    )

    # =====================================================================
    # 05_training_on_the_odyssey/05_07_train_the_gpt.ipynb
    # =====================================================================
    save(
        "05_training_on_the_odyssey/05_07_train_the_gpt.ipynb",
        [
            title(
                "05_07 Train the GPT",
                "section 5.2 (training); the learning-rate schedule is in appendix D",
            ),
            where(
                "model learns",
                ["src/homer_ai/train_homer.py", "src/homer_ai/gpt.py"],
            ),
            learn(
                [
                    "How `train_gpt` feeds the model: batches of random windows",
                    "Why the scores are flattened before the loss",
                    "How the cosine schedule ticks once per step, not once per epoch",
                    "What weight decay and dropout do, and why the GPT needs dropout 0.3",
                    "How to train your own GPT with one command, and read what it prints",
                ]
            ),
            words(
                [
                    (
                        "batch",
                        "several pieces of text the model works on in one step, stacked into one tensor",
                        "16 windows of 64 letters",
                    ),
                    (
                        "window",
                        "one piece of text of `context_length` letters, cut from anywhere in the training text",
                        "64 letters starting at letter 5,120",
                    ),
                    (
                        "weight decay",
                        "at every step, the optimizer also pulls each weight a tiny bit towards 0, so a weight only stays big if the loss keeps asking for it",
                        "`weight_decay=0.01`",
                    ),
                    (
                        "held-out loss",
                        "another name for the validation loss: the loss on text held out of training",
                        "`2.27` on the last 10%",
                    ),
                ]
            ),
            *background(),
            q(
                "We already have a training loop. Why does the GPT get its own?",
                "Because there are two models in this course, and each has its own loop in `src/homer_ai/train_homer.py`. "
                "Everything in 05_01 to 05_04 used the small one. This notebook is about the other one.\n\n"
                "| | The small model | The GPT |\n|---|---|---|\n"
                "| Code | `model.py`, 11,840 weights | `gpt.py`, 835,072 weights ([04_05](../04_building_the_model/04_05_the_full_model.ipynb)) |\n"
                "| Loop | `train()` | `train_gpt()` |\n"
                "| One step reads | one chunk of 24 letters | a batch: 16 windows of 64 letters |\n"
                "| Which pieces | back to back, in order | picked at random |\n"
                "| Trains on | all of Book 1 | the first 90% ([05_05](05_05_validation.ipynb)) |\n"
                "| Schedule ticks | once per epoch | once per step |\n"
                "| Dropout | none | 0.3 |\n"
                "| Checks on hidden text | no | every 20 epochs |\n"
                "| Command | `uv run homer-train` | `uv run homer-train --model gpt` |\n"
                "| Writes | `src/homer_ai/homer_model.pt` | `src/homer_ai/homer_gpt.pt` |\n\n"
                "The four steps inside are the ones you know: guess, measure the loss, work out the nudges, nudge. Here's the whole inner loop of `train_gpt`:\n\n"
                "```python\n"
                "inputs, targets = random_batch(train_data, size, batch_size)       # step 1\n"
                "logits = model(inputs)\n"
                "loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())  # step 2\n"
                "optimizer.zero_grad()\n"
                "loss.backward()\n"
                "optimizer.step()\n"
                "scheduler.step()                                                   # step 3\n"
                "```\n\n"
                "Three lines are new, and steps 1 to 3 below take them one at a time. Then weight decay and dropout, which sit in the lines around the loop.",
            ),
            step_intro(),
            q(
                "Step 1: what's a batch of random windows?",
                "The small model's loop walks along the text, one chunk after the other. `random_batch` does it differently. It picks `batch_size` starting places at random, cuts a window at each one, and stacks the windows into one tensor. "
                "The targets are the same windows moved one letter on, as in [02_04](../02_text_to_numbers/02_04_sliding_window.ipynb).\n\n"
                "Let's shrink it so we can read it. The text is the first 23 characters of Book 1 (`␣` is a space), the windows are 4 letters long, and the batch is 3 windows:\n\n"
                "```\n"
                "ἄ ν δ ρ α ␣ μ ο ι ␣ ἔ ν ν ε π ε , ␣ μ ο ῦ σ α\n"
                "            └─────┘                          a window starting at μ\n"
                "inputs :    μ  ο  ι  ␣\n"
                "targets:    ο  ι  ␣  ἔ                       read it column by column\n"
                "```\n\n"
                "Why random? With windows cut back to back, the model would see the same 277 windows of the training text in every epoch. A window may start at any letter, so there are 17,665 different ones, and every step shows the model a fresh mix.\n\n"
                "👉 **Predict first:** what shape will `inputs` have: 3 windows of 4 letters?",
            ),
            code(
                """import torch

from homer_ai.train_homer import BOOKS_DIR, load_data, random_batch

data, tokenizer, text = load_data(BOOKS_DIR)  # Book 1 as numbers
line = data[:23]
print(repr(tokenizer.decode(line.tolist())))

torch.manual_seed(0)
inputs, targets = random_batch(line, context_length=4, batch_size=3)
print("inputs:", tuple(inputs.shape), " targets:", tuple(targets.shape))
print()
for window, answers in zip(inputs, targets, strict=True):
    print("inputs :", "  ".join(tokenizer.decode(window.tolist()).replace(" ", "␣")))
    print("targets:", "  ".join(tokenizer.decode(answers.tolist()).replace(" ", "␣")))
    print()"""
            ),
            saw(
                "Both are `(3, 4)`: 3 windows, 4 letters each. That first number is the batch.\n\n"
                "```\ninputs : ι  ␣  ἔ  ν\ntargets: ␣  ἔ  ν  ν\n```\n\n"
                "Each targets row is its inputs row moved one letter on. And the windows overlap (the second one starts one letter after the first), which back-to-back chunks never do. "
                "The real run uses `context_length=64` and `batch_size=16`, so one step reads 16 × 64 = 1,024 letters."
            ),
            q(
                "Step 2: why `flatten(0, 1)` before the loss?",
                'The GPT answers every letter of every window at once. For our batch that\'s 3 × 4 = 12 questions ("what comes after this letter?"), and for each one it gives 137 scores. So the logits have **three** dimensions: `(3, 4, 137)`.\n\n'
                "`F.cross_entropy` wants something simpler: one row of scores per question, and one right answer per question. It doesn't care which window a question came from. So we lay the 12 questions out in one long list:\n\n"
                "```\n"
                "logits   (3, 4, 137)  → .flatten(0, 1) →  (12, 137)   12 questions, 137 scores each\n"
                "targets  (3, 4)       → .flatten()     →  (12,)       12 right answers\n"
                "```\n\n"
                "`flatten(0, 1)` merges dimension 0 and dimension 1 (windows and letters) into one, and leaves the 137 scores alone. "
                "The small model never needed this: it reads one chunk with no batch dimension, so its logits are already `(24, 137)`.\n\n"
                "We try it on a small untrained GPT (2 blocks, 64 numbers per letter), so the cell runs at once.\n\n"
                "👉 **Predict first:** the model is untrained. About what loss do you expect? (Random guessing among 137 gives `ln(137) = 4.92`.)",
            ),
            code(
                """import math

import torch.nn.functional as F

from homer_ai.gpt import GPTConfig, GPTModel

SMALL = {"context_length": 32, "emb_dim": 64, "n_heads": 4, "n_layers": 2}

torch.manual_seed(42)
small_gpt = GPTModel(GPTConfig(vocab_size=tokenizer.vocab_size, drop_rate=0.0, **SMALL))

logits = small_gpt(inputs)
print("logits: ", tuple(logits.shape))
print("flat logits:", tuple(logits.flatten(0, 1).shape))
print("flat targets:", tuple(targets.flatten().shape))
loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
print("loss:", round(loss.item(), 3), " ln(137):", round(math.log(137), 3))"""
            ),
            saw(
                "`(3, 4, 137)` becomes `(12, 137)`, and the targets become 12 numbers. The loss is one number, the average over all 12 questions: about 5.16, close to the 4.92 of pure guessing. "
                "(A new model's scores aren't perfectly even, so it's a little worse than guessing. Training fixes that within a few steps.)"
            ),
            q(
                "Step 3: when does the learning rate change?",
                "In [05_02](05_02_learning_rate.ipynb) the cosine schedule moved one notch per **epoch**. `train_gpt` moves it one notch per **step**, so the learning rate slides down smoothly and doesn't go down in 200 stairs. "
                "For that, the schedule has to know how many steps there will be in all:\n\n"
                "```python\n"
                "steps_per_epoch = len(train_data) // (size * batch_size)\n"
                "scheduler = CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch)\n"
                "```\n\n"
                'With random windows there\'s no real "pass through the text" any more. So here an **epoch** just means: as many steps as it takes to read about as many letters as the training text has.\n\n'
                "The settings are in `GPT_SETTINGS` and `GPT_TRAINING`, at the top of `train_homer.py`. We read them from there and do the sums. The schedule below is a real `CosineAnnealingLR`. It has only one dummy number to look after, so nothing is trained.\n\n"
                "👉 **Predict first:** 17,730 training letters, and 1,024 letters per step. How many steps is one epoch? And 200 epochs?",
            ),
            code(
                """from torch.optim.lr_scheduler import CosineAnnealingLR

from homer_ai.train_homer import GPT_SETTINGS, GPT_TRAINING, split_ids

train_data, val_data = split_ids(data)  # the first 90%, the last 10% (05_05)
size, batch_size = GPT_SETTINGS["context_length"], GPT_TRAINING["batch_size"]
steps_per_epoch = len(train_data) // (size * batch_size)
total_steps = GPT_TRAINING["epochs"] * steps_per_epoch
print("training letters:", len(train_data), " letters per step:", size * batch_size)
print("steps per epoch: ", steps_per_epoch, " steps in all:", total_steps)

dummy = torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=GPT_TRAINING["lr"])
schedule = CosineAnnealingLR(dummy, T_max=total_steps)
for step in range(total_steps + 1):
    if step % 850 == 0:
        epoch = step // steps_per_epoch
        print(f"step {step:4} (epoch {epoch:3}): lr {schedule.get_last_lr()[0]:.5f}")
    dummy.step()
    schedule.step()"""
            ),
            saw(
                "17 steps per epoch (17 × 1,024 = 17,408 letters, about one pass), so 200 epochs is 3,400 steps. "
                "The learning rate starts at its peak, 0.001, is exactly half (0.0005) at step 1,700, and reaches 0 at step 3,400. It's the curve of [05_02](05_02_learning_rate.ipynb), with 3,400 small notches."
            ),
            q(
                "Step 4: what's weight decay?",
                "The optimizer line of `train_gpt` has one setting we haven't met:\n\n"
                "```python\noptimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)\n```\n\n"
                "**Weight decay** is a small pull towards 0. At every step, on top of the usual nudge, AdamW shrinks each weight by a tiny share of itself: `learning rate × weight_decay × weight`. "
                "A weight the loss really needs gets pushed back up by its gradient at the next step. A weight that grew big only to memorise one odd spot of the text slowly fades. "
                "So it's a mild brake on memorising.\n\n"
                "To see the pull by itself, we take two weights whose gradient is exactly 0, so the loss asks for no nudge at all. (The learning rate here is 0.1, so the numbers are easy to check.)\n\n"
                "👉 **Predict first:** a weight of 1.0, learning rate 0.1, weight decay 0.01. What is it after one step?",
            ),
            code(
                """weights = torch.tensor([1.0, -2.0], requires_grad=True)
optimizer = torch.optim.AdamW([weights], lr=0.1, weight_decay=0.01)

loss = (weights * 0).sum()  # a loss that doesn't care about the weights: gradient 0
loss.backward()
optimizer.step()
print("gradient:", weights.grad.tolist())
print("after one step:", [round(w, 4) for w in weights.tolist()])
default = torch.optim.AdamW([weights]).defaults["weight_decay"]
print("AdamW's default weight decay:", default)"""
            ),
            saw(
                "The gradient is 0, and still both weights moved towards 0: 1.0 became 0.999 (it lost `0.1 × 0.01 × 1.0 = 0.001`), and −2.0 became −1.998. Each weight shrinks by the same share, so bigger weights lose more.\n\n"
                "The last line tells you something about the small model too. AdamW's default is already 0.01, so `train()` has used the same weight decay all along. `train_gpt` just writes it out."
            ),
            q(
                "Step 5: what does dropout 0.3 do while training?",
                "**Dropout** ([03_03](../03_attention/03_03_hiding_the_future.ipynb)) switches off numbers at random while the model trains. With `drop_rate=0.3`, each number has a 3 in 10 chance of becoming 0 at every step, and the ones that stay are scaled up (divided by 0.7), so the total stays about the same. "
                "The model can't count on any single number being there, so it has to spread what it knows. That makes learning a text by heart harder.\n\n"
                "Dropout is only for training. `model.train()` switches it on, and `model.eval()` switches it off, so measuring and writing use every number.\n\n"
                "👉 **Predict first:** ten 1s go through `nn.Dropout(0.3)` in training mode. What comes out? And in eval mode?",
            ),
            code(
                """from torch import nn

torch.manual_seed(0)
dropout = nn.Dropout(0.3)
ones = torch.ones(10)

dropout.train()
print("training:", [round(v, 2) for v in dropout(ones).tolist()])
dropout.eval()
print("eval:    ", [round(v, 2) for v in dropout(ones).tolist()])"""
            ),
            saw(
                "In training mode some of the 1s became 0 and the others became 1.43 (that's 1 ÷ 0.7). In this draw 4 of the 10 were switched off. On average it's 3. In eval mode all ten come out as 1, untouched."
            ),
            q(
                "Step 6: does dropout really matter? And what's the held-out check?",
                "Let's run `train_gpt` itself and watch. Not the real training (that's 3 minutes, and it's a terminal command), but a small one that takes seconds:\n\n"
                "```\n"
                "text:    the first 4,000 letters of Book 1 (3,600 to train on, 400 held out)\n"
                "model:   a small GPT: 2 blocks, 64 numbers per letter, reads 32 letters\n"
                "run 1:   dropout 0.0\n"
                "run 2:   dropout 0.3     <- the ONE change (same seed, same everything else)\n"
                "```\n\n"
                "Every 20 epochs `train_gpt` prints a line with two losses. The **train loss** is the average over the steps of that epoch, measured while learning. "
                "The **held-out loss** comes from `evaluate_gpt`, which is the `evaluate` of [05_05](05_05_validation.ipynb) for the GPT: dropout off (`model.eval()`), no learning (`torch.no_grad()`), on the text the model never trains on. "
                "At the end the function returns both losses measured that same careful way.\n\n"
                "This small run uses a learning rate of 0.003, so things happen fast. Both runs together take about 20 seconds, and nothing is saved.\n\n"
                "👉 **Predict first:** without dropout the train loss will go lower. What will its held-out loss do?",
            ),
            code(
                """from homer_ai.train_homer import train_gpt

slice_train, slice_val = split_ids(data[:4000])
print("training letters:", len(slice_train), " held out:", len(slice_val))

final = {}
for drop_rate in (0.0, 0.3):
    print(f"\\ndropout {drop_rate}")
    torch.manual_seed(42)
    config = GPTConfig(vocab_size=tokenizer.vocab_size, drop_rate=drop_rate, **SMALL)
    final[drop_rate] = train_gpt(
        GPTModel(config), slice_train, slice_val,
        seed=42, epochs=120, lr=0.003, batch_size=16, eval_every=20,
    )

print()
for drop_rate, losses in final.items():
    print(
        f"dropout {drop_rate}: train loss {losses['final_train_loss']:.2f}"
        f" | held-out loss {losses['val_loss']:.2f}"
    )"""
            ),
            saw(
                "| | Train loss at the end | Held-out loss, epoch 20 → 120 |\n|---|---|---|\n"
                "| dropout 0.0 | 0.77 | 2.91 → 2.93 → 3.19 → 3.47 → 3.63 → **3.65** |\n"
                "| dropout 0.3 | 2.29 | 2.96 → 2.93 → 2.93 → 2.91 → 2.91 → **2.90** |\n\n"
                "Without dropout the train loss falls to 0.77. If that were all we printed, it would look like a great run. But the held-out loss turns round after epoch 20 and climbs to 3.65. "
                "The model is learning its 3,600 letters by heart, and getting worse at every other piece of Greek. That's the overfitting of [05_03](05_03_memorising_vs_understanding.ipynb), caught while it happens.\n\n"
                "With dropout 0.3 the train loss stays much higher (2.29), and the held-out loss keeps going down, slowly. It's the better model of the two on text it hasn't seen, 2.90 against 3.65.\n\n"
                "That's why `GPT_SETTINGS` says `drop_rate: 0.3`. The real model is far bigger than this one, and Book 1 is still a small text. (Your last digits may differ a little on another machine. The picture won't.)"
            ),
            md(
                "## 5. Full example: train your own GPT\n\n"
                "### ❓ How do I train the real one?\n\n"
                "In a terminal, in the project folder:\n\n"
                "```\nuv run homer-train --model gpt\n```\n\n"
                "It runs `main_gpt()` in `train_homer.py`: load Book 1, split it 90/10, build a `GPTModel` from `GPT_SETTINGS`, call `train_gpt` with `GPT_TRAINING`, and save a checkpoint ([05_06](05_06_saving_and_loading.ipynb)). "
                "It takes about 3 minutes on a laptop, and prints this (the `…` are losses we left out; the `lr` column is the schedule of step 3):\n\n"
                "```\n"
                "Training a GPT (835,072 weights) on 17730 Greek characters, checking on the last 1971 (Vocab size: 137)...\n"
                "Epoch 20/200 | train loss: … | held-out loss: … | lr: 0.00098\n"
                "Epoch 40/200 | train loss: … | held-out loss: … | lr: 0.00090\n"
                "…\n"
                "Epoch 200/200 | train loss: … | held-out loss: … | lr: 0.00000\n"
                "Done in 2.9 minutes | train loss 1.5129 | held-out loss 2.2712\n"
                "Saved trained model to …/src/homer_ai/homer_gpt.pt\n"
                "```\n\n"
                "Watch the held-out column while it runs. As long as it keeps falling, the model is still learning. If it turned upwards, as in run 1 above, you'd stop there.\n\n"
                "The run writes `src/homer_ai/homer_gpt.pt` (and replaces the one from an earlier run). The ready-made file in `models/` is never touched. "
                "From then on the notebooks print `Using your own GPT`, and the MCP server of [chapter 6](../06_making_it_talk/06_00_chapter_overview.ipynb) picks your file first.\n\n"
                "Every checkpoint carries notes on how it was made. So we can read the last line of that log from the ready-made file, without training anything.\n\n"
                "👉 **Predict first:** which will be lower, the train loss or the held-out loss?"
            ),
            code(
                """from homer_ai.lessons import READY_MADE_GPT

notes = torch.load(READY_MADE_GPT, weights_only=True)["train_meta"]  # read only
for key in ("seed", "epochs", "lr", "batch_size", "train_chars", "val_chars"):
    print(f"{key:12}{notes[key]}")
print(f"train loss    {notes['final_train_loss']:.4f}")
print(f"held-out loss {notes['val_loss']:.4f}")
print(f"time          {notes['train_seconds'] / 60:.1f} minutes")"""
            ),
            saw(
                "Seed 42, 200 epochs, peak learning rate 0.001, 16 windows per step, 17,730 letters to learn from and 1,971 held out. "
                "The train loss is **1.51** and the held-out loss **2.27**, after 2.9 minutes. Those are the numbers in [models/README.md](../../models/README.md) and in [05_05](05_05_validation.ipynb).\n\n"
                "The gap between the two is the price of never having seen the text: some of that 1.51 is Book 1 known by heart. The 2.27 is the number to quote."
            ),
            *questions(
                [
                    (
                        "Will my run print exactly 1.5129 and 2.2712?",
                        "On the same machine and PyTorch version, yes: the run is seeded ([05_04](05_04_running_fair_experiments.ipynb)). On another machine the last digits can differ, because processors don't all round the same way. "
                        "The time differs more. Three to four minutes is normal on a laptop.",
                    ),
                    (
                        "Can I train the small model too?",
                        "Yes: `uv run homer-train` with no option (about 8 to 10 minutes). It writes `src/homer_ai/homer_model.pt`, see [05_04](05_04_running_fair_experiments.ipynb). The two commands write different files, so neither run disturbs the other.",
                    ),
                    (
                        "What would I change to get a better model?",
                        "More text, first of all. `BOOKS` in `train_homer.py` lists the books to train on, and Book 1 is only about 20,000 letters. With more text a bigger model stops memorising and starts learning, and then a lower dropout and more epochs begin to pay. "
                        "Change one thing at a time, and watch the held-out loss.",
                    ),
                ]
            ),
            recap(
                [
                    "`train_gpt` is the loop you know, fed with batches of random windows. `flatten(0, 1)` turns the `(batch, letters, 137)` scores into one row per question for the loss.",
                    "The cosine schedule ticks once per step. Weight decay pulls every weight a little towards 0, and dropout 0.3 keeps this big model from learning Book 1 by heart.",
                    "`uv run homer-train --model gpt` trains your own in about 3 minutes: train loss 1.51, held-out loss 2.27.",
                ],
                "[chapter 6: Making it talk](../06_making_it_talk/06_00_chapter_overview.ipynb): making the saved model write.",
                can=[
                    "read `train_gpt` line by line: batches, schedule, weight decay, dropout",
                    "train your own GPT with `uv run homer-train --model gpt`",
                ],
            ),
        ],
    )
