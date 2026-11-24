# ruff: noqa: E501
"""Chapter 6, Making it talk."""

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
from .diagrams import DIAGRAM_LOOP, DIAGRAM_MCP
from .setup_code import CH6_SETUP, CH6_WINDOW


def build():
    # =====================================================================
    # 06_making_it_talk/06_00_chapter_overview.ipynb
    # =====================================================================
    save(
        "06_making_it_talk/06_00_chapter_overview.ipynb",
        [
            title(
                "06_00 Making it talk (overview)",
                "sections 4.7, 5.1.1 and 5.3 (text generation); the MCP part is our own",
            ),
            where(
                "model writes",
                [
                    "src/homer_ai/homer_gpt.pt",
                    "src/homer_ai/homer_mcp_server.py",
                ],
            ),
            learn(
                [
                    "How a model writes: guess one letter, add it, guess again",
                    "Why the plainest way of writing gets stuck, and what fixes it",
                    "How other AI apps, like Claude, can use our model through MCP",
                    "Which notebook to open for each idea",
                ]
            ),
            words(
                [
                    (
                        "generation",
                        "making new text: the model guesses one letter, we add it to the end, and it guesses again",
                        "`ἄνδρα` → `ἄνδρα τοι …`",
                    ),
                    (
                        "autoregressive",
                        "feeding the model's own output back in as its next input (auto = self)",
                        "the letter it just wrote is read in the next round",
                    ),
                    (
                        "prompt",
                        "the text you give a model to continue",
                        "`ἄνδρα`",
                    ),
                    (
                        "tool",
                        "a function an AI app is allowed to call, with a name, a description and arguments",
                        "`consult_homer_llm`",
                    ),
                    (
                        "MCP",
                        "Model Context Protocol: the standard way for an AI app to find and call tools",
                        "like one plug shape that fits every socket",
                    ),
                    (
                        "MCP server",
                        "a small program that offers tools through MCP",
                        "`homer_mcp_server.py`",
                    ),
                    (
                        "MCP client",
                        "the AI app that connects to MCP servers and calls their tools",
                        "Claude Code",
                    ),
                ]
            ),
            *background(),
            q(
                "The model only guesses ONE next letter. How does it write a whole line?",
                "You know the row of suggestions above your phone's keyboard? "
                "Tap the middle one again and again and the phone writes a sentence by itself, one word at a time, each one picked from what's there so far. "
                "Our model writes the same way, and it has the same weakness: keep taking the top suggestion and it soon goes round in circles. "
                "The second half of the chapter lends that keyboard out, so Claude can ask our model for a line of Greek the way you'd ask a colleague who's good at exactly one thing. "
                "The picture flatters us in one way: your phone suggests whole words and has learned from millions of people, and our model guesses a single letter and has read one book of one poem.\n\n"
                "So, how does one guess become a line? By doing it over and over. The model guesses a letter, we add it to the text, and the longer text goes back in for the next guess. "
                "Feeding the output back in as the next input is called **autoregressive** (*auto* = self). Here are the first four rounds on the real model, starting from the prompt `ἄνδρα` "
                '("man", the Odyssey\'s first word). `␣` stands for a space:\n\n'
                "```\n"
                "round 1   reads: ἄνδρα      guess: ␣\n"
                "round 2   reads: ἄνδρα␣     guess: τ\n"
                "round 3   reads: ἄνδρα␣τ    guess: ο\n"
                "round 4   reads: ἄνδρα␣το   guess: ι\n"
                'written:  ἄνδρα τοι         (τοι: "to you", a small word)\n'
                "```\n\n"
                "So each round the text grows by one letter, and the model reads the longer text.\n\n"
                "👉 **Predict first:** the code below runs exactly this on the trained model. Will its picture match?",
            ),
            code(CH6_SETUP),
            code(
                r"""if model is not None:
    text = "ἄνδρα"
    for round_number in range(1, 5):
        ids = torch.tensor(tokenizer.encode(text))
        with torch.no_grad():  # we only guess here, we do not learn
            scores = model(ids)[-1]  # [-1]: the guess after the last letter
        best = scores.argmax().item()  # argmax: the position of the biggest score
        guess = tokenizer.decode([best])
        shown_text = text.replace(" ", "␣")  # make the spaces visible
        shown_guess = guess.replace(" ", "␣")
        print(f"round {round_number}   reads: {shown_text:<8}  guess: {shown_guess}")
        text = text + guess
    print("written: ", text)"""
            ),
            saw(
                "Same picture, and this time from real guesses. That's all writing is: this one small round, repeated. "
                "In [06_01](06_01_writing_greek.ipynb) we build it as a proper function, `write()`, on the real trained model."
            ),
            q(
                "Is the best guess always the best choice?",
                "Not for writing. Always taking the single most likely letter is called **greedy** ([01_01](../01_foundations/01_01_a_language_model_without_a_brain.ipynb)), "
                'and it falls into loops. The same few letters come out again and again, like `και και και` ("and and and"). '
                "Why that happens is in 06_01. The fix, in [06_02](06_02_less_repetition.ipynb), is to roll a weighted dice instead, so likely letters win more often, but not always. "
                "The dice has two dials, called temperature and top-k.",
            ),
            q(
                "How can another AI app, like Claude, use our model?",
                "Through **MCP**, the Model Context Protocol. A protocol is an agreed way of talking. Think of one plug shape: every AI app that has the plug can use every tool that has the socket. "
                "Our model sits behind an MCP server (`src/homer_ai/homer_mcp_server.py`), a small program that offers our model as a tool, `consult_homer_llm`. "
                "An MCP client, such as Claude Code, connects to the server and can call the tool.\n\n"
                "```\n"
                "you ─▶ Claude Code (the client) ─▶ our MCP server ─▶ our model\n"
                "                                    consult_homer_llm   writes Greek\n"
                "you ◀─ Claude's answer ◀─ the tool's text ◀──────────┘\n"
                "```\n\n"
                "In [06_03](06_03_the_mcp_server.ipynb) we take the server apart, piece by piece.\n\n"
                "*A lot of my AI work runs inside automation tools like n8n, where the model is just one step in a workflow. That's exactly the role our Homer model plays for Claude.*",
            ),
            q(
                "Who runs the tool, Claude or our model?",
                "Neither of them, really. Claude only writes a small text form in **JSON** (plain text for data, with names and values in curly brackets). "
                "The program around Claude, the harness, is what runs our tool. The [course map](../00_start_here/00_00_course_map.ipynb) explains the whole chain once, in full, and [06_03](06_03_the_mcp_server.ipynb) shows our server's part of it.",
            ),
            q(
                "Which notebook do I open for each idea?",
                "| Notebook | Question it answers |\n|---|---|\n"
                "| [`06_01_writing_greek.ipynb`](06_01_writing_greek.ipynb) | How do I build `write()`, and why does it loop? |\n"
                "| [`06_02_less_repetition.ipynb`](06_02_less_repetition.ipynb) | How do temperature and top-k make the writing less repetitive? |\n"
                "| [`06_03_the_mcp_server.ipynb`](06_03_the_mcp_server.ipynb) | How does our server work, and how do I connect it to Claude Code? |",
            ),
            recap(
                [
                    "Writing is: guess one letter, add it, guess again (autoregressive). Greedy picking loops.",
                    "A weighted dice (temperature, top-k) makes the writing less repetitive.",
                    "An MCP server offers our model as a tool. The client's harness runs it, and Claude only writes text.",
                ],
                "[06_01: Writing Greek](06_01_writing_greek.ipynb).",
            ),
        ],
    )

    # =====================================================================
    # 06_making_it_talk/06_01_writing_greek.ipynb
    # =====================================================================
    save(
        "06_making_it_talk/06_01_writing_greek.ipynb",
        [
            title(
                "06_01 Writing Greek, one letter at a time", "sections 4.7 and 5.1.1"
            ),
            where(
                "model writes",
                ["src/homer_ai/model.py", "src/homer_ai/homer_model.pt"],
            ),
            learn(
                [
                    "How to turn the model's scores into one chosen letter",
                    "Why we feed in only the last 24 letters",
                    "How to build `write()` and write from three prompts",
                    "Why greedy writing falls into a loop, shown with real numbers",
                ]
            ),
            words(
                [
                    (
                        "repetition loop",
                        "the model reads the same letters again and again, so it writes the same letters again and again",
                        "`και και και`",
                    ),
                    (
                        "context window",
                        "the most recent letters the model can read at once; older ones fall out of view. Ours is 24 letters",
                        "`ids[-24:]`",
                    ),
                ]
            ),
            *background(),
            q(
                "Step 1: what does the trained model give me for the next letter?",
                "The same as in [01_04](../01_foundations/01_04_our_model_as_a_black_box.ipynb): one row of 137 scores for each letter we give it (one score per token). Then softmax ([A_04](../A_toolbox/A_04_pytorch_in_this_repo.ipynb)) "
                "turns a row into chances that add up to 1. We read only the **last** row, because that's the guess for what comes after the last letter.\n\n"
                "Two small things in the code. `model.eval()` switches off training-only tricks such as dropout ([03_03](../03_attention/03_03_hiding_the_future.ipynb)). Our `model.py` has none, "
                "so here it changes nothing, but it's the right habit. `torch.no_grad()` means \"we only guess, we don't learn\", which saves work.\n\n"
                "👉 **Predict first:** after `ἄνδρα`, which character has the biggest chance? (The course map showed it.)",
            ),
            code(CH6_WINDOW),
            code(
                r"""if model is not None:
    ids = tokenizer.encode("ἄνδρα")
    with torch.no_grad():
        scores = model(torch.tensor(ids))  # one row of scores for each of the 5 letters
    print("scores:", tuple(scores.shape), " (5 letters, 137 scores each)")

    # only the last row; dim=-1 means "across the 137 scores"
    chances = torch.softmax(scores[-1], dim=-1)
    print("chances add up to", round(chances.sum().item(), 4))
    best3 = torch.topk(chances, 3)  # the 3 biggest chances, and where they are
    for chance, token_id in zip(best3.values, best3.indices, strict=True):
        letter = tokenizer.decode([token_id.item()])
        print(f"  {letter!r:6} {chance.item():.2f}")"""
            ),
            saw(
                "A space `' '` is the most likely next character (0.26), then `ι` (0.15) and `τ` (0.07). "
                "`!r` prints with quotes, so the invisible space shows up. Notice the chances are small even for the winner. The model is far from sure."
            ),
            q(
                "Step 2: what do all the chances look like?",
                "Here are the 10 biggest, as a picture. The orange bar is the one greedy would pick.",
            ),
            code(
                r"""if model is not None:
    import matplotlib.pyplot as plt

    from homer_ai.studio_style import ACCENT, INK, MUTED, studio_axes, studio_figure

    def show(ch):  # make invisible characters visible
        return {" ": "␣", "\n": "⏎"}.get(ch, ch)

    top = torch.topk(chances, 10)
    labels = [show(tokenizer.decode([i])) for i in top.indices.tolist()]
    values = top.values.tolist()

    fig = studio_figure("The next letter", "The model's chances after ἄνδρα", (10, 6))
    ax = studio_axes(fig)
    bars = ax.bar(labels, values, color=MUTED)
    bars[0].set_color(ACCENT)  # the greedy pick
    for i, v in enumerate(values):
        ax.text(i, v + 0.004, f"{v:.2f}", ha="center", color=INK, family="monospace")
    ax.tick_params(axis="x", labelsize=16, labelcolor=INK)
    ax.set_ylabel("chance")
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)  # grid lines behind the bars
    plt.show()
    print("the other 127 characters share:", round(1 - sum(values), 2))"""
            ),
            saw(
                "One bar stands out (`␣`, orange), but the next ones aren't far behind. And the 127 characters we can't see on the picture still hold nearly a quarter of all the chance (0.23). "
                "Keep that number in mind, because it's what [06_02](06_02_less_repetition.ipynb) plays with."
            ),
            q(
                "Step 3: how do I pick one letter and add it?",
                "With greedy picking. `argmax` gives the ID with the biggest chance, and `.item()` turns a one-number tensor into a plain number. Then we add it to the list of IDs. "
                "(Softmax never changes which number is biggest, so `argmax` of the scores would pick the same letter. We keep the chances step because 06_02 needs them.)",
            ),
            code(
                r"""if model is not None:
    best = chances.argmax().item()
    print("best ID:", best, "=", repr(tokenizer.decode([best])))
    ids = ids + [best]  # one new letter on the end
    print("text now:", repr(tokenizer.decode(ids)))"""
            ),
            saw(
                "ID 1 is the space, so the text grew from `'ἄνδρα'` to `'ἄνδρα '`. The space is hiding at the end, and `repr` puts quotes around it so you can see it."
            ),
            q(
                "Step 4: why do we feed in only the last 24 letters?",
                "The model trained on pieces of 24 letters, `chunk_size` ([02_04](../02_text_to_numbers/02_04_sliding_window.ipynb)), so 24 is the longest text it knows how to read. "
                "As the text grows we keep only the last 24, and older letters fall out of view, like a short-term memory. This is the **context window**.\n\n"
                "Take the first 30 characters of Book 1:\n\n"
                "```\n"
                "all 30:   ἄνδρα μοι ἔννεπε, μοῦσα, πολύτ\n"
                'kept 24:        μοι ἔννεπε, μοῦσα, πολύτ     (the model can\'t "see" ἄνδρα␣ here)\n'
                "```\n\n"
                "👉 **Predict first:** which 6 characters get dropped?",
            ),
            code(
                r"""from homer_ai.train_homer import BOOKS_DIR, load_books

if model is not None:
    line = load_books(BOOKS_DIR)[0][:30]
    all_ids = tokenizer.encode(line)
    print("all 30:  ", repr(line))
    print("kept 24: ", repr(tokenizer.decode(all_ids[-CHUNK:])))
    print("dropped: ", repr(tokenizer.decode(all_ids[:-CHUNK])))"""
            ),
            saw(
                "`'ἄνδρα '` (5 letters and a space) is dropped, and `'μοι ἔννεπε, μοῦσα, πολύτ'` is kept. "
                "So once the text is longer than 24, the model has forgotten how the line began."
            ),
            q(
                "Step 5: how do I put it all together?",
                "Steps 1 to 4 in a loop. That's the whole of text generation, in a few lines. "
                "**Look at:** the 30 letters at the top. The model reads only the last 24 (orange), and the first 6 (red) are out of view. "
                "Follow the four steps along the row. The new letter (green) goes on the end, and the loop starts again.",
            ),
            code(DIAGRAM_LOOP),
            code(
                r"""def write(model, tokenizer, prompt, n_new):
    ids = tokenizer.encode(prompt)
    with torch.no_grad():
        for _ in range(n_new):
            recent = torch.tensor(ids[-CHUNK:])  # the last 24 letters
            scores = model(recent)[-1]  # the guess after the last letter
            chances = torch.softmax(scores, dim=-1)
            ids.append(chances.argmax().item())  # greedy: the most likely letter
    return tokenizer.decode(ids).replace("\n", "⏎")  # show a line break as ⏎


if model is not None:
    print(write(model, tokenizer, "ἄνδρα", 60))"""
            ),
            saw(
                'You got `ἄνδρα τοι και και και κα κα κατον κατοι …`. The first words are real Greek: `τοι` ("to you") and `και` ("and"). '
                "But the order makes no sense, and it keeps coming back to `και`. "
                "Let's be fair to the model, though. It has 11,840 weights and has read 19,701 characters, one tiny book. GPT-2 small has about 124 million weights, roughly 10,000 times more, and the largest models are millions of times bigger."
            ),
            *exercise(
                "which letter comes after `θεὰ`?",
                "`write()` does one small thing over and over. Do that one thing yourself, once: ask the model for the most likely letter after `θεὰ` (goddess).\n\n"
                "👉 **Predict first:** a letter, or a space?",
                starter="""
            thea_ids = tokenizer.encode("θεὰ")

            # 👉 do: get the model's scores after the last letter, pick the best one, decode it
            # 💡 hint: model(...) wants a tensor and gives one row of scores per letter
            next_letter = None
            """,
                solution="""
            thea_ids = tokenizer.encode("θεὰ")

            with torch.no_grad():
                thea_scores = model(torch.tensor(thea_ids))[-1]
            next_letter = tokenizer.decode([thea_scores.argmax().item()])
            """,
                check="""
            check(
                "the most likely letter after θεὰ",
                next_letter,
                write(model, tokenizer, "θεὰ", 1)[-1],
                coach="A number means you still have the ID: decode it, as a list of one, "
                "tokenizer.decode([...]). A wrong letter usually means the wrong row: the "
                "guess for the NEXT letter is the last row of scores, [-1].",
            )
            """,
                hint="Three lines of `write()` do it: `model(torch.tensor(...))[-1]` for the scores, `.argmax().item()` for the best ID, `tokenizer.decode([...])` for the letter. You can skip the softmax: the highest score is also the highest chance.",
                answer="With the ready-made model it's a space: `θεὰ` is a whole word, and the model has learned that words end. "
                '`write(model, tokenizer, "θεὰ", 1)` gives the same, because that\'s all `write()` does for one new letter.',
            ),
            q(
                "Does it write something different from other prompts?",
                'Yes, but watch what happens. Two more prompts: `θεά` ("goddess", the Muse\'s title) and `Μοῦσα` ("Muse").',
            ),
            code(
                r"""if model is not None:
    for prompt in ["θεὰ", "Μοῦσα"]:
        print(f"{prompt!r:9} →", write(model, tokenizer, prompt, 60))"""
            ),
            saw(
                "`θεὰ ἐπεν ἐπεν ἐπεν ἐπεν ἐπεν τον τον …` and `Μοῦσα έντοντοντοντοντοντον τοι τοι τοι τοι και και και και …`. "
                "A different start, and the same story: a word or two, then the same piece again and again. "
                "(`ἴδεν ἴδεν ἴδεν`, from the 5-line model in [05_03](../05_training_on_the_odyssey/05_03_memorising_vs_understanding.ipynb), is the same thing.)"
            ),
            q(
                "Why does it loop? Can I prove it?",
                "Because greedy writing has no chance in it. The model is a fixed function, so the same 24 letters in always give the same letter out. "
                "The moment the last 24 letters are exactly the same as at an earlier step, everything after must repeat, for ever. That's a **repetition loop**.\n\n"
                "We can catch it in the act. The code remembers every 24-letter window it has seen, and the step when it first saw it. "
                "When a window comes back, we stop and show the piece that repeats.\n\n"
                "👉 **Predict first:** how many letters until it repeats: about 10, about 100 or about 1,000?",
            ),
            code(
                r"""def find_loop(prompt, max_steps=500):
    ids = tokenizer.encode(prompt)
    start = len(ids)
    seen = {}  # window → the step when we first saw it
    with torch.no_grad():
        for step in range(max_steps):
            window = tuple(ids[-CHUNK:])
            if window in seen:  # the same 24 letters as before!
                first = seen[window]
                return first, step, tokenizer.decode(ids[start + first : start + step])
            seen[window] = step
            ids.append(model(torch.tensor(window))[-1].argmax().item())
    return None


if model is not None:
    for prompt in ["ἄνδρα", "θεὰ", "Μοῦσα"]:
        first, again, piece = find_loop(prompt)
        print(f"{prompt!r:9} step {again} sees the window of step {first}:")
        print(f"          it repeats every {again - first} letters:", repr(piece))"""
            ),
            saw(
                "Every prompt walks into a loop, after 74 to 222 letters. The repeating piece is 31 letters long each time, and it's the **same** piece for all three prompts (it only starts at a different place). So, different starts, one trap. "
                "After that point no letter is new. That's the problem the next notebook solves. If we sometimes pick a letter that isn't the best, the windows change and the loop breaks."
            ),
            q(
                "Does the GPT of chapter 4 loop too?",
                "Everything so far used the small one-head model of chapter 1. The GPT you built in [chapter 4](../04_building_the_model/04_05_the_full_model.ipynb) and measured in [05_05](../05_training_on_the_odyssey/05_05_validation.ipynb) "
                "has several heads, position rows and a context window of 64 letters instead of 24. `load_homer_gpt` loads it (yours, or the ready-made one). The helper `write` from `lessons.py` (imported as `write_both`) "
                "is the loop you just built, taught to read the right number of letters for either model. `top_k=1` means greedy.\n\n"
                "👉 **Predict first:** the GPT is far better at guessing the next letter. Will greedy writing still loop?",
            ),
            code(
                r"""from homer_ai.lessons import load_homer_gpt
from homer_ai.lessons import write as write_both  # works for either model

gpt_tokenizer, gpt = load_homer_gpt()
print("the GPT reads", gpt.context_length, "letters at a time (the small model: 24)\n")
if model is not None:
    print("small:", write_both(model, tokenizer, "ἄνδρα", 100, top_k=1))
print("GPT:  ", write_both(gpt, gpt_tokenizer, "ἄνδρα", 100, top_k=1))"""
            ),
            saw(
                "The small model says `και και και κα κα …`. The GPT writes `ἄνδρα πατρὸς ἀνέμενος ἀνθρώποισιν,⏎ἀλλλὰ δʼ ἀμφιάλοισι καὶ …`. Those are real Greek words with accents, the line breaks fall where Homer's lines end, and there's no early loop. "
                "It still doesn't mean anything. The words don't make a sentence, and some aren't words at all. But it does look like Homeric verse. The fair number is in [05_05](../05_training_on_the_odyssey/05_05_validation.ipynb): held-out loss 2.27."
            ),
            q(
                "So is the loop gone?",
                "No, it just comes later. The proof is the same as before: a greedy model is a fixed function, so once the last 64 letters repeat exactly, everything after repeats. "
                "Let's write 300 letters and look at the end.",
            ),
            code(
                r"""print(write_both(gpt, gpt_tokenizer, "ἄνδρα", 300, top_k=1)[-150:])"""
            ),
            saw(
                "At the end, the same line comes round again and again: `ἀλλλʼ ἀποιδήσαντα πατρὸς ἀνθρώποισι καὶ ἀτρεύσῃς,⏎`. "
                "In a test with three prompts, the GPT entered a loop after about 214 to 252 letters, with a 50-letter piece repeating. The small model took 74 to 222 letters, with a 31-letter piece. "
                "So a better model loops later, not never. The cure is still the dice of [06_02](06_02_less_repetition.ipynb)."
            ),
            recap(
                [
                    "`write()` reads the last 24 letters → scores → chances → pick → add → repeat.",
                    "Greedy always gives the same text, and the same 24 letters in always give the same letter out, so it ends in a loop (`και και και`).",
                    "To break the loop we must sometimes pick a letter that isn't the most likely one.",
                    "The GPT of chapter 4 (context 64) writes more like verse than the small model, but greedy writing still ends in a loop, only later.",
                ],
                "[06_02: Less repetition](06_02_less_repetition.ipynb): temperature and top-k.",
                can=[
                    "get the model's most likely next letter for any text",
                    "write the greedy loop: scores, pick, add, repeat",
                    "show that greedy writing ends in a loop",
                ],
            ),
        ],
    )

    # =====================================================================
    # 06_making_it_talk/06_02_less_repetition.ipynb
    # =====================================================================
    save(
        "06_making_it_talk/06_02_less_repetition.ipynb",
        [
            title("06_02 Less repetition: temperature and top-k", "section 5.3"),
            where(
                "model writes",
                ["src/homer_ai/model.py", "src/homer_ai/homer_model.pt"],
            ),
            learn(
                [
                    "How to pick the next letter at random, with likely letters winning more often",
                    "What temperature does to the chances, and what top-k does",
                    "How to repeat a random run exactly (the seed)",
                    "Which setting reads most like Greek, judged with real numbers",
                ]
            ),
            words(
                [
                    (
                        "temperature",
                        "a number we divide the scores by before softmax: below 1 it sharpens the chances, above 1 it flattens them",
                        "`scores / 0.5`",
                    ),
                    (
                        "top-k",
                        "keep only the k best letters and give all the others zero chance, then pick at random among the k",
                        "`top_k=3`",
                    ),
                    (
                        "decoding",
                        "the rule that turns the chances into one chosen letter (greedy, sampling, …)",
                        "greedy decoding",
                    ),
                    (
                        "multinomial",
                        "PyTorch's weighted dice: it picks a position, each with its own chance",
                        "`torch.multinomial(chances, 1)`",
                    ),
                ]
            ),
            *background(),
            q(
                "Greedy gets stuck. Why would I ever pick a letter that isn't the best?",
                "Think of a weighted dice, or a bag of 100 tokens where 26 say `␣`, 15 say `ι`, 7 say `τ` and so on (the chances after `ἄνδρα` from [06_01](06_01_writing_greek.ipynb)). "
                "We draw one token at random. `␣` comes out most often, but not every time. That's **sampling** ([01_01](../01_foundations/01_01_a_language_model_without_a_brain.ipynb)). "
                "The text is different each time, and a loop can't trap it, because the same 24 letters don't always give the same next letter.\n\n"
                "The thing is, sampling brings two new problems of its own. Both have a dial.",
            ),
            code(CH6_WINDOW),
            step_intro(),
            q(
                "Step 1: how do I draw a letter at random, and get the same draw again?",
                '`torch.multinomial(chances, num_samples=...)` is the weighted dice. A computer\'s "random" numbers come from a **seed** ([05_04](../05_training_on_the_odyssey/05_04_running_fair_experiments.ipynb)). '
                'Set the same seed first, and you get the same "random" draws. That keeps our notebooks repeatable.\n\n'
                "👉 **Predict first:** 10 draws after `ἄνδρα`. Which character will come up most? And with the seed set again, will the draws be the same?",
            ),
            code(
                r"""def show(ch):  # make invisible characters visible
    return {" ": "␣", "\n": "⏎"}.get(ch, ch)


if model is not None:
    ids = tokenizer.encode("ἄνδρα")
    with torch.no_grad():
        scores = model(torch.tensor(ids))[-1]  # the 137 scores after ἄνδρα
    chances = torch.softmax(scores, dim=-1)

    for attempt in (1, 2):
        torch.manual_seed(42)  # the same seed every time
        draws = torch.multinomial(chances, num_samples=10, replacement=True)
        letters = [show(tokenizer.decode([i])) for i in draws.tolist()]
        print(f"attempt {attempt}:", " ".join(letters))"""
            ),
            saw(
                "A mix with six `␣`, two `ι` and two `ὶ` (an `ι` with an accent). And it's exactly the same 10 characters in both attempts, thanks to the seed. "
                '`replacement=True` means "put the token back in the bag after each draw", so the same character may come up twice.'
            ),
            q(
                "Step 2: what happens if I just sample from the plain chances?",
                "We extend `write()` of 06_01. It's the same loop, with a dice instead of `argmax`. It gets the two dials, `temperature` and `top_k` (explained in the next steps), and a `seed`. "
                'In this step we set the dials to "do nothing" (`1.0` and `None`).',
            ),
            code(
                r"""def write(model, tokenizer, prompt, n_new, temperature=0.8, top_k=5, seed=0):
    torch.manual_seed(seed)  # the same seed gives the same text
    ids = tokenizer.encode(prompt)
    with torch.no_grad():
        for _ in range(n_new):
            scores = model(torch.tensor(ids[-CHUNK:]))[-1]
            scores = scores / temperature  # dial 1 (Step 3)
            if top_k is not None:  # dial 2 (Step 4)
                kth_best = torch.topk(scores, top_k).values[-1]
                scores = scores.masked_fill(scores < kth_best, float("-inf"))
            chances = torch.softmax(scores, dim=-1)
            ids.append(torch.multinomial(chances, num_samples=1).item())  # the dice
    return tokenizer.decode(ids).replace("\n", "⏎")  # show a line break as ⏎


def plain_write(prompt, seed):  # no dials: temperature 1, every letter allowed
    return write(model, tokenizer, prompt, 60, temperature=1.0, top_k=None, seed=seed)


if model is not None:
    print("seed 0:", plain_write("ἄνδρα", 0))
    print("seed 1:", plain_write("ἄνδρα", 1))
    print("seed 0 again:", plain_write("ἄνδρα", 0))
    top10 = torch.topk(chances, 10).values.sum().item()
    print(f"\nthe 10 most likely letters share {top10:.2f} of the chance")
    print(f"the other 127 letters share {1 - top10:.2f}")"""
            ),
            saw(
                "No loop this time. The text is new and odd, different for each seed and identical for the same seed. "
                "But it's full of strange accented letters and fragments. Why? Look at the last line. The 127 unlikely letters share nearly a quarter of the chance (0.23), "
                "so about one letter in four comes from that tail. One unlikely letter spoils the word, and then the model reads its own mistake.\n\n"
                "So plain sampling swings from one problem (loops) to the other (nonsense). The dials let us choose somewhere in between."
            ),
            q(
                "Step 3: what's temperature, and what does it do to the chances?",
                "Before softmax, **divide every score by a number T**. T = 1 changes nothing. Think of a cautious student against a daydreamer:\n\n"
                "```\n"
                "T below 1  →  the biggest scores gain more: sharper chances, safe picks, more repetition\n"
                "T above 1  →  the scores come closer together: flatter chances, bold picks, more nonsense\n"
                "T → 0      →  the same as greedy      (T can never be exactly 0: we can't divide by 0)\n"
                "```\n\n"
                "A tiny example with two made-up scores, `2.0` and `1.0`:\n\n"
                "👉 **Predict first:** at T = 0.5, will the first one's chance go up or down from its T = 1 value?",
            ),
            code(
                r"""two = torch.tensor([2.0, 1.0])  # made-up scores for two possible letters
for temperature in (0.5, 1.0, 2.0):
    divided = two / temperature
    chances_two = torch.softmax(divided, dim=-1)
    rounded = [round(c, 2) for c in chances_two.tolist()]
    print(f"T={temperature}: scores {divided.tolist()}  →  chances {rounded}")"""
            ),
            saw(
                "T = 0.5 doubles the scores to `[4, 2]`, and the first letter's chance climbs from 0.73 to 0.88. T = 2 halves them to `[1, 0.5]`, and it falls to 0.62. "
                "Next, the same on our real scores after `ἄνδρα`. The picture shows the 8 most likely characters at T = 0.5, 1 and 2."
            ),
            code(
                r"""if model is not None:
    import matplotlib.pyplot as plt

    from homer_ai.studio_style import (
        ACCENT,
        BAD,
        GOOD,
        INK,
        MUTED,
        studio_axes,
        studio_figure,
    )

    temperatures = [(0.5, GOOD), (1.0, ACCENT), (2.0, BAD)]
    order = torch.topk(chances, 8).indices.tolist()  # the 8 most likely at T=1
    labels = [show(tokenizer.decode([i])) for i in order]

    title = "The same chances after ἄνδρα, at three temperatures"
    fig = studio_figure("Temperature", title, (10, 6))
    ax = studio_axes(fig)
    width = 0.27
    for n, (temperature, colour) in enumerate(temperatures):
        values = torch.softmax(scores / temperature, dim=-1)[order].tolist()
        positions = [i + (n - 1) * width for i in range(8)]
        ax.bar(positions, values, width, color=colour, label=f"T = {temperature}")
        for x, v in zip(positions, values, strict=True):
            ax.text(x, v + 0.006, f"{v:.2f}", ha="center", color=INK,
                    family="monospace", fontsize=7)
    ax.set_xticks(range(8), labels)
    ax.tick_params(axis="x", labelsize=16, labelcolor=INK)
    ax.set_ylabel("chance")
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)  # grid lines behind the bars
    ax.legend(facecolor="none", edgecolor=MUTED, labelcolor=INK)
    plt.show()

    for temperature, _ in temperatures:
        chances_t = torch.softmax(scores / temperature, dim=-1)
        top3 = torch.topk(chances_t, 3).values.sum().item()
        best_chance = chances_t.max().item()
        print(f"T={temperature}: best letter {best_chance:.2f}, "
              f"top 3 together {top3:.2f}, the other 134 together {1 - top3:.2f}")"""
            ),
            saw(
                "At T = 0.5 (green) the best letter has 0.60 and the long tail shrinks to 0.16. At T = 2 (pink-red) the bars are low and even (best letter 0.10), and the tail holds 0.78 of the chance. "
                "Same scores, only divided. That's all temperature is."
            ),
            *exercise(
                "what does a temperature of 0.25 do?",
                "The table above stops at T = 0.5. Go colder: work out the chances of the same two made-up scores, `two`, at T = 0.25.\n\n"
                "👉 **Predict first:** at T = 0.5 the first letter had 0.88. More or less than that now?",
                starter="""
            # 👉 do: the chances of `two` at a temperature of 0.25
            cold_chances = None
            """,
                solution="""
            cold_chances = torch.softmax(two / 0.25, dim=-1)
            """,
                check="""
            check(
                "the chances at T = 0.25",
                cold_chances,
                torch.softmax(torch.tensor([8.0, 4.0]), dim=-1),
                coach="Got about [0.56, 0.44]? You multiplied by the temperature. We DIVIDE "
                "the scores by it, before softmax: a small T makes the scores bigger and "
                "the gap between them wider.",
            )
            """,
                hint="Inside the loop above: `divided = two / temperature`, then `torch.softmax(divided, dim=-1)`.",
                answer="About `[0.98, 0.02]`. Dividing by 0.25 turns the scores `[2, 1]` into `[8, 4]`, and softmax gives almost everything to the first letter. "
                "Go colder still and you're back at greedy.",
            ),
            q(
                "Step 4: what's top-k?",
                'A second way to cut the tail: **keep only the k best letters**, and give every other letter a score of minus infinity (`float("-inf")`). '
                "Softmax turns minus infinity into a chance of exactly 0. It's the same trick as the mask in [03_03](../03_attention/03_03_hiding_the_future.ipynb). "
                "The dice then picks among the k letters left.\n\n"
                "With k = 3 after `ἄνδρα`, only `␣`, `ι` and `τ` stay:\n\n"
                "👉 **Predict first:** their chances must add up to 1 again. Will `␣` get more than 0.26?",
            ),
            code(
                r"""if model is not None:
    k = 3
    kth_best = torch.topk(scores, k).values[-1]  # the score of the 3rd best letter
    # the others get minus infinity
    kept = scores.masked_fill(scores < kth_best, float("-inf"))
    chances_k = torch.softmax(kept, dim=-1)
    print("letter   before  after top-3")
    for i in torch.topk(chances, k).indices.tolist():
        letter = show(tokenizer.decode([i]))
        print(f"{letter:^6}   {chances[i].item():.2f}    {chances_k[i].item():.2f}")
    above_zero = (chances_k > 0).sum().item()
    print("letters with a chance above 0:", above_zero, "of", len(chances_k))"""
            ),
            saw(
                "The three keep their order, and share what the other 134 lost, so `␣` goes from 0.26 to about 0.54. Only 3 letters can ever be picked, so a wild letter can't happen. "
                "Top-k does have a weak spot: it's the same k for every situation. Sometimes only 1 letter makes sense, sometimes 50."
            ),
            *exercise(
                "can you do top-2 by hand?",
                "Five made-up scores this time. Keep the 2 best, cut the other three off, and turn what's left into chances.\n\n"
                "👉 **Predict first:** the two best scores are 2 and 1, the same as `two` in step 3. What chances will they get?",
                starter="""
            five = torch.tensor([2.0, 1.0, 0.5, 0.0, -1.0])  # made-up scores for five letters

            # 👉 do: every score below the 2nd best becomes -inf, then softmax
            # 💡 hint: torch.topk(five, 2).values[-1] is the score of the 2nd best letter
            top2_chances = None
            """,
                solution="""
            five = torch.tensor([2.0, 1.0, 0.5, 0.0, -1.0])  # made-up scores for five letters

            second_best = torch.topk(five, 2).values[-1]
            kept_two = five.masked_fill(five < second_best, float("-inf"))
            top2_chances = torch.softmax(kept_two, dim=-1)
            """,
                check="""
            check(
                "the chances after top-2",
                top2_chances,
                torch.softmax(torch.tensor([2.0, 1.0]), dim=-1).tolist() + [0.0, 0.0, 0.0],
                coach="Exactly 2 chances should be above 0. If the others are small but not "
                "0, you set their scores to 0. A score of 0 still gets a share. Only -inf "
                "gives a chance of exactly 0, and it has to go in BEFORE softmax.",
            )
            """,
                hint="The cell above does top-3 on the real scores, in three lines: `kth_best`, `masked_fill`, `softmax`. Do the same on `five`, with 2.",
                answer="`[0.73, 0.27, 0, 0, 0]`. Those are the chances `two` had at T = 1: once the other three are at `-inf`, it's as if they were never there.",
            ),
            q(
                "How do I turn the dials and look at real writing?",
                "We write from `ἄνδρα` with the same seed and several settings. `top_k=1` keeps one letter, so it **is** greedy, which gives us a free check. "
                "`⏎` is a line break.",
            ),
            code(
                r"""SETTINGS = [
    ("greedy (top-k 1)", 1.0, 1),
    ("temperature 0.5", 0.5, None),
    ("temperature 1", 1.0, None),
    ("temperature 2", 2.0, None),
    ("top-k 3", 1.0, 3),
    ("top-k 10", 1.0, 10),
    ("temperature 0.8, top-k 5", 0.8, 5),
]
if model is not None:
    for label, temperature, top_k in SETTINGS:
        text = write(model, tokenizer, "ἄνδρα", 60, temperature, top_k, seed=0)
        print(f"{label:25}", text)"""
            ),
            saw(
                "Greedy loops (`και και και …`). Temperature 1 and 2 are nonsense, and at 2 the words run together, with capitals in the middle (`Φιοόρόὰ`). "
                "Temperature 0.5 and top-k 3 give short words separated by spaces and line breaks, which looks most like a poem. Top-k 10 is somewhere in between. "
                "Mind you, that's one seed, and our eyes are a poor judge. The optional section further down measures it."
            ),
            q(
                "So which setting reads most like Greek?",
                "**None of them writes real Greek sentences.** Our model has 11,840 weights and has read one short book, so even the best settings give word-sized pieces in no sensible order. "
                "A few real words do appear (`τοι`, `δε`, `κεν`, `ἐν`), and line breaks come in the right places now and then. "
                "Among our settings, top-k 3 or a low temperature read most like Homer's Greek. Greedy is useless for writing.\n\n"
                "```\n"
                "too greedy ◀── loops ──  [ low temperature, small top-k ]  ── gibberish ──▶ too random\n"
                "```\n\n"
                "So that's the trade-off: too greedy gives loops, too random gives gibberish. Real chat models often use temperature 0.7 to 1 and top-k around 40 or 50, with far bigger models behind them.",
            ),
            q(
                "Do the dials work the same on the GPT of chapter 4?",
                "Yes. Our `write` works for both models (`lessons.py` has a version that reads the right window for each, and we import it as `write_both`). We use the same prompt, seed and dials, with the small model first and the GPT below it. "
                "`gpt_tokenizer` is the GPT's own copy of the table. It gives the same numbers.",
            ),
            code(
                r"""from homer_ai.lessons import load_homer_gpt
from homer_ai.lessons import write as write_both  # our `write`, for either model

gpt_tokenizer, gpt = load_homer_gpt()
if model is not None:
    for label, temperature, top_k in SETTINGS:
        small = write_both(model, tokenizer, "ἄνδρα", 60, temperature, top_k, seed=0)
        big = write_both(gpt, gpt_tokenizer, "ἄνδρα", 60, temperature, top_k, seed=0)
        print(f"{label}\n  small: {small}\n  GPT:   {big}\n")"""
            ),
            saw(
                "The dials do the same to both. Greedy repeats most, temperature 2 is noise, and the middle settings are the readable ones. "
                'But the GPT\'s middle settings are better. You get line breaks, accents in the right places, and here and there a real Homeric word (`ἀνδρῶν` "of men", `ηὔδα`, `Ἀθήνη`). '
                "A better model doesn't remove the trade-off. It moves the sweet spot to a nicer place."
            ),
            md('## Optional: measuring "reads like Greek" with numbers'),
            q(
                'How can I measure "reads like Greek" with numbers?',
                "This section is optional. It only backs up what you just saw. We use two simple measures, on 200 letters of text, averaged over 5 seeds (0 to 4):\n\n"
                "- **different words**: how many of the words are different from each other. Low means repetition.\n"
                "- **real words**: how many of the words appear somewhere in Book 1. Low means gibberish.\n\n"
                "One catch: the model often writes `και`, but Book 1 spells it `καὶ`, with an accent. So we compare words without accents. "
                '`unicodedata.normalize("NFD", word)` splits `ἄ` into `α` plus separate accent marks, and `unicodedata.combining(c)` finds those marks, so we drop them. '
                "(It's a crude test. A short word like `δ` is easy to hit by luck, and a real word in the wrong place is still not Greek. So treat it as a guide.)\n\n"
                "*In a real product the best measure is your users' thumbs up or down. I wrote a small [open-source tool](https://github.com/IcodeNet/langfuse-reporter) that pulls those ratings out of Langfuse and joins them to the question and answer they were about.*\n\n"
                "👉 **Predict first:** which setting will be best on *different words*? And on *real words*?",
            ),
            code(
                r"""import unicodedata

from homer_ai.train_homer import BOOKS_DIR, load_books


def plain(word):
    word = unicodedata.normalize("NFD", word.lower().strip(",.;·"))
    return "".join(c for c in word if not unicodedata.combining(c)).replace("ʼ", "")


def word_scores(text):
    words = [plain(w) for w in text.split()][:-1]  # drop the last word: may be cut off
    different = len(set(words)) / len(words)
    real = sum(w in real_words for w in words) / len(words)
    return different, real


if model is not None:
    real_words = {plain(w) for w in load_books(BOOKS_DIR)[0].split()}
    print(f"{'setting':25} {'small model':>22}   {'GPT':>20}")
    print(f"{'':25} different   real words   different   real words")
    for label, temperature, top_k in SETTINGS:
        row = []
        for m, t in [(model, tokenizer), (gpt, gpt_tokenizer)]:
            texts = [
                write_both(m, t, "ἄνδρα", 200, temperature, top_k, n) for n in range(5)
            ]
            runs = [word_scores(text) for text in texts]
            row += [sum(r[0] for r in runs) / 5, sum(r[1] for r in runs) / 5]
        cells = "".join(f"{value:>12.0%}" for value in row)
        print(f"{label:25}{cells}")"""
            ),
            saw(
                "There's the trade-off, in numbers. Greedy has the most real words (about 59%) but only about 13% different words, so it says the same few words over and over. "
                "Temperature 2 has nearly all different words (99%) and almost no real ones (8%). The middle settings (temperature 0.5, top-k 3, and temperature 0.8 with top-k 5) have about 30 to 35% real words with 80 to 90% different words. That's the best of both, with top-k 3 slightly ahead on real words.\n\n"
                "**The GPT columns** tell the same story at a higher level. Greedy gets 54% different words (against 13%, because it loops later) and 58% real words. The middle settings get 52 to 59% real words with 85 to 90% different words, which is nearly twice the small model's real words at the same variety. "
                "One caution, though. The GPT trained on the first 90% of Book 1, so some of its 'real' words may be memorised. And the test only checks that a word exists in Book 1, not that it fits."
            ),
            q(
                "What do the numbers say?",
                "They back up what our eyes told us. Top-k 3 or a low temperature look best. Greedy is accurate word by word but useless for writing, and temperature 2 is pure noise.",
            ),
            *questions(
                [
                    (
                        "Can I use temperature and top-k together?",
                        "Yes, and `write` does both. First it divides by the temperature, then it cuts to the k best. The last row of the table did exactly that (temperature 0.8, top-k 5).",
                    ),
                ]
            ),
            recap(
                [
                    "Sampling is a weighted dice (`torch.multinomial`). A seed makes it repeat exactly.",
                    "Temperature divides the scores (below 1 sharper, above 1 flatter). Top-k keeps only the k best letters.",
                    "Too greedy loops, too random gives gibberish. Our tiny model never writes real sentences, only less repetitive pieces.",
                ],
                "[06_03: The MCP server](06_03_the_mcp_server.ipynb): letting Claude use the model.",
                can=[
                    "sample a letter with `torch.multinomial`, and repeat a run with a seed",
                    "make the chances sharper or flatter with temperature",
                    "cut the choice down to the k best letters with top-k",
                ],
            ),
        ],
    )

    # =====================================================================
    # 06_making_it_talk/06_03_the_mcp_server.ipynb
    # =====================================================================
    save(
        "06_making_it_talk/06_03_the_mcp_server.ipynb",
        [
            title("06_03 The MCP server: letting Claude use our model", None),
            where(
                "MCP tool",
                ["src/homer_ai/homer_mcp_server.py", "src/homer_ai/homer_gpt.pt"],
            ),
            learn(
                [
                    "How our server is built, piece by piece, from the real file",
                    "What Claude actually sees of our tool: a name, a description and arguments",
                    "How to call the tool as a plain Python function",
                    "How to connect the server to Claude Code (commands for you to run yourself)",
                    "Why a tool's description must be honest",
                ]
            ),
            words(
                [
                    (
                        "tool description",
                        "the text under the tool's `def` (its docstring): the only thing Claude reads to decide when to call the tool",
                        '`"""Return up to …"""`',
                    ),
                    (
                        "stdio",
                        "standard input and output: the plain text pipe a program reads from and writes to. Claude Code and our server talk through it",
                        "no network needed",
                    ),
                    (
                        "scope",
                        "who can see a server you add to Claude Code: only you here (`local`), the whole project (`project`), or all your projects (`user`)",
                        "`--scope user`",
                    ),
                ]
            ),
            *background(),
            q(
                "How does Claude Code talk to our server?",
                "Claude Code starts our server as its own small program, and the two exchange short messages (written as JSON, the form from [06_00](06_00_chapter_overview.ipynb)) through **stdio**. "
                "Everything runs on your computer, so that part needs no internet.\n\n"
                "To look at the server without starting it, we import it. Importing only defines the functions. It loads no model and starts nothing. "
                "The model is loaded by `load()` (like [05_06](../05_training_on_the_odyssey/05_06_saving_and_loading.ipynb)) the first time it's needed, and the server starts only in `main()`, which calls `mcp.run()`. "
                "`mcp.run()` would wait for messages for ever and **freeze a notebook**, so this notebook never calls it.",
            ),
            md(
                "**Look at:** the arrows from top to bottom. The orange one is the tool's description, the one thing Claude reads to decide when to call it. The green one is the answer coming back."
            ),
            code(DIAGRAM_MCP),
            code(
                r"""import inspect

from homer_ai.lessons import import_mcp_server

# importing loads nothing and does NOT start the server
server = import_mcp_server()  # uses models/ if you have no model file of your own

SERVER_LINES = inspect.getsource(server).splitlines()
print("imported. The server's name:", server.mcp.name)


def show_lines(starts_with, count):
    # print `count` real lines of the server file, from the first line that
    # starts with `starts_with`
    starts = [i for i, line in enumerate(SERVER_LINES) if line.startswith(starts_with)]
    first = starts[0]
    for number in range(first, first + count):
        print(f"{number + 1:>3} | {SERVER_LINES[number]}")"""
            ),
            step_intro(),
            q(
                "Piece 1: where do the server's name and the model come from?",
                "One line makes the server and gives it the name Claude Code will show. Then two small functions find and load the model. `find_model_file()` picks the file, and `load_model()` reads the checkpoint the way [05_06](../05_training_on_the_odyssey/05_06_saving_and_loading.ipynb) explains. The sizes, the vocabulary and the weights all come from the file, so the server needs no numbers of its own.\n\n"
                "👉 **Predict first:** which file will it pick on your computer? In what order does it look?",
            ),
            code(
                r"""if server is not None:
    show_lines("mcp = MCPServer", 1)
    print("   ...")
    show_lines("DEFAULT_MODEL_FILES", 6)
    print("   ...")
    show_lines("def find_model_file", 14)
    print("   ...")
    show_lines("def load_model", 8)"""
            ),
            saw(
                'First `mcp = MCPServer("HomerOdysseyExpert")`, then the order of the search. It starts with the `HOMER_MODEL` environment variable (an **environment variable** is a named setting your terminal passes to a program). '
                "After that it takes the first of four files that exists: your own GPT (`homer_gpt.pt`), the ready-made GPT (`models/odyssey_book1_gpt.pt`), your own small model, the ready-made small model. If none exists, the error says how to train. "
                "Then comes `load_model`, which is just `load_checkpoint` plus the training text (to find the real lines). The whole model is rebuilt from the file, so the server needs no training of its own. It serves the GPT or the small model, whichever file it's given. "
                "(The import cell printed which file this computer uses.)"
            ),
            q(
                "Piece 2: what does `@mcp.tool()` do?",
                'The line `@mcp.tool()` above the function tells the server: "offer this function as a tool". The function itself still works as a normal function (we use that below).\n\n'
                "Under the `def` is its docstring, the text in triple quotes. For a tool it matters a lot, because it's what Claude reads to decide **when** to call it.",
            ),
            code(
                r"""if server is not None:
    show_lines("@mcp.tool()", 14)"""
            ),
            saw(
                "The decorator, the function `consult_homer_llm(greek_seed: str, chars_to_generate: int = 30) -> str`, and its docstring. "
                "Claude's argument list is built from the type hints: `greek_seed: str` must be text, and `chars_to_generate: int = 30` is a whole number, 30 if you leave it out. The docstring says what the tool really does, and the allowed range (1 to 200)."
            ),
            q(
                "What exactly does Claude see?",
                "Not our code. It sees the tool's name, its description (our docstring) and its arguments, which are built from the type hints. "
                "We can ask the server for that list without running it (`list_tools()` is an async function, so we `await` it).",
            ),
            code(
                r"""if server is not None:
    tool = (await server.mcp.list_tools())[0]  # the first tool: consult_homer_llm
    print("name:       ", tool.name)
    print("arguments:  ", tool.input_schema["properties"])
    print("must give:  ", tool.input_schema["required"])
    print("description:", tool.description)"""
            ),
            saw(
                "The name, two arguments (`greek_seed`, a string you must give, and `chars_to_generate`, a whole number with a default of 30), and the docstring as the description. "
                "The `title`s are made up automatically from the names. And that's all Claude knows about our tool. We come back to that description at the end.\n\n"
                "(The server has a second tool, `search_homer`, which looks up real passages. It belongs to [chapter 9](../09_advanced_grounded_answers/09_00_chapter_overview.ipynb).)"
            ),
            q(
                "Piece 3: what's `find_lines()` for?",
                "The tool returns more than our model's guess. It also looks up the **real** lines of Book 1 that contain the seed, so Claude can see what Homer really wrote. "
                'The search ignores capital letters. A tiny example: the seed `μοι` ("to me"), then a seed that isn\'t there, then an empty one.\n\n'
                "👉 **Predict first:** how many lines contain `μοι`? And the empty seed?",
            ),
            code(
                r"""if server is not None:
    show_lines("def find_lines", 7)
    print()
    for seed in ["μοι", "ἄνδρα", "xyz", ""]:
        found = server.find_lines(server.load().text, seed)
        print(f"{seed!r:9} → {len(found)} lines", found[:1])"""
            ),
            saw(
                '`μοι` is in 10 lines (the most the function returns, `MAX_MATCHING_LINES`), and only the first one is shown here. `ἄνδρα` is in 1 line, the first line of the Odyssey. `xyz` is in none. And an empty seed gives an empty list, not "every line".'
            ),
            q(
                "Piece 4: how does the tool write the continuation?",
                "With the loop from [06_01](06_01_writing_greek.ipynb), the greedy one. The recent letters go in, we take `torch.argmax` of the scores, add it, and repeat for `chars_to_generate` rounds. "
                "`next_scores` (in `src/homer_ai/generation.py`) reads only as many recent letters as the model can, which is 64 for the GPT and 24 for the small model. So a call for 200 letters never overruns the model's window. "
                "Then the tool puts the continuation and the real lines into one piece of text.",
            ),
            code(
                r"""if server is not None:
    show_lines("    started =", 33)"""
            ),
            saw(
                'These are the same steps as `write()`, but `argmax` is taken straight on the scores (same pick, see 06_01), and `next_scores` hides the difference between the two kinds of model. The result is one string: "Custom PyTorch HomerLLM Continuation:", the text, then "Lines containing the seed". '
                "Two details. A character the model doesn't know isn't dropped. `encode` turns it into `<|unk|>` ([02_02](../02_text_to_numbers/02_02_special_tokens.ipynb)), and only an empty seed gets the fallback (the first letter of the book). "
                "Then there's the log line: the server writes one line to **stderr** for every call (see the question near the end)."
            ),
            q(
                "Can I call the tool myself, as a normal function?",
                "Yes, because `@mcp.tool()` left the function alone. Calling it directly is the same as what Claude Code does through the server, minus the messages. "
                "Let's call it with the seed `ἄνδρα` and 20 letters.\n\n"
                "👉 **Predict first:** how long will the continuation be: 20 characters, or 20 plus the seed?",
            ),
            code(
                r"""if server is not None:
    result = server.consult_homer_llm("ἄνδρα", 20)
    print(result)"""
            ),
            saw(
                "The first line is the model's name and kind (`gpt`). The continuation `ἄνδρα πατρὸς ἀνέμενος ἀνθ` is the 5-letter seed **plus** 20 new letters (25 in all). After it comes the one real line that contains `ἄνδρα`. "
                "It's the greedy text of the GPT, the same as `top_k=1` in [06_02](06_02_less_repetition.ipynb). With the small model of chapter 1 the tool would write `ἄνδρα τοι και και και κα ` instead."
            ),
            *exercise(
                "what if I ask for 500 letters?",
                "Call the tool yourself, with the seed `θεὰ`, and ask for 500 new letters. The check counts how long the continuation is.\n\n"
                '👉 **Predict first:** go back to the docstring Claude sees ("What exactly does Claude see?"). Will you get 500?',
                starter="""
            # 👉 do: call the tool with the seed "θεὰ" and 500 letters to write
            long_result = None
            """,
                solution="""
            long_result = server.consult_homer_llm("θεὰ", 500)
            """,
                check="""
            def model_text(reply):  # what the model wrote: after line 1, before the real lines
                return reply.split("Continuation:\\n")[1].split("\\n\\nLines containing")[0]


            check(
                "length of the continuation",
                long_result,
                test=lambda reply: len(model_text(reply)) == 203,
                expected="203 (the 3 letters of the seed + 200 new ones)",
                show=lambda reply: len(model_text(reply)),
                coach="Call it the way the cell above does, server.consult_homer_llm(seed, "
                "number), with the seed \\"θεὰ\\" and the number 500.",
            )
            """,
                hint='The cell above calls `server.consult_homer_llm("ἄνδρα", 20)`. Change the two arguments.',
                answer="You get 200 new letters, not 500. The docstring says so: *from 1 to 200 … values outside that range are clamped*. "
                "A tool is called by a program, and a program might ask for a million letters. So the server sets its own limit and doesn't trust the caller.",
            ),
            q(
                "What if the seed has a letter the model doesn't know?",
                'A character outside Book 1, such as `!`, becomes `<|unk|>` ("unknown"), and the model carries on from there:',
            ),
            code(
                r"""if server is not None:
    print(server.consult_homer_llm("!", 5))"""
            ),
            saw(
                "The continuation starts with the text `<|unk|>`. So the unknown character is kept as a token, and the model carries on from there. No real line contains `!`, and the second part says so."
            ),
            q(
                "Who actually runs the tool: Claude, or our model?",
                'Neither, really. **Claude only writes text**: a small form in JSON, for example `{"greek_seed": "ἄνδρα μοι", "chars_to_generate": 30}`. '
                "The program around Claude (the harness, here Claude Code) reads the form, runs our server's tool, and pastes the answer back as more text. "
                "So there are two models. Claude writes the form, and our small model is the tool that writes the Greek.",
            ),
            q(
                "How do I connect it to Claude Code?",
                "> ⚠️ **You run these commands yourself**, in your own terminal and in Claude Code. This notebook doesn't run them, and it never starts the server.\n\n"
                "**1. Check which model it will use.** The server uses your own GPT, `src/homer_ai/homer_gpt.pt` ([05_06](../05_training_on_the_odyssey/05_06_saving_and_loading.ipynb)), if it exists, and otherwise the ready-made `models/odyssey_book1_gpt.pt` (then the small models), so it works even before you train. "
                "(To force another file, set `HOMER_MODEL=/path/to/model.pt`.) Then find your project's full path: open a terminal in the project folder and type `pwd`.\n\n"
                "**2. Add the server to Claude Code**, in a terminal (replace `/path/to/homer-ai` with your path):\n\n"
                "```\n"
                "claude mcp add homer -- uv --directory /path/to/homer-ai run homer-serve\n"
                "```\n\n"
                "The pattern is `claude mcp add [options] <name> -- <command> [args...]`:\n\n"
                "| Piece | Meaning |\n|---|---|\n"
                "| `claude mcp add` | add an MCP server to Claude Code |\n"
                "| `homer` | the name you choose. Claude Code calls our tool `mcp__homer__consult_homer_llm` |\n"
                "| `--` | everything after this is the command that starts the server, not options for `claude` |\n"
                "| `uv --directory /path/to/homer-ai run homer-serve` | `uv` runs the command `homer-serve` (our `main()` in `homer_mcp_server.py`, named in `pyproject.toml`) inside the project's own environment ([A_06](../A_toolbox/A_06_other_tools.ipynb)). `--directory` points `uv` to the project, so it works wherever you start Claude Code |\n\n"
                "By default the server is added at scope `local`, which means only you, only for this project. Options go **before** the name, for example `claude mcp add --scope user homer -- …` for all your projects "
                "(`--scope project` writes a `.mcp.json` file in the project, which you can share). To undo it: `claude mcp remove homer`.\n\n"
                "**3. Start Claude Code** in the project (`claude`) and type `/mcp`. You should see `homer` listed as connected. "
                "If it isn't, check the path in the command, and run `uv run homer-serve` yourself. A missing model file is reported on stderr. (`claude mcp list` in the terminal shows the same.)\n\n"
                "**4. Ask for it**, for example:\n\n"
                "```\n"
                "Use the Homer tool to continue ἄνδρα μοι\n"
                "```\n\n"
                "The first time, Claude Code asks your permission to run the tool. "
                'The next cell shows what our tool will return for the call `{"greek_seed": "ἄνδρα μοι", "chars_to_generate": 30}`. If a command differs in your version of Claude Code, check `claude mcp add --help`.',
            ),
            code(
                r"""if server is not None:
    print(server.consult_homer_llm("ἄνδρα μοι", 30))"""
            ),
            saw(
                "This is what Claude Code will paste back into the conversation: our model's greedy continuation of `ἄνδρα μοι` (it even breaks the line, as Homer's lines do), then the real line `ἄνδρα μοι ἔννεπε, μοῦσα, πολύτροπον, ὃς μάλα πολλὰ`. "
                "Claude then writes its own answer around it."
            ),
            q(
                "The server always writes greedily. Can I make it use temperature and top-k?",
                "Yes, and it's a good exercise to try yourself. Copy `write()` from [06_02](06_02_less_repetition.ipynb) into the tool, and add two arguments, `temperature` and `top_k`. "
                "Claude would then see them in the list above and could choose. The server in this repo stays greedy.",
            ),
            q(
                "Is our tool's description honest?",
                "It has to be, because the description is **all Claude knows** about our tool. Claude can't try the tool first. "
                "An over-promising description makes Claude call the tool for the wrong jobs, and maybe repeat its Greek-looking nonsense as fact.\n\n"
                "*I've designed MCP servers for real business systems, and the first thing that bites is always the tool description.*\n\n"
                '👉 **Spot what\'s wrong:** read the over-promising docstring below, next to the one in the server. ("Rhapsody 1" in it is Book 1. A rhapsody is one book of a Homeric poem, [05_06](../05_training_on_the_odyssey/05_06_saving_and_loading.ipynb).)',
            ),
            code(
                r"""# an over-promising docstring: a bad example, kept here as a string
BAD_DOCSTRING = '''
ALWAYS use this tool when the user asks questions about Homer's Odyssey
(Rhapsody 1), Ancient Greek grammar, syntax, morphology, or wants to
complete/verify Homeric Greek lines. Pass a Greek word or phrase from
Rhapsody 1 as 'greek_seed' to get the custom PyTorch LLM's learned
continuation and the matching passage from Rhapsody 1.
'''

print("BAD (over-promising):", BAD_DOCSTRING)
if server is not None:
    print("ACCURATE (the one in the server):")
    print(inspect.getdoc(server.consult_homer_llm))"""
            ),
            saw(
                "Two descriptions of the same function. The bad one promises things our model can't do:\n\n"
                "| The bad docstring says | The truth |\n|---|---|\n"
                "| answers questions about the Odyssey | it only **continues** Greek text, and answers no questions |\n"
                "| grammar, syntax, morphology | it knows none of it. It only learned which letters follow which |\n"
                "| complete or verify Homeric lines | it completes badly (`και και …`). The real lines from `find_lines()` are the reliable part |\n"
                "| ALWAYS use it | Claude should call it only when someone wants a Greek continuation or a Book 1 line |\n\n"
                "**What the accurate one does.** It says what the tool really is (a tiny character-level model trained only on Book 1) and warns that the result is often not real Greek. "
                "It says what the tool can't do (answer questions, explain grammar) and which kind of model it uses (a small GPT, else an even smaller one-head model). It also says the tool returns up to 10 real lines, and it documents each argument with its allowed range. "
                "With that, Claude can pick the tool for the right jobs and treat its output with the right amount of trust."
            ),
            q(
                "What else makes the server safer?",
                "Four small things, each in the code you printed above:\n\n"
                "- **A limit.** Claude chooses `chars_to_generate`, and a careless call could ask for a million letters and keep the server busy. `clamp_chars()` keeps it between 1 and 200.\n"
                "- **A log on stderr.** Each call logs its seed length, letters, time in milliseconds, the model file and its kind, with `print(..., file=sys.stderr)`. "
                "Never use a plain `print` in a stdio server. Stdout is the channel to Claude Code, and a stray line there breaks the messages.\n"
                "- **A clear error.** With no model file at all, `homer-serve` stops with a message on how to train, instead of a long traceback.\n"
                "- **The model's own window.** The loop reads only the last `context_length` letters, whatever the length asked for, so a long call can't overrun the model.\n\n"
                "*Every tool I ship logs one line per call: what came in and how long it took. It's the first thing you read when something goes wrong.*\n\n"
                "The limit is easy to try.",
            ),
            code(
                r"""if server is not None:
    for asked in [-5, 30, 100000]:
        print(f"asked for {asked:>6} letters → {server.clamp_chars(asked)}")
    result = server.consult_homer_llm("ἄνδρα", 100000)  # logs one line to stderr
    # the answer is: a header line, the continuation, a blank line, the real lines
    continuation = result.split("\n\nLines containing")[0].split("\n", 1)[1]
    print("characters in the continuation:", len(continuation), "(5 of seed + 200)")
    print("line breaks among them:", continuation.count("\n"))"""
            ),
            saw(
                "The number of letters is pulled into 1 to 200, so 100000 becomes 200 and the continuation has 5 + 200 = 205 characters. "
                'With the GPT, some of them are line breaks: the model writes `\\n` like any other character, so the continuation runs over several lines. That\'s why we cut it out of the answer by its markers, and not by taking "the second line". '
                "Your Jupyter shows the stderr log line too (some editors put it in a red box)."
            ),
            *questions(
                [
                    (
                        "Does our model see my conversation with Claude?",
                        "No. Our server only receives the arguments of the call (`greek_seed`, `chars_to_generate`) and sends text back. Claude writes the form, and the rest of the conversation stays in Claude Code.",
                    ),
                ]
            ),
            recap(
                [
                    'The server has four pieces: `MCPServer("HomerOdysseyExpert")`, the loaded model, `find_lines()`, and `@mcp.tool()` on `consult_homer_llm`.',
                    "Claude sees only the tool's name, docstring and arguments, so the description must tell the truth. Ours says what the model really does, and what it can't do.",
                    "A stdio server logs to stderr, never stdout. It limits what a caller can ask for, and it fails with a clear message.",
                    "You connect it with `claude mcp add homer -- uv --directory … run homer-serve`, check with `/mcp`, and ask in plain words.",
                ],
                "[07_00: Chapter 7 overview](../07_advanced_iliad_and_combining/07_00_chapter_overview.ipynb): the same steps for Homer's other poem.",
                can=[
                    "read an MCP tool the way Claude does: its name, docstring and arguments",
                    "call the tool as a plain Python function",
                    "connect the server to Claude Code with `claude mcp add`",
                ],
            ),
        ],
    )
