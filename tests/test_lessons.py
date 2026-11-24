import math

import torch

from homer_ai import lessons
from homer_ai.lessons import evaluate, split_90_10


def test_split_90_10_sizes():
    train, held_out = split_90_10(list(range(100)))
    assert (len(train), len(held_out)) == (90, 10)
    assert train + held_out == list(range(100))  # no overlap, nothing lost


def test_evaluate_returns_a_finite_float(tiny_model):
    data = torch.randint(0, 20, (100,))
    loss = evaluate(tiny_model, data, chunk_size=10)
    assert isinstance(loss, float)
    assert math.isfinite(loss)
    assert loss > 0


def test_practice_data_is_the_first_five_lines():
    text, tokenizer, data = lessons.practice_data()
    assert len(text.splitlines()) == 5
    assert tokenizer.decode(data.tolist()) == text


def test_tiny_train_is_repeatable_and_learns():
    first = lessons.tiny_train(seed=42, epochs=3)
    again = lessons.tiny_train(seed=42, epochs=3)
    assert first == again  # the same seed gives the same numbers
    assert len(first) == 3
    assert first[-1] < first[0]  # the loss goes down


def test_tiny_train_cosine_changes_only_the_schedule():
    constant = lessons.tiny_train(seed=42, epochs=3)
    cosine = lessons.tiny_train(seed=42, epochs=3, cosine=True)
    assert cosine[0] == constant[0]  # the first epoch still runs at the full lr
    assert cosine != constant


def test_tiny_model_returns_a_model_that_reads_its_own_tokenizer():
    model, tokenizer, losses = lessons.tiny_model(seed=0, epochs=1)
    scores = model(torch.tensor(tokenizer.encode("ἄνδρα")))
    assert scores.shape == (5, tokenizer.vocab_size)
    assert len(losses) == 1


# ---- check(): the helper behind every "Your turn" exercise ----


def printed(capsys, *args, **kwargs):
    """Call check() and return the lines it printed."""
    assert lessons.check(*args, **kwargs) is None
    return capsys.readouterr().out.strip().splitlines()


def test_check_says_not_filled_in_for_none(capsys):
    lines = printed(capsys, "the IDs", None, [1, 2], coach="use encode")
    assert len(lines) == 1  # no coaching yet: there is nothing to coach
    assert lines[0].startswith("⏳ the IDs")
    assert "replace the None" in lines[0]


def test_check_says_right_and_shows_the_answer(capsys):
    assert printed(capsys, "the IDs", [58, 38], [58, 38], coach="use encode") == [
        "✅ the IDs: [58, 38]"
    ]


def test_check_says_wrong_then_coaches(capsys):
    lines = printed(capsys, "w after one nudge", 0.7, 1.3, coach="you added it")
    assert lines == [
        "❌ w after one nudge: expected 1.3, got 0.7",
        "🚨 you added it",
    ]


def test_check_without_coach_prints_one_line(capsys):
    assert printed(capsys, "a number", 1, 2) == ["❌ a number: expected 2, got 1"]


def test_check_compares_tensors_lists_and_tuples_by_their_numbers(capsys):
    mask = torch.triu(torch.ones(2, 2) * float("-inf"), diagonal=1)
    assert printed(capsys, "mask", mask, [[0.0, float("-inf")], [0.0, 0.0]])[
        0
    ].startswith("✅")
    assert printed(capsys, "shape", torch.ones(3, 32).shape, (3, 32)) == [
        "✅ shape: (3, 32)"
    ]
    assert printed(capsys, "ids", torch.tensor([1, 2]), [1, 2, 3])[0].startswith("❌")
    assert printed(capsys, "ids", torch.tensor([1, 2]), [1, 3])[0].startswith("❌")


def test_check_allows_a_rounding_error_but_not_more(capsys):
    assert printed(capsys, "a third", 0.33333, 1 / 3)[0].startswith("✅")
    assert printed(capsys, "a third", 0.33, 1 / 3)[0].startswith("❌")
    assert printed(capsys, "a third", torch.tensor(1 / 3), 1 / 3)[0].startswith("✅")
    assert printed(capsys, "nan", float("nan"), 1.0)[0].startswith("❌")


def test_check_with_a_test_function(capsys):
    def below_3(loss):
        return loss < 3

    assert printed(capsys, "loss", torch.tensor(2.5), test=below_3) == ["✅ loss: 2.5"]
    lines = printed(
        capsys, "loss", 4.0, test=below_3, expected="a loss below 3", coach="step()"
    )
    assert lines == ["❌ loss: expected a loss below 3, got 4.0", "🚨 step()"]


def test_check_show_makes_the_message_readable(capsys):
    assert printed(capsys, "inputs", [1, 2], [1, 2, 3], show=len) == [
        "❌ inputs: expected 3, got 2"
    ]
    shown = printed(capsys, "inputs", "μοι ", "μοι ", show=str.upper)
    assert shown == ["✅ inputs: 'ΜΟΙ '"]  # repr: the space at the end stays visible


def test_check_shortens_a_long_answer(capsys):
    (line,) = printed(capsys, "big", list(range(1000)), list(range(1000)))
    assert line.endswith("…")
    assert len(line) < 90


def test_check_never_raises(capsys):
    def broken(_answer):
        raise ZeroDivisionError("oops")

    # a test that fails, an answer of the wrong kind, a `show` that cannot read it
    lines = printed(capsys, "a", 1, test=broken, coach="look here")
    assert lines[0].startswith("❌ a: the check couldn't read your answer")
    assert lines[1] == "🚨 look here"
    assert printed(capsys, "b", object(), [1, 2])[0].startswith("❌ b: expected [1, 2]")
    assert printed(capsys, "c", 5, [1, 2], show=len)[0].startswith("❌ c")
    assert printed(capsys, "d", "text", 1.5)[0].startswith("❌ d")
    assert printed(capsys, "e", {"a": 1}, torch.ones(2))[0].startswith("❌ e")
