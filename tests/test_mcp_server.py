import asyncio

import pytest

from homer_ai import homer_mcp_server as server


def test_find_lines_empty_seed_gives_nothing(book1):
    assert server.find_lines(book1, "") == []
    assert server.find_lines(book1, "   ") == []


def test_find_lines_ignores_case(book1):
    assert server.find_lines(book1, "ἌΝΔΡΑ") == server.find_lines(book1, "ἄνδρα")
    assert len(server.find_lines(book1, "ἄνδρα")) >= 1


def test_find_lines_respects_the_limit(book1):
    assert len(server.find_lines(book1, "μοι")) == server.MAX_MATCHING_LINES
    assert len(server.find_lines(book1, "μοι", limit=3)) == 3


def test_find_lines_no_match(book1):
    assert server.find_lines(book1, "xyz") == []


@pytest.mark.parametrize(
    ("asked", "used"), [(-5, 1), (0, 1), (1, 1), (30, 30), (200, 200), (10_000, 200)]
)
def test_chars_to_generate_is_clamped(asked, used):
    assert server.clamp_chars(asked) == used


def test_tool_output_length_follows_the_clamp():
    seed = "ἄνδρα"
    for asked, used in [(0, 1), (10_000, 200)]:
        result = server.consult_homer_llm(seed, asked)
        continuation = result.split("\n")[1]
        assert len(continuation) == len(seed) + used


def test_logs_go_to_stderr_not_stdout(capsys):
    server.consult_homer_llm("ἄνδρα", 5)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "seed_chars=5 chars=5" in captured.err
    assert "odyssey_book1.pt" in captured.err


def test_missing_model_gives_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.setenv("HOMER_MODEL", str(tmp_path / "nope.pt"))
    with pytest.raises(FileNotFoundError, match="HOMER_MODEL"):
        server.find_model_file()


def test_list_tools_smoke():
    tools = asyncio.run(server.mcp.list_tools())
    assert [tool.name for tool in tools] == ["consult_homer_llm", "search_homer"]
    schema = tools[0].input_schema
    assert set(schema["properties"]) == {"greek_seed", "chars_to_generate"}
    assert schema["required"] == ["greek_seed"]
    assert "1 to 200" in tools[0].description


def test_search_tool_schema():
    tool = asyncio.run(server.mcp.list_tools())[1]
    schema = tool.input_schema
    assert set(schema["properties"]) == {"question", "poem", "language", "k"}
    assert schema["required"] == ["question"]
    assert schema["properties"]["poem"]["enum"] == ["odyssey", "iliad", "both"]
    assert schema["properties"]["language"]["enum"] == ["english", "greek"]
    assert "1 to 10" in tool.description
    assert "not meaning" in tool.description


@pytest.mark.parametrize(
    ("asked", "used"), [(-3, 1), (0, 1), (1, 1), (5, 5), (10, 10), (500, 10)]
)
def test_k_is_clamped(asked, used):
    assert server.clamp_k(asked) == used
    result = server.search_homer("Cyclops", k=asked)
    assert result.count("] (score ") == used


def test_search_tool_is_deterministic_and_cites_real_lines():
    question = "What false name does Odysseus give to the Cyclops?"
    result = server.search_homer(question, k=3)
    assert result == server.search_homer(question, k=3)
    assert result.startswith("[odyssey 9:24] (score ")
    assert "my name is Noman" in result
    only_iliad = server.search_homer("Achilles Hector", poem="iliad", k=10)
    assert "[odyssey" not in only_iliad
    greek = server.search_homer("μηνιν αειδε", "iliad", "greek", 1)
    assert greek.startswith("[iliad 1:1-5]")


def test_search_tool_cuts_long_passages_and_handles_no_match():
    result = server.search_homer("Hector Achilles Priam", k=10)
    for part in result.split("\n\n"):
        text = part.split("\n", 1)[1]
        assert len(text) <= server.MAX_PASSAGE_CHARS + 1
    assert server.search_homer("qwertyuiop") == (
        "No passage shares a word with this question."
    )


def test_search_tool_logs_to_stderr_not_stdout(capsys):
    server.search_homer("Sirens wax", poem="odyssey", k=2)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "search question_chars=10 poem=odyssey language=english k=2" in captured.err


def test_importing_the_server_builds_no_index():
    import subprocess
    import sys

    code = (
        "from homer_ai import homer_mcp_server, retrieval; "
        "print(len(retrieval._indexes))"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert done.stdout.strip() == "0"


def make_checkpoint(path, kind, tokenizer):
    """Write a tiny untrained checkpoint of the given kind."""
    from homer_ai.checkpoint import save_checkpoint
    from homer_ai.gpt import GPTConfig, GPTModel
    from homer_ai.model import HomerGreekLLM

    if kind == "gpt":
        config = GPTConfig(
            vocab_size=tokenizer.vocab_size,
            context_length=16,
            emb_dim=16,
            n_heads=2,
            n_layers=1,
        )
        model, config = GPTModel(config), config.to_dict()
    else:
        config = {"vocab_size": tokenizer.vocab_size, "embed_dim": 32}
        model = HomerGreekLLM(tokenizer.vocab_size, embed_dim=32)
    meta = {"books": ["book_01.txt"], "poem": "odyssey"}
    save_checkpoint(path, model, tokenizer, kind, config, meta)


def test_default_order_and_fallback(monkeypatch, tmp_path):
    names = ["own_gpt.pt", "ready_gpt.pt", "own_small.pt", "ready_small.pt"]
    files = [tmp_path / name for name in names]
    monkeypatch.setattr(server, "DEFAULT_MODEL_FILES", tuple(files))
    monkeypatch.delenv("HOMER_MODEL", raising=False)

    with pytest.raises(FileNotFoundError, match="homer-train --model gpt"):
        server.find_model_file()
    for file in reversed(files):  # create the files from the last choice upwards
        file.write_bytes(b"")
        assert server.find_model_file() == file

    env_file = tmp_path / "from_env.pt"
    env_file.write_bytes(b"")
    monkeypatch.setenv("HOMER_MODEL", str(env_file))  # $HOMER_MODEL beats them all
    assert server.find_model_file() == env_file


def test_real_default_order_puts_the_gpt_first():
    names = [path.name for path in server.DEFAULT_MODEL_FILES]
    assert names == [
        "homer_gpt.pt",
        "odyssey_book1_gpt.pt",
        "homer_model.pt",
        "odyssey_book1.pt",
    ]


def test_server_serves_a_gpt_and_respects_its_context(
    monkeypatch, tmp_path, tokenizer, capsys
):
    path = tmp_path / "gpt.pt"
    make_checkpoint(path, "gpt", tokenizer)
    monkeypatch.setenv("HOMER_MODEL", str(path))
    monkeypatch.setattr(server, "_loaded", None)
    # 100 new letters from a model that reads only 16: must not fail
    result = server.consult_homer_llm("ἄνδρα", 100)
    assert len(result.split("\n")[1]) == len("ἄνδρα") + 100
    assert "kind=gpt" in capsys.readouterr().err
    monkeypatch.setattr(server, "_loaded", None)  # do not leak into other tests


def test_server_serves_a_small_model_checkpoint_of_format_2(
    monkeypatch, tmp_path, tokenizer
):
    path = tmp_path / "small.pt"
    make_checkpoint(path, "char-1head", tokenizer)
    monkeypatch.setenv("HOMER_MODEL", str(path))
    monkeypatch.setattr(server, "_loaded", None)
    loaded = server.load()
    assert loaded.kind == "char-1head"
    assert len(server.consult_homer_llm("ἄνδρα", 30).split("\n")[1]) == 35
    monkeypatch.setattr(server, "_loaded", None)
