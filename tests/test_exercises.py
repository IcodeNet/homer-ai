"""The "Your turn" exercises: the cells they need, and how the runner reads them."""

import ast
import importlib.util
from pathlib import Path

import nbformat

ROOT = Path(__file__).parent.parent
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*/*.ipynb"))


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_notebooks", ROOT / "scripts" / "run_notebooks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_runner()


def tagged(notebook, tag):
    return [c for c in notebook.cells if runner.has_tag(c, tag)]


def test_every_exercise_has_a_starter_a_check_and_a_solution():
    exercises = 0
    for path in NOTEBOOKS:
        notebook = nbformat.read(path, as_version=4)
        starters = tagged(notebook, "exercise-starter")
        assert len(starters) == len(tagged(notebook, "exercise-check")), path.name
        assert len(starters) == len(tagged(notebook, "exercise-solution")), path.name
        for starter in starters:
            assert "None" in starter.source, path.name  # a blank to fill in
            assert "# 👉 do:" in starter.source, path.name
            ast.parse(starter.source)  # it runs as it is
        exercises += len(starters)
    assert exercises >= 20


def test_use_solutions_fills_in_every_blank():
    for path in NOTEBOOKS:
        notebook = nbformat.read(path, as_version=4)
        count = runner.use_solutions(notebook)
        starters = tagged(notebook, "exercise-starter")
        assert count == len(starters)
        for starter in starters:
            assert "= None" not in starter.source, path.name
            assert "👉" not in starter.source, path.name
            ast.parse(starter.source)


def test_a_blank_run_expects_only_hourglasses():
    assert runner.wrong_checks([["⏳ a. Not filled in yet"]], solutions=False) is None
    assert runner.wrong_checks([["✅ a: 1"]], solutions=False) is not None
    assert runner.wrong_checks([[]], solutions=False) is not None  # printed nothing


def test_a_solutions_run_expects_only_ticks():
    assert runner.wrong_checks([["✅ a: 1", "✅ b: 2"]], solutions=True) is None
    assert runner.wrong_checks([["✅ a: 1", "⏳ b."]], solutions=True) is not None
    wrong = [["❌ a: expected 2, got 1", "🚨 look here"]]
    assert runner.wrong_checks(wrong, solutions=True) is not None
    assert runner.wrong_checks([[]], solutions=True) is not None
