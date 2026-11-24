# Appendix A: The toolbox

## ❓ Why does this chapter matter?

Building an LLM uses a handful of tools again and again: Python, PyTorch, a bit of maths (the dot product), tiktoken, and the everyday tools uv, Ruff, Jupyter, git and MCP. This appendix explains each one, and shows where this repo uses it.

You don't read it first. **Dip in whenever you meet a tool you don't know yet.** New to tensors? See A_02. Meeting `@` in the model? See A_03.

The theory, the new words and other learners' questions live in the overview notebook, [A_00_overview.ipynb](A_00_overview.ipynb).

## ❓ Which notebook do I open first?

Open them in this order.

| Notebook | What it teaches |
|---|---|
| [A_00_overview.ipynb](A_00_overview.ipynb) | Every tool in one line, and which chapter needs it |
| [A_01_python_essentials.ipynb](A_01_python_essentials.ipynb) | Variables, lists, dictionaries, loops, functions, classes |
| [A_02_pytorch_tensors.ipynb](A_02_pytorch_tensors.ipynb) | What PyTorch is, then tensors, shape, dtype, slicing and the seed |
| [A_03_dot_product.ipynb](A_03_dot_product.ipynb) | The dot product, matrix multiplication, and `Q @ K.T` |
| [A_04_pytorch_in_this_repo.ipynb](A_04_pytorch_in_this_repo.ipynb) | Embedding, Linear, softmax, the loss, backward and the optimizer |
| [A_05_tiktoken_bpe.ipynb](A_05_tiktoken_bpe.ipynb) | GPT-2's tokenizer on English and on Greek |
| [A_06_other_tools.ipynb](A_06_other_tools.ipynb) | uv, Ruff, Jupyter, git and MCP |
| [A_07_organising_code.ipynb](A_07_organising_code.ipynb) | Why the model lives in one shared file, `model.py` |

## ❓ What should I know before I start?

Nothing. Open any notebook here whenever you need it.

## ❓ Where is this in the project code?

- `pyproject.toml`: the shopping list of libraries
- `src/homer_ai/model.py`: the model, built from PyTorch parts (A_04)
- `src/homer_ai/train_homer.py`: the training loop (A_04) and the data as a tensor (A_02)

## ❓ Do I need the book?

No, everything is explained here. If you do have it, this chapter matches Appendix A (Introduction to PyTorch) covers part of this.
