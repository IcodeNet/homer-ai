# 0003. Ship small trained models in git

Status: Accepted

## Context

Training the GPT takes 2.9 minutes on a laptop CPU, and the small model about 8 minutes (`models/README.md`). That's quick, but a learner who hasn't trained anything should still be able to open any notebook and see it work. And the MCP server needs a model to serve.

`.gitignore` ignores `*.pt` (trained weights are normally build output, and should not be committed), so shipping models is an exception that needs a reason.

## Decision

Commit exactly two files, with an exception in `.gitignore` (`!/models/*.pt`):

| File | Size on disk |
|---|---|
| `models/odyssey_book1_gpt.pt` (GPT, 835,072 weights) | 3,380,850 bytes (3.4 MB) |
| `models/odyssey_book1.pt` (small one-head model, 11,840 weights) | 88,453 bytes (88 KB) |

Your own trained files go to `src/homer_ai/homer_gpt.pt` and `homer_model.pt`, which stay ignored and are never committed. Notebooks and the server prefer your file and fall back to `models/` (see [ARCHITECTURE.md](../ARCHITECTURE.md)). `models/README.md` records how each was made.

Git LFS (large file storage) and release downloads were not used. Total weights are about 3.5 MB, so the clone stays small (the README says a few MB). LFS needs an extra install and, on some hosts, bandwidth quotas. A release download adds a setup step where the notebooks fail until the learner finds and runs it. Both would break the rule that the notebooks work right after `uv sync`.

## Consequences

- No setup step for the models. Tests and CI use the committed small model (`HOMER_MODEL` is set to it in `tests/conftest.py`).
- Each retrain adds a new binary to git history. The GPT file is 3.4 MB, so a handful of retrains is fine, but history grows and cannot be squeezed back without rewriting it.
- A reader must trust a binary file. Format 2 files hold only tensors and plain data, and `load_checkpoint` uses `weights_only=True`, so loading cannot run code.
- The shipped files are not rebuilt in CI. `models/README.md` only claims the same numbers on the same machine and PyTorch version; other machines may differ in the last digits.
- This decision should be revisited if a shipped model grows to tens of megabytes, if models are retrained often, or if a bigger model (more books, more weights) is added. At that point use Git LFS or a release asset, with a small download script and a clear message in the notebooks when the file is missing.
