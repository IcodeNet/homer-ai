# Decision records

Short notes on why the project is built the way it is. Each one gives the context, the decision, its consequences and its status. See also [ARCHITECTURE.md](../ARCHITECTURE.md).

| No. | Decision | Status |
|---|---|---|
| [0001](0001-character-level-tokenizer.md) | A character-level tokenizer instead of BPE for Homeric Greek | Accepted |
| [0002](0002-notebooks-generated-from-one-script.md) | Notebooks generated from one script | Accepted |
| [0003](0003-ship-small-models-in-git.md) | Ship small trained models in git | Accepted |
| [0004](0004-mcp-over-stdio.md) | MCP over stdio for serving the model | Accepted |
| [0005](0005-cpu-only-laptop-first.md) | CPU-only, laptop-first | Accepted |
| [0006](0006-bm25-from-scratch-for-search.md) | BM25 from scratch for the search of chapter 9, embeddings as an optional comparison | Accepted |
| [0007](0007-synthetic-data-for-the-domain-chapter.md) | Invented, generated data for the company examples of chapter 10 | Accepted |

To add one, copy the layout of an existing file, use the next number, and add a row here. Do not edit an accepted record to change history: write a new one and mark the record it replaces "Superseded by NNNN".
