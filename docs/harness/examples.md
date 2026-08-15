---
title: Examples
description: Complete, self-contained agents built from harness capabilities, written to be read and copied.
---

# Examples

The [`examples/`](https://github.com/pydantic/pydantic-ai-harness/tree/main/examples)
directory contains complete agents assembled from individual capabilities. They are
meant to be read as much as run: every capability choice has the reasoning next to it,
and each example writes out its full configuration so you can copy it into your own
code and tweak it, without chasing imports.

| Example | What it does |
|---|---|
| `coding_agent.py` | A coding agent for the current repo, built from the blocks that make up [`Coder`](coder.md) |
| `research_agent.py` | A web-research agent that cites every claim, built from the blocks that make up [`Researcher`](researcher.md) |

If you just want the assembled version, every packaged harness ([`Coder`](coder.md),
[`Researcher`](researcher.md), …) is one import, or zero, via the
[CLI](https://pydantic.dev/docs/ai/cli/#custom-agents):

```bash
uvx --with pydantic-ai-harness clai -a pydantic_ai_harness.coder:coder_agent
```

## Running an example

From the repo root:

```bash
make install
uv run examples/coding_agent.py
```

Each example states its default model at the top and reads the corresponding API key
from the environment (e.g. `ANTHROPIC_API_KEY`). Set `PYDANTIC_AI_MODEL=provider:model`
to run it against a different model; you'll then need that provider's key instead. See
the [model configuration docs](https://pydantic.dev/docs/ai/models/overview/) for
provider setup.

Every example exposes a `build_agent()` factory you can import and embed in your own
code, and a `main()` that runs a small demo. See
[`examples/README.md`](https://github.com/pydantic/pydantic-ai-harness/blob/main/examples/README.md)
for per-example details.
