---
title: Harness Examples
description: "Example Pydantic AI Harness agents to read and copy: a coding agent built on `Coder` and a web research agent built from the blocks that make up `Researcher`."
---

# Harness Examples

The [`examples/`](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/examples)
directory contains complete agents to read, run, and copy into your own code.

| Example | What it does |
|---|---|
| `coding_agent.py` | A coding agent for the current repo: [`Coder`](coder.md) in a local workspace |
| `research_agent.py` | A web-research agent that cites every claim, built from the blocks that make up [`Researcher`](researcher.md) |

If you just want the assembled version, every packaged harness ([`Coder`](coder.md),
[`Researcher`](researcher.md), ...) is one import, or zero, via the
[CLI](https://pydantic.dev/docs/ai/cli/#custom-agents):

```bash
uvx --with "pydantic-ai-harness[coder]" clai -a pydantic_ai_harness.coder:coder_agent
```

## Running an example

From the repo root:

```bash
make install
uv run src/pydantic_ai_harness/examples/coding_agent.py
```

Each example states its default model at the top and reads the corresponding API key
from the environment (e.g. `ANTHROPIC_API_KEY`). Set `PYDANTIC_AI_MODEL=provider:model`
to run it against a different model with that provider's key. See
the [model configuration docs](https://pydantic.dev/docs/ai/models/overview/) for
provider setup.

Every example exposes a `build_agent()` factory you can import and embed in your own
code, and a `main()` that starts an interactive session. See
[`examples/README.md`](https://github.com/pydantic/pydantic-ai/blob/main/src/pydantic_ai_harness/examples/README.md)
for per-example details.
