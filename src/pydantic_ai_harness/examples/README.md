# Examples

Complete agents assembled from individual harness capabilities, written to be
read as much as run: every capability choice has the reasoning next to it, and
each example writes out its full configuration so you can copy it into your own
code and tweak it.

If you just want the assembled version, use the packaged harnesses instead
([`Coder`](../docs/coder.md), [`Researcher`](../docs/researcher.md)) — or run one
with zero setup: `uvx --with pydantic-ai-harness clai -a pydantic_ai_harness.coder:coder_agent`.

## Setup

From the repo root:

```bash
make install                      # or: uv sync --all-extras
uv run examples/coding_agent.py
```

Each example states its default model at the top and reads that provider's API
key from the environment (e.g. `ANTHROPIC_API_KEY`). Set
`PYDANTIC_AI_MODEL=provider:model` to run against a different model — you'll
then need that provider's key instead.

## The examples

| Example | What it does | Default model |
|---|---|---|
| [`coding_agent.py`](coding_agent.py) | A coding agent for the current repo, built from the blocks that make up `Coder` | `anthropic:claude-fable-5` |
| [`research_agent.py`](research_agent.py) | A web-research agent that cites every claim, built from the blocks that make up `Researcher` | `openai:gpt-5.6-sol` |

Every example exposes a `build_agent()` factory (imported by the test suite, and
handy for embedding the agent in your own code) and a `main()` that runs a small
demo.
