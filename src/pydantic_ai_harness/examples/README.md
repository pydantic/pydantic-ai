# Examples

Complete agents to read, run, and copy into your own code.

If you just want the assembled version, use the packaged harnesses instead
([`Coder`](../../../docs/harness/coder.md), [`Researcher`](../../../docs/harness/researcher.md)), or run one
with zero setup: `uvx --with "pydantic-ai-harness[coder]" clai -a pydantic_ai_harness.coder:coder_agent`.

## Setup

From the repo root:

```bash
make install                      # or: uv sync --all-extras
uv run src/pydantic_ai_harness/examples/coding_agent.py
```

Each example states its default model at the top and reads that provider's API
key from the environment (e.g. `ANTHROPIC_API_KEY`). Set
`PYDANTIC_AI_MODEL=provider:model` to run against a different model with that
provider's key.
The video example also needs `OPENAI_API_KEY` for ingestion and search.

## The examples

| Example | What it does | Default model |
|---|---|---|
| [`agent_control.py`](agent_control.py) | A support agent whose prompt, model, settings and tool descriptions are editable from Logfire, block by block. Runs on the code-default path with no Logfire configured | `anthropic:claude-fable-5` |
| [`coding_agent.py`](coding_agent.py) | A coding agent for the current repo: `Coder` in a local workspace that passes on `PATH`, `HOME`, `LANG`, `LC_ALL` and `LC_CTYPE` | `anthropic:claude-opus-5-5` |
| [`research_agent.py`](research_agent.py) | A web-research agent that cites every claim, built from the blocks that make up `Researcher` | `openai:gpt-5.6-sol` |
| [`pixeltable_video.py`](pixeltable_video.py) | Search stored video transcripts and frame captions with timestamps and persistent memory; see the [Pixeltable guide](../../../docs/harness/pixeltable.md#video-audio-and-images) for setup | `openai:gpt-5.6-sol` |

Every example exposes a `build_agent()` factory (imported by the test suite, and
handy for embedding the agent in your own code) and a `main()`. The coding and research
examples' `main()` starts an interactive session.

### Pointing `agent_control.py` at a Logfire

With no Logfire configured, `agent_control.py` resolves nothing and runs exactly
as the code says -- it reports `reason='code_default'` and prints the prompt it
assembled. That is the path to check first, and it needs no setup at all.

Two more environment variables point it at a Logfire instead:

| Variable | What it is |
|---|---|
| `LOGFIRE_BASE_URL` | The Logfire origin: UI, OTLP and API. Without it the SDK infers a region from the token, so a local platform needs it set (e.g. `http://localhost:3000`) |
| `LOGFIRE_API_KEY` | An API key with `project:read_variables` (and `project:write_variables` to publish a config from outside the UI). This is a different credential from the write token that sends spans: a write token cannot serve the variables API |

With both set, the example calls `logfire.configure()` and
`logfire.instrument_pydantic_ai()`, which is how the agent reports itself to your
project and how the config version behind a run reaches a trace. Editing a block
in Logfire then changes the next run's output, with no restart. See
[the Agent Control docs](../../../docs/harness/agent-control.md) for what a published config can
hold, and
[`integration_tests/logfire_platform/`](../integration_tests/logfire_platform/README.md)
for the suite that checks all of it against a running platform.
