Example of Pydantic AI with persistent workspace memory using [sayou](https://github.com/pixell-global/sayou).

Demonstrates:

- [toolsets](../toolsets.md) — using `AbstractToolset` to add external tools
- [message history](../message-history.md) — persisting conversations across sessions

This example shows how to give an agent persistent file storage, search, and conversation
history that survives across sessions. The agent can write notes, search past findings,
and pick up conversations where it left off.

## How It Works

[`sayou-pydantic-ai`](https://github.com/pixell-global/sayou-pydantic-ai) provides two components:

- **`SayouToolset`** — An `AbstractToolset` that gives the agent 7 workspace tools (write, read, list, search, grep, glob, kv). Files are versioned and persist across agent runs.
- **`SayouMessageHistory`** — Serializes `ModelMessage` lists to sayou's KV store, enabling conversation persistence across sessions.

## Running the Example

Install the additional dependency:

```bash
pip install sayou-pydantic-ai
```

With [dependencies installed and environment variables set](./setup.md#usage), run:

```bash
python/uv-run -m pydantic_ai_examples.persistent_memory
```

## Example Code
```snippet {path="/examples/pydantic_ai_examples/persistent_memory.py"}```
