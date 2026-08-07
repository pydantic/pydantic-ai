# Claude Code `stream-json`

Pydantic AI can emit its run events as [Claude Code](https://docs.claude.com/en/docs/claude-code/overview) CLI `stream-json` output: the JSONL that `claude -p --output-format stream-json --verbose` writes to stdout. Emitting it lets an agent stand in for the Claude Code CLI in tooling built around that stream, such as [GitHub Agentic Workflows](https://github.com/github/gh-aw), which renders its run summaries and token metrics by parsing it.

!!! note
    This API is a draft, pending maintainer review.

Unlike the [AG-UI](./ag-ui.md) and [Vercel AI](./vercel-ai.md) protocols, `stream-json` is an output format rather than a request/response protocol: there is no protocol-specific run input to receive from a frontend, so there is no [`UIAdapter`][pydantic_ai.ui.UIAdapter] and you use the [`ClaudeCodeEventStream`][pydantic_ai.ui.claude_code.ClaudeCodeEventStream] directly.

## Usage

Construct a [`ClaudeCodeEventStream`][pydantic_ai.ui.claude_code.ClaudeCodeEventStream], pass it the agent's [native event stream](../agent.md#streaming-all-events) via [`transform_stream()`][pydantic_ai.ui.UIEventStream.transform_stream], and encode the result with [`encode_stream()`][pydantic_ai.ui.UIEventStream.encode_stream]:

```py {title="claude_code_stream.py"}
import asyncio
import sys

from pydantic_ai import Agent
from pydantic_ai.ui.claude_code import ClaudeCodeEventStream

agent = Agent('anthropic:claude-sonnet-4-5')


async def main():
    event_stream = ClaudeCodeEventStream(model='anthropic:claude-sonnet-4-5')
    async with agent.run_stream_events('What is 2+2?') as events:
        async for line in event_stream.encode_stream(event_stream.transform_stream(events)):
            sys.stdout.write(line)


asyncio.run(main())
```

The `session_id`, `model` and `cwd` the stream reports are constructor parameters, since none of them can be derived from agent events. Everything else the [`system`/`init` record](https://docs.claude.com/en/docs/claude-code/sdk) carries is left empty rather than fabricated.

## Message-level by default

`stream-json` is message-level: each `assistant` line carries one complete content block — a `text`, `thinking` or `tool_use` block — rather than a delta, and all blocks of one model response share a message id. Nothing is emitted until a part is complete, so no partial text ever reaches the stream. Tool results arrive as `user` lines carrying `tool_result` blocks, paired to their call by id, and a run's history compaction becomes a `system`/`compact_boundary` record.

Set `include_partial_messages=True` to additionally emit the `stream_event` records the CLI's `--include-partial-messages` flag produces, which wrap Anthropic-shaped streaming deltas. This is a superset: the whole-block `assistant` lines are still emitted.

```py {title="claude_code_partial.py"}
from pydantic_ai.ui.claude_code import ClaudeCodeEventStream

event_stream = ClaudeCodeEventStream(include_partial_messages=True)
```

## The terminal `result` record

Every stream ends with a `result` record reporting the run's turn count, duration, final answer text and token usage, and nothing follows it — consumers read the last line of the stream to find it. A run that fails still closes with one, carrying `is_error: true`; exhausting a [usage limit](../agent.md#usage-limits) additionally reports `subtype: 'error_max_turns'`.

Token usage appears only on that record, using Anthropic's names (`input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`). Per-message usage is deliberately omitted: Pydantic AI reports usage once the run has finished, and reporting zeros per response would understate it.
