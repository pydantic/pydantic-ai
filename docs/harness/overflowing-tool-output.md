---
title: Overflowing Tool Output
description: Reduce oversized tool returns when they are produced -- truncate, spill to a queryable file, or summarize -- so a large payload does not persist in history.
---

# Overflowing Tool Output

`OverflowingToolOutput` reduces a tool return that is large enough to dominate the context
window. Tool returns persist in history as `ToolReturnPart`s, so an oversized one is re-sent
on every later model request, paying its token cost for the rest of the run. This capability
intercepts a return when it is produced, reduces it once, and lets the reduced form persist --
the reduction is not recomputed per request.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/overflowing_tool_output/)

> The API may change between releases. Where practical, breaking changes ship with a deprecation warning.

## The problem

A tool can return a payload large enough to dominate the context window: a big file read, a
verbose log, a large JSON document. Because tool returns persist in history, an oversized one
is re-sent on every later model request, paying its token cost for the rest of the run.

This is the overflow-to-file follow-up the [compaction](compaction.md) capability names as out
of scope: it moves large tool outputs *out* of the window at production time, rather than
compressing or dropping context already inside it.

## The three modes

| Mode | Cost | Lossy? | What the model gets |
|---|---|---|---|
| `Truncate` | zero-LLM | yes | A head / tail / head+tail clamp of the text |
| `Spill` | zero-LLM | no | A handle + preview + shape sketch; full payload read back on demand |
| `Summarize` | one LLM call | yes | A size-gated summary (inherits the run's model by default) |

`Spill` is lossless: the full payload is persisted and the model reads slices of it through
the registered `read_tool_result(handle, offset, limit, from_end, pattern)` tool (the Claude
Code pattern, the core [#4352](https://github.com/pydantic/pydantic-ai/issues/4352) design).
That tool is bounded: `offset >= 0`, `limit` clamped to a built-in line cap, the joined output
capped, and `pattern` is a literal substring (not a regex), so a model-supplied value cannot
hang the host with catastrophic backtracking.

## Usage

Construct an `Agent` with `OverflowingToolOutput()` in its `capabilities`. With no arguments it
uses the default band: spill returns of 10,000 characters or more, with a bounded truncation
fallback if the store cannot accept the write.

```python
from pydantic_ai import Agent
from pydantic_ai_harness.overflowing_tool_output import OverflowingToolOutput

agent = Agent('openai:gpt-4o', capabilities=[OverflowingToolOutput()])
```

The capability registers a single `read_tool_result` tool so the model can page back into any
spilled payload. Its own returns are exempt from reduction.

## Bands: combine the modes

Configure an ordered list of size `bands`. Each band is a `(over, action)` pair: when a
return's measured size reaches `over`, its action runs. The band with the largest threshold
that fits wins; anything below the smallest threshold passes through.

```python
from pydantic_ai import Agent
from pydantic_ai_harness.overflowing_tool_output import (
    Band,
    OverflowingToolOutput,
    Spill,
    Summarize,
    Truncate,
)

agent = Agent(
    'openai:gpt-4o',
    capabilities=[
        OverflowingToolOutput(
            bands=[
                Band(over=100_000, action=Spill()),      # huge: keep losslessly, read back on demand
                Band(over=20_000, action=Summarize()),    # large: compress with the run's model
                Band(over=5_000, action=Truncate()),      # medium: cheap clamp
            ],
            # below 5,000: passthrough
        )
    ],
)
```

The default band, when you pass no `bands`, is `Spill(then=Truncate())` at a 10,000-character
threshold: lossless when a store accepts the write, a bounded truncation otherwise -- zero LLM
cost and no silent drop.

`Passthrough()` is an explicit no-op action for `bands` or `per_tool` lists, leaving matching
returns untouched.

### Fallbacks with `then`

Every action takes an optional `then`, applied when the action cannot run: a `Spill` whose
store errors, a `Truncate` / `Summarize` on a binary payload, a `Summarize` whose model call
raises. `then` chains, so `Summarize(then=Spill(then=Truncate()))` degrades summarize ->
spill -> truncate.

### Per-tool overrides and filtering

`per_tool` replaces the global band list for named tools (file reads to `head`, logs to
`tail`); `tool_filter` (a `ToolSelector`) scopes which tools the capability touches at all.

```python
from pydantic_ai import Agent
from pydantic_ai_harness.overflowing_tool_output import (
    Band,
    OverflowingToolOutput,
    Truncate,
    TruncationStrategy,
)

agent = Agent(
    'openai:gpt-4o',
    capabilities=[
        OverflowingToolOutput(
            per_tool={
                'read_file': [Band(over=8_000, action=Truncate(strategy=TruncationStrategy.head))],
                'run_shell': [Band(over=8_000, action=Truncate(strategy=TruncationStrategy.tail))],
            },
            tool_filter=['read_file', 'run_shell', 'search'],
        )
    ],
)
```

`TruncationStrategy` has three members: `head` (keep the first characters, good for headers and
schemas), `tail` (keep the last characters, good for build and test output where errors land
last), and `head_tail` (keep both ends, elide the middle -- the default).

## Both `return_value` and `content` are reduced

A `ToolReturn` carries a `return_value` and an optional `content` that core renders as a
separate, model-visible part which also persists in history. This capability measures and
reduces both with the same band logic (they spill to distinct handles). Text `content` is
reduced in place; non-text `content` (multimodal parts) that overflows is left unreduced with
a `warnings.warn`, since it cannot be safely truncated.

## Size unit

Thresholds are measured in characters by default. Set `over_tokens=True` to measure in
estimated tokens (the same ~4-chars-per-token heuristic as [compaction](compaction.md)); pass a
`tokenizer` callable for accuracy. `Truncate.max_chars` is always characters -- truncation is a
character operation regardless of the threshold unit. Set `strip_ansi=True` to strip ANSI
escape sequences from text returns before measuring and reducing.

## Spill store

Spilled payloads go through the narrow `OverflowStore` protocol. The default `LocalFileStore`
writes one file per `(run_id, tool_call_id, retry)` under a stable root directory and keeps it
after the run, so a later `read_tool_result` -- in this run or a subsequent agent/run -- can
still reach it. The handle is backend-addressable (a relative key), not an absolute local
path, so a durable backend (Temporal, a blob store, or the core `ExecutionEnvironment`
workspace once #4352 lands) can resolve the same handle in another process. Supply your own
backend with `store=...`.

```python
from typing import Protocol


class OverflowStore(Protocol):
    async def write(self, key: str, data: bytes) -> str: ...   # returns a handle
    async def read(self, handle: str) -> bytes: ...
```

### Security model (shared root, not isolation)

The store root is stable and shareable on purpose -- spilled files must be readable by a later
agent or run -- so security does not come from per-instance isolation. It comes from two
mechanisms: the root is created with `0700` (owner-only) permissions, and `read` resolves the
target (following symlinks) and rejects anything that escapes the root via symlink, `..`, or
an absolute path. Handle segments are also sanitized so a crafted handle cannot traverse out.

### Cleanup: keep-forever by default, opt-in TTL pruning

By default the store keeps spilled files forever -- deleting on run end would break a later
agent that still wants to read a spill. To bound disk use, opt into age-based pruning:

```python
from datetime import timedelta

from pydantic_ai import Agent
from pydantic_ai_harness.overflowing_tool_output import LocalFileStore, OverflowingToolOutput

store = LocalFileStore(cleanup_after=timedelta(hours=6))  # default: None = keep forever
agent = Agent('openai:gpt-4o', capabilities=[OverflowingToolOutput(store=store)])
```

When set, a `write` schedules a background prune (a daemon thread, off the hot path) that
deletes files whose modification time (`st_mtime`) is older than `cleanup_after`. Pruning is
non-blocking and non-erroring: any failure is caught and surfaced via `warnings.warn`, never
propagated into the agent run, so cleanup can never fail a run or block the hot path.
Last-read time (`st_atime`) is unreliable on `noatime`/`relatime` mounts and is not used.

Prefer external cleanup (cron, a sweeper) over the in-process TTL? Point it at the store root
and delete by mtime:

```python
import time
from pathlib import Path

root = Path('/tmp/pyai_harness_overflow')  # or your configured base_dir
cutoff = time.time() - 6 * 3600
for path in root.rglob('*'):
    if path.is_file() and path.stat().st_mtime < cutoff:
        path.unlink(missing_ok=True)
```

## Usage accounting

A `Summarize` call is a real request to the model, so its full usage -- tokens and the
request itself -- folds into the run's `ctx.usage`, exactly like `SummarizingCompaction`. No
token caps are imposed on the summary call. A `UsageLimits` request limit will see it.

By default `Summarize` inherits the running agent's model (`ctx.model`). Pass a model id or
instance to `Summarize(model=...)` to override, or a `summarize` callable to bypass the
built-in prompt entirely. The `summary_prompt` template on the capability must contain both
`{tool_name}` and `{output}` placeholders.

## Edge cases

- Binary returns spill verbatim and are never stringify-truncated; `Truncate` / `Summarize`
  on binary fall through to `then`.
- Structured / nested returns spill (or summarize) by preference -- truncating JSON produces
  invalid JSON. `Spill` includes a one-line shape sketch of the top level.
- `ModelRetry` and tool errors never reach this hook (they are raised, not returned), so the
  model always gets the full error it needs to recover.
- A large `ToolReturn.content` is reduced with the same bands as `return_value`; non-text
  content that overflows is left unreduced with a warning.
- Multiple oversized returns in one step get distinct handles (keyed per `tool_call_id`);
  retries get distinct handles too (keyed per `retry`), so a retried call never clobbers the
  earlier attempt's spill.

## Relationship to other capabilities

- Distinct from [compaction](compaction.md), which compresses or drops context already inside
  the window; this capability moves large tool outputs out of the window at production time.
- Consumes core [#4352](https://github.com/pydantic/pydantic-ai/issues/4352) (the canonical
  queryable-file primitive) through the `OverflowStore` seam once it lands.
- Distinct from `ClampOversizedMessages`, which clamps runaway model responses, not tool
  returns.

## API reference

::: pydantic_ai_harness.overflowing_tool_output.OverflowingToolOutput
