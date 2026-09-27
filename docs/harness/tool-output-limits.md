---
title: Tool Output Limits
description: "Keep large tool outputs from filling a Pydantic AI agent's context window: truncate them, spill them to a file the model can page through, or summarize them."
---

# Tool Output Limits

`ToolOutputLimits` reduces a tool return that is large enough to dominate the context
window. Tool returns persist in history as `ToolReturnPart`s, so an oversized one is re-sent
on every later model request, paying its token cost for the rest of the run. This capability
intercepts a return when it is produced, reduces it once, and lets the reduced form persist --
the reduction is not recomputed per request.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/tool_output_limits/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

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

Construct an `Agent` with `ToolOutputLimits()` in its `capabilities`. With no arguments it
uses the default band: spill returns of 10,000 characters or more, with a bounded truncation
fallback if the store cannot accept the write. Spills go to the run's
[workspace](https://pydantic.dev/docs/ai/core-concepts/workspace/), so attach one
([Spill store](#spill-store) has other options):

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness import ToolOutputLimits

agent = Agent('anthropic:claude-opus-5-5', capabilities=[LocalWorkspace('.'), ToolOutputLimits()])
```

The capability registers a single `read_tool_result` tool so the model can page back into any
spilled payload. Its own returns are exempt from reduction. A file tool can also read workspace spills,
but its results are subject to the configured reduction bands.

## Bands: combine the modes

Configure an ordered list of size `bands`. Each band is a `(over, action)` pair: when a
return's measured size reaches `over`, its action runs. The band with the largest threshold
that fits wins; anything below the smallest threshold passes through.

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness import ToolOutputLimits
from pydantic_ai_harness.tool_output_limits import Band, Spill, Summarize, Truncate

agent = Agent(
    'anthropic:claude-opus-5-5',
    capabilities=[
        LocalWorkspace('.'),
        ToolOutputLimits(
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
store errors (for example, the workspace is read-only), a `Truncate` / `Summarize` on a binary
payload, a `Summarize` whose model call raises. `then` chains, so
`Summarize(then=Spill(then=Truncate()))` degrades summarize -> spill -> truncate.

### Per-tool overrides and filtering

`per_tool` replaces the global band list for named tools (file reads to `head`, logs to
`tail`); `tool_filter` (a `ToolSelector`) scopes which tools the capability touches at all.

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness import ToolOutputLimits
from pydantic_ai_harness.tool_output_limits import Band, Truncate, TruncationStrategy

agent = Agent(
    'anthropic:claude-opus-5-5',
    capabilities=[
        LocalWorkspace('.'),
        ToolOutputLimits(
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

### Prefer complete tail lines

Set `Truncate(keep_tail_lines=N)` to reserve the final N lines before allocating the rest of
the character budget. The default is zero, which leaves existing truncation behavior unchanged.

```python
from pydantic_ai_harness.tool_output_limits import Band, ToolOutputLimits, Truncate, TruncationStrategy

truncate = Truncate(max_chars=4_000, strategy=TruncationStrategy.head, keep_tail_lines=2)
limits = ToolOutputLimits(bands=[], per_tool={'run_command': [Band(over=4_000, action=truncate)]})
```

With `head`, the remaining content budget keeps the beginning of the output. With
`head_tail`, it is split 2:3 between the beginning and the text immediately before the
reserved lines. `tail` continues to keep the end. Markers and all retained characters count
toward `max_chars`.

The cap takes priority. If the requested lines exceed it, truncation falls back to the
usual tail strategy; if they fit but a full marker would displace them, the result is a
bare tail slice within the cap. This does not invoke `then` or guarantee that an oversized
control trailer remains intact.

Lines are separated by LF or CRLF, and their existing endings are retained. A terminating
newline does not add a line, but a blank final line counts. Requesting more lines than exist
selects the whole text, subject to the same cap. Negative `keep_tail_lines` values are rejected.

This applies to the text after serialization and optional ANSI stripping, independently
for `ToolReturn.return_value` and textual `content`. Binary fallbacks are unchanged.
Tail-line selection adds no telemetry spans: it is a slicing choice within the existing
tool-result reduction, rather than a separate operation.

## Layering with Shell

Keep Shell's native `max_output_chars` above the `ToolOutputLimits` thresholds. Use
`tail` truncation for moderate command output and `Spill` for large output:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace

from pydantic_ai_harness.shell import Shell
from pydantic_ai_harness.tool_output_limits import (
    Band,
    Spill,
    ToolOutputLimits,
    Truncate,
    TruncationStrategy,
)

tail = Truncate(max_chars=4_000, strategy=TruncationStrategy.tail)
agent = Agent(
    'anthropic:claude-opus-5-5',
    capabilities=[
        LocalWorkspace('.'),
        Shell(allowed_commands=['git', 'rg', 'pytest'], max_output_chars=100_000),
        ToolOutputLimits(
            bands=[],
            per_tool={
                'run_command': [
                    Band(over=20_000, action=Spill(then=tail)),
                    Band(over=4_000, action=tail),
                ],
            },
        ),
    ],
)
```

Only `run_command` uses these bands; `bands=[]` leaves other tools to their native limits.
A tail slice retains an exit-code trailer when it fits in the retained suffix. It does not
guarantee that an entire line survives a small budget; `head` can remove the trailer.

Shell applies its native cap before `ToolOutputLimits` sees the result. With the spill
threshold below that cap, a natively truncated result is stored instead of being truncated
again, unless the store fails. The spill preview shows both ends and the model can use
`read_tool_result` to inspect the stored text.

Spilling preserves only the result received from Shell. It cannot recover content already
removed by the native cap. A preview can contain both spill and native truncation notices;
a native notice describes the stored result, not the shorter preview. A positive
`Spill.preview_chars` value controls the preview's content budget; its header and omission
marker add to that length.

## Both `return_value` and `content` are reduced

A `ToolReturn` carries a `return_value` and an optional `content` that core renders as a
separate, model-visible part which also persists in history. This capability measures and
reduces both with the same band logic (they spill to distinct handles). Text `content` is
reduced in place; non-text `content` (multimodal parts) that overflows is left unreduced with
a `warnings.warn`, since it cannot be safely truncated.

### `ToolReturn.metadata` is preserved

Spill keys (`overflow_handle`, `overflow_bytes`, `overflow_content_handle`) live in
`ToolReturn.metadata` alongside whatever the tool put there. A pre-existing mapping is copied
in with stringified keys; a pre-existing non-mapping value (a string, a dataclass, anything
that is not a `Mapping`) is kept under `original_metadata` rather than dropped. Either way the
caller's own metadata survives a spill and is readable from `metadata` on the resulting
`ToolReturnPart` after `Agent.run`.

## Size unit

Thresholds are measured in characters by default. Set `over_tokens=True` to measure in
estimated tokens (the same ~4-chars-per-token heuristic as [compaction](compaction.md)); pass a
`tokenizer` callable for accuracy. `Truncate.max_chars` is always characters -- truncation is a
character operation regardless of the threshold unit. The cap includes the truncation marker
and applies separately to each reduced text value. If the budget cannot fit both retained
content and a complete marker, truncation keeps the selected slice without a marker. A
non-positive cap returns an empty string. Set `strip_ansi=True` to strip ANSI escape sequences
from text returns before measuring and reducing.

## Pageable structured spills

A spilled structured return is stored as compact JSON: one long line. `read_tool_result`
pages by line, so page 1 returns the whole payload and page 2 is empty. Setting `serializer`
stores the value in a layout with real lines instead:

```python
from pydantic_ai_harness.tool_output_limits import ToolOutputLimits, indented_json, json_lines

ToolOutputLimits(serializer=indented_json)  # one field per line
ToolOutputLimits(serializer=json_lines)  # one record per line
```

Use `json_lines` for tools that return lists of records: line N is record N, so page offsets
and `pattern` matches line up with whole records. Anything that is not a list-like sequence
falls back to `indented_json` -- including a list wrapped in a dict, so return the list
directly for per-record paging. Use `indented_json` for everything else.

Any `(value) -> str` callable works too, but prefer the presets: they escape the Unicode
line separators (U+0085/U+2028/U+2029) that would otherwise knock read-back offsets off the
line grid. The serialized text is also what gets measured, so an indented layout can cross
a size band that compact JSON would not. Strings and binary returns are never serialized,
returns below the smallest band pass through untouched, and a serializer that raises or
returns non-text warns and falls back to compact JSON rather than losing the tool output.

## Spill store

The default `WorkspaceStore` writes each spill as a file in the run's workspace, under
`.pydantic-ai-harness/tool-output/` in the working directory, which is git-ignored. With a
sandbox the files live in the sandbox. The handle is the file's absolute path, so a file tool
such as `FileSystem`'s `read_file` can open it too. `read_tool_result` only opens handles
inside the store's directory.

To keep spills in a different workspace than the run's, pass a backend:

```python
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai_harness.tool_output_limits import ToolOutputLimits, WorkspaceStore

spills_elsewhere = ToolOutputLimits(store=WorkspaceStore(workspace=LocalWorkspaceBackend('/var/spills')))
```

`workspace=` takes a backend, not the `LocalWorkspace` capability. Under a durable engine such
as Temporal it is written in-process, not through the workflow.

Spilled files are kept for the life of the workspace; they are not pruned by this capability.

A read-only workspace cannot take spills: the spill warns and falls back to its `then` action (a
bounded truncation by default). When a band can spill and the run has no workspace for the store
to write to, the run fails at its start. For runs without a workspace, use `LocalFileStore`, which
writes to the host's temp directory:

```python
from datetime import timedelta

from pydantic_ai import Agent
from pydantic_ai_harness.tool_output_limits import LocalFileStore, ToolOutputLimits

store = LocalFileStore(cleanup_after=timedelta(hours=6))  # default: None = keep forever
agent = Agent('anthropic:claude-opus-5-5', capabilities=[ToolOutputLimits(store=store)])
```

`LocalFileStore` keeps its directory owner-only. Set `cleanup_after` to delete spills older than
that age.

Any other backend (a blob store, a durable engine's storage) implements the `OverflowStore`
protocol and is passed as `store=...`:

```python
from typing import Protocol


class OverflowStore(Protocol):
    async def write(self, key: str, data: bytes) -> str: ...   # returns a handle
    async def read(self, handle: str) -> bytes: ...
```

## Usage accounting

A built-in `Summarize` call is a real request to the model, so its full usage -- tokens and the
request itself -- folds into the run's `ctx.usage`, exactly like `SummarizingCompaction`. Its nested
run receives the parent limits unchanged except that a finite request limit reserves one request for
the pending parent request, and is filed under the parent run's `conversation_id`.

By default `Summarize` inherits the running agent's model (`ctx.model`). Pass a model id or
instance to `Summarize(model=...)` to override, or a `summarize` callable to bypass the
built-in prompt entirely. The `summary_prompt` template on the capability must contain both
`{tool_name}` and `{output}` placeholders.

With a durable-execution capability attached, built-in model summarization is a journaled
capability operation. `ToolOutputLimits` carries the stable default `id='tool_output_limits'`, so
durable recovery works without configuration.

Two details matter when choosing a band under durability. The text being summarized is part of the
journaled operation input, so prefer `Spill` over built-in `Summarize` for outputs near the
engine's payload limit. And a custom `summarize` callable runs directly rather than as a durable
operation -- arbitrary callables cannot be reconstructed on the worker side -- so it may be called
again on replay.

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
- Writes spills through the run's workspace, so they sit beside the files that
  [FileSystem](filesystem.md) and [Shell](shell.md) tools act on.
- Distinct from `ClampOversizedMessages`, which clamps runaway model responses, not tool
  returns.

## API reference

::: pydantic_ai_harness.tool_output_limits.ToolOutputLimits
