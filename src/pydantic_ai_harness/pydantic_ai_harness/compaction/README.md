# Compaction capabilities

> [!NOTE]
> Import these capabilities from their submodule -- there is no top-level `pydantic_ai_harness` re-export:
>
> ```python
> from pydantic_ai_harness.compaction import TieredCompaction
> ```
>
> The API may change between releases. Where practical, breaking changes ship with a deprecation warning.

A menu of strategies for keeping an agent's conversation history within a model's context
window. Each is a Pydantic AI `Capability` that runs in the `before_model_request` hook; edits
**persist** into the run's message history, so a trim/clear/summary carries forward to later
steps (it is not recomputed from the full history every turn).

All strategies preserve tool-call / tool-return **pairing** -- core does not validate this, and a
provider rejects an orphaned pair. The zero-LLM strategies never call a model.

## The menu

| Capability | Cost | What it does | Reach for it when |
|---|---|---|---|
| `ClampOversizedMessages` | zero-LLM | Head/tail-truncates a single oversized part (response text, tool-call args) | One runaway generation blew past the context cap and no other strategy can reach it |
| `SlidingWindow` | zero-LLM | Drops the oldest whole messages down to a tail | You only need the recent turns and can discard old context entirely |
| `ClearToolResults` | zero-LLM | Blanks the content of old tool *results* in place, keeping the last `keep_pairs` | Tool outputs dominate context and can be re-fetched on demand (the cheap first tier) |
| `DeduplicateFileReads` | zero-LLM | Blanks every file read superseded by a newer read of the same file | The agent re-reads files and only the latest version matters |
| `SummarizingCompaction` | one LLM call | Summarizes older messages into a structured summary, keeping the recent tail | Old context still matters but must be compressed; use behind the cheap tiers |
| `TieredCompaction` | escalates | Runs cheap passes first, summarizes only if still over `target_tokens` | You want the SOTA default: spend the expensive summary only when needed |
| `LimitWarner` | zero-LLM | Injects an URGENT/CRITICAL warning as limits approach | You want the agent to wrap up rather than have its history rewritten |

## Triggers

Every size-based strategy triggers on `max_messages` and/or `max_tokens` (estimated). Token counts
use a ~4-chars-per-token heuristic by default; pass a `tokenizer` callable (e.g. `tiktoken`) for
accuracy. `DeduplicateFileReads` runs on every request when no trigger is set (it is cheap and
near-lossless). `TieredCompaction` triggers and stops on a single `target_tokens` budget.
`ClampOversizedMessages` triggers per *part* (`max_part_tokens` / `max_part_chars`), not on the
whole history -- the failure it targets is one oversized part, not a large total.

## `ClampOversizedMessages`: surviving a runaway generation

A single model response of repeated whitespace, or a single tool call with a giant payload, can
produce one part so large the *next* request exceeds the provider's context cap. None of the other
strategies can reach it: `SlidingWindow` drops the oldest messages but the offender is the newest;
`ClearToolResults` only touches tool *results*; `LimitWarner` never edits history; and feeding the
history to `SummarizingCompaction` hits the same cap.

`ClampOversizedMessages` truncates the offending part in place, keeping a head slice and a tail slice
with a `[clamped: removed N of M characters]` marker between them. Degenerate generations are
low-entropy repetition, so a head/tail slice loses little.

```python
from pydantic_ai import Agent
from pydantic_ai_harness.compaction import ClampOversizedMessages

agent = Agent(
    'openai:gpt-4o',
    capabilities=[ClampOversizedMessages(max_part_tokens=50_000, keep_head_chars=2_000, keep_tail_chars=2_000)],
)
```

A part is clamped only when it is oversized *and* the clamp actually shrinks it, so keep
`keep_head_chars + keep_tail_chars` well below your per-part threshold.

It clamps two kinds of part inside each `ModelResponse`:

- **Response text** (`TextPart`) -- the critical case, a runaway model-response text part.
- **Tool-call args** (`ToolCallPart`), when `clamp_tool_call_args=True` (default) -- the same failure
  shape for a giant payload (e.g. a runaway `write_plan`). The args are replaced with a small JSON
  object `{"_clamped": "<head>...<tail>"}` so they stay valid function arguments; the original call
  already executed, so this only shrinks the history copy. Set `clamp_tool_call_args=False` to clamp
  response text only.

Request-side parts (user prompts, tool *returns*, system prompts) are deliberately out of scope:
user input should not be silently rewritten, and oversized tool returns are the job of
`ClearToolResults`.

Use it as the first tier of `TieredCompaction`, before `ClearToolResults`:

```python
from pydantic_ai_harness.compaction import (
    ClampOversizedMessages,
    ClearToolResults,
    TieredCompaction,
)

TieredCompaction(
    tiers=[
        ClampOversizedMessages(max_part_tokens=50_000),
        ClearToolResults(max_tokens=1, keep_pairs=3),
    ],
    target_tokens=120_000,
)
```

## Cost: why summarization is the last resort

Summarization turns input tokens into output tokens, which are billed at a premium and generated
serially -- so it is genuinely expensive. The zero-LLM strategies touch only the cheaper input side.
The field consensus (Anthropic, OpenCode, Letta) is to clear/dedupe first and summarize only when
that is not enough -- which is exactly what `TieredCompaction` encodes:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.compaction import (
    ClearToolResults,
    DeduplicateFileReads,
    SummarizingCompaction,
    TieredCompaction,
)

agent = Agent(
    'openai:gpt-4o',
    capabilities=[
        TieredCompaction(
            tiers=[
                DeduplicateFileReads(file_key=my_file_key),
                ClearToolResults(max_tokens=1, keep_pairs=3),
                SummarizingCompaction(max_messages=1, keep_messages=20),  # model inherits the run's
            ],
            target_tokens=120_000,
        )
    ],
)
```

A tier inside `TieredCompaction` is driven directly by the orchestrator, which re-measures after each
and stops once under `target_tokens` -- so a tier's own `max_*` trigger is irrelevant there (set it to
anything valid). Any object with `async def compact(messages, ctx) -> list[ModelMessage]`
(`CompactionStrategy`) can be a tier, so you can plug in your own.

## Cache tradeoff (read before using `ClearToolResults`)

Clearing or deduplicating rewrites message content, which invalidates the provider's prompt cache
from the edit point onward -- the next request pays a cache-write. Use `ClearToolResults`'
`min_clear_tokens` to skip clearing that reclaims too little to be worth busting the cache.

## Model inheritance

`SummarizingCompaction(model=...)` accepts a model name or `Model`; when left `None` it inherits the
running agent's model. No token caps are imposed on the summary call.

## Usage accounting

The summary call is a real request to the model, so its full usage -- tokens **and** the request
itself -- is folded into the run's `ctx.usage`. This is deliberate: it keeps cost honest, keeps the
request count consistent (a model request that didn't count as one would be the surprise), and lets a
`UsageLimits` request limit catch a runaway compaction. A run-request / iteration limiter will
therefore see compaction calls among its requests.

## `DeduplicateFileReads.file_key`

There is no default `file_key`: identifying a file read is agent-specific, and a wrong guess would
drop live data. Supply a callable mapping a `ToolCallPart` to a stable file key, or `None` when the
call is not a file read:

```python
from pydantic_ai.messages import ToolCallPart

def my_file_key(call: ToolCallPart) -> str | None:
    if call.tool_name != 'read_file':
        return None
    args = call.args
    return args.get('path') if isinstance(args, dict) else None
```

## Tracing

When core instrumentation is active (the `Instrumentation` capability, `agent.instrument`, or
`Agent.instrument_all()`), each strategy emits a `compact_messages` span on the run's tracer the
moment it actually compacts -- that is, in `before_model_request`, once the strategy's threshold is
exceeded (`ClampOversizedMessages` emits only when a part is actually clamped). `TieredCompaction`
emits a single span for the whole escalation rather than one per tier, because it drives each tier's
`compact` directly. Without instrumentation the tracer is a no-op, so the span adds no overhead.

The span name is the static `compact_messages`; the strategy is an attribute, not part of the name,
to keep span cardinality low. Attributes:

| Attribute | Type | Meaning |
|---|---|---|
| `gen_ai.conversation.compacted` | bool | Always `true`; the OpenTelemetry GenAI convention's flag for a compacted context |
| `compaction.strategy` | str | Strategy class name (e.g. `SlidingWindow`, `SummarizingCompaction`) |
| `compaction.messages_before` | int | Message count before compaction |
| `compaction.messages_after` | int | Message count after compaction |
| `compaction.tokens_before` | int | Estimated token count before compaction |
| `compaction.tokens_after` | int | Estimated token count after compaction |

`gen_ai.conversation.compacted` is the GenAI semantic convention's flag; the rest is
harness-specific. Token counts use the strategy's `tokenizer` when set, otherwise the
~4-chars-per-token heuristic.
Raw message content is not recorded.

## Out of scope

These strategies compress or drop context *inside* the window. Moving large tool outputs *out* of the
window -- overflowing them to a file the agent (or a subagent) can query on demand -- is a separate
capability, not lossy truncation. Prefer it over capping individual tool outputs.
