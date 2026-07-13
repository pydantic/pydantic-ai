---
title: Compaction
description: A menu of strategies -- clear, dedupe, trim, or summarize -- for keeping an agent's conversation history within the model's context window.
---

# Compaction

Compaction is a menu of strategies for keeping an agent's conversation history within a model's context window. Each strategy is a Pydantic AI `Capability` that edits the message history just before each request goes out. The edits **persist** into the run's message history, so a trim, clear, or summary carries forward to later steps -- it is not recomputed from the full history every turn.

All strategies preserve tool-call / tool-return **pairing**. Core does not validate this, and a provider rejects an orphaned pair, so the pairing guarantee is what makes these safe to drop into an agent. The zero-LLM strategies never call a model; only `SummarizingCompaction` (and `TieredCompaction` when it escalates that far) spends tokens.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/compaction/)

> The API may change between releases. Where practical, breaking changes ship with a deprecation warning.

## The problem

An agent that runs for many turns accumulates history: tool outputs, file reads, model reasoning, repeated content. Left unchecked, that history outgrows the model's context window and the next request fails. Compaction keeps the history bounded, and the right strategy depends on where the bloat lives and how much you can afford to spend reclaiming it.

## The menu

| Capability | Cost | What it does | Reach for it when |
|---|---|---|---|
| `ClampOversizedMessages` | zero-LLM | Head/tail-truncates a single oversized part (response text, tool-call args) | One runaway generation blew past the context cap and no other strategy can reach it |
| `SlidingWindow` | zero-LLM | Drops the oldest whole messages down to a tail | You only need the recent turns and can discard old context entirely |
| `ClearToolResults` | zero-LLM | Blanks the content of old tool *results* in place, keeping the last `keep_pairs` | Tool outputs dominate context and can be re-fetched on demand (the cheap first tier) |
| `DeduplicateFileReads` | zero-LLM | Blanks every file read superseded by a newer read of the same file | The agent re-reads files and only the latest version matters |
| `SummarizingCompaction` | one LLM call | Summarizes older messages into a structured summary, keeping the recent tail | Old context still matters but must be compressed; use behind the cheap tiers |
| `TieredCompaction` | escalates | Runs cheap passes first, summarizes only if still over `target_tokens` | You want a sensible default: spend the expensive summary only when needed |
| `LimitWarner` | zero-LLM | Injects an URGENT/CRITICAL warning as limits approach | You want the agent to wrap up rather than have its history rewritten |

## Triggers

Every size-based strategy triggers on `max_messages` and/or `max_tokens` (estimated). Token counts use a ~4-chars-per-token heuristic by default; pass a `tokenizer` callable (for example `tiktoken`) for accuracy. `DeduplicateFileReads` runs on every request when no trigger is set (it is cheap and near-lossless). `TieredCompaction` triggers and stops on a single `target_tokens` budget. `ClampOversizedMessages` triggers per *part* (`max_part_tokens` / `max_part_chars`), not on the whole history -- the failure it targets is one oversized part, not a large total.

## The recommended default: `TieredCompaction`

The field consensus (Anthropic, OpenCode, Letta) is to clear and dedupe first, and summarize only when that is not enough. Summarization turns input tokens into output tokens, which are billed at a premium and generated serially, so it is genuinely expensive. The zero-LLM strategies touch only the cheaper input side.

`TieredCompaction` encodes that escalation: it runs each tier in order, re-measures the token count after each, and stops as soon as the conversation fits `target_tokens`. Order the tiers cheap-to-expensive so the expensive summarization tier is only reached when the cheap passes cannot reclaim enough.

```python
from pydantic_ai import Agent
from pydantic_ai_harness.compaction import (
    ClearToolResults,
    DeduplicateFileReads,
    SummarizingCompaction,
    TieredCompaction,
)
from pydantic_ai.messages import ToolCallPart


def my_file_key(call: ToolCallPart) -> str | None:
    if call.tool_name != 'read_file':
        return None
    return call.args_as_dict().get('path')


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

A tier inside `TieredCompaction` is driven directly by the orchestrator, which re-measures after each tier and stops once under `target_tokens`. A tier's own `max_*` trigger is therefore irrelevant when it runs inside `TieredCompaction` -- set it to anything valid (for example `ClearToolResults(max_tokens=1)`). Any object with `async def compact(messages, ctx) -> list[ModelMessage]` (the `CompactionStrategy` protocol) can be a tier, so you can plug in your own.

## `ClampOversizedMessages`: surviving a runaway generation

A single model response of repeated whitespace, or a single tool call with a giant payload, can produce one part so large the *next* request exceeds the provider's context cap. None of the other strategies can reach it: `SlidingWindow` drops the oldest messages but the offender is the newest; `ClearToolResults` only touches tool *results*; `LimitWarner` never edits history; and feeding the history to `SummarizingCompaction` hits the same cap.

`ClampOversizedMessages` truncates the offending part in place, keeping a head slice and a tail slice with a `[clamped: removed N of M characters]` marker between them. Degenerate generations are low-entropy repetition, so a head/tail slice loses little.

```python
from pydantic_ai import Agent
from pydantic_ai_harness.compaction import ClampOversizedMessages

agent = Agent(
    'openai:gpt-4o',
    capabilities=[
        ClampOversizedMessages(max_part_tokens=50_000, keep_head_chars=2_000, keep_tail_chars=2_000)
    ],
)
```

A part is clamped only when it is oversized *and* the clamp actually shrinks it, so keep `keep_head_chars + keep_tail_chars` well below your per-part threshold.

It clamps two kinds of part inside each `ModelResponse`:

- **Response text** (`TextPart`) -- the critical case, a runaway model-response text part.
- **Tool-call args** (`ToolCallPart`), when `clamp_tool_call_args=True` (the default) -- the same failure shape for a giant payload (for example a runaway `write_plan`). The args are replaced with a small JSON object `{"_clamped": "<head>...<tail>"}` so they stay valid function arguments; the original call already executed, so this only shrinks the history copy. Set `clamp_tool_call_args=False` to clamp response text only.

Request-side parts (user prompts, tool *returns*, system prompts) are deliberately out of scope: user input should not be silently rewritten, and oversized tool returns are the job of `ClearToolResults`.

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

## `ClearToolResults`: the cheap first tier

Tool outputs typically dominate an agent's context, and the agent can usually re-run a tool if it needs the data again. `ClearToolResults` replaces the content of the oldest tool *results* with a short placeholder while keeping the most recent `keep_pairs` tool-call / tool-return pairs intact. The tool calls stay paired with their now-blanked results, so the history stays valid.

```python
from pydantic_ai import Agent
from pydantic_ai_harness.compaction import ClearToolResults

agent = Agent(
    'openai:gpt-4o',
    capabilities=[ClearToolResults(max_tokens=100_000, keep_pairs=3)],
)
```

Set `clear_tool_inputs=True` to also blank the arguments of the cleared calls, and `exclude_tools` to a set of tool names whose results are never cleared.

## `DeduplicateFileReads`: drop superseded reads

When the same file is read more than once, only the latest read keeps its content; earlier reads are blanked with a placeholder, with pairing preserved.

There is no default `file_key`: identifying a file read is agent-specific, and a wrong guess would drop live data. Supply a callable mapping a `ToolCallPart` to a stable file key, or `None` when the call is not a file read:

```python
from pydantic_ai import Agent
from pydantic_ai.messages import ToolCallPart
from pydantic_ai_harness.compaction import DeduplicateFileReads


def file_key(call: ToolCallPart) -> str | None:
    if call.tool_name != 'read_file':
        return None
    return call.args_as_dict().get('path')


agent = Agent('openai:gpt-4o', capabilities=[DeduplicateFileReads(file_key=file_key)])
```

With no `max_messages` or `max_tokens` trigger set, `DeduplicateFileReads` runs on every request. It is cheap and near-lossless, so that default is usually what you want.

## `SlidingWindow`: keep only the recent tail

When the conversation exceeds the configured threshold, `SlidingWindow` discards the oldest whole messages down to a tail, preserving tool-call / tool-return pairs. Reach for it when you only need the recent turns and can discard old context entirely.

```python
from pydantic_ai import Agent
from pydantic_ai_harness.compaction import SlidingWindow

agent = Agent(
    'openai:gpt-4o',
    capabilities=[SlidingWindow(max_messages=80, keep_messages=40)],
)
```

By default `preserve_first_user_message=True` keeps the first user turn (in addition to system prompts) even when it falls outside the window, so the agent does not lose the original task. Pass `keep_tokens` instead of `keep_messages` to trim to a token budget rather than a message count.

## `SummarizingCompaction`: compress, do not discard

When old context still matters but must be compressed, `SummarizingCompaction` summarizes the older messages with a dedicated model call and replaces them with a single structured summary, preserving the recent tail and tool-call integrity. It is the expensive tier, so it is best used behind the cheaper passes (see `TieredCompaction`).

```python
from pydantic_ai import Agent
from pydantic_ai_harness.compaction import SummarizingCompaction

agent = Agent(
    'openai:gpt-4o',
    capabilities=[
        SummarizingCompaction(
            model='openai:gpt-4o-mini',
            max_messages=60,
            keep_messages=20,
        )
    ],
)
```

`model` accepts a model name or a `Model`; when left `None` it inherits the running agent's model. No token caps are imposed on the summary call. By default `incremental=True` extends any existing summary from a prior compaction rather than regenerating it from scratch.

### Usage accounting

The summary call is a real request to the model, so its full usage -- tokens **and** the request itself -- is folded into the run's `ctx.usage`. This is deliberate: it keeps cost honest, keeps the request count consistent (a model request that did not count as one would be the surprise), and lets a `UsageLimits` request limit catch a runaway compaction. A run-request or iteration limiter will therefore see compaction calls among its requests.

## `LimitWarner`: warn instead of rewrite

`LimitWarner` never edits history. As the run approaches a configured limit, it injects an URGENT (then CRITICAL) warning as a trailing user turn, so the model wraps up rather than having its context rewritten under it. Models tend to pay more attention to user messages than system messages, which is why the warning is a user turn. Previous warnings from this capability are stripped before deciding whether to inject a new one.

```python
from pydantic_ai import Agent
from pydantic_ai_harness.compaction import LimitWarner

agent = Agent(
    'openai:gpt-4o',
    capabilities=[
        LimitWarner(
            max_iterations=40,
            max_context_tokens=100_000,
        )
    ],
)
```

Warnings begin at `warning_threshold` (default `0.7`, a fraction of the limit) and become CRITICAL for iterations once the remaining request count drops to `critical_remaining_iterations` (default `3`). It watches three kinds of limit -- `max_iterations`, `max_context_tokens`, and `max_total_tokens` -- and by default warns on whichever are configured; narrow that with `warn_on`.

## Cache tradeoff

Clearing, deduplicating, clamping, and summarizing all rewrite message content, which invalidates the provider's prompt cache from the edit point onward -- the next request pays a cache-write. For `ClearToolResults`, use `min_clear_tokens` to skip clearing that reclaims too little to be worth busting the cache. For `ClampOversizedMessages` the cache bust is unavoidable, because the alternative is a failed request.

## Tracing

When core instrumentation is active (the `Instrumentation` capability, `agent.instrument`, or `Agent.instrument_all()`), each strategy emits a `compact_messages` span on the run's tracer the moment it actually compacts -- that is, in `before_model_request`, once the strategy's threshold is exceeded (`ClampOversizedMessages` emits only when a part is actually clamped). `TieredCompaction` emits a single span for the whole escalation rather than one per tier, because it drives each tier's `compact` directly. Without instrumentation the tracer is a no-op, so the span adds no overhead.

The span name is the static `compact_messages`; the strategy is an attribute, not part of the name, to keep span cardinality low. Attributes:

| Attribute | Type | Meaning |
|---|---|---|
| `gen_ai.conversation.compacted` | bool | Always `true`; the OpenTelemetry GenAI convention's flag for a compacted context |
| `compaction.strategy` | str | Strategy class name (for example `SlidingWindow`, `SummarizingCompaction`) |
| `compaction.messages_before` | int | Message count before compaction |
| `compaction.messages_after` | int | Message count after compaction |
| `compaction.tokens_before` | int | Estimated token count before compaction |
| `compaction.tokens_after` | int | Estimated token count after compaction |

`gen_ai.conversation.compacted` is the GenAI semantic convention's flag; the rest is harness-specific. Token counts use the strategy's `tokenizer` when set, otherwise the ~4-chars-per-token heuristic. Raw message content is not recorded.

## Out of scope

These strategies compress or drop context *inside* the window. Moving large tool outputs *out* of the window -- overflowing them to a file the agent (or a subagent) can query on demand -- is a separate capability ([overflowing tool output](overflowing-tool-output.md)), not lossy truncation. Prefer it over capping individual tool outputs.

## API reference

The recommended default is `TieredCompaction`; the other strategies below can be used standalone or plugged in as its tiers.

::: pydantic_ai_harness.compaction.TieredCompaction

::: pydantic_ai_harness.compaction.ClampOversizedMessages

::: pydantic_ai_harness.compaction.ClearToolResults

::: pydantic_ai_harness.compaction.DeduplicateFileReads

::: pydantic_ai_harness.compaction.SlidingWindow

::: pydantic_ai_harness.compaction.SummarizingCompaction

::: pydantic_ai_harness.compaction.LimitWarner
