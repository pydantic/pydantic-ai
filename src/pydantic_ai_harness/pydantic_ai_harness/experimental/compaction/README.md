# Compaction capabilities

> [!WARNING]
> **Experimental.** These capabilities live under `pydantic_ai_harness.experimental` and may
> change or be removed in any release, without a deprecation period. Import them from the
> experimental path — there is no top-level export:
>
> ```python
> from pydantic_ai_harness.experimental.compaction import TieredCompaction
> ```
>
> Importing any experimental capability emits a `HarnessExperimentalWarning`. Silence **all**
> harness experimental warnings with a single filter (no per-capability lines needed):
>
> ```python
> import warnings
> from pydantic_ai_harness.experimental import HarnessExperimentalWarning
>
> warnings.filterwarnings('ignore', category=HarnessExperimentalWarning)
> ```

A menu of strategies for keeping an agent's conversation history within a model's context
window. Each is a Pydantic AI `Capability` that runs in the `before_model_request` hook; edits
**persist** into the run's message history, so a trim/clear/summary carries forward to later
steps (it is not recomputed from the full history every turn).

All strategies preserve tool-call / tool-return **pairing** — core does not validate this, and a
provider rejects an orphaned pair. The zero-LLM strategies never call a model.

## The menu

| Capability | Cost | What it does | Reach for it when |
|---|---|---|---|
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

## Cost: why summarization is the last resort

Summarization turns input tokens into output tokens, which are billed at a premium and generated
serially — so it is genuinely expensive. The zero-LLM strategies touch only the cheaper input side.
The field consensus (Anthropic, OpenCode, Letta) is to clear/dedupe first and summarize only when
that is not enough — which is exactly what `TieredCompaction` encodes:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.experimental.compaction import (
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
and stops once under `target_tokens` — so a tier's own `max_*` trigger is irrelevant there (set it to
anything valid). Any object with `async def compact(messages, ctx) -> list[ModelMessage]`
(`CompactionStrategy`) can be a tier, so you can plug in your own.

## Cache tradeoff (read before using `ClearToolResults`)

Clearing or deduplicating rewrites message content, which invalidates the provider's prompt cache
from the edit point onward — the next request pays a cache-write. Use `ClearToolResults`'
`min_clear_tokens` to skip clearing that reclaims too little to be worth busting the cache.

## Model inheritance

`SummarizingCompaction(model=...)` accepts a model name or `Model`; when left `None` it inherits the
running agent's model. No token caps are imposed on the summary call.

## Usage accounting

The summary call is a real request to the model, so its full usage — tokens **and** the request
itself — is folded into the run's `ctx.usage`. This is deliberate: it keeps cost honest, keeps the
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

## Out of scope

These strategies compress or drop context *inside* the window. Moving large tool outputs *out* of the
window — overflowing them to a file the agent (or a subagent) can query on demand — is a separate
capability, not lossy truncation. Prefer it over capping individual tool outputs.
