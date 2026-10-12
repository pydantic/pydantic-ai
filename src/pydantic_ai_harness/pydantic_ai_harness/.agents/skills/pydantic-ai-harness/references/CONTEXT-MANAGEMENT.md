# Context Management

Capabilities that keep a long run inside the model's context window and its prompt cache healthy:
the compaction family (edit history before each request), `ToolOutputLimits` (shrink or spill big
tool returns when they are produced), `WarnOnCacheBusts` (observe cache collapses), and the media
stores `StepPersistence` uses. None needs an extra, except `MongoMediaStore`, which needs
`pydantic-ai-harness[mongodb]` (`PostgresMediaStore` takes your own `asyncpg` pool and needs no
harness extra). All compaction strategies keep tool-call /
tool-return pairs intact, and their edits persist into the run's message history.

## Pick a capability

| Symptom | Use |
|---|---|
| Long runs overflow the window, any model | `TieredCompaction` (clamp/dedupe/clear first, summarize last) |
| Tool outputs dominate context and can be re-fetched | `ClearToolResults` |
| Agent re-reads the same files | `DeduplicateFileReads` |
| Many turns: slow or costly but not overflowing, or only recent turns matter | `SlidingWindowCompaction(max_messages=...)` (no LLM call), or `SummarizingCompaction` to keep the gist |
| Summarization can fail (API error) and the run must survive | `FallbackCompaction` |
| One runaway response or tool-call arg blows the cap | `ClampOversizedMessages` |
| Model should wrap up rather than have history rewritten | `WarnNearLimits` |
| UI needs a live "context: 73%" gauge | `ReportContextUsage` |
| A single tool return is huge (file read, logs, JSON) | `ToolOutputLimits` |
| Cache hit rate drops / costs jump with no error | `WarnOnCacheBusts` |
| OpenAI Responses or Anthropic only, server-side compaction is fine | core `OpenAICompaction` / `AnthropicCompaction` |
| Need exact details compaction dropped | `ConversationSearch` (see KNOWLEDGE-AND-MEMORY.md) |

## Harness compaction vs provider-native compaction

Core ships provider-native compaction as capabilities in the model modules (not in
`pydantic_ai.capabilities`): `pydantic_ai.models.openai.OpenAICompaction` (Responses API;
`token_threshold=`, or stateless mode via `message_count_threshold=` / `trigger=`) and
`pydantic_ai.models.anthropic.AnthropicCompaction(token_threshold=150_000, instructions=None, pause_after_compaction=False)`.
The provider summarizes server-side; it only works on that provider.

Use the harness strategies when the agent may run on any model (including `FallbackModel` across
providers), when you want zero-LLM trimming, or when you need control over what is kept (pins,
receipts, tiers). Do not stack a harness summarizer and provider-native compaction on the same
agent without a reason: both rewrite history.

## Triggers (shared by all size-based strategies)

- `max_messages`, `max_tokens`, or `max_fraction` (keyword-only). `max_tokens` and `max_fraction`
  are mutually exclusive (`ValueError`). Most standalone strategies require at least one trigger.
  `TieredCompaction` takes no `max_*`: its `target_tokens`/`target_fraction` is both the trigger
  and the stopping point.
- `max_fraction` resolves per request against the model's `context_window` (profile, else
  `genai-prices`); prefer it over absolute tokens so one config fits every model.
- `context_window=` overrides resolution; `fallback_context_window=` (default `200_000`) is used
  only when the window cannot be resolved. `TestModel` never resolves, so `max_fraction=0.9` means
  a 180,000-token trigger in tests: set `context_window=` there.
- Token counts anchor on the last response's provider usage, then estimate the suffix with
  `tokenizer=` or ~4 chars per token.

## TieredCompaction (recommended default)

Before each model request, if the estimated history is over `target_tokens` / `target_fraction`,
runs `tiers` in order, re-measuring after each, and stops once under it. Order tiers cheap to
expensive.

```python
from pydantic_ai import Agent
from pydantic_ai.messages import ToolCallPart

from pydantic_ai_harness import (
    ClampOversizedMessages,
    ClearToolResults,
    DeduplicateFileReads,
    SummarizingCompaction,
    TieredCompaction,
)


def file_key(call: ToolCallPart) -> str | None:
    if call.tool_name != 'read_file':
        return None
    return call.args_as_dict().get('path')


agent = Agent(
    'test',
    capabilities=[
        TieredCompaction(
            tiers=[
                ClampOversizedMessages(max_part_tokens=50_000),
                DeduplicateFileReads(file_key=file_key),
                ClearToolResults(max_tokens=1, keep_pairs=3),
                SummarizingCompaction(max_messages=1, keep_messages=20),
            ],
            target_fraction=0.8,
        )
    ],
)
```

Gotchas: a tier's own `max_*` trigger is ignored inside `TieredCompaction` but must still be valid,
hence `max_tokens=1` / `max_messages=1`. `tiers` must be non-empty and exactly one of
`target_tokens` / `target_fraction` is required. Any object with
`async def compact(messages, ctx) -> list[ModelMessage]` (`CompactionStrategy`) can be a tier.

## The individual strategies

### `ClearToolResults`

`ClearToolResults(max_messages=None, max_tokens=None, keep_pairs=3, placeholder='[tool result cleared]', exclude_tools=frozenset(), clear_tool_inputs=False, min_clear_tokens=None)`:
blanks old tool results, keeping the newest `keep_pairs`. `min_clear_tokens` skips clears too
small to be worth a cache bust. Core `search_tools` / `load_capability` returns are never cleared.

### `DeduplicateFileReads`

`DeduplicateFileReads(file_key, ...)`: `file_key` is required (no default; a wrong guess would
drop live data). With no trigger it runs on every request.

### `SlidingWindowCompaction`

`SlidingWindowCompaction(max_messages=None, max_tokens=None, keep_messages=40, keep_tokens=None, preserve_first_user_message=True, receipts=False)`:
drops the oldest whole messages.

### `SummarizingCompaction`

`SummarizingCompaction(model=None, max_messages=None, max_tokens=None, keep_messages=20, keep_tokens=None, ...)`:
one LLM call per compaction. `model=None` inherits the run's model; a non-text model (for
example a decision model) needs `model=` set or the first compaction raises `UserError`.
Useful keyword options: `model_settings`, `summarization_capabilities` (outer capabilities do not
run on the summary call), `incremental=True` (update the previous summary rather than
re-summarize), `keep_user_messages=False`, `tool_return_max_chars=500`, `summary_prompt` (must
contain `{messages}`), `instructions`, `bridge_prefix=False`, `receipts=False`,
`event_stream_handler` (pass `drain_summary_events` for endpoints that reject non-streaming).
Usage (tokens and the request) folds into the run's `ctx.usage`, so `UsageLimits` sees it. Keep
the default `id='summarizing_compaction'` under durable execution.

### `ClampOversizedMessages`

`ClampOversizedMessages(max_part_tokens=None, max_part_chars=None, keep_head_chars=2_000, keep_tail_chars=2_000, clamp_tool_call_args=True)`:
head/tail-truncates one oversized `TextPart` or `ToolCallPart` args in a `ModelResponse`. Needs
one of the two `max_part_*`. Keep head + tail well below the threshold. Never touches tool
returns or user prompts.

### `FallbackCompaction`

`FallbackCompaction(fallback_chain, fallback_on=(ModelAPIError, FallbackExceptionGroup), *, max_tokens=None, max_fraction=None)`:
tries the next strategy only when one raises a `fallback_on` exception; re-raises the last one
if all fail. With no trigger, its request hook does nothing (it still works as a tier or via
`compact_now`).

```python
from pydantic_ai import Agent

from pydantic_ai_harness import (
    FallbackCompaction,
    SlidingWindowCompaction,
    SummarizingCompaction,
)

agent = Agent(
    'test',
    capabilities=[
        FallbackCompaction(
            max_fraction=0.85,
            fallback_chain=[
                SummarizingCompaction(max_messages=1, keep_tokens=20_000),
                SlidingWindowCompaction(max_messages=1, keep_tokens=20_000),
            ],
        )
    ],
)
```

## Observe instead of edit: WarnNearLimits and ReportContextUsage

`WarnNearLimits(max_iterations=None, max_context_tokens=None, max_total_tokens=None, warn_on=None, warning_threshold=0.7, critical_remaining_iterations=3, *, max_context_fraction=None)`
never edits history; it appends an URGENT then CRITICAL user-turn warning as limits approach. At
least one limit is required; `warn_on` may only name configured kinds (`'iterations'`,
`'context_window'`, `'total_tokens'`).

`ReportContextUsage(context_window=None, fallback_context_window=200_000, tokenizer=None)` emits a
`ContextUsageEvent` (`used_tokens`, `window_tokens`, `fraction`, `resolved`) before each request.
`resolved=False` means the window is the fallback guess. The `on_usage=` callback is deprecated;
subscribe to the event.

```python
from pydantic_ai import Agent

from pydantic_ai_harness import (
    ReportContextUsage,
    SummarizingCompaction,
    WarnNearLimits,
)
from pydantic_ai_harness.compaction import ContextUsageEvent

agent = Agent(
    'test',
    capabilities=[
        SummarizingCompaction(max_fraction=0.9, keep_messages=20),
        ReportContextUsage(),
        WarnNearLimits(max_iterations=40, max_context_fraction=0.8),
    ],
)


@agent.on_event(ContextUsageEvent)
async def show(ctx, event):
    print(f'{event.fraction:.0%}')
```

`WarnNearLimits` tells the model; to alert an operator, log from the `ContextUsageEvent` handler
(`if event.fraction > 0.8: logger.warning(...)`). Register `ReportContextUsage` after a compaction
capability to see post-compaction usage, before it to see what triggered compaction.

## Pins, receipts, and manual compaction

- `pin(text)` (from `pydantic_ai_harness.compaction`) returns a `UserPromptPart` to place in a
  `ModelRequest` in history; every strategy preserves or re-injects it. `Planning` does not need it.
- `receipts=True` on `SlidingWindowCompaction` / `SummarizingCompaction` appends a note saying
  history was compacted; with `StepPersistence` attached it includes the run handle.
- `await compact_now(strategy, history, model=..., focus=..., conversation_id=...)` (plus `deps=` for typed deps) runs a
  strategy between runs (for a `/compact` command). It applies no trigger of its own.

Every rewrite (clear, dedupe, clamp, summarize) busts the provider prompt cache from the edit
point onward.

## ToolOutputLimits

Reduces an oversized tool return once, when produced, so it is not re-sent at full size on every
later request. Actions: `Truncate` (zero-LLM, lossy), `Spill` (lossless: stores the payload and
returns a handle plus preview), `Summarize` (one LLM call), `Passthrough`. The spill preview reads
`[Tool output too large (N chars); stored to handle '<handle>'. Read it with read_tool_result(...)]`,
then a `shape:` line for JSON and a head/tail excerpt; the model pages it with the registered
`read_tool_result(handle, offset=0, limit=200, from_end=False, pattern=None)` (`pattern` is a literal
substring; at most 1,000 lines / 50,000 chars per call). Default with no `bands`:
`Band(over=10_000, action=Spill(then=Truncate()))`, measured in characters.

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace

from pydantic_ai_harness import ToolOutputLimits
from pydantic_ai_harness.tool_output_limits import (
    Band,
    Spill,
    Summarize,
    Truncate,
    TruncationStrategy,
)

agent = Agent(
    'test',
    capabilities=[
        LocalWorkspace('.'),
        ToolOutputLimits(
            bands=[
                Band(over=100_000, action=Spill()),
                Band(over=20_000, action=Summarize(then=Spill(then=Truncate()))),
                Band(over=5_000, action=Truncate()),
            ],
            per_tool={
                'run_command': [
                    Band(over=4_000, action=Truncate(strategy=TruncationStrategy.tail))
                ],
            },
        ),
    ],
)
```

Key parameters: `bands` (largest matching `over` wins; below the smallest passes through),
`per_tool` (replaces `bands` for named tools), `tool_filter` (which tools are touched at all),
`over_tokens=False` / `tokenizer`, `strip_ansi=False`, `serializer` (`indented_json` or
`json_lines` so structured spills page by line), `summary_prompt` (must contain `{tool_name}` and
`{output}`), `store`. `Truncate(strategy=head_tail, max_chars=4_000, then=None, *, keep_tail_lines=0)`,
`Spill(preview_chars=1_000, then=None)`, `Summarize(model=None, summarize=None, then=None)`.
`then` runs when the action cannot (store error, binary payload, failed summary call).

Stores:
- Default `WorkspaceStore()` writes under `.pydantic-ai-harness/tool-output/` in the run's
  workspace. If any band can spill and the run has no workspace, the run fails at start with
  `UserError`. `WorkspaceStore(workspace=LocalWorkspaceBackend(...))` takes a backend, not the
  `LocalWorkspace` capability. A read-only workspace falls back to `then`.
- `LocalFileStore(base_dir=None, cleanup_after=None)` from `pydantic_ai_harness.tool_output_limits`, for runs without a workspace (host temp dir).
- Anything implementing `OverflowStore` (`async write(key, data) -> handle`, `async read(handle)`).

Gotchas: both `ToolReturn.return_value` and text `content` are reduced; `ModelRetry` and tool errors
are never reduced. Keep `Shell(max_output_chars=...)` above your thresholds. `Summarize` usage counts
toward the run; under durable execution prefer `Spill` for payloads near the engine's size limit.
Distinct from `ClampOversizedMessages`, which clamps model responses, not tool returns. With
`CodeMode`, the limits apply to the `run_code` return, not to each tool call inside the script.

## WarnOnCacheBusts

Emits a `CacheBustWarning` (a `UserWarning`) once when a request falls short of the cached prefix
the conversation established (per provider, endpoint, and model) by more than `min_missed_ratio=0.05`
of it and at least `min_missed_tokens=2000` tokens, unless the provider's cache retention window or a
`CompactionPart` explains it. It shares core instrumentation's detector: `warning.reason` is
`'unexpected'` (inside the retention window) or `'unknown'` (no published window); `ttl_expired`,
`compacted`, and `unreported` collapses don't warn. It adds no tools or instructions. Deprecated:
`collapse_ratio` (honored as `min_missed_ratio=1 - collapse_ratio`), `min_prefix_tokens` (honored;
use `min_missed_tokens`), and `cache_ttl_seconds` (ignored: retention comes from the model's
settings, its profile's `default_cache_retention`, and `CachePoint` TTLs).

```python
import warnings

from pydantic_ai import Agent

from pydantic_ai_harness import WarnOnCacheBusts
from pydantic_ai_harness.warn_on_cache_busts import CacheBustWarning

agent = Agent('test', capabilities=[WarnOnCacheBusts()])
warnings.filterwarnings('error', category=CacheBustWarning)  # fail CI on busts
```

Gotchas: reuse one instance across runs (marks are per `conversation_id`, held in process memory);
it only fires when the provider reports cache tokens; route to Logfire with
`logging.captureWarnings(True)`. On a provider with no published retention window (`unknown`)
it can't tell a moved prefix from an expired cache.

## Media externalization (not a capability)

`pydantic_ai_harness.media` holds content-addressed stores (`DiskMediaStore`, `SqliteMediaStore`,
`S3MediaStore`, `MongoMediaStore` with the `mongodb` extra, `PostgresMediaStore` over a
caller-owned `asyncpg` pool) and the `externalize_media` /
`restore_media` walkers. `StepPersistence` stores use them automatically (`media_store='auto'`,
`media_threshold_bytes` 64 KiB) to keep snapshots small; configure a store only to change where
payloads live. Nothing here goes in `capabilities=[...]`.

## Deprecated names

Use `warn_on_cache_busts.WarnOnCacheBusts` (not `cache_stability` / `CacheStabilityMonitor`),
`tool_output_limits.ToolOutputLimits` (not `overflowing_tool_output` / `OverflowingToolOutput`),
`SlidingWindowCompaction` (not `SlidingWindow`), and `WarnNearLimits` (not `LimitWarner`). The
old `context` module is `repo_context`.

## See also

- https://pydantic.dev/docs/ai/harness/compaction/
- https://pydantic.dev/docs/ai/harness/tool-output-limits/
- https://pydantic.dev/docs/ai/harness/warn-on-cache-busts/
- https://pydantic.dev/docs/ai/harness/media/
- https://pydantic.dev/docs/ai/capabilities/compaction/
