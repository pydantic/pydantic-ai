# Compaction

A menu of strategies for keeping an agent's conversation history within a model's context
window. Most are Pydantic AI `Capability` classes that edit the message history just before each
request goes out. `FallbackCompaction` is instead a composing `CompactionStrategy` used through
`TieredCompaction` or `compact_now`; it has no request trigger of its own. Edits **persist** into the
run's message history, so a trim, clear, or summary carries forward to later steps (it is not
recomputed from the full history every turn).

All strategies preserve tool-call / tool-return **pairing** -- core does not validate this, and a
provider rejects an orphaned pair. The zero-LLM strategies never call a model.

On OpenAI and Anthropic, core also ships [provider-native compaction](https://pydantic.dev/docs/ai/capabilities/compaction/) --
the provider summarizes history server-side. The strategies here are the model-agnostic
alternative: they work with every model and keep the compaction logic (and its costs) under your control.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/compaction/)

## The menu

| Component | Cost | What it does | Reach for it when |
|---|---|---|---|
| `ClampOversizedMessages` | zero-LLM | Head/tail-truncates a single oversized part (response text, tool-call args) | One runaway generation blew past the context cap and no other strategy can reach it |
| `SlidingWindowCompaction` | zero-LLM | Drops the oldest whole messages down to a tail | You only need the recent turns and can discard old context entirely |
| `ClearToolResults` | zero-LLM | Blanks the content of old tool *results* in place, keeping the last `keep_pairs` | Tool outputs dominate context and can be re-fetched on demand (the cheap first tier) |
| `DeduplicateFileReads` | zero-LLM | Blanks every file read superseded by a newer read of the same file | The agent re-reads files and only the latest version matters |
| `SummarizingCompaction` | one LLM call | Summarizes older messages into a structured summary, keeping the recent tail | Old context still matters but must be compressed; use behind the cheap tiers |
| `TieredCompaction` | escalates | Runs cheap passes first, summarizes only if still over `target_tokens` | You want a sensible default: spend the expensive summary only when needed |
| `FallbackCompaction` | depends on chain | Tries the next strategy when one raises | Summarization can fail and deterministic truncation must keep the run alive |
| `WarnNearLimits` | zero-LLM | Injects an URGENT/CRITICAL warning as limits approach | You want the agent to wrap up rather than have its history rewritten |
| `ReportContextUsage` | zero-LLM | Reports context usage to your application; never edits history | You want a live context gauge in a UI |

## Triggers

Every size-based strategy triggers on `max_messages`, `max_tokens` (estimated), or `max_fraction`.
Token counts anchor on the provider-reported usage of the most recent model response when one is
available: its `input_tokens` measured the whole request that produced it (instructions, tool
definitions, every prior message), so only the messages added since are estimated. The estimate for
those, and the fallback when no response carries usage yet, is a ~4-chars-per-token heuristic; pass
a `tokenizer` callable (e.g. `tiktoken`) to sharpen it. The anchor is what keeps triggers honest on
token-dense content -- minified JSON or base64 tokenizes at ~1-2 chars per token, which the
heuristic underestimates severely enough for a history to blow the context window without any
trigger firing. Newly revealed tool schemas that are pending in the request are conservatively
estimated by the implementation. `DeduplicateFileReads` runs on every request when no trigger is set (it is
cheap and near-lossless). `TieredCompaction` triggers and stops on a single `target_tokens` /
`target_fraction` budget. `ClampOversizedMessages` triggers per *part* (`max_part_tokens` /
`max_part_chars`), not on the whole history -- the failure it targets is one oversized part, not a
large total.

### `max_fraction`: one setting for every model

An absolute `max_tokens` is only correct for the model it was measured against. Configure `180_000`
and a 1M-context model compacts at a fifth of its capacity, paying for summaries it did not need; a
128K model configured for `1_000_000` never compacts before the provider rejects the request.

`max_fraction` is resolved per request against the model's real context window, so one configuration
is correct everywhere:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import SummarizingCompaction

agent = Agent(
    'anthropic:claude-sonnet-5',
    capabilities=[SummarizingCompaction(max_fraction=0.9, keep_messages=20)],
)
```

That compacts at 900K on a 1M model and at 115K on a 128K one. `WarnNearLimits` takes the same shape as
`max_context_fraction`, and `TieredCompaction` as `target_fraction`.

`max_tokens` and `max_fraction` are mutually exclusive -- a strategy taking both would have to
pick one and discard the other, leaving the caller unable to tell which budget was in force.

The window comes from [`genai-prices`](https://github.com/pydantic/genai-prices), already a
dependency of `pydantic-ai-slim`; `resolve_context_window` is exported if you want the number
yourself. Pydantic AI does not expose it yet (`ModelProfile` has no `context_window` field), so when
it does, that one function switches over. Nothing is cached: only a registry-confirmed number is
ever treated as the real window.

The model consulted is `ModelRequestContext.model`, the one the request will be sent to, not the one
the run started with. A capability ordered earlier may replace it, and the budget follows.

### When the window does not resolve

Not every model is in the registry. A local endpoint, a bespoke deployment, a Bedrock-prefixed
reference such as `bedrock:us.anthropic.claude-sonnet-5`, a model the registry knows without a
recorded window, and any `FallbackModel` (its `model_id` is a
composite `fallback:...`) all resolve to nothing. The fraction is then taken of
`fallback_context_window`, which defaults to a conservative 200K (`DEFAULT_CONTEXT_WINDOW`):
compacting earlier than necessary costs one summary, overestimating costs the whole request.

Every capability that takes a fraction takes the fallback too, so you are not stuck with 200K on a
model you know the size of:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import SummarizingCompaction

agent = Agent(
    'bedrock:us.anthropic.claude-sonnet-5',
    capabilities=[SummarizingCompaction(max_fraction=0.9, fallback_context_window=1_000_000)],
)
```

It is only consulted when resolution fails, so it costs nothing on a model the registry does know.

`TestModel` is one of the models that does not resolve: its `model_id` is `test:test`, so a fraction
is taken of `fallback_context_window` and `max_fraction=0.9` becomes a 180,000-token trigger. A
compaction config exercised only against `TestModel` will look like it never fires; pass
`context_window=` or `fallback_context_window=` in the test to put the trigger where you can reach it.

### When the window resolves to the wrong number

Resolution can also succeed and be wrong, which `fallback_context_window` cannot help with -- it
applies only when resolution fails. Three cases:

- **The registry entry itself is wrong.** Harness reads `genai-prices` and cannot validate it.
  Measured against `genai-prices` 0.1.3:

  | model id | registry records | real window |
  |---|---|---|
  | `anthropic:claude-sonnet-4-5` | 1,000,000 | 200,000 |
  | `anthropic:claude-opus-4-6` | 200,000 | 1,000,000 |

  An over-recorded window is the direction that breaks a run. On `anthropic:claude-sonnet-4-5`,
  `max_fraction=0.9` resolves to a 900,000-token trigger against a 200,000-token window: compaction
  never fires, and the provider rejects the request instead. **Pass `context_window=200_000`
  explicitly on `claude-sonnet-4-5`.** `claude-sonnet-5`'s recorded 1,000,000 matches Anthropic's
  model documentation, so it needs no override; check any other Sonnet id you use against the
  provider's own documentation before relying on the resolved number. An under-recorded window is
  safe but wasteful -- it compacts earlier than it has to.
- The registry records the maximum a model can be made to accept. Where that maximum is gated --
  a beta header, a pricing tier -- an ordinary request gets less, and a fraction of the recorded
  number never triggers before the provider rejects the request.
- A self-hosted or proxied endpoint reports a model id whose registry entry describes someone
  else's deployment.

`context_window` overrides resolution outright, on every capability that takes a fraction:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import SummarizingCompaction

agent = Agent(
    'openai:gpt-5.6-luna',  # served by a local endpoint with a smaller window than the registry records
    capabilities=[SummarizingCompaction(max_fraction=0.9, context_window=32_000)],
)
```

### What counts toward the fraction

With a usage anchor, everything the provider billed for the anchored request counts -- including
tool definitions and `FilePart` payloads, which no character estimate can see. For the messages
after the anchor (and for whole histories with no reported usage), the estimator counts every part
that is sent: prompts, system prompts, tool calls and their results, retry prompts,
extended-thinking blocks, provider-side tool results, and the instructions, once (or again after
the anchor only when they changed since it). That estimated portion is a ~4-characters-per-token
approximation, not a tokenizer; pass `tokenizer=` to any strategy to measure with the real one.
`FilePart` is not counted there -- its payload is binary, and its length in characters would mean
nothing. Newly revealed tool schemas pending in the current request are conservatively estimated
by the implementation, since they are not covered by the earlier anchor.

**If you already set an absolute `max_tokens`, re-check it.** The estimator used to count only user
and system prompts, tool returns, response text, and tool calls. `ThinkingPart` / `CompactionPart`
content, `RetryPromptPart` content, `NativeToolCallPart` / `NativeToolReturnPart`, and the most
recent `ModelRequest.instructions` are now counted too, so the same history measures higher and an
unchanged `max_tokens` compacts earlier. How much earlier depends on how much of the history is
thinking blocks, retries, and instructions; on a thinking-heavy tool-calling history it can be
several times the old count. What each strategy clears is unchanged -- only when it runs.

## Reporting usage: `ReportContextUsage`

A strategy knows when to act but says nothing about how close the run is to the limit, so an
application that wants to show `context: 73%` ends up re-counting the history and guessing the
denominator. `ReportContextUsage` does neither -- it reuses the same estimator and the same resolved
window, and only observes:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import ReportContextUsage, SummarizingCompaction

agent = Agent(
    'anthropic:claude-sonnet-5',
    capabilities=[
        SummarizingCompaction(max_fraction=0.9, keep_messages=20),
        ReportContextUsage(on_usage=lambda usage: print(f'{usage.fraction:.0%}')),
    ],
)
```

Each reading carries `used_tokens`, `window_tokens`, and `resolved` -- `False` when the window is the
fallback rather than the model's real one, so a gauge can show that the percentage is a guess.
`on_usage` may be a coroutine function, so a gauge that pushes over a socket does not need a sync
bridge.

Order matters: register the monitor *after* a compaction capability to observe the corrected current
history after same-cycle compaction, or before it to see what triggered the compaction.

`used_tokens` follows the accounting above: provider usage anchors include instructions, tool
definitions, and `FilePart` payloads from the anchored request. The suffix after the anchor, or a
history with no anchor, uses `tokenizer` or a ~4-characters-per-token heuristic and cannot see
`FilePart` payloads. Pending newly revealed tool schemas are conservatively estimated by the
implementation.

## Compacting outside a run: `compact_now`

A strategy's `compact` takes a `RunContext`, which an application holding a conversation *between*
runs does not have -- and that is exactly when a user types `/compact`. `compact_now` builds a
throwaway context so the same strategy the agent uses can be driven from a command handler:

```python {test="skip"}
from pydantic_ai_harness import SummarizingCompaction
from pydantic_ai_harness.compaction import compact_now

strategy = SummarizingCompaction(max_fraction=0.9, keep_messages=20)
history = await compact_now(
    strategy,
    history,
    model='anthropic:claude-sonnet-5',
    focus='the auth refactor, not the earlier CSS work',
)
```

`compact_now` applies no trigger of its own, so a strategy whose `compact` is unconditional runs
whatever the history size. A strategy that defines its own stop condition still honours it:
`TieredCompaction` escalates only until the history fits its target, so a history already under
target comes back unchanged. Pass the tier directly if you need it to run regardless.

`focus` steers strategies that write prose -- `SummarizingCompaction`, via the exported
`SupportsFocus` protocol's `with_focus` -- and is passed over by the ones that drop or blank content
by rule, since they have nothing to steer. `TieredCompaction` is focusable when any of its tiers is, so a focus reaches the
summarizing tier rather than stopping at the wrapper.

A compaction that changes the history emits the same `compact_messages` span the in-run path emits,
so an instrumented application sees one shape however compaction was triggered. Pass `tracer=` to
record it; without one the span goes to a no-op tracer.

## `FallbackCompaction`: recover when a strategy fails

`TieredCompaction` advances when a successful tier does not reclaim enough. `FallbackCompaction`
advances only when a strategy raises an exception selected by `fallback_on`, which defaults to
Pydantic AI's `ModelAPIError` and `FallbackExceptionGroup`. The latter is raised when every model
in a `FallbackModel` fails. Each attempt receives a fresh list containing the original message
objects, so list-level changes by a failed strategy do not affect its fallback. Strategies must
still avoid mutating message objects. If every strategy fails, the
last exception is re-raised. Non-matching exceptions, cancellation, and other `BaseException`
subclasses pass through immediately; `fallback_on` rejects types that do not derive from
`Exception`.

```python
from pydantic_ai_harness import FallbackCompaction, SlidingWindowCompaction, SummarizingCompaction

fallback = FallbackCompaction(
    fallback_chain=[
        SummarizingCompaction(max_messages=1, keep_tokens=20_000),
        SlidingWindowCompaction(max_messages=1, keep_tokens=20_000),
    ]
)
```

The strategies' trigger fields are not consulted when a composing strategy calls `compact`
directly. Put `fallback` inside `TieredCompaction` to give the chain a context trigger, or pass it
to `compact_now` for manual compaction.

## `ClampOversizedMessages`: surviving a runaway generation

A single model response of repeated whitespace, or a single tool call with a giant payload, can
produce one part so large the *next* request exceeds the provider's context cap. None of the other
strategies can reach it: `SlidingWindowCompaction` drops the oldest messages but the offender is the newest;
`ClearToolResults` only touches tool *results*; `WarnNearLimits` never edits history; and feeding the
history to `SummarizingCompaction` hits the same cap.

`ClampOversizedMessages` truncates the offending part in place, keeping a head slice and a tail slice
with a `[clamped: removed N of M characters]` marker between them. Degenerate generations are
low-entropy repetition, so a head/tail slice loses little.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import ClampOversizedMessages

agent = Agent(
    'openai:gpt-5.6-terra',
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
  response text only. Framework-typed call parts -- core's `search_tools` and `load_capability`
  calls -- are never clamped, because their typed args are validated when persisted history is
  restored (for example a `StepPersistence` resume) and the `_clamped` object would fail that
  round-trip.

Request-side parts (user prompts, tool *returns*, system prompts) are deliberately out of scope:
user input should not be silently rewritten, and oversized tool returns are the job of
`ClearToolResults`.

Use it as the first tier of `TieredCompaction`, before `ClearToolResults`:

```python
from pydantic_ai_harness import ClampOversizedMessages, ClearToolResults, TieredCompaction

TieredCompaction(
    tiers=[
        ClampOversizedMessages(max_part_tokens=50_000),
        ClearToolResults(max_tokens=1, keep_pairs=3),
    ],
    target_tokens=120_000,
)
```

## `SlidingWindowCompaction` and `ClearToolResults` options

`SlidingWindowCompaction` keeps the last `keep_messages` down to a tail; pass `keep_tokens` instead for a token
budget rather than a message count. By default `preserve_first_user_message=True` keeps the first user
turn even when it falls outside the window, so the agent does not lose the original task.

`ClearToolResults` keeps the last `keep_pairs` intact. Set `clear_tool_inputs=True` to also blank the
arguments of the cleared calls, and `exclude_tools` to a set of tool names whose results are never
cleared. Framework-typed tool results -- core's `search_tools` and `load_capability` returns -- are
left intact (a small token floor), because their structured content is re-parsed on later requests and
rewriting it via `dataclasses.replace` would bypass validation and corrupt the part.

## `WarnNearLimits` thresholds

Warnings begin at `warning_threshold` (default `0.7`, a fraction of the limit) and escalate to CRITICAL
for iterations once the remaining request count drops to `critical_remaining_iterations` (default `3`).
It watches `max_iterations`, `max_context_tokens` (or `max_context_fraction`), and `max_total_tokens`,
warning on whichever are configured; narrow that with `warn_on`.

## Cost: why summarization is the last resort

Summarization turns input tokens into output tokens, which are billed at a premium and generated
serially -- so it is genuinely expensive. The zero-LLM strategies touch only the cheaper input side.
The field consensus (Anthropic, OpenCode, Letta) is to clear/dedupe first and summarize only when
that is not enough -- which is exactly what `TieredCompaction` encodes:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import ClearToolResults, DeduplicateFileReads, SummarizingCompaction, TieredCompaction

agent = Agent(
    'openai:gpt-5.6-luna',
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
running agent's model. Its nested summary run inherits the parent usage limits and reserves one request from a
finite request limit for the pending parent request. Pass `model_settings` to give the dedicated summary call
settings that differ from defaults carried by that model; the supplied settings merge over the model defaults
without mutating the model or the settings dictionary.

By default `incremental=True` updates the newest existing summary from a prior compaction as an
anchor rather than regenerating it from scratch. This changes the summary-call prompt from earlier
releases; set `incremental=False` to retain the prior regeneration behavior. `preserve_first_user_message=True` keeps the original task turn even
when it falls outside the window. Pass `keep_tokens` to trim the retained tail to a token budget instead
of `keep_messages`.

Both prompt surfaces of the summary request are fields: `summary_prompt` is the user-turn template (it
must contain a `{messages}` placeholder), and `instructions` sets the internal agent's static instructions,
which Pydantic AI sends in the request's system prompt. Override `instructions` when the summarizer
endpoint requires a fixed leading instruction.

## Usage accounting

The summary call is a real request to the model, so its full usage -- tokens **and** the request
itself -- is folded into the run's `ctx.usage`. This is deliberate: it keeps cost honest, keeps the
request count consistent (a model request that didn't count as one would be the surprise), and lets a
`UsageLimits` request limit catch a runaway compaction. The nested run receives the other parent limits unchanged;
the finite request limit is reduced by one so it cannot spend the slot already approved for the parent request.
A run-request / iteration limiter will therefore see compaction calls among its requests.

With a durable-execution capability attached, the summary call runs as a contributed durable
operation, so replay uses the recorded summary instead of calling the model again. When `model` is
not set, the operation uses the run's model. The capability carries a stable default `id`, which
durable execution uses to recover the operation by the same identity. Overriding it with a custom
value orphans recorded operations for in-flight workflows, so keep it fixed once a workflow is live.

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
| `compaction.strategy` | str | Strategy class name (e.g. `SlidingWindowCompaction`, `SummarizingCompaction`) |
| `compaction.messages_before` | int | Message count before compaction |
| `compaction.messages_after` | int | Message count after compaction |
| `compaction.tokens_before` | int | Estimated token count before compaction |
| `compaction.tokens_after` | int | Estimated token count after compaction |

`gen_ai.conversation.compacted` is the GenAI semantic convention's flag; the rest is
harness-specific. Token counts use the strategy's `tokenizer` when set, otherwise the
~4-chars-per-token heuristic.
Raw message content is not recorded.

## Compaction receipts

Compaction is a memory wipe the model cannot veto and often cannot detect, which invites
*resumption drift* -- the model confabulates continuity with history it no longer has. A
receipt makes the wipe legible: after a boundary-crossing strategy rewrites history it can
append a short, deterministic note recording how much was compacted, warning that what
survives is secondhand, and -- when a handle provider is attached -- an identifier for persisted
run history.

```python
SummarizingCompaction(max_messages=60, keep_messages=20, receipts=True)
SlidingWindowCompaction(max_messages=80, keep_messages=40, receipts=True)
```

- **Deterministic receipt text.** The receipt text carries no timestamp and is a pure function
  of the compaction. The message part still has its ordinary request timestamp.
- **Honest wording.** `SummarizingCompaction` leaves a summary, so its receipt says the summary
  above is secondhand; `SlidingWindowCompaction` drops history outright, so its receipt says that context
  is gone. The blank-in-place strategies (`ClearToolResults`, `DeduplicateFileReads`,
  `ClampOversizedMessages`) keep every message and cross no boundary, so they emit no receipt.
- **Transcript handle.** Attach any capability exposing `compaction_transcript_handle() -> str | None`
  (the `TranscriptHandleProvider` protocol) and the receipt gains a `Persisted run handle:` pointer.
  `StepPersistence` implements it (returning its `run_id`), so attaching it is enough. The handle
  addresses the persisted *run*, not a pristine transcript: compaction's edits persist into the run's
  message history, so the run's latest snapshot reflects the **compacted** history and reading it back
  does not recover what the receipt says was dropped. A store keeping per-step snapshots may still hold
  pre-compaction steps, subject to its own retention (`max_snapshots_per_run` on the shipped stores).
- **Attribution.** The receipt's `by` field uses the same coarse family heuristic as the bridge prefix,
  with the same approximations -- see [the note below](#anchored-incremental-summarization-and-the-cross-model-bridge).
- **Observability.** Each receipt is also emitted as a `compaction.receipt` event on the
  `compact_messages` span.

> The receipt *text* is content, so it is opt-in (`receipts=False` by default) and its exact
> wording is provisional pending the benchmark eval-rig pass; the mechanism is structural.

## Pinning: content that survives compaction

Mark content that every shipped strategy must preserve verbatim with `pin`:

```python
from pydantic_ai_harness.compaction import pin

# In a ModelRequest placed in the run's message history (by a capability or the user):
pinned = pin('Durable task state the model must never lose across compaction.')
```

A pinned part is never summarized away or dropped; if a strategy would have discarded it, the
strategy re-injects it verbatim near the top of the surviving history. This is the least
invasive marking available today: pins use model-invisible `TextContent.metadata`, so their
contents remain ordinary user context while compaction can distinguish them from user turns.

`Planning` does **not** need pinning: its plan is re-injected ephemerally every request in
`wrap_model_request`, so it already survives compaction by construction. Pinning is for durable
task state and scratchpads that live *in* the history.

## Keeping user messages (`keep_user_messages`)

User turns are the highest signal-per-token content in a conversation, and losing them is the
main driver of resumption drift. `SummarizingCompaction(keep_user_messages=True)` preserves
the newest user turns from the summarized prefix alongside the summary. They consume the
existing `keep_messages` tail budget, so at most that many retained user messages and tail
messages survive together; compaction therefore does not grow retained copies on each cycle.
When `keep_tokens` is set, those same retained user messages and tail messages also share its
token budget; a user turn that does not fit is summarized instead.
Each retained turn is bounded to `keep_user_messages_max_chars` (default 20k) with an explicit
truncation marker when it overruns. The character budget applies per part, shared across the
text items of a multi-part prompt; images, audio, and cache points pass through untouched. This
supersedes `preserve_first_user_message` (which keeps only the first).

```python
SummarizingCompaction(max_tokens=120_000, keep_messages=20, keep_user_messages=True)
```

Retaining user turns leaves the summary, any receipt, and the retained turns as adjacent
`ModelRequest`s. Providers that require one request per turn -- Bedrock Converse and Gemini among
them -- never see that shape: Pydantic AI normalizes the history with `_merge_consecutive_messages`
after the `before_model_request` hooks run, combining adjacent requests into a single turn before
dispatch. `keep_user_messages` therefore needs no provider-specific handling.

## Anchored incremental summarization and the cross-model bridge

With `incremental=True` (the default), a prior summary is not re-summarized (which decays over
successive compactions). It is fed back as an anchored `<previous-summary>` block with an
*update* instruction -- preserve still-true details, remove stale ones, merge in new facts --
so the summary is a living document updated in place under a fixed structure.

> **Behavior change: `incremental=True` is the default.** Every existing `SummarizingCompaction`
> user gets a different summary-call prompt from this release on: once a prior summary exists, the
> summarizer is asked to *update* it under `<previous-summary>` rather than regenerate one from the
> conversation. The summaries it produces will read differently. Set `incremental=False` to keep the
> previous regenerate-from-scratch behavior.

`bridge_prefix=True` prepends a one-line note to the summary **only** when the
summarizer's model family differs from the family that produced the history (derived from the
history's `model_name` and the summarizer config), marking the summary as a cross-model
handoff so the resuming model builds on it rather than confabulating that it did the work
itself. It never fires in the common same-model case, so it is cheap. It defaults to `False`
because the note is prompt content.

The family token is a coarse approximation: drop any `provider:` prefix, then take the leading
token before the first `-` or `/`. It separates `gpt` from `claude` on ordinary references
(`openai:gpt-5.6-luna` -> `gpt`, `google:gemini-3.6-flash` -> `gemini`) and misreads several real ones:
`us.anthropic.claude-sonnet-4-5-v1:0` reduces to `0`, `ollama/llama3` to `ollama`, and a `fallback:`
model *string* to its last listed model rather than its first (a `FallbackModel` object is read
correctly, from its first model). Bridge and receipt attribution are therefore best-effort: a
misread family can suppress a bridge note or fire one between two same-family models. Neither
outcome changes what compaction keeps or drops.

> The update instruction and bridge-prefix wording are content, shipped minimal/neutral and
> flagged pending the eval-rig pass; the anchoring and family-gating mechanisms are structural.

## Out of scope

These strategies compress or drop context *inside* the window. Moving large tool outputs *out* of the
window -- overflowing them to a file the agent (or a subagent) can query on demand -- is a separate
capability, not lossy truncation. Prefer it over capping individual tool outputs.
