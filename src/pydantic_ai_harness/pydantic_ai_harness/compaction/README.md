# Compaction

A menu of strategies for keeping an agent's conversation history within a model's context
window. These Pydantic AI `Capability` classes edit the message history just before each request goes out. `FallbackCompaction` optionally triggers its chain at a token threshold and also works as a composing `CompactionStrategy`. Edits **persist** into the
run's message history, so a trim, clear, or summary carries forward to later steps (it is not
recomputed from the full history every turn).

All strategies preserve tool-call / tool-return **pairing** -- core does not validate this, and a
provider rejects an orphaned pair. The zero-LLM strategies never call a model.

On OpenAI and Anthropic, core also ships [provider-native compaction](https://pydantic.dev/docs/ai/capabilities/compaction/) --
the provider summarizes history server-side. The strategies here are the model-agnostic
alternative: they work with every model and keep the compaction logic (and its costs) under your control.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/compaction/)

## The menu

| Component | Cost | What it does | Reach for it when |
|---|---|---|---|
| `ClampOversizedMessages` | zero-LLM | Head/tail-truncates a single oversized part (response text, tool-call args) | One runaway generation blew past the context cap and no other strategy can reach it |
| `SlidingWindowCompaction` | zero-LLM | Drops the oldest whole messages down to a tail | You only need the recent turns and can discard old context entirely |
| `ClearToolResults` | zero-LLM | Blanks the content of old tool *results* in place, keeping the last `keep_pairs` | Tool outputs dominate context and can be re-fetched on demand (the cheap first tier) |
| `DeduplicateFileReads` | zero-LLM | Blanks every file read superseded by a newer read of the same file | The agent re-reads files and only the latest version matters |
| `SummarizingCompaction` | one LLM call | Summarizes older messages into a structured summary, keeping the recent tail | Old context still matters but must be compressed; use behind the cheap tiers |
| `RelevanceCompaction` | one judge request per new step, plus an optional summary | Keeps, reduces, or drops each older step by a judge model's answers | Some old tool results are still needed verbatim and others are dead weight |
| `TieredCompaction` | escalates | Runs cheap passes first, summarizes only if still over `target_tokens` | You want a sensible default: spend the expensive summary only when needed |
| `FallbackCompaction` | depends on chain | Tries the next strategy when one raises | Summarization can fail and deterministic truncation must keep the run alive |
| `CompactOnRequest` | depends on strategy | Gives the model a tool to ask for a compaction before its next request | The model knows when a phase of work ends better than a size trigger does |
| `WarnNearLimits` | zero-LLM | Injects an URGENT/CRITICAL warning as limits approach | You want the agent to wrap up rather than have its history rewritten |
| `ReportContextUsage` | zero-LLM | Reports context usage to your application; never edits history | You want a live context gauge in a UI |

Compaction updates persistent run history and replaces the current request view. List request-only injectors such as `Memory` after compaction so they apply to the compacted request. Earlier request-only edits are discarded, even when `ClampOversizedMessages` has nothing to shorten.

## Triggers

Instruction replacement and withdrawal records contribute their full rendered system text to token estimates. Superseded updates before a new instruction baseline are excluded.

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

### What counts toward `keep_tokens`

The retained tail is measured per message with `estimate_message_tokens`, so `keep_tokens` means
tokens as the provider counts them. A response counts at least its reported `output_tokens`, which
includes thinking that some providers return as an empty part with only a signature. The messages
between two responses that carry usage share the difference between their `input_tokens`, which is
exactly what the provider counted for them. A response whose billed usage sums several model
requests enters that difference with the input tokens of the one that measures its context: the
first request of a [Google long decode](https://pydantic.dev/docs/ai/models/google/#long-decode), or the last sampling pass
of an [Anthropic response that reports its passes](https://pydantic.dev/docs/ai/models/anthropic/#server-side-tool-loops).
That last pass also read the response's own earlier passes, so the difference credits them to the
messages before it, and the tail measures high. Before the first such response, after the last, and
where the history did not grow between two of them (as across an earlier compaction), `tokenizer` or the
~4-characters-per-token heuristic applies. Instructions are not part of the tail. A history with no
provider usage is measured with the heuristic as before.

**If you set `keep_tokens`, re-check it.** It used to be measured with the heuristic alone, which
on a thinking-heavy history can see a third of the real tokens or less, so the same budget kept
several times more history than it named. It now keeps the named amount, so fewer messages.

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

The window is the model's own `context_window`: the value its profile sets, including one you pass
as `profile={'context_window': ...}`, filled from
[`genai-prices`](https://github.com/pydantic/genai-prices) when no profile does. A `FallbackModel`
reports the smallest window among its models, since any of them may answer.
`resolve_context_window` is exported if you want the number yourself. Nothing is cached: only a
window the model or the registry states is ever treated as the real one.

The model consulted is `ModelRequestContext.model`, the one the request will be sent to, not the one
the run started with. A capability ordered earlier may replace it, and the budget follows.

### When the window does not resolve

Not every model has a known window. A local endpoint, a bespoke deployment, a model the registry
knows without a recorded window, and a `FallbackModel` none of whose models has a window all resolve
to nothing. The fraction is then taken of
`fallback_context_window`, which defaults to a conservative 200K (`DEFAULT_CONTEXT_WINDOW`):
compacting earlier than necessary costs one summary, overestimating costs the whole request.

Every capability that takes a fraction takes the fallback too, so you are not stuck with 200K on a
model you know the size of:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import SummarizingCompaction

agent = Agent(
    'ollama:qwen3',
    capabilities=[SummarizingCompaction(max_fraction=0.9, fallback_context_window=40_000)],
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
tool definitions and `FilePart` payloads, which no character estimate can see. A response whose
billed usage sums several model requests anchors on the input tokens of the one that measures its
context, plus its output: the first request of a [Google long decode](https://pydantic.dev/docs/ai/models/google/#long-decode),
whose billed input re-counts the earlier requests' output, or the last sampling pass of an
[Anthropic response that reports its passes](https://pydantic.dev/docs/ai/models/anthropic/#server-side-tool-loops),
whose billed input re-counts the context of each pass. For the messages
after the anchor (and for whole histories with no reported usage), the estimator counts every part
that is sent: prompts, system prompts, tool calls and their results, retry prompts,
extended-thinking blocks, provider-side tool results, and the instructions, once. When the
instructions changed since the anchor, the new set replaces the old one in the estimate rather than
adding to it, since a request sends only one. That estimated portion is a ~4-characters-per-token
approximation, not a tokenizer; pass `tokenizer=` to any strategy to measure with the real one.
`FilePart` and the files a provider-side tool returns are not counted there -- their payload is
binary, and its length in characters would mean nothing. For the same reason a file inside a tool
return counts as the `See file <identifier>.` reference a text-only tool result carries in its place. Newly revealed tool schemas pending in the current request are conservatively estimated
by the implementation, since they are not covered by the earlier anchor.

**If you already set an absolute `max_tokens`, re-check it.** The estimator used to count only user
and system prompts, tool returns, response text, and tool calls. `ThinkingPart` / `CompactionPart`
content, retry content, `NativeToolCallPart` / `NativeToolReturnPart`, and the most recent
`ModelRequest.instructions` are now counted too, so the same history measures higher and an
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
from pydantic_ai_harness.compaction import ContextUsageEvent

agent = Agent(
    'anthropic:claude-sonnet-5',
    capabilities=[
        SummarizingCompaction(max_fraction=0.9, keep_messages=20),
        ReportContextUsage(),
    ],
)

@agent.on_event(ContextUsageEvent)
async def show(ctx, event):
    print(f'{event.fraction:.0%}')
```

Each reading carries `used_tokens`, `window_tokens`, and `resolved` -- `False` when the window is the
fallback rather than the model's real one, so a gauge can show that the percentage is a guess.
Migration: `on_usage` remains supported but is deprecated. Move its callback body to a
`ContextUsageEvent` subscription.

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

`focus` steers strategies that write prose or judge relevance -- `SummarizingCompaction` and
`RelevanceCompaction`, via the exported `SupportsFocus` protocol's `with_focus` -- and is passed over
by the ones that drop or blank content by rule, since they have nothing to steer. `TieredCompaction` is focusable when any of its tiers is, so a focus reaches the
summarizing tier rather than stopping at the wrapper.

A compaction that changes the history emits the same `compact_messages` span the in-run path emits,
so an instrumented application sees one shape however compaction was triggered. Pass `tracer=` to
record it; without one the span goes to a no-op tracer.

Pass `conversation_id=` with the id of the conversation being compacted. It is set on the throwaway
context, so the summary run of `SummarizingCompaction` is recorded under that conversation (its
messages and its `gen_ai.conversation.id` span attribute) rather than under a fresh id of its own.
Inside a run, the summary run takes the parent run's `conversation_id` without this.

## `FallbackCompaction`: recover when a strategy fails

`TieredCompaction` advances when a successful tier does not reclaim enough. `FallbackCompaction`
advances when a strategy raises an exception selected by `fallback_on`, which defaults to
Pydantic AI's `ModelAPIError` and `FallbackExceptionGroup`. The latter is raised when every model
in a `FallbackModel` fails. When its trigger runs the chain, it also advances past a
`SummarizingCompaction` whose result isn't smaller than the history (see
[A summary that isn't smaller](#a-summary-that-isnt-smaller)), and leaves the history unchanged if
no strategy shrinks it. Each attempt receives a fresh list containing the original message
objects, so list-level changes by a failed strategy do not affect its fallback. Strategies must
still avoid mutating message objects. If every strategy fails, the
last exception is re-raised. Non-matching exceptions, cancellation, and other `BaseException`
subclasses pass through immediately; `fallback_on` rejects types that do not derive from
`Exception`.

```python
from pydantic_ai_harness import FallbackCompaction, SlidingWindowCompaction, SummarizingCompaction

fallback = FallbackCompaction(
    max_fraction=0.85,
    fallback_chain=[
        SummarizingCompaction(max_messages=1, keep_tokens=20_000),
        SlidingWindowCompaction(max_messages=1, keep_tokens=20_000),
    ]
)
```

Register `fallback` directly with `Agent(..., capabilities=[fallback])`. Its optional
`max_tokens` or `max_fraction` trigger runs the chain only when estimated context tokens
exceed the threshold. Fractions resolve against the request's model; `context_window`
overrides its window and `fallback_context_window` supplies an unknown model's window.
`tokenizer` customizes token estimation. The hook preserves pinned parts and persists
compacted history, emitting the standard `compact_messages` span when history changes.

With neither trigger configured, the request hook does nothing. Direct `compact()` and
`compact_now()` calls run the chain regardless of its threshold, so it remains usable
inside other composing strategies and for manual compaction. Each child's own trigger
is bypassed when the chain calls its `compact()` method.

### Background compaction (experimental)

Set `background_start` to start the chain's first strategy before the trigger fires, so a summary
is usually ready by the time compaction is due and the run does not wait for a model call then.

```python
from pydantic_ai_harness import FallbackCompaction, SlidingWindowCompaction, SummarizingCompaction

compaction = FallbackCompaction(
    max_fraction=0.85,
    background_start=0.85,
    fallback_chain=[
        SummarizingCompaction(max_messages=1, keep_tokens=20_000),
        SlidingWindowCompaction(max_messages=1, keep_tokens=20_000),
    ],
)
```

- **Start.** Before each model request and when a run ends, once the context passes
  `background_start` of the trigger (here 85% of the 85% threshold), the first strategy starts
  compacting a copy of the history in a background task. Neither the request nor the run waits for
  it. A run that ends already over the trigger starts none; its next request compacts as it would
  without `background_start`. It needs a `max_tokens` or `max_fraction` trigger.
- **Hold.** The result is held, not applied, so the history, and the prompt cache built on it, stay
  as they are until compaction is due.
- **Swap.** When the trigger fires, the chain waits for a result still being written and then swaps
  it in: the compacted copy replaces the messages it was made from, and every message added since
  follows it verbatim. The strategy keeps tool calls and returns paired within the copy, and the
  messages added since start where a model request or a run did, so pairs stay intact.
- **Fall back.** The held result is dropped, and the chain runs as it would without
  `background_start`, when the history no longer starts with the copy (rewritten, rewound, or
  compacted another way), the request goes to another model, the strategy raised or was cancelled,
  or the swapped-in history would still be over the trigger because too much arrived meanwhile.
  Instructions are not compared: the summary is written from the messages alone.
- **Lifetime.** Each conversation (`RunContext.conversation_id`) holds at most one. It outlives the
  run that started it, so it can finish between runs, until it is used, replaced, or cancelled.
  Below the start, one whose conversation was rewritten (same first message) or moved to another
  model is cancelled at the next check; a run whose history starts elsewhere, such as a side run with
  an empty `message_history` under the same conversation ID, leaves it alone. A conversation that
  will not continue, such as a finished delegated task, keeps its job until you call
  `discard_background_compaction(conversation_id)`, or until it is the oldest of eight held and
  another starts. `aclose()` cancels them all: call it when you are done with the instance.
- **Outside a run.** `take_background_compaction(messages, model=..., conversation_id=...)` waits
  for the conversation's held result and returns the history with it swapped in, or `None` when it
  does not apply, so a manual compaction can fall back to `compact_now`.
  `discard_background_compaction(conversation_id)` also suits rewriting the history another way, and
  `background_conversations` lists the conversations that hold one.
- **Usage.** The work runs on the context of the run that started it, so a summary call counts
  toward that run's usage and usage limits, as a blocking one does. When it finishes after that run
  has ended, its usage is added to that run's `RunUsage` then. A result that goes unused still cost
  its tokens.

Only the first strategy runs in the background, so put the model-backed one first. A `with_focus()`
copy holds nothing. Background compaction is not recorded as a durable operation, so a run inside a
durable workflow or flow with `background_start` set raises `UserError` before its first request.

## `CompactOnRequest`: let the model choose the moment

Size triggers do not know where the task is. One can fire in the middle of a sub-task and summarize away the file the model was about to edit, or stay quiet just after a phase ends, when a large amount of investigation output is no longer needed. The model often knows when that moment comes. `CompactOnRequest` gives it a `compact_context(focus=None)` tool to say so, and runs a strategy you choose before the next model request of the same run:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import CompactOnRequest, SummarizingCompaction, TieredCompaction

agent = Agent(
    'openai:gpt-5.6-luna',
    capabilities=[
        # Automatic size-based compaction stays as is.
        TieredCompaction(tiers=[SummarizingCompaction(max_messages=1)], target_fraction=0.8),
        CompactOnRequest(
            # The strategy's own trigger is required but bypassed: `compact` is called directly.
            SummarizingCompaction(max_messages=1, keep_messages=6),
            min_tokens=30_000,  # refuse requests that would save little
            max_per_run=2,
        ),
    ],
)
```

The tool compacts nothing itself. It records the request and returns a short acknowledgement, and the compaction runs in the next `before_model_request`, with the same `messages` and context automatic compaction gets. The tool call and its return are the newest messages, so they stay paired in the kept tail and the model can see that it asked. The `focus` the model passes reaches the strategy through `with_focus` when the strategy implements `SupportsFocus` (as `SummarizingCompaction`, `RelevanceCompaction`, `TieredCompaction` and `FallbackCompaction` do), and is ignored otherwise.

The guards answer the model in the tool return, so it learns why nothing happened:

- below `min_tokens`, estimated the way the token triggers estimate, the request is refused;
- after `max_per_run` accepted requests in a run, further requests are refused;
- a second call while one is already scheduled, such as two parallel calls in one response, is refused.

`CompactOnRequest` orders itself after the automatic compaction capabilities on the agent, wherever you list it. When one of them already compacted the history for the request, the model's request is dropped rather than compacting twice in one step, and it does not count against `max_per_run`.

Under durable execution, the guards are evaluated in a capability hook rather than in the tool function, because the tool may run in an activity that sees neither the history nor the run's state. The strategy runs as the `compact` durable operation of `CompactOnRequest`, so a replay reuses the recorded history instead of calling the summarizer's model again. Under Temporal that operation is an activity whose run context omits `tracer` and `capabilities`, so a strategy that reads them fails there: `RelevanceCompaction`, and `receipts=True` on `SummarizingCompaction` or `SlidingWindowCompaction`. For the same reason, receipts recorded inside the activity, or skipped on a DBOS replay, add no `compaction.receipt` events to the span.

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
[`ToolOutputLimits`](../tool_output_limits/), which reduces a return when the tool produces it.
`ClearToolResults` keeps the newest `keep_pairs` results intact, and with `keep_pairs=0` it blanks a
fresh return outright rather than shrinking it.

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

## `RelevanceCompaction`: keep what the task still needs

`RelevanceCompaction` asks a judge model about each older step of the history and keeps the step verbatim, reduces it, or drops it, based on the answers. Reach for it when the older context is uneven: some tool results hold code or error text the agent still needs exactly, others are dead weight, and one summary of everything would lose the exact parts.

```python
from pydantic_ai import Agent
from pydantic_ai_harness.compaction import RelevanceCompaction, SummarizingCompaction

agent = Agent(
    'anthropic:claude-opus-5-5',
    capabilities=[
        RelevanceCompaction(
            'system-one:nimble',
            summarizer=SummarizingCompaction(model='anthropic:claude-haiku-5-5', max_messages=1),
            max_fraction=0.8,
        )
    ],
)
```

The judge, `model`, is required. A [decision model](https://pydantic.dev/docs/ai/models/decision/), such as `system-one:<model>` on [System One](https://pydantic.dev/docs/ai/models/system-one/) (including the models Ollama runs locally) or TypeSafe's `typesafe:jev-latest`, answers with a probability. Any other model that supports structured output fills the same two fields, so `model` also takes a `FallbackModel` that tries a decision model first and a language model second.

A *step* is a model response together with the requests that answer it: its tool results, retry feedback, and the next user prompt. A step is kept or removed whole, so a tool call never loses its result. Everything before the first response (the system prompt, the first user message, and an earlier summary) and the most recent `keep_tokens` (default 8,000) of history are never judged. Pinned content survives even when its step is removed.

Each remaining step is judged in one request with two questions:

- `needed_verbatim`: "Will the agent need the exact content of this earlier step, such as code, file contents, identifiers, numbers, commands or error text, to finish the current task?"
- `still_matters`: "Does this earlier step record a fact, decision, change, result or user instruction that still matters for the current task?"

The judge is shown the current task (the first and the latest user prompts, plus any `focus` passed to `compact_now`), the most recent part of the conversation, and the step itself, including its tool results. A step longer than `judge_step_tokens` (default 2,000) is shown by its start and end, and the task and the recent conversation are each clamped to `judge_context_tokens` (default 500). The defaults keep a request within the 8,192-token input of small local decision models, since code and JSON take about twice as many tokens as the ~4-characters-per-token estimate suggests. The judge is told that all of it is untrusted data, and text that looks like the prompt's own tags is escaped.

Steps are chosen by rank against a token budget, never by comparing a probability to a fixed threshold: a judge's probabilities shift as a whole with the shape of the request, while the order it puts steps in holds. The history is compacted to `target_tokens` (or `target_fraction`), which defaults to half the trigger:

1. Steps are kept verbatim in order of `needed_verbatim` while they fit the target.
2. The rest are reduced in order of `still_matters`. With a `summarizer`, they are written into one summary placed after the head of the history, up to `summary_input_tokens` (default 30,000) of steps per compaction. Without one, each of their tool results is cut to its first `truncated_result_chars` (default 300) characters plus a note that the tool can be called again, while that still fits the target.
3. Whatever is left is dropped, except a step holding a user prompt: without a summarizer it is cut, and with one it is summarized while it fits `summary_input_tokens` and kept verbatim otherwise.

The `summarizer` is a `SummarizingCompaction` used for its model, prompts and settings; its trigger and `keep_*` fields are never consulted, so any trigger such as `max_messages=1` works. With the summarizer's `incremental=True` (the default) the earlier summary is updated; with `incremental=False` it is summarized again along with the steps. `receipts=True` adds a [receipt](#compaction-receipts) when steps were summarized or dropped.

Judge requests run concurrently, up to `judge_concurrency` (default 8) at once. Every judgement is remembered on the instance under the task and the step's content, so a later compaction with the same task judges only the new steps. All the judge requests of one compaction count as one request against a `UsageLimits` request limit, checked before any of them is sent. Their tokens are added to the run's usage and checked against its token and cost limits once they return, like the summary request's. On a laptop running Ollama, one judge request for a step of a few thousand tokens took 5 to 10 seconds, so the first compaction of a long history can take minutes there; later ones judge only the new steps.

A judge error propagates, as a summary error does. Wrap the strategy in `FallbackCompaction` to fall back to one that makes no model requests:

```python
from pydantic_ai_harness.compaction import FallbackCompaction, RelevanceCompaction, SlidingWindowCompaction

compaction = FallbackCompaction(
    max_fraction=0.8,
    fallback_chain=[
        RelevanceCompaction('system-one:nimble', max_fraction=0.8),
        SlidingWindowCompaction(max_messages=1, keep_tokens=20_000),
    ],
)
```

As a `TieredCompaction` tier, set its `target_tokens` explicitly: a tier's `compact` runs with its own target, and the default of half the trigger means nothing when the trigger is only there to satisfy the constructor.

With a durable-execution capability attached, the judgements and the summary run as the durable operations `judge` and `summarize` of the capability's stable default `id`, `relevance_compaction`, so replay reuses them instead of calling the models again.

The two-question split and concurrent per-step requests come from [fast-jev-compaction](https://github.com/tamaratran/fast-jev-compaction) (MIT), which compacts Claude Code transcripts with TypeSafe's Jev. Its measurements also motivated showing the judge each step's content and choosing steps by rank. No code is shared.

## `SlidingWindowCompaction` and `ClearToolResults` options

`SlidingWindowCompaction` keeps the last `keep_messages` down to a tail; pass `keep_tokens` instead to retain
a minimum amount of original message text rather than a message count. By default `preserve_first_user_message=True` keeps the first user
turn even when it falls outside the window, so the agent does not lose the original task. `keep_messages=0` still keeps
the newest message, the request being answered.

`ClearToolResults` keeps the last `keep_pairs` intact. A retried return, which asks the model to make
the call again, is never cleared and doesn't count toward `keep_pairs`. Set `clear_tool_inputs=True` to also blank the
arguments of the cleared calls. A `load_capability` call keeps its argument, because Pydantic AI reads from
it which capability is loaded. Set `exclude_tools` to a set of tool names whose results are never
cleared. Framework-typed tool results -- core's `search_tools` and `load_capability` returns -- are
left intact (a small token floor), because their structured content is re-parsed on later requests and
rewriting it via `dataclasses.replace` would bypass validation and corrupt the part. `FileSystem`'s typed
`read_file` results (`ReadFileReturnPart`) are cleared like any other result and keep their type, unless
their `metadata` is something other than `None` or a `dict`: a cleared result is marked in its
`metadata`, so one whose metadata can't hold the mark is left uncleared.

With `max_messages`, the count starts after the last cleared result, because clearing blanks content
but removes no messages. Clearing then happens in batches: after a clear, the next one waits until
more than `max_messages` messages follow the last cleared result, so the requests in between extend
the cached prefix instead of rewriting it. The kept pairs count toward that: right after a clear,
about `2 * keep_pairs` messages follow the last cleared result, and each tool step adds two more. So
each clear is followed by another after about `(max_messages - 2 * keep_pairs) / 2` tool steps, and
at or below `2 * keep_pairs` it clears on every request. Pick the value with that gap in mind.

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
running agent's model. That model has to write text, so an agent running a model that can't, such as a
decision model like TypeSafe's Jev, needs `model=` set to a language model; without it, the first compaction
raises a `UserError` saying so. A summary that is empty or only whitespace is retried within the summary
run's output retries. Exhausting those retries raises `UnexpectedModelBehavior` rather than replacing
history with an empty summary. Its nested summary run inherits the parent usage limits and reserves
one request from a finite request limit for the pending parent request. These limits also apply to
retries: if the summary run exhausts its budget first, it raises `UsageLimitExceeded` instead. To fall
back on either failure, set `FallbackCompaction`'s
`fallback_on=(UnexpectedModelBehavior, UsageLimitExceeded)`, preferably with a fallback strategy that
makes no model requests. This does not increase the parent run's limits.

Pass `model_settings` to give the dedicated summary call
settings that differ from defaults carried by that model; the supplied settings merge over the model defaults
without mutating the model or the settings dictionary. Pass `summarization_capabilities` to attach
capabilities to the summary agent; capabilities on the outer agent do not run on the summary call.

The summary request is non-streaming unless `event_stream_handler` is set. Supply a handler to watch
the summary as it is written, or pass `drain_summary_events` to take the streaming request path
without handling the events -- which is what a summarizer endpoint that rejects non-streaming
requests needs:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.compaction import SummarizingCompaction, drain_summary_events

agent = Agent(
    'openai:gpt-5.6-terra',
    capabilities=[
        SummarizingCompaction(max_messages=60, event_stream_handler=drain_summary_events),
    ],
)
```

Neither transport works everywhere, which is why this is a choice rather than a default: some
endpoints reject non-streaming requests and others reject streaming ones. The handler receives the
summary run's own `RunContext` and event stream; the outer `Agent.run(...)` handler is not inherited
and never sees the summary token deltas.

By default `incremental=True` updates the newest existing summary from a prior compaction as an
anchor rather than regenerating it from scratch. This changes the summary-call prompt from earlier
releases; set `incremental=False` to retain the prior regeneration behavior. `preserve_first_user_message=True` keeps the original task turn even
when it falls outside the window. Pass `keep_tokens` to retain a minimum amount of original message text
instead of using `keep_messages`.
`keep_messages=0` still keeps the newest message, the request being answered, alongside the summary.

The summary is a `CompactionPart` in a `ModelRequest` of its own, after the original system prompts,
which stay exactly as they were, and the original first user turn when `preserve_first_user_message`
keeps it. Each provider model receives it as user-voice text inside `<conversation_summary>`, saying a
separate summarization model wrote it (a `FunctionModel` gets the part itself), never as part of the
system prompt: a model wrote it, from a history that can include tool output. Because it is a compaction
part, Pydantic AI also treats it as a compaction boundary, so tool discoveries from before it are announced
again, on-demand capability loads from before it are recorded again after it (see "On-demand capabilities
across compaction" below), and `WarnOnCacheBusts` reports the cache drop it causes as
`compacted`. The part carries `{'pydantic-ai-harness.compaction.summary.v1': True}` in its `metadata`,
which the model never sees; that marker, together with the part having no provider data, is how
`SummarizingCompaction` recognizes its own summaries, so a `CompactionPart` your application wrote is
left alone. It's on the part, not the message, so it survives Pydantic AI merging the summary's request
with the requests next to it. So does `'pydantic-ai-harness.compaction.summary.created_at'`, the ISO 8601
time the summary was written: the merged request keeps the timestamp of a request next to it, which can
be an older prompt the compaction preserved. A history saved before this, with the summary in a system prompt opening
with `Summary of previous conversation:`, keeps working: that summary anchors the next one and is
replaced by it.

Both prompt surfaces of the summary request are fields: `summary_prompt` is the user-turn template (it
must contain a `{messages}` placeholder and no other replacement field; any other prompt emits a `HarnessDeprecationWarning` and will raise `ValueError` in a future release), and `instructions` sets the internal agent's static instructions,
which Pydantic AI sends in the request's system prompt. Override `instructions` when the summarizer
endpoint requires a fixed leading instruction.

The messages served into that template are rendered as a transcript with one `<message>` element per part
(each user prompt, system prompt, response text, tool call and tool return), tagged with its role and,
for a tool call or return, its tool name. Retry feedback, the correction a rejected tool call or output
got, is a `role="retry_feedback"` element carrying the retry's `cause` (and, for a tool call, its tool
name), so the summary records the correction rather than the rejected call as something that happened;
feedback that answers no tool call is shown as the model saw it. Content that looks like the transcript's
own tags or a framing tag Pydantic AI writes (`<system>`, `<context>`, `<tool_result>` and the rest), in
any case or spacing, is escaped, so a tool return stays inside its own message and can't end the
transcript early. Each tool return and retry feedback is capped, after that escaping, at
`tool_return_max_chars` (default 500) characters using the same explicit truncation marker as kept user
turns. Raise it, or set it to `None` to render each return and retry feedback whole, when the summarizer's context window is large enough to absorb the payloads.
Content a harness capability added on the user channel and tagged with an `origin` in its
`TextContent.metadata` starting with `pydantic-ai-harness.` is not a user message. Today that's a
`BackgroundTools` result, `TrajectoryJudge` feedback and a `Memory` snapshot, including one an older release marked with a plain string. Tagged content is
rendered as a `<harness_data origin="...">` element, capped like a tool return, rather than as
`role="user"`. When the transcript holds such content, the summary request also tells the summarizer not to
treat it as the user's requests or standing constraints; other summary requests are unchanged.
`max_tokens` and `keep_tokens` control when compaction runs and which history messages are retained; they
do not cap the summary-request payload.

### Token retention after automatic compaction

For both strategies, `max_tokens` is the trigger and positive `keep_tokens` is the minimum unchanged
recent suffix. Retention includes the boundary message and required tool dependencies, including named
retries. Counts follow [What counts toward `keep_tokens`](#what-counts-toward-keep_tokens), excluding
attached `ModelRequest.instructions`. Histories below the target stay unchanged. Summaries, receipts, pins, and older user copies do not replace the protected tail.

Allow headroom for whole-message/tool-dependency overshoot and added context; `keep_tokens` is not a
request-size limit. A value of `0` sets a zero-token target rather than a minimum-retention floor;
`None` selects message-count retention with `keep_messages`.

When everything before the suffix is already accounted for (an earlier summary, receipts, pins, and
messages kept verbatim anyway, such as the preserved first user turn), `SummarizingCompaction` has
nothing new to summarize and leaves the history unchanged without a summary call, even if it is
still above the trigger.

See the [token-retention guide](https://pydantic.dev/docs/ai/harness/compaction/#token-retention-after-automatic-compaction)
for the automatic-compaction example, tokenizer accounting, and fallback/budget guidance.

## A summary that isn't smaller

The summary replaces only part of the history: system prompts, the first user turn, the recent tail,
pinned parts and receipts are kept as they are. When the summarized span is short or the summary is
long, the result can be larger than the history it replaces. Automatic compaction is there to make
the request smaller, so when `SummarizingCompaction` runs on its own trigger, the compacted history is
kept only when it is estimated smaller than the original, using the `tokenizer` of the capability
whose trigger fired (`FallbackCompaction`'s inside a chain), or the ~4-characters-per-token estimate
without one. Otherwise the request goes out with the history unchanged, any receipt the attempt
wrote is dropped, and a `compact_messages` span with `compaction.skipped` set to `not_smaller`
records the attempt and its estimated size. The summary call's usage still counts, and while the
trigger stays exceeded, the next request in the run tries again with another summary call. Inside
`FallbackCompaction`, the next strategy runs instead, so putting a `SlidingWindowCompaction` after
the summarizer bounds both the history and that cost. A direct `compact()` call, `compact_now()` or
a `TieredCompaction` tier returns the summary either way.

## Usage accounting

The summary call is a real request to the model, so its full usage -- tokens **and** the request
itself -- is folded into the run's `ctx.usage`. This is deliberate: it keeps cost honest, keeps the
request count consistent (a model request that didn't count as one would be the surprise), and lets a
`UsageLimits` request limit catch a runaway compaction. The nested run receives the other parent limits unchanged;
the finite request limit is reduced by one so it cannot spend the slot already approved for the parent request.
The summary run is filed under the parent run's `conversation_id`, so its requests, spans, and cost
group with the conversation it compacts.
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
    return call.args_as_dict().get('path')
```

## Tracing

When core instrumentation is active (the `Instrumentation` capability, `agent.instrument`, or
`Agent.instrument_all()`), each strategy emits a `compact_messages` span on the run's tracer the
moment it actually compacts -- that is, in `before_model_request`, once the strategy's threshold is
exceeded (`ClampOversizedMessages` emits only when a part is actually clamped). `TieredCompaction`
emits a single span for the whole escalation rather than one per tier, because it drives each tier's
`compact` directly. An automatic summary that is discarded for not being smaller also emits one,
with `compaction.skipped`. Without instrumentation the tracer is a no-op, so the span adds no overhead.

The span name is the static `compact_messages`; the strategy is an attribute, not part of the name,
to keep span cardinality low. Attributes:

| Attribute | Type | Meaning |
|---|---|---|
| `gen_ai.conversation.compacted` | bool | `true` when the history was compacted, absent on a skipped attempt; the OpenTelemetry GenAI convention's flag for a compacted context |
| `compaction.strategy` | str | Strategy class name (e.g. `SlidingWindowCompaction`, `SummarizingCompaction`) |
| `compaction.messages_before` | int | Message count before compaction |
| `compaction.messages_after` | int | Message count after compaction |
| `compaction.tokens_before` | int | Estimated token count before compaction |
| `compaction.tokens_after` | int | Estimated token count after compaction |
| `compaction.model_requested` | bool | `true` when the model asked for the compaction through `CompactOnRequest`; absent otherwise |
| `compaction.background` | str | Only with `background_start`: `used`, `stale`, or `failed`, what became of the background compaction |
| `compaction.skipped` | str | `not_smaller` when an automatic summary was discarded because it wasn't smaller than the history; the `*_after` attributes then describe the discarded result. Absent otherwise |

`gen_ai.conversation.compacted` is the GenAI semantic convention's flag; the rest is
harness-specific. Token counts use the strategy's `tokenizer` when set, otherwise the
~4-chars-per-token heuristic.
Raw message content is not recorded.

`SummarizingCompaction` runs its summarizer as a nested `Agent` named `summarizing_compaction`,
so under `Agent.instrument_all()` (or `logfire.instrument_pydantic_ai()`) its runs carry
`agent_name = summarizing_compaction`. Filter on that to track summarization usage and cost
separately from the parent agent.

`FallbackCompaction` with `background_start` also emits a `background_compaction` span around the
background work, with `compaction.strategy` and `compaction.messages_before`. It ends when the work
does, which can be after the run that started it; a failure is recorded on it. The summary run inside
it is the same `summarizing_compaction` agent run.

`RelevanceCompaction` adds a `select compaction steps` span inside `compact_messages`. It records the number of steps and how many were judged rather than remembered (`relevance_compaction.steps`, `relevance_compaction.steps_judged`), the steps and tokens of each tier (`relevance_compaction.keep.steps`, `relevance_compaction.keep.tokens`, and the same for `reduce` and `drop`), and one `relevance_compaction.step` event per step with its position, size, both answers, whether it was remembered, and its tier. No message content is recorded. The judge's own runs are not instrumented, because their prompts carry the conversation.

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
- **Receipt slot.** `SlidingWindowCompaction` counts its receipt toward `keep_messages`, so
  `keep_messages=40` keeps the receipt and 39 messages. The receipt never takes the last slot:
  `keep_messages=1` keeps the receipt and the newest message.
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

## On-demand capabilities across compaction

Pydantic AI counts an on-demand capability as loaded while its `load_capability` exchange is in the
history after the latest compaction, so a strategy that summarizes or drops the exchange would unload it:
its tools would leave the request and its instructions would no longer be sent, while the summary may
still say it's loaded. `SummarizingCompaction`, `SlidingWindowCompaction` and `RelevanceCompaction` remove
history, so when they remove a capability's load they record it again: a `load_capability` call and its
return, with the original return's content, right after the summary (or ahead of the kept messages when
there's no summary). The capability stays loaded, its tools stay in the request, and Pydantic AI states
its current instructions again in the system voice. The recorded exchange carries
`{'pydantic-ai-harness.compaction.recorded-load.v1': True}` in the `metadata` of its `ModelResponse` and
of its `LoadCapabilityReturnPart`, which the model never sees. Summaries leave every `load_capability`
exchange out, the model's own included: which capabilities are loaded is carried by the exchanges, recorded
again after each compaction, never by the summary.
Only capabilities the run still registers as on-demand are recorded. `compact_now`, which has no run to
ask, records every load, and the next run ignores a load of a capability it doesn't register.

Set `preserve_loaded_capabilities=False` on the strategy to let them unload instead. The strategy then
leaves a note in the same place naming the capabilities that are no longer loaded, so the model can load
the ones it still needs rather than finding out from a refused tool call. The note is user-channel content
whose `TextContent.metadata` is
`{'origin': 'pydantic-ai-harness.compaction.unloaded-capabilities.v1', 'capabilities': [...]}`, so a later
summary treats it as data rather than as the user's words.

`ClearToolResults`, `DeduplicateFileReads` and `ClampOversizedMessages` keep every message and leave
`load_capability` calls and returns as they are, so loads survive them unchanged. Provider-native
compaction (`OpenAICompaction`, `AnthropicCompaction`) runs no harness strategy, so it still unloads them.

## Keeping user messages (`keep_user_messages`)

User turns are the highest signal-per-token content in a conversation, and losing them is the
main driver of resumption drift. `SummarizingCompaction(keep_user_messages=True)` preserves
the newest user turns from the summarized prefix alongside the summary. In message-count mode, they consume the
existing `keep_messages` tail budget, so at most that many retained user messages and tail
messages survive together, except that one tail slot is reserved for the request being
answered when retained user turns would otherwise consume it; compaction therefore does not
grow retained copies on each cycle. With a positive `keep_tokens`, the original suffix takes
precedence: `keep_messages` caps only the additional older user turns, which do not reduce the
retained tail. With `keep_tokens=0`, retained user copies must fit a zero-token budget, and the
newest tail message still survives.
Each retained turn is bounded to `keep_user_messages_max_chars` (default 20k) with an explicit
truncation marker when it overruns. The character budget applies per part, shared across the
text items of a multi-part prompt; images, audio, and cache points pass through untouched.
Harness-added content on the user channel, such as a background tool's result, isn't a user turn: it is
left out of what `keep_user_messages` and `preserve_first_user_message` carry over from the summarized
prefix (including from a turn an earlier compaction kept), and summarized as data rather than as the
user's intent. The recent suffix compaction leaves unsummarized is kept as is, harness content included.
This supersedes `preserve_first_user_message` (which keeps only the first).

```python
SummarizingCompaction(max_tokens=120_000, keep_messages=20, keep_user_messages=True)
```

Retaining user turns leaves any receipt and the retained turns as adjacent `ModelRequest`s after the
summary. Providers that require one request per turn -- Bedrock Converse and Gemini among
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
capability, not lossy truncation. Within `SummarizingCompaction`, `tool_return_max_chars` makes the
summarizer's per-return cap tunable (or `None` to render returns whole), but the summary request still
reads a lossy rendering; prefer tool output limits when a payload must be queryable in full.
