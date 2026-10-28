# System Reminders

> [!NOTE]
> The `Reminder` helper is not re-exported at the top level -- import it from the submodule:
>
> ```python
> from pydantic_ai_harness import SystemReminders
> from pydantic_ai_harness.system_reminders import Reminder
> ```
>
> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

Re-inject behavioral guidance mid-run to counter instruction fade -- without invalidating the prompt cache.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/system_reminders/)

## The problem

Long multi-turn runs suffer instruction fade: after many tool-use turns the model progressively ignores the guidance it was given at the start. A single start-of-session system prompt is not enough for extended work. The fix is to re-state targeted guidance mid-run -- on a fixed cadence, or reactively when a condition is detected.

## The solution

`SystemReminders` injects reminders on each model request, either statically (`Reminder`, on a cadence) or dynamically (a callable that reads the run context). Each firing enqueues a [request-scoped system prompt](https://pydantic.dev/docs/ai/core-concepts/message-history/#request-scoped-system-prompts) at the **tail** of the request. The model sees a request-scoped prompt only in the model request it's part of: the next request, even the next step of the same run, no longer shows it. It's recorded in the message history like any other part.

- On Anthropic models that take mid-conversation system messages, it's sent as a turn-scoped system message (`clear_at`): every copy stays on the wire, the API stops rendering each one once the next request arrives, and a cleared one costs no tokens. Leaving earlier reminders in place is what keeps the cached prefix and the model's earlier reasoning valid; deleting one would invalidate every thinking block after it on models with preserved thinking.
- Everywhere else, it's sent with its own request only, at the end, rendered like any other [mid-conversation system prompt](https://pydantic.dev/docs/ai/core-concepts/message-history/#mid-conversation-system-prompts), and left out of later ones.
- The cache breakpoints your model's caching settings place (such as `anthropic_cache`) go *before* the reminder, so the cached prefix (tools + system + real conversation) is reused turn over turn and only the small reminder falls outside it.

Injecting into the system prompt or instructions instead would sit at the front of the request, so every reminder would bust the cached prefix. This capability avoids that.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import SystemReminders
from pydantic_ai_harness.system_reminders import Reminder

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[
        SystemReminders(
            reminders=[Reminder('Stay focused on the original request.', interval=5, tag=None)],
        )
    ],
)

result = agent.run_sync('Refactor the auth module and add tests.')
print(result.output)
```

## Static reminders

A `Reminder` fires on a cadence within a run:

| Field | Purpose |
|---|---|
| `content` | The reminder text. |
| `interval` | Fire every N model requests (`interval=3` fires on the 3rd, 6th, ...). |
| `first_after` | Request number of the first fire, then every `interval` after. `None` = first multiple of `interval` (plain modulo). |
| `trigger` | Predicate over `RunContext`. When set, fires only when it returns `True` *and* the cadence matches. |
| `max_fires` | Cap the number of fires per run. `None` = no limit. |
| `tag` | Also wrap the content in `<tag>\ncontent\n</tag>`. `None` sends the raw content. Left unset, it's `None` and warns that the default changed. |

A reminder adds recency; it doesn't keep a rule in force. The model sees each firing only in the request it fires on, so with `interval>1` it sees the reminder on the requests where it fires and not on the ones in between, and a `trigger` or `max_fires` narrows that further. Text the model must follow at every step belongs in the agent's [instructions](https://pydantic.dev/docs/ai/core-concepts/agent/#instructions), which every request sends.

A reminder is a system prompt, so the model is already shown it as one: as a native system message where the model supports it, and in a `<system>` tag otherwise. It needs no tag of its own, so pass `tag=None`. The default used to be `'system-reminder'`: a `Reminder` that leaves `tag` unset sends the raw content and warns, and `tag='system-reminder'` restores the old wrapping.

The `tag` wrapping applies only to static `Reminder` content. A `str` a dynamic callable returns (including `GoalReanchor`'s) is sent as written; an `LLMReminder`'s `GeneratedReminder` is quoted as data, as described under [Dynamic reminders](#dynamic-reminders).

## Dynamic reminders

A dynamic reminder is any callable `(RunContext) -> str | None` (sync or async), evaluated on every model request. Return a string to inject, or `None` to skip. This is the general seam for conditions that need run state -- token budget, post-compaction, mode switches -- without hardcoded detectors:

```python
from pydantic_ai_harness import SystemReminders

SystemReminders(
    dynamic_reminders=[
        lambda ctx: 'Wrap up soon.' if ctx.run_step > 20 else None,
    ],
)
```

Every reminder is sent as a system prompt, so the model reads it with the authority of your instructions. Static `Reminder` content and a `str` a dynamic callable returns are operator text, sent in system voice as written: return only text you wrote yourself. The built-in dynamic reminders never put text they didn't write in system voice.

`GoalReanchor` doesn't quote the user at all: it points the model back at the user's first request, which is already in the conversation. `LLMReminder` returns a `GeneratedReminder`, a distinct type rather than a `str`, so string operations can't turn the generated text into operator text; a wrapper that passes it through keeps it as data.

`SystemReminders` sends a `GeneratedReminder` as a fixed framework note in system voice, which ranks the reminder below your instructions and the user's request, and the generated text on the user channel in a `<generated_reminder model="...">` tag. The two are one record in the message history, kept and sent together on every later request and seen together by compaction and summaries, so in history kept on the server the generated text doesn't reach the model without the note that says what it is. History that round-trips through a UI client is the one exception: see [UI history](#ui-history). Both parts carry `origin` metadata, so the reminders recognize their own records.

Every opening and closing spelling of a framing tag inside the generated text, `<system>` and the reminder tags included, is escaped (`&lt;system>`), so it can't end its tag, open another or speak as the system. `LLMReminder` also quotes the transcript it sends the reminder model, in `<summary>`, `<goal>` and `<message role="...">` elements, starting at the latest native compaction boundary.

### `GoalReanchor` -- zero-cost goal anchoring

`GoalReanchor` points the model back at the run's first user request and asks it to check its next action advances it. After native compaction it points at the request as the compaction summary records it, when the request itself is no longer in what the model sees. No model call, no dependencies:

```python
from pydantic_ai_harness import SystemReminders
from pydantic_ai_harness.system_reminders import GoalReanchor

SystemReminders(dynamic_reminders=[GoalReanchor()])
```

### `LLMReminder` -- model-generated nudges

`LLMReminder` has a model summarize a compact transcript (original goal + recent activity) into a short stay-on-task nudge. It requires an explicit `model` -- there is no default model id -- and falls back to `GoalReanchor` text on any error, so a failed generation never blocks the run:

```python
from pydantic_ai_harness import SystemReminders
from pydantic_ai_harness.system_reminders import LLMReminder

SystemReminders(dynamic_reminders=[LLMReminder(model='anthropic:claude-haiku-4-5', interval=10)])
```

`LLMReminder` generates every `interval` model requests: the 10th, 20th, 30th, ... by default, the cadence `TrajectoryJudge` reviews a run on. Each firing issues one extra model call (its usage is threaded onto the parent run via `ctx.usage`, so it shows up in `result.usage`, and the generation run is filed under the parent's `conversation_id`), and adds its note and generated text to the message history for good, sent again on every later request. So `interval` bounds both: a run of R requests makes `R // interval` extra calls and gains `2 * (R // interval)` parts. A run shorter than `interval` requests gets no generated reminder, which is the point of the default: instruction fade sets in over long runs. Set `interval` lower for a tighter cadence, and add `GoalReanchor` alongside it for a free check on every request in between:

```python
from pydantic_ai_harness import SystemReminders
from pydantic_ai_harness.system_reminders import GoalReanchor, LLMReminder

SystemReminders(dynamic_reminders=[GoalReanchor(), LLMReminder(model='anthropic:claude-haiku-4-5', interval=5)])
```

`LLMReminder` defaults to `interval=10`. Earlier releases generated on every model request, and each generation adds two parts to the history for good, so an agent with an `LLMReminder` and no cadence of its own grew its history by two parts on every step. Leaving `interval` unset warns that the default changed: pass `interval=10` to keep the new cadence without the warning, or `interval=1` to generate on every request as before. A wrapper that gates `LLMReminder` itself should pass `interval=1`, so the wrapper alone decides.

The nested call runs under the parent's `usage_limits` with one request held back for the model request it precedes, so the reminder cannot push a run past its `request_limit`; once the budget is that tight the generation is skipped and `GoalReanchor` text is used instead. Every other generation failure is logged at warning level with its traceback under the `pydantic_ai_harness.system_reminders` logger namespace, once per run, so a misconfigured `model` (bad id, missing key) shows up in the logs once per run rather than on every model request, while the run carries on with the fallback text.

Under a durability engine, a `LLMReminder` listed directly in `dynamic_reminders` is a journaled
capability operation: replay restores the recorded reminder instead of repeating the model call,
and a generation error is recorded as the `GoalReanchor` fallback rather than inheriting the
engine's retry policy, so a best-effort reminder cannot stall the run. `SystemReminders` carries the
stable default `id='system_reminders'`, so durable recovery works without configuration.

Only a direct entry takes that route. Two shapes do not:

- A wrapper of your own that calls `LLMReminder` (to gate it on a condition, say) calls it from
  orchestration context, where engines that forbid I/O can fail the call outright. Called that way,
  it still applies its `interval`, counted in run steps (`ctx.run_step`).
- An `LLMReminder` subclass that overrides `__call__` runs that override directly, so it cannot be
  journaled either.

Without a durability engine, generation runs directly in all three cases, with the same fallback to
`GoalReanchor` on error.

## Configuration

Subscribe to `ReminderFiredEvent` to observe reminders after they are appended:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import SystemReminders
from pydantic_ai_harness.system_reminders import Reminder, ReminderFiredEvent

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[SystemReminders(reminders=[Reminder('...', interval=5, tag=None)])],
)

@agent.on_event(ReminderFiredEvent)
async def record(ctx, event):
    print(event.text)
```

Migration: `on_fire` remains supported but is deprecated. Move its callback body to this
subscription.

```python
from pydantic_ai_harness import SystemReminders
from pydantic_ai_harness.system_reminders import Reminder

SystemReminders(
    reminders=[Reminder('...', interval=5, tag=None)],
    dynamic_reminders=[],       # callables evaluated every request
)
```

Per-run state (the request counter and per-reminder fire counts) is isolated via `for_run`, so concurrent runs on the same agent never share fire state.

## Caching guarantee

Reminders never go into the leading system prompt or instructions, and the history only ever grows by appending, so it stays eligible for a cache hit. Plain reminders, `GoalReanchor` included, are request-scoped system prompts at the tail of the request, and the only added cost is reading each one once. An `LLMReminder` record is not request-scoped: every firing adds its note and generated text to the history for good, and every later request sends all of them again. Those accumulated records are read from cache when the prefix before them is cached, and paid for in full when it isn't, so keep `LLMReminder`'s `interval` as long as the run allows and turn on caching for your model.

`SystemReminders` doesn't place a cache breakpoint of its own. Turn on caching for your model (for example `anthropic_cache` or `anthropic_cache_messages`, `bedrock_cache_messages`, or `openrouter_cache_messages`), and the breakpoint it places lands before the reminder.

If you relied on `SystemReminders` for prompt caching, turn it on for your model. Earlier releases placed a cache breakpoint ahead of every reminder, with a `'5m'` TTL by default, so an agent got caching on Anthropic without configuring any. That breakpoint is gone, and leaving `cache_ttl` unset doesn't warn about it: an agent with no caching configured on its model now pays full price for its prefix on every request. `cache_ttl` is deprecated and has no effect; set to `'5m'` or `'1h'`, it warns that it's ignored, so remove it.

## Composition

- **`Planning`** surfaces the plan in a request-only reminder of its own; both compose in one agent. The plan reminder opens with a cache breakpoint on the content before it, and the `SystemReminders` reminder is sent after the plan reminder with no breakpoint of its own, so neither displaces the other.
- **Loop detection** (detect-and-interrupt with a nudge that stays in history) is [Detect Loops](../detect_loops/). `SystemReminders` is cadence/condition steering for the current request.

The reminder is only added when the last message in the request is a `ModelRequest` and at least one reminder fires, so a turn where nothing fires adds nothing to the request. Provider-resume turns (where the request tail is a suspended `ModelResponse` that is echoed back verbatim) are skipped and do not consume a cadence slot.

## UI history

With a [UI adapter](https://pydantic.dev/docs/ai/integrations/ui/overview/#trust-model-for-client-submitted-messages) in its default `manage_system_prompt='server'` mode, history that goes through the client doesn't keep an `LLMReminder` record intact. The dump leaves out the record's system note but keeps its generated text, and AG-UI also drops the metadata that marks the text as reminder data. When the client sends that history back, the generated reminder reads as plain user text, and compaction may keep it as a user turn. Plain reminders, `GoalReanchor` included, are request-scoped and left out of the dump altogether, so they're unaffected.

If you use `LLMReminder` with a UI adapter, keep the [message history on the server](https://pydantic.dev/docs/ai/core-concepts/message-history/) and don't rely on the client's copy. A durable fix is planned with `CapabilityPart` ([#11029](https://github.com/pydantic/pydantic-ai/issues/11029)), and a server-held history mode for UI adapters is proposed in [#11031](https://github.com/pydantic/pydantic-ai/issues/11031).

## Not spec-serializable

`SystemReminders.get_serialization_name()` returns `None`: reminders take arbitrary callables, which cannot be serialized to an agent spec.

## Further reading

- [Pydantic AI capabilities](https://ai.pydantic.dev/capabilities/)
- [Request-scoped system prompts](https://pydantic.dev/docs/ai/core-concepts/message-history/#request-scoped-system-prompts) -- the core primitive the reminders are built on
- [Anthropic prompt caching](https://docs.claude.com/en/docs/build-with-claude/prompt-caching)
