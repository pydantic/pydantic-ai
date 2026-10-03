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

`SystemReminders` injects reminders on each model request, either statically (`Reminder`, on a cadence) or dynamically (a callable that reads the run context). Each firing adds a [turn-scoped system prompt](https://pydantic.dev/docs/ai/core-concepts/message-history/#turn-scoped-system-prompts) at the **tail** of the request: the model sees it for that request only, and it's recorded in the message history like any other part.

- On Anthropic models that take mid-conversation system messages, it's sent as a turn-scoped system message (`clear_at`): every copy stays on the wire, the API stops rendering each one once the next request arrives, and a cleared one costs no tokens. Leaving earlier reminders in place is what keeps the cached prefix and the model's earlier reasoning valid; deleting one would invalidate every thinking block after it on models with preserved thinking.
- Everywhere else, it's sent with its own request only, at the end, and left out of later ones.
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
            reminders=[Reminder('Stay focused on the original request.', interval=5)],
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
| `tag` | Wrap the content in `<tag>\ncontent\n</tag>`. Defaults to `'system-reminder'`; set `None` for raw content. |

The default `tag='system-reminder'` wraps every reminder in `<system-reminder>...</system-reminder>`, following Claude Code's convention so the model reads it as an out-of-band steering note rather than user text.

The `tag` wrapping applies only to static `Reminder` content. Dynamic callables (including `GoalReanchor` and `LLMReminder`) inject their returned text raw and own their own formatting.

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

### `GoalReanchor` -- zero-cost goal anchoring

`GoalReanchor` re-states the run's first user request as the anchor and asks the model to check its next action advances it. No model call, no dependencies:

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

SystemReminders(dynamic_reminders=[LLMReminder(model='anthropic:claude-haiku-4-5')])
```

Dynamic reminders have no cadence of their own -- they run on every model request. `LLMReminder` therefore issues one extra model call per turn (its usage is threaded onto the parent run via `ctx.usage`, so it shows up in `result.usage()`, and the generation run is filed under the parent's `conversation_id`). The nested call also runs under the parent's `usage_limits` with one request held back for the model request it precedes, so the reminder cannot push a run past its `request_limit`; once the budget is that tight the generation is skipped and `GoalReanchor` text is used instead. Every other generation failure is logged at warning level with its traceback under the `pydantic_ai_harness.system_reminders` logger namespace, once per run, so a misconfigured `model` (bad id, missing key) shows up in the logs once per run rather than on every model request, while the run carries on with the fallback text. To bound the cost, gate it behind a cadence with an async wrapper:

```python
_llm = LLMReminder(model='anthropic:claude-haiku-4-5')

async def every_tenth(ctx):
    return await _llm(ctx) if ctx.run_step % 10 == 0 else None

SystemReminders(dynamic_reminders=[every_tenth])
```

Under a durability engine, a `LLMReminder` listed directly in `dynamic_reminders` is a journaled
capability operation: replay restores the recorded reminder instead of repeating the model call,
and a generation error is recorded as the `GoalReanchor` fallback rather than inheriting the
engine's retry policy, so a best-effort reminder cannot stall the run. `SystemReminders` carries the
stable default `id='system_reminders'`, so durable recovery works without configuration.

Only a direct entry takes that route. Two shapes do not:

- A wrapper like `every_tenth` above calls `LLMReminder` from orchestration context, where engines
  that forbid I/O can fail the call outright.
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
    capabilities=[SystemReminders(reminders=[Reminder('...', interval=5)])],
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
    reminders=[Reminder('...', interval=5)],
    dynamic_reminders=[],       # callables evaluated every request
)
```

Per-run state (the request counter and per-reminder fire counts) is isolated via `for_run`, so concurrent runs on the same agent never share fire state.

## Caching guarantee

Reminders never go into the leading system prompt or instructions. They're turn-scoped system prompts at the tail of the request, so across turns the history grows append-only and stays eligible for a cache hit, and the only added cost is reading each reminder once.

`SystemReminders` doesn't place a cache breakpoint of its own. Turn on caching for your model (for example `anthropic_cache` or `anthropic_cache_messages`, `bedrock_cache_messages`, or `openrouter_cache_messages`), and the breakpoint it places lands before the reminder. `cache_ttl` is deprecated and has no effect.

## Composition

- **`Planning`** surfaces the plan in a request-only reminder of its own; both compose in one agent. Planning anchors its cache breakpoint on the last user content in the request, which `SystemReminders` doesn't add, so neither displaces the other.
- **Loop detection** (detect-and-interrupt with a durable nudge) is a separate concern. `SystemReminders` is cadence/condition steering for the current request; a dynamic reminder can read loop state from your deps if you want to steer on it.

The reminder is only added when the last message in the request is a `ModelRequest` and at least one reminder fires, so a turn where nothing fires adds nothing to the request. Provider-resume turns (where the request tail is a suspended `ModelResponse` that is echoed back verbatim) are skipped and do not consume a cadence slot.

## Not spec-serializable

`SystemReminders.get_serialization_name()` returns `None`: reminders take arbitrary callables, which cannot be serialized to an agent spec.

## Further reading

- [Pydantic AI capabilities](https://ai.pydantic.dev/capabilities/)
- [Turn-scoped system prompts](https://pydantic.dev/docs/ai/core-concepts/message-history/#turn-scoped-system-prompts) -- the core primitive the reminders are built on
- [Anthropic prompt caching](https://docs.claude.com/en/docs/build-with-claude/prompt-caching)
