# System Reminders

> [!NOTE]
> The `Reminder` helper is not re-exported at the top level -- import it from the submodule:
>
> ```python
> from pydantic_ai_harness import SystemReminders
> from pydantic_ai_harness.system_reminders import Reminder
> ```
>
> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

Re-inject behavioral guidance mid-run to counter instruction fade -- without invalidating the prompt cache.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/system_reminders/)

## The problem

Long multi-turn runs suffer instruction fade: after many tool-use turns the model progressively ignores the guidance it was given at the start. A single start-of-session system prompt is not enough for extended work. The fix is to re-state targeted guidance mid-run -- on a fixed cadence, or reactively when a condition is detected.

## The solution

`SystemReminders` injects reminders on each model request, either statically (`Reminder`, on a cadence) or dynamically (a callable that reads the run context). Reminders are appended to the **tail** of the request as an ephemeral `UserPromptPart` behind a `CachePoint`:

- The injection happens in `wrap_model_request`, which runs *after* the durable history is persisted, so the reminder reaches the model but is never written to `message_history`. No reminders accumulate across turns.
- A `CachePoint` is placed immediately *before* the reminder, so the cached prefix (tools + system + real conversation) stays byte-identical turn over turn. Only the small reminder falls outside the cache.

Injecting into the system prompt (or any persisted part) instead would sit at the front of the request, so every reminder would bust the cached prefix and stale reminders would pile up in history. This capability avoids both.

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

Dynamic reminders have no cadence of their own -- they run on every model request. `LLMReminder` therefore issues one extra model call per turn (its usage is threaded onto the parent run via `ctx.usage`, so it shows up in `result.usage()`). The nested call also runs under the parent's `usage_limits` with one request held back for the model request it precedes, so the reminder cannot push a run past its `request_limit`; once the budget is that tight the generation is skipped and `GoalReanchor` text is used instead. Because the fallback is silent, a persistently misconfigured `model` (bad id, missing key) looks like normal operation. To bound the cost, gate it behind a cadence with an async wrapper:

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

```python
from pydantic_ai_harness import SystemReminders
from pydantic_ai_harness.system_reminders import Reminder

SystemReminders(
    reminders=[Reminder('...', interval=5)],
    dynamic_reminders=[],       # callables evaluated every request
    cache_ttl='5m',             # TTL for the cache breakpoint before the reminder ('5m' | '1h')
    on_fire=None,               # optional callback invoked with each rendered reminder
)
```

Per-run state (the request counter and per-reminder fire counts) is isolated via `for_run`, so concurrent runs on the same agent never share fire state.

## Caching guarantee

Reminders are never injected into the system prompt or instructions. They ride the ephemeral tail behind a `CachePoint`, so across turns the durable history grows append-only and stays eligible for a cache hit (subject to the provider's cache TTL -- a gap longer than `cache_ttl` expires the entry even under an unchanged prefix), while the reminder and its `CachePoint` live only in the per-request copy. The only added cost is re-reading the reminder each turn.

`CachePoint` is supported on Anthropic, Amazon Bedrock (Converse API), and OpenRouter (Anthropic and Gemini models); on providers without prompt caching it is ignored (nothing to bust). The reminder leads with its `CachePoint` only when the request already carries user content for the breakpoint to attach to -- on a turn whose only tail content is the reminder (for example an `instructions`-only run's first request), the reminder is injected without a breakpoint, since there is no prefix to protect.

## Composition

- **`Planning`** uses the same ephemeral-tail mechanism to surface the plan. Both compose in one agent: each appends its own tail part behind its own `CachePoint`, and neither is persisted. Each ephemeral-tail capability adds a cache breakpoint; Anthropic allows 4 (3 with automatic caching) and core trims the excess oldest-first, so stacking several tail-injecting capabilities alongside `anthropic_cache_instructions` / `anthropic_cache_tool_definitions` can evict an older breakpoint. Two capabilities plus the defaults stay within budget.
- **Loop detection** (detect-and-interrupt with a durable nudge) is a separate concern. `SystemReminders` is cadence/condition steering that stays ephemeral; a dynamic reminder can read loop state from your deps if you want to steer on it.

The tail reminder is only appended when the last message in the request is a `ModelRequest` and at least one reminder fires, so a turn where nothing fires adds nothing to the request. Provider-resume turns (where the request tail is a suspended `ModelResponse` that is echoed back verbatim) are skipped and do not consume a cadence slot.

## Not spec-serializable

`SystemReminders.get_serialization_name()` returns `None`: reminders take arbitrary callables, which cannot be serialized to an agent spec.

## Further reading

- [Pydantic AI capabilities](https://ai.pydantic.dev/capabilities/)
- [Hooks](https://ai.pydantic.dev/hooks/) -- `wrap_model_request` is the ephemeral injection point used here
- [Anthropic prompt caching](https://docs.claude.com/en/docs/build-with-claude/prompt-caching)
