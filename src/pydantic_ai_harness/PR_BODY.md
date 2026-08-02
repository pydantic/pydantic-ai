## Summary

Adds the `SystemReminders` capability: it re-injects behavioral guidance mid-run -- on a
cadence or reactively from a condition -- to counter instruction fade in long sessions,
without invalidating the prompt cache.

The capability lands as a self-contained submodule `pydantic_ai_harness/system_reminders/`
(not re-exported from the root package), with tests under `tests/system_reminders/`, a
`README.md` next to the code, and a `docs/system-reminders.md` page.

Public API (`from pydantic_ai_harness.system_reminders import ...`):

- `SystemReminders` -- the capability. Fields: `reminders`, `dynamic_reminders`, `cache_ttl`,
  `on_fire`.
- `Reminder` -- static reminder with `content`, `interval`, `first_after`, `trigger`,
  `max_fires`, `tag`.
- `DynamicReminder` / `AsyncDynamicReminder` / `ReminderGenerator` -- callable type aliases.
- `GoalReanchor` -- opt-in, dependency-free dynamic reminder that re-states the first user
  request as the anchor.
- `LLMReminder` -- opt-in dynamic reminder that has a model generate a stay-on-task nudge;
  `model` is required (no default id), fails soft to `GoalReanchor` text.

## Linked Issue

Closes #83

## Supersedes #181

This rebuilds @DouweM's DRAFT PR #181 (`Add SystemReminders capability`). #181 is left open
and untouched; this PR carries its design forward and should supersede it. Credit to
@DouweM for the model, the `SystemReminders` name, and the audit items (`trigger`,
`max_fires`, XML `tag`) that #181 already implemented.

### What changed vs #181, and why

- **Injection point (the reason for the rebuild).** #181 appended a `SystemPromptPart` to the
  last `ModelRequest` inside `before_model_request`, which runs before core persists the
  durable history. That mutates a message in the cached prefix (busting the prompt cache on
  every fire) and risks reminders entering `message_history` and accumulating. This PR injects
  in `wrap_model_request` as an ephemeral tail `UserPromptPart` behind a `CachePoint`,
  mirroring the verified `Planning` pattern: it runs after core persists the durable history,
  and the per-request message list it mutates is never written back, so reminders reach the
  model but never enter `message_history` and the cached prefix stays byte-identical.
- **Default `tag='system-reminder'` (behavior change).** #181 defaulted `tag=None` (raw
  content). This PR defaults `tag='system-reminder'`, so reminder content is wrapped in
  `<system-reminder>...</system-reminder>` by default (Claude Code's convention). Set
  `tag=None` for raw content. This is an explicit behavior change vs #181.
- **Name kept: `SystemReminders`.** deepagents' independent implementation named the analogous
  capability `PeriodicReminder` (cadence-only). `SystemReminders` covers cadence + condition +
  dynamic reminders, so DouweM's broader name is kept for continuity and scope accuracy.
- **Package relocation.** #181's flat `src/pydantic_harness/system_reminders.py` moves to the
  current submodule layout `pydantic_ai_harness/system_reminders/`.

## Visible decisions

- **Decision: inject via `wrap_model_request` ephemeral tail + `CachePoint`, not
  `before_model_request` + persisted `SystemPromptPart`** -- reason: cache-safety. Injecting
  into the cached prefix busts the cache every fire and lets stale reminders accumulate in
  history; the tail-behind-`CachePoint` pattern (same as `Planning`) reaches the model without
  entering `message_history` or invalidating the prefix. This is the load-bearing change.
- **Decision: default `tag='system-reminder'`** -- reason: Claude Code convention and harness
  norm; the model reads it as an out-of-band steering note. Behavior change vs #181's
  `tag=None`.
- **Decision: fold deepagents extras in as opt-in dynamic-reminder callables (`GoalReanchor`,
  `LLMReminder`), not core defaults** -- reason: keep `SystemReminders` core dependency-free
  and deterministic. `LLMReminder` requires an explicit `model` (no hardcoded/default model
  id) and fails soft to `GoalReanchor` text on any error.
- **Decision: keep the `SystemReminders` name over deepagents' `PeriodicReminder`** -- reason:
  it covers cadence + condition + dynamic, which the narrower name would misdescribe.
- **Decision: public reminder callables take `RunContext[Any]` (deps-agnostic)** -- reason:
  avoids forcing users to parametrize reminders by deps type; matches `step_persistence`
  precedent.

## Carry-forward ledger (#181 / deepagents -> disposition)

| Item | Disposition | Note |
|---|---|---|
| `Reminder(content, interval)` static cadence | Carried | Core mechanism. |
| Dynamic `(RunContext) -> str \| None` (sync + async) | Carried | General seam over the trigger table. |
| `for_run()` per-run counter reset | Carried | `replace(self)` resets `_request_count`/`_fire_counts`. |
| `get_serialization_name() -> None` | Carried | Callables aren't spec-serializable. |
| `trigger: Callable[[RunContext], bool]` | Carried | AND-composed with cadence. |
| `max_fires: int \| None` | Carried | Per-reminder cap. |
| XML `tag` wrapping | Carried + strengthened | Default `tag='system-reminder'`. |
| Injection via `before_model_request` + `SystemPromptPart` | Changed | Moved to cache-safe `wrap_model_request` tail. |
| `_inject_into_last_request` create-request-if-none branch | Dropped | `wrap_model_request` always has a tail `ModelRequest`. |
| Flat `src/pydantic_harness/` module | Changed | Relocated to `pydantic_ai_harness/system_reminders/`. |
| deepagents goal re-anchoring | Carried as `GoalReanchor` | Zero-cost, opt-in dynamic reminder. |
| deepagents LLM generator | Carried as `LLMReminder` | Opt-in; `model` required; error-fallback to `GoalReanchor`. |
| deepagents `first_after` cadence | Carried | `None` preserves #181 modulo. |
| deepagents `render_style` enum | Skipped (net-neutral) | `tag` XML + raw covers it. |
| deepagents `make_config_for_mode` factory | Skipped (net-neutral) | Sugar over the constructor. |
| deepagents `on_reminder` callback | Carried as `on_fire` | Cheap observability hook. |
| deepagents transcript self-filtering | Dropped | Ephemeral tail never persists, so no self-pollution. |
| #83 priority / cooldown / template substitution | Deferred | Cooldown ~= `interval`; the rest add surface without a demonstrated need. |
| #83 built-in trigger detectors | Deferred / composed | Loop detection is a separate capability; the rest are user-written dynamic callables. |

## Tests

`tests/system_reminders/test_system_reminders.py`, `TestModel`/`FunctionModel` only, no
private-helper imports. Covers static intervals and the multi-reminder cadence matrix,
`first_after`, sync + async dynamic reminders, `trigger`, `max_fires` (including per-reminder
independence), the default XML `tag` wrap, `for_run` isolation across concurrent copies,
`on_fire`, `GoalReanchor`, `LLMReminder` (transcript build, truncation, agent caching, blank
output, error fallback), and composition alongside `Planning`.

The key invariant test (`test_reminder_reaches_model_but_not_persisted`) asserts a fired
reminder reaches the model behind a `CachePoint` but never enters `result.all_messages()` --
the cache-safety guarantee.

The Pydantic bars are met: full suite passes and branch coverage is 100% (module and repo-wide).

## Checklist

- [x] Linked issue exists and is referenced above
- [x] Tests added/updated for new behavior
- [x] `make lint && make typecheck && make test` passes locally (don't stress about CI -- we'll help)
- [x] No changes to `pyproject.toml` or `uv.lock` (dependency changes require a separate issue)
- [x] Docstrings use single backticks (not RST double backticks)
