## Backwards compatibility in UI adapters (specially AG-UI)

Since [3971](https://github.com/pydantic/pydantic-ai/pull/3971#discussion_r3011028336) we decided to introduce the policy of sticking to the lower (existing) version requirement. In short, this means:
- version requirement bumps are disallowed
- new functionality should be gated behind version checks (including imports)
- older versions don't error out when they encounter new functionality, but instead skip it

The inbound half of that last rule lives in `ag_ui/_forward_compat.py`: AG-UI's `Message` and `InputContent` are discriminated unions, so a `role` or `type` added after the installed version is a hard validation failure for the whole request unless it's skipped first. Skipping is scoped to exactly that — an unknown tag only counts as new functionality when the item also satisfies the contract every member of its union shares (for `Message`, a string `id`), so anything else malformed must still fail.

Tests for version-gated behavior belong behind `requires_ag_ui('<version>')` in `tests/test_ag_ui.py`, never behind the module-level `imports_successful()` gate: a name that only exists above the floor in that import block skips the entire module on CI's `test-lowest-versions` job, which is the only job that exercises the floor these gates exist for.

## Adapter properties are shared concepts the adapter itself consumes

An unread field on a protocol's run input is not a gap. `run_input` is public, so every field is already reachable as `adapter.run_input.<field>`; a property that only forwards one adds no capability and takes on a permanent public-API commitment. AG-UI's `context`, `forwardedProps` and `parentRunId` are deliberately left that way — see [7106](https://github.com/pydantic/pydantic-ai/pull/7106#discussion_r3723844005), which closed [7105](https://github.com/pydantic/pydantic-ai/issues/7105) by documenting the wiring instead of exposing `AGUIAdapter.context`.

A field earns an adapter property when **both** hold:
- the adapter consumes it, feeding it into run args or the event stream — that's what `messages`, `toolset`, `state`, `conversation_id` and `deferred_tool_results` all do
- it names a concept every UI protocol has, so it can live on `UIAdapter` with one normalized type

One without the other is the trap: a protocol-specific property with a generic name means the day a second protocol grows the same concept, the base-class version can't be added without breaking the first adapter's return type. Normalizing early to dodge that is not the fix either — a shape derived from a single protocol is a guess, and a lossy one when it discards structure the protocol chose (AG-UI's `context` is a `list` of `description`/`value` pairs, and `description` is not unique, so a `dict` silently drops entries).

The agent run is not a sink for the leftovers, either: `RunContext.metadata` is attached to the run span, so routing client-submitted text there by default would put unbounded untrusted content into every user's traces.

The generic-name trap has a second form, on the event stream rather than the adapter. `run_id` and `thread_id` already mean something here: `UIAdapter.run_stream()` takes the *agent run's* `run_id`, and `UIAdapter.conversation_id` is the agent's conversation. `AGUIEventStream.thread_id` / `.run_id` are AG-UI's own, read off `RunAgentInput` and emitted on `RunStartedEvent` / `RunFinishedEvent`, and the two are never fed into each other. Keeping the protocol-side pair on the protocol's own class is what leaves the base free to claim the name later; say in each docstring which of the two it is, so a reader doesn't have to guess.

Run identity Pydantic AI *mints* uses `uuid7()` from `_uuid.py`, matching `GraphAgentState.run_id` / `conversation_id`. `uuid4()` is for message and part IDs (`UIEventStream.message_id`), not run identity.
