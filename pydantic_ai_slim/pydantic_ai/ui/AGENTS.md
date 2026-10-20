## Backwards compatibility in UI adapters (specially AG-UI)

Since [3971](https://github.com/pydantic/pydantic-ai/pull/3971#discussion_r3011028336) we decided to introduce the policy of sticking to the lower (existing) version requirement. In short, this means:
- version requirement bumps are disallowed
- new functionality should be gated behind version checks (including imports)
- older versions don't error out when they encounter new functionality, but instead skip it

Gate a test for version-gated behavior on the installed version (`detect_ag_ui_version()` in `ag_ui/_utils.py`) on the test itself, never by importing a newer-only name in the module-level `imports_successful()` block of `tests/test_ag_ui.py`: a name that only exists above the floor in that import block skips the entire module on CI's `test-lowest-versions` job, which is the only job that exercises the floor these gates exist for.

## Adapter properties are shared concepts the adapter itself consumes

An unread field on a protocol's run input is not a gap. `run_input` is public, so every field is already reachable as `adapter.run_input.<field>`; a property that only forwards one adds no capability and takes on a permanent public-API commitment. AG-UI's `context`, `forwardedProps` and `parentRunId` are deliberately left that way — see [7106](https://github.com/pydantic/pydantic-ai/pull/7106#discussion_r3723844005), which closed [7105](https://github.com/pydantic/pydantic-ai/issues/7105) by documenting the wiring instead of exposing `AGUIAdapter.context`.

A field earns an adapter property when **both** hold:
- the adapter consumes it, feeding it into run args or the event stream — that's what `messages`, `toolset`, `state`, `conversation_id` and `deferred_tool_results` all do
- it names a concept every UI protocol has, so it can live on `UIAdapter` with one normalized type

One without the other is the trap: a protocol-specific property with a generic name means the day a second protocol grows the same concept, the base-class version can't be added without breaking the first adapter's return type. Normalizing early to dodge that is not the fix either — a shape derived from a single protocol is a guess, and a lossy one when it discards structure the protocol chose (AG-UI's `context` is a `list` of `description`/`value` pairs, and `description` is not unique, so a `dict` silently drops entries).

The agent run is not a sink for the leftovers, either: `RunContext.metadata` is attached to the run span, so routing client-submitted text there by default would put unbounded untrusted content into every user's traces.

## AG-UI response identities follow real `Agent` turn transitions

Each model response gets a fresh AG-UI parent message ID. A regular
`FunctionToolCallEvent` moves `UIEventStream` from the response turn to the
request turn; its result is emitted in that request turn, and the next
response's `PartStartEvent` invokes `before_response()` to replace the parent
ID. This is the lifecycle established by [#3325](https://github.com/pydantic/pydantic-ai/pull/3325).

Native tool returns differ because another native call can follow inside the
same model response. That path uses a one-off result ID without replacing the
response parent; regular tool results intentionally retain the request-turn ID
mutation. The distinction was retained explicitly in
[#6659](https://github.com/pydantic/pydantic-ai/pull/6659#discussion_r3632483479).

Regression tests for cross-response identity must run through the supported
`Agent` and `AGUIAdapter` boundary. A synthetic stream that includes a regular
tool result must include its preceding `FunctionToolCallEvent`; otherwise it
does not exercise a production-reachable turn sequence. Assert relationships
between emitted string IDs, not between matcher objects.
