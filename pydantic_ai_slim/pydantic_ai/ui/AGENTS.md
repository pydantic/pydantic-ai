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

## AG-UI response identities follow real `Agent` turn transitions

Each model response gets a fresh AG-UI parent message ID. A regular
`FunctionToolCallEvent` moves `UIEventStream` from the response turn to the
request turn; its result is emitted in that request turn, and the next
response's `PartStartEvent` invokes `before_response()` to replace the parent
ID. This is the lifecycle established by [#3325](https://github.com/pydantic/pydantic-ai/pull/3325).

Every parent ID the stream emits is announced before it is referenced.
`_handle_tool_call_start` starts a content-free message when the response has no
text before its first tool call, so a consumer never has to synthesize the
assistant message that ID names — a synthesized one carries an ID the server
never emitted, which is what breaks telling new messages apart from replayed
history ([#7527](https://github.com/pydantic/pydantic-ai/issues/7527)). Close
that pair on the spot: the AG-UI *client's* event verifier (`verifyEvents` in the
TypeScript `@ag-ui/client`, not the Python package) rejects `RUN_FINISHED` while a
text message is still open.

That content-free pair is not the shape [#2754](https://github.com/pydantic/pydantic-ai/pull/2754)
removed. Those `THINKING_TEXT_MESSAGE_*` envelopes carried no ID anyone needed —
thinking already has an outer `THINKING_START`/`THINKING_END` envelope, so the
inner empty message signalled state and nothing else. An assistant message has no
outer envelope, so `TEXT_MESSAGE_START` is the only event that can name its ID,
which is the thing a consumer needs. Emit an empty envelope only when it carries
something; here it carries the identity.

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

## The event stream is an encoder, so it owns what it emits

`UIEventStream.run_input` is optional because a stream is constructible with no request behind it: transports that carry native events out of band — a durable execution workflow, a queue, a websocket fan-out — encode at an API edge the adapter never reaches ([6970](https://github.com/pydantic/pydantic-ai/issues/6970)).

So a value a subclass emits is a field of that subclass, overwritten from `run_input` in `__post_init__` when one is given — the request's identity wins over an explicitly passed value, it doesn't merely default it, and passing both warns rather than discarding one in silence — that's `AGUIEventStream.thread_id` / `run_id`, which the protocol requires on `RUN_STARTED` and `RUN_FINISHED`. Telling an explicitly passed value apart from a generated default is what `_GeneratedID` is for: comparing against the run input's value can't, since a generated ID differs from it too. Reading `self.run_input.<field>` at emit time is what makes a stream un-constructible without a request, and it's the reason a `run_input` stub had to be fabricated before.

This is the opposite case from the adapter-property rule above, not an exception to it: the field exists because the stream *emits* it, not to forward a protocol object's contents to the caller. An unread `run_input` field still earns nothing.
