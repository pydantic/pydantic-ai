---
description: "Integrate a new durable execution engine with Pydantic AI using the stable backend builder, routing model requests, tool calls and events through your engine."
---

# Building a durable execution backend

Pydantic AI's durable execution builder lets an integration route model requests, tool discovery,
tool validation, tool calls, event handling, message compaction, token counting, and
[decorated capability operations](../capabilities/custom.md#durable-capability-operations) through one engine
backend. Use it when you are integrating a durable execution system that is not already supported.

The complete implementations for [Temporal][pydantic_ai.durable_exec.temporal.TemporalDurability],
[DBOS][pydantic_ai.durable_exec.dbos.DBOSDurability], and
[Prefect][pydantic_ai.durable_exec.prefect.PrefectDurability] are useful references. The external
[Restate](https://github.com/restatedev/sdk-python/tree/main/python/restate/ext/pydantic),
[AWS Lambda](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/aws_lambda),
[Absurd](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/absurd),
and [pgtask](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/pgtask)
integrations show the same public builder with JSON journals, and
[Render Workflows](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/render)
shows an engine whose durable units run in another process.

## Choose a backend tier

Subclass [`CallableOperationBackend`][pydantic_ai.durable_exec.CallableOperationBackend] when the
engine SDK accepts a callback each time a durable unit is invoked. Implement `execute` to pass the
typed operation identity, name, callback, cache identity, and resolved config to the SDK. The
operation identity lets an engine apply behavior to an operation kind without depending on its
persisted name.

Use [`RegisteredOperationBackend`][pydantic_ai.durable_exec.RegisteredOperationBackend] when the
engine requires handlers to be registered before the worker starts. Implement `register` to
create a bound caller and return its registration handles. Its `registrations()` method returns
the collected SDK registration handles in binding order. During agent assembly, the base binds the
five model operations before a worker can start, so these registrations are present without first
running a model request. Pass the complete `registrations()` result to the engine SDK when creating
the worker.

Both tiers implement [`DurableOperationBackend`][pydantic_ai.durable_exec.DurableOperationBackend].
They own result encoding, cache projection, config resolution, and naming around the engine-specific
primitive.

## Minimal callable backend

This complete in-process example runs each operation immediately. A real integration replaces
`ImmediateBackend.execute` with its SDK's activity, step, or task call and changes
`in_durable_context` to query the engine runtime.

```python
from collections.abc import Awaitable, Callable

from pydantic_ai import Agent
from pydantic_ai.durable_exec import (
    JSON_CODEC,
    BaseDurabilityCapability,
    DurabilityEngineSpec,
    DurableOperationId,
    JournalCallableOperationBackend,
    RoleBasedOperationConfig,
)
from pydantic_ai.models.test import TestModel


class TerminalError(Exception):
    pass


class ImmediateBackend(JournalCallableOperationBackend):
    def __init__(self, agent_name: str, default_model_id: str | None) -> None:
        super().__init__(
            agent_name=agent_name,
            default_model_id=default_model_id,
            config=RoleBasedOperationConfig(model=None, event=None, capability=None, tool=None),
        )

    async def execute(
        self,
        *,
        operation_id: DurableOperationId,
        name: str,
        body: Callable[[], Awaitable[object]],
        cache_key: tuple[object, ...],
        config: None,
    ) -> object:
        return await body()


class ImmediateDurability(BaseDurabilityCapability):
    engine_spec = DurabilityEngineSpec(
        engine_name='Immediate',
        durable_unit_noun='operation',
        durable_container_noun='run',
        codec=JSON_CODEC,
        serialization_failure=lambda exc: TerminalError(str(exc)),
    )

    @property
    def in_durable_context(self) -> bool:
        return True

    def get_durable_operation_backend(self) -> ImmediateBackend:
        return ImmediateBackend(self.name, self.default_model_id)


agent = Agent(TestModel(), name='example', capabilities=[ImmediateDurability()])
result = agent.run_sync('hello')
assert result.output == 'success (no tool calls)'
```

The public [`DurabilityEngineSpec`][pydantic_ai.durable_exec.DurabilityEngineSpec] groups the
declarative engine surface in one immutable object. Its required fields name the engine, durable
unit, and durable container. Its optional fields select the codec, wrapped toolset kinds, lifecycle
policy, upgrade compatibility, durable discovery, sequential tool execution, unsupported runtime
toolsets, per-tool configuration key, and whether a run inside the container may take a
[`CancellationToken`][pydantic_ai.CancellationToken] (only safe on a container loop that rejects
delivery from other threads; see the field's docs). The default lifecycle policy enters function and MCP
toolsets for every run and never enters dynamic toolsets.

Define every field whose engine behavior differs from the defaults. The spec validates non-empty
nouns and requires a lifecycle for every wrapped toolset kind when it is constructed, so an invalid
engine declaration fails while its class is being defined. Lifecycle choices must ensure toolset
resources close on success, errors, and cancellation.

## Serialization and configuration

Use [`IDENTITY_CODEC`][pydantic_ai.durable_exec.IDENTITY_CODEC] when the engine SDK owns Python
object serialization. Use [`JSON_CODEC`][pydantic_ai.durable_exec.JSON_CODEC] when the integration
writes JSON-compatible journal payloads itself. Both implement
[`DurabilityCodec`][pydantic_ai.durable_exec.DurabilityCodec]. Arguments, results, tool control-flow
signals, and decorated capability operations all cross the selected codec boundary.

JSON-journal engines should set `DurabilityEngineSpec.serialization_failure` to convert deterministic
codec failures into the engine's terminal or non-retryable exception type. Such values cannot become
serializable on retry.

[`RoleBasedOperationConfig`][pydantic_ai.durable_exec.RoleBasedOperationConfig] supplies one config
per operation role and accepts an optional `resolve_tool` callback for per-tool overrides. The
callback receives the complete typed operation ID, tool object, and tool name.

[`JournalCallableOperationBackend`][pydantic_ai.durable_exec.JournalCallableOperationBackend]
combines the callable backend with `JournalOperationNamer`. Pass the bound capability's
`default_model_id` so the agent's default string model keeps the standard unsuffixed persisted name.

The backend config object implements the backend configuration protocol. Its public `base` and
`for_tool` methods receive an
[`OperationConfigRole`][pydantic_ai.durable_exec.OperationConfigRole] and a
[`DurableOperationId`][pydantic_ai.durable_exec.DurableOperationId]. Match the concrete ID variants
when config differs by model, toolset, or operation. The role is a coarse config bucket: `'model'`,
`'event'`, `'tool'`, or `'capability'`. The operation ID carries the fine-grained identity. A
capability operation ID includes the explicit name from `@durable_operation(name='...')`. That name
is required because it becomes persisted compatibility data and must remain stable if the Python
method is renamed. The ID union represents the IDs available in the installed Pydantic AI version.
Per-tool config can return
`False` to opt a function or dynamic tool out of a durable unit.
MCP tools perform I/O and always run in their durable unit, so returning `False` for one raises a
[`UserError`][pydantic_ai.exceptions.UserError].

The built-in IDs are `ModelRequestId`, `ModelCompactMessagesId`, `ModelCountTokensId`,
`ModelCancelSuspendedResponseId`, `EventStreamHandlerId`, `ToolsetGetToolsId`,
`ToolsetGetInstructionsId`, `ToolsetValidateToolArgumentsId`, `ToolsetCallToolId`, and
`CapabilityOperationId`. Their Python class names do not determine persisted operation names.

### API evolution

[`DurableOperationId`][pydantic_ai.durable_exec.DurableOperationId] grows in minor releases as
Pydantic AI adds durable units. Sandbox operations are one planned example. Engine configuration
must therefore include a default branch when matching IDs. Use that branch to apply a safe base
configuration or raise an actionable unsupported-operation error. Do not rely on an exhaustive
match that assumes the current union will never gain another arm.

## Durable units in another process

Some engines run a durable unit on a different worker than the run that dispatched it, so its
parameters have to cross the boundary as data. A backend's
[`parameter_transports`][pydantic_ai.durable_exec.DurableOperationBackend.parameter_transports]
decides how. The default [`ParameterTransports`][pydantic_ai.durable_exec.ParameterTransports]
passes every operation's parameters through unchanged. An engine whose units run out of process
subclasses it and passes an instance to `RegisteredOperationBackend.__init__`. Each method returns a
[`ParameterTransport`][pydantic_ai.durable_exec.ParameterTransport] that dumps one of the public
parameter types to the engine's wire format at dispatch and loads it back worker-side:
[`ModelRequestParams`][pydantic_ai.durable_exec.ModelRequestParams],
[`ModelCompactMessagesParams`][pydantic_ai.durable_exec.ModelCompactMessagesParams],
[`ModelCancelSuspendedResponseParams`][pydantic_ai.durable_exec.ModelCancelSuspendedResponseParams],
[`EventStreamHandlerParams`][pydantic_ai.durable_exec.EventStreamHandlerParams],
[`ToolsetGetToolsParams`][pydantic_ai.durable_exec.ToolsetGetToolsParams],
[`ToolsetCallToolParams`][pydantic_ai.durable_exec.ToolsetCallToolParams],
[`DynamicToolsetCallToolParams`][pydantic_ai.durable_exec.DynamicToolsetCallToolParams],
[`CapabilityOperationParams`][pydantic_ai.durable_exec.CapabilityOperationParams], and
[`WorkspaceCallParams`][pydantic_ai.durable_exec.WorkspaceCallParams].
The token count ahead of a request takes the request's `ModelRequestParams` and the transport
`model_request(result_type)` returns, so an engine that carries model requests carries counts too.
The backend's `register` receives each [`DurableOperation`][pydantic_ai.durable_exec.DurableOperation]
and returns a [`BoundDurableOperation`][pydantic_ai.durable_exec.BoundDurableOperation]. Run its
worker-side handler inside [`durable_unit_scope`][pydantic_ai.durable_exec.durable_unit_scope], so
a durable operation called from inside the unit runs directly instead of scheduling another unit.

With the default dynamic toolset transports, a unit gets the run's own `RunContext`, so a
[`DynamicToolset`][pydantic_ai.toolsets.DynamicToolset] built with `per_run_step=False` is resolved
once per run and held for the units, as a non-durable run holds it. An engine whose
[`dynamic_get_tools`][pydantic_ai.durable_exec.ParameterTransports.dynamic_get_tools] or
[`dynamic_call`][pydantic_ai.durable_exec.ParameterTransports.dynamic_call] returns a transport of its
own resolves the toolset again in each unit.

Every parameter type carries the run's `RunContext` (as `run_context`, or `ctx` on the toolset
parameters; it is `None` when a suspended response is cancelled outside a run), and that context
holds objects that cannot be serialized. Use [`DurableRunContext`][pydantic_ai.durable_exec.DurableRunContext] for it:
`DurableRunContext.serialize_run_context(ctx)` returns a mapping of the fields that can cross,
which the engine encodes with its own codec, and `DurableRunContext.restore(...)` rebuilds the
context worker-side and re-attaches the worker's agent and workspace. The run's validation context
is rebuilt from the agent's `validation_context` setting the first time it's read.

Function tool arguments arrive as JSON-decoded values, not the Python values the tool's schema
produces. Override `ParameterTransports.prepare_function_call` to return
[`revalidate_function_call`][pydantic_ai.durable_exec.revalidate_function_call]'s result, which
looks the tool up again when it wasn't transported and validates the arguments worker-side.
Reading a field that wasn't carried raises a `UserError` instead of returning a default that could
be mistaken for run state. Override `unavailable_field_message` to word that error for your engine.
Your users can subclass your context and override `serialize_run_context` to carry more fields.
Temporal's [`TemporalRunContext`][pydantic_ai.durable_exec.temporal.TemporalRunContext] is one such subclass.

The other building blocks an out-of-process engine works with:

- [`DurableOperationConfig`][pydantic_ai.durable_exec.DurableOperationConfig] is the protocol
  `RoleBasedOperationConfig` implements, for engines that resolve config their own way.
- [`ResultCodec`][pydantic_ai.durable_exec.ResultCodec] and
  [`CacheIdentity`][pydantic_ai.durable_exec.CacheIdentity] are the `result_codec` and
  `cache_identity` of a `DurableOperation`: how its result crosses the boundary, and which inputs a
  hash-keyed engine uses as its cache key.
- [`CallToolResult`][pydantic_ai.durable_exec.CallToolResult],
  [`DynamicToolsResult`][pydantic_ai.durable_exec.DynamicToolsResult] and
  [`WorkspaceCallResult`][pydantic_ai.durable_exec.WorkspaceCallResult] are the result types of the
  tool call, dynamic toolset discovery and workspace operations. A capability operation's result
  type is [`capability_operation_result_type`][pydantic_ai.durable_exec.capability_operation_result_type]
  applied to its [`CapabilityMethodDeclaration`][pydantic_ai.durable_exec.CapabilityMethodDeclaration]'s
  `result_type`. Pass these to your codec when an engine decodes results itself.
- [`ModelRequestContextProjection`][pydantic_ai.durable_exec.ModelRequestContextProjection] is the
  serializable form of a model request context, which carries a live model that cannot cross the
  boundary.

## Persisted names and recovery

Operation names are persisted compatibility data. They do not depend on the Python class name.
[`JournalOperationNamer`][pydantic_ai.durable_exec.JournalOperationNamer] supplies the standard
sequence-based naming scheme; implement
[`DurableOperationNamer`][pydantic_ai.durable_exec.DurableOperationNamer] if the engine has different
requirements. Pin every generated name in tests before refactoring agent names, model IDs, toolset
IDs, capability IDs, or operation names. A rename without a migration prevents in-flight executions
from finding their recorded work.

The namer receives the typed operation ID and an optional keyword-only `label`. Operations that
need an invocation-specific suffix provide a typed `invocation_label` callable when they are
declared. Built-in tool-call and argument-validation operations use the tool name. Namers should
not inspect an operation's parameter object.

A durable unit can run more than once when a worker fails after its side effect but before the
checkpoint commits. Tools and decorated capability operations should therefore be idempotent unless the
engine provides a suitable at-most-once mode. Test replay and recovery, resource teardown,
control-flow exceptions, persisted-output upgrades, and ordinary execution outside the durable
runtime before publishing an integration.

A replayed or recovered run must also get the same identity. Inside the durable context, override
[`get_default_run_id()`][pydantic_ai.capabilities.AbstractCapability.get_default_run_id] to return a
replay-stable ID, such as one derived from the workflow instance ID, and return
[`conversation_id_from_run_id(run_id)`][pydantic_ai.durable_exec.conversation_id_from_run_id] from
[`get_default_conversation_id()`][pydantic_ai.capabilities.AbstractCapability.get_default_conversation_id],
as the Temporal and DBOS capabilities do. Each agent run in the same workflow needs its own
`run_id`, associated with the same logical run even if concurrent calls reorder on retry. See
[Prefect run and conversation IDs](prefect.md#run-and-conversation-ids) for its cached-ID policy.

See [durable capability operations](../capabilities/custom.md#durable-capability-operations) for how
capability authors contribute decorated methods that arrive through the same backend.

## Waiting for external events

[`wait_for_external_event`][pydantic_ai.external_events.wait_for_external_event] waits until
something outside the run resolves a token the run handed out, unless the `ask` function it hands
the token to answers right away, and returns the JSON payload. A
[`HandleDeferredToolCalls`](../capabilities/handle-deferred-tool-calls.md) handler that takes a resume
token uses it to ask for an answer on deferred tool calls without ending the run. Outside a durable run it
waits in process, and [`resolve_external_event`][pydantic_ai.external_events.resolve_external_event]
ends the wait. In a durable run, an in-process wait would not be recorded: a replay would ask again,
could get a different answer than the one the run already acted on and diverge from the recorded
history, and a restart would lose the wait. So in the durable context,
`BaseDurabilityCapability.wrap_run` installs a waiter that refuses with a `UserError`, unless your
engine installed its own first. Overriding `wrap_run`, call `super().wrap_run(ctx, handler=handler)`
rather than `handler()` to keep that guard.

To implement it, subclass [`ExternalEventWaiter`][pydantic_ai.external_events.ExternalEventWaiter] and
install an instance for each durable agent run with
[`external_event_scope`][pydantic_ai.external_events.external_event_scope] in your capability's
`wrap_run` when it is in the durable context, around the call to `super().wrap_run(...)`. Both are re-exported from
`pydantic_ai.durable_exec` with [`ExternalEventError`][pydantic_ai.external_events.ExternalEventError],
and live in `pydantic_ai.external_events`. The `wait` method receives a stable `name`, an async
`ask` callback, and an optional `timeout`, and must journal three things so a replay returns the
recorded payload without side effects:

1. Create a resume token: a callback ID, an awakeable ID, or a waitpoint token, depending on the engine.
2. Call `ask(token)` once, inside a durable unit that records its result. It returns the payload
   when it answered right away, and the wait returns that payload without suspending. It returns
   `None` when it handed the token to whoever resolves it.
3. Otherwise suspend until the token is resolved, then return its JSON payload. Raise
   `ExternalEventError` when the timeout elapses or the token is resolved with a failure.

```python {title="external_event_waiter.py"}
from collections.abc import Awaitable, Callable
from datetime import timedelta

from pydantic import JsonValue

from pydantic_ai.durable_exec import ExternalEventWaiter, external_event_scope
from pydantic_ai.external_events import wait_for_external_event


class ApprovingWaiter(ExternalEventWaiter):
    async def wait(
        self,
        name: str,
        *,
        ask: Callable[[str], Awaitable[JsonValue]],
        timeout: timedelta | None,
    ) -> JsonValue:
        token = f'{name}-1'  # (1)!
        if (answer := await ask(token)) is not None:  # (2)!
            return answer
        return {'approved': True}  # (3)!


async def main():
    with external_event_scope(ApprovingWaiter()):  # (4)!
        payload = await wait_for_external_event('review', ask=lambda token: None)
    print(payload)
    #> {'approved': True}
```

1. A real engine creates a resume token through its API and records it.
2. Run inside a durable unit the engine records, so a replay does not ask again and reuses an answer `ask` gave.
3. A real engine suspends the run here until the token is resolved, then returns the recorded payload.
4. An engine enters the scope for each durable agent run, such as in its capability's `wrap_run`.

_(To run this example, ensure `asyncio` is imported and add `asyncio.run(main())`; no other changes are needed.)_

Engines that hand out tokens map onto this directly: AWS Lambda callbacks, Restate awakeables,
Trigger.dev waitpoint tokens, Vercel Workflow hooks. Engines that address external events by name
instead (Temporal signals, Azure Durable Task external events, DBOS messages) can encode the
execution's ID and an event key in the token, and have whoever resolves it send the named event. Two
implementations are references: the AWS Lambda one in Pydantic AI Harness's
[`aws_lambda` package](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/aws_lambda)
hands out Lambda callback IDs, and the
[Temporal one](https://github.com/pydantic/pydantic-ai/blob/main/pydantic_ai_slim/pydantic_ai/durable_exec/temporal/_external_events.py)
encodes the workflow ID and an event key, calls `ask` in a local activity, and waits for a signal that
[`resolve_external_event`][pydantic_ai.durable_exec.temporal.resolve_external_event] sends.

Code inside a durable unit, such as a tool, can't wait: the wait has to be recorded in the run
itself. `wait_for_external_event` rejects a call made inside
[`durable_unit_scope`][pydantic_ai.durable_exec.durable_unit_scope] before it reaches your waiter, so
run each unit's body in that scope.

## Waiting a provider's `Retry-After`

A model error can say how long to wait before the request is attempted again, in its
[`retry_after`][pydantic_ai.exceptions.ModelAPIError.retry_after]. When your engine retries a model
operation that failed, wait that long instead of its configured backoff, as a durable wait the engine
schedules rather than a sleep inside the durable unit.
[`model_error_retry_after(error, maximum=...)`][pydantic_ai.durable_exec.model_error_retry_after] finds
the model error in the chain of the exception the unit failed with and caps the wait at `maximum`,
typically the engine's maximum retry interval. It returns `None` when no model error asked for a wait,
so the configured backoff applies. Temporal sets the result as the failed activity's
`next_retry_delay`. An engine with no way to set the delay of one retry, or no durable timer to wait
it with, should say in its documentation that it doesn't wait `retry_after`, rather than sleep in the
process that runs the unit.

## Replays and telemetry

A replayed run executes the agent loop again, so its instrumentation runs again too. Tell it which
work came from the journal, so replays emit no duplicate spans and no duplicate token or cost
metrics: wrap each durable agent run in [`replay_scope()`][pydantic_ai.durable_exec.replay_scope].

An engine that knows, per durable unit, whether its body ran reports each one with
[`record_durable_unit(executed=...)`][pydantic_ai.durable_exec.record_durable_unit], in the context of
the task that requested it, as AWS Lambda does. Every span instrumentation opens counts the units that
completed under it, so concurrent tool calls are told apart: a span whose units were all replayed is
not reported, and neither are a replayed model request's metrics. A span that is reported keeps the
spans around it, so live work never loses its parent.

An engine that knows instead whether it is replaying at a given moment passes
`replay_scope(is_replaying=...)`, as Temporal does with `workflow.unsafe.is_replaying_history_events()`:
a span or model request that ends while it returns `True` is not reported.

## Checking an engine

Subclass [`DurableEngineSuite`][pydantic_ai.durable_exec.conformance.DurableEngineSuite] in your pytest suite
and provide its `engine` fixture. Each test runs one cell under your engine: a capability's tool calls, or a
scenario such as an event emitted from a tool. It runs the agent, recovers the run from what your engine recorded,
and checks the recovered run repeats no completed work and observes the same events and messages. The built-in
engines run the same suite, and the [compatibility tables](compatibility.md) show their results.

```python {test="skip" lint="skip"}
import pytest
from pydantic_ai.durable_exec.conformance import DurableEngineSuite, EngineRun, Support


class MyEngine:
    name = 'my-engine'

    def durability(self) -> MyDurability:
        return MyDurability()

    async def run(self, agent, prompt: str) -> EngineRun:
        ...  # run `agent` as a workflow of your engine, in process, and read back its record

    async def recover(self, run: EngineRun, *, after: int | None = None) -> EngineRun:
        ...  # recover `run` from its record, keeping only its first `after` units when given


class TestMyEngine(DurableEngineSuite):
    support = {
        'RunContext.emit_from_tool': Support(
            'refused', 'tools run in a unit, which refuses an emitted event', match='not supported inside'
        ),
    }

    @pytest.fixture(scope='class')
    @classmethod
    def engine(cls) -> MyEngine:
        return MyEngine()
```

- The `engine` fixture returns a [`DurableEngine`][pydantic_ai.durable_exec.conformance.DurableEngine]:
    - `durability()` returns a new instance of your durability capability, which the suite puts last on each agent.
    - `run(agent, prompt)` runs the agent the way your engine runs a workflow, and returns an
      [`EngineRun`][pydantic_ai.durable_exec.conformance.EngineRun] with the output, or the error the run raised,
      and the units your engine recorded, in order. Unwrap a refusal your engine wraps back into the
      [`UserError`][pydantic_ai.exceptions.UserError] it was.
    - `recover(run, after=None)` recovers the run from its record, as your engine does after a crash. With
      `after=k`, it keeps only the first `k` units, as if the run had stopped there.
- Start the engine once per class, with a class-scoped fixture, and use no real waits: inject retry delays and poll
  intervals. The suite needs the anyio pytest plugin.
- Declare each cell that isn't supported in `support`, keyed by the case's capability class name or the
  scenario's name, as a [`Support`][pydantic_ai.durable_exec.conformance.Support]:
    - `'limited'`: the cell works with a limit, which its `check` asserts on the run, its `reason` documents, and
      its `issue` tracks.
    - `'refused'`: your engine raises a `UserError` matching `match` before the refused operation records anything.
    - `'broken'`: your engine runs it and gets it wrong; `match` names the symptom, and `issue` the bug tracking it.
- A cell you don't list must be supported. Set `unlisted = 'pending'` to report a failing unlisted cell as an
  expected failure instead, so a case a new release adds doesn't fail your CI until you declare its support.
- Set `recovery` to a [`Recovery('restart', ...)`][pydantic_ai.durable_exec.conformance.Recovery] if your engine retries a run by running it again from the start, rather
  than replaying its record.
- Return `False` from the `can_fork_mid_run` fixture if your engine can't recover a run from part of its record,
  and document that limit for your users.
- The suite counts the requests a recovery must not make again: a capability's own model requests, and, under
  `HarnessDurableEngineSuite`, the writes and reads of the stores the Harness cases use. They're counted in the test's
  process. A case you add whose stand-ins make requests of their own overrides `count_requests` to count them too,
  which keeps the suite's own counts.
- Override `expected_units` and `model_request_unit` to pin the names your engine records, which a run recorded by
  an earlier release recovers by.

[`HarnessDurableEngineSuite`][pydantic_ai_harness.conformance.HarnessDurableEngineSuite],
from `pydantic_ai_harness.conformance`, is the same suite with the [Pydantic AI Harness](../harness/index.md)
capabilities' cases added. Subclass it instead to check your engine against both.

A case whose capability needs an optional dependency that isn't installed is skipped, naming the missing module, so
run pytest with `-rs` to see what went unchecked. Set the `PYDANTIC_AI_REQUIRE_CONFORMANCE_EXTRAS` environment
variable to `1` (or `true`) to fail those cases instead, in a CI job that installs every extra; `0`, `false` or unset
keeps them skipped, and any other value is an error. A module missing from `pydantic_ai` or `pydantic_ai_harness`
themselves always fails. The cases that need one are:

| Case | Needs |
|---|---|
| `MCP` | the `fastmcp` package, whose server `pydantic-ai-slim[mcp]` doesn't include |
| `CodeMode` | `pydantic-ai-harness[code-mode]` |
| `Coder` | `pydantic-ai-harness[coder]` |
| `IPythonCodeMode` | `pydantic-ai-harness[ipython]` |
| `PromptInjectionDefender` | `pydantic-ai-harness[prompt-injection-defender]` |
| `RepairToolArguments` | `pydantic-ai-harness[repair-tool-arguments]`, or `[coder]` |
| `Skills` | `pydantic-ai-harness[skills]` |

The `Macroscope` case runs a stand-in `#!/bin/sh` script, so it needs a POSIX platform.

Only a module that isn't installed skips a case. Any other `ImportError`, such as a name a mismatched version lacks,
or one your engine's own code raises, fails it.

The cases are the [`Calls`][pydantic_ai.capabilities.conformance.Calls] in
[`pydantic_ai.capabilities.conformance.CALLS`][pydantic_ai.capabilities.conformance.CALLS] and
[`pydantic_ai_harness.conformance.CALLS`][pydantic_ai_harness.conformance.CALLS], and the scenarios are
[`SCENARIO_RUNS`][pydantic_ai.capabilities.conformance.SCENARIO_RUNS]. To also check a capability of your own, add
its calls to `cases`:

```python {test="skip" lint="skip"}
from pydantic_ai.capabilities.conformance import Calls
from pydantic_ai_harness.conformance import HarnessDurableEngineSuite


class TestMyEngine(HarnessDurableEngineSuite):
    cases = {
        **HarnessDurableEngineSuite.cases,
        'MyCapability': Calls((('my_tool', {'query': 'q'}),), build=lambda directory: [MyCapability()]),
    }
```

The suite's class, fixtures, `DurableEngine`, `EngineRun`, `Support`, `Recovery`, the level names, `Calls` and
the case and scenario names are stable in 2.x. Minor releases can add cases, scenarios and checks, or tighten them,
so pin `pydantic-ai` or set `unlisted = 'pending'` if a new cell failing your CI is a problem.
