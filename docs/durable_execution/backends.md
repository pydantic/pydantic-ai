# Building a durable execution backend

Pydantic AI's durable execution builder lets an integration route model requests, tool discovery,
tool validation, tool calls, event handling, message compaction, and
[decorated capability operations](../capabilities/custom.md#durable-capability-operations) through one engine
backend. Use it when you are integrating a durable execution system that is not already supported.

The complete implementations for [Temporal][pydantic_ai.durable_exec.temporal.TemporalDurability],
[DBOS][pydantic_ai.durable_exec.dbos.DBOSDurability], and
[Prefect][pydantic_ai.durable_exec.prefect.PrefectDurability] are useful references. The external
[Restate](https://github.com/restatedev/sdk-python/tree/main/packages/integrations/pydantic-ai),
[AWS Lambda](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/aws_lambda),
and [Absurd](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/absurd)
integrations show the same public builder with JSON journals.

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
four model operations before a worker can start, so these registrations are present without first
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


class ImmediateBackend(JournalCallableOperationBackend[None]):
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


class ImmediateDurability(BaseDurabilityCapability[None]):
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
toolsets, and per-tool configuration key. The default lifecycle policy enters function and MCP
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

The built-in IDs are `ModelRequestId`, `ModelCompactMessagesId`,
`ModelCancelSuspendedResponseId`, `EventStreamHandlerId`, `ToolsetGetToolsId`,
`ToolsetGetInstructionsId`, `ToolsetValidateToolArgumentsId`, `ToolsetCallToolId`, and
`CapabilityOperationId`. Their Python class names do not determine persisted operation names.

### API evolution

[`DurableOperationId`][pydantic_ai.durable_exec.DurableOperationId] grows in minor releases as
Pydantic AI adds durable units. Sandbox operations are one planned example. Engine configuration
must therefore include a default branch when matching IDs. Use that branch to apply a safe base
configuration or raise an actionable unsupported-operation error. Do not rely on an exhaustive
match that assumes the current union will never gain another arm.

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

See [durable capability operations](../capabilities/custom.md#durable-capability-operations) for how
capability authors contribute decorated methods that arrive through the same backend.
