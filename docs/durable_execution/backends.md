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
engine SDK accepts a callback each time a durable unit is invoked. Implement `_execute` to pass the
name, callback, cache identity, and resolved config to the SDK.

Use [`RegisteredOperationBackend`][pydantic_ai.durable_exec.RegisteredOperationBackend] when the
engine requires handlers to be registered before the worker starts. Implement `_register` to
create a bound caller and return its registration handles. The capability exposes the collected
handles to the worker.

Both tiers implement [`DurableOperationBackend`][pydantic_ai.durable_exec.DurableOperationBackend].
They own result encoding, cache projection, config resolution, and naming around the engine-specific
primitive.

## Minimal callable backend

This complete in-process example runs each operation immediately. A real integration replaces
`ImmediateBackend._execute` with its SDK's activity, step, or task call and changes
`in_durable_context` to query the engine runtime.

```python
from collections.abc import Awaitable, Callable, Mapping
from typing import ClassVar, Literal

from pydantic_ai import Agent
from pydantic_ai.durable_exec import (
    IDENTITY_CODEC,
    BaseDurabilityCapability,
    CallableOperationBackend,
    DurableOperationId,
    JournalOperationNamer,
    OperationConfigRole,
    ToolsetKind,
)
from pydantic_ai.models.test import TestModel


class ImmediateConfig:
    def base(self, role: OperationConfigRole, operation_id: DurableOperationId) -> None:
        return None

    def for_tool(
        self,
        role: OperationConfigRole,
        operation_id: DurableOperationId,
        tool: object | None,
        tool_name: str,
    ) -> None | Literal[False]:
        return None


class ImmediateBackend(CallableOperationBackend[None]):
    def __init__(self, agent_name: str) -> None:
        super().__init__(namer=JournalOperationNamer(agent_name), config=ImmediateConfig())

    async def _execute(
        self,
        *,
        name: str,
        body: Callable[[], Awaitable[object]],
        cache_key: tuple[object, ...],
        config: object,
    ) -> object:
        return await body()


class ImmediateDurability(BaseDurabilityCapability[None]):
    engine_name = 'Immediate'
    _codec: ClassVar = IDENTITY_CODEC
    _unsupported_runtime_toolset_kinds: ClassVar = frozenset()
    _wrapped_toolset_kinds: ClassVar[frozenset[ToolsetKind]] = frozenset({'function', 'mcp', 'dynamic'})
    _toolset_lifecycles: ClassVar[
        Mapping[ToolsetKind, Literal['enter-outside-durable', 'enter-always', 'enter-never']]
    ] = {
        'function': 'enter-always',
        'mcp': 'enter-always',
        'dynamic': 'enter-never',
    }
    _durable_unit_noun = 'operation'
    _durable_container_noun = 'run'

    @property
    def in_durable_context(self) -> bool:
        return True

    def _build_operation_backend(self) -> ImmediateBackend:
        return ImmediateBackend(self.name)


agent = Agent(TestModel(), name='example', capabilities=[ImmediateDurability()])
result = agent.run_sync('hello')
assert result.output == 'success (no tool calls)'
```

The declarative fields tell the shared base which toolsets to wrap, where their async context
managers open, whether discovery is durable, and whether calls must be sequential. Define every
field deliberately for a production engine. In particular, lifecycle choices must ensure toolset
resources close on success, errors, and cancellation.

## Serialization and configuration

Use [`IDENTITY_CODEC`][pydantic_ai.durable_exec.IDENTITY_CODEC] when the engine SDK owns Python
object serialization. Use [`JSON_CODEC`][pydantic_ai.durable_exec.JSON_CODEC] when the integration
writes JSON-compatible journal payloads itself. Both implement
[`DurabilityCodec`][pydantic_ai.durable_exec.DurabilityCodec]. Arguments, results, tool control-flow
signals, and decorated capability operations all cross the selected codec boundary.

The backend config object receives an
[`OperationConfigRole`][pydantic_ai.durable_exec.OperationConfigRole] and a
[`DurableOperationId`][pydantic_ai.durable_exec.DurableOperationId]. Match the concrete ID variants
when config differs by model, toolset, or operation. The union is closed, so exhaustive matching
will make a newly added operation visible to your type checker. Per-tool config can return `False`
to opt that tool out of a durable unit.

## Persisted names and recovery

Operation names are persisted compatibility data. They do not depend on the Python class name.
[`JournalOperationNamer`][pydantic_ai.durable_exec.JournalOperationNamer] supplies the standard
sequence-based naming scheme; implement
[`DurableOperationNamer`][pydantic_ai.durable_exec.DurableOperationNamer] if the engine has different
requirements. Pin every generated name in tests before refactoring agent names, model IDs, toolset
IDs, capability IDs, or operation names. A rename without a migration prevents in-flight executions
from finding their recorded work.

A durable unit can run more than once when a worker fails after its side effect but before the
checkpoint commits. Tools and decorated capability operations should therefore be idempotent unless the
engine provides a suitable at-most-once mode. Test replay and recovery, resource teardown,
control-flow exceptions, persisted-output upgrades, and ordinary execution outside the durable
runtime before publishing an integration.

See [durable capability operations](../capabilities/custom.md#durable-capability-operations) for how
capability authors contribute decorated methods that arrive through the same backend.
