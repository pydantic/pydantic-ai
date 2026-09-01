from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Literal

from pydantic_ai.durable_exec import (
    JSON_CODEC,
    BaseDurabilityCapability,
    CallableOperationBackend,
    DurabilityEngineSpec,
    DurableOperationId,
    JournalOperationNamer,
    OperationConfigRole,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import RunContext


class RestrictedRunContext(RunContext[None]):
    """Run context stand-in that rejects fields excluded from a worker payload."""

    unavailable_fields: frozenset[str] = frozenset()

    def __getattribute__(self, name: str) -> object:
        if name in object.__getattribute__(self, 'unavailable_fields'):
            raise UserError(f'{name!r} is not available in this durable operation')
        return super().__getattribute__(name)


class _RecordingConfig:
    def base(self, role: OperationConfigRole, *, operation_id: DurableOperationId) -> None:
        return None

    def for_tool(
        self,
        role: OperationConfigRole,
        *,
        operation_id: DurableOperationId,
        tool: object | None,
        tool_name: str,
    ) -> None | Literal[False]:
        return None  # pragma: no cover - protocol stub; these tests do not wrap tools


class _RecordingBackend(CallableOperationBackend[None]):
    def __init__(
        self,
        agent_name: str,
        calls: list[tuple[str, tuple[object, ...]]],
        fail_operations: frozenset[str],
    ) -> None:
        super().__init__(namer=JournalOperationNamer(agent_name), config=_RecordingConfig())
        self.calls = calls
        self.fail_operations = fail_operations

    async def execute(
        self,
        *,
        operation_id: DurableOperationId,
        name: str,
        body: Callable[[], Awaitable[object]],
        cache_key: tuple[object, ...],
        config: None,
    ) -> object:
        self.calls.append((name, cache_key))
        if name in self.fail_operations:
            raise RuntimeError(f'{name} failed')
        return await body()


class RecordingDurability(BaseDurabilityCapability[object]):
    engine_spec = DurabilityEngineSpec(
        engine_name='recording',
        durable_unit_noun='unit',
        durable_container_noun='journal',
        codec=JSON_CODEC,
        wrapped_toolset_kinds=frozenset(),
    )

    def __init__(self, *, fail_operations: frozenset[str] = frozenset()) -> None:
        super().__init__()
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self.fail_operations = fail_operations

    @property
    def in_durable_context(self) -> bool:
        return True

    def get_durable_operation_backend(self) -> _RecordingBackend:
        return _RecordingBackend(self.name, self.calls, self.fail_operations)
