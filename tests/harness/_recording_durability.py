from __future__ import annotations

import asyncio
import contextvars
from collections.abc import Awaitable, Callable, Mapping
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
        results: list[tuple[str, object]],
        isolated: bool,
        journal: Mapping[str, object],
        failure: type[Exception] = RuntimeError,
        wraps_errors: bool = False,
    ) -> None:
        super().__init__(namer=JournalOperationNamer(agent_name), config=_RecordingConfig())
        self.isolated = isolated
        self.calls = calls
        self.fail_operations = fail_operations
        self.failure = failure
        self.wraps_errors = wraps_errors
        self.results = results
        self.journal = journal

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
        if name in self.journal:
            # As on replay, the journaled result comes back at once, without running the operation.
            return self.journal[name]
        if name in self.fail_operations:
            raise self.failure(f'{name} failed')
        try:
            if self.isolated:
                # As on a worker, the operation doesn't see the caller's context variables.
                async def run() -> object:
                    return await body()

                result = await asyncio.create_task(run(), context=contextvars.Context())
            else:
                result = await body()
        except Exception as e:  # pragma: no cover - reached only by an operation that lets an error escape
            if not self.wraps_errors:
                raise
            # As a real engine reports an operation that raised: as its own error, losing the original's type.
            raise RuntimeError(f'{name} failed') from e
        self.results.append((name, result))
        return result


class RecordingDurability(BaseDurabilityCapability[object]):
    engine_spec = DurabilityEngineSpec(
        engine_name='recording',
        durable_unit_noun='unit',
        durable_container_noun='journal',
        codec=JSON_CODEC,
        wrapped_toolset_kinds=frozenset(),
    )

    def __init__(
        self,
        *,
        fail_operations: frozenset[str] = frozenset(),
        isolated: bool = False,
        journal: Mapping[str, object] | None = None,
        failure: type[Exception] = RuntimeError,
        wraps_errors: bool = False,
    ) -> None:
        super().__init__()
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self.results: list[tuple[str, object]] = []
        """Each completed operation's result, as an engine would journal it."""
        self.fail_operations = fail_operations
        self.isolated = isolated
        self.journal: Mapping[str, object] = journal or {}
        """Results by operation name, replayed instead of running the operation."""
        self.failure = failure
        """What the engine raises for an operation in `fail_operations`."""
        self.wraps_errors = wraps_errors
        """Whether an error an operation raises reaches the caller as the engine's own `RuntimeError`, as on a real
        engine, rather than as itself."""

    @property
    def in_durable_context(self) -> bool:
        return True

    def get_durable_operation_backend(self) -> _RecordingBackend:
        return _RecordingBackend(
            self.name,
            self.calls,
            self.fail_operations,
            self.results,
            self.isolated,
            self.journal,
            self.failure,
            self.wraps_errors,
        )
