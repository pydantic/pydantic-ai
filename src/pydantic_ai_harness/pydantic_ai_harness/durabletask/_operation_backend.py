from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from contextlib import AbstractAsyncContextManager, contextmanager
from dataclasses import dataclass
from typing import Annotated, Generic, Literal, TypeVar, cast

from anyio.from_thread import BlockingPortal, start_blocking_portal
from durabletask import task
from durabletask.worker import TaskHubGrpcWorker
from pydantic import Field, JsonValue, TypeAdapter, ValidationError
from pydantic_core import PydanticSerializationError

from pydantic_ai.durable_exec import JournalOperationNamer, RegisteredOperationBackend, RoleBasedOperationConfig
from pydantic_ai.durable_exec._operation import DurableOperation, DurableOperationConfig
from pydantic_ai.durable_exec._operation_backend import BoundDurableOperation, durable_unit_scope
from pydantic_ai.exceptions import UserError

from ._bridge import current_loop

ParamsT = TypeVar('ParamsT')
WireT = TypeVar('WireT')
ResultT = TypeVar('ResultT')
__all__ = ('DurableTaskOperationBackend',)


@dataclass
class _ActivityInput:
    parameters: JsonValue
    deps_input: JsonValue


_INPUT = TypeAdapter(_ActivityInput)
_JSON = TypeAdapter[JsonValue](JsonValue)


class _SerializationFailure(UserError):
    def __init__(self, error: Exception) -> None:
        super().__init__(f'Durable Task serialization failed: {error}')


@dataclass(kw_only=True)
class _ActivityResult:
    format: Literal['pydantic-ai-durabletask-result-v1'] = 'pydantic-ai-durabletask-result-v1'
    value: JsonValue = None
    serialization_error: str | None = None
    user_error: Annotated[str | None, Field(exclude_if=lambda error: error is None)] = None


_RESULT = TypeAdapter(_ActivityResult)


def _serialization_failure(error: Exception) -> object:
    message = str(error if isinstance(error, _SerializationFailure) else _SerializationFailure(error))
    return _RESULT.dump_python(_ActivityResult(serialization_error=message), mode='json')


class _BoundOperation(BoundDurableOperation[ParamsT, WireT, ResultT], Generic[ParamsT, WireT, ResultT]):
    def __init__(
        self, operation: DurableOperation[ParamsT, WireT, ResultT], *, name: str, retry_policy: task.RetryPolicy | None
    ) -> None:
        self._operation = operation
        self._name = name
        self._retry_policy = retry_policy

    @property
    def operation(self) -> DurableOperation[ParamsT, WireT, ResultT]:
        return self._operation

    async def __call__(self, params: ParamsT, *, config: object | None = None) -> ResultT:
        loop = current_loop()
        if loop is None:
            return await self.operation.handler(params)
        if config is not None and not isinstance(config, task.RetryPolicy):
            raise UserError('Durable Task activity configuration must be a `durabletask.task.RetryPolicy`.')
        pending = loop.context.call_activity(
            self._name,
            input=_JSON.validate_python(self.operation.parameter_transport.dump(params)),
            retry_policy=config if isinstance(config, task.RetryPolicy) else self._retry_policy,
            return_type=object,
        )
        result = await loop.wait(pending)
        # Older histories contain bare codec results. New results always use the envelope,
        # including values that themselves resemble it, so user data cannot impersonate a failure.
        wire = _JSON.validate_python(result)
        if isinstance(wire, dict) and wire.get('format') == 'pydantic-ai-durabletask-result-v1':
            recorded = _RESULT.validate_python(wire)
            if recorded.serialization_error is not None:
                raise UserError(recorded.serialization_error)
            if recorded.user_error is not None:
                raise UserError(recorded.user_error)
            wire = recorded.value
        return self.operation.result_codec.load(wire)


class DurableTaskOperationBackend(RegisteredOperationBackend[task.RetryPolicy | None]):
    """Register named Durable Task activities for the base durability declarations.

    Normally constructed by `DurableTaskDurability`. Native retry policies apply to
    activities, not the agent's semantic `ModelRetry` control flow.
    """

    def __init__(
        self,
        *,
        agent_name: str,
        default_model_id: str | None = None,
        retry_policy: task.RetryPolicy | None = None,
        operation_config: DurableOperationConfig[task.RetryPolicy | None] | None = None,
        activity_scope: Callable[[JsonValue], AbstractAsyncContextManager[None]],
    ) -> None:
        self._portal: BlockingPortal | None = None
        self._activity_scope = activity_scope

        super().__init__(
            namer=JournalOperationNamer(agent_name, default_model_id=default_model_id or 'default'),
            config=operation_config
            or RoleBasedOperationConfig(
                model=retry_policy,
                event=retry_policy,
                capability=retry_policy,
                tool=retry_policy,
            ),
        )

    def register(
        self,
        operation: DurableOperation[ParamsT, WireT, ResultT],
        *,
        name: str,
        config: task.RetryPolicy | None,
    ) -> tuple[BoundDurableOperation[ParamsT, WireT, ResultT], Sequence[Callable[..., object]]]:
        async def execute(payload: object) -> object:
            try:
                wire = _INPUT.validate_python(payload)
            except ValidationError as exc:
                return _serialization_failure(exc)
            try:
                async with self._activity_scope(wire.deps_input):
                    with durable_unit_scope():
                        try:
                            # JSON erases the wire type; its transport validates it on load.
                            params = operation.parameter_transport.load(cast(WireT, payload), runtime=self)
                        except ValidationError as exc:
                            return _serialization_failure(exc)
                        result = await operation.handler(params)
                        try:
                            value = _JSON.validate_python(operation.result_codec.dump(result))
                        except (ValidationError, PydanticSerializationError, UnicodeDecodeError) as exc:
                            return _serialization_failure(exc)
                        return _RESULT.dump_python(_ActivityResult(value=value), mode='json')
            except _SerializationFailure as exc:
                # SDK RetryPolicy retries every activity exception. Raise recorded codec
                # failures in orchestration code instead, outside its retry boundary.
                return _serialization_failure(exc)
            except UserError as exc:
                return _RESULT.dump_python(_ActivityResult(user_error=str(exc)), mode='json')

        def activity(context: task.ActivityContext, payload: object) -> object:
            if self._portal is None:
                raise UserError('Enter `DurableTaskAgent.register(worker)` before starting the worker.')
            return self._portal.call(execute, payload)

        activity.__name__ = name
        return _BoundOperation(operation, name=name, retry_policy=config), [activity]

    @contextmanager
    def worker_scope(self, worker: TaskHubGrpcWorker) -> Generator[None]:
        """Own the activity event loop while its registered worker accepts and drains work.

        Enter before starting the worker and exit after stopping it. Dependency and
        toolset resources belong to activity attempts on this loop, not the orchestrator.
        """
        if self._portal is not None:
            raise UserError('This Durable Task agent is already registered with an active worker.')
        with start_blocking_portal(name='pydantic-ai-durabletask') as portal:
            self._portal = portal
            try:
                for activity in self.registrations():
                    worker.add_activity(activity)
                yield
            finally:
                self._portal = None
