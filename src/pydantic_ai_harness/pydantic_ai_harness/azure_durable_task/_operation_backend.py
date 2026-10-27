from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Generator, Sequence
from contextlib import AbstractAsyncContextManager, contextmanager
from dataclasses import dataclass
from typing import Annotated, Generic, Literal, TypeVar, cast

from anyio.from_thread import BlockingPortal, start_blocking_portal
from durabletask import task
from durabletask.worker import TaskHubGrpcWorker
from pydantic import Field, JsonValue, TypeAdapter, ValidationError
from pydantic_core import PydanticSerializationError

from pydantic_ai.durable_exec import (
    BoundDurableOperation,
    DurableOperation,
    DurableOperationConfig,
    DurableOperationId,
    JournalOperationNamer,
    ParameterTransports,
    RegisteredOperationBackend,
    ToolsetGetInstructionsId,
    ToolsetGetToolsId,
    durable_unit_scope,
)
from pydantic_ai.exceptions import UserError

from ._bridge import current_loop

ParamsT = TypeVar('ParamsT')
WireT = TypeVar('WireT')
ResultT = TypeVar('ResultT')
__all__ = ('AzureDurableTaskOperationBackend',)


@dataclass
class ActivityInput:
    parameters: JsonValue
    deps_input: JsonValue


INPUT = TypeAdapter(ActivityInput)
_JSON = TypeAdapter[JsonValue](JsonValue)
_RESULT_FORMAT: Literal['pydantic-ai-durabletask-result-v1'] = 'pydantic-ai-durabletask-result-v1'
_ESCAPED_NAME_CHARACTER = re.compile(r'-|[^a-zA-Z0-9_.`+:-]')
_ACTIVITY_NAME = re.compile(r'[a-zA-Z0-9_.`+:-]{1,100}')
_NAME_RULE = '1-100 ASCII characters: letters, digits, underscores, periods, backticks, plus signs, hyphens or colons'
_DEFAULT_CONFIG = object()


class _AzureDurableTaskNamer(JournalOperationNamer):
    def operation_name(self, operation_id: DurableOperationId) -> str:
        original = super().operation_name(operation_id)
        name = _ESCAPED_NAME_CHARACTER.sub(lambda match: '--' if match[0] == '-' else f'-x{ord(match[0]):X}-', original)
        if len(name) > 100:
            name = f'{name[:83]}-{hashlib.sha256(original.encode()).hexdigest()[:16]}'
        return name


class SerializationFailure(UserError):
    def __init__(self, error: Exception) -> None:
        super().__init__(f'Durable Task serialization failed: {error}')


@dataclass(kw_only=True)
class _ActivityResult:
    format: Literal['pydantic-ai-durabletask-result-v1']
    value: JsonValue = None
    serialization_error: str | None = None
    user_error: Annotated[str | None, Field(exclude_if=lambda error: error is None)] = None


_RESULT = TypeAdapter(_ActivityResult)


def _serialization_failure(error: Exception) -> object:
    message = str(error if isinstance(error, SerializationFailure) else SerializationFailure(error))
    return _RESULT.dump_python(_ActivityResult(format=_RESULT_FORMAT, serialization_error=message), mode='json')


class _BoundOperation(BoundDurableOperation[ParamsT, WireT, ResultT], Generic[ParamsT, WireT, ResultT]):
    def __init__(
        self, operation: DurableOperation[ParamsT, WireT, ResultT], *, name: str, options: task.RetryPolicy | None
    ) -> None:
        self._operation = operation
        self._name = name
        self._options = options

    @property
    def operation(self) -> DurableOperation[ParamsT, WireT, ResultT]:
        return self._operation

    async def __call__(self, params: ParamsT, *, config: object | None = _DEFAULT_CONFIG) -> ResultT:
        loop = current_loop()
        if loop is None:
            return await self.operation.handler(params)
        if loop.abandoned:
            # Fail fast, so best-effort cleanup like withdrawing a pending model response gives up instead of waiting.
            raise UserError('Agent cleanup started durable work after its orchestration was abandoned.')
        # Discovery builders pass None for inherited configuration, not a per-tool override.
        if config is _DEFAULT_CONFIG or (
            config is None and isinstance(self.operation.operation_id, (ToolsetGetToolsId, ToolsetGetInstructionsId))
        ):
            config = self._options
        if config is not None and not isinstance(config, task.RetryPolicy):
            raise UserError('Durable Task activity configuration must be a `durabletask.task.RetryPolicy`.')
        pending = loop.context.call_activity(
            self._name,
            input=_JSON.validate_python(self.operation.parameter_transport.dump(params)),
            retry_policy=config,
            return_type=object,
        )
        result = await loop.wait(pending)
        recorded = _RESULT.validate_python(result)
        if recorded.serialization_error is not None:
            raise UserError(recorded.serialization_error)
        if recorded.user_error is not None:
            raise UserError(recorded.user_error)
        return self.operation.result_codec.load(recorded.value)


class AzureDurableTaskOperationBackend(RegisteredOperationBackend[task.RetryPolicy | None]):
    """Register named Durable Task activities for the base durability declarations.

    Normally constructed by `AzureDurableTaskDurability`. Native retry policies apply to
    activities, not the agent's semantic `ModelRetry` control flow.
    """

    def __init__(
        self,
        *,
        agent_name: str,
        default_model_id: str | None = None,
        config: DurableOperationConfig[task.RetryPolicy | None],
        activity_scope: Callable[[JsonValue], AbstractAsyncContextManager[None]],
        parameter_transports: ParameterTransports | None = None,
    ) -> None:
        self._portal: BlockingPortal | None = None
        self._activity_scope = activity_scope
        self._activity_names: set[str] = set()

        super().__init__(
            namer=_AzureDurableTaskNamer(agent_name, default_model_id=default_model_id or 'default'),
            config=config,
            parameter_transports=parameter_transports,
        )

    def register(
        self,
        operation: DurableOperation[ParamsT, WireT, ResultT],
        *,
        name: str,
        config: task.RetryPolicy | None,
    ) -> tuple[BoundDurableOperation[ParamsT, WireT, ResultT], Sequence[Callable[..., object]]]:
        if _ACTIVITY_NAME.fullmatch(name) is None:
            raise UserError(f'Azure Durable Task activity name {name!r} must use {_NAME_RULE}.')
        if name in self._activity_names:
            raise UserError(
                f'Azure Durable Task activity name {name!r} is not unique after escaping. '
                f'Choose distinct agent, model and toolset identities; names must use {_NAME_RULE}.'
            )
        self._activity_names.add(name)

        async def execute(payload: object) -> object:
            try:
                wire = INPUT.validate_python(payload)
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
                        return _RESULT.dump_python(_ActivityResult(format=_RESULT_FORMAT, value=value), mode='json')
            except SerializationFailure as exc:
                # SDK RetryPolicy retries every activity exception. Raise recorded codec
                # failures in orchestration code instead, outside its retry boundary.
                return _serialization_failure(exc)
            except UserError as exc:
                return _RESULT.dump_python(_ActivityResult(format=_RESULT_FORMAT, user_error=str(exc)), mode='json')

        def activity(context: task.ActivityContext, payload: object) -> object:
            if self._portal is None:
                raise UserError('Enter `AzureDurableTaskDurability.register(worker)` before starting the worker.')
            return self._portal.call(execute, payload)

        activity.__name__ = name
        return _BoundOperation(operation, name=name, options=config), [activity]

    @contextmanager
    def worker_scope(self, worker: TaskHubGrpcWorker) -> Generator[None]:
        """Own the activity event loop while its registered worker accepts and drains work.

        Enter before starting the worker and exit after stopping it. Dependency and
        toolset resources belong to activity attempts on this loop, not the orchestrator.
        """
        if self._portal is not None:
            raise UserError('These Azure Durable Task activities are already registered with an active worker.')
        with start_blocking_portal(name='pydantic-ai-durabletask') as portal:
            self._portal = portal
            try:
                for activity in self.registrations():
                    worker.add_activity(activity)
                yield
            finally:
                self._portal = None
