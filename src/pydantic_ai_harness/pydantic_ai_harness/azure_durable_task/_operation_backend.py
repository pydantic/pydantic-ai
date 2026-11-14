from __future__ import annotations

import asyncio
import hashlib
import math
import re
from collections.abc import Callable, Generator, Sequence
from contextlib import AbstractAsyncContextManager, contextmanager
from dataclasses import dataclass, replace
from datetime import timedelta
from typing import Generic, Literal, TypeVar

from anyio.from_thread import BlockingPortal, start_blocking_portal
from durabletask import task
from durabletask.grpc_options import GrpcChannelOptions
from durabletask.internal import helpers as _helpers, orchestrator_service_pb2 as _pb
from durabletask.worker import TaskHubGrpcWorker
from google.protobuf.json_format import MessageToDict, ParseDict
from pydantic import JsonValue, TypeAdapter, ValidationError
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
    model_error_retry_after,
)
from pydantic_ai.exceptions import UserError

from .._retry_after import MINIMUM_RETRY_AFTER_CAP, MODEL_OPERATION_IDS
from ._bridge import current_loop
from ._transports import INPUT, JsonTransport

ParamsT = TypeVar('ParamsT')
WireT = TypeVar('WireT')
ResultT = TypeVar('ResultT')
__all__ = ('AzureDurableTaskOperationBackend',)


_JSON = TypeAdapter[JsonValue](JsonValue)
_JSON_OBJECT = TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue])
_RESULT_FORMAT: Literal['pydantic-ai-durabletask-result-v1'] = 'pydantic-ai-durabletask-result-v1'
_ESCAPED_NAME_CHARACTER = re.compile(r'-|[^a-zA-Z0-9_.`+:-]')
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
    error: str | None = None
    retry_after: float | None = None
    """The wait a model error asked for before the model activity is attempted again.

    See `_BoundOperation._call_waiting_retry_after`.
    """
    failure: dict[str, JsonValue] | None = None
    """The `TaskFailureDetails` the model error would have failed the activity with, as JSON."""


_RESULT = TypeAdapter(_ActivityResult)


def _failure(error: UserError) -> object:
    return _RESULT.dump_python(_ActivityResult(format=_RESULT_FORMAT, error=str(error)), mode='json')


def _retry_after_result(error: Exception) -> object | None:
    """The result a model activity records instead of failing with `error`, when a model error asked to wait.

    The SDK's retry policy can't wait longer or shorter than its own backoff for one failure, so the activity completes
    with the wait instead, and the orchestration waits it on a durable timer before attempting the activity again; see
    `_BoundOperation._call_waiting_retry_after`. The failure details travel with it, so an orchestration that doesn't
    attempt the activity again fails the way the activity would have.
    """
    retry_after = model_error_retry_after(error, maximum=math.inf)
    if retry_after is None:
        return None
    # The SDK builds an activity's failure details with this internal helper; there is no public equivalent.
    failure = _JSON_OBJECT.validate_python(MessageToDict(_helpers.new_failure_details(error)))
    return _RESULT.dump_python(
        _ActivityResult(format=_RESULT_FORMAT, retry_after=retry_after, failure=failure), mode='json'
    )


def _retry_after_cap(policy: task.RetryPolicy, attempts: int) -> float:
    """The longest wait before the attempt after `attempts`: the policy's `max_retry_interval`, if it sets one.

    Without one, the policy's own backoff for the attempt or a minute, whichever is longer.
    """
    if policy.max_retry_interval is not None:
        return policy.max_retry_interval.total_seconds()
    backoff = policy.first_retry_interval.total_seconds() * (policy.backoff_coefficient or 1.0) ** (attempts - 1)
    return max(backoff, MINIMUM_RETRY_AFTER_CAP)


def _continued_policy(policy: task.RetryPolicy, attempts: int, elapsed: timedelta) -> task.RetryPolicy:
    """The retry policy for the attempts `policy` has left after `attempts`, with the backoff where it left off."""
    # In seconds, as the SDK works out its backoff, so a long one is capped before it can overflow a `timedelta`.
    first_retry_interval = policy.first_retry_interval.total_seconds() * (policy.backoff_coefficient or 1.0) ** attempts
    if policy.max_retry_interval is not None:
        first_retry_interval = min(first_retry_interval, policy.max_retry_interval.total_seconds())
    retry_timeout = policy.retry_timeout
    return task.RetryPolicy(
        first_retry_interval=timedelta(seconds=first_retry_interval),
        max_number_of_attempts=policy.max_number_of_attempts - attempts,
        backoff_coefficient=policy.backoff_coefficient,
        max_retry_interval=policy.max_retry_interval,
        retry_timeout=max(retry_timeout - elapsed, timedelta()) if retry_timeout is not None else None,
    )


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
        if loop is None:  # pragma: no cover - core only calls bound operations inside a durable context
            return await self.operation.handler(params)
        if loop.abandoned:
            # Fail fast, so best-effort cleanup like withdrawing a pending model response gives up instead of waiting.
            raise UserError('Agent cleanup started durable work after its orchestration was abandoned.')
        # Discovery builders pass None for inherited configuration, not a per-tool override.
        if config is _DEFAULT_CONFIG or (
            config is None and isinstance(self.operation.operation_id, (ToolsetGetToolsId, ToolsetGetInstructionsId))
        ):
            config = self._options
        if config is not None and not isinstance(
            config, task.RetryPolicy
        ):  # pragma: no cover - typed resolvers return a policy
            raise UserError('Durable Task activity configuration must be a `durabletask.task.RetryPolicy`.')
        recorded = await self._call_waiting_retry_after(
            _JSON.validate_python(self.operation.parameter_transport.dump(params)), config
        )
        if recorded.error is not None:
            raise UserError(recorded.error)
        return self.operation.result_codec.load(recorded.value)

    async def _call_waiting_retry_after(self, payload: JsonValue, policy: task.RetryPolicy | None) -> _ActivityResult:
        """Call the activity, and again after each wait a model error asked for, within the retry policy.

        Only a model activity whose retry policy allows another attempt asks for a wait; see `_retry_after_result`.
        Each call carries the retry policy, continued from the attempts before it, so a failure that asks for no wait
        gets the policy's own backoff and the activity is attempted no more often than the policy allows.
        `asyncio.sleep` waits on a durable timer here.
        """
        loop = current_loop()
        assert loop is not None
        started = loop.time()
        attempts = 0
        call_policy = policy
        while True:
            pending = loop.context.call_activity(
                self._name, input=payload, retry_policy=call_policy, return_type=object
            )
            recorded = _RESULT.validate_python(await loop.wait(pending))
            if recorded.retry_after is None:
                return recorded
            assert policy is not None, 'Only a model activity with a retry policy asks for a wait.'
            # The SDK counts a retried activity's attempts on the task, but doesn't expose the count.
            attempts += pending._attempt_count if isinstance(pending, task.RetryableTask) else 1  # pyright: ignore[reportPrivateUsage]
            if attempts >= policy.max_number_of_attempts or (
                policy.retry_timeout is not None and loop.time() - started >= policy.retry_timeout.total_seconds()
            ):
                assert recorded.failure is not None
                failure = ParseDict(recorded.failure, _pb.TaskFailureDetails())
                raise task.TaskFailedError(
                    f'{loop.context.instance_id}: Activity {self._name!r} failed: {failure.errorMessage}', failure
                )
            await asyncio.sleep(min(recorded.retry_after, _retry_after_cap(policy, attempts)))
            call_policy = _continued_policy(policy, attempts, timedelta(seconds=loop.time() - started))


async def _run_operation(
    operation: DurableOperation[ParamsT, WireT, ResultT], params: ParamsT, *, wait_retry_after: bool
) -> object:
    """Run the operation in its activity, and record its result, or the wait a model error asked for."""
    try:
        result = await operation.handler(params)
    except Exception as exc:
        if not wait_retry_after or (waiting := _retry_after_result(exc)) is None:
            raise
        return waiting
    try:
        value = _JSON.validate_python(operation.result_codec.dump(result))
    except (ValidationError, PydanticSerializationError, UnicodeDecodeError) as exc:
        return _failure(SerializationFailure(exc))
    return _RESULT.dump_python(_ActivityResult(format=_RESULT_FORMAT, value=value), mode='json')


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
        if name in self._activity_names:  # pragma: no cover - needs a SHA-256 prefix collision
            raise UserError(
                f'Azure Durable Task activity name {name!r} is not unique after escaping. '
                'Choose distinct agent, model and toolset identities.'
            )
        self._activity_names.add(name)
        transport = operation.parameter_transport
        if not isinstance(transport, JsonTransport):  # pragma: no cover - this capability installs every transport
            raise TypeError(f'{type(transport).__name__} does not accept the Durable Task JSON wire.')
        json_transport: JsonTransport[ParamsT] = transport
        # Bound model operations are always called with the policy they're registered with. Without one that allows
        # another attempt, a model activity fails natively, as it would without waiting `retry_after`.
        wait_retry_after = (
            isinstance(operation.operation_id, MODEL_OPERATION_IDS)
            and config is not None
            and config.max_number_of_attempts > 1
        )

        async def execute(payload: object) -> object:
            try:
                wire = INPUT.validate_python(payload)
            except ValidationError as exc:
                return _failure(SerializationFailure(exc))
            try:
                async with self._activity_scope(wire.deps_input):
                    with durable_unit_scope():
                        try:
                            params = json_transport.load(payload, runtime=self)
                        except ValidationError as exc:
                            return _failure(SerializationFailure(exc))
                        return await _run_operation(operation, params, wait_retry_after=wait_retry_after)
            except UserError as exc:
                # SDK RetryPolicy retries every activity exception. Raise recorded codec and
                # configuration failures in orchestration code instead, outside its retry boundary.
                return _failure(exc)

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
        # The SDK creates its channel lazily at start; in-memory workers may not configure a channel.
        if hasattr(worker, '_channel_options'):
            # The SDK has no public setter for channel options after construction.
            channel_options = worker._channel_options or GrpcChannelOptions()  # pyright: ignore[reportPrivateUsage]
            raw_options = list(channel_options.raw_options)
            for index, (key, value) in enumerate(raw_options):
                if key == 'grpc.primary_user_agent':
                    if 'pydantic-ai' not in value.split():
                        raw_options[index] = (key, f'{value} pydantic-ai'.strip())
                    break
            else:
                raw_options.append(('grpc.primary_user_agent', 'pydantic-ai'))
            worker._channel_options = replace(channel_options, raw_options=raw_options)  # pyright: ignore[reportPrivateUsage]
        with start_blocking_portal(name='pydantic-ai-durabletask') as portal:
            self._portal = portal
            try:
                for activity in self.registrations():
                    worker.add_activity(activity)
                yield
            finally:
                self._portal = None
