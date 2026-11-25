"""Registered Pydantic AI operations backed by Hatchet tasks.

Each durable operation is a Hatchet task, registered when the agent is constructed so the worker can
serve it. Inside a durable task, an operation runs by spawning its task as a child run: the durable
task's event log records the spawn and its result, so a replayed durable task gets the result back
without running the operation again.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from hatchet_sdk import (
    ConcurrencyStrategy,
    Context,
    DesiredWorkerLabel,
    Hatchet,
    NonRetryableError,
    RateLimit,
    RetryAfterError,
    Standalone,
    TaskRunError,
)
from hatchet_sdk.types.priority import Priority
from pydantic import BaseModel, JsonValue, TypeAdapter
from typing_extensions import TypedDict

from pydantic_ai.durable_exec._model_error_codec import decode_model_error, encode_model_error
from pydantic_ai.durable_exec._operation import (
    CapabilityOperationId,
    DurableOperation,
    DurableOperationId,
    EventStreamHandlerId,
    ModelCancelSuspendedResponseId,
    ModelCompactMessagesId,
    ModelCountTokensId,
    ModelRequestId,
    ParameterTransports,
    ToolsetCallToolId,
    ToolsetGetInstructionsId,
    ToolsetGetToolsId,
    ToolsetValidateToolArgumentsId,
)
from pydantic_ai.durable_exec._operation_backend import (
    BoundDurableOperation,
    RegisteredOperationBackend,
    RoleBasedOperationConfig,
    durable_unit_scope,
)
from pydantic_ai.durable_exec._operation_names import (
    DurableInvocationName,
    DurableOperationNamer,
    JournalOperationNamer,
)
from pydantic_ai.durable_exec._retry_after import model_error_retry_after
from pydantic_ai.durable_exec._utils import mark_streamed_before_failing, streamed_before_failing
from pydantic_ai.exceptions import (
    FallbackExceptionGroup,
    ModelAPIError,
    RetryModelRequest,
    UnexpectedModelBehavior,
    UserError,
)
from pydantic_ai.messages import ModelResponse

from ._clock import advance_durable_clock
from ._transports import TaskInput

if TYPE_CHECKING:
    from ._durability import HatchetDurability

ParamsT = TypeVar('ParamsT')
WireT = TypeVar('WireT')
ResultT = TypeVar('ResultT')


class HatchetTaskConfig(TypedDict, total=False):
    """Options for the Hatchet tasks that run an agent's model requests, tool calls, and other operations.

    The registration options are fixed when the agent is constructed, as Hatchet registers each task
    once, so they can only vary per toolset (see `toolset_task_config`). The run options are applied
    each time a task is spawned, so a tool's `'hatchet'` metadata can also set them.
    """

    # Registration options, passed to `Hatchet.task()`.
    retries: int
    """How many times Hatchet retries a failed task. Defaults to `0`."""
    execution_timeout: timedelta
    """How long one attempt may run before Hatchet fails it. Defaults to 60 seconds."""
    schedule_timeout: timedelta
    """How long a task may wait for a worker before Hatchet fails it. Defaults to 5 minutes."""
    backoff_factor: float
    """The exponential backoff factor between retries."""
    backoff_max: timedelta
    """The maximum backoff between retries."""
    rate_limits: list[RateLimit]
    """Hatchet rate limits the task consumes, e.g. to cap the requests a model provider gets across workers."""
    concurrency: int | ConcurrencyStrategy | list[ConcurrencyStrategy]
    """Hatchet concurrency limits for the task."""
    default_priority: Priority
    """The priority the task's runs get unless `priority` is set."""

    # Run options, passed to `Standalone.aio_run()`.
    priority: Priority
    """The priority of each run of the task."""
    sticky: bool
    """Whether to run the task on the worker running the durable task, which keeps its local state warm."""
    desired_worker_labels: list[DesiredWorkerLabel]
    """Labels the worker running the task should have, e.g. to run a tool on a worker with a GPU."""
    additional_metadata: dict[str, str]
    """Extra metadata attached to each run, shown and filterable in the Hatchet dashboard."""


_REGISTRATION_KEYS = frozenset(
    {
        'retries',
        'execution_timeout',
        'schedule_timeout',
        'backoff_factor',
        'backoff_max',
        'rate_limits',
        'concurrency',
        'default_priority',
    }
)
_RUN_KEYS = frozenset({'priority', 'sticky', 'desired_worker_labels', 'additional_metadata'})

NON_RETRYABLE_ERRORS: tuple[type[Exception], ...] = (UserError, UnexpectedModelBehavior, FallbackExceptionGroup)
"""Errors a retry can't fix, which fail a task without Hatchet retrying it, subclasses included."""

_RETRY_AFTER_MAXIMUM = 60.0
"""The longest `Retry-After` a model error can make Hatchet wait before retrying its task, in seconds."""

_ERROR_PREFIX = 'pydantic_ai_error:'
"""Marks a task failure message that holds an encoded error, which the durable task rebuilds."""

_response_adapter: TypeAdapter[ModelResponse] = TypeAdapter(ModelResponse)


class TaskOutput(BaseModel):
    """The JSON output of every Hatchet task that runs a durable operation."""

    payload: JsonValue
    completed_at: datetime
    """When the task finished, which the durable task's replay-safe clock advances to."""


def validate_task_config(config: Mapping[str, object], label: str) -> HatchetTaskConfig:
    """Reject a key `HatchetTaskConfig` doesn't have, which would otherwise only fail when the task runs."""
    unknown = set(config) - _REGISTRATION_KEYS - _RUN_KEYS
    if unknown:
        raise UserError(f'{label} has unknown Hatchet task config keys: {sorted(unknown)!r}.')
    return cast(HatchetTaskConfig, dict(config))


def _encoded_error_message(kind: str, payload: JsonValue, message: str) -> str:
    # `TaskRunError` keeps a single-line message intact, which `json.dumps` without an indent always is.
    return f'{_ERROR_PREFIX}{json.dumps({"kind": kind, "payload": payload, "message": message})}'


def _model_error_message(error: ModelAPIError) -> str:
    """The failure message for a model error, encoded so the durable task can rebuild it, if it can be encoded."""
    encoded = encode_model_error(error)
    if encoded is None:
        return str(error)
    # What the stream produced before failing, which tells the run its events already went out.
    streamed = streamed_before_failing(error)
    model_error: JsonValue = {
        'error': encoded,
        'streamed': _response_adapter.dump_python(streamed, mode='json') if streamed is not None else None,
    }
    return _encoded_error_message('model_error', model_error, str(error))


def _rebuild_error(error: TaskRunError) -> BaseException | None:
    """The error a task encoded in its failure message, rebuilt with its original type, if it encoded one."""
    message = error.exc
    if not message.startswith(_ERROR_PREFIX):
        return None
    try:
        encoded = json.loads(message[len(_ERROR_PREFIX) :])
    except json.JSONDecodeError:
        return None
    kind = encoded.get('kind')
    payload = encoded.get('payload')
    if kind == 'model_error' and isinstance(payload, dict):
        model_error = cast(dict[str, Any], payload)
        rebuilt = decode_model_error(model_error['error'])
        if rebuilt is not None and (streamed := model_error.get('streamed')) is not None:
            mark_streamed_before_failing(rebuilt, _response_adapter.validate_python(streamed))
        return rebuilt
    if kind == 'retry_model_request':
        model = cast(dict[str, Any], payload).get('model') if isinstance(payload, dict) else None
        return RetryModelRequest(model if isinstance(model, str) else None)
    if kind == 'user_error':
        return UserError(str(encoded.get('message', '')))
    return None


async def run_operation_in_task(
    operation: DurableOperation[ParamsT, WireT, ResultT],
    params: ParamsT,
    *,
    non_retryable_model_errors: tuple[type[ModelAPIError], ...],
) -> TaskOutput:
    """Run an operation's handler as a task body, raising each failure as Hatchet should treat it."""
    try:
        with durable_unit_scope():
            value = await operation.handler(params)
    except RetryModelRequest as retry:
        if retry.model is not None and not isinstance(retry.model, str):
            raise NonRetryableError(
                _encoded_error_message(
                    'user_error',
                    None,
                    '`RetryModelRequest` raised from a capability hook that runs as a durable operation can only '
                    'name the model to attempt by its name: register a model instance with the durability '
                    "capability's `models` and pass its name.",
                )
            ) from retry
        # Asking for another attempt is the hook's answer, not a failure to retry.
        raise NonRetryableError(
            _encoded_error_message('retry_model_request', {'model': retry.model}, str(retry))
        ) from retry
    except ModelAPIError as error:
        message = _model_error_message(error)
        if isinstance(error, non_retryable_model_errors):
            raise NonRetryableError(message) from error
        if (retry_after := model_error_retry_after(error, maximum=_RETRY_AFTER_MAXIMUM)) is not None:
            # Hatchet waits the provider's `Retry-After` before the next attempt, instead of its own backoff.
            raise RetryAfterError(message, after=timedelta(seconds=retry_after)) from error
        raise _RetryableModelError(message) from error
    except NON_RETRYABLE_ERRORS as error:
        payload = _encoded_error_message('user_error', None, str(error)) if isinstance(error, UserError) else str(error)
        raise NonRetryableError(payload) from error
    try:
        return TaskOutput(payload=cast(JsonValue, operation.result_codec.dump(value)), completed_at=datetime.now(UTC))
    except UserError as error:
        # The result codec reports a result it can't encode as a `UserError`. It fails the same way on every
        # attempt, so it isn't retried.
        raise NonRetryableError(_encoded_error_message('user_error', None, str(error))) from error


class _RetryableModelError(Exception):
    """A retryable model error, raised from its task with the error encoded so the durable task can rebuild it."""


_ESCAPED_NAME_CHARACTERS = re.compile(r'[^a-zA-Z0-9._]')


def hatchet_task_name(name: str) -> str:
    """Escape each character Hatchet doesn't allow in a task name, like the `<agent>` toolset ID's brackets.

    Hatchet names may only contain letters, digits, `.`, `-` and `_`. Each other character becomes `-<hex>-`, and
    `-` itself is escaped too, so two different names never escape to the same task name. The escape is
    deterministic, so a task keeps its name across deploys.
    """
    return _ESCAPED_NAME_CHARACTERS.sub(lambda match: f'-{ord(match.group()):x}-', name)


class HatchetOperationNamer(DurableOperationNamer):
    """The standard journal operation names, escaped into valid Hatchet task names."""

    def __init__(self, agent_name: str) -> None:
        self._journal = JournalOperationNamer(agent_name)

    def operation_name(self, operation_id: DurableOperationId) -> str:
        return hatchet_task_name(self._journal.operation_name(operation_id))

    def invocation_name(self, operation_id: DurableOperationId, *, label: str | None) -> DurableInvocationName:
        return DurableInvocationName(self.operation_name(operation_id))


def _summary_metadata(operation_id: DurableOperationId, params: object) -> dict[str, str]:
    """What a task run is doing, attached as metadata so the Hatchet dashboard can filter runs by it."""
    match operation_id:
        case ModelRequestId(model_name=model_name) | ModelCompactMessagesId(model_name=model_name):
            return {'pydantic_ai_model': cast(Any, params).model_id or model_name}
        case ModelCancelSuspendedResponseId(model_name=model_name) | ModelCountTokensId(model_name=model_name):
            return {'pydantic_ai_model': cast(Any, params).model_id or model_name}
        case ToolsetCallToolId(toolset_id=toolset_id) | ToolsetValidateToolArgumentsId(toolset_id=toolset_id):
            return {'pydantic_ai_toolset': toolset_id, 'pydantic_ai_tool': cast(Any, params).name}
        case ToolsetGetToolsId(toolset_id=toolset_id) | ToolsetGetInstructionsId(toolset_id=toolset_id):
            return {'pydantic_ai_toolset': toolset_id}
        case CapabilityOperationId(capability_id=capability_id, operation=operation):
            return {'pydantic_ai_capability': f'{capability_id}.{operation}'}
        case EventStreamHandlerId():
            return {'pydantic_ai_event': cast(Any, params).event.event_kind}
        case _:
            return {}


class HatchetBoundOperation(BoundDurableOperation[ParamsT, WireT, ResultT], Generic[ParamsT, WireT, ResultT]):
    """Run one operation by spawning its Hatchet task from the durable task."""

    def __init__(
        self,
        operation: DurableOperation[ParamsT, WireT, ResultT],
        *,
        task: Standalone[TaskInput, TaskOutput],
        name: str,
        config: HatchetTaskConfig,
        agent_name: str,
    ) -> None:
        self._operation = operation
        self.task = task
        self._name = name
        self._config = config
        self._agent_name = agent_name

    @property
    def operation(self) -> DurableOperation[ParamsT, WireT, ResultT]:
        return self._operation

    async def __call__(self, params: ParamsT, *, config: object | None = None) -> ResultT:
        resolved = cast(HatchetTaskConfig, config) if config is not None else self._config
        changed = {key for key in _REGISTRATION_KEYS if resolved.get(key) != self._config.get(key)}
        if changed:
            raise UserError(
                f'Hatchet task options {sorted(changed)!r} are fixed when the agent is constructed, as Hatchet '
                f'registers the {self._name!r} task once, so a tool can only override its run options '
                f'({", ".join(sorted(_RUN_KEYS))}). Put tools needing different task options in a separate toolset '
                'and configure it with `toolset_task_config`.'
            )
        payload = cast(TaskInput, self._operation.parameter_transport.dump(params))
        metadata = {
            'pydantic_ai_agent': self._agent_name,
            **_summary_metadata(self._operation.operation_id, params),
            **resolved.get('additional_metadata', {}),
        }
        try:
            output = await self.task.aio_run(
                payload,
                additional_metadata=metadata,
                priority=resolved.get('priority'),
                sticky=resolved.get('sticky', False),
                desired_worker_labels=resolved.get('desired_worker_labels'),
            )
        except TaskRunError as error:
            rebuilt = _rebuild_error(error)
            if rebuilt is not None:
                raise rebuilt from error
            raise
        advance_durable_clock(output.completed_at)
        return self._operation.result_codec.load(output.payload)


class HatchetOperationBackend(RegisteredOperationBackend[HatchetTaskConfig]):
    """Register each Pydantic AI operation as a Hatchet task."""

    def __init__(
        self,
        hatchet: Hatchet,
        *,
        runtime: HatchetDurability[Any],
        agent_name: str,
        config: RoleBasedOperationConfig[HatchetTaskConfig],
        parameter_transports: ParameterTransports,
        non_retryable_model_errors: tuple[type[ModelAPIError], ...],
    ) -> None:
        super().__init__(
            namer=HatchetOperationNamer(agent_name), config=config, parameter_transports=parameter_transports
        )
        self._hatchet = hatchet
        self._runtime = runtime
        self._agent_name = agent_name
        self._non_retryable_model_errors = non_retryable_model_errors

    def register(
        self,
        operation: DurableOperation[ParamsT, WireT, ResultT],
        *,
        name: str,
        config: HatchetTaskConfig,
    ) -> tuple[BoundDurableOperation[ParamsT, WireT, ResultT], Sequence[Callable[..., object]]]:
        runtime = self._runtime
        non_retryable_model_errors = self._non_retryable_model_errors

        async def operation_task(input: TaskInput, ctx: Context) -> TaskOutput:
            params = operation.parameter_transport.load(cast(WireT, input), runtime=runtime)
            return await run_operation_in_task(operation, params, non_retryable_model_errors=non_retryable_model_errors)

        registration: dict[str, Any] = {key: config[key] for key in _REGISTRATION_KEYS if key in config}
        task = self._hatchet.task(name=name, input_validator=TaskInput, **registration)(operation_task)
        runtime._register_task(task)  # pyright: ignore[reportPrivateUsage]
        bound = HatchetBoundOperation(operation, task=task, name=name, config=config, agent_name=self._agent_name)
        return bound, (cast(Callable[..., object], task),)
