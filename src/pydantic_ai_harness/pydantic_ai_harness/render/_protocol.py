"""JSON wire protocol used by Render Workflows operation tasks.

Besides the request and result envelopes, this module owns the *effects* an operation
produces for its caller: the usage a child task added and the events it emitted. A child
task cannot reach the caller's `RunUsage` object or its event stream, so both are buffered
by an `EffectRecorder` while the task runs and travel back inside the successful result
envelope, where `apply_effects` replays them through the caller's public run context.
"""

from __future__ import annotations

import json
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, TypeGuard, TypeVar

from pydantic import TypeAdapter, ValidationError
from typing_extensions import NotRequired, TypedDict

from pydantic_ai.durable_exec import JSON_CODEC
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelRetry,
    SkipModelRequest,
    SkipToolExecution,
    SkipToolValidation,
    ToolFailed,
    UserError,
)
from pydantic_ai.messages import AgentStreamEvent, CapabilityEvent, CustomEvent, ModelResponse
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from ._compat import (
    JSONObject as JsonObject,
    JSONValue as JsonValue,
    dump_json_object,
    load_json_object,
    load_json_type,
    normalize_json_value,
)

PROTOCOL_VERSION = 2
_SUPPORTED_PROTOCOL_VERSIONS = frozenset({1, PROTOCOL_VERSION})
MAX_ARGUMENT_BYTES = 4 * 1024 * 1024

AgentDepsT = TypeVar('AgentDepsT')

_OBJECT_ADAPTER: TypeAdapter[dict[str, object]] = TypeAdapter(dict[str, object])
_LIST_ADAPTER: TypeAdapter[list[object]] = TypeAdapter(list[object])


class OperationRequest(TypedDict):
    version: int
    operation: str
    payload: JsonObject


class OperationEffects(TypedDict, total=False):
    """Caller-side state one child task produced, encoded beside its result payload."""

    usage: JsonObject
    events: list[JsonObject]


class OperationSuccess(TypedDict):
    version: int
    status: Literal['ok']
    payload: JsonValue
    # Absent whenever a task changed nothing its caller has to apply, which keeps the
    # three-key envelope every existing worker already writes and reads.
    effects: NotRequired[OperationEffects]


class OperationControlFlowError(TypedDict):
    version: int
    status: Literal['control-flow']
    error: JsonObject


class OperationPermanentError(TypedDict):
    version: int
    status: Literal['error']
    error: JsonObject


OperationResult: TypeAlias = OperationSuccess | OperationControlFlowError | OperationPermanentError


class RenderProtocolError(ValueError):
    """Raised when a Render operation receives an invalid wire envelope."""


class RenderPayloadTooLargeError(RenderProtocolError):
    """Raised before dispatch when task arguments exceed Render's limit."""


@dataclass(frozen=True)
class ChildEffects:
    """The decoded effects one child task reported for its caller to apply."""

    usage: RunUsage | None
    """What the child added to the run usage it was given, not the usage it ended with."""
    events: tuple[CustomEvent | CapabilityEvent, ...]
    """Every event the child emitted, in emission order."""


@dataclass(frozen=True)
class OperationOutcome:
    """A successful operation result and whatever its caller still has to apply."""

    payload: JsonValue
    effects: ChildEffects | None


class EffectRecorder:
    """Buffer the effects of one child task run while that run is in progress.

    A task run that fails, or that encodes expected control flow instead of a result,
    reports nothing: effects reach the caller only on the successful result, so an
    abandoned or retried attempt cannot apply part of what it did.
    """

    def __init__(self) -> None:
        self._usage: RunUsage | None = None
        self._usage_before: RunUsage | None = None
        self._events: list[CustomEvent | CapabilityEvent] = []
        self._event_capability_id: str | None = None

    def watch_usage(self, usage: RunUsage) -> None:
        """Snapshot the run usage this task starts from, so only its own delta is reported."""
        if self._usage is not None:
            return
        self._usage = usage
        # `RunUsage() + usage` is the public way to take a detached copy: the operands stop
        # sharing their `details` mapping, which the task is about to mutate in place.
        self._usage_before = RunUsage() + usage

    def record_event(self, event: CustomEvent | CapabilityEvent) -> None:
        """Buffer one emitted event, preserving emission order.

        `RunContext.emit` is the only way a child produces an event and the only way its
        caller replays one, so these two families are what the envelope carries.
        """
        self._events.append(event)

    def set_event_capability(self, capability_id: str | None) -> None:
        """Record which capability owns the function tool running in this task."""
        self._event_capability_id = capability_id

    @property
    def event_capability_id(self) -> str | None:
        """Return the capability ID that owns the current function tool, if any."""
        return self._event_capability_id

    def effects(self) -> OperationEffects | None:
        """Encode what the caller must apply, or `None` when this task changed nothing."""
        effects: OperationEffects = {}
        usage = self._usage_delta()
        if usage is not None:
            effects['usage'] = dump_json_object(RunUsage, usage)
        if self._events:
            effects['events'] = _dump_events(self._events)
        return effects or None

    def _usage_delta(self) -> RunUsage | None:
        usage, before = self._usage, self._usage_before
        if usage is None or before is None:
            return None
        delta = usage - before
        return delta if delta.has_values() else None


_EFFECT_RECORDER: ContextVar[EffectRecorder | None] = ContextVar(
    'pydantic_ai_harness_render_effect_recorder', default=None
)


@contextmanager
def recording_effects() -> Generator[EffectRecorder, None, None]:
    """Collect the effects of one child task run, isolated from concurrent sibling runs."""
    recorder = EffectRecorder()
    token = _EFFECT_RECORDER.set(recorder)
    try:
        yield recorder
    finally:
        _EFFECT_RECORDER.reset(token)


def current_effect_recorder() -> EffectRecorder | None:
    """The recorder collecting effects for the child task run in progress, if there is one."""
    return _EFFECT_RECORDER.get()


async def apply_effects(
    effects: ChildEffects | None,
    *,
    ctx: RunContext[AgentDepsT],
) -> None:
    """Apply one child task's reported effects to the caller's run.

    Usage and events go through `ctx`, the live context the operation was called with.
    For a capability-owned tool, that context retains the tool definition Pydantic AI
    uses to validate the event's capability attribution.

    The caller applies each result once, so a replayed workflow rebuilds its own usage and
    event stream from the journaled results rather than counting them twice.
    """
    if effects is None:
        return
    if effects.usage is not None:
        ctx.usage.incr(effects.usage)
    for event in effects.events:
        await ctx.emit(event)


def make_request(operation: str, payload: object) -> OperationRequest:
    """Build and preflight the single positional argument sent to a Render task."""
    json_payload = _as_json_object(payload, label='operation payload')
    request: OperationRequest = {'version': PROTOCOL_VERSION, 'operation': operation, 'payload': json_payload}
    _check_argument_size(request)
    return request


def read_request(value: object, *, expected_operation: str) -> JsonObject:
    """Validate a task request and return its encoded operation parameters."""
    envelope = _object(value, label='operation request')
    _check_keys(envelope, required={'version', 'operation', 'payload'}, label='operation request')
    _check_version(envelope)
    operation = envelope['operation']
    if operation != expected_operation:
        raise RenderProtocolError(
            f'Render operation request was routed to {expected_operation!r}, but names operation {operation!r}.'
        )
    return _as_json_object(envelope['payload'], label='operation payload')


def success(
    payload: object,
    *,
    effects: OperationEffects | None = None,
    version: int = PROTOCOL_VERSION,
) -> OperationSuccess:
    result: OperationSuccess = {
        'version': version,
        'status': 'ok',
        'payload': _as_json_value(payload, label='operation result'),
    }
    if effects is not None and version >= 2:
        result['effects'] = effects
    return result


def control_flow_error(
    exc: Exception,
    *,
    version: int = PROTOCOL_VERSION,
) -> OperationControlFlowError | None:
    """Encode expected Pydantic AI control flow as a successful Render task result."""
    error: JsonObject
    if isinstance(exc, ModelRetry):
        error = {'kind': 'model-retry', 'message': exc.message}
    elif isinstance(exc, ToolFailed):
        error = {'kind': 'tool-failed', 'message': exc.message}
    elif isinstance(exc, ApprovalRequired):
        error = {'kind': 'approval-required', 'metadata': _as_json_value(exc.metadata, label='approval metadata')}
    elif isinstance(exc, CallDeferred):
        error = {'kind': 'call-deferred', 'metadata': _as_json_value(exc.metadata, label='deferred-call metadata')}
    elif isinstance(exc, SkipModelRequest):
        error = {'kind': 'skip-model-request', 'response': dump_json_object(ModelResponse, exc.response)}
    elif isinstance(exc, SkipToolValidation):
        error = {
            'kind': 'skip-tool-validation',
            'validated_args': dump_json_object(dict[str, Any], exc.validated_args),
        }
    elif isinstance(exc, SkipToolExecution):
        error = {
            'kind': 'skip-tool-execution',
            'result': _as_json_value(JSON_CODEC.dump(Any, exc.result), label='skipped tool result'),
        }
    else:
        return None
    return {'version': version, 'status': 'control-flow', 'error': error}


def permanent_error(
    kind: Literal['invalid-request', 'invalid-result'],
    exc: Exception,
    *,
    version: int = PROTOCOL_VERSION,
) -> OperationPermanentError:
    """Finish a task when retrying cannot repair its boundary data."""
    message = str(exc).strip() or type(exc).__name__
    error: JsonObject = {'kind': kind, 'message': message[:500]}
    return {'version': version, 'status': 'error', 'error': error}


def read_result(value: object) -> JsonValue:
    """Decode the payload of a task result, for a caller that applies no effects."""
    return read_outcome(value).payload


def read_outcome(value: object) -> OperationOutcome:
    """Decode a task result and its effects, recreating expected Pydantic AI control flow."""
    envelope = _object(value, label='operation result')
    _check_version(envelope)
    status = envelope.get('status')
    if status == 'ok':
        version = _protocol_version(envelope)
        _check_keys(
            envelope,
            required={'version', 'status', 'payload'},
            optional={'effects'} if version >= 2 else None,
            label='successful operation result',
        )
        return OperationOutcome(
            payload=_as_json_value(envelope['payload'], label='operation result payload'),
            effects=_read_effects(envelope['effects']) if 'effects' in envelope else None,
        )
    if status == 'control-flow':
        _check_keys(envelope, required={'version', 'status', 'error'}, label='control-flow operation result')
        _raise_control_flow(envelope['error'])
    if status == 'error':
        _check_keys(envelope, required={'version', 'status', 'error'}, label='failed operation result')
        _raise_permanent_error(envelope['error'])
    raise RenderProtocolError(f'Render operation result has unknown status {status!r}.')


def _raise_control_flow(value: object) -> None:
    error = _object(value, label='control-flow error')
    kind = error.get('kind')
    if kind == 'model-retry':
        _check_keys(error, required={'kind', 'message'}, label='model-retry error')
        raise ModelRetry(_string(error['message'], label='model-retry message'))
    if kind == 'tool-failed':
        _check_keys(error, required={'kind', 'message'}, label='tool-failed error')
        raise ToolFailed(_string(error['message'], label='tool-failed message'))
    if kind == 'approval-required':
        _check_keys(error, required={'kind', 'metadata'}, label='approval-required error')
        raise ApprovalRequired(metadata=_metadata(error['metadata'], label='approval metadata'))
    if kind == 'call-deferred':
        _check_keys(error, required={'kind', 'metadata'}, label='call-deferred error')
        raise CallDeferred(metadata=_metadata(error['metadata'], label='deferred-call metadata'))
    if kind == 'skip-model-request':
        _check_keys(error, required={'kind', 'response'}, label='skip-model-request error')
        raise SkipModelRequest(
            load_json_type(
                ModelResponse,
                _as_json_object(error['response'], label='skip-model-request response'),
            )
        )
    if kind == 'skip-tool-validation':
        _check_keys(error, required={'kind', 'validated_args'}, label='skip-tool-validation error')
        raise SkipToolValidation(
            load_json_object(_as_json_object(error['validated_args'], label='skip-tool-validation arguments'))
        )
    if kind == 'skip-tool-execution':
        _check_keys(error, required={'kind', 'result'}, label='skip-tool-execution error')
        raise SkipToolExecution(JSON_CODEC.load(Any, _as_json_value(error['result'], label='skipped tool result')))
    raise RenderProtocolError(f'Render operation result has unknown control-flow kind {kind!r}.')


def _raise_permanent_error(value: object) -> None:
    error = _object(value, label='operation error')
    _check_keys(error, required={'kind', 'message'}, label='operation error')
    kind = error['kind']
    if kind not in ('invalid-request', 'invalid-result'):
        raise RenderProtocolError(f'Render operation result has unknown error kind {kind!r}.')
    message = _string(error['message'], label='operation error message')
    raise UserError(f'Render operation {kind.replace("-", " ")}: {message}')


def _read_effects(value: object) -> ChildEffects:
    effects = _object(value, label='operation effects')
    _check_keys(effects, required=set(), optional={'usage', 'events'}, label='operation effects')
    if not effects:
        raise RenderProtocolError('Operation effects must carry usage, events, or both.')
    return ChildEffects(
        usage=_effects_usage(effects['usage']) if 'usage' in effects else None,
        events=_effects_events(effects['events']) if 'events' in effects else (),
    )


def _effects_usage(value: object) -> RunUsage:
    payload = _as_json_object(value, label='operation effects usage')
    if _contains_negative_count(payload):
        raise RenderProtocolError('Operation effects usage deltas cannot contain negative counts.')
    try:
        return load_json_type(RunUsage, payload)
    except ValidationError as exc:
        raise RenderProtocolError(f'Operation effects usage is not a run usage delta: {exc}') from exc


def _contains_negative_count(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value < 0
    if _is_object_dict(value):
        return any(_contains_negative_count(item) for item in value.values())
    if _is_object_list(value):
        return any(_contains_negative_count(item) for item in value)
    return False


def _is_object_dict(value: object) -> TypeGuard[dict[object, object]]:
    return isinstance(value, dict)


def _is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _effects_events(value: object) -> tuple[CustomEvent | CapabilityEvent, ...]:
    try:
        payloads = _LIST_ADAPTER.validate_python(value, strict=True)
    except ValidationError as exc:
        raise RenderProtocolError('Operation effects events must be a JSON array.') from exc
    return tuple(_effects_event(payload) for payload in payloads)


def _effects_event(value: object) -> CustomEvent | CapabilityEvent:
    payload = _as_json_object(value, label='operation effects event')
    try:
        event = JSON_CODEC.load(AgentStreamEvent, payload)
    except ValidationError as exc:
        raise RenderProtocolError(f'Operation effects event is not an agent stream event: {exc}') from exc
    if not isinstance(event, CustomEvent | CapabilityEvent):
        raise RenderProtocolError(f'Operation effects carry a {type(event).__name__} its caller cannot emit.')
    return event


def _dump_events(events: Sequence[CustomEvent | CapabilityEvent]) -> list[JsonObject]:
    # Each event is dumped as its own type rather than through the `AgentStreamEvent` union:
    # the union's members are registered as event classes are defined, so serializing through
    # it asks every member to encode an event it may never have seen. The tag each event
    # carries is what routes it back to its class on the reader's side.
    return [dump_json_object(type(event), event) for event in events]


def _check_argument_size(request: OperationRequest) -> None:
    # TaskContext.run serializes positional arguments as a JSON list. Measure that final shape,
    # including JSON punctuation and whitespace, instead of only measuring the semantic payload.
    encoded = _json_bytes([request], label='operation request')
    if len(encoded) > MAX_ARGUMENT_BYTES:
        raise RenderPayloadTooLargeError(
            f'Render operation arguments are {len(encoded)} bytes, exceeding the {MAX_ARGUMENT_BYTES}-byte limit.'
        )


def _as_json_value(value: object, *, label: str) -> JsonValue:
    try:
        return normalize_json_value(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise RenderProtocolError(f'{label.capitalize()} must be JSON serializable: {exc}') from exc


def _as_json_object(value: object, *, label: str) -> JsonObject:
    json_value = _as_json_value(value, label=label)
    if not isinstance(json_value, dict):
        raise RenderProtocolError(f'{label.capitalize()} must be a JSON object.')
    return _OBJECT_ADAPTER.validate_python(json_value, strict=True)


def _json_bytes(value: object, *, label: str) -> bytes:
    normalized = _as_json_value(value, label=label)
    try:
        return json.dumps(normalized, allow_nan=False).encode()
    except (OverflowError, TypeError, ValueError) as exc:
        raise RenderProtocolError(f'{label.capitalize()} must be JSON serializable: {exc}') from exc


def _object(value: object, *, label: str) -> dict[str, object]:
    try:
        return _OBJECT_ADAPTER.validate_python(value, strict=True)
    except ValidationError as exc:
        raise RenderProtocolError(f'{label.capitalize()} must be a JSON object.') from exc


def _check_keys(value: dict[str, object], *, required: set[str], label: str, optional: set[str] | None = None) -> None:
    actual = set(value)
    missing = sorted(required - actual)
    extra = sorted(actual - required - (optional or set()))
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f'missing {missing!r}')
        if extra:
            details.append(f'unexpected {extra!r}')
        raise RenderProtocolError(f'{label.capitalize()} has invalid fields: {", ".join(details)}.')


def _check_version(value: dict[str, object]) -> None:
    _protocol_version(value)


def _protocol_version(value: dict[str, object]) -> int:
    version = value.get('version')
    if not isinstance(version, int) or isinstance(version, bool) or version not in _SUPPORTED_PROTOCOL_VERSIONS:
        raise RenderProtocolError(
            f'Render operation protocol version {version!r} is unsupported; '
            f'expected one of {sorted(_SUPPORTED_PROTOCOL_VERSIONS)}.'
        )
    return version


def read_protocol_version(value: object) -> int:
    """Return the supported protocol version named by an operation envelope."""
    return _protocol_version(_object(value, label='operation envelope'))


def _string(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise RenderProtocolError(f'{label.capitalize()} must be a string.')
    return value


def _metadata(value: object, *, label: str) -> dict[str, object] | None:
    if value is None:
        return None
    return _object(value, label=label)
