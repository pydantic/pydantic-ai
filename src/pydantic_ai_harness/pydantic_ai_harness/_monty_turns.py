"""Run a remote Monty session one turn at a time, each turn in a call that can leave the workflow.

A turn is the stretch of a snippet between two host interactions: from `feed_start` or a `resume`
until the sandbox pauses at an external call, waits on deferred calls, finishes, or fails. Inside a
Temporal workflow, a remote turn keeps the workflow task waiting on the network for as long as the
snippet computes, and Temporal fails a workflow task that does not yield within 2 seconds. Running
each turn as an activity moves that wait out of the workflow.

The activity can run in any worker process, so the session cannot stay open between turns. A Monty
server that stores sessions saves the state after each turn as a record that never changes and
returns its ID, and the next turn starts from that ID. Everything a turn takes or returns is
therefore plain data: the types below are what crosses the activity boundary.
"""

from __future__ import annotations

import base64
import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from functools import cache
from typing import Any, Literal, TypeGuard, TypeVar, cast, get_args

from pydantic_core import PydanticSerializationError, to_json
from pydantic_monty import (
    AsyncFunctionSnapshot,
    AsyncFutureSnapshot,
    AsyncMontyRemote,
    AsyncMontySession,
    AsyncNameLookupSnapshot,
    AsyncSnapshot,
    CollectString,
    ExcType,
    ExternalExceptionData,
    ExternalResult,
    ExternalReturnValue,
    ExternalSettledResult,
    MontyComplete,
    MontyCrashedError,
    MontyDisconnectError,
    MontyRuntimeError,
    MontySyntaxError,
    MontyTypingError,
    MountDir,
    OSPolicy,
    ResourceLimits,
)

from pydantic_ai.messages import is_multi_modal_content
from pydantic_ai_harness._typeguards import is_mapping

SettledData = ExternalReturnValue | ExternalExceptionData
"""A settled call result as plain data, which a turn in another process can resume the sandbox with."""

SLEEP_FUNCTIONS = frozenset({'time.sleep', 'asyncio.sleep', 'system.sleep', 'system.async_sleep'})
"""The OS calls CodeMode answers on the host, charging the wait to the snippet's sleep allowance."""

_EXCEPTION_TYPES: dict[str, list[ExcType]] = {}
for _name in get_args(ExcType):
    _EXCEPTION_TYPES.setdefault(_name.rpartition('.')[2], []).append(_name)


def to_wire(value: object) -> object:
    """`value` as JSON that `from_wire` turns back into an equal value of the same types.

    Temporal's converter writes activity inputs and results as JSON, which would turn bytes into text
    and tuples into lists, and fails on bytes that are not UTF-8. Every value whose JSON form would
    differ is tagged, and every dict is tagged too, so a dict the snippet built is never read as a tag.
    That covers every type Monty hands the host: dates, times, `timedelta`, `Decimal` and sets too. A
    value of any other type crosses as its JSON form and comes back as that form, as a tool result
    does when CodeMode hands it to the sandbox.

    Raises `TypeError` once it has written more values than `TURN_PAYLOAD_BYTES`. Each value
    takes at least one byte of JSON, so the result could not cross anyway, and a list that holds the
    same list many times stops being expanded here instead of growing exponentially.
    """
    return _Encoder().encode(value)


_TAGS = frozenset(
    {'float', 'bytes', 'tuple', 'dict', 'set', 'frozenset', 'date', 'datetime', 'time', 'timedelta', 'Decimal'}
)
"""The keys of a one-key object that `to_wire` wrote for a value JSON has no form of its own for."""


class _Encoder:
    """`to_wire`'s walk, which counts the values it writes so it can stop as soon as they cannot fit."""

    def __init__(self) -> None:
        self.values_left = TURN_PAYLOAD_BYTES

    def encode(self, value: object) -> object:
        self.values_left -= 1
        if self.values_left < 0:
            raise TurnTooLarge(f'a value with more than {TURN_PAYLOAD_BYTES} parts cannot cross as JSON')
        if _is_list_or_tuple(value):
            items = [self.encode(item) for item in value]
            return {'tuple': items} if isinstance(value, tuple) else items
        if is_mapping(value):
            return self._mapping(value)
        if isinstance(value, set | frozenset):
            items = cast(set[object] | frozenset[object], value)
            return {type(items).__name__: [self.encode(item) for item in items]}
        return _scalar_to_wire(value)

    def _mapping(self, value: Mapping[Any, Any]) -> object:
        if all(isinstance(key, str) for key in value) and not (len(value) == 1 and next(iter(value)) in _TAGS):
            # A plain object, one level deep as in the workflow, so nesting that fits there fits here.
            return {key: self.encode(item) for key, item in value.items()}
        return {'dict': [[self.encode(key), self.encode(item)] for key, item in value.items()]}


def _output_to_wire(output: object) -> object:
    """A snippet's result as a turn returns it, or a note saying why it cannot be returned."""
    try:
        wired = to_wire(model_safe_result(output))
        to_json(MontyTurn('complete', output=wired))  # the depth limit applies to the turn as a whole
    except TurnTooLarge:
        return f'[the result has more than {TURN_PAYLOAD_BYTES} parts, more than a turn can return]'
    except PydanticSerializationError:
        return '[the result is nested more deeply than a turn can return]'
    return wired


def _scalar_to_wire(value: object) -> object:
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {'float': repr(value)}
    if isinstance(value, bytes):
        return {'bytes': base64.b64encode(value).decode()}
    # The other types Monty hands the host, tagged so they come back as themselves.
    if isinstance(value, date | time):  # `datetime` is a `date`
        return {type(value).__name__: value.isoformat()}
    if isinstance(value, timedelta):
        return {'timedelta': [value.days, value.seconds, value.microseconds]}
    if isinstance(value, Decimal):
        return {'Decimal': str(value)}
    # As a live session does, for example for a `bytearray` tool result.
    raise TypeError(f'Cannot convert {type(value).__module__}.{type(value).__qualname__} to Monty value')


def from_wire(value: object) -> Any:
    """The value `to_wire` encoded."""
    if _is_list_or_tuple(value):
        return [from_wire(item) for item in value]
    if not is_mapping(value):
        return value
    if len(value) != 1 or next(iter(value)) not in _TAGS:
        return {key: from_wire(item) for key, item in value.items()}
    ((tag, data),) = value.items()
    match tag:
        case 'float':
            return float(str(data))
        case 'bytes':
            return base64.b64decode(str(data))
        case 'tuple':
            return tuple(from_wire(data))
        case 'dict':
            assert _is_list_or_tuple(data)
            return dict(_decoded_pair(pair) for pair in data)
        case 'set' | 'frozenset':
            return (set if tag == 'set' else frozenset)(from_wire(data))
        case 'date' | 'datetime' | 'time':
            return {'date': date, 'datetime': datetime, 'time': time}[str(tag)].fromisoformat(str(data))
        case 'timedelta':
            days, seconds, microseconds = cast(list[int], data)
            return timedelta(days, seconds, microseconds)
        case _:  # 'Decimal'
            return Decimal(str(data))


def _decoded_pair(pair: object) -> tuple[Any, Any]:
    assert _is_list_or_tuple(pair)
    key, item = pair
    return from_wire(key), from_wire(item)


def settled_data(result: ExternalSettledResult) -> SettledData:
    """`result` with any exception instance replaced by the type and message Monty gives the sandbox for it.

    Monty raises a host exception in the sandbox as the nearest type it knows, with the exception's only
    string argument as the message, or `str()` of the exception otherwise. Dotted names such as
    `json.JSONDecodeError` match only classes from that module, so a user's own `Error` class is not
    read as `binascii.Error`.
    """
    if 'return_value' in result:
        return ExternalReturnValue(return_value=to_wire(result['return_value']))
    if 'exception' not in result:  # pragma: no cover
        # Already data. The executor settles every call with a value or an exception instance.
        return result
    exc = result['exception']
    # `BaseException` is one of Monty's types, so every exception's MRO reaches a match.
    exc_type: ExcType = next(
        name
        for base in type(exc).__mro__
        for name in _EXCEPTION_TYPES.get(base.__name__, ())
        if base.__module__.partition('.')[0] in ('builtins', name.partition('.')[0])
    )
    message = exc.args[0] if len(exc.args) == 1 and isinstance(exc.args[0], str) else str(exc)
    return ExternalExceptionData(exc_type=exc_type, message=message)


# Scalars every tool-return serializer handles. Unlike CodeMode's `_SANDBOX_NATIVE_SCALARS` this
# leaves out `Ellipsis`, which Monty holds but JSON has no form for.
_MODEL_NATIVE_SCALARS = (str, bytes, bytearray, bool, int, float, type(None))


def model_safe_result(value: object) -> object:
    """Render the parts of a snippet's result that have no JSON form as their `repr`.

    Monty hands some sandbox values back as host objects: `type(x)` and `ValueError` arrive
    as `type` objects, `len` as a `MontyStdTypeProxy`, a bare exception instance as itself.
    None of them serialize, so the tool return would abort the run in whichever layer
    renders it first (`ToolOutputLimits`, or pydantic-ai building the model request). The
    `repr` is what the snippet's author would have seen in a Python REPL. Non-finite floats
    do serialize, but as `null`, which would hide the result, so they render as `repr` too.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, _MODEL_NATIVE_SCALARS) or is_multi_modal_content(value):
        return value
    if is_mapping(value):
        return _model_safe_mapping(value)
    if _is_list_or_tuple(value):
        items = [model_safe_result(item) for item in value]
        return tuple(items) if isinstance(value, tuple) else items
    return value if _serializes(value) else repr(value)


def _model_safe_mapping(value: Mapping[object, object]) -> object:
    """`model_safe_result` for a mapping, rendering each key that has no JSON form as its `repr`."""
    rendered = {key if _serializes({key: None}) else repr(key): model_safe_result(item) for key, item in value.items()}
    # A rendered key can equal an existing one (`{int: 1, "<class 'int'>": 2}`), and
    # the dict would silently drop an entry; the whole mapping's `repr` loses nothing.
    return rendered if len(rendered) == len(value) else repr(value)


def _is_list_or_tuple(value: object) -> TypeGuard[list[object] | tuple[object, ...]]:
    return isinstance(value, (list, tuple))


def _serializes(value: object) -> bool:
    try:
        to_json(value)
    except PydanticSerializationError:
        return False
    return True


@dataclass
class TurnResume:
    """How to resume the snapshot a turn paused at."""

    kind: Literal['call', 'deferred', 'settled']
    """`call`: the paused call ended with `result`. `deferred`: it runs in the background (`ExternalFuture`).
    `settled`: the deferred calls the sandbox awaits ended with `settled`."""
    result: SettledData | None = None
    settled: dict[int, SettledData] = field(default_factory=dict[int, SettledData])

    @classmethod
    def from_result(cls, result: ExternalResult) -> TurnResume:
        if 'future' in result:
            return cls('deferred')
        return cls('call', result=settled_data(result))


@dataclass
class TurnRequest:
    """One turn: feed `code` into the session stored as `state`, or resume the snapshot stored as `state`."""

    state: str | None
    """The stored record to start from, base64-encoded; `None` starts a fresh session."""
    limits: ResourceLimits
    """The run's configured limits, applied to this turn."""
    print_bytes: int
    """How much this turn may print: what earlier turns of the feed left of the feed's print cap."""
    code: str | None = None
    resume: TurnResume | None = None
    type_check_stubs: str | None = None
    """Type-check `code` against these stubs; only a fresh session's first snippet is type-checked."""


@dataclass
class TurnCall:
    """The external call a turn paused at."""

    function_name: str
    args: object
    """The positional arguments, as `to_wire` encodes them."""
    kwargs: object
    """The keyword arguments, as `to_wire` encodes them."""
    call_id: int
    is_os_function: bool


@dataclass
class MontyFailure:
    """What CodeMode reads from an error a Monty call raised, as data that can leave the worker process.

    Monty's error types can be neither constructed nor pickled, so a turn returns this in their place,
    and the code that turns a failure into a retry reads only this.
    """

    kind: Literal[
        'syntax', 'typing', 'runtime', 'crashed', 'disconnected', 'other', 'panic', 'not_stored', 'no_storage'
    ]
    """`not_stored`: the server refused to store the session; `display` says why. `no_storage`: the server
    stores no sessions at all, found before the turn ran any code."""
    display: str = ''
    """The full report: `display()` for a sandbox error, `str()` for any other."""
    message: str = ''
    """A runtime error's message alone (`display(format='msg')`)."""
    type_name: str = ''
    """A runtime error's sandbox exception type, or the type of an error that is not Monty's."""
    exception_types: tuple[str, ...] = ()
    """A runtime error's sandbox exception type and its bases, for `isinstance`-style checks."""
    has_traceback: bool = False
    timed_out: bool = False
    """Whether a crashed worker was killed for running past its time limit."""
    cause: str | None = None
    """Why the server dropped the connection, for a disconnect."""
    session_lost: bool = False
    message_sha256: str | None = None
    """The SHA-256 of `message` before `_fit_turn` cut it, so it still matches the error it came from."""
    """Whether the server could not keep the session a runtime error left, so the next turn starts fresh."""

    def carries(self, text: str) -> bool:
        """Whether `message` is `text`, or was before `_fit_turn` cut it."""
        if self.message_sha256 is None:
            return self.message == text
        return self.message_sha256 == _sha256(text)

    @classmethod
    def from_error(cls, error: BaseException) -> MontyFailure | None:
        """Describe `error`, or return `None` for an interruption, which must propagate unchanged."""
        match error:
            case MontySyntaxError():
                return cls('syntax', display=error.display())
            case MontyTypingError():
                return cls('typing', display=error.display())
            case MontyRuntimeError():
                exception = error.exception()
                return cls(
                    'runtime',
                    display=error.display(),
                    message=error.display(format='msg'),
                    type_name=type(exception).__name__,
                    exception_types=tuple(base.__name__ for base in type(exception).__mro__),
                    has_traceback=bool(error.traceback()),
                )
            case MontyCrashedError():
                return cls('crashed', timed_out=error.timed_out)
            case MontyDisconnectError():
                return cls('disconnected', cause=error.cause)
            case Exception():
                return cls('other', display=str(error), type_name=type(error).__name__)
            case _:
                # pyo3 raises a sandbox panic as a `BaseException` from a module that cannot be imported.
                return cls('panic') if type(error).__name__ == 'PanicException' else None


@dataclass
class MontyTurn:
    """How a turn ended, and the stored record to start the next turn from."""

    kind: Literal['call', 'waiting', 'complete', 'failed']
    state: str | None = None
    """The record of the session after this turn, base64-encoded; `None` when it could not be saved.

    Encoded here rather than typed `Base64Bytes`: Temporal's Pydantic converter serializes a value
    by its runtime types, so bytes would go out as UTF-8 text whatever this field declares."""
    prints: str = ''
    printed_bytes: int = 0
    """How much the snippet printed in this turn, counting any part `prints` leaves out."""
    suspensions: int = 0
    """How many times the snippet paused in this turn; each pause counts toward `max_suspensions`."""
    call: TurnCall | None = None
    pending_call_ids: list[int] = field(default_factory=list[int])
    output: object = None
    """The snippet's result, as `to_wire` encodes it."""
    failure: MontyFailure | None = None


_Snapshot = TypeVar('_Snapshot', bound=AsyncSnapshot | MontyComplete)


@dataclass
class _Pauses:
    """Counts the snapshots a turn pauses at: every pause, answered in the turn or not, is a suspension."""

    count: int = 0

    def seen(self, snapshot: _Snapshot) -> _Snapshot:
        if not isinstance(snapshot, MontyComplete):
            self.count += 1
        return snapshot


class MontyServerError(Exception):
    """A turn could not reach the Monty server, named by the type of the error it met.

    The error's own text is left out because it can carry the pool's URL or credentials.
    """


@cache
def redacted_error_type(type_name: str) -> type[MontyServerError]:
    """A `MontyServerError` subclass named `type_name`.

    Temporal records a failure's type as its class name, so a retry policy's
    `non_retryable_error_types` that names the original type still matches.
    """
    return type(type_name, (MontyServerError,), {})


TURN_PAYLOAD_BYTES = 1024 * 1024
"""The most a turn takes or returns, encoded: half Temporal's default 2 MiB payload limit.

The other half leaves room for the converter's envelope. Past the limit the activity result fails
to upload, while workflow code would hand the same text on for the model's tool output limits to cut.
"""

_NOTE_BYTES = 100
"""Room kept for the note that says how much of a text `_fit_turn` cut."""

_DUMP_REFUSED = "The Monty server could not store the session's state, for example because its snapshot quota is full."


async def run_monty_turn(
    pool: AsyncMontyRemote,
    request: TurnRequest,
    *,
    os_policy: OSPolicy,
    mount: MountDir | list[MountDir] | None,
) -> MontyTurn:
    """Run one turn on a session checked out of `pool`, and store the state it leaves.

    Monty errors, and a server that will not store the session, come back as a `failed` turn rather
    than raising, so Temporal neither retries a snippet that failed in the sandbox nor records the
    server's own text. Errors before the turn starts, such as a refused checkout, raise and are
    retried by the activity's retry policy: a retry starts again from `request.state`, which never
    changes.
    """
    capture = CollectString(max_bytes=request.print_bytes)
    async with pool.checkout(
        limits=request.limits,
        type_check=request.type_check_stubs is not None,
        type_check_stubs=request.type_check_stubs,
        os_policy=os_policy,
        ephemeral=False,
    ) as session:
        turn = await _run_turn(session, request, mount=mount, capture=capture)
    printed = capture.output
    return _fit_turn(replace(turn, prints=printed, printed_bytes=len(printed.encode())))


async def _run_turn(
    session: AsyncMontySession,
    request: TurnRequest,
    *,
    mount: MountDir | list[MountDir] | None,
    capture: CollectString,
) -> MontyTurn:
    if await _session_id(session) is None:
        # Before anything ran, CodeMode falls back to workflow code. Later, this checkout reached a
        # server that cannot have the run's record, so CodeMode reports it as a dropped session.
        return MontyTurn('failed', failure=MontyFailure('no_storage' if request.state is None else 'disconnected'))
    pauses = _Pauses()
    try:
        paused = await _load(session, request, mount=mount, capture=capture)
    except MontyRuntimeError:
        # The stored record could not be loaded, for example because the server let it expire.
        # Connection errors are other `MontyError`s and raise, so the activity retry loads it again.
        # CodeMode reports this as a dropped session.
        return MontyTurn('failed', failure=MontyFailure('disconnected'))
    try:
        if paused is None:
            assert request.code is not None, 'a turn that resumes nothing feeds code'
            snapshot = pauses.seen(
                await session.feed_start(
                    request.code,
                    print_callback=capture,
                    mount=mount,
                    skip_type_check=request.type_check_stubs is None,
                )
            )
        else:
            assert request.resume is not None, 'a turn that loads a snapshot resumes it'
            snapshot = pauses.seen(await _resume(paused, request.resume))
        snapshot = await _answer_in_turn(snapshot, pauses)
    except BaseException as error:
        failure = MontyFailure.from_error(error)
        if failure is None:  # pragma: no cover -- an activity cancelled while the snippet runs
            raise
        if failure.kind == 'other':
            # A remote error's text can carry the pool's URL or credentials, and this result is kept
            # in workflow history. CodeMode shows the model only the error's type.
            failure.display = ''
        return MontyTurn(
            'failed', state=await _state_after(session, request, failure), suspensions=pauses.count, failure=failure
        )
    try:
        state = await _dump(session)
    except MontyRuntimeError:
        return MontyTurn('failed', failure=MontyFailure('not_stored', display=_DUMP_REFUSED))
    except Exception:  # see `_dump`; CodeMode reports this as a dropped session
        return MontyTurn('failed', failure=MontyFailure('disconnected'))
    if isinstance(snapshot, MontyComplete):
        return MontyTurn('complete', state=state, suspensions=pauses.count, output=_output_to_wire(snapshot.output))
    if isinstance(snapshot, AsyncFunctionSnapshot):
        call = TurnCall(
            function_name=snapshot.function_name,
            args=to_wire(snapshot.args),
            kwargs=to_wire(snapshot.kwargs),
            call_id=snapshot.call_id,
            is_os_function=snapshot.is_os_function,
        )
        return MontyTurn('call', state=state, suspensions=pauses.count, call=call)
    return MontyTurn('waiting', state=state, suspensions=pauses.count, pending_call_ids=snapshot.pending_call_ids)


async def _state_after(session: AsyncMontySession, request: TurnRequest, failure: MontyFailure) -> str | None:
    """The record to continue from after `failure`, marking the failure when the session is lost.

    Only a syntax or runtime error leaves the session usable; CodeMode resets it after any other.
    Monty finishes the checkout when the print cap fires in a snippet with later calls, and Logfire
    Agent Sandboxes has refused to store a session a snippet error left, so there may be nothing to
    store. A syntax error ran nothing, so the turn's starting record then still holds.
    """
    if failure.kind not in ('syntax', 'runtime'):
        return None
    try:
        return await _dump(session)
    except Exception:  # see `_dump`
        if failure.kind == 'syntax':
            return request.state
        failure.session_lost = True
        return None


class TurnTooLarge(TypeError):
    """A value has more parts than a turn can carry."""


def oversized(request: TurnRequest) -> str | None:
    """Why `request` is more than a turn may take, or `None` when it fits."""
    try:
        size = len(to_json(request))
    except PydanticSerializationError:
        return 'the sandbox would receive a value nested more deeply than one step can carry'
    if size <= TURN_PAYLOAD_BYTES:
        return None
    return f'the sandbox would receive {size} bytes in one step, more than the {TURN_PAYLOAD_BYTES} it can take'


def _fit_turn(turn: MontyTurn) -> MontyTurn:
    """`turn`, its longest text cut, pass after pass, until the encoded turn fits `TURN_PAYLOAD_BYTES`.

    Measured on the encoded turn, so JSON escaping and error text count. A nested call's arguments
    are never cut: `_answer_in_turn` keeps them to half the budget, so the texts always hold the excess.
    """
    while (excess := len(to_json(turn)) - TURN_PAYLOAD_BYTES) > 0:
        output = turn.output if isinstance(turn.output, str) else repr(from_wire(turn.output))
        failure = turn.failure or MontyFailure('other')
        texts = {'prints': turn.prints, 'output': output, 'display': failure.display, 'message': failure.message}
        name = max(texts, key=lambda name: len(texts[name]))
        data = texts[name].encode()
        assert len(data) > _NOTE_BYTES, 'the excess is always in a text longer than the note'
        # At most half per pass, so a cut spreads over the texts rather than emptying the first.
        kept = max(len(data) - excess, len(data) // 2) - _NOTE_BYTES
        note = f'\n[{len(data) - kept} more bytes not shown: a turn returns at most {TURN_PAYLOAD_BYTES} bytes]'
        cut = data[:kept].decode(errors='ignore') + note
        if name == 'message' and failure.message_sha256 is None:
            turn = replace(turn, failure=replace(failure, message=cut, message_sha256=_sha256(failure.message)))
        elif name in ('display', 'message'):
            turn = replace(turn, failure=replace(failure, **{name: cut}))
        else:
            turn = replace(turn, **{name: cut})
    return turn


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode(errors='surrogatepass')).hexdigest()


async def _answer_in_turn(
    snapshot: AsyncSnapshot | MontyComplete, pauses: _Pauses
) -> AsyncFunctionSnapshot | AsyncFutureSnapshot | MontyComplete:
    """Answer the pauses CodeMode answers the same way every time, without leaving the turn.

    A name lookup is left undefined, so the sandbox raises `NameError`. An OS call other than a sleep
    is answered from the mounts, as CodeMode does without an `os_access` handler, which this path
    never has. Its arguments, such as a `PurePosixPath`, need not then cross the activity boundary.
    A call whose arguments `to_wire` cannot encode, or that take more than half of what a turn may
    return, cannot reach the tool, so it raises `TypeError` at the call site.
    """
    while True:
        if isinstance(snapshot, AsyncNameLookupSnapshot):
            snapshot = pauses.seen(await snapshot.resume())
        elif not isinstance(snapshot, AsyncFunctionSnapshot):
            return snapshot
        elif snapshot.is_os_function and snapshot.function_name not in SLEEP_FUNCTIONS:
            snapshot = pauses.seen(await snapshot.resume_auto())
        elif not _args_cross(snapshot):
            message = (
                f'{snapshot.function_name}() got arguments that cannot be sent to the tool: they need a JSON '
                f'form and at most {TURN_PAYLOAD_BYTES // 2} bytes'
            )
            snapshot = pauses.seen(await snapshot.resume({'exception': TypeError(message)}))
        else:
            return snapshot


def _args_cross(snapshot: AsyncFunctionSnapshot) -> bool:
    try:
        wired = to_wire((snapshot.args, snapshot.kwargs))
        return len(to_json(wired)) <= TURN_PAYLOAD_BYTES // 2
    except (TypeError, PydanticSerializationError):
        return False


async def _load(
    session: AsyncMontySession,
    request: TurnRequest,
    *,
    mount: MountDir | list[MountDir] | None,
    capture: CollectString,
) -> AsyncSnapshot | None:
    """Restore the record the turn starts from: the session to feed, or the snapshot to resume and return."""
    if request.state is None:
        return None
    state = base64.b64decode(request.state)
    if request.resume is None:
        await session.load_session(state)
        return None
    return await session.load_snapshot(state, mount=mount, print_callback=capture)


async def _session_id(session: AsyncMontySession) -> bytes | None:
    """The ID a storing server gave `session`, or `None` from a server that stores nothing.

    Monty 1.6 opens a remote session on its first request, so its `session_id` property reads `None`
    until then; `get_session_id()` opens the session to answer. Monty 1.5 has only the property, set
    on checkout.
    """
    get_session_id = getattr(session, 'get_session_id', None)
    if get_session_id is None:  # pragma: no cover -- Monty 1.5; the tests' storing server models 1.6
        return session.session_id
    session_id: bytes | None = await get_session_id()
    return session_id


async def _dump(session: AsyncMontySession) -> str:
    """Store the session and return its record ID.

    The snippet has run by then, so callers handle every error here, a dropped connection included,
    as an unsaved session: letting it escape would make Temporal retry the turn and run it again.
    """
    return base64.b64encode(await session.dump()).decode()


def _settled(data: SettledData) -> ExternalSettledResult:
    if 'return_value' in data:
        return ExternalReturnValue(return_value=from_wire(data['return_value']))
    return data


async def _resume(snapshot: AsyncSnapshot, resume: TurnResume) -> AsyncSnapshot:
    if isinstance(snapshot, AsyncFutureSnapshot):
        return await snapshot.resume({call_id: _settled(data) for call_id, data in resume.settled.items()})
    assert isinstance(snapshot, AsyncFunctionSnapshot), 'a turn only pauses at a call or at deferred calls'
    match resume.kind:
        case 'deferred':
            return await snapshot.resume({'future': ...})
        case _:
            assert resume.result is not None
            return await snapshot.resume(_settled(resume.result))
