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
import io
import math
import pickle
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from functools import cache
from types import EllipsisType, NotImplementedType
from typing import Literal, assert_never, get_args

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
from pydantic_ai_harness._typeguards import (
    is_mapping,
    is_object_dict,
    is_object_frozenset,
    is_object_list,
    is_object_list_or_tuple,
    is_object_set,
    is_object_tuple,
)

SettledData = ExternalReturnValue | ExternalExceptionData
"""A settled call result as plain data, which a turn in another process can resume the sandbox with."""

SLEEP_FUNCTIONS = frozenset({'time.sleep', 'asyncio.sleep', 'system.sleep', 'system.async_sleep'})
"""The OS calls CodeMode answers on the host, charging the wait to the snippet's sleep allowance."""

_EXCEPTION_TYPES: dict[str, list[ExcType]] = {}
for _name in get_args(ExcType):
    _EXCEPTION_TYPES.setdefault(_name.rpartition('.')[2], []).append(_name)


_UNDOTTED_MODULES = {'FrozenInstanceError': 'dataclasses'}
"""The module of each exception Monty names without one that is not a builtin."""


def to_wire(value: object) -> str:
    """`value` as text that `from_wire` turns back into an equal value of the same types.

    Temporal's converter writes activity inputs and results as JSON, which would turn bytes into text,
    tuples into lists, and has no form for sets, dates, `Decimal`, `complex` or `...`; its decoder also
    refuses deep nesting and very large integers. So a value crosses as one string: its pickle, which
    keeps every type Monty hands the host and the values a snippet shares, in base64.

    The pickle names only the classes in `_WIRE_CLASSES`, and `from_wire` loads only those, so loading
    a recorded turn never imports or calls anything else. A subclass of a container crosses as the
    container, as Monty itself hands it to the host, and a named tuple as a plain tuple. Raises
    `TypeError` for any other type, with the message a live session gives for a host value it cannot
    convert, and for a value nested too deeply to write.
    """
    buffer = io.BytesIO()
    try:
        _WirePickler(buffer, protocol=5).dump(value)
    except RecursionError:
        raise TypeError('the value is nested too deeply to send') from None
    return base64.b64encode(buffer.getvalue()).decode()


def from_wire(data: str) -> object:
    """The value `to_wire` encoded as `data`."""
    return _WireUnpickler(io.BytesIO(base64.b64decode(data))).load()


_WIRE_CLASSES: dict[tuple[str, str], type] = {
    (cls.__module__, cls.__qualname__): cls
    for cls in (
        tuple,
        list,
        dict,
        set,
        frozenset,
        int,
        float,
        str,
        bytes,
        complex,
        type(...),
        date,
        datetime,
        time,
        timedelta,
        timezone,
        Decimal,
    )
}
"""The classes a wire value may name, by module and name. `...` pickles as the name `builtins.Ellipsis`."""


class _WirePickler(pickle.Pickler):
    def reducer_override(self, obj: object, /) -> tuple[type, tuple[object]] | NotImplementedType:
        # Called for every value except `None`, `bool`, and exact `int`, `float`, `str`, `bytes`,
        # `bytearray` and containers, and for each class the pickle names.
        if type(obj) in _WIRE_CLASSES.values() or obj in _WIRE_CLASSES.values():
            return NotImplemented
        if is_object_tuple(obj):
            return tuple, (tuple(obj),)
        if is_object_list(obj):
            return list, (list(obj),)
        if is_object_dict(obj):
            return dict, (dict(obj),)
        if is_object_frozenset(obj):
            return frozenset, (frozenset(obj),)
        if is_object_set(obj):
            return set, (set(obj),)
        # A live session takes an `IntEnum` or `StrEnum` as the plain value, so the wire does too.
        if isinstance(obj, int):
            return int, (int(obj),)
        if isinstance(obj, str):
            return str, (str.__str__(obj),)
        if isinstance(obj, float):
            return float, (float(obj),)
        if isinstance(obj, bytes):
            return bytes, (bytes(obj),)
        if is_mapping(obj):
            return dict, (dict(obj),)
        cls = type(obj)
        raise TypeError(f'Cannot convert {cls.__module__}.{cls.__qualname__} to Monty value')


class _WireUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> type | EllipsisType:
        if (module, name) == ('builtins', 'Ellipsis'):
            return ...
        if (cls := _WIRE_CLASSES.get((module, name))) is None:
            raise pickle.UnpicklingError(f'{module}.{name} is not a wire type')
        return cls


def _output_to_wire(output: object) -> str:
    """A snippet's result as a turn returns it; one that cannot cross returns as its `repr`."""
    try:
        safe = model_safe_result(output)
        try:
            return to_wire(safe)
        # No value Monty returns has a JSON form but no wire form; this keeps a new one from failing the turn
        # after the snippet ran.
        except TypeError:  # pragma: no cover
            return to_wire(repr(safe))
    except RecursionError:
        # The snippet already ran, so failing here would let the activity retry run it again.
        return to_wire('[the result is nested too deeply to show]')


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
        if base.__module__.partition('.')[0] in ('builtins', _UNDOTTED_MODULES.get(name, name.partition('.')[0]))
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
    if is_object_list_or_tuple(value):
        items = [model_safe_result(item) for item in value]
        return tuple(items) if isinstance(value, tuple) else items
    return value if _serializes(value) else repr(value)


def _model_safe_mapping(value: Mapping[object, object]) -> object:
    """`model_safe_result` for a mapping, rendering each key that has no JSON form as its `repr`."""
    rendered = {key if _serializes({key: None}) else repr(key): model_safe_result(item) for key, item in value.items()}
    # A rendered key can equal an existing one (`{int: 1, "<class 'int'>": 2}`), and
    # the dict would silently drop an entry; the whole mapping's `repr` loses nothing.
    return rendered if len(rendered) == len(value) else repr(value)


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
    """The run's limits, with `max_suspensions` lowered by what earlier turns of the feed used."""
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
    args: str
    """The positional arguments, as `to_wire` encodes them."""
    kwargs: str
    """The keyword arguments, as `to_wire` encodes them."""
    call_id: int
    is_os_function: bool


@dataclass
class MontyFailure:
    """What CodeMode reads from an error a Monty call raised, as data that can leave the worker process.

    Monty's error types can be neither constructed nor pickled, so a turn returns this in their place,
    and the code that turns a failure into a retry reads only this.
    """

    kind: Literal['syntax', 'typing', 'runtime', 'crashed', 'disconnected', 'other', 'panic', 'not_stored']
    """`not_stored`: the server refused to store the session; `display` says why."""
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
    """Whether the server could not keep the session a runtime error left, so the next turn starts fresh."""

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
                return cls('panic') if is_sandbox_panic(error) else None


def is_sandbox_panic(exc: BaseException) -> bool:
    """Whether `exc` is a Rust-side sandbox panic surfacing through pyo3.

    pyo3 raises `pyo3_runtime.PanicException`, a `BaseException` (not `Exception`) subclass
    from a module that cannot be imported, so it is matched by name. A panic can surface from
    monty's host-side bindings, so callers should convert it to a retry rather than let it
    tear down the whole agent run.
    """
    return type(exc).__name__ == 'PanicException'


@dataclass
class MontyTurn:
    """How a turn ended, and the stored record to start the next turn from."""

    kind: Literal['call', 'waiting', 'complete', 'failed', 'no_storage']
    """`no_storage`: the server stores no sessions at all, found before the turn ran any code."""
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
    output: str | None = None
    """The snippet's result, as `to_wire` encodes it."""
    failure: MontyFailure | None = None


@dataclass
class _Pauses:
    """Counts the snapshots a turn pauses at: every pause, answered in the turn or not, is a suspension."""

    count: int = 0

    def seen(self, snapshot: AsyncSnapshot | MontyComplete) -> AsyncSnapshot | MontyComplete:
        if not isinstance(snapshot, MontyComplete):
            self.count += 1
        return snapshot


@cache
def redacted_error_type(type_name: str) -> type[Exception]:
    """An exception class named `type_name`, for a turn that could not reach the Monty server.

    Temporal records a failure's type as its class name, so a retry policy's
    `non_retryable_error_types` that names the original type still matches. The original error's
    text is left out because it can carry the pool's URL or credentials.
    """
    return type(type_name, (Exception,), {})


TURN_PAYLOAD_BYTES = 1024 * 1024
"""The most a turn takes or returns, encoded: half Temporal's default 2 MiB payload limit.

The other half leaves room for the converter's envelope. Past the limit the activity result fails
to upload, while workflow code would hand the same text on for the model's tool output limits to cut.
"""

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
        if request.state is None:
            return MontyTurn('no_storage')
        return MontyTurn('failed', failure=MontyFailure('disconnected'))
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
        # An activity cancelled while the snippet runs.
        if failure is None:  # pragma: no cover
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


def oversized(request: TurnRequest) -> str | None:
    """Why `request` is more than a turn may take, or `None` when it fits."""
    size = len(to_json(request))
    if size <= TURN_PAYLOAD_BYTES:
        return None
    return f'the sandbox would receive {size} bytes in one step, more than the {TURN_PAYLOAD_BYTES} it can take'


def _fit_turn(turn: MontyTurn) -> MontyTurn:
    """`turn`, its longest text halved, pass after pass, until the encoded turn fits `TURN_PAYLOAD_BYTES`.

    Measured on the encoded turn, so JSON escaping and error text count. A result picked to cut first
    becomes the text the model would read; a result that fits keeps its types. A nested call's arguments are never cut:
    `_answer_in_turn` keeps them to half the budget, so the texts always hold the excess.
    """
    shown: str | None = None
    while len(to_json(turn)) > TURN_PAYLOAD_BYTES:
        failure = turn.failure or MontyFailure('other')
        output = shown if shown is not None else turn.output or ''  # an unrendered result counts at its wire size
        texts: dict[str, str] = {
            'prints': turn.prints,
            'output': output,
            'display': failure.display,
            'message': failure.message,
        }
        name = _longest(texts)
        if name == 'output' and shown is None:
            assert turn.output is not None
            shown = _shown(from_wire(turn.output))
            turn = replace(turn, output=to_wire(shown))
            continue
        text = texts[name]
        kept = len(text) // 2
        assert kept > 100, 'the excess is always in a text longer than the note'
        cut = f'{text[:kept]}\n[the rest is not shown: a turn returns at most {TURN_PAYLOAD_BYTES} bytes]'
        if name == 'output':
            shown = cut
            turn = replace(turn, output=to_wire(cut))
        elif name == 'prints':
            turn = replace(turn, prints=cut)
        else:
            turn = replace(turn, failure=replace(failure, **{name: cut}))
    return turn


def _longest(texts: dict[str, str]) -> str:
    return max(texts, key=lambda key: len(texts[key]))


def _shown(value: object) -> str:
    """`value` as the model would read it: a string as itself, anything else as its `repr`."""
    if isinstance(value, str):
        return value
    try:
        return repr(value)
    except (ValueError, RecursionError):  # too many digits to convert to text, or nested too deeply
        return f'[a {type(value).__name__} too large to show]'


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
                f'{snapshot.function_name}() got arguments that cannot be sent to the tool: they need types '
                f'the sandbox can hand the host and at most {TURN_PAYLOAD_BYTES // 2} bytes'
            )
            snapshot = pauses.seen(await snapshot.resume({'exception': TypeError(message)}))
        else:
            return snapshot


def _args_cross(snapshot: AsyncFunctionSnapshot) -> bool:
    try:
        # Encoded as `TurnCall` encodes them, so a value both share is counted twice, as it is sent.
        return len(to_wire(snapshot.args)) + len(to_wire(snapshot.kwargs)) <= TURN_PAYLOAD_BYTES // 2
    except TypeError:
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
    # Monty 1.5; the tests' storing server models 1.6.
    if get_session_id is None:  # pragma: no cover
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
        case 'call':
            assert resume.result is not None
            return await snapshot.resume(_settled(resume.result))
        # Settled results resume deferred calls, which pause at a future snapshot, handled above.
        case 'settled':  # pragma: no cover
            raise AssertionError('settled results cannot resume a call')
        case _:
            assert_never(resume.kind)
