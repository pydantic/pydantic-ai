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
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

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

SettledData = ExternalReturnValue | ExternalExceptionData
"""A settled call result as plain data, which a turn in another process can resume the sandbox with."""

SLEEP_FUNCTIONS = frozenset({'time.sleep', 'asyncio.sleep', 'system.sleep', 'system.async_sleep'})
"""The OS calls CodeMode answers on the host, charging the wait to the snippet's sleep allowance."""

_EXCEPTION_TYPES = {name.rpartition('.')[2]: name for name in get_args(ExcType)}


def settled_data(result: ExternalSettledResult) -> SettledData:
    """`result` with any exception instance replaced by the type and message Monty gives the sandbox for it.

    Monty raises a host exception in the sandbox as the nearest type it knows, with the exception's only
    string argument as the message, or `str()` of the exception otherwise. Dotted names such as
    `json.JSONDecodeError` match only classes from that module, so a user's own `Error` class is not
    read as `binascii.Error`.
    """
    if 'exception' not in result:
        return result
    exc = result['exception']
    # `BaseException` is one of Monty's types, so every exception's MRO reaches a match.
    exc_type: ExcType = next(
        name
        for base in type(exc).__mro__
        if (name := _EXCEPTION_TYPES.get(base.__name__)) is not None
        and base.__module__.partition('.')[0] in ('builtins', name.partition('.')[0])
    )
    message = exc.args[0] if len(exc.args) == 1 and isinstance(exc.args[0], str) else str(exc)
    return ExternalExceptionData(exc_type=exc_type, message=message)


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
    code: str | None = None
    resume: TurnResume | None = None
    type_check_stubs: str | None = None
    """Type-check `code` against these stubs; only a fresh session's first snippet is type-checked."""


@dataclass
class TurnCall:
    """The external call a turn paused at."""

    function_name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    call_id: int
    is_os_function: bool


@dataclass
class MontyFailure:
    """What CodeMode reads from an error a Monty call raised, as data that can leave the worker process.

    Monty's error types can be neither constructed nor pickled, so a turn returns this in their place,
    and the code that turns a failure into a retry reads only this.
    """

    kind: Literal['syntax', 'typing', 'runtime', 'crashed', 'disconnected', 'other', 'panic']
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

    kind: Literal['call', 'waiting', 'complete', 'failed', 'not_stored']
    """`not_stored`: the server stores no sessions, so the turn ran nothing."""
    state: str | None = None
    """The record of the session after this turn, base64-encoded; `None` when it could not be saved.

    Encoded here rather than typed `Base64Bytes`: Temporal's Pydantic converter serializes a value
    by its runtime types, so bytes would go out as UTF-8 text whatever this field declares."""
    prints: str = ''
    call: TurnCall | None = None
    pending_call_ids: list[int] = field(default_factory=list[int])
    output: Any = None
    failure: MontyFailure | None = None


async def run_monty_turn(
    pool: AsyncMontyRemote,
    request: TurnRequest,
    *,
    limits: ResourceLimits,
    os_policy: OSPolicy,
    mount: MountDir | list[MountDir] | None,
) -> MontyTurn:
    """Run one turn on a session checked out of `pool`, and store the state it leaves.

    Monty errors come back as a `failed` turn rather than raising, so Temporal does not retry a
    snippet that failed in the sandbox. Errors before the turn starts, such as a refused checkout,
    raise and are retried by the activity's retry policy: a retry starts again from `request.state`,
    which never changes.
    """
    type_check = request.type_check_stubs is not None
    async with pool.checkout(
        limits=limits, type_check=type_check, type_check_stubs=request.type_check_stubs, os_policy=os_policy
    ) as session:
        if session.session_id is None:
            return MontyTurn('not_stored')
        capture = CollectString()
        try:
            if request.resume is None:
                if request.state is not None:
                    await session.load_session(base64.b64decode(request.state))
                assert request.code is not None, 'a turn that resumes nothing feeds code'
                snapshot = await session.feed_start(
                    request.code, print_callback=capture, mount=mount, skip_type_check=not type_check
                )
            else:
                assert request.state is not None, 'a resumed turn starts from the snapshot it paused at'
                paused = await session.load_snapshot(
                    base64.b64decode(request.state), mount=mount, print_callback=capture
                )
                snapshot = await _resume(paused, request.resume)
            snapshot = await _answer_in_turn(snapshot)
        except BaseException as error:
            failure = MontyFailure.from_error(error)
            if failure is None:  # pragma: no cover -- an activity cancelled while the snippet runs
                raise
            # Only these leave the session usable; the caller resets it after any other failure.
            state = await _dump(session) if failure.kind in ('syntax', 'runtime') else None
            return MontyTurn('failed', state=state, prints=capture.output, failure=failure)
        state = await _dump(session)
        prints = capture.output
        if isinstance(snapshot, MontyComplete):
            return MontyTurn('complete', state=state, prints=prints, output=snapshot.output)
        if isinstance(snapshot, AsyncFunctionSnapshot):
            call = TurnCall(
                function_name=snapshot.function_name,
                args=snapshot.args,
                kwargs=snapshot.kwargs,
                call_id=snapshot.call_id,
                is_os_function=snapshot.is_os_function,
            )
            return MontyTurn('call', state=state, prints=prints, call=call)
        return MontyTurn('waiting', state=state, prints=prints, pending_call_ids=snapshot.pending_call_ids)


async def _answer_in_turn(
    snapshot: AsyncSnapshot | MontyComplete,
) -> AsyncFunctionSnapshot | AsyncFutureSnapshot | MontyComplete:
    """Answer the pauses CodeMode answers the same way every time, without leaving the turn.

    A name lookup is left undefined, so the sandbox raises `NameError`. An OS call other than a sleep
    is answered from the mounts, as CodeMode does without an `os_access` handler, which this path
    never has. Its arguments, such as a `PurePosixPath`, need not then cross the activity boundary.
    """
    while True:
        if isinstance(snapshot, AsyncNameLookupSnapshot):
            snapshot = await snapshot.resume()
        elif (
            isinstance(snapshot, AsyncFunctionSnapshot)
            and snapshot.is_os_function
            and snapshot.function_name not in SLEEP_FUNCTIONS
        ):
            snapshot = await snapshot.resume_auto()
        else:
            return snapshot


async def _dump(session: AsyncMontySession) -> str:
    return base64.b64encode(await session.dump()).decode()


async def _resume(snapshot: AsyncSnapshot, resume: TurnResume) -> AsyncSnapshot:
    if isinstance(snapshot, AsyncFutureSnapshot):
        return await snapshot.resume(dict[int, ExternalSettledResult](resume.settled))
    assert isinstance(snapshot, AsyncFunctionSnapshot), 'a turn only pauses at a call or at deferred calls'
    match resume.kind:
        case 'deferred':
            return await snapshot.resume({'future': ...})
        case _:
            assert resume.result is not None
            return await snapshot.resume(resume.result)
