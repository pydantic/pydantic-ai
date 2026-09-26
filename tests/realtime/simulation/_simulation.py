"""The simulation: one real `RealtimeSession` driven step by step against a simulated provider.

A `Simulation` owns the event loop, the agent (whose tools block on gates the trace opens), the session,
and the simulated server. Its public methods are the *steps* of a trace: client operations
(`send_text`, `send_audio`, `interrupt`, `wait_for_reply`, `finish_tool`, `close`, ...), provider
behavior (defined per protocol in the subclasses), and scheduling (`deliver`, `tick`, `settle`).

Every step is recorded in `trace` as the Python call that performed it, so a failing trace is also the
code that replays it:

```python
sim = OpenAISimulation()
sim.send_text()
sim.speak()
sim.finish()
sim.settle()
```

Each step takes `ticks`: how many event-loop iterations to run afterwards. `None` (the default) runs
until nothing is runnable, which is what hand-written scenarios want; a small number leaves tasks
mid-flight, which is how the randomized driver makes a send race a frame, or a tool finish while a
response is streaming. After every step the invariants that must hold at *any* point are checked;
`settle()` drives the server and the tools to a quiet state and checks the ones that must hold then.
"""

from __future__ import annotations as _annotations

import asyncio
import functools
import random
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Literal, ParamSpec, TypeVar

from typing_extensions import Self

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.messages import BinaryImage, ModelMessage
from pydantic_ai.realtime import RealtimeEvent, RealtimeModel, RealtimeSession
from pydantic_ai.realtime.settings import RealtimeModelSettings
from pydantic_ai.usage import UsageLimits

from . import _invariants
from ._invariants import SimulatedToolError
from ._loop import SimulatedLoop, SimulationStuck
from ._truth import GroundTruth

P = ParamSpec('P')
R = TypeVar('R')

ToolOutcome = Literal['ok', 'retry', 'error']
"""How a gated tool call settles: return a result, ask the model to retry, or raise (ending the session)."""


class InvariantViolation(AssertionError):
    """A property the session must always have did not hold. `code` names which one."""

    def __init__(self, code: str, detail: str, trace: list[str], context: dict[str, Any] | None = None) -> None:
        self.code = code
        self.detail = detail
        self.trace = list(trace)
        self.context = context or {}
        self.findings: list[str] = []
        """Ids of the known findings this violation matches, if any."""
        super().__init__(code, detail)

    def __str__(self) -> str:
        known = f' (known finding: {", ".join(self.findings)})' if self.findings else ''
        replay = '\n'.join(self.trace)
        return f'[{self.code}] {self.detail}{known}\n\nReplay:\n{replay}'


class FindingReproduced(InvariantViolation):
    """The violation a pinned scenario exists to reproduce: the known finding it enforces is still there."""

    def __init__(
        self, code: str, detail: str, trace: list[str], context: dict[str, Any] | None, *, findings: list[str]
    ) -> None:
        super().__init__(code, detail, trace, context)
        self.findings = findings


def step(method: Callable[P, R]) -> Callable[P, R]:
    """Record a public simulation method as a trace step, run its ticks, and check the invariants."""

    @functools.wraps(method)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        self = args[0]
        assert isinstance(self, Simulation)
        arguments = [repr(arg) for arg in args[1:]] + [f'{key}={value!r}' for key, value in kwargs.items()]
        self.trace.append(f'sim.{method.__name__}({", ".join(arguments)})')
        result = method(*args, **kwargs)
        self.check()
        return result

    return wrapper


@dataclass
class Operation:
    """A client operation the trace issued, and how it ended."""

    name: str
    key: str | None
    """The ground-truth key of what it sent, when it sent something the server can identify."""
    caller: str
    issued: int
    completed: int | None = None
    error: BaseException | None = None

    @property
    def done(self) -> bool:
        return self.completed is not None


@dataclass
class Waiter:
    """A `wait_for_reply()` call in flight."""

    index: int
    started: int
    returned: int | None = None
    error: BaseException | None = None
    snapshot: frozenset[str] = frozenset()
    """Keys of the operations that had completed when the wait began."""


class ToolGates:
    """Gated tool execution: each call blocks until the trace settles it."""

    def __init__(self) -> None:
        self.started: dict[str, asyncio.Future[ToolOutcome]] = {}
        self.order: list[str] = []
        self.settled: dict[str, ToolOutcome] = {}

    async def run(self, call_id: str) -> str:
        future = self.started[call_id] = asyncio.get_running_loop().create_future()
        self.order.append(call_id)
        outcome = await future
        if outcome == 'retry':
            raise ModelRetry(f'retry {call_id}')
        if outcome == 'error':
            raise SimulatedToolError(f'tool {call_id} crashed')
        return f'result:{call_id}'

    def pending(self) -> list[str]:
        return [call_id for call_id in self.order if not self.started[call_id].done()]

    def settle(self, call_id: str, outcome: ToolOutcome) -> None:
        self.settled[call_id] = outcome
        self.started[call_id].set_result(outcome)


@dataclass
class SessionOptions:
    """Session-level knobs a trace starts with."""

    handle_barge_in: bool = False
    request_limit: int | None = None
    latency: bool = False
    """Whether sends yield a random number of loop iterations (seeded) before reaching the server."""


class Simulation(ABC):
    """A real `RealtimeSession` against one simulated provider, driven one step at a time."""

    provider: str
    """Which provider the simulated server stands in for: known findings are scoped by it."""
    output_bytes_per_ms = 48
    """Model audio is 24 kHz PCM16 on every simulated provider."""

    def __init__(self, *, seed: int = 0, options: SessionOptions | None = None, strict: bool | None = None) -> None:
        from ._invariants import strict_from_environment

        self.options = options or SessionOptions()
        self.strict = strict_from_environment() if strict is None else strict
        """Whether a violation matching a known finding raises too, rather than being tolerated."""
        self.enforce: frozenset[str] = frozenset()
        """Ids of known findings that raise anyway: set by a pinned scenario for the finding it pins."""
        self.seed = seed
        self.rng = random.Random(seed)
        self.loop = SimulatedLoop()
        self.trace: list[str] = []
        self.tools = ToolGates()
        self.operations: list[Operation] = []
        self.waiters: list[Waiter] = []
        self.events: list[RealtimeEvent] = []
        self.consumer_error: BaseException | None = None
        self.close_error: BaseException | None = None
        self.close_requested: int | None = None
        self.closed = False
        self.session: RealtimeSession | None = None
        self._exit_stack = ExitStack()
        self._callers: dict[str, deque[tuple[Operation, Callable[[], Awaitable[object]]]]] = {}
        self._caller_wakeups: dict[str, asyncio.Event] = {}
        self._close_event: asyncio.Event | None = None
        self._play_permits = 0
        self._player_wakeup: asyncio.Event | None = None
        self.played_chunks = 0
        self._text_counter = 0
        self._image_counter = 0
        self._checker: _invariants.Checker | None = None
        self._view: AsyncIterator[bytes] | None = None

    # --- provider-specific hooks ------------------------------------------------------------------

    @property
    @abstractmethod
    def truth(self) -> GroundTruth: ...

    @abstractmethod
    def build_model(self) -> RealtimeModel: ...

    @abstractmethod
    def model_settings(self) -> RealtimeModelSettings: ...

    @abstractmethod
    @contextmanager
    def transport(self) -> Iterator[None]:
        """Patch the provider's transport so the model dials the simulated server."""
        ...

    @abstractmethod
    def drive_server_to_rest(self) -> bool:
        """Make one move that brings the server closer to quiet; `False` when it already is."""
        ...

    def before_settle_checks(self) -> None:
        """A hook for provider-specific settling."""

    def observe_connection(self, connection: Any) -> None:
        """A hook to record provider-internal facts that known findings are recognized by (never invariants)."""

    def observe_codec_event(self, event: Any) -> None:
        """A hook for a protocol whose response boundaries are inferred by the connection, not sent (GPT-Live)."""

    def expected_requests(self) -> int | None:
        """What `usage.requests` should be, if not one per recorded response (a model reporting requests with usage)."""
        return None

    @property
    @abstractmethod
    def failed_sends(self) -> list[tuple[str | None, str | None, str]]:
        """`(frame, last frame read, fault)` for every send a fault failed."""
        ...

    # --- lifecycle ------------------------------------------------------------------------------

    def start(self) -> Simulation:
        self._checker = _invariants.Checker(self, strict=self.strict, enforce=self.enforce)
        self._exit_stack.enter_context(self.transport())
        self._exit_stack.callback(self.loop.close)
        agent = self._build_agent()
        model = self.build_model()
        realtime = agent.realtime(
            model,
            model_settings=self.model_settings(),
            usage_limits=UsageLimits(request_limit=self.options.request_limit)
            if self.options.request_limit is not None
            else None,
        )
        ready = asyncio.Event()
        self._close_event = asyncio.Event()
        self._player_wakeup = asyncio.Event()

        async def host() -> None:
            assert self._close_event is not None
            try:
                async with realtime.session(handle_barge_in=self.options.handle_barge_in) as session:
                    self.session = session
                    self.checker.attach(session)
                    self.observe_connection(session._connection)  # pyright: ignore[reportPrivateUsage]
                    self._view = session.stream_audio()
                    ready.set()
                    await self._close_event.wait()
            except BaseException as e:  # the outcome is data the invariants read
                self.close_error = e
            finally:
                ready.set()
                self.closed = True

        self._host = self.loop.create_task(host())
        self.loop.run_until_idle()
        assert self.session is not None, f'the session never opened: {self.close_error!r}'
        self._consumer = self.loop.create_task(self._consume())
        self._player = self.loop.create_task(self._play())
        self.loop.run_until_idle()
        return self

    def _build_agent(self) -> Agent[None, str]:
        agent = Agent(instructions='You are a simulated voice assistant.')
        gates = self.tools

        @agent.tool
        async def lookup(ctx: RunContext[None]) -> str:
            """Look something up."""
            assert ctx.tool_call_id is not None
            return await gates.run(ctx.tool_call_id)

        return agent

    async def _consume(self) -> None:
        assert self.session is not None
        try:
            async for event in self.session:
                self.events.append(event)
        except BaseException as e:
            self.consumer_error = e
        finally:
            self.receive_ended = True

    receive_ended = False
    """Whether the session's event stream has ended: it reads nothing more from the provider.

    Usually with `consumer_error` set, but not when a send was the first to be told why it ended.
    """

    async def _play(self) -> None:
        assert self._player_wakeup is not None and self._view is not None
        view = self._view
        try:
            while True:
                while self._play_permits <= 0:
                    self._player_wakeup.clear()
                    await self._player_wakeup.wait()
                self._play_permits -= 1
                await anext(view)
                self.played_chunks += 1
        except StopAsyncIteration:
            return

    def close_simulation(self) -> None:
        """Tear the simulation down, whatever state it is in."""
        if self._close_event is not None and not self._close_event.is_set():
            self._close_event.set()
        try:
            for task in asyncio.all_tasks(self.loop):
                task.cancel()
            self.loop.run_until_complete(asyncio.sleep(0))
            if pending := [task for task in asyncio.all_tasks(self.loop) if not task.done()]:  # pragma: lax no cover
                self.loop.run_until_complete(asyncio.wait(pending))
        finally:
            self._exit_stack.close()

    def enforcing(self, *finding_ids: str) -> Self:
        """Raise `FindingReproduced` when these known findings are hit, instead of tolerating them."""
        self.enforce = frozenset(finding_ids)
        return self

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close_simulation()

    # --- scheduling -------------------------------------------------------------------------------

    def _advance(self, ticks: int | None) -> None:
        if ticks is None:
            self.loop.run_until_idle()
        else:
            self.loop.run_ticks(ticks)

    def _now(self) -> int:
        return self.truth.tick()

    @step
    def tick(self, ticks: int | None = None) -> None:
        """Let the event loop run."""
        self._advance(ticks)

    @step
    def advance_time(self, seconds: float, ticks: int | None = None) -> None:
        """Move the virtual clock forward, firing any timers that fall due."""
        self.loop.advance_clock(seconds)
        self._advance(ticks)

    # --- client operations ------------------------------------------------------------------------

    def _issue(
        self,
        name: str,
        make: Callable[[], Awaitable[object]],
        *,
        key: str | None = None,
        caller: str = 'main',
        ticks: int | None,
    ) -> Operation:
        operation = Operation(name=name, key=key, caller=caller, issued=self._now())
        self.operations.append(operation)
        queue = self._callers.get(caller)
        if queue is None:
            queue = self._callers[caller] = deque()
            self._caller_wakeups[caller] = asyncio.Event()
            self.loop.create_task(self._run_caller(caller))
        queue.append((operation, make))
        self._caller_wakeups[caller].set()
        self._advance(ticks)
        return operation

    async def _run_caller(self, caller: str) -> None:
        queue = self._callers[caller]
        wakeup = self._caller_wakeups[caller]
        while True:
            while not queue:
                wakeup.clear()
                await wakeup.wait()
            operation, make = queue.popleft()
            try:
                await make()
            except BaseException as e:  # recorded; the invariants judge it
                operation.error = e
                if isinstance(e, asyncio.CancelledError):  # pragma: lax no cover (torn down mid-operation)
                    raise
            finally:
                operation.completed = self._now()

    def _require_session(self) -> RealtimeSession:
        assert self.session is not None
        return self.session

    @step
    def send_text(self, respond: bool = True, ticks: int | None = None) -> str:
        """Send a text turn (`t1`, `t2`, ...), soliciting a reply unless `respond=False`."""
        self._text_counter += 1
        text = f't{self._text_counter}'
        session = self._require_session()
        self._issue(
            'send_text' if respond else 'send_context',
            lambda: session.send(text, respond=respond),
            key=text,
            ticks=ticks,
        )
        return text

    @step
    def send_image(self, respond: bool = False, ticks: int | None = None) -> str:
        """Send a (tiny, unique) image, as context unless `respond=True`."""
        self._image_counter += 1
        key = f'img{self._image_counter:08d}'
        image = BinaryImage(data=b'\x89PNG\r\n\x1a\n' + key.encode(), media_type='image/png')
        session = self._require_session()
        self._issue(
            'send_image_respond' if respond else 'send_image',
            lambda: session.send(image, respond=respond),
            key=image.base64[-12:],
            ticks=ticks,
        )
        return key

    @step
    def send_audio(self, chunks: int = 1, voiced: bool = True, ticks: int | None = None) -> None:
        """Stream `chunks` × 100 ms of microphone audio from the microphone task."""
        session = self._require_session()
        chunk = self.input_audio_chunk(voiced=voiced)

        async def stream() -> None:
            for _ in range(chunks):
                await session.send_audio(chunk)

        self._issue('send_audio', stream, caller='mic', ticks=ticks)

    def input_audio_chunk(self, *, voiced: bool) -> bytes:
        rate = self._require_session().audio_input_sample_rate
        samples = rate // 10
        return (b'\x00\x10' if voiced else b'\x00\x00') * samples

    @step
    def commit_audio(self, ticks: int | None = None) -> None:
        session = self._require_session()
        self._issue('commit_audio', session.commit_audio, ticks=ticks)

    @step
    def clear_audio(self, ticks: int | None = None) -> None:
        session = self._require_session()
        self._issue('clear_audio', session.clear_audio, ticks=ticks)

    @step
    def create_response(self, ticks: int | None = None) -> None:
        session = self._require_session()
        self._issue('create_response', session.create_response, ticks=ticks)

    @step
    def interrupt(
        self, mode: Literal['cancel', 'played_ms', 'played_bytes'] = 'played_bytes', ticks: int | None = None
    ) -> None:
        """Barge in: cancel, truncate at a playback position, or let the session map the played bytes."""
        session = self._require_session()

        async def interrupt() -> None:
            if mode == 'played_bytes':
                await session.interrupt(played_bytes=session.played_audio_bytes)
            elif mode == 'played_ms':
                await session.interrupt(played_ms=self.played_chunks * 100)
            else:
                await session.interrupt()

        self._issue(f'interrupt_{mode}', interrupt, ticks=ticks)

    @step
    def play(self, chunks: int = 1, ticks: int | None = None) -> None:
        """Let the speaker pull `chunks` more chunks from `stream_audio()`."""
        assert self._player_wakeup is not None
        self._play_permits += chunks
        self._player_wakeup.set()
        self._advance(ticks)

    @step
    def wait_for_reply(self, ticks: int | None = None) -> Waiter:
        """Start a `wait_for_reply()` in a task of its own; the invariants judge when it returns."""
        session = self._require_session()
        waiter = Waiter(
            index=len(self.waiters),
            started=self._now(),
            snapshot=frozenset(op.key for op in self.operations if op.done and op.key is not None and op.error is None),
        )
        self.waiters.append(waiter)

        async def wait() -> None:
            try:
                await session.wait_for_reply()
            except BaseException as e:  # pragma: lax no cover (the session ended under the wait)
                waiter.error = e
            finally:
                waiter.returned = self._now()

        self.loop.create_task(wait())
        self._advance(ticks)
        return waiter

    @step
    def finish_tool(self, call: int = 0, outcome: ToolOutcome = 'ok', ticks: int | None = None) -> str:
        """Settle the `call`-th still-pending tool call (in start order)."""
        pending = self.tools.pending()
        assert pending, 'no tool call is waiting to be finished'
        call_id = pending[call % len(pending)]
        self.tools.settle(call_id, outcome)
        self._advance(ticks)
        return call_id

    @step
    def close(self, ticks: int | None = None) -> None:
        """Hang up: leave the session's `async with`."""
        assert self._close_event is not None
        if self.close_requested is None:
            self.close_requested = self._now()
        self._close_event.set()
        self._advance(ticks)

    # --- settling ---------------------------------------------------------------------------------

    @step
    def settle(self) -> None:
        """Bring everything to rest, then check the invariants that must hold at rest.

        Pending tool calls finish successfully, the server finishes and delivers whatever it has in
        progress, virtual time moves on to any timer still pending, and the loop runs until nothing is
        runnable.
        """
        for _ in range(500):
            self.loop.run_until_idle()
            if (pending := self.tools.pending()) and not self.closed:
                self.tools.settle(pending[0], 'ok')
                continue
            if not self.closed and not self.receive_ended and self.drive_server_to_rest():
                # (A session that ended on an error reads nothing more, so there is no server left to drive.)
                continue
            if (when := self.loop.next_timer()) is not None:
                # Nothing else will happen until a timer fires (a reconnect's backoff, a turn clock).
                self.loop.advance_clock(max(0.0, when - self.loop.time()))
                continue
            break
        else:  # pragma: no cover
            raise SimulationStuck('the simulation never came to rest')
        self.loop.run_until_idle()
        self.before_settle_checks()
        self.checker.check_at_rest()

    @property
    def checker(self) -> _invariants.Checker:
        assert self._checker is not None, 'the simulation has not started'
        return self._checker

    def check(self) -> None:
        self.checker.check_step()

    @step
    def check_handoff(self) -> None:
        """Hand the history to a standard agent run, as an app handing a call over to text would."""
        from pydantic_ai.messages import ModelResponse, TextPart
        from pydantic_ai.models.function import AgentInfo, FunctionModel

        async def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart('Continuing in text.')])

        messages = self.history()
        agent: Agent[None, str] = Agent(FunctionModel(reply))

        async def handoff() -> None:
            await agent.run('Continue in text.', message_history=messages)

        try:
            self.loop.run_until_complete(handoff())
        except Exception as e:  # pragma: no cover (only when a standard run rejects the history)
            self.checker.fail('history.handoff', f'a standard run rejected the history: {e!r}')

    # --- views for the invariants -------------------------------------------------------------------

    def history(self) -> list[ModelMessage]:
        return self._require_session().all_messages()
