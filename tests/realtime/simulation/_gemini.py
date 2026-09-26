"""The simulated Gemini Live server, its `google-genai` session fake, and the simulation over them.

Gemini Live is reached through the `google-genai` SDK rather than a raw WebSocket, so the fake sits one
level up: `client.aio.live.connect` hands out a `FakeGeminiSession` that the real
`GoogleRealtimeConnection` drives through the SDK's own `send_*` methods and `receive()` (which, like the
SDK's, yields one model turn and then returns). Server messages are real `LiveServerMessage`s.

The server model encodes what the stress runs and the recorded cassettes show Gemini doing, which is
quite unlike the OpenAI protocol:

- no response ids and no response-start event: a model turn is only known from its first output;
- replies are automatic: a typed turn (`turn_complete=True`), the end of the user's speech, and the last
  result of a batch of tool calls each make the model speak, with no request frame;
- one `tool_call` message can carry several calls; the model answers the whole batch once, in the same
  turn, after the last result arrives;
- usage is reported only with `turn_complete`, covering the whole turn (tool call and answer);
- some models close the tool-call turn with a `turn_complete` of its own before the answer (Vertex
  `gemini-live-2.5-flash`), and the extended-thinking model ends a filler turn with
  `interaction_status=IN_PROGRESS` before calling the tool it was stalling for;
- the user's speech interrupts the model (`interrupted`, then `turn_complete`) and cancels its pending
  tool calls (`tool_call_cancellation`);
- the user's words stream as input transcript fragments, usually without an explicit end;
- resumption handles arrive on the server's schedule (at turn start on 3.x, after a turn on 2.5); a
  resumed session knows only what it knew when that handle was issued, so a tool call made after it is
  forgotten and its result silently ignored.
"""

from __future__ import annotations as _annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal
from unittest import mock

from google.genai import types as gt
from google.genai.live import ConnectionClosed
from hypothesis import strategies as st
from hypothesis.stateful import precondition, rule
from websockets.frames import Close

from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.realtime import RealtimeModel
from pydantic_ai.realtime.google import GoogleRealtimeModel, GoogleRealtimeModelSettings
from pydantic_ai.realtime.settings import RealtimeModelSettings

from ._machine import TICKS, SessionMachine
from ._simulation import SessionOptions, Simulation, step
from ._truth import GroundTruth, ToolCallTruth, TruthResponse
from ._wire import SendFault

AUDIO_CHUNK = b'\x00\x10' * 2400
"""100 ms of 24 kHz PCM16 output audio."""


@dataclass
class _Closed:
    pass


@dataclass(eq=False)
class FakeGeminiSession:
    """One simulated Live connection, standing in for `google.genai.live.AsyncSession`."""

    server: GeminiServer
    index: int
    handle: str | None
    inbox: deque[gt.LiveServerMessage | _Closed] = field(default_factory=deque[gt.LiveServerMessage | _Closed])
    in_flight: deque[gt.LiveServerMessage] = field(default_factory=deque[gt.LiveServerMessage])
    broken: bool = False
    closed_by_client: bool = False
    _readable: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def alive(self) -> bool:
        return not self.broken and not self.closed_by_client

    # --- the SDK surface the connection uses -------------------------------------------------------

    async def _outbound(self, kind: str, payload: Any) -> None:
        for _ in range(self.server.latency()):
            await asyncio.sleep(0)
        if not self.alive:
            raise ConnectionClosed(None, Close(1006, 'simulated drop'))
        fault = self.server.send_faults.popleft() if self.server.send_faults else None
        if fault is not None:
            self.server.failed_sends.append((kind, None, fault))
        if fault == 'lost':
            self.break_connection()
            raise ConnectionClosed(None, Close(1006, 'simulated send failure'))
        self.server.on_client_message(self, kind, payload)
        if fault == 'ambiguous':
            self.break_connection()
            raise ConnectionClosed(None, Close(1006, 'simulated send failure'))

    async def send_realtime_input(self, **kwargs: Any) -> None:
        await self._outbound('realtime', kwargs)

    async def send_client_content(self, *, turns: Any = None, turn_complete: bool = True) -> None:
        await self._outbound('client_content', (turns, turn_complete))

    async def send_tool_response(self, *, function_responses: Any) -> None:
        await self._outbound('tool_response', function_responses)

    async def receive(self) -> AsyncIterator[gt.LiveServerMessage]:
        while True:
            while not self.inbox:
                self._readable.clear()
                await self._readable.wait()
            item = self.inbox[0]
            if isinstance(item, _Closed):
                raise ConnectionClosed(Close(1006, 'simulated drop'), None)
            self.inbox.popleft()
            self.server.on_client_read(self, item)
            yield item
            content = item.server_content
            if (
                content is not None
                and content.turn_complete
                and content.interaction_status != gt.InteractionStatus.IN_PROGRESS
            ):
                return

    async def close(self) -> None:
        if self.alive:
            self.closed_by_client = True
            self.in_flight.clear()
            self._push(_Closed())

    # --- server side --------------------------------------------------------------------------------

    def emit(self, message: gt.LiveServerMessage) -> None:
        if self.alive:  # pragma: lax no cover (a race: the server answers a message on a socket that just dropped)
            self.in_flight.append(message)

    def deliver(self, count: int | None = None) -> None:
        moved = 0
        while self.in_flight and (count is None or moved < count):
            self._push(self.in_flight.popleft())
            moved += 1

    def break_connection(self) -> None:
        if not self.alive:  # pragma: lax no cover (a drop racing the client's own close)
            return
        self.broken = True
        self.in_flight.clear()
        self._push(_Closed())
        self.server.on_disconnect(self)

    def _push(self, item: gt.LiveServerMessage | _Closed) -> None:
        self.inbox.append(item)
        self._readable.set()


@dataclass
class GeminiBehavior:
    """Which Gemini Live model family the simulated server behaves like."""

    closes_tool_turn_separately: bool = False
    """Send a `turn_complete` when the last tool result arrives, before the answer (Vertex `gemini-live-2.5-flash`)."""
    handles_at_turn_start: bool = False
    """Issue resumption handles as a turn starts (3.x) rather than after it ends (2.5)."""
    stalls_in_progress: bool = False
    """End a filler turn `IN_PROGRESS` before calling a tool (`gemini-3.8-live-extended-thinking`)."""
    input_transcription: bool = True

    @property
    def model(self) -> str:
        if self.stalls_in_progress:
            return 'gemini-3.8-live-extended-thinking'
        if self.handles_at_turn_start:
            return 'gemini-3.8-live'
        return 'gemini-2.5-flash-native-audio-latest'


@dataclass
class _Turn:
    """A model turn in progress: one or more responses (a tool call, then its answer) up to `turn_complete`."""

    response: TruthResponse | None
    responses: list[TruthResponse] = field(default_factory=list[TruthResponse])
    awaiting: set[str] = field(default_factory=set[str])
    """Tool calls whose results the model is waiting for."""
    outputs: list[str] = field(default_factory=list[str])
    stalled: TruthResponse | None = None
    """A filler response the model ended `IN_PROGRESS`: held open for the tool call it was stalling for."""


@dataclass
class _ServerSession:
    socket: FakeGeminiSession
    known_calls: set[str] = field(default_factory=set[str])
    turn: _Turn | None = None
    triggers: list[str] = field(default_factory=list[str])
    """Inputs the model will answer on its own next (a typed turn, the user's speech, a batch of results)."""
    audio_ms: int = 0
    user_turn: str | None = None


class GeminiServer:
    """The simulated Gemini Live service."""

    def __init__(self, behavior: GeminiBehavior) -> None:
        self.behavior = behavior
        self.truth = GroundTruth()
        self.sessions: list[_ServerSession] = []
        self.handles: dict[str, set[str]] = {}
        """Resumption handle -> the tool calls the server knew about when it was issued."""
        self.send_faults: deque[SendFault] = deque()
        self.failed_sends: list[tuple[str | None, str | None, SendFault]] = []
        self.dial_failures = 0
        self.latency: Any = lambda: 0
        self._next_handle = 1

    # --- transport ---------------------------------------------------------------------------------

    @property
    def session(self) -> _ServerSession | None:
        return self.sessions[-1] if self.sessions else None

    @property
    def socket(self) -> FakeGeminiSession | None:
        return self.session.socket if self.session is not None else None

    def dial(self, config: gt.LiveConnectConfig) -> FakeGeminiSession:
        if self.dial_failures:
            self.dial_failures -= 1
            raise ConnectionClosed(None, Close(1011, 'simulated dial failure'))
        resumption = config.session_resumption
        handle = resumption.handle if resumption is not None else None
        socket = FakeGeminiSession(self, len(self.sessions), handle)
        known = set(self.handles.get(handle, set())) if handle is not None else set()
        self.truth.connections += 1
        self.sessions.append(_ServerSession(socket=socket, known_calls=known))
        return socket

    def patch(self, provider: GoogleProvider) -> Any:
        server = self

        @asynccontextmanager
        async def connect(
            *, model: str, config: gt.LiveConnectConfig | None = None
        ) -> AsyncIterator[FakeGeminiSession]:
            del model
            socket = server.dial(config or gt.LiveConnectConfig())
            try:
                yield socket
            finally:
                await socket.close()

        return mock.patch.object(provider.client.aio.live, 'connect', connect)

    def on_disconnect(self, socket: FakeGeminiSession) -> None:
        self.truth.connection_losses.append(self.truth.tick())
        for response in self.truth.responses.values():
            if response.connection == socket.index + 1 and response.terminal_read is None:
                self.truth.lose(response)
        session = self._session_for(socket)
        session.turn = None
        session.triggers.clear()

    def _session_for(self, socket: FakeGeminiSession) -> _ServerSession:
        return next(session for session in self.sessions if session.socket is socket)

    def on_client_message(self, socket: FakeGeminiSession, kind: str, payload: Any) -> None:
        session = self._session_for(socket)
        if kind == 'realtime':
            if (audio := payload.get('audio')) is not None:
                session.audio_ms += len(audio.data) // 32
            else:  # The session streams nothing else as realtime input.
                self.truth.add_input(payload['video'].data[-8:].decode(errors='replace'), 'image')
            return
        if kind == 'client_content':
            turns, turn_complete = payload
            text = ''.join(part.text or '' for part in (turns.parts or []))
            self.truth.add_input(text, 'text' if turn_complete else 'context', solicits=bool(turn_complete))
            if turn_complete:
                self._barge_in(session)
                session.triggers.append(text)
            return
        assert kind == 'tool_response'
        responses: Sequence[gt.FunctionResponse] = payload if isinstance(payload, list) else [payload]
        for response in responses:
            call_id = response.id or ''
            call = self.truth.tool_calls[call_id]
            call.output_received = True
            if call_id not in session.known_calls:
                # A resumed session never issued this call: the result goes nowhere, silently.
                self.truth.add_input(call_id, 'tool_output').answer_lost = True
                continue
            self.truth.add_input(call_id, 'tool_output')
            turn = session.turn
            if turn is None or call_id not in turn.awaiting:  # pragma: lax no cover (the user barged in first)
                continue
            turn.awaiting.discard(call_id)
            turn.outputs.append(call_id)
            if not turn.awaiting:
                # The whole batch is in: the model answers it, once.
                if self.behavior.closes_tool_turn_separately:
                    self._turn_complete(session, turn.responses[-1] if turn.responses else None)
                    session.turn = None
                session.triggers.extend(turn.outputs)

    def on_client_read(self, socket: FakeGeminiSession, message: gt.LiveServerMessage) -> None:
        now = self.truth.tick()
        tags: dict[str, Any] = getattr(message, '_simulation', {}) or {}
        response = self.truth.responses.get(tags.get('response', ''))
        if response is not None:
            if response.started_read is None:
                response.started_read = now
            if tags.get('content') and response.content_read is None:
                response.content_read = now
            if tags.get('terminal') and response.terminal_read is None:
                response.terminal_read = now
            if tags.get('stall') and response.stall_read is None:
                response.stall_read = now
        for call_id in tags.get('calls', ()):
            self.truth.tool_calls[call_id].read = True
        for key, tokens in tags.get('usage', {}).items():
            self.truth.usage_read.setdefault(key, tokens)

    # --- emitting ------------------------------------------------------------------------------------

    def _emit(self, session: _ServerSession, message: gt.LiveServerMessage, **tags: Any) -> None:
        # Truth bookkeeping rides along on the message object (the connection never looks at it).
        object.__setattr__(message, '_simulation', tags)
        session.socket.emit(message)

    def _start_response(self, session: _ServerSession, *, answers: list[str]) -> TruthResponse:
        number = self.truth.next_response_number
        response = self.truth.new_response(f'gemini-r{number}', trigger='auto', answers=answers)
        if session.turn is None:
            session.turn = _Turn(response=response)
            if self.behavior.handles_at_turn_start:
                self.issue_handle()
        else:
            session.turn.response = response
        session.turn.responses.append(response)
        return response

    def _active_response(self, session: _ServerSession) -> TruthResponse:
        turn = session.turn
        if turn is not None and turn.response is not None and not turn.awaiting:
            return turn.response
        self._settle_stall(session)
        triggers, session.triggers = session.triggers, []
        return self._start_response(session, answers=triggers)

    def _settle_stall(self, session: _ServerSession) -> None:
        """The model did something other than the tool call it stalled for: the filler was a response of its own."""
        turn = session.turn
        if turn is None or (stalled := turn.stalled) is None:
            return
        turn.stalled = None
        self._end(stalled, 'completed')
        stalled.terminal_read = stalled.stall_read

    def _end(self, response: TruthResponse, status: Literal['completed', 'cancelled', 'failed', 'incomplete']) -> None:
        if response.status == 'in_progress':  # pragma: no branch
            response.status = status
            response.seq_end = self.truth.tick()

    def _turn_complete(
        self, session: _ServerSession, response: TruthResponse | None, *, in_progress: bool = False
    ) -> None:
        number = self.truth.next_response_number
        prompt, output = 100 * number + 1, 100 * number + 2
        usage_key = response.key if response is not None else f'gemini-turn-{number}'
        if response is not None:
            response.input_tokens += prompt
            response.output_tokens += output
        self._emit(
            session,
            gt.LiveServerMessage(
                server_content=gt.LiveServerContent(
                    turn_complete=True,
                    interaction_status=gt.InteractionStatus.IN_PROGRESS if in_progress else None,
                ),
                usage_metadata=gt.UsageMetadata(
                    prompt_token_count=prompt,
                    response_token_count=output,
                    total_token_count=prompt + output,
                    prompt_tokens_details=[gt.ModalityTokenCount(modality=gt.MediaModality.TEXT, token_count=prompt)],
                    response_tokens_details=[
                        gt.ModalityTokenCount(modality=gt.MediaModality.AUDIO, token_count=output)
                    ],
                ),
            ),
            response=response.key if response is not None else '',
            terminal=not in_progress,
            stall=in_progress,
            usage={f'{usage_key}@{self.truth.tick()}': (prompt, output)},
        )
        if not self.behavior.handles_at_turn_start and not in_progress:
            self.issue_handle()

    def _barge_in(self, session: _ServerSession) -> None:
        """Whatever the model was saying is cut off by a new user turn."""
        if session.turn is None:
            if not session.triggers:
                return
            # A reply the model hadn't started on yet is cut off too: it ends, empty and interrupted, so every
            # typed turn gets a turn boundary of its own.
            self._active_response(session)
        turn = session.turn
        assert turn is not None
        self._settle_stall(session)
        if turn.awaiting:
            cancelled = sorted(turn.awaiting)
            for call_id in cancelled:
                self.truth.tool_calls[call_id].cancelled_by_server = True
            self._emit(
                session,
                gt.LiveServerMessage(tool_call_cancellation=gt.LiveServerToolCallCancellation(ids=cancelled)),
            )
        response = turn.response
        if response is not None and response.status == 'in_progress':
            self._end(response, 'cancelled')
        self._emit(
            session,
            gt.LiveServerMessage(server_content=gt.LiveServerContent(interrupted=True)),
            response=response.key if response is not None else '',
        )
        self._turn_complete(session, response)
        session.turn = None

    # --- actions the simulation drives ------------------------------------------------------------

    def can_finish(self) -> bool:
        session = self.session
        assert session is not None
        turn = session.turn
        return turn is not None and turn.response is not None and not turn.awaiting

    def speak(self, chunks: int = 1) -> None:
        session = self.session
        assert session is not None
        response = self._active_response(session)
        for _ in range(chunks):
            self._emit(
                session,
                gt.LiveServerMessage(
                    server_content=gt.LiveServerContent(
                        model_turn=gt.Content(
                            role='model',
                            parts=[gt.Part(inline_data=gt.Blob(data=AUDIO_CHUNK, mime_type='audio/pcm;rate=24000'))],
                        )
                    )
                ),
                response=response.key,
                content=True,
            )
            response.audio_bytes += len(AUDIO_CHUNK)
        word = f'r{response.number}w{len(response.words) + 1}'
        response.words.append(word)
        self._emit(
            session,
            gt.LiveServerMessage(
                server_content=gt.LiveServerContent(
                    output_transcription=gt.Transcription(text=word if len(response.words) == 1 else f' {word}')
                )
            ),
            response=response.key,
            content=True,
        )

    def call_tools(self, count: int = 1) -> list[str]:
        """The model calls `count` tools in one `tool_call` message, and waits for all their results."""
        session = self.session
        assert session is not None
        turn = session.turn
        if turn is not None and turn.stalled is not None and not session.triggers:
            # The tool call the filler was stalling for: it belongs to the same response.
            response, turn.stalled = turn.stalled, None
            turn.response = response
        else:
            response = self._active_response(session)
        calls: list[gt.FunctionCall] = []
        ids: list[str] = []
        for _ in range(count):
            call_id = self.truth.new_call_id()
            ids.append(call_id)
            self.truth.tool_calls[call_id] = ToolCallTruth(call_id=call_id, response=response.key, name='lookup')
            response.tool_calls.append(call_id)
            session.known_calls.add(call_id)
            calls.append(gt.FunctionCall(id=call_id, name='lookup', args={}))
        # A tool call ends the response it was in: the answer is a response of its own, after the results.
        self._end(response, 'completed')
        assert session.turn is not None
        session.turn.awaiting.update(ids)
        session.turn.response = None
        self._emit(
            session,
            gt.LiveServerMessage(tool_call=gt.LiveServerToolCall(function_calls=calls)),
            response=response.key,
            content=True,
            terminal=True,
            calls=ids,
        )
        return ids

    def finish(self, *, in_progress: bool = False) -> None:
        """End the model's turn (`turn_complete`, with the turn's usage); `in_progress` for a stalled filler turn."""
        session = self.session
        assert session is not None and session.turn is not None and session.turn.response is not None
        turn = session.turn
        response = turn.response
        stall = in_progress and self.behavior.stalls_in_progress
        self._emit(
            session,
            gt.LiveServerMessage(server_content=gt.LiveServerContent(generation_complete=True)),
            response=response.key,
        )
        if stall:
            # The exchange isn't over: the model is still working, and will call a tool next.
            self._turn_complete(session, response, in_progress=True)
            turn.stalled, turn.response = response, None
            return
        self._end(response, 'completed')
        self._turn_complete(session, response)
        session.turn = None

    def user_speaks(self, *, finished: bool = False) -> str:
        """The user says something: it cuts off whatever the model was saying, and the model answers it."""
        session = self.session
        assert session is not None
        key = self.truth.new_user_turn()
        self._barge_in(session)
        session.audio_ms = 0
        self.truth.add_input(key, 'speech', solicits=True)
        if self.behavior.input_transcription:
            self._emit(
                session,
                gt.LiveServerMessage(
                    server_content=gt.LiveServerContent(
                        input_transcription=gt.Transcription(text=key, finished=finished)
                    )
                ),
            )
        session.triggers.append(key)
        return key

    def issue_handle(self) -> str:
        session = self.session
        assert session is not None
        handle = f'h{self._next_handle}'
        self._next_handle += 1
        self.handles[handle] = set(session.known_calls)
        session.socket.emit(
            gt.LiveServerMessage(
                session_resumption_update=gt.LiveServerSessionResumptionUpdate(new_handle=handle, resumable=True)
            )
        )
        return handle

    def drop(self) -> None:
        socket = self.socket
        assert socket is not None
        socket.break_connection()


class GeminiSimulation(Simulation):
    """A session with a Gemini Live model over the simulated Gemini Live service."""

    provider = 'gemini'

    def __init__(
        self,
        *,
        seed: int = 0,
        options: SessionOptions | None = None,
        behavior: GeminiBehavior | None = None,
        strict: bool | None = None,
    ) -> None:
        super().__init__(seed=seed, options=options, strict=strict)
        self.behavior = behavior or GeminiBehavior()
        self.server = GeminiServer(self.behavior)
        if self.options.latency:
            self.server.latency = lambda: self.rng.choice((0, 0, 0, 1, 2, 4))
        self._provider = GoogleProvider(api_key='simulated')

    @property
    def failed_sends(self) -> list[tuple[str | None, str | None, SendFault]]:
        return self.server.failed_sends

    @property
    def truth(self) -> GroundTruth:
        return self.server.truth

    def build_model(self) -> RealtimeModel:
        return GoogleRealtimeModel(self.behavior.model, provider=self._provider)

    def model_settings(self) -> RealtimeModelSettings:
        settings: GoogleRealtimeModelSettings = {'google_input_transcription': self.behavior.input_transcription}
        settings['reconnect'] = {'max_attempts': 2, 'base_delay': 0.1, 'jitter': False}
        return settings

    @contextmanager
    def transport(self) -> Iterator[None]:
        with self.server.patch(self._provider):
            yield

    # --- provider behavior --------------------------------------------------------------------------

    def _after_server(self, deliver: bool, ticks: int | None) -> None:
        if deliver:
            self._socket().deliver()
        self._advance(ticks)

    def _socket(self) -> FakeGeminiSession:
        socket = self.server.socket
        assert socket is not None
        return socket

    @step
    def deliver(self, count: int | None = None, ticks: int | None = None) -> None:
        self._socket().deliver(count)
        self._advance(ticks)

    @step
    def speak(self, chunks: int = 1, deliver: bool = True, ticks: int | None = None) -> None:
        self.server.speak(chunks)
        self._after_server(deliver, ticks)

    @step
    def call_tools(self, count: int = 1, deliver: bool = True, ticks: int | None = None) -> list[str]:
        ids = self.server.call_tools(count)
        self._after_server(deliver, ticks)
        return ids

    @step
    def finish(self, in_progress: bool = False, deliver: bool = True, ticks: int | None = None) -> None:
        self.server.finish(in_progress=in_progress)
        self._after_server(deliver, ticks)

    @step
    def user_speaks(self, finished: bool = False, deliver: bool = True, ticks: int | None = None) -> None:
        self.server.user_speaks(finished=finished)
        self._after_server(deliver, ticks)

    @step
    def issue_handle(self, deliver: bool = True, ticks: int | None = None) -> None:
        self.server.issue_handle()
        self._after_server(deliver, ticks)

    @step
    def fail_next_send(self, fault: SendFault = 'lost', ticks: int | None = None) -> None:
        self.server.send_faults.append(fault)
        self._advance(ticks)

    @step
    def drop(self, refuse_dials: int = 0, ticks: int | None = None) -> None:
        self.server.dial_failures += refuse_dials
        self.server.drop()
        self._advance(ticks)

    # --- settling -----------------------------------------------------------------------------------

    def drive_server_to_rest(self) -> bool:
        server = self.server
        session = server.session
        if session is None or not session.socket.alive:
            return False
        socket = session.socket
        if socket.in_flight:
            socket.deliver()
            return True
        if session.turn is not None and session.turn.stalled is not None and not session.triggers:
            server.call_tools(1)
        elif server.can_finish():
            server.finish()
        elif session.triggers:
            server.speak(1)
            server.finish()
        else:
            return False
        socket.deliver()
        return True


class GeminiMachine(SessionMachine):  # pragma: lax no cover (driven only by randomized exploration)
    """Randomized traces against the simulated Gemini Live service."""

    provider_options = st.builds(
        GeminiBehavior,
        closes_tool_turn_separately=st.booleans(),
        handles_at_turn_start=st.booleans(),
        stalls_in_progress=st.booleans(),
        input_transcription=st.booleans(),
    )

    @staticmethod
    def make_simulation(seed: int, options: SessionOptions, provider_options: GeminiBehavior) -> Simulation:
        return GeminiSimulation(seed=seed, options=options, behavior=provider_options)

    @property
    def gemini(self) -> GeminiSimulation:
        sim = self.s
        assert isinstance(sim, GeminiSimulation)
        return sim

    def alive(self) -> bool:
        socket = self.gemini.server.socket if self.live() else None
        return socket is not None and socket.alive

    def can_speak(self) -> bool:
        session = self.gemini.server.session
        assert session is not None
        turn = session.turn
        return (turn is not None and turn.response is not None and not turn.awaiting) or bool(session.triggers)

    def can_call_tools(self) -> bool:
        turn = self.gemini.server.session.turn  # pyright: ignore[reportOptionalMemberAccess]
        return self.can_speak() or (turn is not None and turn.stalled is not None)

    @precondition(lambda self: self.alive() and self.can_speak())
    @rule(chunks=st.integers(min_value=1, max_value=2), deliver=st.booleans(), ticks=TICKS)
    def speak(self, chunks: int, deliver: bool, ticks: int | None) -> None:
        self.run(lambda: self.gemini.speak(chunks=chunks, deliver=deliver, ticks=ticks))

    @precondition(lambda self: self.alive() and self.can_call_tools())
    @rule(count=st.integers(min_value=1, max_value=3), deliver=st.booleans(), ticks=TICKS)
    def call_tools(self, count: int, deliver: bool, ticks: int | None) -> None:
        self.run(lambda: self.gemini.call_tools(count=count, deliver=deliver, ticks=ticks))

    @precondition(lambda self: self.alive() and self.gemini.server.can_finish())
    @rule(in_progress=st.booleans(), deliver=st.booleans(), ticks=TICKS)
    def finish(self, in_progress: bool, deliver: bool, ticks: int | None) -> None:
        self.run(lambda: self.gemini.finish(in_progress=in_progress, deliver=deliver, ticks=ticks))

    @precondition(lambda self: self.alive() and self.gemini.server.session.audio_ms > 0)  # pyright: ignore[reportOptionalMemberAccess]
    @rule(finished=st.booleans(), deliver=st.booleans(), ticks=TICKS)
    def user_speaks(self, finished: bool, deliver: bool, ticks: int | None) -> None:
        self.run(lambda: self.gemini.user_speaks(finished=finished, deliver=deliver, ticks=ticks))

    @precondition(lambda self: self.alive())
    @rule(deliver=st.booleans(), ticks=TICKS)
    def issue_handle(self, deliver: bool, ticks: int | None) -> None:
        self.run(lambda: self.gemini.issue_handle(deliver=deliver, ticks=ticks))

    @precondition(lambda self: self.alive() and bool(self.gemini.server.socket.in_flight))  # pyright: ignore[reportOptionalMemberAccess]
    @rule(count=st.one_of(st.none(), st.integers(min_value=1, max_value=3)), ticks=TICKS)
    def deliver(self, count: int | None, ticks: int | None) -> None:
        self.run(lambda: self.gemini.deliver(count=count, ticks=ticks))

    @precondition(lambda self: self.alive())
    @rule(fault=st.sampled_from(['lost', 'ambiguous']), ticks=TICKS)
    def fail_next_send(self, fault: SendFault, ticks: int | None) -> None:
        self.run(lambda: self.gemini.fail_next_send(fault=fault, ticks=ticks))

    @precondition(lambda self: self.alive())
    @rule(refuse_dials=st.integers(min_value=0, max_value=2), ticks=TICKS)
    def drop(self, refuse_dials: int, ticks: int | None) -> None:
        self.run(lambda: self.gemini.drop(refuse_dials=refuse_dials, ticks=ticks))
