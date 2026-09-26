"""The simulated OpenAI GPT-Live server, and the simulation over it.

GPT-Live has no turn events at all, which is what makes it a protocol family of its own:

- the model's output is a continuous audio track, voiced while it speaks and silent otherwise, with its
  words as timeline-stamped transcript fragments; the *connection* ends a reply once the model has been
  quiet for `openai_live_turn_silence_ms` (so here, once virtual time moves past that silence);
- the user's words arrive the same way, as input transcript fragments with no end;
- work is delegated: `session.delegation.created` opens a delegation whose Responses backend runs inside
  `response.event` frames: `response.created`, function calls as `response.output_item.done`, and a
  terminal (`response.completed`, with the backend's token usage) per backend response; the client
  returns results with `response.item.create` and continues the delegation with `response.create`;
- the session itself is billed by the second, cumulatively, in `session.usage.updated`.

The model's spoken replies are the ground-truth responses. Two stretches of speech with less than the
turn silence between them are one reply, as the protocol defines it; which one the session makes of
them is the adapter's inference, checked by the invariants like everything else.
"""

from __future__ import annotations as _annotations

import base64
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal
from unittest import mock

from hypothesis import strategies as st
from hypothesis.stateful import precondition, rule

from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.realtime import RealtimeModel, openai_live as live_module
from pydantic_ai.realtime.codec import RealtimeCodecEvent, ResponseDone
from pydantic_ai.realtime.openai_live import OpenAILiveConnection, OpenAILiveModel, OpenAILiveModelSettings
from pydantic_ai.realtime.settings import RealtimeModelSettings

from ._machine import TICKS, SessionMachine
from ._simulation import SessionOptions, Simulation, step
from ._truth import GroundTruth, ToolCallTruth, TruthResponse
from ._wire import FakeWebSocket, Network, SendFault

VOICED = base64.b64encode(b'\x00\x10' * 2400).decode()
"""100 ms of 24 kHz speech."""
SILENT = base64.b64encode(b'\x00' * 4800).decode()
"""100 ms of the idle track."""
TURN_SILENCE_MS = 500
BACKEND_MODEL = 'gpt-6-sol'


def _backend_response(response_id: str, status: str, usage: tuple[int, int] | None = None) -> dict[str, Any]:
    return {
        'id': response_id,
        'object': 'response',
        'created_at': 0,
        'status': status,
        'model': BACKEND_MODEL,
        'output': [],
        'parallel_tool_calls': True,
        'tool_choice': 'auto',
        'tools': [],
        'error': {'code': 'server_error', 'message': 'simulated failure'} if status == 'failed' else None,
        'usage': None
        if usage is None
        else {
            'input_tokens': usage[0],
            'input_tokens_details': {'cached_tokens': 0, 'cache_write_tokens': 0},
            'output_tokens': usage[1],
            'output_tokens_details': {'reasoning_tokens': 0},
            'total_tokens': sum(usage),
        },
    }


@dataclass
class _Delegation:
    id: str
    spoken: TruthResponse
    """The spoken reply the model delegated from: its tool calls belong to it."""
    backend: str | None = None
    """The backend response in flight, if any."""
    backend_calls: list[str] = field(default_factory=list[str])
    pending_calls: set[str] = field(default_factory=set[str])
    outputs: list[str] = field(default_factory=list[str])
    backends: int = 0


class LiveServer:
    """The simulated GPT-Live service."""

    def __init__(self, now: Any) -> None:
        self.truth = GroundTruth()
        self.network = Network(self)
        self._now = now
        self.speaking: TruthResponse | None = None
        self.last_voice: float | None = None
        self.triggers: list[str] = []
        self.delegations: dict[str, _Delegation] = {}
        self.seconds = 0.0
        self.backend_terminals_read = 0
        self.backends_failed = 0
        self._next_event = 1
        self._next_delegation = 1
        self._next_backend = 1
        self._timeline_ms = 0

    # --- transport ---------------------------------------------------------------------------------

    @property
    def socket(self) -> FakeWebSocket | None:
        return self.network.socket

    def on_connect(self, socket: FakeWebSocket, url: str) -> None:
        self.truth.connections += 1

    def on_disconnect(self, socket: FakeWebSocket) -> None:
        self.truth.connection_losses.append(self.truth.tick())
        for response in self.truth.responses.values():
            if response.terminal_read is None:
                self.truth.lose(response)

    def _emit(self, frame: dict[str, Any], *, immediately: bool = False) -> None:
        socket = self.socket
        assert socket is not None
        frame = {'event_id': f'event_{self._next_event}', **frame}
        self._next_event += 1
        socket.emit(frame, immediately=immediately)

    def _timeline(self) -> tuple[int, int]:
        start = self._timeline_ms
        self._timeline_ms += 100
        return start, self._timeline_ms

    def on_client_frame(self, socket: FakeWebSocket, frame: dict[str, Any]) -> None:
        kind = frame.get('type')
        if kind == 'session.start':
            self._emit(
                {'type': 'session.started', 'session': {'id': 'live_simulated', 'model': 'gpt-live-1'}},
                immediately=True,
            )
        elif kind == 'session.commentary.append':
            text = frame.get('content', '')
            self.truth.add_input(text, 'text', solicits=True)
            self.triggers.append(text)
        elif kind == 'session.thinking.append':
            self.truth.add_input(frame.get('content', ''), 'context')
        elif kind == 'response.item.create':
            item = frame.get('item', {})
            if item.get('type') == 'function_call_output':  # pragma: no branch
                call_id = item['call_id']
                self.truth.tool_calls[call_id].output_received = True
                self.truth.add_input(call_id, 'tool_output')
                for delegation in self.delegations.values():
                    if call_id in delegation.pending_calls:  # pragma: no branch
                        delegation.pending_calls.discard(call_id)
                        delegation.outputs.append(call_id)
        elif kind == 'response.create':
            delegation = next((d for d in self.delegations.values() if d.backend is None and d.outputs), None)
            if delegation is not None:  # pragma: no branch
                self._start_backend(delegation)

    def on_client_read(self, socket: FakeWebSocket, frame: dict[str, Any]) -> None:
        """Nothing: the connection reads a frame ahead of handling it, so a read proves nothing (see `on_frame_handled`)."""

    def on_frame_handled(self, frame: dict[str, Any]) -> None:
        """The connection turned `frame` into codec events, which the session handles before anything else runs."""
        now = self.truth.tick()
        tags = frame.get('_simulation', {})
        response = self.truth.responses.get(tags.get('response', ''))
        if response is not None:
            if response.started_read is None:
                response.started_read = now
            if tags.get('content') and response.content_read is None:
                response.content_read = now
        for call_id in tags.get('calls', ()):
            self.truth.tool_calls[call_id].read = True
        if (usage := tags.get('usage')) is not None:
            self.truth.usage_read.setdefault(tags['backend'], tuple(usage))
            self.backend_terminals_read += 1
            spoken = self.truth.responses[tags['spoken']]
            if spoken.tool_calls and spoken.terminal_read is None:
                # The reply that delegated a tool call is recorded once its backend reports usage.
                spoken.terminal_read = now

    # --- the model ---------------------------------------------------------------------------------

    def _spoken_response(self) -> TruthResponse:
        """The reply the model is speaking, or a new one if it has been quiet longer than the turn silence."""
        now = self._now()
        # Delegated work suspends the turn clock: however long the model is quiet then, the reply goes on.
        quiet = not self.delegations and (self.last_voice is None or now - self.last_voice >= TURN_SILENCE_MS / 1000)
        if self.speaking is None or quiet:
            if self.speaking is not None and self.speaking.status == 'in_progress':  # pragma: lax no cover (unread end)
                self.speaking.status = 'completed'
                self.speaking.seq_end = self.truth.tick()
            triggers, self.triggers = self.triggers, []
            self.speaking = self.truth.new_response(trigger='auto', answers=triggers)
        return self.speaking

    def speak(self, chunks: int = 1) -> None:
        response = self._spoken_response()
        for _ in range(chunks):
            self._emit(
                {
                    'type': 'session.output_audio.delta',
                    'delta': VOICED,
                    '_simulation': {'response': response.key, 'content': True},
                }
            )
        word = f'r{response.number}w{len(response.words) + 1}'
        response.words.append(word)
        start, end = self._timeline()
        self._emit(
            {
                'type': 'session.output_transcript.delta',
                'start_ms': start,
                'end_ms': end,
                'delta': word if len(response.words) == 1 else f' {word}',
                '_simulation': {'response': response.key, 'content': True},
            }
        )
        self.last_voice = self._now()

    def user_says(self) -> str:
        key = self.truth.new_user_turn()
        self.truth.add_input(key, 'speech', solicits=True)
        self.triggers.append(key)
        start, end = self._timeline()
        self._emit({'type': 'session.input_transcript.delta', 'start_ms': start, 'end_ms': end, 'delta': key})
        return key

    def idle(self, frames: int = 1) -> None:
        for _ in range(frames):
            self._emit({'type': 'session.output_audio.delta', 'delta': SILENT})

    def delegate(self) -> str:
        """The model hands the request to its Responses backend."""
        response = (
            self.speaking
            if self.speaking is not None and self.speaking.status == 'in_progress'
            else self._spoken_response()
        )
        delegation_id = f'item_d{self._next_delegation}'
        self._next_delegation += 1
        delegation = self.delegations[delegation_id] = _Delegation(id=delegation_id, spoken=response)
        # A delegation opens the reply (and restarts its turn clock) just as speech does.
        self.last_voice = self._now()
        self._emit(
            {
                'type': 'session.delegation.created',
                'offset_ms': self._timeline_ms,
                'delegation': {
                    'id': delegation_id,
                    'type': 'delegation',
                    'response_id': 'resp_pending',
                    'target': 'responses',
                },
                '_simulation': {'response': response.key},
            }
        )
        self._start_backend(delegation)
        return delegation_id

    def _start_backend(self, delegation: _Delegation) -> None:
        backend = f'resp_b{self._next_backend}'
        self._next_backend += 1
        delegation.backend = backend
        delegation.backends += 1
        self._emit(
            {
                'type': 'response.event',
                'delegation_id': delegation.id,
                'event': {
                    'type': 'response.created',
                    'response': _backend_response(backend, 'in_progress'),
                    'sequence_number': 0,
                },
            }
        )

    def running_delegation(self) -> _Delegation | None:
        return next((d for d in self.delegations.values() if d.backend is not None), None)

    def backend_call(self, count: int = 1) -> list[str]:
        delegation = self.running_delegation()
        assert delegation is not None
        ids: list[str] = []
        for _ in range(count):
            call_id = self.truth.new_call_id()
            ids.append(call_id)
            self.truth.tool_calls[call_id] = ToolCallTruth(
                call_id=call_id, response=delegation.spoken.key, name='lookup'
            )
            delegation.spoken.tool_calls.append(call_id)
            delegation.pending_calls.add(call_id)
            delegation.backend_calls.append(call_id)
            self._emit(
                {
                    'type': 'response.event',
                    'delegation_id': delegation.id,
                    'event': {
                        'type': 'response.output_item.done',
                        'item': {
                            'id': f'fc_{call_id}',
                            'type': 'function_call',
                            'status': 'completed',
                            'arguments': '{}',
                            'call_id': call_id,
                            'name': 'lookup',
                        },
                        'output_index': 0,
                        'sequence_number': 1,
                    },
                    '_simulation': {'response': delegation.spoken.key, 'content': True, 'calls': [call_id]},
                }
            )
        return ids

    def backend_finish(self, status: Literal['completed', 'failed'] = 'completed') -> None:
        delegation = self.running_delegation()
        assert delegation is not None and delegation.backend is not None
        backend, delegation.backend = delegation.backend, None
        number = int(backend.removeprefix('resp_b'))
        usage = (1000 * number + 1, 1000 * number + 2)
        self._emit(
            {
                'type': 'response.event',
                'delegation_id': delegation.id,
                'event': {
                    'type': 'response.completed' if status == 'completed' else 'response.failed',
                    'response': _backend_response(backend, status, usage),
                    'sequence_number': 2,
                },
                '_simulation': {'backend': backend, 'usage': usage, 'spoken': delegation.spoken.key},
            }
        )
        asked_for_tools, delegation.backend_calls = bool(delegation.backend_calls), []
        if status == 'failed':
            # The backend gave up: the calls it asked for lead nowhere, answered or not.
            self.backends_failed += 1
            for call_id in delegation.spoken.tool_calls:
                self.truth.tool_calls[call_id].cancelled_by_server = True
            for output in delegation.outputs:
                input_ = self.truth.input(output)
                assert input_ is not None
                input_.answer_lost = True
        if asked_for_tools and self.speaking is delegation.spoken:
            # The reply that asked for the tools is complete with this terminal's usage; what the model says
            # next is a reply of its own.
            delegation.spoken.status = 'completed'
            delegation.spoken.seq_end = self.truth.tick()
            self.speaking = None
        if status == 'failed' or not asked_for_tools:
            # The delegation owes nothing more: the model speaks its outcome (the tool results it answered).
            self.triggers.extend(delegation.outputs)
            del self.delegations[delegation.id]
            # A settled delegation restarts the turn clock, as the model's voice does.
            self.last_voice = self._now()
        # Otherwise the backend waits for the results and the client's `response.create` to continue it.

    def bill(self) -> None:
        self.seconds += 1.0
        self._emit({'type': 'session.usage.updated', 'usage': {'seconds': self.seconds}})

    def observe_codec_event(self, event: RealtimeCodecEvent) -> None:
        """The connection inferred a reply's end: that is this protocol's terminal."""
        if (
            isinstance(event, ResponseDone)
            and (response := self.speaking) is not None
            and response.terminal_read is None
        ):
            response.terminal_read = self.truth.tick()
            if response.status == 'in_progress':  # pragma: no branch
                response.status = 'completed'
                response.seq_end = response.terminal_read
            self.speaking = None


class LiveSimulation(Simulation):
    """A session with `gpt-live-1` over the simulated GPT-Live service."""

    provider = 'gpt-live'

    def __init__(self, *, seed: int = 0, options: SessionOptions | None = None, strict: bool | None = None) -> None:
        super().__init__(seed=seed, options=options, strict=strict)
        self.server = LiveServer(self.loop.time)
        if self.options.latency:
            self.server.network.latency = lambda: self.rng.choice((0, 0, 0, 1, 2, 4))

    @property
    def truth(self) -> GroundTruth:
        return self.server.truth

    @property
    def failed_sends(self) -> list[tuple[str | None, str | None, SendFault]]:
        return self.server.network.failed_sends

    def build_model(self) -> RealtimeModel:
        return OpenAILiveModel('gpt-live-1', provider=OpenAIProvider(api_key='simulated'))

    def model_settings(self) -> RealtimeModelSettings:
        settings: OpenAILiveModelSettings = {'openai_live_turn_silence_ms': TURN_SILENCE_MS}
        return settings

    @contextmanager
    def transport(self) -> Iterator[None]:
        with self.server.network.patch(), mock.patch.object(live_module, '_now', self.loop.time):
            yield

    def observe_codec_event(self, event: RealtimeCodecEvent) -> None:
        self.server.observe_codec_event(event)

    def observe_connection(self, connection: Any) -> None:
        """Note each frame when the connection handles it, not when it reads it: it reads one frame ahead."""
        assert isinstance(connection, OpenAILiveConnection)
        map_frame = connection._map_frame  # pyright: ignore[reportPrivateUsage]

        def mapped(raw: str | bytes) -> list[RealtimeCodecEvent]:
            self.server.on_frame_handled(json.loads(raw))
            return map_frame(raw)

        connection._map_frame = mapped  # pyright: ignore[reportPrivateUsage]

    def expected_requests(self) -> int | None:
        return self.server.backend_terminals_read

    # --- provider behavior --------------------------------------------------------------------------

    def _after_server(self, deliver: bool, ticks: int | None) -> None:
        if deliver:
            self.server.network.deliver()
        self._advance(ticks)

    @step
    def deliver(self, count: int | None = None, ticks: int | None = None) -> None:
        self.server.network.deliver(count)
        self._advance(ticks)

    @step
    def speak(self, chunks: int = 1, deliver: bool = True, ticks: int | None = None) -> None:
        self.server.speak(chunks)
        self._after_server(deliver, ticks)

    @step
    def user_says(self, deliver: bool = True, ticks: int | None = None) -> None:
        self.server.user_says()
        self._after_server(deliver, ticks)

    @step
    def delegate(self, deliver: bool = True, ticks: int | None = None) -> None:
        self.server.delegate()
        self._after_server(deliver, ticks)

    @step
    def backend_call(self, count: int = 1, deliver: bool = True, ticks: int | None = None) -> None:
        self.server.backend_call(count)
        self._after_server(deliver, ticks)

    @step
    def backend_finish(
        self, status: Literal['completed', 'failed'] = 'completed', deliver: bool = True, ticks: int | None = None
    ) -> None:
        self.server.backend_finish(status)
        self._after_server(deliver, ticks)

    @step
    def bill(self, deliver: bool = True, ticks: int | None = None) -> None:
        self.server.bill()
        self._after_server(deliver, ticks)

    @step
    def advance_time(self, seconds: float, ticks: int | None = None) -> None:
        """Let time pass; the idle track keeps flowing while it does.

        GPT-Live's turn boundaries are timing, so frames can't be held back across a stretch of time the way
        they can on the other protocols: whatever is in flight is read first, and only then does the clock
        move, so the client sees the model's pauses as long as the model made them.
        """
        self.server.network.deliver()
        self.loop.run_until_idle()
        self.server.idle(min(5, max(1, round(seconds * 10))))
        self.server.network.deliver()
        self.loop.advance_clock(seconds)
        self._advance(ticks)

    @step
    def fail_next_send(self, fault: SendFault = 'lost', ticks: int | None = None) -> None:
        self.server.network.send_faults.append(fault)
        self._advance(ticks)

    @step
    def drop(self, ticks: int | None = None) -> None:
        """The connection drops. GPT-Live has no reconnect, so this ends the session."""
        self.server.network.drop()
        self._advance(ticks)

    # --- settling -----------------------------------------------------------------------------------

    def drive_server_to_rest(self) -> bool:
        server = self.server
        socket = server.socket
        if socket is None or not socket.alive:  # pragma: lax no cover (the session ended with its connection)
            return False
        if server.network.in_flight():
            server.network.deliver()
            return True
        if server.running_delegation() is not None:
            server.backend_finish()
        elif server.speaking is not None and server.speaking.terminal_read is None:
            # Let the model's silence end its reply (before it answers anything else).
            server.idle(1)
            server.network.deliver()
            self.loop.run_until_idle()
            self.loop.advance_clock(TURN_SILENCE_MS / 1000)
            return True
        elif server.triggers:
            server.speak(1)
        else:
            return False
        server.network.deliver()
        return True


class LiveMachine(SessionMachine):  # pragma: lax no cover (driven only by randomized exploration)
    """Randomized traces against the simulated GPT-Live service."""

    @staticmethod
    def make_simulation(seed: int, options: SessionOptions, provider_options: Any) -> Simulation:
        return LiveSimulation(seed=seed, options=options)

    @property
    def live_sim(self) -> LiveSimulation:
        sim = self.s
        assert isinstance(sim, LiveSimulation)
        return sim

    def alive(self) -> bool:
        socket = self.live_sim.server.socket
        return self.live() and socket is not None and socket.alive

    @precondition(lambda self: self.alive())
    @rule(chunks=st.integers(min_value=1, max_value=2), deliver=st.booleans(), ticks=TICKS)
    def speak(self, chunks: int, deliver: bool, ticks: int | None) -> None:
        self.run(lambda: self.live_sim.speak(chunks=chunks, deliver=deliver, ticks=ticks))

    @precondition(lambda self: self.alive())
    @rule(deliver=st.booleans(), ticks=TICKS)
    def user_says(self, deliver: bool, ticks: int | None) -> None:
        self.run(lambda: self.live_sim.user_says(deliver=deliver, ticks=ticks))

    @precondition(lambda self: self.alive() and self.live_sim.server.running_delegation() is None)
    @rule(deliver=st.booleans(), ticks=TICKS)
    def delegate(self, deliver: bool, ticks: int | None) -> None:
        self.run(lambda: self.live_sim.delegate(deliver=deliver, ticks=ticks))

    @precondition(lambda self: self.alive() and self.live_sim.server.running_delegation() is not None)
    @rule(count=st.integers(min_value=1, max_value=2), deliver=st.booleans(), ticks=TICKS)
    def backend_call(self, count: int, deliver: bool, ticks: int | None) -> None:
        self.run(lambda: self.live_sim.backend_call(count=count, deliver=deliver, ticks=ticks))

    @precondition(lambda self: self.alive() and self.live_sim.server.running_delegation() is not None)
    @rule(status=st.sampled_from(['completed', 'completed', 'failed']), deliver=st.booleans(), ticks=TICKS)
    def backend_finish(self, status: Literal['completed', 'failed'], deliver: bool, ticks: int | None) -> None:
        self.run(lambda: self.live_sim.backend_finish(status=status, deliver=deliver, ticks=ticks))

    @precondition(lambda self: self.alive())
    @rule(deliver=st.booleans(), ticks=TICKS)
    def bill(self, deliver: bool, ticks: int | None) -> None:
        self.run(lambda: self.live_sim.bill(deliver=deliver, ticks=ticks))

    @precondition(lambda self: self.alive() and bool(self.live_sim.server.network.in_flight()))
    @rule(count=st.one_of(st.none(), st.integers(min_value=1, max_value=3)), ticks=TICKS)
    def deliver(self, count: int | None, ticks: int | None) -> None:
        self.run(lambda: self.live_sim.deliver(count=count, ticks=ticks))

    @precondition(lambda self: self.alive())
    @rule(fault=st.sampled_from(['lost', 'ambiguous']), ticks=TICKS)
    def fail_next_send(self, fault: SendFault, ticks: int | None) -> None:
        self.run(lambda: self.live_sim.fail_next_send(fault=fault, ticks=ticks))

    @precondition(lambda self: self.alive())
    @rule(ticks=TICKS)
    def drop(self, ticks: int | None) -> None:
        self.run(lambda: self.live_sim.drop(ticks=ticks))
