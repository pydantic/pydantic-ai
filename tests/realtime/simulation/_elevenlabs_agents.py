"""The simulated ElevenLabs Agents server, and the simulation over it.

ElevenLabs Agents runs a cascaded pipeline (ASR, an LLM, TTS) behind a hosted agent, and its WebSocket
protocol names little of what happens, which is what makes it a protocol family of its own. The server
model encodes what the recorded cassettes show:

- the connection opens with a REST preflight (the agent's configuration, a signed URL), then a
  `conversation_initiation_client_data` frame the server answers with `conversation_initiation_metadata`;
- replies are automatic: a typed turn (`user_message`), the end of the user's speech, and the results of
  the client tool calls the agent made each make it speak, with no request frame;
- a reply has no start event: its `audio` frames come first, and the whole-text `agent_response` (with the
  reply's `response_id`) arrives once the LLM is done writing, while TTS may still be voicing it: more audio can
  follow, and the reply ends with its last audio, which `context_usage` arrives with, or with the next turn;
  a reply that calls client tools is only the `client_tool_call`s, and the agent goes on in a reply of its own
  once their results are in;
- the user's words come as one final `user_transcript` per utterance, before the reply to them;
- the user barging in on a reply still being generated gets an `interruption`, then that reply's
  `agent_response` cut down to what was heard; on a reply already delivered (synthesis outruns
  playback), an `interruption` and an `agent_response_correction` instead;
- usage is the LLM's context size, in a `context_usage` report after the reply, for the conversation
  rather than any one reply;
- a conversation waiting for a free slot in the workspace's concurrency queue gets `queue_status`
  `waiting`, hold audio over regular `audio` frames, then `admitted`;
- the server pings, and there is no reconnect: a drop ends the conversation.

The model's words are the ground-truth responses, as on the other simulators.
"""

from __future__ import annotations as _annotations

import base64
from collections.abc import AsyncIterator, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import httpx2
from hypothesis import strategies as st
from hypothesis.stateful import precondition, rule

from pydantic_ai.providers.elevenlabs import ElevenLabsProvider
from pydantic_ai.realtime import RealtimeModel, elevenlabs_agents as rt_elevenlabs
from pydantic_ai.realtime.codec import RealtimeCodecEvent
from pydantic_ai.realtime.elevenlabs_agents import (
    ElevenLabsAgentsRealtimeConnection,
    ElevenLabsAgentsRealtimeModel,
    ElevenLabsAgentsRealtimeModelSettings,
)
from pydantic_ai.realtime.lifecycle import RealtimeFrame, StaleEvent
from pydantic_ai.realtime.settings import RealtimeModelSettings

from ._machine import TICKS, SessionMachine
from ._simulation import SessionOptions, Simulation, step
from ._truth import GroundTruth, ToolCallTruth, TruthResponse
from ._wire import FakeWebSocket, Network, SendFault

AGENT_ID = 'agent_simulated'
SPEECH = base64.b64encode(b'\x00\x10' * 1600).decode()
"""100 ms of 16 kHz speech."""
HOLD = base64.b64encode(b'\x00\x20' * 1600).decode()
"""100 ms of hold music."""


@dataclass
class ElevenLabsAgentsOptions:
    """How the simulated agent behaves."""

    queued: bool = False
    """Whether the conversation first waits in the workspace's concurrency queue (with hold audio)."""


def _agent_config() -> dict[str, Any]:
    """The `GET /v1/convai/agents/{id}` response: the prompt override is permitted, and no client tools."""
    return {
        'agent_id': AGENT_ID,
        'conversation_config': {
            'agent': {'prompt': {'prompt': 'Existing', 'tools': [], 'tool_ids': []}},
            'conversation': {'client_events': ['audio', 'agent_response', 'context_usage']},
        },
        'platform_settings': {'overrides': {'conversation_config_override': {'agent': {'prompt': {'prompt': True}}}}},
    }


def _rest(request: httpx2.Request) -> httpx2.Response:
    if request.url.path == f'/v1/convai/agents/{AGENT_ID}':
        return httpx2.Response(200, json=_agent_config())
    assert request.url.path == '/v1/convai/conversation/get-signed-url', request.url
    return httpx2.Response(200, json={'signed_url': f'wss://api.elevenlabs.io/signed?agent_id={AGENT_ID}'})


class ElevenLabsAgentsServer:
    """The simulated ElevenLabs Agents service."""

    def __init__(self, options: ElevenLabsAgentsOptions) -> None:
        self.options = options
        self.truth = GroundTruth()
        self.network = Network(self)
        self.speaking: TruthResponse | None = None
        """The reply being generated: its terminal (`agent_response`, or its tool calls) is still to come."""
        self.voicing: TruthResponse | None = None
        """The reply whose text is complete, while TTS may still be voicing it."""
        self.calling: TruthResponse | None = None
        """The reply that made tool calls, which the connection takes to be over once no more of its calls come."""
        self.delivered: TruthResponse | None = None
        """The latest reply the server voiced in full, which the user may still be hearing."""
        self.triggers: list[str] = []
        """Inputs the agent will reply to next."""
        self.pending_calls: dict[str, TruthResponse] = {}
        """Client tool calls whose results the agent waits for, by call id."""
        self.answered_calls: list[str] = []
        self.message_ids: dict[str, str] = {}
        """The `id` each typed turn went out with, which the replies answering it name."""
        self.callers: dict[str, TruthResponse] = {}
        """The reply that made each client tool call."""
        self._next_event = 1
        self._next_usage = 1
        self._context_tokens = 100

    # --- transport ---------------------------------------------------------------------------------

    @property
    def socket(self) -> FakeWebSocket | None:
        return self.network.socket

    def on_connect(self, socket: FakeWebSocket, url: str) -> None:
        self.truth.connections += 1

    def on_disconnect(self, socket: FakeWebSocket) -> None:
        self.truth.connection_losses.append(self.truth.tick())
        # Nothing under way on a lost conversation is said any more. (A reply's tool calls were all it had to say.)
        self.speaking = self.voicing = None
        if (called := self.calling) is not None:
            self.calling = None
            called.terminal_read = self.truth.tick()
        for response in self.truth.responses.values():
            if response.terminal_read is None:
                self.truth.lose(response)

    def _emit(self, frame: dict[str, Any], *, immediately: bool = False) -> None:
        socket = self.socket
        assert socket is not None
        if (called := self.calling) is not None and frame.get('type') != 'client_tool_call':
            # Something else comes: the reply's calls were all it had.
            self.calling = None
            frame = {**frame, '_simulation': {**frame.get('_simulation', {}), 'called': called.key}}
        socket.emit(frame, immediately=immediately)

    def on_client_frame(self, socket: FakeWebSocket, frame: dict[str, Any]) -> None:
        kind = frame.get('type')
        if kind == 'conversation_initiation_client_data':
            self._emit(
                {
                    'type': 'conversation_initiation_metadata',
                    'conversation_initiation_metadata_event': {
                        'conversation_id': 'conv_simulated',
                        'agent_output_audio_format': 'pcm_16000',
                        'user_input_audio_format': 'pcm_16000',
                    },
                },
                immediately=True,
            )
            if self.options.queued:
                self._emit({'type': 'queue_status', 'queue_status_event': {'status': 'waiting'}})
                self._emit({'type': 'audio', 'audio_event': {'audio_base_64': HOLD, 'event_id': 0}})
                self._emit({'type': 'queue_status', 'queue_status_event': {'status': 'admitted'}})
        elif kind == 'user_message':
            text = frame['text']
            # (The server makes an id up for a typed turn sent without one.)
            self.message_ids[text] = frame.get('id', f'server_{text}')
            self.truth.add_input(text, 'text', solicits=True)
            self.triggers.append(text)
        elif kind == 'contextual_update':
            self.truth.add_input(frame['text'], 'context')
        elif kind == 'client_tool_result':
            call_id = frame['tool_call_id']
            self.truth.tool_calls[call_id].output_received = True
            self.truth.add_input(call_id, 'tool_output')
            caller = self.pending_calls.pop(call_id)
            # Verified live: every result is acknowledged, so the next reply's calls never follow its calls directly.
            self._emit(
                {
                    'type': 'agent_tool_response',
                    'agent_tool_response': {'tool_name': 'lookup', 'tool_call_id': call_id, 'tool_type': 'client'},
                }
            )
            self.answered_calls.append(call_id)
            if not any(response is caller for response in self.pending_calls.values()):
                # Every result the reply asked for is in: the agent goes on answering them.
                self.triggers.extend(self.answered_calls)
                self.answered_calls = []
        # `user_audio_chunk` is speech the server's ASR hears (see `user_says`), and `pong` answers a ping.

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
            if tags.get('terminal') and response.terminal_read is None:
                response.terminal_read = now
        if (called := self.truth.responses.get(tags.get('called', ''))) is not None and called.terminal_read is None:
            called.terminal_read = now
        if (voiced := self.truth.responses.get(tags.get('voiced', ''))) is not None and voiced.terminal_read is None:
            # The frame that tells the connection the reply before it was voiced in full ends that reply.
            voiced.terminal_read = now
        for call_id in tags.get('calls', ()):
            self.truth.tool_calls[call_id].read = True
        if (usage := tags.get('usage')) is not None:
            self.truth.usage_read.setdefault(usage, (frame['context_usage_event']['context_tokens'], 0))
            self.truth.usage_reports_read += 1
        if (transcript := tags.get('transcript')) is not None:
            self.truth.transcripts_read.setdefault(transcript, now)
            input_ = self.truth.input(transcript)
            assert input_ is not None
            input_.transcript_read = input_.transcript_read or now

    # --- the agent ---------------------------------------------------------------------------------

    def _reply(self) -> TruthResponse:
        """The reply being generated, or a new one answering what the agent has to reply to."""
        if self.speaking is None:
            triggers, self.triggers = self.triggers, []
            self.speaking = self.truth.new_response(trigger='auto', answers=triggers)
        return self.speaking

    def voiced(self) -> dict[str, str]:
        """The reply being voiced is over, as the next turn begins: tags for the frame that says so."""
        response, self.voicing = self.voicing, None
        if response is None:
            return {}
        response.status = 'completed'
        response.seq_end = self.truth.tick()
        self.delivered = response
        return {'voiced': response.key}

    def speak(self, chunks: int = 1) -> None:
        voiced = self.voiced() if self.speaking is None else {}
        response = self._reply()
        word = f'r{response.number}w{len(response.words) + 1}'
        response.words.append(word)
        self.truth.word_seq[word] = self.truth.tick()
        for chunk in range(chunks):
            response.audio_bytes += 3200
            self._emit(
                {
                    'type': 'audio',
                    'audio_event': {'audio_base_64': SPEECH, 'event_id': response.number},
                    '_simulation': {'response': response.key, 'content': True, **(voiced if chunk == 0 else {})},
                }
            )

    def voice(self, chunks: int = 1) -> None:
        """More audio of the reply whose text is complete."""
        response = self.voicing
        assert response is not None
        for _ in range(chunks):
            response.audio_bytes += 3200
            self._emit(
                {
                    'type': 'audio',
                    'audio_event': {'audio_base_64': SPEECH, 'event_id': response.number},
                    '_simulation': {'response': response.key, 'content': True},
                }
            )

    def finish(self) -> None:
        """The LLM is done writing the reply: its whole text, while TTS may still be voicing it."""
        response = self.speaking
        assert response is not None and response.words
        self.speaking = None
        self.voicing = response
        self._emit(
            {
                'type': 'agent_response',
                'agent_response_event': {
                    'agent_response': ' '.join(response.words),
                    'event_id': response.number,
                    'response_id': response.key,
                    'in_response_to_ids': self.in_response_to(response),
                },
                '_simulation': {'response': response.key, 'content': True},
            }
        )

    def in_response_to(self, response: TruthResponse) -> list[str]:
        """The ids a reply names as what it answers: those of the typed turns that made the agent speak.

        Speech gets an id the server makes up, and the answer to tool results names what the reply that called the
        tools did (verified live).
        """
        ids: list[str] = []
        for key in response.answers:
            if (caller := self.callers.get(key)) is not None:
                ids.extend(self.in_response_to(caller))
            else:
                ids.append(self.message_ids.get(key, f'server_{key}'))
        return list(dict.fromkeys(ids))

    def _cut(self, response: TruthResponse) -> None:
        """The reply being written is cut off: its text, cut down to what was heard, ends it."""
        self.speaking = None
        response.status = 'cancelled'
        response.seq_end = self.truth.tick()
        self._emit(
            {
                'type': 'agent_response',
                'agent_response_event': {
                    'agent_response': ' '.join(response.words),
                    'event_id': response.number,
                    'response_id': response.key,
                    'in_response_to_ids': self.in_response_to(response),
                },
                '_simulation': {'response': response.key, 'content': True, 'terminal': True},
            }
        )

    def call_tool(self, count: int = 1) -> list[str]:
        """A reply that calls client tools, which is all it does: the agent goes on once their results are in."""
        assert self.speaking is None
        voiced = self.voiced()
        response = self._reply()
        ids: list[str] = []
        for _ in range(count):
            call_id = self.truth.new_call_id()
            ids.append(call_id)
            self.truth.tool_calls[call_id] = ToolCallTruth(
                call_id=call_id, response=response.key, name='lookup', seq=self.truth.tick()
            )
            response.tool_calls.append(call_id)
            self.pending_calls[call_id] = response
            self.callers[call_id] = response
            self._emit(
                {
                    'type': 'client_tool_call',
                    'client_tool_call': {
                        'tool_name': 'lookup',
                        'tool_call_id': call_id,
                        'parameters': {},
                        'event_id': response.number,
                        'expects_response': True,
                    },
                    '_simulation': {
                        'response': response.key,
                        'content': True,
                        'calls': [call_id],
                        **voiced,
                    },
                }
            )
        self.speaking = None
        # The calls end the reply that makes them, which the connection learns once no more come.
        response.status = 'completed'
        response.seq_end = self.truth.tick()
        self.calling = response
        return ids

    def user_says(self) -> str:
        """The user's utterance, as the server's ASR transcribes it once they stop: the agent replies to it."""
        key = self.truth.new_user_turn()
        self.truth.add_input(key, 'speech', solicits=True)
        self.truth.speech_committed.add(key)
        self.triggers.append(key)
        voiced = self.voiced()
        self._emit(
            {
                'type': 'user_transcript',
                'user_transcription_event': {'user_transcript': key, 'event_id': self._next_event},
                '_simulation': {'transcript': key, **voiced},
            }
        )
        self._next_event += 1
        return key

    def barge_in(self) -> None:
        """The user starts talking over the agent: the reply being generated is cut to what was heard."""
        self._emit({'type': 'interruption', 'interruption_event': {'event_id': self._next_event}})
        if (response := self.speaking) is not None and response.words:
            self._cut(response)
        elif (response := self.voicing) is not None:
            # Its text is complete, but not yet voiced in full: the correction ends it, with what was heard.
            self.voicing = None
            response.status = 'cancelled'
            response.seq_end = self.truth.tick()
            self._emit(
                {
                    'type': 'agent_response_correction',
                    'agent_response_correction_event': {
                        'original_agent_response': ' '.join(response.words),
                        'corrected_agent_response': response.words[0],
                        'event_id': response.number,
                        'response_id': response.key,
                    },
                    '_simulation': {'response': response.key, 'terminal': True},
                }
            )
        elif (response := self.delivered) is not None:
            # Delivered already, and still playing: the server corrects the transcript it keeps.
            self._emit(
                {
                    'type': 'agent_response_correction',
                    'agent_response_correction_event': {
                        'original_agent_response': ' '.join(response.words),
                        'corrected_agent_response': response.words[0],
                        'event_id': response.number,
                        'response_id': response.key,
                    },
                }
            )
        self.delivered = None

    def report_usage(self) -> None:
        """The LLM's context size, for the conversation so far, which arrives with a reply's last audio."""
        self._context_tokens += 37
        key = f'context_{self._next_usage}'
        self._next_usage += 1
        self._emit(
            {
                'type': 'context_usage',
                'context_usage_event': {'context_tokens': self._context_tokens, 'context_limit_tokens': 1_000_000},
                '_simulation': {'usage': key, **self.voiced()},
            }
        )

    def ping(self) -> None:
        self._emit({'type': 'ping', 'ping_event': {'event_id': self._next_event, 'ping_ms': 50}})
        self._next_event += 1


class ElevenLabsAgentsSimulation(Simulation):
    """A session with a hosted ElevenLabs agent over the simulated ElevenLabs Agents service."""

    provider = 'elevenlabs-agents'
    output_bytes_per_ms = 32
    """The agent's output is 16 kHz PCM16."""

    def __init__(
        self,
        *,
        seed: int = 0,
        options: SessionOptions | None = None,
        elevenlabs: ElevenLabsAgentsOptions | None = None,
        strict: bool | None = None,
    ) -> None:
        super().__init__(seed=seed, options=options, strict=strict)
        self.server = ElevenLabsAgentsServer(elevenlabs or ElevenLabsAgentsOptions())
        if self.options.latency:
            self.server.network.latency = lambda: self.rng.choice((0, 0, 0, 1, 2, 4))

    @property
    def truth(self) -> GroundTruth:
        return self.server.truth

    def build_model(self) -> RealtimeModel:
        client = httpx2.AsyncClient(transport=httpx2.MockTransport(_rest))
        return ElevenLabsAgentsRealtimeModel(
            AGENT_ID, provider=ElevenLabsProvider(api_key='simulated', http_client=client)
        )

    def model_settings(self) -> RealtimeModelSettings:
        # The simulated agent has no client tools of its own: the session's are trusted rather than synced.
        settings: ElevenLabsAgentsRealtimeModelSettings = {'elevenlabs_agents_tool_sync': 'off'}
        return settings

    @contextmanager
    def transport(self) -> Generator[None]:
        with self.server.network.patch():
            yield

    def observe_connection(self, connection: Any) -> None:
        """Note each frame as the connection handles it, and check the lifecycle stream it makes."""
        assert isinstance(connection, ElevenLabsAgentsRealtimeConnection)
        map_event = connection._map_event  # pyright: ignore[reportPrivateUsage]

        async def mapped(data: dict[str, Any]) -> list[RealtimeCodecEvent]:
            self.server.on_frame_handled(data)
            return await map_event(data)

        connection._map_event = mapped  # pyright: ignore[reportPrivateUsage]

        connection_frames = connection.frames
        lifecycle = connection._lifecycle  # pyright: ignore[reportPrivateUsage]
        observe = self.checker.observe_lifecycle_stream(lambda: lifecycle._inputs)  # pyright: ignore[reportPrivateUsage]

        async def observed() -> AsyncIterator[RealtimeFrame]:
            async for frame in connection_frames():
                for event in frame:
                    # (The simulated server sends a cut reply's terminal right after its interruption, so nothing
                    # comes between them for the connection to make the terminal's transcript stale.)
                    if not isinstance(event, StaleEvent):  # pragma: no branch
                        observe(event)
                yield frame
            observe(None)

        connection.frames = observed

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
    def advance_time(self, seconds: float, ticks: int | None = None) -> None:
        """Let time pass. A reply being voiced that goes quiet this long is over: the connection takes it to be.

        Whatever is in flight is read first, so a quiet stretch is quiet on the client's side too.
        """
        server = self.server
        server.network.deliver()
        self.loop.run_until_idle()
        if (called := server.calling) is not None and seconds >= rt_elevenlabs._TOOL_CALL_BATCH_WINDOW:  # pyright: ignore[reportPrivateUsage]
            server.calling = None
            called.terminal_read = self.truth.tick()
        quieted = server.voicing if seconds >= rt_elevenlabs._AUDIO_TAIL_GRACE else None  # pyright: ignore[reportPrivateUsage]
        if quieted is not None:
            server.voiced()
            # The connection's own clock ends it, as the clock moves on below.
            quieted.terminal_read = self.truth.tick()
        self.loop.advance_clock(seconds)
        self._advance(ticks)

    @step
    def voice(self, chunks: int = 1, deliver: bool = True, ticks: int | None = None) -> None:
        self.server.voice(chunks)
        self._after_server(deliver, ticks)

    @step
    def finish(self, deliver: bool = True, ticks: int | None = None) -> None:
        self.server.finish()
        self._after_server(deliver, ticks)

    @step
    def call_tool(self, count: int = 1, deliver: bool = True, ticks: int | None = None) -> None:
        self.server.call_tool(count)
        self._after_server(deliver, ticks)

    @step
    def user_says(self, deliver: bool = True, ticks: int | None = None) -> None:
        self.server.user_says()
        self._after_server(deliver, ticks)

    @step
    def barge_in(self, deliver: bool = True, ticks: int | None = None) -> None:
        self.server.barge_in()
        self._after_server(deliver, ticks)

    @step
    def report_usage(self, deliver: bool = True, ticks: int | None = None) -> None:
        self.server.report_usage()
        self._after_server(deliver, ticks)

    @step
    def ping(self, deliver: bool = True, ticks: int | None = None) -> None:
        self.server.ping()
        self._after_server(deliver, ticks)

    @step
    def fail_next_send(self, fault: SendFault = 'lost', ticks: int | None = None) -> None:
        self.server.network.send_faults.append(fault)
        self._advance(ticks)

    @step
    def drop(self, ticks: int | None = None) -> None:
        """The connection drops. ElevenLabs conversations can't be resumed, so this ends the session."""
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
        if server.speaking is not None:
            server.finish()
        elif server.voicing is not None:
            server.report_usage()
        elif server.triggers:
            server.speak()
        else:
            return False
        server.network.deliver()
        return True


class ElevenLabsAgentsMachine(SessionMachine):  # pragma: lax no cover (driven only by randomized exploration)
    """Randomized traces against the simulated ElevenLabs Agents service."""

    provider_options = st.builds(ElevenLabsAgentsOptions, queued=st.booleans())

    @staticmethod
    def make_simulation(seed: int, options: SessionOptions, provider_options: Any) -> Simulation:
        return ElevenLabsAgentsSimulation(seed=seed, options=options, elevenlabs=provider_options)

    @property
    def el_sim(self) -> ElevenLabsAgentsSimulation:
        sim = self.s
        assert isinstance(sim, ElevenLabsAgentsSimulation)
        return sim

    def alive(self) -> bool:
        socket = self.el_sim.server.socket
        return self.live() and socket is not None and socket.alive

    def has_cue(self) -> bool:
        """Whether the agent has something to say: a reply under way, something to reply to, or its greeting.

        The agent never speaks on its own otherwise: its replies follow the user's words, a typed turn, or the
        results of its tool calls.
        """
        server = self.el_sim.server
        truth = server.truth
        return server.speaking is not None or bool(server.triggers) or not (truth.inputs or truth.responses)

    @precondition(lambda self: self.alive() and self.has_cue())
    @rule(chunks=st.integers(min_value=1, max_value=2), deliver=st.booleans(), ticks=TICKS)
    def speak(self, chunks: int, deliver: bool, ticks: int | None) -> None:
        self.run(lambda: self.el_sim.speak(chunks=chunks, deliver=deliver, ticks=ticks))

    @precondition(lambda self: self.alive() and self.el_sim.server.voicing is not None)
    @rule(chunks=st.integers(min_value=1, max_value=2), deliver=st.booleans(), ticks=TICKS)
    def voice(self, chunks: int, deliver: bool, ticks: int | None) -> None:
        self.run(lambda: self.el_sim.voice(chunks=chunks, deliver=deliver, ticks=ticks))

    @precondition(lambda self: self.alive() and self.el_sim.server.speaking is not None)
    @rule(deliver=st.booleans(), ticks=TICKS)
    def finish(self, deliver: bool, ticks: int | None) -> None:
        self.run(lambda: self.el_sim.finish(deliver=deliver, ticks=ticks))

    # Not straight after another reply's calls: live, something always comes between the agent's replies (an
    # `agent_tool_response` for each result, say), and the connection takes calls in a row for one reply's.
    @precondition(
        lambda self: (
            self.alive()
            and self.el_sim.server.speaking is None
            and self.el_sim.server.calling is None
            and bool(self.el_sim.server.triggers)
        )
    )
    @rule(count=st.integers(1, 3), deliver=st.booleans(), ticks=TICKS)
    def call_tool(self, count: int, deliver: bool, ticks: int | None) -> None:
        # Parallel calls come in frames of their own, back to back (verified live).
        self.run(lambda: self.el_sim.call_tool(count=count, deliver=deliver, ticks=ticks))

    @precondition(lambda self: self.alive())
    @rule(deliver=st.booleans(), ticks=TICKS)
    def user_says(self, deliver: bool, ticks: int | None) -> None:
        self.run(lambda: self.el_sim.user_says(deliver=deliver, ticks=ticks))

    @precondition(lambda self: self.alive())
    @rule(deliver=st.booleans(), ticks=TICKS)
    def barge_in(self, deliver: bool, ticks: int | None) -> None:
        self.run(lambda: self.el_sim.barge_in(deliver=deliver, ticks=ticks))

    @precondition(lambda self: self.alive())
    @rule(deliver=st.booleans(), ticks=TICKS)
    def report_usage(self, deliver: bool, ticks: int | None) -> None:
        self.run(lambda: self.el_sim.report_usage(deliver=deliver, ticks=ticks))

    @precondition(lambda self: self.alive())
    @rule(deliver=st.booleans(), ticks=TICKS)
    def ping(self, deliver: bool, ticks: int | None) -> None:
        self.run(lambda: self.el_sim.ping(deliver=deliver, ticks=ticks))

    @precondition(lambda self: self.alive() and bool(self.el_sim.server.network.in_flight()))
    @rule(count=st.one_of(st.none(), st.integers(min_value=1, max_value=3)), ticks=TICKS)
    def deliver(self, count: int | None, ticks: int | None) -> None:
        self.run(lambda: self.el_sim.deliver(count=count, ticks=ticks))

    @precondition(lambda self: self.alive())
    @rule(fault=st.sampled_from(['lost', 'ambiguous']), ticks=TICKS)
    def fail_next_send(self, fault: SendFault, ticks: int | None) -> None:
        self.run(lambda: self.el_sim.fail_next_send(fault=fault, ticks=ticks))

    @precondition(lambda self: self.alive())
    @rule(ticks=TICKS)
    def drop(self, ticks: int | None) -> None:
        self.run(lambda: self.el_sim.drop(ticks=ticks))
