"""A simulated OpenAI Realtime server (also serving the Azure OpenAI and xAI dialects of the protocol).

It is a small model of the provider's server, not a script: it keeps the conversation, at most one
active response, the input audio buffer, and server-VAD state, and it reacts to client frames the way
the recorded cassettes and the live stress runs show the real API does. The behaviors that matter to
the session, each observed live:

- one response at a time: a `response.create` while one is active is refused with
  `conversation_already_has_active_response`, echoing the frame's `event_id`;
- function calls stream before the response's `response.done`, which carries the usage;
- a response the client cancels can keep streaming (stragglers) until its cancelled `response.done`;
- server VAD cancels the active response when the user starts speaking (`interrupt_response`), and
  commits the user's audio item, and asks for a response on its own (`create_response`), when they stop;
- the input transcript arrives on its own schedule, often after the response it prompted;
- an empty `input_audio_buffer.commit` is refused;
- a dropped connection loses whatever was in flight; a re-dial starts a fresh server session, with an
  empty conversation (a tool output for a call it never made is refused) unless xAI resumes it.

Content the server can generate is driven by the simulation (`speak`, `call_tool`, `finish`, ...), so
the trace decides *what* the model says and *when*; the server decides what the protocol makes of it.
"""

from __future__ import annotations as _annotations

import base64
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from ._truth import GroundTruth, Restoration, TruthResponse
from ._wire import FakeWebSocket, Network

AUDIO_CHUNK = b'\x00\x10' * 2400
"""100 ms of 24 kHz PCM16 audio, audible (well above any silence floor)."""
AUDIO_CHUNK_B64 = base64.b64encode(AUDIO_CHUNK).decode()
_BYTES_PER_MS = 48

_EVENT_ID = re.compile(r'pydantic_ai\.(content|response)\.(\d+(?:-\d+)*)')

Dialect = Literal['openai', 'azure', 'xai']

_CONTENT_FRAME_PREFIXES = (
    'response.output_audio',
    'response.audio',
    'response.output_text',
    'response.function_call_arguments',
)


def _usage(input_tokens: int, output_tokens: int) -> dict[str, Any]:
    return {
        'total_tokens': input_tokens + output_tokens,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'input_token_details': {'text_tokens': input_tokens, 'audio_tokens': 0, 'image_tokens': 0, 'cached_tokens': 0},
        'output_token_details': {'text_tokens': output_tokens, 'audio_tokens': 0},
    }


def conversation_fingerprint(item: dict[str, Any]) -> str | None:
    """What identifies a conversation item across sessions, whoever sent it (see `Restoration`)."""
    if item.get('type') in ('function_call', 'function_call_output'):
        return f'{item["type"]}:{item.get("call_id")}'
    content: list[dict[str, Any]] = item.get('content') or [{}]
    part = content[0]
    text = part.get('text') or part.get('transcript')
    if item.get('type') != 'message' or not text:
        return None
    return f'{item.get("role")}:{" ".join(text.split())}'


def _error(code: str, message: str, event_id: str | None = None) -> dict[str, Any]:
    return {
        'type': 'error',
        'error': {
            'type': 'invalid_request_error',
            'code': code,
            'message': message,
            'param': None,
            'event_id': event_id,
        },
    }


@dataclass
class _ActiveResponse:
    truth: TruthResponse
    message_item: str | None = None
    audio_ms: int = 0
    output: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    message_words: list[str] = field(default_factory=list[str])
    cancel_requested: bool = False
    metadata: dict[str, str] | None = None
    commit_reply: bool = False
    """xAI push-to-talk: the reply xAI started on its own for a commit."""


@dataclass
class ServerSession:
    """The state one server session keeps; a re-dial starts a new one."""

    index: int
    socket: FakeWebSocket
    turn_detection: dict[str, Any] | None = None
    transcription: bool = False
    active: _ActiveResponse | None = None
    audio_ms: int = 0
    """Input audio buffered since the last commit or clear."""
    speaking: str | None = None
    """The user turn server VAD currently hears, if any."""
    pending_transcripts: list[str] = field(default_factory=list[str])
    pending_vad_responses: list[str] = field(default_factory=list[str])
    """Spoken turns committed while a response was active, each answered in turn once the one before ends (VAD asks
    for a response at every turn's end; no recording shows what it does with several behind one reply)."""
    unanswered_tool_outputs: list[str] = field(default_factory=list[str])
    item_audio_ms: dict[str, int] = field(default_factory=dict[str, int])
    ended_responses: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    """The `response.done` frames sent on this session, for duplicate-terminal faults."""
    late_done: dict[str, Any] | None = None
    resumed: bool = False
    """An xAI re-dial that resumed the conversation."""
    answers_every_turn: bool = False
    """xAI echoes `create_response: False` back but answers anyway (see `XaiRealtimeModelSettings`)."""
    ptt_speech: str | None = None
    """xAI push-to-talk: the user turn xAI reported speech for (`speech_started`) in the uncommitted buffer."""
    answered_everything: bool = False
    """xAI push-to-talk: a reply ended and nothing new arrived since, so xAI drops a `response.create`."""
    uncommitted_answers: list[str] = field(default_factory=list[str])
    """Spoken turns committed without a reply (push-to-talk, or VAD not answering): the next request answers them."""
    conversation: set[str] = field(default_factory=set[str])
    """What this session's conversation holds, as `conversation_fingerprint`s: a re-dial starts empty."""
    restoration_checked: bool = False

    @property
    def server_vad(self) -> bool:
        return self.turn_detection is not None

    @property
    def interrupt_response(self) -> bool:
        """Whether server VAD cancels the active response when the user starts speaking."""
        return bool((self.turn_detection or {}).get('interrupt_response', True))

    @property
    def create_response(self) -> bool:
        """Whether server VAD answers a turn when the user stops speaking."""
        return self.answers_every_turn or bool((self.turn_detection or {}).get('create_response', True))


class OpenAIServer:
    """The simulated server; the `Network` it is attached to carries its frames."""

    def __init__(self, *, dialect: Dialect = 'openai', model: str = 'gpt-realtime') -> None:
        self.dialect = dialect
        self.model = model
        self.truth = GroundTruth()
        self.network = Network(self)
        self.sessions: list[ServerSession] = []
        self._armed_rejections: list[Literal['content', 'response']] = []
        self._next_item = 1
        # Server event id of an `error` frame -> the inputs it refused, resolved when the client reads it.
        self._refusals: dict[str, list[str]] = {}
        self._next_error = 1
        self._next_event = 1
        self.late_terminals: set[str] = set()
        """Responses whose (cancelled) `response.done` is, or was, held back past the start of the next response."""
        self.late_cancels = 0
        """How many `response.cancel`s arrived with no response left to cancel (it had already finished)."""
        # Client input index (from `event_id`) -> the input's ground-truth key. The connection numbers its
        # inputs for its whole life, across re-dials, so this outlives any one server session.
        self._client_items: dict[int, str] = {}
        self._client_images: dict[int, bool] = {}
        self._conversation_id = 'conv_simulated'
        # What an xAI resumption replays of the conversation, in order. Assistant messages are recorded
        # (`test_xai_ws/test_session_resumption_after_drop`, which replays no user message); function calls
        # and their outputs are inferred: no recording has a tool round before a resumption.
        self._finished_items: list[dict[str, Any]] = []
        # Azure OpenAI speaks the GA event names too (every `test_azure_ws` cassette does); only Voice Live, which
        # has a connection class of its own that isn't simulated, still uses the beta names.
        self._audio_delta = 'response.output_audio.delta'
        self._transcript_delta = 'response.output_audio_transcript.delta'
        self._transcript_done = 'response.output_audio_transcript.done'

    # --- transport hooks ------------------------------------------------------------------------

    def on_connect(self, socket: FakeWebSocket, url: str) -> None:
        self.truth.connections += 1
        session = ServerSession(index=len(self.sessions), socket=socket)
        # xAI resumes a conversation natively: a re-dial naming it gets the finished conversation back.
        session.resumed = self.dialect == 'xai' and f'conversation_id={self._conversation_id}' in url
        if session.resumed and self.sessions:
            session.conversation = set(self.sessions[-1].conversation)
        session.answers_every_turn = self.dialect == 'xai'
        self.sessions.append(session)
        socket.emit(
            {
                'type': 'session.created',
                'event_id': 'evt_created',
                'session': {
                    'type': 'realtime',
                    'object': 'realtime.session',
                    'id': f'sess_{socket.index}',
                    'model': self.model,
                },
            },
            immediately=True,
        )
        if self.dialect == 'xai':
            socket.emit(
                {
                    'type': 'conversation.created',
                    'event_id': 'evt_conversation',
                    'conversation': {'id': self._conversation_id, 'object': 'realtime.conversation'},
                },
                immediately=True,
            )

    def on_client_frame(self, socket: FakeWebSocket, frame: dict[str, Any]) -> None:
        session = self._session_for(socket)
        # A client frame the simulator doesn't model fails loudly: the server must learn it first.
        handler = {
            'session.update': self._on_session_update,
            'input_audio_buffer.append': self._on_audio_append,
            'input_audio_buffer.commit': self._on_audio_commit,
            'input_audio_buffer.clear': self._on_audio_clear,
            'conversation.item.create': self._on_item_create,
            'response.create': self._on_response_create,
            'response.cancel': self._on_response_cancel,
            'conversation.item.truncate': self._on_truncate,
        }[frame['type']]
        handler(session, frame)

    def on_client_read(self, socket: FakeWebSocket, frame: dict[str, Any]) -> None:
        now = self.truth.tick()
        frame_type = frame.get('type', '')
        response_object: dict[str, Any] = frame.get('response') or {}
        response_id = frame.get('response_id') or response_object.get('id')
        response = self.truth.responses.get(response_id) if isinstance(response_id, str) else None
        if response is not None and response.started_read is None:
            response.started_read = now
        if response is not None and response.content_read is None and frame_type.startswith(_CONTENT_FRAME_PREFIXES):
            response.content_read = now
        if frame_type == 'response.function_call_arguments.done' and (
            call := self.truth.tool_calls.get(frame.get('call_id', ''))
        ):
            call.read = True
        if frame_type == 'response.done' and response is not None:
            if response.terminal_read is None:
                response.terminal_read = now
            else:
                self.truth.repeated_terminals.add(response.key)
            self.truth.usage_reports_read += 1
            # xAI reports usage on the frame itself, leaving `response.usage` empty.
            usage: dict[str, int] = response_object.get('usage') or frame.get('usage') or {}
            self.truth.usage_read.setdefault(
                response.key, (usage.get('input_tokens', 0), usage.get('output_tokens', 0))
            )
        elif frame_type in (
            'conversation.item.input_audio_transcription.completed',
            'conversation.item.input_audio_transcription.failed',
        ):
            key = frame['item_id'].removeprefix('item_')
            input_ = self.truth.input(key)
            assert input_ is not None
            input_.transcript_read = input_.transcript_read or now
        elif frame_type == 'input_audio_buffer.committed':
            key = frame['item_id'].removeprefix('item_')
            input_ = self.truth.input(key)
            assert input_ is not None
            input_.committed_read = input_.committed_read or now
        elif frame_type == 'error':
            for key in self._refusals.pop(frame.get('event_id', ''), ()):
                input_ = self.truth.input(key)
                assert input_ is not None
                input_.refused_read = input_.refused_read or now

    def on_disconnect(self, socket: FakeWebSocket) -> None:
        session = self._session_for(socket)
        self.truth.connection_losses.append(self.truth.tick())
        for response in self.truth.responses.values():
            if response.connection == session.index + 1 and response.terminal_read is None:
                self.truth.lose(response)
        if session.active is not None and session.active.message_words:
            # The conversation keeps what a reply cut off had said so far.
            session.conversation.add(f'assistant:{" ".join(session.active.message_words)}')
        session.active = None

    # --- queries used by the simulation ---------------------------------------------------------

    @property
    def session(self) -> ServerSession | None:
        return self.sessions[-1] if self.sessions else None

    def _session_for(self, socket: FakeWebSocket) -> ServerSession:
        return next(session for session in self.sessions if session.socket is socket)

    def _emit(self, session: ServerSession, frame: dict[str, Any]) -> None:
        if 'event_id' not in frame:
            frame = {'event_id': f'evt_{self._next_event}', **frame}
            self._next_event += 1
        session.socket.emit(frame)

    def _refuse(self, session: ServerSession, frame: dict[str, Any], refused: list[str]) -> None:
        """Emit an `error` frame that refuses `refused` (input keys) once the client reads it."""
        event_id = f'evt_error_{self._next_error}'
        self._next_error += 1
        self._refusals[event_id] = refused
        now = self.truth.tick()
        for key in refused:
            input_ = self.truth.input(key)
            assert input_ is not None
            input_.refused_at = input_.refused_at or now
        self._emit(session, {**frame, 'event_id': event_id})

    def _new_item(self, prefix: str = 'item') -> str:
        item = f'{prefix}_{self._next_item}'
        self._next_item += 1
        return item

    # --- client frames ----------------------------------------------------------------------------

    def _on_session_update(self, session: ServerSession, frame: dict[str, Any]) -> None:
        config = frame.get('session', {})
        audio_input = config.get('audio', {}).get('input', {})
        # xAI puts `turn_detection` at the session's top level; OpenAI nests it under `audio.input`.
        session.turn_detection = (
            config['turn_detection'] if 'turn_detection' in config else audio_input.get('turn_detection')
        )
        session.transcription = audio_input.get('transcription') is not None
        if session.resumed:
            # xAI replays the resumed conversation during the handshake, under fresh item ids (recorded:
            # `test_xai_ws/test_session_resumption_after_drop`).
            for item in self._finished_items:
                session.socket.emit(
                    {
                        'type': 'conversation.item.added',
                        'event_id': f'evt_replay_{self._next_item}',
                        'item': {
                            'id': self._new_item('item_replayed'),
                            'object': 'realtime.item',
                            'status': 'completed',
                            **item,
                        },
                    },
                    immediately=True,
                )
        session.socket.emit(
            {'type': 'session.updated', 'event_id': 'evt_updated', 'session': {**config, 'model': self.model}},
            immediately=True,
        )

    def _xai_ptt(self, session: ServerSession) -> bool:
        return self.dialect == 'xai' and not session.server_vad

    def _on_audio_append(self, session: ServerSession, frame: dict[str, Any]) -> None:
        audio = base64.b64decode(frame['audio'])
        session.audio_ms += len(audio) // _BYTES_PER_MS
        if not self._xai_ptt(session) or not any(audio):
            return
        # Recorded (`test_xai_ws/test_push_to_talk_replies_only_when_asked`, #9070): with turn detection off,
        # xAI still reports speech it hears, and speech appended during a reply stops it with no `response.done`.
        session.answered_everything = False
        if session.active is not None:  # pragma: lax no cover (the session holds audio back during a reply)
            self._end_truth(session.active.truth, 'cancelled')
            session.active = None
        if session.ptt_speech is None:
            session.ptt_speech = self.truth.new_user_turn()
            self.truth.speech_started[session.ptt_speech] = self.truth.tick()
            self._emit(
                session,
                {
                    'type': 'input_audio_buffer.speech_started',
                    'item_id': f'item_{session.ptt_speech}',
                    'audio_start_ms': 0,
                },
            )

    def _on_audio_commit(self, session: ServerSession, frame: dict[str, Any]) -> None:
        if self._xai_ptt(session):
            self._xai_ptt_commit(session, frame.get('event_id'), answer=True)
            return
        if session.audio_ms <= 0:
            self._emit(
                session,
                _error('input_audio_buffer_commit_empty', 'Error committing input audio buffer: buffer too small.'),
            )
            return
        self._commit_user_turn(session)

    def _xai_ptt_commit(self, session: ServerSession, event_id: str | None, *, answer: bool) -> str | None:
        """xAI with turn detection off: a commit of speech is answered at once; one of silence does nothing."""
        key, session.ptt_speech = session.ptt_speech, None
        session.audio_ms = 0
        if key is None:
            return None
        item_id = f'item_{key}'
        session.answered_everything = False
        self._emit(session, {'type': 'input_audio_buffer.speech_stopped', 'item_id': item_id, 'audio_end_ms': 1000})
        self.truth.add_input(key, 'speech', solicits=True)
        self.truth.speech_committed.add(key)
        self._emit(session, {'type': 'input_audio_buffer.committed', 'item_id': item_id, 'previous_item_id': None})
        self._audio_item_added(session, item_id)
        if session.transcription:
            session.pending_transcripts.append(key)
        if answer:
            # (Speech appended during a reply has stopped it, so nothing is active by the time it is committed.)
            assert session.active is None
            self._start_response(session, trigger='vad', answers=[key], user_turn=key)
            assert session.active is not None
            session.active.commit_reply = True
        return key

    def _on_audio_clear(self, session: ServerSession, frame: dict[str, Any]) -> None:
        del frame
        if self._xai_ptt(session):
            session.answered_everything = False
            session.ptt_speech = None
            active = session.active
            if active is not None and active.commit_reply and not active.truth.words:  # pragma: lax no cover
                # (Only a clear between a commit the session sent and its reply's first word.)
                self._finish_active(session, 'cancelled')  # A clear right after a commit cancels its reply.
        session.audio_ms = 0
        session.speaking = None
        self._emit(session, {'type': 'input_audio_buffer.cleared'})

    def _on_item_create(self, session: ServerSession, frame: dict[str, Any]) -> None:
        session.answered_everything = False
        item = frame.get('item', {})
        event_id = frame.get('event_id')
        client_index = self._client_index(event_id)
        if item.get('type') == 'function_call_output':
            call_id = item.get('call_id', '')
            call = self.truth.tool_calls.get(call_id)
            if (
                call is None or call.cancelled_by_server or f'function_call:{call_id}' not in session.conversation
            ):  # pragma: lax no cover (only an output for no call)
                # The call was never made on this conversation (or was abandoned, or made on a session a
                # re-dial replaced without replaying it): the real API refuses it.
                self._emit(session, _error('invalid_value', f'No tool call found with call_id {call_id!r}.'))
                return
            self._item_added(session, item)
            if call.output_received:
                # Replayed with the rest of the history on a re-dial: already part of the conversation.
                return
            call.output_received = True
            self._finished_items.append(
                {'type': 'function_call_output', 'call_id': call_id, 'output': item.get('output', '')}
            )
            self.truth.add_input(call_id, 'tool_output')
            session.unanswered_tool_outputs.append(call_id)
            return
        if item.get('type') != 'message' or item.get('role') != 'user' or event_id is None:
            # Replayed history on a re-dial, or seeded history: already part of the conversation.
            self._item_added(session, item)
            return
        content: list[dict[str, str]] = item['content']
        part = content[0]
        kind: Literal['image', 'text']
        if part['type'] == 'input_image':
            key = part['image_url'].rsplit(',', 1)[-1][-12:]
            kind = 'image'
        else:
            key = part['text']
            kind = 'text'
        if self._armed_rejections and self._armed_rejections[0] == 'content':
            self._armed_rejections.pop(0)
            rejected = self.truth.add_input(key, kind, client_index=client_index)
            rejected.rejected = True
            self._refuse(session, _error('string_above_max_length', 'Invalid content: refused.', event_id), [key])
            return
        self.truth.add_input(key, kind, client_index=client_index)
        self._item_added(session, item)
        assert client_index is not None
        self._client_items[client_index] = key
        self._client_images[client_index] = kind == 'image'

    def _on_response_create(self, session: ServerSession, frame: dict[str, Any]) -> None:
        event_id = frame.get('event_id')
        if self._xai_ptt(session):
            if session.ptt_speech is not None:
                # Speech still in the buffer is committed by the request, and answered by its response.
                key = self._xai_ptt_commit(session, event_id, answer=False)
                assert key is not None
                session.uncommitted_answers.append(key)
            elif (
                session.answered_everything and not session.unanswered_tool_outputs
            ):  # pragma: lax no cover (the session asks again only with something new)
                return  # Recorded (#9070): after answering committed audio, with nothing new, xAI drops it.
        indexes = self._client_indexes(event_id)
        self.truth.merged_requests += max(0, len(indexes) - 1)
        requested: list[str] = []
        for index in indexes:
            if index in self._client_items:
                requested.append(self._client_items[index])
            elif index - 1 in self._client_items and self._client_images.get(index - 1):
                # `send(image, respond=True)` sends the image and the request for a response as two inputs.
                requested.append(self._client_items[index - 1])
        requested.extend(session.unanswered_tool_outputs)
        requested.extend(session.uncommitted_answers)
        if self._armed_rejections and self._armed_rejections[0] == 'response':
            self._armed_rejections.pop(0)
            if not requested:  # A bare request for a response: track the refusal on the request itself.
                requested.append(self.truth.add_input(f'create{len(self.truth.inputs)}', 'create', solicits=True).key)
            self._refuse(
                session, _error('server_refused', 'The server refused the response request.', event_id), requested
            )
            return
        if session.active is not None:  # pragma: lax no cover (a race: the session defers requests it knows about)
            self._refuse(
                session,
                _error(
                    'conversation_already_has_active_response',
                    f'Conversation already has an active response in progress: {session.active.truth.key}.',
                    event_id,
                ),
                requested,
            )
            return
        answers = requested
        session.unanswered_tool_outputs.clear()
        session.uncommitted_answers.clear()
        for key in answers:
            input_ = self.truth.input(key)
            assert input_ is not None
            input_.solicits = True
        if not answers:
            answers.append(self.truth.add_input(f'create{len(self.truth.inputs)}', 'create', solicits=True).key)
        # The request's `metadata` comes back on the response it starts, as it does live on every dialect.
        request: dict[str, Any] = frame.get('response') or {}
        metadata: dict[str, str] | None = request.get('metadata')
        self._start_response(session, trigger='create', answers=answers, metadata=metadata)

    def _on_response_cancel(self, session: ServerSession, frame: dict[str, Any]) -> None:
        del frame
        if session.active is None or session.active.cancel_requested:
            if session.active is None:  # pragma: no branch (a second cancel of a live response is a race)
                self.late_cancels += 1
            self._emit(session, _error('response_cancel_not_active', 'Cancellation failed: no active response found.'))
            return
        session.active.cancel_requested = True

    def _on_truncate(self, session: ServerSession, frame: dict[str, Any]) -> None:
        item_id = frame.get('item_id', '')
        audio_end_ms = int(frame.get('audio_end_ms', 0))
        available = session.item_audio_ms.get(item_id)
        if available is None:  # pragma: lax no cover (only a mistargeted truncation)
            self.truth.refused_truncations.append(f'{item_id!r} is not an item of this conversation')
            self._emit(session, _error('invalid_value', f'Item {item_id!r} not found.'))
            return
        if audio_end_ms > available:  # pragma: lax no cover (only a truncation past the audio sent)
            self.truth.refused_truncations.append(f'{item_id!r} has {available} ms of audio, not {audio_end_ms}')
            self._emit(
                session,
                _error('invalid_value', f'Audio content of {available}ms is already shorter than {audio_end_ms}ms'),
            )
            return
        self.truth.truncations.append((item_id, audio_end_ms))
        self._emit(session, {'type': 'conversation.item.truncated', 'item_id': item_id, 'audio_end_ms': audio_end_ms})

    def _item_added(self, session: ServerSession, item: dict[str, Any], item_id: str | None = None) -> None:
        """Acknowledge an item joining the conversation, as the real API does for every item, whoever made it."""
        if (fingerprint := conversation_fingerprint(item)) is not None:
            session.conversation.add(fingerprint)
        self._emit(
            session,
            {
                'type': 'conversation.item.added',
                'previous_item_id': None,
                'item': {**item, 'id': item_id or self._new_item(), 'object': 'realtime.item', 'status': 'completed'},
            },
        )

    def _audio_item_added(self, session: ServerSession, item_id: str) -> None:
        self._item_added(
            session,
            {'type': 'message', 'role': 'user', 'content': [{'type': 'input_audio', 'transcript': None}]},
            item_id,
        )

    def _client_index(self, event_id: str | None) -> int | None:
        indexes = self._client_indexes(event_id)
        return indexes[0] if len(indexes) == 1 else None

    @staticmethod
    def _client_indexes(event_id: str | None) -> list[int]:
        if event_id is None or (match := _EVENT_ID.fullmatch(event_id)) is None:
            return []
        return [int(index) for index in match[2].split('-')]

    # --- response lifecycle -----------------------------------------------------------------------

    def _start_response(
        self,
        session: ServerSession,
        *,
        trigger: Literal['create', 'vad'],
        answers: list[str],
        user_turn: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> None:
        truth = self.truth.new_response(trigger=trigger, answers=answers)
        truth.user_turn = user_turn
        if session.index > 0 and not session.restoration_checked:
            session.restoration_checked = True
            self.truth.restorations.append(
                Restoration(
                    connection=session.index + 1,
                    response=truth.key,
                    before={
                        fingerprint
                        for earlier in self.sessions[: session.index]
                        for fingerprint in earlier.conversation
                    },
                    held=set(session.conversation),
                )
            )
        session.active = _ActiveResponse(truth=truth, metadata=metadata)
        self._emit(
            session,
            {
                'type': 'response.created',
                'response': {
                    'id': truth.key,
                    'object': 'realtime.response',
                    'status': 'in_progress',
                    'output': [],
                    'metadata': metadata,
                },
            },
        )

    def _end_truth(
        self, truth: TruthResponse, status: Literal['completed', 'cancelled', 'failed', 'incomplete', 'lost']
    ) -> None:
        truth.status = status
        truth.seq_end = self.truth.tick()

    def _finish_active(
        self,
        session: ServerSession,
        status: Literal['completed', 'cancelled', 'failed', 'incomplete'],
        *,
        reason: str | None = None,
        late: bool = False,
    ) -> None:
        active = session.active
        assert active is not None
        truth = active.truth
        self._close_message(session, active)
        truth.input_tokens = 10 * truth.number + 1
        truth.output_tokens = 10 * truth.number + 2
        status_details: dict[str, Any] | None = None
        if status == 'cancelled':
            status_details = {'type': 'cancelled', 'reason': reason or 'client_cancelled'}
        elif status == 'incomplete':
            status_details = {'type': 'incomplete', 'reason': 'max_output_tokens'}
        elif status == 'failed':
            status_details = {
                'type': 'failed',
                'error': {'type': 'server_error', 'code': 'simulated', 'message': 'failed'},
            }
        usage = _usage(truth.input_tokens, truth.output_tokens)
        done: dict[str, Any] = {
            'type': 'response.done',
            'response': {
                'id': truth.key,
                'object': 'realtime.response',
                'status': status,
                'status_details': status_details,
                'output': active.output,
                'metadata': active.metadata,
                # xAI reports the usage on the frame itself, and an empty `response.usage`.
                'usage': {} if self.dialect == 'xai' else usage,
            },
        }
        if self.dialect == 'xai':
            done['usage'] = usage
        self._end_truth(truth, status)
        session.active = None
        # Recorded (#9070): xAI drops a `response.create` with nothing new only after answering committed audio.
        # (Anything that arrived while it was answering is new.)
        session.answered_everything = (
            self._xai_ptt(session)
            and any((input_ := self.truth.input(key)) is not None and input_.kind == 'speech' for key in truth.answers)
            and not any(
                input_.seq > truth.seq_start and input_.key not in truth.answers for input_ in self.truth.inputs
            )
        )
        session.ended_responses.append(done)
        if late:
            assert session.late_done is None, 'one late `response.done` at a time'
            session.late_done = done
            self.late_terminals.add(truth.key)
        else:
            self._emit(session, done)
        if session.pending_vad_responses:
            pending = session.pending_vad_responses.pop(0)
            self._start_response(session, trigger='vad', answers=[pending], user_turn=pending)

    def _close_message(self, session: ServerSession, active: _ActiveResponse) -> None:
        """End the assistant message item being spoken: its transcript is final once the item is done."""
        if active.message_item is None:
            return
        item = next(item for item in active.output if item['id'] == active.message_item)
        transcript = item['content'][0]['transcript']
        self._finished_items.append(
            {'type': 'message', 'role': 'assistant', 'content': [{'type': 'text', 'text': transcript}]}
        )
        session.conversation.add(f'assistant:{transcript}')
        self._emit(
            session,
            {
                'type': self._transcript_done,
                'response_id': active.truth.key,
                'item_id': active.message_item,
                'output_index': active.output.index(item),
                'content_index': 0,
                'transcript': transcript,
            },
        )
        active.message_item = None
        active.message_words = []

    def _commit_user_turn(self, session: ServerSession) -> str:
        """Commit the buffered audio as a user turn, which the next request for a response answers."""
        key = self.truth.new_user_turn()
        item_id = f'item_{key}'
        session.audio_ms = 0
        self.truth.add_input(key, 'speech').committed_by_client = True
        self.truth.speech_committed.add(key)
        self._emit(session, {'type': 'input_audio_buffer.committed', 'item_id': item_id, 'previous_item_id': None})
        self._audio_item_added(session, item_id)
        if session.transcription:
            session.pending_transcripts.append(key)
        session.uncommitted_answers.append(key)
        return key

    # --- actions the simulation drives ------------------------------------------------------------

    def speak(self, chunks: int = 1) -> None:
        """The active response says one more word, with `chunks` × 100 ms of audio."""
        session = self.session
        assert session is not None and session.active is not None
        active = session.active
        truth = active.truth
        if active.message_item is None:
            active.message_item = self._new_item('item_msg')
            active.output.append(
                {
                    'id': active.message_item,
                    'type': 'message',
                    'role': 'assistant',
                    'status': 'completed',
                    'content': [{'type': 'output_audio', 'transcript': ''}],
                }
            )
            session.item_audio_ms[active.message_item] = 0
            active.audio_ms = 0
        item = active.output[-1]
        common = {
            'response_id': truth.key,
            'item_id': active.message_item,
            'output_index': len(active.output) - 1,
            'content_index': 0,
        }
        for _ in range(chunks):
            self._emit(session, {'type': self._audio_delta, **common, 'delta': AUDIO_CHUNK_B64})
            truth.audio_bytes += len(AUDIO_CHUNK)
            active.audio_ms += len(AUDIO_CHUNK) // _BYTES_PER_MS
            session.item_audio_ms[active.message_item] = active.audio_ms
        word = f'r{truth.number}w{len(truth.words) + 1}'
        truth.words.append(word)
        self.truth.word_seq[word] = self.truth.tick()
        active.message_words.append(word)
        item['content'][0]['transcript'] = ' '.join(active.message_words)
        self._emit(
            session,
            {
                'type': self._transcript_delta,
                **common,
                'delta': word if len(active.message_words) == 1 else f' {word}',
            },
        )

    def call_tool(self, name: str) -> str:
        """The active response calls tool `name`; returns its call id."""
        session = self.session
        assert session is not None and session.active is not None
        active = session.active
        self._close_message(session, active)
        call_id = self.truth.new_call_id()
        item_id = self._new_item('item_fc')
        from ._truth import ToolCallTruth

        self.truth.tool_calls[call_id] = ToolCallTruth(call_id=call_id, response=active.truth.key, name=name)
        active.truth.tool_calls.append(call_id)
        item = {
            'id': item_id,
            'type': 'function_call',
            'status': 'completed',
            'name': name,
            'call_id': call_id,
            'arguments': '{}',
        }
        active.output.append(item)
        session.conversation.add(f'function_call:{call_id}')
        self._finished_items.append({key: item[key] for key in ('type', 'call_id', 'name', 'arguments')})
        self._emit(
            session,
            {
                'type': 'response.function_call_arguments.done',
                'response_id': active.truth.key,
                'item_id': item_id,
                'output_index': len(active.output) - 1,
                'call_id': call_id,
                'name': name,
                'arguments': '{}',
            },
        )
        return call_id

    def finish(self, status: Literal['completed', 'incomplete', 'failed'] = 'completed', *, late: bool = False) -> None:
        """End the active response. A requested cancel ends it as cancelled instead.

        `late=True` withholds the `response.done` until `release_late_done()`, so it can land after the next
        response's `response.created`. Only a cancelled response's can (#8801): the provider starts no other
        response while one that completes normally is still active.
        """
        session = self.session
        assert session is not None and session.active is not None
        assert not late or session.active.cancel_requested, "only a cancelled response's `response.done` comes late"
        if session.active.cancel_requested:
            self._finish_active(session, 'cancelled', late=late)
        else:
            self._finish_active(session, status, late=late)

    def release_late_done(self) -> None:
        session = self.session
        assert session is not None and session.late_done is not None
        done, session.late_done = session.late_done, None
        if done['response']['id'] == self.truth.responses_by_number[self.truth.next_response_number - 1].key:
            self.late_terminals.discard(done['response']['id'])  # No other response started meanwhile.
        self._emit(session, done)

    def repeat_done(self) -> None:
        """Send the most recent `response.done` a second time.

        A robustness fault, like a failed send: no recording shows a provider repeating a terminal, and the lifecycle
        contract (`codec.duplicate_terminal`) forbids it. It checks the session survives one anyway.
        """
        session = self.session
        assert session is not None and session.ended_responses
        self._emit(session, session.ended_responses[-1])

    def speech_start(self, *, late: bool = False) -> str:
        """Server VAD hears the user start speaking, cancelling the active response.

        `late=True` withholds the cancelled response's `response.done` until `release_late_done()` (#8801).
        """
        session = self.session
        assert session is not None and session.server_vad and session.speaking is None
        key = self.truth.new_user_turn()
        session.speaking = key
        self.truth.speech_started[key] = self.truth.tick()
        self._emit(
            session, {'type': 'input_audio_buffer.speech_started', 'item_id': f'item_{key}', 'audio_start_ms': 0}
        )
        if self.dialect == 'xai':
            # xAI adds the spoken turn's item as soon as it hears speech, not when it commits it (recorded:
            # `test_xai_ws/test_audio_in_server_vad_turn`): that is where the turn sits in its conversation.
            self.truth.add_input(key, 'speech', solicits=True)
            self._audio_item_added(session, f'item_{key}')
        if session.active is not None and session.interrupt_response:
            self._finish_active(session, 'cancelled', reason='turn_detected', late=late)
        return key

    def speech_stop(self) -> str:
        """Server VAD hears the user stop: the audio is committed as a user turn, and answered if configured to."""
        session = self.session
        assert session is not None and session.speaking is not None
        key, session.speaking = session.speaking, None
        item_id = f'item_{key}'
        self._emit(session, {'type': 'input_audio_buffer.speech_stopped', 'item_id': item_id, 'audio_end_ms': 1000})
        session.audio_ms = 0
        self.truth.speech_committed.add(key)
        if self.dialect != 'xai':
            self.truth.add_input(key, 'speech', solicits=session.create_response)
        self._emit(session, {'type': 'input_audio_buffer.committed', 'item_id': item_id, 'previous_item_id': None})
        if self.dialect != 'xai':
            self._audio_item_added(session, item_id)
        if session.transcription:
            session.pending_transcripts.append(key)
        if not session.create_response:
            # The turn is committed as it is under push-to-talk: the app asks for the reply itself.
            session.uncommitted_answers.append(key)
        elif session.active is None:
            self._start_response(session, trigger='vad', answers=[key], user_turn=key)
        else:
            session.pending_vad_responses.append(key)
        return key

    def transcribe(self, *, fail: bool = False) -> str:
        """Deliver the oldest pending input transcript (or its transcription failure)."""
        session = self.session
        assert session is not None and session.pending_transcripts
        key = session.pending_transcripts.pop(0)
        item_id = f'item_{key}'
        if fail:
            self._emit(
                session,
                {
                    'type': 'conversation.item.input_audio_transcription.failed',
                    'item_id': item_id,
                    'content_index': 0,
                    'error': {'type': 'transcription_error', 'code': 'simulated', 'message': 'could not transcribe'},
                },
            )
        else:
            self._emit(
                session,
                {
                    'type': 'conversation.item.input_audio_transcription.completed',
                    'item_id': item_id,
                    'content_index': 0,
                    'transcript': key,
                    'usage': {'type': 'tokens', 'input_tokens': 3, 'output_tokens': 1, 'total_tokens': 4},
                },
            )
        return key

    def arm_rejection(self, kind: Literal['content', 'response']) -> None:
        """Refuse the next user content item, or the next response request, with an `error` naming it."""
        self._armed_rejections.append(kind)

    def drop(self) -> None:
        """The connection drops: everything in flight is lost."""
        self.network.drop()
