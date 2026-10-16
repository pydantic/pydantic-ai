"""Known invariant violations on current main, each tied to the PR, or the structural change, that fixes it.

A violation is matched against this registry by invariant code, provider, and a predicate over the
simulation that recognizes the finding's trigger. A match is tolerated in the default exploration mode
(and counted in `_machine.KNOWN_HIT_COUNTS`) so the suite stays green on main while anything *new* still
fails; `REALTIME_SIMULATION_STRICT=1` reports them all. Each finding also has a pinned, minimal scenario
in `test_simulation.py`, marked as a strict expected failure, which starts passing when the fix lands:
then delete the finding here and turn its scenario into an ordinary regression test.

Finding ids are the ones used in the realtime stress reports and review rounds (`OR*` OpenAI Realtime,
`G*` Gemini Live, `L*` GPT-Live, `87xx #n` the n-th finding of a Codex review of that PR), so a violation
here can be traced back to its write-up.

Every finding says how well the behavior it rests on is established (`evidence`): only a `recorded` or
`live-stress` finding is a known bug; a `simulated` one may be an artifact of a fake server's guess and
needs a recording before it's treated as one. A few findings are `accepted` limitations rather than bugs:
they stay, and have no pinned scenario to retire.
"""

from __future__ import annotations as _annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ._simulation import InvariantViolation, Simulation
    from ._truth import TruthResponse

Predicate = Callable[['Simulation', 'InvariantViolation'], bool]

Evidence = Literal['recorded', 'live-stress', 'simulated']
"""How the provider behavior a finding rests on is known.

- `recorded`: every provider behavior it needs is in the cassettes (what the app does, like a tool raising or
  `clear_audio`, needs no evidence);
- `live-stress`: seen in the live stress runs against the provider, but not recorded;
- `simulated`: rests on provider behavior, or a network fault, that only the fake server shows.
"""


@dataclass(frozen=True)
class Finding:
    id: str
    """The finding's reference in the stress reports and reviews."""
    title: str
    tracked_by: str
    """The open PR that fixes it, or the structural change that would."""
    evidence: Evidence
    codes: frozenset[str]
    providers: frozenset[str]
    matches: Predicate
    accepted: bool = False
    """A documented limitation rather than a bug: it stays in the registry, with no pinned scenario to retire."""

    def __str__(self) -> str:
        return f'{self.id} ({self.evidence}): {self.title} (tracked by {self.tracked_by})'


def _context_responses(sim: Simulation, violation: InvariantViolation) -> list[TruthResponse] | None:
    """The server responses a violation names, if it names any (by key, or by number)."""
    truth = sim.truth
    refs = [violation.context['response']] if 'response' in violation.context else []
    refs += violation.context.get('responses', [])
    if not refs:
        return None
    found = [truth.responses.get(ref) if isinstance(ref, str) else truth.responses_by_number.get(ref) for ref in refs]
    return [response for response in found if response is not None]


ALL = frozenset({'openai', 'azure', 'xai', 'gemini', 'gpt-live'})
OPENAI_PROTOCOL = frozenset({'openai', 'azure', 'xai'})
GEMINI = frozenset({'gemini'})


def _late_cancel(sim: Simulation, violation: InvariantViolation) -> bool:
    return getattr(getattr(sim, 'server', None), 'late_cancels', 0) > 0


LATE_CANCEL_DROPS_CONTENT = Finding(
    id='SIM-10',
    title=(
        'a cancel that reaches the server after the response already finished still has the connection drop that '
        "response's content as stragglers: history records it empty, though the model said it and the provider kept it "
        '(no cassette has a `response.cancel`)'
    ),
    tracked_by='per-response-id state: a cancel targets a response id, and is a no-op once that response is done; found by this simulator',
    evidence='simulated',
    codes=frozenset(
        {'response.truncated', 'response.missing', 'usage.attribution', 'wait.hang', 'wait.early', 'history.order'}
    ),
    providers=OPENAI_PROTOCOL,
    matches=_late_cancel,
)


def _stray_terminal(sim: Simulation, violation: InvariantViolation) -> bool:
    """A repeated or late `response.done` for the response the violation names, or for one before it.

    (A stray terminal lands on whatever response the session is assembling when it's read, which can be any later one.)
    """
    truth = sim.truth
    stray = truth.repeated_terminals | getattr(getattr(sim, 'server', None), 'late_terminals', set[str]())
    responses = _context_responses(sim, violation)
    if responses is None:
        return bool(stray)
    first = min((truth.responses[key].number for key in stray), default=None)
    return first is not None and any(response.number >= first for response in responses)


REPEATED_TERMINAL = Finding(
    id='8801',
    title=(
        "a cancelled response's `response.done` (and its usage) arriving after the next response started lands on "
        'whichever response the session is assembling: stamped onto the next response, or recorded as an empty one '
        '(8801 #1, #4, #7). The simulator also sends a `response.done` twice, a robustness fault no provider was '
        'recorded doing, which lands the same way'
    ),
    tracked_by='#8801 (per-response-id session state; adapters report one terminal per response)',
    evidence='live-stress',
    codes=frozenset(
        {
            'codec.content_after_terminal',
            'codec.duplicate_terminal',
            'codec.overlapping_responses',
            'history.order',
            'history.tool_round_order',
            'response.duplicated',
            'response.mixed',
            'response.truncated',
            'usage.total',
            'usage.attribution',
            'usage.requests',
            'wait.early',
            'wait.hang',
        }
    ),
    providers=OPENAI_PROTOCOL,
    matches=_stray_terminal,
)

LOST_RESPONSE_RESERVATION = Finding(
    id='SIM-1',
    title=(
        'a reply lost with a dropped connection keeps its reservation: the reconnect does not ask for it again (it '
        'had started, or the provider resumes without it) and the session does not settle it, so `wait_for_reply()` hangs'
    ),
    tracked_by='a reconnect resolves the reply obligations its connection lost; found by this simulator',
    evidence='simulated',
    codes=frozenset({'wait.hang'}),
    # Fixed for Gemini and GPT-Live by their lifecycle tracker: a drop loses every reply still owed.
    providers=OPENAI_PROTOCOL,
    matches=lambda sim, violation: (
        any(response.lost and response.answers for response in sim.truth.responses.values())
        or any(input_.answer_lost for input_ in sim.truth.inputs)
    ),
)


def _xai_push_to_talk(sim: Simulation) -> bool:
    options = getattr(sim, 'openai', None)
    return options is not None and options.dialect == 'xai' and options.turn_detection == 'manual'


def _receive_loop_send_failed(sim: Simulation, violation: InvariantViolation) -> bool:
    network = getattr(getattr(sim, 'server', None), 'network', None)
    # On xAI push-to-talk, the receive loop sends more once the `response.done` that ends a reply is handled (#9070):
    # the audio held back behind the reply, and the clear a deferred request needs to be answered.
    xai_frames = ('input_audio_buffer.append', 'input_audio_buffer.clear') if _xai_push_to_talk(sim) else ()
    sent = ('response.create', *xai_frames)
    return network is not None and any(
        frame in sent and last_read in ('response.done', 'error') for frame, last_read, _ in network.failed_sends
    )


RECEIVE_LOOP_SEND_FAILURE = Finding(
    id='SIM-4',
    title=(
        'a deferred `response.create` (or, on xAI push-to-talk, audio held back behind the reply, or the clear the '
        'request needs) that the connection sends while handling a `response.done` (or a refusal) fails on a dying '
        "socket, and the whole frame is dropped: that response's usage and terminal never reach the session, and the "
        'deferred request is neither re-asked nor released'
    ),
    tracked_by='an ordered outbox, so the receive loop never sends on the socket itself; found by this simulator',
    evidence='simulated',
    codes=frozenset(
        {
            'usage.total',
            'usage.attribution',
            'response.missing',
            'response.truncated',
            'usage.requests',
            'wait.hang',
            'history.not_restored',
        }
    ),
    providers=OPENAI_PROTOCOL,
    matches=_receive_loop_send_failed,
)


def _sent_before_reply_content(sim: Simulation, violation: InvariantViolation) -> bool:
    """The input went out while a reply was requested or started, but before any of its content had arrived."""
    key, response_key = violation.context.get('input'), violation.context.get('response')
    input_ = sim.truth.input(key) if isinstance(key, str) else None
    response = sim.truth.responses.get(response_key) if isinstance(response_key, str) else None
    if input_ is None or response is None or input_.kind not in ('text', 'context', 'image'):
        return False
    # When the client started sending it: the session places a sent turn when the send begins. A cancelled
    # response's content is dropped as stragglers, so the session never learns of it from that either.
    issued = min((operation.issued for operation in sim.operations if operation.key == key), default=input_.seq)
    return response.content_read is None or response.content_read > issued or response.status == 'cancelled'


UNACKNOWLEDGED_INPUT_PLACED_AFTER_THE_REPLY = Finding(
    id='SIM-39',
    title=(
        'Gemini Live and GPT-Live neither acknowledge an input nor say when a response starts, so an input that asks '
        'for no reply (context, an image), sent while a reply is owed but before any of its content arrived, is '
        'placed after that reply: the server may have had it before it started'
    ),
    tracked_by=(
        "the inferred lifecycle's documented inference (`_inferred_lifecycle.py`): an input sent while the model "
        'owes a reply reached it while it was working on that reply'
    ),
    evidence='simulated',
    codes=frozenset({'history.order'}),
    providers=GEMINI | {'gpt-live'},
    matches=_sent_before_reply_content,
    accepted=True,
)


def _reply_taken_by_earlier_response(sim: Simulation, violation: InvariantViolation) -> bool:
    """A response already under way when the input arrived ended the wait for the input's own reply.

    Or, for a wait on a tool round: a response the provider started on its own ended it before the round's answer.
    """
    input_ = sim.truth.input(violation.context.get('input', ''))
    if input_ is None:
        return _tool_round_reply_taken(sim, violation)
    # Under way as far as the client could see: it hadn't read the response's end when the input went out.
    return any(
        response.seq_start < input_.seq
        and (
            response.seq_end is None
            or response.seq_end > input_.seq
            or response.terminal_read is None
            or response.terminal_read > input_.seq
        )
        for response in sim.truth.responses.values()
    )


def _tool_round_reply_taken(sim: Simulation, violation: InvariantViolation) -> bool:
    responses = sim.truth.responses.values()
    round_ = sim.truth.responses.get(violation.context.get('response', ''))
    if round_ is None or not round_.tool_calls:
        return False
    answer = next((response for response in responses if set(round_.tool_calls) & set(response.answers)), None)
    return any(
        response.trigger == 'auto'
        and response.seq_start > round_.seq_start
        and response is not answer
        and (answer is None or (response.seq_end is not None and response.seq_end < answer.seq_start))
        for response in responses
    )


RESERVATION_TAKEN_BY_OTHER_RESPONSE = Finding(
    id='8763c #3',
    title=(
        "a response already under way when a turn was sent (server VAD, a GPT-Live delegation) takes that turn's "
        "reservation, so `wait_for_reply()` returns when it ends, before the turn's own reply (likewise a response "
        'the provider starts on its own while a tool round is owed its answer)'
    ),
    tracked_by='reply obligations resolved only by the response that answers them',
    evidence='recorded',
    codes=frozenset({'wait.early'}),
    providers=ALL,
    matches=_reply_taken_by_earlier_response,
)

KNOWN_FINDINGS: list[Finding] = [
    RESERVATION_TAKEN_BY_OTHER_RESPONSE,
    REPEATED_TERMINAL,
    LATE_CANCEL_DROPS_CONTENT,
    RECEIVE_LOOP_SEND_FAILURE,
]
"""Checked in order: the more specific findings for a code come before the more general ones."""


def _gemini_behavior(sim: Simulation, name: str) -> bool:
    return bool(getattr(getattr(sim, 'behavior', None), name, False))


def _cut_off_by_the_input(sim: Simulation, violation: InvariantViolation) -> bool:
    """The input the wait was owed a reply for cut off a model turn (or one the model hadn't started on yet)."""
    if any(call.cancelled_by_server for call in sim.truth.tool_calls.values()):
        return True  # It cut off a turn waiting on tool results: Gemini cancelled the calls.
    input_ = sim.truth.input(violation.context.get('input', ''))
    return input_ is not None and any(
        response.status == 'cancelled' and response.seq_end is not None and response.seq_end > input_.seq
        for response in sim.truth.responses.values()
    )


CUT_OFF_TURN_COMPLETE = Finding(
    id='SIM-13',
    title=(
        'a typed turn that cuts off a model turn (one waiting on tool results, whose calls Gemini cancels, or one '
        "the model hadn't started on yet) gets its `wait_for_reply()` ended by the cut-off turn's `turn_complete`, "
        'before its own reply'
    ),
    tracked_by=(
        'turn boundaries mapped to the exchange they close, and obligations resolved only by their answer; '
        'the cut-off-turn case #8766 left; found by this simulator'
    ),
    evidence='simulated',
    codes=frozenset({'wait.early'}),
    providers=GEMINI,
    matches=_cut_off_by_the_input,
)


def _continued_after_calling(sim: Simulation, responses: Iterable[TruthResponse] | None = None) -> bool:
    """A response went on after its first tool call: it said more, or called another tool in a later message.

    Any response, unless `responses` narrows it to the ones a violation names.
    """
    truth = sim.truth
    return any(
        any(truth.word_seq[word] > first for word in response.words)
        or any(truth.tool_calls[call_id].seq > first for call_id in response.tool_calls)
        for response in (truth.responses.values() if responses is None else responses)
        if response.tool_calls
        for first in [truth.tool_calls[response.tool_calls[0]].seq]
    )


def _reply_continued_across_a_round(sim: Simulation, violation: InvariantViolation) -> bool:
    """A response the violation names continued after its first tool call."""
    return _continued_after_calling(sim, _context_responses(sim, violation) or [])


def _spoke_after_calling(sim: Simulation, violation: InvariantViolation) -> bool:
    """The model kept going after an asynchronous tool call it made (before or after the result came back)."""
    return _gemini_behavior(sim, 'talks_through_tool_calls') and _continued_after_calling(sim)


def _in_flight_at_the_result(sim: Simulation, violation: InvariantViolation) -> bool:
    """Something the calling response said before its result reached the server hadn't reached the client yet."""
    truth = sim.truth
    responses = _context_responses(sim, violation) or []
    for response in responses:
        outputs = [truth.input(call_id) for call_id in response.tool_calls]
        arrived = [output.seq for output in outputs if output is not None and output.kind == 'tool_output']
        if arrived and any(
            truth.word_seq[word] < min(arrived) and truth.word_read.get(word, min(arrived) + 1) > min(arrived)
            for word in response.words
        ):
            return True
    return False


ASYNC_SPEECH_IN_FLIGHT_AT_THE_RESULT = Finding(
    id='8760-accepted',
    title=(
        'with asynchronous Gemini tool calls, speech the model said before the result went out, but which the '
        'session had not received yet, goes after the tool return: the calling response is recorded when the '
        'result is sent, so an `asap` message can still go out ahead of it'
    ),
    tracked_by="#8760's documented trade-off: speech under way when the result goes out is split there",
    evidence='simulated',
    codes=frozenset({'response.duplicated', 'history.tool_round_order'}),
    providers=GEMINI,
    matches=_in_flight_at_the_result,
    accepted=True,
)


ASYNC_RESULT_CUT_IN_ENDS_THE_WAIT = Finding(
    id='SIM-33',
    title=(
        "with asynchronous Gemini tool calls, the `interrupted` + `turn_complete` the batch's result cuts in with "
        'ends `wait_for_reply()` before the model has answered the result'
    ),
    tracked_by='turn boundaries mapped to the exchange they close (#8760 is parked); found by this simulator',
    evidence='recorded',
    codes=frozenset({'wait.early'}),
    providers=GEMINI,
    matches=lambda sim, violation: _gemini_behavior(sim, 'talks_through_tool_calls') and bool(sim.truth.tool_calls),
)


GEMINI_ASYNC_TOOL_ROUND = Finding(
    id='8760',
    title=(
        'with asynchronous (`NON_BLOCKING`) Gemini tool calls, what the model says (or calls) after the call, in the '
        "same turn, is recorded as a response of its own after the tool's result, as if it had the result in hand"
    ),
    tracked_by='#8760 (parked: per-response-id state, so a response the session already recorded can be continued)',
    evidence='recorded',
    codes=frozenset({'response.duplicated', 'history.tool_round_order', 'wait.hang'}),
    providers=GEMINI,
    matches=_spoke_after_calling,
)


def _raw_transport_error(sim: Simulation, violation: InvariantViolation) -> bool:
    from websockets.exceptions import WebSocketException

    errors = [operation.error for operation in sim.operations] + [sim.consumer_error]
    return any(isinstance(error, WebSocketException) for error in errors)


LIVE_RAW_CLOSE_ERROR = Finding(
    id='SIM-8',
    title=(
        'an abnormal close of a GPT-Live connection escapes as a raw `websockets.ConnectionClosedError` (from iterating '
        'the session and from the next send) instead of `RealtimeError`: the connection only handles a clean close'
    ),
    tracked_by='an adapter-local fix in `OpenAILiveConnection.__aiter__`; found by this simulator',
    evidence='simulated',
    codes=frozenset({'api.unexpected_error'}),
    providers=frozenset({'gpt-live'}),
    matches=_raw_transport_error,
)


def _parked_error(sim: Simulation, violation: InvariantViolation) -> bool:
    from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded

    from ._invariants import SimulatedToolError

    parked = (SimulatedToolError, UsageLimitExceeded, UnexpectedModelBehavior)
    errors = [sim.consumer_error, *(operation.error for operation in sim.operations)]
    return 'error' in sim.tools.settled.values() or any(isinstance(error, parked) for error in errors)


PARKED_ERROR_LEAVES_REQUEST_OWED = Finding(
    id='SIM-15',
    title=(
        'the session ends on a parked error (a tool that raised, or a tool result over `request_limit`) while '
        'another request is still owed (deferred behind the tool round), so `wait_for_reply()` hangs: the part '
        'of OR8 that #8765 left'
    ),
    tracked_by='every reply reservation settled when the session parks an error; found by this simulator',
    evidence='recorded',
    codes=frozenset({'wait.hang'}),
    providers=ALL,
    matches=_parked_error,
)


LIVE_REPLY_SPLIT_BY_TOOL_ROUND = Finding(
    id='SIM-18',
    title=(
        'on GPT-Live, a spoken reply that goes on across a delegated tool round is recorded in two pieces around '
        "the tool's return (the GPT-Live counterpart of #8760); when the model has started another reply by then, "
        "the delegation's later tool call is recorded in that reply instead, with the words around it"
    ),
    tracked_by='per-response-id state, so a response the session already recorded can be continued; found by this simulator',
    evidence='recorded',
    codes=frozenset({'response.duplicated', 'response.mixed', 'response.truncated'}),
    providers=frozenset({'gpt-live'}),
    matches=_reply_continued_across_a_round,
)


def _terminal_read_as_the_connection_dropped(sim: Simulation, violation: InvariantViolation) -> bool:
    # Read off its connection right before that connection dropped (within a few clock ticks), or after, from what
    # it had already received: either way the session handles it on a connection it already gave up on.
    truth = sim.truth
    losses = truth.connection_losses  # One per connection, in order: connection `n` dropped at `losses[n - 1]`.
    return any(
        response.terminal_read is not None
        and len(losses) >= response.connection
        and response.terminal_read > losses[response.connection - 1] - 4
        for response in truth.responses.values()
    )


TERMINAL_DISCARDED_WITH_THE_CONNECTION = Finding(
    id='SIM-22',
    title=(
        'a `response.done` read from a connection as it drops (just before, or from what it had already received) is '
        "discarded with it: the response's usage never reaches `session.usage`, though the provider billed it"
    ),
    tracked_by='a reconnect that drains what the dropped connection already read; found by exploration on the refactor branch',
    evidence='simulated',
    codes=frozenset({'usage.total', 'usage.attribution'}),
    providers=OPENAI_PROTOCOL,
    matches=_terminal_read_as_the_connection_dropped,
)


def _turn_without_its_transcript(sim: Simulation) -> bool:
    """With transcription on, a spoken turn's transcript never reached the client before the session ended or dropped."""
    ended = sim.close_requested is not None or sim.receive_ended or bool(sim.truth.connection_losses)
    return (
        ended
        and getattr(getattr(sim, 'openai', None), 'transcription', False)
        and any(input_.kind == 'speech' and input_.transcript_read is None for input_ in sim.truth.inputs)
    )


def _hand_commit_unanswered_at_the_end(sim: Simulation) -> bool:
    """A turn `commit_audio()` committed whose reply the client hadn't read the end of when the connection dropped or
    the session ended."""
    truth = sim.truth
    ended = bool(truth.connection_losses) or sim.close_requested is not None or sim.receive_ended
    return ended and any(
        input_.kind == 'speech'
        and input_.committed_by_client
        and (input_.answered_by is None or truth.responses[input_.answered_by].terminal_read is None)
        for input_ in truth.inputs
    )


def _push_to_talk_audio_after_a_repeated_terminal(sim: Simulation) -> bool:
    return (
        _xai_push_to_talk(sim)
        and bool(sim.truth.repeated_terminals)
        and any(operation.name == 'send_audio' for operation in sim.operations)
    )


PUSH_TO_TALK_AUDIO_AFTER_A_REPEATED_TERMINAL = Finding(
    id='SIM-36',
    title=(
        'on xAI push-to-talk, audio sent after a `response.done` the server repeated is held behind a reply that '
        'already ended: the session core leaves `wait_for_reply()` hanging, or holds back the reply after it (a '
        'repeated terminal is a robustness fault no provider was recorded sending, see 8801)'
    ),
    tracked_by="#9070's held audio released by the response that ended, not by the next terminal; found by this simulator",
    evidence='simulated',
    codes=frozenset({'history.turn_missing', 'wait.hang', 'response.missing', 'usage.requests'}),
    providers=frozenset({'xai'}),
    matches=lambda sim, violation: _push_to_talk_audio_after_a_repeated_terminal(sim),
)


HAND_COMMIT_UNANSWERED_AT_THE_END = Finding(
    id='SIM-34',
    title=(
        'a spoken turn `commit_audio()` committed is lost when the connection drops, or the session closes or ends (on '
        "a failed send, an exceeded limit), before its reply is over (the session core's own: the current one loses "
        'it only on xAI, whose held commit goes out with the request)'
    ),
    tracked_by='user turns recorded from the provider committing them; found by this simulator on #9422',
    evidence='simulated',
    codes=frozenset({'history.turn_missing'}),
    providers=OPENAI_PROTOCOL,
    matches=lambda sim, violation: _hand_commit_unanswered_at_the_end(sim),
)


TURN_LOST_AT_CLOSE = Finding(
    id='SIM-25',
    title=(
        'a spoken turn committed with input transcription on, whose transcript never arrives (the session closes or '
        'ends on an error, or the connection drops, first), is never recorded: it waits for a transcript that never '
        'comes (surfaced by Macroscope for closing after `commit_audio()`)'
    ),
    tracked_by='user turns recorded from the provider committing them, not from their transcript; found reviewing #9070',
    evidence='recorded',
    codes=frozenset({'history.turn_missing'}),
    providers=OPENAI_PROTOCOL,
    matches=lambda sim, violation: _turn_without_its_transcript(sim),
)


def _send_failed_across_reconnect(sim: Simulation, violation: InvariantViolation) -> bool:
    from pydantic_ai.realtime import RealtimeError

    if violation.code == 'send.failed_across_reconnect':
        return violation.context.get('operation') != 'send_audio'
    error = sim.consumer_error
    return (
        isinstance(error, RealtimeError)
        and error.message.startswith('Realtime connection failed while sending')
        and sim.truth.connections > 1
    )


def _request_deferred_behind_an_answer_to_speech(sim: Simulation) -> bool:
    """xAI push-to-talk: `create_response()` was called once a reply answering a spoken turn had started."""
    options = getattr(sim, 'openai', None)
    # (A send that asks for a response sends a request too.)
    requests = [
        operation.issued
        for operation in sim.operations
        if operation.name in ('create_response', 'send_text', 'send_image_respond')
    ]
    answering_speech = [
        response
        for response in sim.truth.responses.values()
        if any((input_ := sim.truth.input(key)) is not None and input_.kind == 'speech' for key in response.answers)
    ]
    return (
        options is not None
        and options.turn_detection == 'manual'
        and any(response.seq_start < issued for response in answering_speech for issued in requests)
    )


DEFERRED_REQUEST_DROPPED_AFTER_SPEECH = Finding(
    id='SIM-30',
    title=(
        'on xAI push-to-talk, a `create_response()` made while the reply to a spoken turn is in flight, or after a '
        'reply to speech a request committed (rather than `commit_audio()`), goes out without the buffer clear that '
        'makes xAI answer it: xAI drops a request with nothing new after answering committed audio, so '
        '`wait_for_reply()` hangs'
    ),
    tracked_by='the clear-then-request of #9070 also for a request deferred behind a reply; found by this simulator',
    evidence='simulated',
    codes=frozenset({'wait.hang'}),
    providers=frozenset({'xai'}),
    matches=lambda sim, violation: _request_deferred_behind_an_answer_to_speech(sim),
)


NON_AUDIO_SEND_DURING_RECONNECT = Finding(
    id='G3b',
    title=(
        'a send other than audio (a typed turn, context, an image, `clear_audio`) that hits the dropped socket raises '
        '`RealtimeError` although the reconnect succeeds: #8806 drops audio sent mid-reconnect, but nothing else. '
        'When it is a tool result the session sends, the error ends the event stream over a re-dialed connection, '
        'and a reply asked for afterwards is waited on forever'
    ),
    tracked_by='an ordered outbox that holds sends across a re-dial (#8806 covered audio only)',
    evidence='live-stress',
    codes=frozenset({'send.failed_across_reconnect', 'wait.hang'}),
    providers=OPENAI_PROTOCOL | GEMINI,
    matches=_send_failed_across_reconnect,
)


FRAME_CUT_OFF_BY_CLOSE = Finding(
    id='SIM-23',
    title=(
        'closing the session while the connection is handling a `response.done` that sends a deferred '
        "`response.create` cancels it mid-frame: the whole frame is dropped, so that response's usage and "
        'terminal never reach the session, though it was billed'
    ),
    tracked_by='an ordered outbox, so the receive loop never awaits a send; found by exploration on the refactor branch',
    evidence='simulated',
    codes=frozenset({'usage.total', 'usage.attribution', 'response.missing', 'response.truncated', 'usage.requests'}),
    providers=OPENAI_PROTOCOL,
    matches=lambda sim, violation: sim.close_requested is not None and getattr(sim, 'requests_cut_off', 0) > 0,
)


def _provider_reply_before_any_echo(sim: Simulation, violation: InvariantViolation) -> bool:
    """A response the provider started on its own was read after the input went out, before any of ours was.

    When the violation names a response (`history.order`), it is that one.
    """
    key = violation.context.get('input')
    if not isinstance(key, str) or (input_ := sim.truth.input(key)) is None:  # pragma: lax no cover
        return False
    issued = min((operation.issued for operation in sim.operations if operation.key == key), default=input_.seq)
    responses = sim.truth.responses.values()
    first_echo = min(
        (r.started_read for r in responses if r.trigger == 'create' and r.started_read is not None), default=None
    )
    named = _context_responses(sim, violation)
    return any(
        response.trigger != 'create'
        and response.started_read is not None
        and response.started_read > issued
        and (first_echo is None or response.started_read < first_echo)
        for response in (responses if named is None else named)
    )


PROVIDER_REPLY_BEFORE_ANY_ECHO = Finding(
    id='SIM-24',
    title=(
        'until the server has echoed the metadata of one of our `response.create`s, the connection takes a response '
        'it started on its own (server VAD) for the one we asked for, if ours is outstanding: it answers our input, '
        "so the input is recorded before it and a wait for the input's reply ends with it"
    ),
    tracked_by=(
        "the OpenAI-protocol lifecycle tracker's documented inference: whether a server echoes request metadata is "
        'learned from its first echo, not assumed per provider'
    ),
    evidence='simulated',
    codes=frozenset({'history.order', 'wait.early'}),
    providers=OPENAI_PROTOCOL,
    matches=_provider_reply_before_any_echo,
    accepted=True,
)


CLEARED_TURN_LOSES_ITS_TRANSCRIPT = Finding(
    id='SIM-37',
    title=(
        'clearing the input audio while the user is still speaking ends a spoken turn xAI already added (it adds '
        'one at speech start): the turn is recorded at once with what transcript it had, so the transcript xAI '
        'still sends for it is dropped'
    ),
    tracked_by=(
        'an audio clear that settles only a turn not yet in the conversation, leaving one that joined to wait for '
        'its transcript; found by review of the session core'
    ),
    evidence='simulated',
    codes=frozenset({'history.transcript_lost'}),
    providers=frozenset({'xai'}),
    matches=lambda sim, violation: violation.context.get('input') in sim.truth.speech_cleared,
)


UNCOMMITTED_STOP_RECORDED_AS_A_TURN = Finding(
    id='SIM-38',
    title=(
        'a spoken turn joins the conversation at `speech_stopped`, which server VAD normally commits at once; if it '
        "doesn't (the user goes on, and a later start's item is committed instead), the turn waits for a "
        'transcript that never comes, holding back what follows it for the transcript wait, and is then recorded '
        'as an empty user turn the provider never had'
    ),
    tracked_by=(
        'placing a spoken turn at its commit wherever the provider commits it separately; found by review of the '
        'session core'
    ),
    evidence='simulated',
    codes=frozenset({'history.phantom_turn'}),
    providers=frozenset({'openai', 'azure'}),
    # No more phantom turns than stops taken back: each of those makes at most one.
    matches=lambda sim, violation: violation.context.get('extra', 0) <= len(sim.truth.speech_stopped_uncommitted),
)


KNOWN_FINDINGS.extend(
    [
        FRAME_CUT_OFF_BY_CLOSE,
        NON_AUDIO_SEND_DURING_RECONNECT,
        TURN_LOST_AT_CLOSE,
        PUSH_TO_TALK_AUDIO_AFTER_A_REPEATED_TERMINAL,
        HAND_COMMIT_UNANSWERED_AT_THE_END,
        DEFERRED_REQUEST_DROPPED_AFTER_SPEECH,
        TERMINAL_DISCARDED_WITH_THE_CONNECTION,
        PARKED_ERROR_LEAVES_REQUEST_OWED,
        LIVE_REPLY_SPLIT_BY_TOOL_ROUND,
        LIVE_RAW_CLOSE_ERROR,
        ASYNC_SPEECH_IN_FLIGHT_AT_THE_RESULT,
        GEMINI_ASYNC_TOOL_ROUND,
        ASYNC_RESULT_CUT_IN_ENDS_THE_WAIT,
        CUT_OFF_TURN_COMPLETE,
        # The general reservation leaks last: a more specific finding explains a hang better.
        LOST_RESPONSE_RESERVATION,
        PROVIDER_REPLY_BEFORE_ANY_ECHO,
        CLEARED_TURN_LOSES_ITS_TRANSCRIPT,
        UNCOMMITTED_STOP_RECORDED_AS_A_TURN,
        UNACKNOWLEDGED_INPUT_PLACED_AFTER_THE_REPLY,
    ]
)


def matching_findings(sim: Simulation, violation: InvariantViolation) -> list[Finding]:
    """The known findings that explain `violation`, most specific first."""
    return [
        finding
        for finding in KNOWN_FINDINGS
        if violation.code in finding.codes and sim.provider in finding.providers and finding.matches(sim, violation)
    ]


FINDINGS_BY_ID: dict[str, Finding] = {finding.id: finding for finding in KNOWN_FINDINGS}
assert len(FINDINGS_BY_ID) == len(KNOWN_FINDINGS), 'finding ids must be unique'
