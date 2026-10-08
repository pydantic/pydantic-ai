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

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ._simulation import InvariantViolation, Operation, Simulation
    from ._truth import TruthInput, TruthResponse

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


def _inserted_user_speech(sim: Simulation, violation: InvariantViolation) -> bool:
    from ._invariants import is_user_speech_request

    return is_user_speech_request(violation.context['message'])


ALL = frozenset({'openai', 'azure', 'xai', 'gemini', 'gpt-live'})
OPENAI_PROTOCOL = frozenset({'openai', 'azure', 'xai'})


def _tool_results_request_refused(sim: Simulation, violation: InvariantViolation) -> bool:
    return any(input_.kind == 'tool_output' and input_.refused_read is not None for input_ in sim.truth.inputs)


REFUSED_TOOL_RESULTS_REQUEST = Finding(
    id='SIM-12',
    title=(
        'a response request for tool results that the provider refuses keeps its reply reservation (a refused request '
        'for a user turn releases it), so `wait_for_reply()` hangs'
    ),
    tracked_by='reply reservations released with a refused request (#8765 did not cover it); found by this simulator',
    evidence='simulated',
    codes=frozenset({'wait.hang'}),
    providers=OPENAI_PROTOCOL,
    matches=_tool_results_request_refused,
)


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

ANCHORED_USER_TURNS = Finding(
    id='E',
    title='a user turn is inserted into already-recorded history where it started (snapshots are not prefixes of later ones)',
    tracked_by='history projected from an append-only event log, so snapshots are prefixes of later ones',
    evidence='live-stress',
    codes=frozenset({'history.inserted'}),
    providers=ALL,
    matches=_inserted_user_speech,
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
    providers=ALL,
    matches=lambda sim, violation: (
        any(response.lost and response.answers for response in sim.truth.responses.values())
        or any(input_.answer_lost for input_ in sim.truth.inputs)
    ),
)

LOST_UNSTARTED_REQUEST = Finding(
    id='SIM-3',
    title=(
        'a reconnect re-asks for the requests deferred behind a lost, not-yet-started response, but not for that '
        'response itself, so its reservation leaks and `wait_for_reply()` hangs'
    ),
    tracked_by='an outbox that replays every unanswered request after a reconnect; found by this simulator',
    evidence='simulated',
    codes=frozenset({'wait.hang'}),
    providers=OPENAI_PROTOCOL,
    matches=lambda sim, violation: (
        getattr(sim, 'deferred_requests', 0) > 0
        and any(response.lost and response.started_read is None for response in sim.truth.responses.values())
    ),
)


def _receive_loop_send_failed(sim: Simulation, violation: InvariantViolation) -> bool:
    network = getattr(getattr(sim, 'server', None), 'network', None)
    return network is not None and any(
        frame == 'response.create' and last_read in ('response.done', 'error')
        for frame, last_read, _ in network.failed_sends
    )


RECEIVE_LOOP_SEND_FAILURE = Finding(
    id='SIM-4',
    title=(
        'a deferred `response.create` that the connection sends while handling a `response.done` (or a refusal) fails '
        "on a dying socket, and the whole frame is dropped: that response's usage and terminal never reach the "
        'session, and the deferred request is neither re-asked nor released'
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


SENT_BEFORE_REPLY_STARTED = Finding(
    id='SIM-2a',
    title=(
        'a turn sent while a reply is requested or already started, but before any of its content arrived, is recorded '
        'ahead of that reply, though the reply never saw it: the session learns a response exists only from its content'
    ),
    tracked_by=(
        'an explicit response-started event from the adapters, and history ordered by causality; found by this simulator'
    ),
    evidence='recorded',
    codes=frozenset({'history.order'}),
    providers=ALL,
    matches=_sent_before_reply_content,
)


def _spoken_before_reply(sim: Simulation, violation: InvariantViolation) -> bool:
    """A spoken turn committed after a response ended, which the user started before that response was over.

    Either server VAD heard them start before it ended, or their voiced audio started streaming before any of its
    content arrived.
    """
    input_ = sim.truth.input(violation.context.get('input', ''))
    response = sim.truth.responses.get(violation.context.get('response', ''))
    if input_ is None or response is None or input_.kind != 'speech' or response.seq_end is None:
        return False
    if input_.seq <= response.seq_end:
        return False
    if (started := sim.truth.speech_started.get(input_.key)) is not None and started <= response.seq_end:
        return True  # Server VAD heard the user start before the response ended.

    content_read, ended, cancelled = response.content_read, response.seq_end, response.status == 'cancelled'

    def before_reply(operation: Operation) -> bool:
        # Audio a failed send never delivered is no one's turn.
        return (
            operation.name == 'send_audio'
            and operation.error is None
            and (content_read is None or operation.issued < content_read or (cancelled and operation.issued < ended))
        )

    # Audio for this turn: streamed since the spoken turn before it was committed.
    earlier = [other.seq for other in sim.truth.inputs if other.kind == 'speech' and other.seq < input_.seq]
    since = max(earlier, default=0)
    if any(before_reply(operation) and operation.issued > since for operation in sim.operations):
        return True
    # Or more audio went out before the reply than the turns before this one account for: the session gives each
    # stretch of audio a turn of its own (each `send_audio` is one here), though the provider may commit several
    # stretches as one turn and this one later.
    return sum(before_reply(operation) for operation in sim.operations) > len(earlier)


SPEAKING_ORDER = Finding(
    id='SIM-11',
    title=(
        'a spoken turn the user started before a response was over (VAD heard them start, or their audio began '
        'before the response said anything), but which the provider committed after that response ended, is '
        "recorded before it (speaking order, since #8764), while the provider's conversation has it after (history "
        "follows the provider's order, decided 2026-09-28)"
    ),
    tracked_by=(
        "history in the provider's conversation order: the session holds the reply back until the user turn before "
        'it is final, and never inserts into what it recorded; found by this simulator'
    ),
    evidence='recorded',
    codes=frozenset({'history.order'}),
    providers=ALL,
    matches=_spoken_before_reply,
)


def _committed_by_hand_under_server_vad(sim: Simulation, violation: InvariantViolation) -> bool:
    """A spoken turn that reached the server before a response started, with a manual commit in the trace."""
    input_ = sim.truth.input(violation.context.get('input', ''))
    response = sim.truth.responses.get(violation.context.get('response', ''))
    if input_ is None or response is None or input_.kind != 'speech':
        return False
    return input_.seq < response.seq_start and any(operation.name == 'commit_audio' for operation in sim.operations)


COMMITTED_BY_HAND_UNDER_SERVER_VAD = Finding(
    id='OR9',
    title=(
        'a spoken turn committed by hand while server VAD is on is filed after the reply to a turn VAD committed '
        'later (the rest of OR9: push-to-talk, barge-in, and transcription off were fixed by #8764)'
    ),
    tracked_by='#8764 (follow-up)',
    evidence='live-stress',
    codes=frozenset({'history.order'}),
    providers=OPENAI_PROTOCOL,
    matches=_committed_by_hand_under_server_vad,
)


def _waited_before_reply_content(sim: Simulation, violation: InvariantViolation) -> bool:
    """The wait began after the client read that a response started, but before any of its content."""
    response = sim.truth.responses.get(violation.context.get('response', ''))
    started = violation.context.get('started')
    if response is None or started is None:
        return False
    return response.content_read is None or response.content_read > started


WAIT_BEFORE_REPLY_CONTENT = Finding(
    id='SIM-2b',
    title=(
        '`wait_for_reply()` returns at once while a response the provider started on its own (server VAD, a GPT-Live '
        'delegation) has produced no content yet: the session learns a response exists only from its content'
    ),
    tracked_by='an explicit response-started event from the adapters opens the exchange; found by this simulator',
    evidence='recorded',
    codes=frozenset({'wait.early'}),
    providers=ALL,
    matches=_waited_before_reply_content,
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
    WAIT_BEFORE_REPLY_CONTENT,
    RESERVATION_TAKEN_BY_OTHER_RESPONSE,
    REFUSED_TOOL_RESULTS_REQUEST,
    ANCHORED_USER_TURNS,
    REPEATED_TERMINAL,
    LATE_CANCEL_DROPS_CONTENT,
    SENT_BEFORE_REPLY_STARTED,
    SPEAKING_ORDER,
    COMMITTED_BY_HAND_UNDER_SERVER_VAD,
    LOST_UNSTARTED_REQUEST,
    RECEIVE_LOOP_SEND_FAILURE,
]
"""Checked in order: the more specific findings for a code come before the more general ones."""


def _parallel_calls(sim: Simulation, violation: InvariantViolation) -> bool:
    return any(len(response.tool_calls) > 1 for response in sim.truth.responses.values())


def _gemini_behavior(sim: Simulation, name: str) -> bool:
    return bool(getattr(getattr(sim, 'behavior', None), name, False))


GEMINI = frozenset({'gemini'})


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


def _continued_after_calling(sim: Simulation) -> bool:
    """A response went on after its first tool call: it said more, or called another tool in a later message."""
    truth = sim.truth
    return any(
        any(truth.word_seq[word] > first for word in response.words)
        or any(truth.tool_calls[call_id].seq > first for call_id in response.tool_calls)
        for response in truth.responses.values()
        if response.tool_calls
        for first in [truth.tool_calls[response.tool_calls[0]].seq]
    )


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


def _results_answered_together(sim: Simulation, violation: InvariantViolation) -> bool:
    """One reply answered the results of several tool calls: parallel calls, or the calls of separate delegations."""
    truth = sim.truth
    return _parallel_calls(sim, violation) or any(
        sum((input_ := truth.input(key)) is not None and input_.kind == 'tool_output' for key in response.answers) > 1
        for response in truth.responses.values()
    )


LIVE_BATCH_RESERVATIONS = Finding(
    id='SIM-6',
    title=(
        "GPT-Live answers the results of several tool calls (a delegation's parallel calls, or the calls of "
        'delegations in a row) with one reply, but the session reserves a reply per result, so `wait_for_reply()` '
        'hangs'
    ),
    tracked_by='reply obligations resolved by the reply that answers them (#8765 fixed this for the other providers, but not GPT-Live); found by this simulator',
    evidence='simulated',
    codes=frozenset({'wait.hang'}),
    providers=frozenset({'gpt-live'}),
    matches=_results_answered_together,
)


def _answered_together(sim: Simulation, violation: InvariantViolation) -> bool:
    """Some reply answered a typed turn and something else it was asked to answer, at once."""
    truth = sim.truth

    def answered(response: TruthResponse) -> list[TruthInput]:
        # The session reserves a reply for every soliciting send and every tool result.
        inputs = [truth.input(key) for key in response.answers]
        return [input_ for input_ in inputs if input_ is not None and (input_.solicits or input_.kind == 'tool_output')]

    return any(
        len(inputs) > 1 and any(input_.kind == 'text' for input_ in inputs)
        for inputs in map(answered, truth.responses.values())
    )


LIVE_QUEUED_TEXT_RESERVATIONS = Finding(
    id='SIM-7',
    title=(
        'GPT-Live answers text turns sent before it speaks with one reply (text is context on its timeline), but each '
        'keeps a reservation, so `wait_for_reply()` hangs (the one-reply answer is a guess, unconfirmed live)'
    ),
    tracked_by='reply obligations resolved by the reply that answers them; found by this simulator',
    evidence='simulated',
    codes=frozenset({'wait.hang'}),
    providers=frozenset({'gpt-live'}),
    matches=_answered_together,
)

LIVE_ABANDONED_CALL_RESERVATIONS = Finding(
    id='SIM-9',
    title=(
        "when a GPT-Live delegation's backend gives up, the results of the calls it had asked for are dropped by the "
        'connection, but the session still reserved a reply for each, so `wait_for_reply()` hangs'
    ),
    tracked_by='adapters report the reply obligations a provider voids; found by this simulator',
    evidence='simulated',
    codes=frozenset({'wait.hang'}),
    providers=frozenset({'gpt-live'}),
    matches=lambda sim, violation: getattr(getattr(sim, 'server', None), 'backends_failed', 0) > 0,
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


def _refused_while_speaking(sim: Simulation, violation: InvariantViolation) -> bool:
    input_ = sim.truth.input(violation.context.get('input', ''))
    return (
        input_ is not None
        and input_.kind == 'speech'
        and any(other.rejected and other.refused_read is not None for other in sim.truth.inputs)
    )


REFUSED_CONTEXT_MISFILES_SPEECH = Finding(
    id='SIM-14',
    title=(
        'context sent while the user is speaking (before the session read `speech_started`) and refused by the '
        'provider takes the spoken turn with it: the turn is filed after its own reply'
    ),
    tracked_by=(
        'history ordered by what the provider saw, not by the send a turn was anchored to: the session falls back '
        'to appending a turn whose anchor was withdrawn; found by this simulator'
    ),
    evidence='simulated',
    codes=frozenset({'history.order'}),
    providers=OPENAI_PROTOCOL,
    matches=_refused_while_speaking,
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


def _speech_cleared_after_barge_in(sim: Simulation, violation: InvariantViolation) -> bool:
    started = sim.truth.speech_started
    return any(key not in sim.truth.speech_committed for key in started) and any(
        operation.name == 'clear_audio' for operation in sim.operations
    )


CLEARED_BARGE_IN = Finding(
    id='SIM-16',
    title=(
        'a request deferred behind a response that server VAD then cut off is dropped for the barge-in, but if '
        'the app clears the buffered speech (`clear_audio`) no spoken turn follows, and the dropped request keeps '
        'its reservation, so `wait_for_reply()` hangs'
    ),
    tracked_by='reply reservations released with the request the barge-in dropped; found by this simulator after #8765',
    evidence='recorded',
    codes=frozenset({'wait.hang'}),
    providers=OPENAI_PROTOCOL,
    matches=_speech_cleared_after_barge_in,
)


def _pair_answered_with_the_turn_that_cut_it_off(sim: Simulation) -> bool:
    """Two calls whose turn a user turn (typed or spoken) cut off, answered by one response together with that turn."""
    truth = sim.truth
    for cut in truth.responses.values():
        if len(cut.tool_calls) != 2 or cut.status != 'cancelled':
            continue
        # (A spoken turn is committed after it cut the response off, so only its start bounds it.)
        turns = {
            input_.key for input_ in truth.inputs if input_.kind in ('text', 'speech') and input_.seq > cut.seq_start
        }
        if any(
            set(cut.tool_calls) <= set(answer.answers) and turns & set(answer.answers)
            for answer in truth.responses.values()
        ):
            return True
    return False


ASYNC_BATCH_OF_THREE = Finding(
    id='SIM-19',
    title=(
        'with asynchronous (`NON_BLOCKING`) Gemini tool calls, a `tool_call` message of three or more calls leaves '
        'a reservation after the model answered the batch, so `wait_for_reply()` hangs (two calls are fine, unless a '
        'user turn, typed or spoken, cut their turn off and one answer covers both with it; no '
        'recording has an async batch, so the fake answering one once may be what is wrong)'
    ),
    tracked_by='one reply per batch (#8765) also for asynchronous calls; found by this simulator',
    evidence='simulated',
    codes=frozenset({'wait.hang'}),
    providers=GEMINI,
    matches=lambda sim, violation: (
        _gemini_behavior(sim, 'talks_through_tool_calls')
        and (
            any(len(response.tool_calls) > 2 for response in sim.truth.responses.values())
            or _pair_answered_with_the_turn_that_cut_it_off(sim)
        )
    ),
)

LIVE_REPLY_SPLIT_BY_TOOL_ROUND = Finding(
    id='SIM-18',
    title=(
        'on GPT-Live, a spoken reply that goes on across a delegated tool round is recorded in two pieces around '
        "the tool's return (the GPT-Live counterpart of #8760)"
    ),
    tracked_by='per-response-id state, so a response the session already recorded can be continued; found by this simulator',
    evidence='recorded',
    codes=frozenset({'response.duplicated'}),
    providers=frozenset({'gpt-live'}),
    matches=lambda sim, violation: _continued_after_calling(sim),
)

EXTENDED_THINKING_PARALLEL_CALLS = Finding(
    id='SIM-17',
    title=(
        'on `gemini-3.8-live-extended-thinking` (which runs every call asynchronously), a batch of parallel calls '
        'leaves a reservation after the model answered it, so `wait_for_reply()` hangs (no recording has an async '
        'batch: the fake answers one once, after its last result, where the adapter expects an answer per result)'
    ),
    tracked_by='#8765 follow-up (one reply per batch, also for async calls); found by this simulator',
    evidence='simulated',
    codes=frozenset({'wait.hang'}),
    providers=GEMINI,
    matches=lambda sim, violation: _gemini_behavior(sim, 'stalls_in_progress') and _parallel_calls(sim, violation),
)


BARGE_IN_ON_A_TOOL_ROUND = Finding(
    id='SIM-20',
    title=(
        'server VAD cuts off a response that called a tool while a request is deferred behind it: the request is '
        'dropped for the barge-in but keeps its reservation, so `wait_for_reply()` hangs (the part of OR3 #8765 left)'
    ),
    tracked_by='reply reservations released with the request the barge-in dropped; found by this simulator',
    evidence='recorded',
    codes=frozenset({'wait.hang'}),
    providers=OPENAI_PROTOCOL,
    matches=lambda sim, violation: (
        getattr(sim, 'deferred_requests', 0) > 0
        and any(response.tool_calls and response.status == 'cancelled' for response in sim.truth.responses.values())
    ),
)


def _refusal_around_a_reconnect(sim: Simulation) -> bool:
    """A refusal the next connection loss followed with no response started in between: nothing released its request."""
    truth = sim.truth
    for input_ in truth.inputs:
        if input_.refused_at is None:
            continue
        refused = input_.refused_at
        loss = next((loss for loss in truth.connection_losses if loss > refused), None)
        if loss is not None and not any(refused < response.seq_start < loss for response in truth.responses.values()):
            return True
    return False


LOST_REFUSAL = Finding(
    id='SIM-21',
    title=(
        'a request for a response the provider refuses around a reconnect (the refusal lost with the connection, or '
        'read just before it drops) leaves a reservation neither re-asked nor released, so `wait_for_reply()` hangs'
    ),
    tracked_by='a reconnect resolves the reply obligations its connection lost; found by this simulator',
    evidence='simulated',
    codes=frozenset({'wait.hang'}),
    providers=OPENAI_PROTOCOL,
    matches=lambda sim, violation: _refusal_around_a_reconnect(sim),
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


def _barged_in_without_a_vad_reply(sim: Simulation) -> bool:
    """Server VAD, set not to answer, cut off a response while a request was deferred behind it (or a tool result awaited its reply)."""
    options = getattr(sim, 'openai', None)
    return (
        options is not None
        and options.turn_detection == 'server_vad'
        and options.vad_interrupts
        and not options.vad_responds
        and (
            getattr(sim, 'deferred_requests', 0) > 0
            or any(input_.kind == 'tool_output' and input_.answered_by is None for input_ in sim.truth.inputs)
        )
        and any(response.status == 'cancelled' for response in sim.truth.responses.values())
        # The user's turn is still there for VAD to answer (a cleared one is SIM-16's).
        and any(sim.truth.input(key) is not None for key in sim.truth.speech_started)
    )


BARGE_IN_WITHOUT_A_VAD_REPLY = Finding(
    id='SIM-32',
    title=(
        "with server VAD's `create_response` off, a request deferred behind a response the user barged in on (or the "
        "reply to that response's tool results) is dropped as if VAD would answer the turn, but nothing does, so "
        '`wait_for_reply()` hangs'
    ),
    tracked_by='a request dropped for a barge-in only when server VAD answers the turn; found by this simulator',
    evidence='simulated',
    codes=frozenset({'wait.hang'}),
    providers=frozenset({'openai', 'azure'}),
    matches=lambda sim, violation: _barged_in_without_a_vad_reply(sim),
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


def _server_vad(sim: Simulation) -> bool:
    """Whether the simulated OpenAI-protocol session runs server VAD (xAI push-to-talk reports speech too)."""
    options = getattr(sim, 'openai', None)
    return options is not None and options.turn_detection == 'server_vad'


def _clear_after_an_unread_vad_commit(sim: Simulation) -> bool:
    """Without transcription, the app cleared the buffer after server VAD committed a turn, before reading that it had."""
    if not _server_vad(sim) or getattr(sim, 'openai').transcription:
        return False
    truth = sim.truth
    clears = [operation.issued for operation in sim.operations if operation.name == 'clear_audio']
    # Turns server VAD committed (not the app), and when the client read that it had.
    vad_turns = [
        (started, input_.committed_read)
        for key, started in truth.speech_started.items()
        if key in truth.speech_committed and (input_ := truth.input(key)) is not None and not input_.committed_by_client
    ]
    return any(started < issued and (read is None or issued < read) for started, read in vad_turns for issued in clears)


def _push_to_talk_audio_after_a_repeated_terminal(sim: Simulation) -> bool:
    options = getattr(sim, 'openai', None)
    return (
        options is not None
        and options.dialect == 'xai'
        and options.turn_detection == 'manual'
        and bool(sim.truth.repeated_terminals)
        and any(operation.name == 'send_audio' for operation in sim.operations)
    )


PUSH_TO_TALK_AUDIO_AFTER_A_REPEATED_TERMINAL = Finding(
    id='SIM-36',
    title=(
        'on xAI push-to-talk, audio sent after a `response.done` the server repeated is held behind a reply that '
        'already ended: the current session loses the turn, and the new core (in shadow) leaves `wait_for_reply()` '
        'hanging, or holds back the reply after it (a repeated terminal is a robustness fault no provider was '
        'recorded sending, see 8801)'
    ),
    tracked_by="#9070's held audio released by the response that ended, not by the next terminal; found by this simulator",
    evidence='simulated',
    codes=frozenset({'history.turn_missing', 'wait.hang', 'response.missing', 'usage.requests'}),
    providers=frozenset({'xai'}),
    matches=lambda sim, violation: _push_to_talk_audio_after_a_repeated_terminal(sim),
)


CLEAR_AFTER_AN_UNREAD_VAD_COMMIT = Finding(
    id='SIM-35',
    title=(
        "without input transcription, a `clear_audio()` made after server VAD committed the user's turn, but before "
        'the session read that it had, drops the turn: the provider keeps it, and answers it, but history never '
        'records it'
    ),
    tracked_by='user turns recorded from the provider committing them; found by this simulator',
    evidence='simulated',
    codes=frozenset({'history.turn_missing'}),
    providers=OPENAI_PROTOCOL,
    matches=lambda sim, violation: _clear_after_an_unread_vad_commit(sim),
)


HAND_COMMIT_UNANSWERED_AT_THE_END = Finding(
    id='SIM-34',
    title=(
        'a spoken turn `commit_audio()` committed is lost when the connection drops, or the session closes or ends (on '
        'a failed send, an exceeded limit), before its reply is over: always in the new session core (in shadow), '
        'and in the current session on xAI, whose held commit goes out with the request'
    ),
    tracked_by='user turns recorded from the provider committing them; found by this simulator on #9422',
    evidence='simulated',
    codes=frozenset({'history.turn_missing'}),
    providers=OPENAI_PROTOCOL,
    matches=lambda sim, violation: _hand_commit_unanswered_at_the_end(sim),
)


def _audio_after_a_clear(sim: Simulation) -> bool:
    """Microphone audio was on its way at the same time as a clear or commit the app made (the two raced)."""

    def span(operation: Operation) -> tuple[int, float]:
        return operation.issued, operation.completed if operation.completed is not None else float('inf')

    audio = [span(operation) for operation in sim.operations if operation.name == 'send_audio']
    buffer_ops = [span(operation) for operation in sim.operations if operation.name in ('clear_audio', 'commit_audio')]
    return any(start < end_b and start_b < end for start, end in audio for start_b, end_b in buffer_ops)


AUDIO_AFTER_A_CLEAR = Finding(
    id='SIM-26',
    title=(
        'audio still streaming from the microphone when the app calls `clear_audio()` or `commit_audio()` lands after '
        'it, and the turn a later `commit_audio()` commits on the provider is never recorded: the session thinks the '
        'buffer is empty'
    ),
    tracked_by='user turns recorded from the provider committing them; found by this simulator',
    evidence='simulated',
    codes=frozenset({'history.turn_missing'}),
    providers=OPENAI_PROTOCOL,
    matches=lambda sim, violation: _audio_after_a_clear(sim),
)


def _hand_commit_under_server_vad(sim: Simulation, violation: InvariantViolation) -> bool:
    if (
        not _server_vad(sim)
        or not sim.truth.speech_started
        or not any(operation.name == 'commit_audio' for operation in sim.operations)
    ):
        return False
    if violation.code != 'history.order':
        return True
    input_ = sim.truth.input(violation.context.get('input', ''))
    return input_ is not None and input_.committed_by_client


HAND_COMMIT_UNDER_SERVER_VAD = Finding(
    id='SIM-27',
    title=(
        'a `commit_audio()` while server VAD is hearing the user commits what was buffered as a turn of its own, and '
        'VAD commits another when the user stops, but the session records only one of the two; on xAI, which adds '
        'the VAD turn when it hears speech start, the hand-committed one is recorded where its audio began, before '
        'a reply that ended before it was committed'
    ),
    tracked_by='user turns recorded from the provider committing them (the rest of OR9); found by this simulator',
    evidence='simulated',
    codes=frozenset({'history.turn_missing', 'history.order'}),
    providers=OPENAI_PROTOCOL,
    matches=lambda sim, violation: _hand_commit_under_server_vad(sim, violation),
)


TURN_LOST_AT_CLOSE = Finding(
    id='SIM-25',
    title=(
        'a spoken turn committed with input transcription on, whose transcript never arrives (the session closes or '
        'ends on an error, or the connection drops, first), is never recorded: it waits for a transcript that never comes (surfaced by '
        'Macroscope for closing after `commit_audio()`). The new session core (in shadow) also holds back the replies '
        'after it, so a reply after a reconnect is missing too (and is counted as a request it does not record)'
    ),
    tracked_by='user turns recorded from the provider committing them, not from their transcript; found reviewing #9070',
    evidence='recorded',
    codes=frozenset({'history.turn_missing', 'response.missing', 'usage.requests'}),
    providers=OPENAI_PROTOCOL,
    matches=lambda sim, violation: _turn_without_its_transcript(sim),
)


def _sent_during_a_lost_reply(sim: Simulation, violation: InvariantViolation) -> bool:
    """Everything the re-dialed conversation lacks is a user input sent while a reply the drop cut off was in flight."""
    lost = [response for response in sim.truth.responses.values() if response.lost]

    def held_back(fingerprint: str) -> bool:
        role, _, key = fingerprint.partition(':')
        input_ = sim.truth.input(key)
        return (
            role == 'user'
            and input_ is not None
            and any(response.connection == input_.connection and response.seq_start < input_.seq for response in lost)
        )

    return all(held_back(fingerprint) for fingerprint in violation.context['missing'])


INPUT_HELD_BEHIND_A_LOST_REPLY = Finding(
    id='SIM-28',
    title=(
        'an input sent while a reply is in flight, which history holds back until that reply is final, is left out of '
        'the replay when a drop cuts the reply off: history records it, but the re-dialed conversation never gets it'
    ),
    tracked_by='replaying history only once the turn the drop cut off is settled; found by this simulator',
    evidence='recorded',
    codes=frozenset({'history.not_restored'}),
    providers=OPENAI_PROTOCOL,
    matches=_sent_during_a_lost_reply,
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


def _request_commit_spoken_over(sim: Simulation) -> bool:
    """xAI push-to-talk without transcription: the user spoke again after a request for a response committed a turn."""
    options = getattr(sim, 'openai', None)
    # (A send that asks for a response sends a request too.)
    requests = [
        operation.issued
        for operation in sim.operations
        if operation.name in ('create_response', 'send_text', 'send_image_respond')
    ]
    audio = [operation.issued for operation in sim.operations if operation.name == 'send_audio']
    return (
        options is not None
        and options.turn_detection == 'manual'
        and not options.transcription
        and any(request < sent for request in requests for sent in audio)
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


def _xai_push_to_talk_speech(sim: Simulation, violation: InvariantViolation) -> bool:
    options = getattr(sim, 'openai', None)
    input_ = sim.truth.input(violation.context.get('input', ''))
    return options is not None and options.turn_detection == 'manual' and input_ is not None and input_.kind == 'speech'


PUSH_TO_TALK_ORDER = Finding(
    id='SIM-31',
    title=(
        "on xAI push-to-talk, a spoken turn the session held back behind a reply (#9070's held audio and commit) is "
        'recorded on the wrong side of a reply: history does not follow the order xAI got them in'
    ),
    tracked_by="history in the provider's conversation order (the refactor); found gating #9070",
    evidence='simulated',
    codes=frozenset({'history.order'}),
    providers=frozenset({'xai'}),
    matches=_xai_push_to_talk_speech,
)


PUSH_TO_TALK_TURN_SPOKEN_OVER = Finding(
    id='SIM-29',
    title=(
        'on xAI push-to-talk without input transcription, the turn a request for a response commits is never recorded '
        'when the user speaks again during its reply (which stops the reply with no `response.done`)'
    ),
    tracked_by='user turns recorded from the provider committing them; found gating #9070',
    evidence='simulated',
    codes=frozenset({'history.turn_missing'}),
    providers=frozenset({'xai'}),
    matches=lambda sim, violation: _request_commit_spoken_over(sim),
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
    """A response the provider started on its own was read after the input went out, before any of ours was."""
    key = violation.context.get('input')
    if not isinstance(key, str) or (input_ := sim.truth.input(key)) is None:
        return False
    issued = min((operation.issued for operation in sim.operations if operation.key == key), default=input_.seq)
    responses = sim.truth.responses.values()
    first_echo = min(
        (r.started_read for r in responses if r.trigger == 'create' and r.started_read is not None), default=None
    )
    return any(
        response.trigger != 'create'
        and response.started_read is not None
        and response.started_read > issued
        and (first_echo is None or response.started_read < first_echo)
        for response in responses
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


KNOWN_FINDINGS.extend(
    [
        FRAME_CUT_OFF_BY_CLOSE,
        NON_AUDIO_SEND_DURING_RECONNECT,
        TURN_LOST_AT_CLOSE,
        INPUT_HELD_BEHIND_A_LOST_REPLY,
        PUSH_TO_TALK_TURN_SPOKEN_OVER,
        CLEAR_AFTER_AN_UNREAD_VAD_COMMIT,
        PUSH_TO_TALK_AUDIO_AFTER_A_REPEATED_TERMINAL,
        HAND_COMMIT_UNANSWERED_AT_THE_END,
        PUSH_TO_TALK_ORDER,
        DEFERRED_REQUEST_DROPPED_AFTER_SPEECH,
        AUDIO_AFTER_A_CLEAR,
        HAND_COMMIT_UNDER_SERVER_VAD,
        BARGE_IN_WITHOUT_A_VAD_REPLY,
        LOST_REFUSAL,
        TERMINAL_DISCARDED_WITH_THE_CONNECTION,
        BARGE_IN_ON_A_TOOL_ROUND,
        REFUSED_CONTEXT_MISFILES_SPEECH,
        PARKED_ERROR_LEAVES_REQUEST_OWED,
        CLEARED_BARGE_IN,
        EXTENDED_THINKING_PARALLEL_CALLS,
        LIVE_REPLY_SPLIT_BY_TOOL_ROUND,
        ASYNC_BATCH_OF_THREE,
        LIVE_BATCH_RESERVATIONS,
        LIVE_QUEUED_TEXT_RESERVATIONS,
        LIVE_RAW_CLOSE_ERROR,
        LIVE_ABANDONED_CALL_RESERVATIONS,
        ASYNC_SPEECH_IN_FLIGHT_AT_THE_RESULT,
        GEMINI_ASYNC_TOOL_ROUND,
        ASYNC_RESULT_CUT_IN_ENDS_THE_WAIT,
        CUT_OFF_TURN_COMPLETE,
        # The general reservation leaks last: a more specific finding explains a hang better.
        LOST_RESPONSE_RESERVATION,
        PROVIDER_REPLY_BEFORE_ANY_ECHO,
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
