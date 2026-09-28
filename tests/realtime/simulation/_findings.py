"""Known invariant violations on current main, each tied to the PR, or the structural change, that fixes it.

A violation is matched against this registry by invariant code, provider, and a predicate over the
simulation that recognizes the finding's trigger. A match is tolerated in the default exploration mode
(and counted in Hypothesis' statistics) so the suite stays green on main while anything *new* still
fails; `REALTIME_SIMULATION_STRICT=1` reports them all. Each finding also has a pinned, minimal scenario
in `test_simulation.py`, marked as a strict expected failure, which starts passing when the fix lands:
then delete the finding here and turn its scenario into an ordinary regression test.

Finding ids are the ones used in the realtime stress reports and review rounds (`OR*` OpenAI Realtime,
`G*` Gemini Live, `L*` GPT-Live, `87xx #n` the n-th finding of a Codex review of that PR), so a violation
here can be traced back to its write-up.
"""

from __future__ import annotations as _annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._simulation import InvariantViolation, Simulation
    from ._truth import TruthInput, TruthResponse

Predicate = Callable[['Simulation', 'InvariantViolation'], bool]


@dataclass(frozen=True)
class Finding:
    id: str
    """The finding's reference in the stress reports and reviews."""
    title: str
    tracked_by: str
    """The open PR that fixes it, or the structural change that would."""
    codes: frozenset[str]
    providers: frozenset[str]
    matches: Predicate

    def __str__(self) -> str:
        return f'{self.id}: {self.title} (tracked by {self.tracked_by})'


def _inserted_user_speech(sim: Simulation, violation: InvariantViolation) -> bool:
    from ._invariants import is_user_speech_request

    return is_user_speech_request(violation.context['message'])


ALL = frozenset({'openai', 'azure', 'xai', 'gemini', 'gpt-live'})
OPENAI_PROTOCOL = frozenset({'openai', 'azure', 'xai'})

MERGED_REQUESTS_LEAK = Finding(
    id='OR3',
    title=(
        'requests for a response the connection defers behind an active one are merged (or dropped for a barge-in), '
        'but each keeps its reservation, so `wait_for_reply()` hangs'
    ),
    tracked_by='#8765',
    codes=frozenset({'wait.hang'}),
    providers=OPENAI_PROTOCOL,
    matches=lambda sim, violation: sim.truth.merged_requests > 0 or getattr(sim, 'deferred_requests', 0) > 0,
)


def _tool_failed(sim: Simulation, violation: InvariantViolation) -> bool:
    from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded

    from ._invariants import SimulatedToolError

    # A tool that raised or ran out of retries, or a tool result whose reservation ran into `request_limit`.
    return 'error' in sim.tools.settled.values() or isinstance(
        sim.consumer_error, (SimulatedToolError, UsageLimitExceeded, UnexpectedModelBehavior)
    )


RAISING_TOOL_HANG = Finding(
    id='OR8',
    title=(
        'a tool that raises or runs out of retries (or whose result runs into `request_limit`) parks the error but '
        'leaves the exchange open, so `wait_for_reply()` hangs while the session keeps running'
    ),
    tracked_by='#8765',
    codes=frozenset({'wait.hang'}),
    providers=ALL,
    matches=_tool_failed,
)


def _tool_results_request_refused(sim: Simulation, violation: InvariantViolation) -> bool:
    return any(input_.kind == 'tool_output' and input_.refused_read is not None for input_ in sim.truth.inputs)


REFUSED_TOOL_RESULTS_REQUEST = Finding(
    id='SIM-12',
    title=(
        'a response request for tool results that the provider refuses keeps its reply reservation (a refused request '
        'for a user turn releases it), so `wait_for_reply()` hangs'
    ),
    tracked_by='reply reservations resolved by the response that answers them (#8765); found by this simulator',
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
        "response's content as stragglers: history records it empty, though the model said it and the provider kept it"
    ),
    tracked_by='per-response-id state: a cancel targets a response id, and is a no-op once that response is done; found by this simulator',
    codes=frozenset({'response.truncated', 'response.missing'}),
    providers=OPENAI_PROTOCOL,
    matches=_late_cancel,
)

ANCHORED_USER_TURNS = Finding(
    id='E',
    title='a user turn is inserted into already-recorded history where it started (snapshots are not prefixes of later ones)',
    tracked_by='history projected from an append-only event log, so snapshots are prefixes of later ones',
    codes=frozenset({'history.inserted'}),
    providers=ALL,
    matches=_inserted_user_speech,
)

REPEATED_TERMINAL = Finding(
    id='8801',
    title=(
        'a repeated or late `response.done` (and its usage) lands on whichever response the session is assembling: '
        'counted again, recorded as an empty response, or stamped onto the next response (8801 #1, #4, #7)'
    ),
    tracked_by='#8801 (per-response-id session state; adapters report one terminal per response)',
    codes=frozenset(
        {
            'codec.content_after_terminal',
            'codec.duplicate_terminal',
            'history.order',
            'history.tool_round_order',
            'response.duplicated',
            'response.mixed',
            'response.truncated',
            'usage.total',
            'usage.attribution',
            'usage.requests',
            'wait.early',
        }
    ),
    providers=OPENAI_PROTOCOL,
    matches=lambda sim, violation: (
        sim.truth.repeated_terminals_read > 0 or getattr(getattr(sim, 'server', None), 'late_terminals', 0) > 0
    ),
)

LOST_RESPONSE_RESERVATION = Finding(
    id='SIM-1',
    title=(
        'a reply lost with a dropped connection keeps its reservation: the reconnect does not ask for it again (it '
        'had started, or the provider resumes without it) and the session does not settle it, so `wait_for_reply()` hangs'
    ),
    tracked_by='a reconnect resolves the reply obligations its connection lost; found by this simulator',
    codes=frozenset({'wait.hang'}),
    providers=ALL,
    matches=lambda sim, violation: bool(sim.truth.connection_losses),
)

LOST_UNSTARTED_REQUEST = Finding(
    id='SIM-3',
    title=(
        'a reconnect re-asks for the requests deferred behind a lost, not-yet-started response, but not for that '
        'response itself, so its reservation leaks and `wait_for_reply()` hangs'
    ),
    tracked_by='an outbox that replays every unanswered request after a reconnect; found by this simulator',
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
        frame == 'response.create' and last_read == 'response.done' for frame, last_read, _ in network.failed_sends
    )


RECEIVE_LOOP_SEND_FAILURE = Finding(
    id='SIM-4',
    title=(
        'a deferred `response.create` that the connection sends while handling a `response.done` fails on a dying '
        "socket, and the whole frame is dropped: that response's usage and terminal never reach the session"
    ),
    tracked_by='an ordered outbox, so the receive loop never sends on the socket itself; found by this simulator',
    codes=frozenset({'usage.total', 'usage.attribution', 'response.missing', 'response.truncated', 'usage.requests'}),
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
    return any(
        operation.name == 'send_audio' and (response.content_read is None or operation.issued < response.content_read)
        for operation in sim.operations
    )


SPEAKING_ORDER = Finding(
    id='SIM-11',
    title=(
        'a spoken turn the user started before a response was over (VAD heard them start, or their audio began '
        'before the response said anything), but which the provider committed after that response ended, is '
        "recorded before it (speaking order, since #8764), while the provider's conversation has it after"
    ),
    tracked_by=(
        'a design question, not necessarily a bug: which order history follows when the two differ '
        '(history projected from an append-only event log would settle it); found by this simulator'
    ),
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
    codes=frozenset({'wait.early'}),
    providers=ALL,
    matches=_waited_before_reply_content,
)


def _reply_taken_by_earlier_response(sim: Simulation, violation: InvariantViolation) -> bool:
    """A response already under way when the input arrived ended the wait for the input's own reply."""
    input_ = sim.truth.input(violation.context.get('input', ''))
    if input_ is None:
        return False
    return any(
        response.seq_start < input_.seq and (response.seq_end is None or response.seq_end > input_.seq)
        for response in sim.truth.responses.values()
    )


RESERVATION_TAKEN_BY_OTHER_RESPONSE = Finding(
    id='8763c #3',
    title=(
        "a response already under way when a turn was sent (server VAD, a GPT-Live delegation) takes that turn's "
        "reservation, so `wait_for_reply()` returns when it ends, before the turn's own reply"
    ),
    tracked_by='reply obligations resolved only by the response that answers them',
    codes=frozenset({'wait.early'}),
    providers=ALL,
    matches=_reply_taken_by_earlier_response,
)

KNOWN_FINDINGS: list[Finding] = [
    WAIT_BEFORE_REPLY_CONTENT,
    RESERVATION_TAKEN_BY_OTHER_RESPONSE,
    RAISING_TOOL_HANG,
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

GEMINI_SPLIT_PARALLEL_CALLS = Finding(
    id='G2b',
    title='the calls of one Gemini `tool_call` frame are each recorded as a `ModelResponse` of their own',
    tracked_by='#8765',
    codes=frozenset({'response.duplicated'}),
    providers=GEMINI,
    matches=_parallel_calls,
)

GEMINI_BATCH_RESERVATIONS = Finding(
    id='G2a',
    title=(
        'Gemini answers a batch of tool results once, but the session reserves a reply per result, '
        'so `wait_for_reply()` hangs and `request_limit` trips early'
    ),
    tracked_by='#8765',
    codes=frozenset({'wait.hang'}),
    providers=GEMINI,
    matches=_parallel_calls,
)


def _cut_off_by_the_input(sim: Simulation, violation: InvariantViolation) -> bool:
    """The input the wait was owed a reply for cut off a model turn (or one the model hadn't started on yet)."""
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
        'related to #8766; found by this simulator'
    ),
    codes=frozenset({'wait.early'}),
    providers=GEMINI,
    matches=_cut_off_by_the_input,
)

GEMINI_EARLY_TURN_COMPLETE = Finding(
    id='8766',
    title=(
        'Vertex `gemini-live-2.5-flash` closes the tool-call turn before the answer, and the session takes that '
        'boundary as the end of the exchange: `wait_for_reply()` returns before the answer'
    ),
    tracked_by='#8766',
    codes=frozenset({'wait.early', 'wait.hang'}),
    providers=GEMINI,
    matches=lambda sim, violation: _gemini_behavior(sim, 'closes_tool_turn_separately'),
)


def _spoke_after_calling(sim: Simulation, violation: InvariantViolation) -> bool:
    """The model kept talking after an asynchronous tool call it made (before or after the result came back)."""
    truth = sim.truth
    return _gemini_behavior(sim, 'talks_through_tool_calls') and any(
        response.tool_calls
        and any(truth.word_seq[word] > truth.tool_calls[response.tool_calls[0]].seq for word in response.words)
        for response in truth.responses.values()
    )


GEMINI_ASYNC_TOOL_ROUND = Finding(
    id='8760',
    title=(
        'with asynchronous (`NON_BLOCKING`) Gemini tool calls, what the model says after the call, in the same turn, '
        "is recorded as a response of its own after the tool's result, as if it had spoken with the result in hand"
    ),
    tracked_by='#8760 (parked: per-response-id state, so a response the session already recorded can be continued)',
    codes=frozenset({'response.duplicated', 'history.tool_round_order'}),
    providers=GEMINI,
    matches=_spoke_after_calling,
)

GEMINI_RESUMED_SESSION_FORGETS_CALLS = Finding(
    id='G6',
    title=(
        'a tool call in flight at a drop is forgotten by the resumed Gemini session (its handle predates the call): '
        'the result goes nowhere and `wait_for_reply()` hangs'
    ),
    tracked_by='#8763',
    codes=frozenset({'wait.hang'}),
    providers=GEMINI,
    matches=lambda sim, violation: any(
        input_.kind == 'tool_output' and input_.answer_lost for input_ in sim.truth.inputs
    ),
)

LIVE_BATCH_RESERVATIONS = Finding(
    id='SIM-6',
    title=(
        'a GPT-Live delegation answers its parallel tool calls once, but the session reserves a reply per result, '
        'so `wait_for_reply()` hangs'
    ),
    tracked_by='reply obligations resolved by the reply that answers them (#8765 does not cover GPT-Live, which it predates); found by this simulator',
    codes=frozenset({'wait.hang'}),
    providers=frozenset({'gpt-live'}),
    matches=_parallel_calls,
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
        'keeps a reservation, so `wait_for_reply()` hangs (simulated behavior; not yet confirmed live)'
    ),
    tracked_by='reply obligations resolved by the reply that answers them; found by this simulator',
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
    codes=frozenset({'api.unexpected_error'}),
    providers=frozenset({'gpt-live'}),
    matches=_raw_transport_error,
)

SEND_DURING_RECONNECT = Finding(
    id='G3',
    title=(
        'a send that hits the dropped socket raises `RealtimeError` although the reconnect succeeds, so the documented '
        '`send_audio(microphone)` task (or a tool result) dies on every reconnect'
    ),
    tracked_by='#8763 (and the planned stopgap that drops audio frames sent mid-reconnect)',
    codes=frozenset({'send.failed_across_reconnect'}),
    providers=OPENAI_PROTOCOL | GEMINI,
    matches=lambda sim, violation: True,
)

KNOWN_FINDINGS.extend(
    [
        SEND_DURING_RECONNECT,
        LIVE_BATCH_RESERVATIONS,
        LIVE_QUEUED_TEXT_RESERVATIONS,
        LIVE_RAW_CLOSE_ERROR,
        LIVE_ABANDONED_CALL_RESERVATIONS,
        GEMINI_ASYNC_TOOL_ROUND,
        GEMINI_SPLIT_PARALLEL_CALLS,
        GEMINI_BATCH_RESERVATIONS,
        GEMINI_EARLY_TURN_COMPLETE,
        CUT_OFF_TURN_COMPLETE,
        GEMINI_RESUMED_SESSION_FORGETS_CALLS,
        # The general reservation leaks last: a more specific finding explains a hang better.
        MERGED_REQUESTS_LEAK,
        LOST_RESPONSE_RESERVATION,
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
