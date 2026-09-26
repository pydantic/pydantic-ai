"""The invariants every simulated session is checked against.

Each check reports a violation with a stable `code` and some context, which is what known findings are
matched on (see `_findings.py`). A violation that matches a known finding is recorded and tolerated
(unless `REALTIME_SIMULATION_STRICT` is set, or the simulation was built with `strict=True`); any other
raises `InvariantViolation`, carrying the trace that reproduces it.

The checks are black-box wherever the public API allows: history comes from `all_messages()`, usage
from `session.usage`, and waits are real `wait_for_reply()` calls whose return is judged against the
server's ground truth. The one internal hook is observational: the codec events a connection yields are
recorded on their way into the session, for the lifecycle contract.

Checked after every step:

- `history.mutated`: a message object that appeared in a snapshot changed afterwards;
- `history.removed`: a message disappeared, other than a refused input taken back or the request of a
  send that failed;
- `history.reordered` / `history.inserted`: recorded messages changed order, or a message appeared
  anywhere but the end, other than a tool return placed after its call (every snapshot should be a
  prefix of the next);
- `response.duplicated`: one server response is recorded as two `ModelResponse`s;
- `response.mixed`: one `ModelResponse` holds what two server responses said;
- `wait.early`: a `wait_for_reply()` returned while a reply it was owed was still coming;
- `codec.*`: the connection broke the lifecycle contract (`_conformance.py`);
- `api.unexpected_error`: a client call, or iterating the session, raised something other than the
  documented errors;
- `simulator.malformed_frame`: the simulated server sent something the real connection can't parse (a
  bug in the simulator, not the session).

Checked at rest (`settle()`):

- `wait.hang`: a `wait_for_reply()` is still waiting, or a new one doesn't return, with nothing left
  to wait for;
- `history.tool_pairing`: a tool call without exactly one return, a return without its call, or a
  return not directly after its call's response;
- `history.order`: a user input and a response recorded in the opposite order to the one the server
  saw them in, or two responses out of order;
- `history.rejected_kept`: an input the provider refused is still in history;
- `response.missing` / `response.truncated`: a response the server completed is missing from history,
  or recorded without all of what it said;
- `usage.total`: `session.usage` tokens differ from what the server billed in the reports the client read;
- `usage.attribution`: a response's recorded usage isn't what the server billed for it;
- `usage.requests`: `usage.requests` differs from the number of responses recorded (or, for a model
  that reports its requests with usage, the number it reported);
- `playback.cut_beyond_audio` / `playback.bad_truncate`: a reply is marked cut past the audio it had,
  or the provider had to refuse a truncation the session sent;
- `wire.duplicate` / `wire.order`: the server got a send twice, or one caller's sends out of order;
- `send.failed_across_reconnect`: a send failed on a dropped connection the session went on to replace;
- `history.roundtrip` / `history.handoff`: history doesn't survive serialization, or a standard
  `Agent.run(message_history=...)` refuses it.
"""

from __future__ import annotations as _annotations

import asyncio
import os
from collections import Counter
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from pydantic_ai.exceptions import UsageLimitExceeded, UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    RealtimeSessionErrorEvent,
    RetryPromptPart,
    SpeechPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.realtime import RealtimeError, RealtimeSession
from pydantic_ai.realtime.codec import RealtimeCodecEvent

from ._conformance import ConformanceIssue, LifecycleChecker
from ._truth import TruthInput, TruthResponse, response_numbers

if TYPE_CHECKING:
    from ._simulation import Simulation, Waiter

Violation = tuple[str, dict[str, Any]]
"""A violation's detail, and the context a known finding is recognized by."""


class SimulatedToolError(RuntimeError):
    """What a gated tool raises for `outcome='error'`: an unhandled tool failure, which ends the session."""


_EXPECTED_CLIENT_ERRORS = (RealtimeError, UserError, UsageLimitExceeded, SimulatedToolError, asyncio.CancelledError)
"""What a client call may raise: a lost connection, a misuse, an exceeded limit, or (from the next outbound
call, or iteration) the tool failure that ended the session."""

_SEND_FAILED = 'Realtime connection failed while sending'


def strict_from_environment() -> bool:
    """Whether known findings fail too (`REALTIME_SIMULATION_STRICT=1`), as when hunting for new ones."""
    return os.environ.get('REALTIME_SIMULATION_STRICT', '') not in ('', '0')


def response_text(message: ModelResponse) -> str:
    texts = [part.transcript for part in message.parts if isinstance(part, SpeechPart) and part.transcript]
    texts += [part.content for part in message.parts if isinstance(part, TextPart)]
    # Whitespace is not what these checks are about: a part can keep the space its first fragment led with.
    return ' '.join(' '.join(texts).split())


def request_keys(message: ModelRequest) -> list[str]:
    """Ground-truth keys of the user inputs and tool outputs a request carries."""
    keys: list[str] = []
    for part in message.parts:
        if isinstance(part, UserPromptPart):
            # Text is its own key; an image (the only other content the simulation sends) the tail of its data.
            contents = [part.content] if isinstance(part.content, str) else part.content
            keys += [item if isinstance(item, str) else getattr(item, 'base64', '')[-12:] for item in contents]
        elif isinstance(part, SpeechPart) and part.transcript:
            keys.append(part.transcript)
        elif isinstance(part, (ToolReturnPart, RetryPromptPart)):
            keys.append(part.tool_call_id)
    return keys


def is_tool_return_request(message: ModelMessage) -> bool:
    """A request carrying tool returns (and the content that follows them), which is placed after its call."""
    return isinstance(message, ModelRequest) and isinstance(message.parts[0], (ToolReturnPart, RetryPromptPart))


def is_user_speech_request(message: ModelMessage) -> bool:
    return isinstance(message, ModelRequest) and any(
        isinstance(part, SpeechPart) and part.speaker == 'user' for part in message.parts
    )


class Checker:
    """Keeps the observations the invariants need across steps, and runs them."""

    def __init__(self, sim: Simulation, *, strict: bool, enforce: frozenset[str] = frozenset()) -> None:
        self.sim = sim
        self.strict = strict
        self.enforce = enforce
        """Ids of known findings that raise anyway: the one a pinned scenario is about."""
        self.lifecycle = LifecycleChecker()
        self.codec_events: list[RealtimeCodecEvent] = []
        self.known_hits: list[tuple[str, str]] = []
        """`(finding id, invariant code)` for every tolerated violation, in order."""
        self._seen: dict[int, tuple[ModelMessage, bytes]] = {}
        self._previous: list[int] = []
        self._judged_waiters: set[int] = set()
        self._judged_operations = 0
        self._judged_events = 0
        self._judged_truncations = 0
        self._consumer_error_judged = False
        self._codec_issues: list[ConformanceIssue] = []

    def attach(self, session: RealtimeSession) -> None:
        """Observe the codec events the connection yields, on their way into the session."""
        handle = session._handle_pump_event  # pyright: ignore[reportPrivateUsage]

        async def observed(event: RealtimeCodecEvent) -> bool:
            self.codec_events.append(event)
            self._codec_issues += self.lifecycle.feed(event)
            self.sim.observe_codec_event(event)
            return await handle(event)

        session._handle_pump_event = observed  # pyright: ignore[reportPrivateUsage]

    def report(self, code: str, violations: Iterable[Violation]) -> None:
        """Report each violation: tolerated if it is a known finding (and not strict), raised otherwise."""
        from ._findings import matching_findings
        from ._simulation import FindingReproduced, InvariantViolation

        for detail, context in violations:
            violation = InvariantViolation(code, detail, self.sim.trace, context)
            findings = [finding.id for finding in matching_findings(self.sim, violation)]
            violation.findings = findings
            if self.enforce.intersection(findings):
                raise FindingReproduced(code, detail, self.sim.trace, context, findings=findings)
            if self.strict or not findings:  # pragma: no cover (only when the session breaks an invariant)
                raise violation
            self.known_hits.append((findings[0], code))

    # --- every step -------------------------------------------------------------------------------

    def check_step(self) -> None:
        session = self.sim.session
        assert session is not None
        issues, self._codec_issues = self._codec_issues, []
        for code in dict.fromkeys(issue.code for issue in issues):
            self.report(
                code,
                [(f'{issue.detail} (codec event #{issue.position})', {}) for issue in issues if issue.code == code],
            )
        self._check_errors()
        self._check_simulator_frames()
        messages = session.all_messages()
        self._check_history_stability(messages)
        self._check_response_identity(messages)
        waiters = [w for w in self.sim.waiters if w.returned is not None and w.index not in self._judged_waiters]
        self._judged_waiters.update(waiter.index for waiter in waiters)
        self.report('wait.early', [violation for waiter in waiters for violation in self._early_return(waiter)])

    def _check_simulator_frames(self) -> None:
        """The simulated server must only send frames the real connection can parse; anything else is a harness bug."""
        events, self._judged_events = self.sim.events[self._judged_events :], len(self.sim.events)
        self.report(
            'simulator.malformed_frame',
            [
                (event.message, {})
                for event in events
                if isinstance(event, RealtimeSessionErrorEvent) and 'Failed to parse' in event.message
            ],
        )

    def _check_errors(self) -> None:
        sim = self.sim
        done = [operation for operation in sim.operations if operation.done]
        judged, self._judged_operations = done[self._judged_operations :], len(done)
        unexpected = [
            (f'{operation.name} raised {operation.error!r}', {'operation': operation.name})
            for operation in judged
            if operation.error is not None and not isinstance(operation.error, _EXPECTED_CLIENT_ERRORS)
        ]
        if (error := sim.consumer_error) is not None and not self._consumer_error_judged:
            self._consumer_error_judged = True
            if not isinstance(error, _EXPECTED_CLIENT_ERRORS):
                unexpected.append((f'iterating the session raised {error!r}', {'operation': 'iterate'}))
        self.report('api.unexpected_error', unexpected)

    def _removal_allowed(self, message: ModelMessage) -> bool:
        """A refused input taken back, or the request of a send that failed."""
        truth = self.sim.truth
        failed_sends = {operation.key for operation in self.sim.operations if operation.error is not None}
        keys = request_keys(message) if isinstance(message, ModelRequest) else []
        return bool(keys) and all(
            key in failed_sends or ((input_ := truth.input(key)) is not None and input_.rejected) for key in keys
        )

    def _check_history_stability(self, messages: list[ModelMessage]) -> None:
        current = [id(message) for message in messages]
        mutated: list[Violation] = []
        for message in messages:
            dumped = ModelMessagesTypeAdapter.dump_json([message])
            before = self._seen.get(id(message), (message, dumped))[1]
            self._seen[id(message)] = (message, dumped)
            if before != dumped:  # pragma: lax no cover (only when the session breaks the invariant)
                changed = f'before: {before.decode()}\n  after:  {dumped.decode()}'
                mutated.append((f'a recorded {type(message).__name__} changed:\n  {changed}', {'message': message}))
        self.report('history.mutated', mutated)

        current_set, previous_set = set(current), set(self._previous)
        removed = [self._seen[message_id][0] for message_id in self._previous if message_id not in current_set]
        self.report(
            'history.removed',
            [
                (f'a recorded message disappeared: {self._describe(message)}', {'message': message})
                for message in removed
                if not self._removal_allowed(message)
            ],
        )
        survivors = [message_id for message_id in current if message_id in previous_set]
        kept = [message_id for message_id in self._previous if message_id in current_set]
        self.report('history.reordered', [('recorded messages changed order', {})] if survivors != kept else [])
        # New messages must all come after the last surviving one: the documented exception is a tool's
        # return, placed directly after its call.
        last_old = current.index(survivors[-1]) if survivors else 0
        self.report(
            'history.inserted',
            [
                (
                    f'{self._describe(messages[position])} was inserted at position {position} of {len(current)}, '
                    'before messages already recorded',
                    {'message': messages[position]},
                )
                for position in range(last_old)
                if current[position] not in previous_set and not is_tool_return_request(messages[position])
            ],
        )
        self._previous = current

    def _describe(self, message: ModelMessage) -> str:
        if isinstance(message, ModelRequest):
            return f'request {request_keys(message) or [part.part_kind for part in message.parts]}'
        return f'response {message.provider_response_id} {response_text(message)!r}'  # pragma: lax no cover

    def response_numbers(self, message: ModelResponse) -> set[int]:
        """The server responses a recorded `ModelResponse` holds content (or the id) of."""
        truth = self.sim.truth
        numbers = response_numbers(response_text(message))
        if (known := truth.responses.get(message.provider_response_id or '')) is not None:
            numbers.add(known.number)
        calls = [part.tool_call_id for part in message.parts if isinstance(part, ToolCallPart)]
        numbers.update(truth.responses[truth.tool_calls[call_id].response].number for call_id in calls)
        return numbers

    def _check_response_identity(self, messages: list[ModelMessage]) -> None:
        recorded = [
            (position, self.response_numbers(message))
            for position, message in enumerate(messages)
            if isinstance(message, ModelResponse)
        ]
        self.report(
            'response.mixed',
            [
                (
                    f'the response at {position} holds what server responses {sorted(numbers)} said: '
                    f'{self._describe(messages[position])}',
                    {'responses': sorted(numbers)},
                )
                for position, numbers in recorded
                if len(numbers) > 1
            ],
        )
        owners: dict[int, list[int]] = {}
        for position, numbers in recorded:
            for number in numbers:
                owners.setdefault(number, []).append(position)
        self.report(
            'response.duplicated',
            [
                (f'server response r{number} is recorded at {positions}', {'response': number})
                for number, positions in owners.items()
                if len(positions) > 1
            ],
        )

    # --- waits ------------------------------------------------------------------------------------

    def _exchange_resolved(self, response: TruthResponse, by: int, seen: set[str]) -> bool:
        """Whether `response` and everything its tool calls led to had ended, as far as the client could see, by `by`."""
        if response.key in seen or response.lost:  # pragma: lax no cover
            # A response lost with its connection is settled by the reconnect, not by a terminal.
            return True
        seen.add(response.key)
        if response.terminal_read is None or response.terminal_read > by:
            return False
        truth = self.sim.truth
        # A call the client never saw, or one from a response that was cancelled or failed, leads nowhere; nor
        # does one the provider cancelled. Any other has to be answered, and its answer spoken.
        calls = [
            call
            for call in map(truth.tool_calls.__getitem__, response.tool_calls)
            if call.read and response.status == 'completed' and not call.cancelled_by_server
        ]
        outputs = [truth.input(call.call_id) for call in calls]
        return all(output is not None and self._input_resolved(output, by, seen) for output in outputs)

    def _input_resolved(self, input_: TruthInput, by: int, seen: set[str]) -> bool:
        if (input_.refused_read is not None and input_.refused_read <= by) or input_.answer_lost:
            return True  # pragma: lax no cover
        # The response that answers it, or any the model gave after it arrived: a request the connection
        # folded into another (a merged or superseded `response.create`) is answered by whichever came next.
        return any(
            self._exchange_resolved(response, by, set(seen))
            for response in self.sim.truth.responses.values()
            if response.key == input_.answered_by or response.seq_start > input_.seq
        )

    def _early_return(self, waiter: Waiter) -> list[Violation]:
        """What a returned `wait_for_reply()` should still have been waiting for, if anything."""
        sim = self.sim
        returned = waiter.returned
        assert returned is not None
        truth = sim.truth
        if (
            waiter.error is not None
            or sim.close_requested is not None
            or any(loss >= waiter.started for loss in truth.connection_losses)
            or 'error' in sim.tools.settled.values()
            or sim.receive_ended
        ):
            # The session ended, lost state to a drop, or a tool failure ended the exchange: returning is right.
            return []
        context = {'waiter': waiter.index, 'started': waiter.started}
        # A reply to a turn it had asked for, by the time the wait began, before the provider saw it...
        soliciting = [
            operation.key
            for operation in sim.operations
            if operation.key in waiter.snapshot and operation.name in ('send_text', 'send_image_respond')
        ]
        inputs = [(key, truth.input(key)) for key in soliciting if key is not None]
        violations: list[Violation] = [
            (f'wait_for_reply() #{waiter.index} returned before the reply to {key!r} ended', {**context, 'input': key})
            for key, input_ in inputs
            if input_ is None or not self._input_resolved(input_, returned, set())
        ]
        # ...or a response it could see was under way when the wait began.
        violations += [
            (
                f'wait_for_reply() #{waiter.index} returned while {response.key} '
                '(already under way when the wait began) was still in progress',
                {**context, 'response': response.key},
            )
            for response in truth.responses.values()
            if response.started_read is not None
            and response.started_read < waiter.started
            and not self._exchange_resolved(response, waiter.started, set())
            and not self._exchange_resolved(response, returned, set())
        ]
        return violations

    # --- at rest ----------------------------------------------------------------------------------

    def check_at_rest(self) -> None:
        sim = self.sim
        self.check_step()
        session = sim.session
        assert session is not None
        messages = session.all_messages()
        self._check_tool_pairing(messages)
        self._check_order(messages)
        self._check_completeness(messages)
        self._check_usage(messages)
        self._check_playback(messages)
        roundtrip = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(messages))
        self.report(
            'history.roundtrip',
            [('history did not survive a serialization round trip', {})] if roundtrip != messages else [],
        )
        self._check_wire()
        if not session.closed:
            # Also once the event stream has ended: `wait_for_reply()` must never outlive the session's ability to reply.
            self._check_sends_survived()
            self._check_wait_liveness()

    def _check_tool_pairing(self, messages: list[ModelMessage]) -> None:
        calls = {
            part.tool_call_id: position
            for position, message in enumerate(messages)
            if isinstance(message, ModelResponse)
            for part in message.parts
            if isinstance(part, ToolCallPart)
        }
        returns = [
            (part.tool_call_id, position)
            for position, message in enumerate(messages)
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, (ToolReturnPart, RetryPromptPart))
        ]
        violations: list[Violation] = [
            (f'a return for {call_id!r}, which no recorded response called', {'call': call_id})
            for call_id, _ in returns
            if call_id not in calls
        ]
        violations += [
            (
                f'the return for {call_id!r} is at {position}, not directly after its call at {calls[call_id]}',
                {'call': call_id},
            )
            for call_id, position in returns
            if call_id in calls
            and not (
                calls[call_id] < position
                and all(is_tool_return_request(message) for message in messages[calls[call_id] + 1 : position])
            )
        ]
        # Every call has exactly one return, once no tool is still running.
        counts = Counter(call_id for call_id, _ in returns)
        violations += [
            (f'tool call {call_id!r} has {counts[call_id]} returns', {'call': call_id})
            for call_id in calls
            if counts[call_id] != 1 and not self.sim.tools.pending()
        ]
        self.report('history.tool_pairing', violations)

    def _check_order(self, messages: list[ModelMessage]) -> None:
        truth = self.sim.truth
        positions: dict[str, int] = {}
        responses: dict[str, int] = {}
        for position, message in enumerate(messages):
            if isinstance(message, ModelRequest):
                for key in request_keys(message):
                    positions.setdefault(key, position)
            else:
                for number in self.response_numbers(message):
                    responses.setdefault(truth.responses_by_number[number].key, position)
        self.report(
            'history.rejected_kept',
            [
                (f'{input_.key!r} was refused by the provider but is still in history', {'input': input_.key})
                for input_ in truth.inputs
                if input_.rejected and input_.refused_read is not None and input_.key in positions
            ],
        )
        # User turns (not tool returns, which are pinned after their call) against the responses around them.
        pairs = [
            (input_, positions[input_.key], truth.responses[key], position)
            for input_ in truth.inputs
            if not input_.rejected and input_.key in positions and input_.kind not in ('tool_output', 'create')
            for key, position in responses.items()
        ]
        violations: list[Violation] = [
            (
                f'{input_.key!r} reached the server before {response.key} started, but is recorded after it '
                f'({input_position} > {response_position})',
                {'input': input_.key, 'response': response.key},
            )
            for input_, input_position, response, response_position in pairs
            if input_.seq < response.seq_start and input_position > response_position
        ]
        violations += [
            (
                f'{input_.key!r} reached the server after {response.key} ended, but is recorded before it '
                f'({input_position} < {response_position})',
                {'input': input_.key, 'response': response.key},
            )
            for input_, input_position, response, response_position in pairs
            if response.seq_end is not None and input_.seq > response.seq_end and input_position < response_position
        ]
        ordered = [truth.responses[key] for key, _ in sorted(responses.items(), key=lambda item: item[1])]
        violations += [
            (f'{second.key} ended before {first.key} started but is recorded after it', {'response': second.key})
            for first, second in zip(ordered, ordered[1:])
            if second.seq_end is not None and second.seq_end < first.seq_start
        ]
        self.report('history.order', violations)

    def _check_completeness(self, messages: list[ModelMessage]) -> None:
        truth = self.sim.truth
        recorded: dict[int, list[str]] = {}
        for message in messages:
            if isinstance(message, ModelResponse):
                for number in self.response_numbers(message):
                    # A response recorded in pieces is `response.duplicated`'s business; here, only what it said.
                    recorded.setdefault(number, []).append(response_text(message))
        completed = [
            (response, ' '.join(filter(None, recorded.get(response.number, []))) or None)
            for response in truth.responses.values()
            if response.words and response.terminal_read is not None and response.status == 'completed'
        ]
        self.report(
            'response.missing',
            [
                (
                    f'{response.key} ({" ".join(response.words)!r}) completed but is not in history',
                    {'response': response.key},
                )
                for response, text in completed
                if response.number not in recorded
            ],
        )
        self.report(
            'response.truncated',
            [
                (
                    f'{response.key} said {" ".join(response.words)!r} but is recorded as {text!r}',
                    {'response': response.key},
                )
                for response, text in completed
                if response.number in recorded and text != ' '.join(response.words)
            ],
        )

    def _check_usage(self, messages: list[ModelMessage]) -> None:
        sim = self.sim
        session = sim.session
        assert session is not None
        if sim.receive_ended and sim.close_requested is None:
            # The session ended on its own (an exceeded usage limit, a lost connection): whatever it read
            # after that point was never going to be accounted.
            return
        truth = sim.truth
        billed = (
            sum(tokens[0] for tokens in truth.usage_read.values()),
            sum(tokens[1] for tokens in truth.usage_read.values()),
        )
        usage = session.usage
        recorded = (usage.input_tokens, usage.output_tokens)
        # Closing stops the session mid-frame: a report the connection had read ahead may go unaccounted, but
        # nothing may ever be counted twice.
        over = any(count > limit for count, limit in zip(recorded, billed))
        wrong = recorded != billed if sim.close_requested is None else over
        self.report(
            'usage.total',
            [(f'session.usage has {recorded} tokens (in, out), but the server billed {billed} in the reports read', {})]
            if wrong
            else [],
        )
        self.report(
            'usage.attribution',
            [
                (
                    f'{message.provider_response_id} is recorded with {message.usage.input_tokens} / '
                    f'{message.usage.output_tokens} tokens, but was billed {truth.usage_read[message.provider_response_id]}',
                    {'response': message.provider_response_id},
                )
                for message in messages
                if isinstance(message, ModelResponse)
                and message.provider_response_id in truth.usage_read
                and (message.usage.input_tokens, message.usage.output_tokens)
                != truth.usage_read[message.provider_response_id]
            ],
        )
        # A model that reports its requests with usage (GPT-Live's delegated backend) counts those instead.
        expected = sim.expected_requests()
        if expected is None:
            expected = sum(isinstance(message, ModelResponse) for message in session.new_messages())
        self.report(
            'usage.requests',
            [(f'usage.requests is {usage.requests}, but {expected} were made', {})]
            if usage.requests != expected
            else [],
        )

    def _check_playback(self, messages: list[ModelMessage]) -> None:
        """A cut lands inside the audio of the reply it cuts, and the provider can apply every truncation sent."""
        truth = self.sim.truth
        refused, self._judged_truncations = (
            truth.refused_truncations[self._judged_truncations :],
            len(truth.refused_truncations),
        )
        self.report(
            'playback.bad_truncate', [(f'the provider refused a truncation: {reason}', {}) for reason in refused]
        )
        cuts = [
            (message, part.interrupted_at_ms, self.response_numbers(message))
            for message in messages
            if isinstance(message, ModelResponse)
            for part in message.parts
            if isinstance(part, SpeechPart) and part.interrupted_at_ms is not None
        ]
        generated = {
            id(message): sum(truth.responses_by_number[number].audio_bytes for number in numbers)
            // self.sim.output_bytes_per_ms
            for message, _, numbers in cuts
        }
        self.report(
            'playback.cut_beyond_audio',
            [
                (
                    f'{self._describe(message)} is cut at {at_ms} ms, but only {generated[id(message)]} ms of it was generated',
                    {'responses': sorted(numbers)},
                )
                for message, at_ms, numbers in cuts
                if at_ms > generated[id(message)]
            ],
        )

    def _check_wire(self) -> None:
        """Each caller's sends reach the server once each, in the order they were made."""
        sim = self.sim
        arrivals: dict[str, list[int]] = {}
        for input_ in sim.truth.inputs:
            if input_.kind in ('text', 'context', 'image'):
                arrivals.setdefault(input_.key, []).append(input_.seq)
        # A send that failed after the frame went out is sent again: the client can't know it arrived.
        ambiguous = any(fault == 'ambiguous' for *_, fault in sim.failed_sends)
        self.report(
            'wire.duplicate',
            [
                (f'the server received {key!r} {len(seqs)} times', {'input': key})
                for key, seqs in arrivals.items()
                if len(seqs) > 1 and not ambiguous
            ],
        )
        callers: dict[str, list[tuple[int, int, str]]] = {}
        for operation in sim.operations:
            if operation.key in arrivals:
                callers.setdefault(operation.caller, []).append(
                    (operation.issued, arrivals[operation.key][0], operation.key)
                )
        orders = [
            (caller, [key for *_, key in sorted(sends)], [key for *_, key in sorted(sends, key=lambda send: send[1])])
            for caller, sends in callers.items()
        ]
        self.report(
            'wire.order',
            [
                (f'the {caller} caller sent {issued}, but the server received {arrived}', {'caller': caller})
                for caller, issued, arrived in orders
                if issued != arrived
            ],
        )

    def _check_sends_survived(self) -> None:
        """A send must not fail on a connection the session went on to replace: only a session that ended may refuse one."""
        if self.sim.receive_ended:
            return
        self.report(
            'send.failed_across_reconnect',
            [
                (
                    f'{operation.name} failed ({operation.error}), though the session reconnected and went on',
                    {'operation': operation.name},
                )
                for operation in self.sim.operations
                if isinstance(operation.error, RealtimeError) and operation.error.message.startswith(_SEND_FAILED)
            ],
        )

    def _check_wait_liveness(self) -> None:
        sim = self.sim
        stuck = [waiter.index for waiter in sim.waiters if waiter.returned is None]
        session = sim.session
        assert session is not None
        probe = sim.loop.create_task(session.wait_for_reply())
        sim.loop.run_until_idle()
        hung = not probe.done()
        probe.cancel()
        sim.loop.run_until_idle()
        if hung:
            stuck.append(len(sim.waiters))
        self.report(
            'wait.hang',
            [(f'wait_for_reply() {stuck} still waiting with nothing left to wait for', {'waiters': stuck})]
            if stuck
            else [],
        )
