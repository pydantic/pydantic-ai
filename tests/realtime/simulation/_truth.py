"""Ground truth: what the simulated provider actually saw, generated, and billed.

Every simulated server keeps one `GroundTruth`, independent of anything the client reports. The
invariants compare the session's history, usage, and waits against it. Everything the server makes is
named so a history can be mapped back onto it without guessing:

- a user text turn is the text the client sent (`t1`, `t2`, ...);
- a spoken user turn transcribes as `u1`, `u2`, ...;
- a response speaks words `r3w1 r3w2 ...` (so any recorded transcript names the response it came from)
  and calls tools with ids `call_1`, `call_2`, ...;
- usage is distinct per response, so a mis-attributed or double-counted report shows up in the sums.
"""

from __future__ import annotations as _annotations

import re
from dataclasses import dataclass, field
from typing import Literal

InputKind = Literal['text', 'context', 'image', 'speech', 'tool_output', 'create']
ResponseStatus = Literal['in_progress', 'completed', 'cancelled', 'failed', 'incomplete', 'lost']

_RESPONSE_WORD = re.compile(r'\br(\d+)w\d+\b')


def response_numbers(text: str | None) -> set[int]:
    """The server responses whose words appear in `text`."""
    return {int(number) for number in _RESPONSE_WORD.findall(text or '')}


@dataclass
class TruthInput:
    """Something the client sent that reached the server."""

    key: str
    kind: InputKind
    seq: int
    """When it joined the conversation, on the server's clock."""
    connection: int
    solicits: bool = False
    """Whether it asked the model for a reply."""
    client_index: int | None = None
    """The session's number for this input (the `event_id` echo), when the frame carried one."""
    rejected: bool = False
    answered_by: str | None = None
    refused_read: int | None = None
    """When the client read an error refusing this input, or the response it asked for."""
    answer_lost: bool = False
    """Whether a response answering this input was lost with its connection (what follows is a reconnect's call)."""


@dataclass
class TruthResponse:
    """One response the server generated."""

    key: str
    number: int
    connection: int
    seq_start: int
    trigger: Literal['create', 'vad', 'auto']
    answers: list[str] = field(default_factory=list[str])
    """Keys of the inputs this response answers."""
    user_turn: str | None = None
    seq_end: int | None = None
    status: ResponseStatus = 'in_progress'
    words: list[str] = field(default_factory=list[str])
    audio_bytes: int = 0
    tool_calls: list[str] = field(default_factory=list[str])
    input_tokens: int = 0
    output_tokens: int = 0
    started_read: int | None = None
    """When the client read the first frame of this response."""
    content_read: int | None = None
    """When the client read the first of this response's content (audio, transcript, a tool call)."""
    terminal_read: int | None = None
    """When the client read this response's terminal (and so could have recorded its usage)."""
    lost: bool = False
    """Whether the connection dropped before the client read this response's terminal."""
    stall_read: int | None = None
    """When the client read an `IN_PROGRESS` boundary ending this response's speech (Gemini extended thinking)."""


@dataclass
class ToolCallTruth:
    call_id: str
    response: str
    name: str
    output_received: bool = False
    cancelled_by_server: bool = False
    read: bool = False
    """Whether the client read the call (a cancelled response's calls can be dropped as stragglers)."""


@dataclass
class GroundTruth:
    """The provider-side record of a simulated conversation."""

    clock: int = 0
    """One clock for everything that happens, on either side: the server receiving or emitting, the client
    reading a frame, starting or finishing an operation. Ordering questions are answered on it."""
    inputs: list[TruthInput] = field(default_factory=list[TruthInput])
    responses: dict[str, TruthResponse] = field(default_factory=dict[str, TruthResponse])
    responses_by_number: dict[int, TruthResponse] = field(default_factory=dict[int, TruthResponse])
    tool_calls: dict[str, ToolCallTruth] = field(default_factory=dict[str, ToolCallTruth])
    connections: int = 0
    next_response_number: int = 1
    next_call_number: int = 1
    next_user_number: int = 1
    truncations: list[tuple[str, int]] = field(default_factory=list[tuple[str, int]])
    """`(item id, audio_end_ms)` for every truncation the server applied."""
    refused_truncations: list[str] = field(default_factory=list[str])
    """Why the server refused each truncation it was asked for."""
    connection_losses: list[int] = field(default_factory=list[int])
    """When each connection was lost (dropped, or refused a re-dial), on the shared clock."""
    usage_read: dict[str, tuple[int, int]] = field(default_factory=dict[str, tuple[int, int]])
    """`(input, output)` tokens per response, for every usage report the client read (first report wins)."""
    usage_reports_read: int = 0
    """How many usage-bearing frames the client read, duplicates included."""
    repeated_terminals_read: int = 0
    """How many terminals the client read for a response it had already read the terminal of."""
    merged_requests: int = 0
    """How many requests for a response the client folded into another one's single request."""

    def tick(self) -> int:
        self.clock += 1
        return self.clock

    def add_input(
        self, key: str, kind: InputKind, *, solicits: bool = False, client_index: int | None = None
    ) -> TruthInput:
        entry = TruthInput(
            key=key,
            kind=kind,
            seq=self.tick(),
            connection=self.connections,
            solicits=solicits,
            client_index=client_index,
        )
        self.inputs.append(entry)
        return entry

    def new_response(
        self, key: str | None = None, *, trigger: Literal['create', 'vad', 'auto'], answers: list[str] | None = None
    ) -> TruthResponse:
        number = self.next_response_number
        self.next_response_number += 1
        response = TruthResponse(
            key=key or f'resp_{number}',
            number=number,
            connection=self.connections,
            seq_start=self.tick(),
            trigger=trigger,
            answers=list(answers or []),
        )
        self.responses[response.key] = response
        self.responses_by_number[number] = response
        for input_ in self.inputs:
            if input_.key in response.answers and input_.answered_by is None:
                input_.answered_by = response.key
        return response

    def new_call_id(self) -> str:
        call_id = f'call_{self.next_call_number}'
        self.next_call_number += 1
        return call_id

    def new_user_turn(self) -> str:
        key = f'u{self.next_user_number}'
        self.next_user_number += 1
        return key

    def lose(self, response: TruthResponse) -> None:
        """The client will never read `response`'s terminal: its connection dropped first."""
        response.lost = True
        if response.status == 'in_progress':
            response.status = 'lost'
            response.seq_end = self.tick()
        for input_ in self.inputs:
            if input_.key in response.answers:
                input_.answer_lost = True

    def input(self, key: str) -> TruthInput | None:
        return next((input_ for input_ in reversed(self.inputs) if input_.key == key), None)
