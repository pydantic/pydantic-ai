"""Properties of saving a run's conversation to a `ConversationStore`, over drawn sequences of turns.

Each example drives one `InMemoryConversationStore` conversation through a drawn list of `Turn`s. A turn takes the
conversation from where the last one left it, from `load()` or from `load(limit=n)`, may append messages to it, runs
an agent on it and saves what the run returned. A run can have a new prompt or none, call a tool, pause on a tool that
needs approval and resume with `DeferredToolResults`, stream, trim or summarize its history with `ProcessHistory`,
and add a message from a hook with `ctx.enqueue(...)` or `ctx.messages.append(...)`. Between the run and the save,
the result can make a JSON round trip, and the user can edit the conversation on purpose: drop its last turn, or put
a new message in place of one.

The test only uses the public API, and keeps a reference of what the user meant: the head as it was before each save,
the messages the user added, what the run made, in the order it made them, and what the model was sent. Every model
response, prompt, tool result and summary has text of its own, so a message can be told apart by its content. After
each save the head is loaded and checked:

- S1, no silent loss: with no edit, every message the head had before the save is still on it, in order.
- S2, no duplicates: no prompt, text, tool call, tool result or summary is on the head twice.
- S3, unedited saves append: with no edit, the head before the save is the start of the head after it, and the
  conversation has no more `branches` than before.
- S4, round trip: the head ends with the messages of the conversation that was saved, and those before them, for a
  conversation loaded as a window, are the ones the head had.
- S5, every message is kept: with no edit, each message the user added, each response the model gave and each tool
  result is on the head once, in the order they were made, and so is each message a hook added.
- S6, summaries: with no edit, each summary is on the head once, before the first message the model was sent after
  it that the head didn't have before the run. A summary of messages the store has goes after them, as the
  [persistence docs](../docs/persistence.md#using-a-conversation-store) say, so stored messages after a summary in
  what the model was sent don't count.
- S7, edits: after an edit at position i, the head ends with the edited messages, and the messages before i are the
  ones it had.

The turns leave out two things a user doesn't do: resuming a paused run from a window that doesn't hold the paused
call, which the run rejects, and editing a tool call or its result rather than a prompt or a reply, which would leave
the other one unanswered.

A property that fails on main for drawn turns is a known finding: `KNOWN_FINDINGS` takes what draws it out of the
turns that property is checked on, and a strict expected failure on a minimal list of turns pins it, so it starts
passing once the finding is fixed: then delete the entry. `test_each_property_can_fail` shows each checker can fail
without them.

`PROPERTY_SETTINGS` runs a derandomized batch of 50 examples per property. `STORE_PROPERTY_EXAMPLES=200` sets a longer,
randomized run to look for new failures.
"""

from __future__ import annotations as _annotations

import asyncio
import collections
import dataclasses
import itertools
import json
import os
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st
from pydantic import TypeAdapter

from pydantic_ai import Agent, AgentRunResult, Conversation, RunContext
from pydantic_ai.capabilities import AbstractCapability, ProcessHistory
from pydantic_ai.conversation import ConversationTypeAdapter
from pydantic_ai.conversation_store import InMemoryConversationStore
from pydantic_ai.messages import (
    CompactionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults

from .function_model_prefix import COMPACTION_REWRITES_HISTORY

pytestmark = [
    pytest.mark.moves_cache_prefix(reason=COMPACTION_REWRITES_HISTORY),
    pytest.mark.filterwarnings(
        'ignore:Do not use the `random` module inside strategies:hypothesis.errors.HypothesisDeprecationWarning'
    ),
]

_long_run_examples = os.environ.get('STORE_PROPERTY_EXAMPLES')
PROPERTY_SETTINGS = settings(
    max_examples=int(_long_run_examples) if _long_run_examples else 50,
    derandomize=_long_run_examples is None,
    database=None,
    deadline=None,
    print_blob=_long_run_examples is not None,
    suppress_health_check=[HealthCheck.too_slow],
)

Property = Literal['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']
PROPERTIES: tuple[Property, ...] = ('S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7')


# --- turns -----------------------------------------------------------------------------------------------


Source = Literal['carry', 'load', 'window']
Appended = Literal['prompt', 'exchange']
Mode = Literal['plain', 'tool', 'deferred']
Trims = Literal['trim', 'summarize']
Hook = Literal['enqueue_before_run', 'enqueue_before_model_request', 'append_before_run', 'append_before_model_request']
RoundTrip = Literal['result', 'conversation']
Edit = Literal['drop_last_turn', 'replace']


@dataclass(frozen=True)
class Turn:
    """One request of a chat app: get the conversation, maybe add to it, run, maybe edit, save."""

    source: Source = 'carry'
    """Go on from the conversation the last save left, `load()` it, or `load(limit=window)` it."""
    window: int = 2
    appended: tuple[Appended, ...] = ()
    """Messages appended before the run: a user prompt, or a prompt and a reply, as a UI that keeps history does."""
    prompt: bool = True
    """Run with a new prompt, or with none, going on from the last appended prompt."""
    mode: Mode = 'plain'
    """Answer at once, call a tool first, or pause on a tool that needs approval and resume in a second run."""
    capability: Trims | None = None
    keep: int = 2
    """How many of the last messages the capability keeps."""
    hook: Hook | None = None
    stream: bool = False
    round_trip: RoundTrip | None = None
    save_result: bool = True
    """Save the run's result, or the `result.conversation` kept in a variable."""
    edit: Edit | None = None
    edit_at: int = 0
    """Which prompt or reply `replace` replaces, modulo how many the conversation has."""

    def __repr__(self) -> str:
        changed = [
            f'{f.name}={getattr(self, f.name)!r}'
            for f in dataclasses.fields(self)
            if getattr(self, f.name) != f.default
        ]
        return f'Turn({", ".join(changed)})'

    def _repr_pretty_(self, printer: Any, cycle: bool) -> None:
        """Hypothesis prints a failing example with this, so it shows only what the turn changes."""
        printer.text(repr(self))


def _one_of(kinds: Any, *, optional: bool = False) -> st.SearchStrategy[Any]:
    return st.sampled_from([*([None] if optional else []), *get_args(kinds)])


@st.composite
def turns(draw: st.DrawFn) -> Turn:
    appended: tuple[Appended, ...] = tuple(draw(st.lists(_one_of(Appended), max_size=3)))
    mode: Mode = draw(_one_of(Mode))
    return Turn(
        source=draw(_one_of(Source)),
        window=draw(st.integers(0, 6)),
        appended=appended,
        prompt=draw(st.booleans()) if appended and appended[-1] == 'prompt' else True,
        mode=mode,
        capability=draw(_one_of(Trims, optional=True)),
        keep=draw(st.integers(1, 4)),
        hook=draw(_one_of(Hook, optional=True)),
        stream=mode != 'deferred' and draw(st.booleans()),
        round_trip=draw(_one_of(RoundTrip, optional=True)),
        save_result=draw(st.booleans()),
        edit=draw(st.one_of(st.none(), _one_of(Edit))),
        edit_at=draw(st.integers(0, 20)),
    )


# --- what the user and the run made ----------------------------------------------------------------------

PartKey = tuple[Any, ...]


def part_key(part: Any) -> PartKey:
    """A part by its content, leaving out what a run fills in, like timestamps and run ids."""
    if isinstance(part, UserPromptPart):
        return ('user', str(part.content))
    if isinstance(part, TextPart):
        return ('text', part.content)
    if isinstance(part, ToolCallPart):
        return ('call', part.tool_name, part.tool_call_id, json.dumps(part.args_as_dict(), sort_keys=True))
    if isinstance(part, ToolReturnPart):
        return ('return', part.tool_name, part.tool_call_id, str(part.content))
    if isinstance(part, CompactionPart):
        return ('summary', part.content)
    return (part.part_kind, repr(getattr(part, 'content', None)))


def message_key(message: ModelMessage) -> PartKey:
    return (message.kind, tuple(part_key(part) for part in message.parts))


_UNIQUE_KINDS = {'user', 'text', 'call', 'return', 'summary'}


@dataclass
class Made:
    """Something the user or the run made, in the order they made it."""

    kind: Literal['part', 'response', 'added', 'summary']
    key: PartKey
    """A part's key, a response's message key, the text a hook added, or a summary's text."""


@dataclass
class Record:
    """What one example made, and what the model was sent, across its turns."""

    counter: itertools.count[int] = field(default_factory=itertools.count)
    made: list[Made] = field(default_factory=list[Made])
    sent: list[list[ModelMessage]] = field(default_factory=list[list[ModelMessage]])

    def next(self) -> int:
        return next(self.counter)


# --- the agent --------------------------------------------------------------------------------------------


@dataclass
class Script:
    """The model: answers each request with the next step of the turn's plan, then with text."""

    record: Record
    plan: list[Literal['lookup', 'approve']] = field(default_factory=list[Literal['lookup', 'approve']])

    def _next(self) -> ModelResponse:
        n = self.record.next()
        step = self.plan.pop(0) if self.plan else None
        if step is None:
            response = ModelResponse(parts=[TextPart(f'reply {n}')])
        else:
            response = ModelResponse(parts=[ToolCallPart(step, {'n': n}, tool_call_id=f'call-{n}')])
        self.record.made.append(Made('response', message_key(response)))
        return response

    def respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.record.sent.append(list(messages))
        return self._next()

    async def stream(self, messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        self.record.sent.append(list(messages))
        response = self._next()
        part = response.parts[0]
        if isinstance(part, TextPart):
            yield part.content
        else:
            assert isinstance(part, ToolCallPart)
            yield {
                0: DeltaToolCall(name=part.tool_name, json_args=part.args_as_json_str(), tool_call_id=part.tool_call_id)
            }


@dataclass
class AddsAMessage(AbstractCapability[Any]):
    """Adds a message of its own, with `ctx.enqueue(...)` or `ctx.messages.append(...)`, before the run or its first request."""

    how: str
    record: Record

    def _add(self, ctx: RunContext[Any]) -> None:
        text = f'<added {self.record.next()}>'
        self.record.made.append(Made('added', (text,)))
        if self.how.startswith('enqueue'):
            ctx.enqueue(ModelRequest(parts=[UserPromptPart(text)]))
        else:
            ctx.messages.append(ModelRequest(parts=[UserPromptPart(text)]))

    async def before_run(self, ctx: RunContext[Any]) -> None:
        if self.how.endswith('before_run'):
            self._add(ctx)

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        if self.how.endswith('before_model_request') and ctx.run_step == 1:
            self._add(ctx)
        return request_context


def _processor(turn: Turn, record: Record) -> Callable[[list[ModelMessage]], list[ModelMessage]]:
    def trim(messages: list[ModelMessage]) -> list[ModelMessage]:
        return messages[-turn.keep :]

    def summarize(messages: list[ModelMessage]) -> list[ModelMessage]:
        """Puts a summary in place of all but the last `keep` messages, as `docs/capabilities/compaction.md` says to store one."""
        if len(messages) <= turn.keep:
            return messages
        text = f'summary {record.next()}'
        record.made.append(Made('summary', (text,)))
        return [ModelRequest(parts=[CompactionPart(content=text)]), *messages[-turn.keep :]]

    return trim if turn.capability == 'trim' else summarize


def _agent(turn: Turn, script: Script, record: Record) -> Agent[None, str | DeferredToolRequests]:
    capabilities: list[AbstractCapability[Any]] = []
    if turn.capability is not None:
        capabilities.append(ProcessHistory(_processor(turn, record)))
    if turn.hook is not None:
        capabilities.append(AddsAMessage(turn.hook, record))
    agent = Agent(
        FunctionModel(script.respond, stream_function=script.stream),
        output_type=[str, DeferredToolRequests],
        capabilities=capabilities,
    )

    @agent.tool
    def lookup(ctx: RunContext[Any], n: int) -> str:
        content = f'fact {n}'
        record.made.append(Made('part', ('return', 'lookup', ctx.tool_call_id, content)))
        return content

    @agent.tool(requires_approval=True)
    def approve(ctx: RunContext[Any], n: int) -> str:
        content = f'approved {n}'
        record.made.append(Made('part', ('return', 'approve', ctx.tool_call_id, content)))
        return content

    return agent


# --- the driver -------------------------------------------------------------------------------------------

_RESULT_ADAPTER: TypeAdapter[AgentRunResult[str | DeferredToolRequests]] = TypeAdapter(
    AgentRunResult[str | DeferredToolRequests]
)


@dataclass
class Session:
    """The conversation as the app holds it between requests, and the reference: the head the store should have."""

    store: InMemoryConversationStore = field(default_factory=InMemoryConversationStore)
    conversation: Conversation | None = None
    head: list[ModelMessage] = field(default_factory=list[ModelMessage])
    """The head as the last save left it, once checked."""
    offset: int = 0
    """How many messages of `head` come before those of `conversation`, when it was loaded as a window."""


def _flat(head: Sequence[ModelMessage]) -> list[tuple[int, int, PartKey]]:
    return [(i, j, part_key(part)) for i, message in enumerate(head) for j, part in enumerate(message.parts)]


def _is_subsequence(small: Sequence[ModelMessage], big: Sequence[ModelMessage]) -> bool:
    it = iter(big)
    return all(any(message == other for other in it) for message in small)


@dataclass
class Save:
    """What a save is checked against."""

    before: list[ModelMessage]
    offset: int
    saved: list[ModelMessage]
    made: list[Made]
    sent: list[list[ModelMessage]]
    branches: int
    edit_at: int | None
    """Where the edit was, in the conversation's messages, or `None` with no edit."""


def _no_silent_loss(save: Save, head: list[ModelMessage], branches: int) -> list[str]:
    return [] if _is_subsequence(save.before, head) else ['a message stored before the save is gone or out of order']


def _no_duplicates(save: Save, head: list[ModelMessage], branches: int) -> list[str]:
    counts = collections.Counter(key for _, _, key in _flat(head) if key[0] in _UNIQUE_KINDS)
    return [f'{key} is on the head {count} times' for key, count in counts.items() if count > 1]


def _appends(save: Save, head: list[ModelMessage], branches: int) -> list[str]:
    found: list[str] = []
    if head[: len(save.before)] != save.before:
        found.append('the head before the save is not the start of the head after it')
    if branches > save.branches:
        found.append(f'an unedited save added {branches - save.branches} branch(es)')
    return found


def _round_trip(save: Save, head: list[ModelMessage], branches: int) -> list[str]:
    if head[save.offset :] != save.saved or head[: save.offset] != save.before[: save.offset]:
        return ['the head is not the saved conversation after the messages before its window']
    return []


def _where(made: Made, head: list[ModelMessage]) -> list[tuple[int, int]]:
    """Where on the head `made` is, by message and part."""
    if made.kind == 'response':
        return [(i, 0) for i, message in enumerate(head) if message_key(message) == made.key]
    if made.kind == 'added':
        return [(i, j) for i, j, key in _flat(head) if any(made.key[0] in str(value) for value in key[1:])]
    return [(i, j) for i, j, key in _flat(head) if key == made.key]


def _everything_kept(save: Save, head: list[ModelMessage], branches: int) -> list[str]:
    found: list[str] = []
    positions: list[tuple[int, int]] = []
    for made in save.made:
        if made.kind == 'summary':
            continue
        where = _where(made, head)
        if len(where) != 1:
            found.append(f'{made.key} is on the head {len(where)} times')
        elif made.kind != 'added':
            # Where a hook's message goes next to the prompt is the run's to decide.
            positions.append(where[0])
    if positions != sorted(set(positions)):
        found.append('what was made is not on the head in the order it was made')
    return found


def _summaries(save: Save, head: list[ModelMessage], branches: int) -> list[str]:
    found: list[str] = []
    flat = _flat(head)
    stored = {key for _, _, key in _flat(save.before)}
    for made in save.made:
        if made.kind != 'summary':
            continue
        key = ('summary', made.key[0])
        where = [i for i, _, k in flat if k == key]
        if len(where) != 1:
            found.append(f'{made.key[0]!r} is on the head {len(where)} times')
            continue
        sent = next(sent for sent in save.sent if any(key in map(part_key, m.parts) for m in sent))
        at = next(i for i, m in enumerate(sent) if key in map(part_key, m.parts))
        for later in sent[at + 1 :]:
            new = [part_key(part) for part in later.parts if part_key(part) not in stored]
            # The first message sent after the summary that the store didn't have. One that isn't on the head at all
            # is S5's to report.
            if found_at := [i for i, _, k in flat if k in new]:
                if where[0] > min(found_at):
                    found.append(f'{made.key[0]!r} is after {new[0]}, which the model was sent after it')
                break
    return found


def _edits_kept(save: Save, head: list[ModelMessage], branches: int) -> list[str]:
    assert save.edit_at is not None
    found: list[str] = []
    kept = save.offset + min(save.edit_at, len(save.before) - save.offset)
    if head[save.offset :] != save.saved:
        found.append('the head is not the edited conversation')
    if head[:kept] != save.before[:kept]:
        found.append('a message before the edit changed')
    return found


_CHECKS: dict[Property, tuple[Callable[[Save, list[ModelMessage], int], list[str]], bool | None]] = {
    'S1': (_no_silent_loss, False),
    'S2': (_no_duplicates, None),
    'S3': (_appends, False),
    'S4': (_round_trip, None),
    'S5': (_everything_kept, False),
    'S6': (_summaries, False),
    'S7': (_edits_kept, True),
}
"""Each property's checker, and whether it checks only edited saves (`True`), only unedited ones (`False`), or all."""


def violations(save: Save, head: list[ModelMessage], branches: int) -> list[tuple[Property, str]]:
    """The properties the head loaded after `save` breaks."""
    edited = save.edit_at is not None
    return [
        (checked, found)
        for checked, (check, on_edited) in _CHECKS.items()
        if on_edited is None or on_edited == edited
        for found in check(save, head, branches)
    ]


def _edit(turn: Turn, conversation: Conversation, record: Record) -> int | None:
    """Edit `conversation` on purpose, as `turn` says, and return where."""
    messages = conversation.messages
    if turn.edit is None or not messages:
        return None
    if turn.edit == 'drop_last_turn':
        # A turn starts at a prompt that doesn't come between a tool call and its result, like a message a hook
        # added before a paused run was resumed, so dropping it leaves no call unanswered.
        at = max(
            (
                i
                for i, m in enumerate(messages)
                if isinstance(m, ModelRequest)
                and any(isinstance(p, UserPromptPart) for p in m.parts)
                and not (i and isinstance(previous := messages[i - 1], ModelResponse) and previous.tool_calls)
            ),
            default=len(messages) - 1,
        )
        del messages[at:]
        return at
    # A user edits a prompt or a reply, not a tool call or its result, which would leave the other unanswered.
    editable = [i for i, m in enumerate(messages) if all(isinstance(p, (UserPromptPart, TextPart)) for p in m.parts)]
    if not editable:
        return None
    at = editable[turn.edit_at % len(editable)]
    text = f'edited {record.next()}'
    if isinstance(messages[at], ModelRequest):
        messages[at] = ModelRequest(parts=[UserPromptPart(text)])
    else:
        messages[at] = ModelResponse(parts=[TextPart(text)])
    return at


async def _save(
    session: Session,
    turn: Turn,
    result: AgentRunResult[Any],
    record: Record,
    made_from: int,
    sent_from: int,
    edit: bool,
) -> list[tuple[Property, str]]:
    """Save `result` as `turn` says and check the head."""
    if turn.round_trip == 'result':
        result = _RESULT_ADAPTER.validate_json(_RESULT_ADAPTER.dump_json(result))
    conversation = result.conversation
    if turn.round_trip == 'conversation':
        conversation = ConversationTypeAdapter.validate_json(ConversationTypeAdapter.dump_json(conversation))
    edit_at = _edit(turn, conversation, record) if edit else None
    record_before = await session.store.read_record('c')
    save = Save(
        before=session.head,
        offset=session.offset,
        saved=list(conversation.messages),
        made=record.made[made_from:],
        sent=record.sent[sent_from:],
        branches=len(record_before.branches) if record_before else 0,
        edit_at=edit_at,
    )
    if turn.save_result and edit_at is None and turn.round_trip != 'conversation':
        await session.store.save(result)
        session.conversation = result.conversation
    else:
        await session.store.save(conversation)
        session.conversation = conversation
    loaded = await session.store.load('c')
    stored = await session.store.read_record('c')
    assert loaded is not None and stored is not None
    found = violations(save, loaded.messages, len(stored.branches))
    session.head = loaded.messages
    return found


async def _turn(session: Session, turn: Turn, record: Record) -> list[tuple[Property, str]]:
    store = session.store
    if session.conversation is None:
        conversation = Conversation(conversation_id='c')
    elif turn.source == 'carry':
        conversation = session.conversation
    else:
        loaded = await store.load('c', limit=turn.window if turn.source == 'window' else None)
        assert loaded is not None
        conversation = loaded
        session.offset = len(session.head) - len(loaded.messages)

    made_from, sent_from = len(record.made), len(record.sent)
    for kind in turn.appended:
        n = record.next()
        request = ModelRequest(parts=[UserPromptPart(f'appended {n}')])
        conversation.messages.append(request)
        record.made.append(Made('part', part_key(request.parts[0])))
        if kind == 'exchange':
            response = ModelResponse(parts=[TextPart(f'appended reply {n}')])
            conversation.messages.append(response)
            record.made.append(Made('response', message_key(response)))
    prompt: str | None = None
    if turn.prompt:
        prompt = f'prompt {record.next()}'
        record.made.append(Made('part', ('user', prompt)))

    script = Script(record, plan=['lookup'] if turn.mode == 'tool' else ['approve'] if turn.mode == 'deferred' else [])
    agent = _agent(turn, script, record)
    if turn.stream:
        async with agent.run_stream(prompt, conversation=conversation) as stream:
            await stream.get_output()
        result = stream.result
    else:
        result = await agent.run(prompt, conversation=conversation)

    if turn.mode != 'deferred':
        return await _save(session, turn, result, record, made_from, sent_from, edit=True)

    # Pause, save, then resume with the approval and save again.
    assert isinstance(result.output, DeferredToolRequests), result.output
    pending = result.output
    found = await _save(
        session, dataclasses.replace(turn, round_trip=None), result, record, made_from, sent_from, False
    )
    if found:
        return found
    if turn.source != 'carry':
        # A window has to hold the paused call to resume it.
        loaded = await store.load('c', limit=max(turn.window, 1) if turn.source == 'window' else None)
        assert loaded is not None
        session.conversation = loaded
        session.offset = len(session.head) - len(loaded.messages)
    assert session.conversation is not None
    made_from, sent_from = len(record.made), len(record.sent)
    results = DeferredToolResults(approvals={call.tool_call_id: True for call in pending.approvals})
    result = await agent.run(conversation=session.conversation, deferred_tool_results=results)
    return await _save(session, turn, result, record, made_from, sent_from, edit=True)


def drive(steps: Sequence[Turn]) -> list[tuple[Property, str]]:
    """Run `steps` on a new store and return what the first save that breaks a property breaks."""

    async def run() -> list[tuple[Property, str]]:
        session, record = Session(), Record()
        for turn in steps:
            found = await _turn(session, turn, record)
            if found:
                return found
        return []

    with asyncio.Runner() as runner:
        return runner.run(run())


_A = ModelRequest(parts=[UserPromptPart('a')])
_B = ModelResponse(parts=[TextPart('b')])
_C = ModelRequest(parts=[UserPromptPart('c')])


def _save_of(**changes: Any) -> Save:
    return dataclasses.replace(
        Save(before=[_A, _B], offset=0, saved=[_A, _B], made=[], sent=[], branches=0, edit_at=None), **changes
    )


@pytest.mark.parametrize(
    'checked,save,head,branches',
    [
        pytest.param('S1', _save_of(saved=[_A]), [_A], 0, id='S1'),
        pytest.param('S2', _save_of(saved=[_A, _B, _A]), [_A, _B, _A], 0, id='S2'),
        pytest.param('S3', _save_of(saved=[_A, _B, _C]), [_A, _B, _C], 1, id='S3'),
        pytest.param('S4', _save_of(saved=[_A, _B, _C]), [_A, _B], 0, id='S4'),
        pytest.param('S5', _save_of(made=[Made('part', part_key(_C.parts[0]))]), [_A, _B], 0, id='S5'),
        pytest.param('S6', _save_of(made=[Made('summary', ('a summary',))]), [_A, _B], 0, id='S6'),
        pytest.param('S7', _save_of(saved=[_C, _B], edit_at=0), [_A, _B], 0, id='S7'),
    ],
)
def test_each_property_can_fail(checked: Property, save: Save, head: list[ModelMessage], branches: int):
    """A property no head breaks checks nothing, so each has a head that breaks it."""
    assert checked in {code for code, _ in violations(save, head, branches)}


# --- known findings ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class KnownFinding:
    id: str
    fails: frozenset[Property]
    leave_out: Callable[[Turn, Sequence[Turn]], Turn]
    """The turn, after the earlier ones, without what can draw the finding, for the properties it fails."""
    reproducers: tuple[tuple[Turn, ...], ...]
    """Minimal lists of turns that break `fails` on main."""
    reason: str


KNOWN_FINDINGS = [
    KnownFinding(
        'summary-after-appended-messages',
        frozenset({'S6'}),
        lambda turn, earlier: (
            dataclasses.replace(
                turn, appended=(), prompt=True, hook=None if turn.hook == 'append_before_run' else turn.hook
            )
            if turn.capability == 'summarize'
            else turn
        ),
        (
            (Turn(), Turn(appended=('prompt',), prompt=False, capability='summarize')),
            (Turn(appended=('exchange',), capability='summarize'),),
            (Turn(), Turn(capability='summarize', keep=3, hook='append_before_run')),
        ),
        'A summary of stored messages is stored after the messages added before the run, like a prompt the run '
        'goes on from, an exchange a UI sent or a message a `before_run` hook added, though the model was sent it '
        'before them. The persistence docs describe this placement: after the messages the run was given.',
    ),
    KnownFinding(
        'message-appended-in-a-trimmed-run-is-lost',
        frozenset({'S5'}),
        lambda turn, earlier: (
            dataclasses.replace(turn, capability=None) if turn.hook == 'append_before_model_request' else turn
        ),
        ((Turn(mode='tool', capability='trim', keep=1, hook='append_before_model_request'),),),
        'A message a `before_model_request` hook adds with `ctx.messages.append(...)` is not in '
        '`result.conversation`, so not saved, when a capability trims it from the history in a later request.',
    ),
    KnownFinding(
        'superseded-summary-is-lost',
        frozenset({'S6'}),
        # Only a run that makes more than one request can summarize its own summary.
        lambda turn, earlier: (
            dataclasses.replace(turn, mode='plain') if turn.capability == 'summarize' and turn.mode == 'tool' else turn
        ),
        ((Turn(), Turn(mode='tool', capability='summarize')),),
        'A summary the model was sent is not stored when a later request in the same run summarizes it away.',
    ),
]


@pytest.mark.parametrize(
    'finding,reproducer',
    [
        pytest.param(
            finding,
            reproducer,
            id=f'{finding.id}-{index}',
            marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason=finding.reason),
        )
        for finding in KNOWN_FINDINGS
        for index, reproducer in enumerate(finding.reproducers)
    ],
)
def test_known_finding_is_still_present(finding: KnownFinding, reproducer: tuple[Turn, ...]):
    assert [v for v in drive(reproducer) if v[0] in finding.fails] == []


FIXED_FINDINGS: dict[str, tuple[tuple[Turn, ...], ...]] = {
    # A message a `before_run` hook added with `ctx.messages.append(...)` to a continued conversation was in
    # `result.conversation` but not saved: the save started after it.
    'message-appended-before-run-is-lost': ((Turn(), Turn(hook='append_before_run')),),
    # Two requests in a row, like an appended prompt and the new one, a prompt and an enqueued message, came back
    # merged, and a window that starts with a tool result came back without it, so `load()` didn't equal the
    # conversation that was saved.
    'conversation-comes-back-cleaned': (
        (Turn(appended=('prompt',)), Turn()),
        (Turn(hook='enqueue_before_run'), Turn()),
        (Turn(mode='tool'), Turn(source='window', window=2)),
    ),
    # With two stored requests in a row and unsaved ones the run merged, an unedited save branched.
    'merging-unsaved-requests-branches-off-adjacent-stored-ones': (
        (Turn(appended=('prompt',)), Turn(source='load', appended=('prompt', 'prompt'), prompt=False)),
    ),
    # An edit after a run merged two stored requests stored a prompt twice, or replaced only one of the two.
    'edit-after-merged-stored-requests-stores-a-prompt-twice': (
        (Turn(appended=('prompt',)), Turn(edit='replace', edit_at=1)),
        (Turn(appended=('prompt',)), Turn(edit='replace')),
    ),
}
"""Minimal lists of turns that broke a property on main before a fix, kept as regression tests."""


@pytest.mark.parametrize(
    'reproducer',
    [
        pytest.param(reproducer, id=f'{finding}-{index}')
        for finding, reproducers in FIXED_FINDINGS.items()
        for index, reproducer in enumerate(reproducers)
    ],
)
def test_fixed_finding_stays_fixed(reproducer: tuple[Turn, ...]):
    assert drive(reproducer) == []


# --- properties -------------------------------------------------------------------------------------------


@pytest.mark.parametrize('checked', PROPERTIES)
@PROPERTY_SETTINGS
@given(data=st.data())
def test_saving_keeps_what_the_user_meant(checked: Property, data: st.DataObject):
    known = [finding.leave_out for finding in KNOWN_FINDINGS if checked in finding.fails]
    steps: list[Turn] = []
    for _ in range(data.draw(st.integers(1, 5), label='turn count')):
        turn = data.draw(turns(), label='turn')
        for leave_out in known:
            turn = leave_out(turn, steps)
        steps.append(turn)
    assert [v for v in drive(steps) if v[0] == checked] == []
