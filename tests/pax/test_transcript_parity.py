"""A web transcript shows the same entries whether the browser watched the turns or loaded them afterwards.

A watching browser builds its transcript from the stream: each turn's prompt from `TurnStarted`, its items from the
item messages `EventTranslator` makes, and its usage line from `TurnEnded`. A browser that connects later is sent
`history_entries` of the saved conversation instead. These two producers drifted apart one item kind at a time, so
one session here runs every kind of item message, and a new kind fails until the session emits it.
"""

import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import get_args, get_type_hints

import pytest
from pydantic import JsonValue

from pydantic_ai import Agent, ModelRetry
from pydantic_ai.messages import ModelMessage, UserPromptPart
from pydantic_ai.models.function import (
    AgentInfo,
    DeltaThinkingCalls,
    DeltaThinkingPart,
    DeltaToolCall,
    DeltaToolCalls,
    FunctionModel,
)
from pydantic_ai_harness.file_system import FileSystem
from pydantic_ai_harness.shell import Shell
from pydantic_ai_pax.builtin_plugins.web.protocol import (
    PROTOCOL,
    ArgsDelta,
    Diff,
    DiffAdded,
    Item,
    ItemAdded,
    ItemKind,
    ItemLog,
    ItemMessage,
    Notice,
    ServerMessage,
    SessionsChanged,
    ShellDone,
    Snapshot,
    SubCallAdded,
    SubCallDone,
    SubCallOutput,
    TextDelta,
    ToolArgs,
    ToolDone,
    ToolOutput,
    TurnEnded,
    TurnStarted,
    TurnSummary,
    history_entries,
)
from tests._timeouts import READINESS_WAIT_TIMEOUT
from tests.manifests import assert_classified, union_members
from tests.pax.test_web import TOKEN, WIRE, MakeWeb, make_web as make_web, receive_until


@dataclass(frozen=True)
class Compared:
    """An `Item` field the two transcripts must agree on."""

    reason: str


@dataclass(frozen=True)
class LiveOnly:
    """An `Item` field only the watching browser fills, because the saved history does not hold it."""

    reason: str
    tools: frozenset[str] | None = None
    """The tools whose items the saved history cannot fill this field for; other tools' items compare it. `None` for
    every tool."""


COMPARED: dict[str, Compared] = {
    'id': Compared('a tool keeps its call ID; other items are matched by position, as each producer numbers its own'),
    'kind': Compared('what the browser renders the entry as'),
    'text': Compared('the prompt, answer, thinking or tool result the entry shows'),
    'name': Compared('the tool the card names'),
    'args': Compared('the arguments the card shows'),
    'state': Compared("the card's outcome"),
    'diffs': Compared(
        "by each file's changed lines: a saved edit does not say where in the file it was, so a reload gives each"
        ' replacement an unnumbered hunk of its own'
    ),
    'by_user': Compared("whether a `run_code` cell is shown as the user's own"),
    'images': Compared("the prompt's images"),
    'run_id': Compared('the run Edit and Retry rewind to'),
}
KNOWN_LIVE_ONLY: dict[str, LiveOnly] = {
    'output': LiveOnly("a shell command's printout as it ran; the saved result holds what the model got back"),
    'exit_code': LiveOnly(
        "a `shell` command's exit code; the saved history keeps it only inside the result's text, which another tool"
        ' named `shell` can imitate'
    ),
    'calls': LiveOnly("the tools a `run_code` cell called from its sandbox; the saved history holds the cell's result"),
    'diffs': LiveOnly(
        'the saved history holds no file content from before a write, so a reload cannot rebuild its diff',
        tools=frozenset({'write_file'}),
    ),
}


@dataclass(frozen=True)
class Scripted:
    """An item message the session below makes."""

    reason: str


SCRIPTED: dict[type, Scripted] = {
    ItemAdded: Scripted('every prompt, answer, thinking block and tool call'),
    TextDelta: Scripted('the answer and the thinking stream in more than one chunk'),
    ArgsDelta: Scripted("the `run_code` cell's code streams in chunks"),
    ToolArgs: Scripted("the streamed cell's complete arguments arrive with its call"),
    ToolOutput: Scripted('the `shell` command prints'),
    ShellDone: Scripted('the `shell` commands exit, one of them with a failure'),
    DiffAdded: Scripted('the edits and the write'),
    ToolDone: Scripted('every tool returns or fails'),
    SubCallAdded: Scripted('the cell calls `lookup` and `shell` from its sandbox'),
    SubCallOutput: Scripted("the cell's `shell` call prints"),
    SubCallDone: Scripted("the cell's calls return"),
}

CALC = 'def add(a, b):\n    return a + b\n\n\ndef mul(a, b):\n    return a * b\n'
EDITS: dict[str, dict[str, JsonValue]] = {
    'prefix.py': {
        'old_text': '    return a * b\n',
        'new_text': '    return a * b\n\n\ndef sub(a, b):\n    return a - b\n',
    },
    'deletion.py': {'old_text': '\n\ndef mul(a, b):\n    return a * b\n', 'new_text': ''},
    'replacements.py': {
        'replacements': [
            {'old_text': '    return a + b\n', 'new_text': '    return b + a\n'},
            {'old_text': 'def mul(a, b):\n', 'new_text': 'def times(a, b):\n'},
        ]
    },
}
"""Each file the first turn edits, with the edit's arguments: a line added after others, lines dropped, and two
replacements."""
TOOLS = 'use the tools'
CODE = 'write a cell'
CELL = "await lookup(query='y')\nawait shell(command='echo sub')\n"


async def turns(
    messages: list[ModelMessage], info: AgentInfo
) -> AsyncIterator[str | DeltaToolCalls | DeltaThinkingCalls]:
    last = messages[-1].parts[-1]
    if isinstance(last, UserPromptPart) and last.content == TOOLS:
        yield {0: DeltaThinkingPart(content='Planning')}
        yield {0: DeltaThinkingPart(content=' the edits.')}
        yield 'Looking '
        yield 'around.'
        calls: list[tuple[str, JsonValue]] = [
            ('lookup', {'query': 'x'}),
            ('broken', {}),
            *[('edit_file', {'path': path, **args}) for path, args in EDITS.items()],
            ('write_file', {'path': 'new.py', 'content': 'new\n'}),
            ('shell', {'command': 'echo hi'}),
            ('shell', {'command': 'exit 3'}),
        ]
        yield {
            index: DeltaToolCall(name=name, json_args=json.dumps(args), tool_call_id=f'call{index}')
            for index, (name, args) in enumerate(calls, start=1)
        }
    elif isinstance(last, UserPromptPart) and last.content == CODE:
        yield {0: DeltaToolCall(name='run_code', json_args='{"code": "', tool_call_id='cell')}
        for line in CELL.splitlines(keepends=True):
            yield {0: DeltaToolCall(json_args=json.dumps(line)[1:-1])}
        yield {0: DeltaToolCall(json_args='"}')}
    else:
        yield 'Done.'


def live_transcript(messages: Sequence[ServerMessage]) -> list[Item | TurnSummary]:
    """The transcript of a browser that watched `messages`.

    Item messages are applied with `ItemLog`, the server's copy of the browser's item fold. The turn messages follow
    `web-ui/src/state.ts`: a turn's prompt arrives with `TurnStarted`, and its usage line with `TurnEnded`, which also
    shows a tool still running as interrupted. The browser does the same to a sandbox call, which `compared` leaves
    out, as `calls` is live-only.
    """
    entries: list[Item | TurnSummary] = []
    log = ItemLog()
    for message in messages:
        if isinstance(message, TurnStarted):
            # Every scripted turn starts from a prompt; only a turn a finished task wakes has none.
            assert message.item is not None
            log.apply(ItemAdded(item=message.item))
        elif isinstance(message, TurnEnded):
            assert message.outcome == 'completed' and message.usage is not None
            entries += [
                replace(item, state='interrupted') if item.state == 'running' else item for item in log.items.values()
            ]
            entries.append(TurnSummary(id='', usage=message.usage))
            log.clear()
        elif isinstance(message, ItemMessage):
            log.apply(message)
    return entries


def compared(entry: Item | TurnSummary) -> Item | TurnSummary:
    """`entry` as the two transcripts are compared: by its `COMPARED` fields, its diffs by each file's changed lines."""
    if isinstance(entry, TurnSummary):
        return replace(entry, id='')
    changed: dict[str, list[str]] = {}
    for diff in entry.diffs:
        lines = [line for line in diff.diff.split('\n') if line[:1] in '+-' and line[:4] not in ('--- ', '+++ ')]
        changed.setdefault(diff.path, []).extend(lines)
    shown: dict[str, object] = {
        'id': entry.id if entry.kind == 'tool' else '',
        'diffs': [Diff(path=path, diff='\n'.join(lines)) for path, lines in changed.items()],
    }
    unset = Item(id='', kind='tool')
    for name, live in KNOWN_LIVE_ONLY.items():
        if live.tools is None or entry.name in live.tools:
            shown[name] = getattr(unset, name)
    return replace(entry, **shown)


@pytest.mark.moves_cache_prefix(reason='`/set run.code_mode true` between the turns replaces the offered tools')
def test_a_reload_shows_what_the_turns_showed_live(make_web: MakeWeb) -> None:
    pytest.importorskip('pydantic_monty')
    partly_live = {name for name, live in KNOWN_LIVE_ONLY.items() if live.tools is not None}
    assert partly_live <= COMPARED.keys(), (
        'a field live-only for some tools is compared on the rest: add it to COMPARED'
    )
    assert COMPARED.keys() & KNOWN_LIVE_ONLY.keys() == partly_live, (
        'a field live-only for every tool is not compared: remove it from COMPARED'
    )
    assert_classified(
        (field.name for field in fields(Item)),
        {**COMPARED, **KNOWN_LIVE_ONLY},
        complete_walk=True,
        table_name='COMPARED | KNOWN_LIVE_ONLY',
    )
    assert_classified(union_members(ItemMessage), SCRIPTED, complete_walk=True, table_name='SCRIPTED')
    # The pax conftest runs each test in a fresh working directory, which is the session's workspace.
    for path in EDITS:
        Path(path).write_text(CALC, encoding='utf-8')
    agent = Agent(FunctionModel(stream_function=turns), capabilities=[FileSystem(), Shell(tools=['shell'])])

    @agent.tool_plain
    def lookup(query: str) -> str:
        """Look something up."""
        return f'found {query}'

    @agent.tool_plain
    def broken() -> str:
        """A tool that asks the model to try again."""
        raise ModelRetry('try again')

    web_ = make_web(agent)
    url = f'/ws?token={TOKEN}&protocol={PROTOCOL}'
    with web_.client.websocket_connect(url, headers={'origin': 'http://testserver'}) as socket:
        socket.receive_text()
        running = web_.run()
        web_.post('/api/input', {'text': TOOLS})
        messages, _ = receive_until(socket, TurnEnded)
        web_.post('/api/input', {'text': '/set run.code_mode true'})
        messages += receive_until(socket, Notice)[0]
        web_.post('/api/input', {'text': CODE})
        messages += receive_until(socket, TurnEnded)[0]
        receive_until(socket, SessionsChanged)
        with web_.client.websocket_connect(url, headers={'origin': 'http://testserver'}) as reloaded:
            snapshot = WIRE.validate_json(reloaded.receive_text())
        web_.post('/api/input', {'text': '/exit'})
        assert running.result(timeout=READINESS_WAIT_TIMEOUT) == 'exit'
    assert isinstance(snapshot, Snapshot) and not snapshot.running

    unscripted = SCRIPTED.keys() - {type(message) for message in messages}
    assert not unscripted, (
        f'the session sends no {sorted(kind.__name__ for kind in unscripted)}: make `turns` send each, as its '
        '`SCRIPTED` entry says'
    )
    live = live_transcript(messages)
    reload = snapshot.items
    missing_kinds = set(get_args(ItemKind)) - {entry.kind for entry in reload if isinstance(entry, Item)}
    assert not missing_kinds, f'the reload shows no {sorted(missing_kinds)} item: make `turns` produce one'
    [entry_types] = get_args(get_type_hints(history_entries)['return'])
    missing_types = union_members(entry_types) - {type(entry) for entry in reload}
    assert not missing_types, (
        f'the reload has no {sorted(kind.__name__ for kind in missing_types)} entry: make `turns` produce one'
    )
    assert [compared(entry) for entry in live] == [compared(entry) for entry in reload], (
        'the reload differs from what the turns showed live: fix whichever of `EventTranslator` and '
        '`history_entries` drifted, or add the field to `KNOWN_LIVE_ONLY` with why a reload cannot show it'
    )
    # Equal transcripts prove nothing if both lost the diffs: each edit must show its changed lines live.
    edited: set[str] = set()
    for entry in live:
        if isinstance(entry, Item) and entry.name == 'edit_file' and isinstance(shown := compared(entry), Item):
            edited |= {diff.path for diff in shown.diffs if diff.diff}
    assert edited == EDITS.keys(), f'live `edit_file` cards show changed lines for {sorted(edited)} only'

    # Each known difference still is one: a fix that lets the reload show it must drop its entry.
    pairs = [
        (seen, loaded) for seen, loaded in zip(live, reload) if isinstance(seen, Item) and isinstance(loaded, Item)
    ]
    for name, live_only in KNOWN_LIVE_ONLY.items():
        differs = [
            seen.name
            for seen, loaded in pairs
            if (live_only.tools is None or seen.name in live_only.tools)
            and getattr(seen, name) != getattr(loaded, name)
        ]
        assert differs, (
            f'`KNOWN_LIVE_ONLY[{name!r}]` no longer differs: remove it, and give the field a `COMPARED` entry with '
            'what the browser shows from it if it has none'
        )
