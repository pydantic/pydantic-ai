"""Rewind menus preserve valid history boundaries and commit before editing a draft."""

from io import StringIO
from pathlib import Path

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console
from rich.text import Text
from termflow.tui.menu import MenuResult

from pydantic_ai import Agent
from pydantic_ai.messages import (
    BinaryContent,
    BinaryImage,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.step_persistence.conversations import ConversationConflict
from pydantic_clai2 import chat
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import Conversation, Transcript
from pydantic_clai2.runtime._session import Session
from pydantic_clai2.ui.menus.rewind import RewindPoint, build_rewind_menu, rewind
from pydantic_clai2.ui.prompt.live_prompt import LivePrompt
from tests.clai2.menu_script import Script, pick
from tests.clai2.test_compaction import make_plugin, two_turns
from tests.clai2.test_live_prompt import editor
from tests.clai2.test_sessions import saved_session


@pytest.mark.parametrize('key', ['enter', 'escape', 'ctrl-c'])
def test_menu_selects_run_boundaries_and_sanitizes_preview(monkeypatch: pytest.MonkeyPatch, key: str) -> None:
    messages: list[ModelMessage] = [
        ModelRequest(parts=[SystemPromptPart('system')]),
        ModelRequest(parts=[UserPromptPart('first\x1b]52;c;payload\x07')], run_id='first'),
        ModelResponse(parts=[ToolCallPart('tool', {}, 'call')], run_id='first'),
        ModelRequest(parts=[ToolReturnPart('tool', 'done', 'call'), UserPromptPart('steer')], run_id='first'),
        ModelRequest(parts=[UserPromptPart('more steering')], run_id='first'),
        ModelResponse(parts=[TextPart('done')], run_id='first'),
        ModelRequest(parts=[RetryPromptPart('retry'), UserPromptPart('not a boundary')]),
        ModelRequest(parts=[UserPromptPart('second')], run_id='second'),
        ModelResponse(parts=[TextPart('legacy answer')]),
    ]
    output = StringIO()
    monkeypatch.setattr('sys.stdout', output)
    keys = iter(['down', key])
    monkeypatch.setattr('pydantic_clai2.ui.menus.rewind.menu_key', lambda: next(keys))
    result = build_rewind_menu(messages).run()
    if key == 'enter':
        assert result.item is not None
        assert result.item.value == RewindPoint(message_index=1, text='first\x1b]52;c;payload\x07', images=())
    else:
        assert result.cancelled
    # The menu paints CRLF rows, and Rich 15.0.0's `from_ansi` blanks each one:
    # https://github.com/Textualize/rich/issues/4090
    text = ' '.join(Text.from_ansi(output.getvalue().replace('\r\n', '\n')).plain.split())
    assert 'Rewind conversation' in text and 'does NOT undo files' in text
    assert 'Replaces your current draft' in text and 'and its attachments.' in text
    assert 'steering' not in text and 'not a boundary' not in text
    assert '\x1b]52;' not in output.getvalue() and '\x07' not in output.getvalue()


@pytest.mark.parametrize(
    'messages', [[], [ModelRequest(parts=[UserPromptPart([ImageUrl('https://example.com/a.png')])])]]
)
def test_empty_and_unsupported_menu_cancels(monkeypatch: pytest.MonkeyPatch, messages: list[ModelMessage]) -> None:
    monkeypatch.setenv('COLUMNS', '140')
    monkeypatch.setenv('LINES', '30')
    output = StringIO()
    monkeypatch.setattr('sys.stdout', output)
    keys = iter(['enter', 'escape'])
    monkeypatch.setattr('pydantic_clai2.ui.menus.rewind.menu_key', lambda: next(keys))
    result = build_rewind_menu(messages).run()
    assert result.cancelled
    assert 'No earlier prompts' in output.getvalue() or 'unsupported attachment' in output.getvalue()


async def test_compacted_prompts_are_disabled_but_new_turns_can_rewind(monkeypatch: pytest.MonkeyPatch) -> None:
    transcript = Transcript(messages=two_turns(), model=TestModel(custom_output_text='answer and later instruction'))
    plugin = make_plugin(transcript, protected_tokens=0)
    await plugin.commands.execute_async('/compact')
    assert len(transcript.messages) == 3
    menu = build_rewind_menu(transcript.messages)
    assert menu.highlighted is not None and menu.highlighted.disabled
    monkeypatch.setattr('sys.stdout', StringIO())
    keys = iter(['enter', 'escape'])
    monkeypatch.setattr('pydantic_clai2.ui.menus.rewind.menu_key', lambda: next(keys))
    assert build_rewind_menu(transcript.messages).run().cancelled

    session = Session(Agent(TestModel()), deps=None, message_history=transcript.messages)
    await session.prompt('after compaction')
    monkeypatch.setattr('sys.stdout', StringIO())
    keys = iter(['down', 'enter'])
    monkeypatch.setattr('pydantic_clai2.ui.menus.rewind.menu_key', lambda: next(keys))
    result = build_rewind_menu(session.messages).run()
    assert result.item is not None
    assert result.item.value == RewindPoint(message_index=2, text='after compaction', images=())


async def test_replacement_images_do_not_count_discarded_draft_against_limit() -> None:
    image = BinaryContent(data=b'x' * (9 * 1024 * 1024), media_type='image/png')
    queued_image = BinaryContent(data=b'queued', media_type='image/png')
    point = RewindPoint(message_index=0, text='restore', images=(image, image))
    script = Script(lists=[pick(point)], choices=[], texts=[])
    session = Session(
        Agent(TestModel()),
        deps=None,
        message_history=[ModelRequest(parts=[UserPromptPart(['restore', image, image])])],
    )
    async with editor() as (live, _, _):
        queued = live.images.attach([queued_image])
        live.submit(queued)
        old_draft = live.images.attach([image, image])
        live.buffer.replace(old_draft)
        await rewind(session, live, runners=script.runners)
        assert live.images.resolve(live.buffer.text) == ('restore', [image, image])
        assert live.images.resolve(await live.read()) == ('', [queued_image])
        assert len(live.images.pending) == 3
        with pytest.raises(ValueError, match='expired'):
            live.images.resolve(old_draft)


@pytest.mark.parametrize('index', [0, 2])
async def test_rewind_is_durable_and_restores_images(tmp_path: Path, index: int) -> None:
    session = saved_session(tmp_path)
    image = BinaryImage(data=b'image', media_type='image/png')
    await session.prompt('first', images=[image])
    await session.prompt('second')
    await session.prompt('third')
    before = session.messages
    menu = build_rewind_menu(before[: index + 1])
    assert menu.highlighted is not None
    script = Script(lists=[MenuResult(item=menu.highlighted)], choices=[], texts=[])
    async with editor() as (live, _, _):
        live.buffer.replace('discard this draft')
        live.buffer.history_index = 0
        async with live.suspended():
            assert 'Conversation rewound' in await rewind(session, live, runners=script.runners)
        text, images = live.images.resolve(live.buffer.text)
        assert text == ('first' if index == 0 else 'second')
        assert images == ([image] if index == 0 else [])
        assert live.buffer.history_index is None
        assert live.history.get_strings() == []
        assert session.messages == before[:index]
        restored = saved_session(tmp_path)
        await restored.resume(session.summary.id)
        assert restored.messages == before[:index]
        await session.prompt(text + ' revised', images=images)
        assert session.messages[:index] == before[:index]
        assert len(session.messages) == index + 2
        assert 'third' not in str(session.messages)


async def test_rewind_neither_replays_nor_undoes_tools(tmp_path: Path) -> None:
    agent = Agent(TestModel(custom_output_text='done'))
    calls = 0
    side_effect = tmp_path / 'tool-output.txt'

    @agent.tool_plain
    def write_file() -> str:
        nonlocal calls
        calls += 1
        side_effect.write_text('external effect')
        return 'written'

    session = Session(agent, deps=None)
    await session.prompt('write a file')
    assert calls == 1
    menu = build_rewind_menu(session.messages)
    script = Script(lists=[MenuResult(item=menu.highlighted)], choices=[], texts=[])
    async with editor() as (live, _, _):
        await rewind(session, live, runners=script.runners)
        assert live.buffer.text == 'write a file'
    assert session.messages == [] and calls == 1
    assert side_effect.read_text() == 'external effect'


@pytest.mark.parametrize('result', [MenuResult(cancelled=True), MenuResult(), pick(None)])
async def test_cancel_preserves_history_draft_and_attachments(tmp_path: Path, result: MenuResult) -> None:
    session = saved_session(tmp_path)
    await session.prompt('original')
    before, summary = session.messages, session.summary
    async with editor() as (live, _, _):
        marker = live.images.attach([BinaryContent(data=b'draft', media_type='image/png')])
        live.buffer.replace(marker + 'draft')
        pending = dict(live.images.pending)
        script = Script(lists=[result], choices=[], texts=[])
        assert await rewind(session, live, runners=script.runners) == 'No changes.'
        assert live.buffer.text == marker + 'draft' and live.images.pending == pending
    assert session.messages == before and session.summary == summary


async def test_save_conflict_preserves_history_and_draft(tmp_path: Path) -> None:
    session = saved_session(tmp_path)
    await session.prompt('original')
    other = saved_session(tmp_path)
    await other.resume(session.summary.id)
    await other.prompt('saved elsewhere')
    before = session.messages
    point = RewindPoint(message_index=0, text='original', images=())
    script = Script(lists=[pick(point)], choices=[], texts=[])
    async with editor() as (live, _, _):
        live.buffer.replace('keep draft')
        with pytest.raises(ConversationConflict):
            await rewind(session, live, runners=script.runners)
        assert live.buffer.text == 'keep draft'
    assert session.messages == before


@pytest.mark.parametrize('outcome', ['rewind', 'cancel', 'error'])
async def test_shell_opens_rewind_between_turns_and_releases_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    read = LivePrompt.read
    reads = 0
    before: list[ModelMessage] = []
    after: list[ModelMessage] = []

    async def request(live: LivePrompt) -> str:
        nonlocal reads
        reads += 1
        if reads <= 2:
            live.submit('first' if reads == 1 else 'second')
        elif reads == 3:
            live.buffer.replace('my draft')
            live.dismiss_completions()
            live.feed('escape')
            live.feed('escape')
        else:
            assert live.buffer.text == ('second' if outcome == 'rewind' else 'my draft')
            live.submit('/exit')
        return await read(live)

    async def choose(conversation: Conversation, live: LivePrompt) -> str:
        assert live.keys._stack is None  # pyright: ignore[reportPrivateUsage]
        before.extend(conversation.messages)
        if outcome == 'error':
            raise ValueError('save failed')
        menu = build_rewind_menu(conversation.messages)
        script = Script(lists=[MenuResult(item=menu.highlighted, cancelled=outcome == 'cancel')], choices=[], texts=[])
        result = await rewind(conversation, live, runners=script.runners)
        after.extend(conversation.messages)
        return result

    monkeypatch.setattr(LivePrompt, 'read', request)
    monkeypatch.setattr('pydantic_clai2._app.rewind', choose)
    output = StringIO()
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        await chat(
            Agent(TestModel(custom_output_text='answer')),
            deps=None,
            console=Console(file=output, force_terminal=True, width=100),
            store=SettingsStore(tmp_path / 'config.db'),
        )
    assert reads == 4 and len(before) == 4
    assert after == (before[:2] if outcome == 'rewind' else before if outcome == 'cancel' else [])
    assert {'rewind': 'Conversation rewound', 'cancel': 'No changes.', 'error': 'Rewind failed: save failed'}[
        outcome
    ] in output.getvalue()
