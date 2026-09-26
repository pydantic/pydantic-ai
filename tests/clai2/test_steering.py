"""Messages queue first; explicit steering uses core delivery."""

from collections.abc import AsyncIterable
from io import StringIO
from pathlib import Path

import anyio
import pytest
from pydantic_ai import Agent, AgentRunResult, AgentStreamEvent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import BinaryContent, ModelRequest, UserPromptPart
from pydantic_ai.models import Model
from pydantic_ai.models.test import TestModel
from rich.console import Console
from rich.text import Text
from test_live_prompt import editor

from pydantic_clai2._app import create_shell
from pydantic_clai2._session import Session
from pydantic_clai2.live_prompt import LivePrompt
from pydantic_clai2.project_settings import ProjectSettings
from pydantic_clai2.settings_store import SettingsStore


@pytest.mark.parametrize('supplied_handler', [False, True])
async def test_enter_during_run_teardown_queues_follow_up(supplied_handler: bool) -> None:
    finishing, release = anyio.Event(), anyio.Event()

    class PauseAfterRun(AbstractCapability[None]):
        async def after_run(self, ctx: RunContext[None], *, result: AgentRunResult[str]) -> AgentRunResult[str]:
            finishing.set()
            await release.wait()
            return result

    async def handler(ctx: RunContext[None], events: AsyncIterable[AgentStreamEvent]) -> None:
        async for _ in events:
            pass
        finishing.set()
        await release.wait()

    class ObservedAgent(Agent[None, str]):
        @property
        def event_stream_handler(self):
            return handler if supplied_handler else None

    session = Session(ObservedAgent(TestModel(), deps_type=type(None), capabilities=[PauseAfterRun()]), deps=None)
    async with editor() as (live, _, _):
        live.steer = session.steer
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(session.prompt, 'start')
            await finishing.wait()
            live.buffer.replace('follow up during teardown')
            live.feed('enter')
            assert await live.read() == 'follow up during teardown'
            release.set()
        assert not session.steer('finished')


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.mark.parametrize('sequence', ['\x1b\r', '\x1b[13;3u', '\x1b[27;3;13~'])
async def test_enter_queues_alt_enter_steers_oldest(sequence: str) -> None:
    accepted: list[str] = []

    delivered = anyio.Event()

    def steer(text: str) -> bool:
        accepted.append(text)
        delivered.set()
        return True

    async with editor() as (live, pipe, _):
        live.steer = steer
        live.buffer.replace('change direction')
        live.feed('enter')
        assert accepted == []
        assert live.queued_messages == ('change direction',)
        assert live.buffer.text == ''
        live.buffer.replace('follow up')
        live.feed('enter')
        live.buffer.replace('unfinished draft')
        pipe.send_text(sequence)
        await delivered.wait()
        assert accepted == ['change direction']
        assert live.queued_messages == ('follow up',)
        assert live.buffer.text == 'unfinished draft'
        assert await live.read() == 'follow up'
        live.buffer.replace('/help')
        live.feed('enter')
        assert await live.read() == '/help'
        assert accepted == ['change direction']
        assert Text.from_ansi(live.frame()[-1]).plain == 'ready'

        def idle(text: str) -> bool:
            return False

        live.steer = idle
        live.buffer.replace('idle prompt')
        live.feed('enter')
        assert await live.read() == 'idle prompt'


async def test_bare_clear_queues_as_a_command_that_steering_skips() -> None:
    attempted: list[str] = []

    def steer(text: str) -> bool:
        attempted.append(text)
        return True

    async with editor() as (live, _, _):
        live.steer = steer
        live.buffer.replace(' Clear ')
        live.feed('enter')
        assert live.queued_messages == ('/clear',)
        live.feed('alt-enter')
        assert attempted == []
        assert await live.read() == '/clear'


@pytest.mark.parametrize('head', ['/help', '!git status', KeyboardInterrupt(), EOFError(), 'follow up'])
@pytest.mark.parametrize('available', [False, True])
async def test_unavailable_steering_preserves_queue(head: str | KeyboardInterrupt | EOFError, available: bool) -> None:
    attempted: list[str] = []

    def idle(text: str) -> bool:
        attempted.append(text)
        return False

    async with editor() as (live, _, _):
        live.steer = idle if available else None
        live.feed('alt-enter')
        live.submit(head)
        live.submit('second')
        live.buffer.replace('draft')
        live.feed('alt-enter')
        assert attempted == (['follow up'] if available and head == 'follow up' else [])
        assert live.buffer.text == 'draft'
        if isinstance(head, str):
            assert await live.read() == head
        else:
            with pytest.raises(type(head)):
                await live.read()
        assert await live.read() == 'second'


async def test_steering_last_message_clears_queue_and_preserves_draft() -> None:
    def steer(text: str) -> bool:
        return True

    async with editor() as (live, _, _):
        live.steer = steer
        live.buffer.replace('queued')
        live.feed('enter')
        live.buffer.replace('draft')
        live.feed('alt-enter')
        assert live.queued_messages == ()
        assert live.buffer.text == 'draft'
        live.feed('alt-enter')
        live.feed('enter')
        assert await live.read() == 'draft'


async def test_steering_reaches_running_agent_and_is_cleared() -> None:
    started = anyio.Event()
    release = anyio.Event()
    agent = Agent(TestModel())

    @agent.tool_plain
    async def wait_for_input() -> str:
        started.set()
        await release.wait()
        return 'done'

    session = Session(agent, deps=None)
    assert not session.steer('idle')
    async with anyio.create_task_group() as tasks:
        tasks.start_soon(session.prompt, 'start')
        await started.wait()
        assert session.steer('change direction')
        assert session.steer('image direction', images=[BinaryContent(data=b'image', media_type='image/png')])
        release.set()
    prompts = [
        part.content
        for message in session.messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart)
    ]
    assert 'change direction' in prompts
    assert not session.steer('finished')


@pytest.mark.parametrize('cancel', [False, True])
async def test_steering_before_model_resolution_and_cancellation(cancel: bool) -> None:
    started = anyio.Event()
    release = anyio.Event()
    model = TestModel(custom_output_text='answer')
    session = Session(Agent(model), deps=None)
    session.model = 'test'

    async def resolve(name: str) -> Model:
        started.set()
        await release.wait()
        return model

    session.resolve_model = resolve
    async with anyio.create_task_group() as tasks:
        tasks.start_soon(session.prompt, 'start')
        await started.wait()
        assert session.steer('early direction')
        assert session.steer('early image', images=[BinaryContent(data=b'image', media_type='image/png')])
        if cancel:
            tasks.cancel_scope.cancel()
        else:
            release.set()
    assert not session.steer('finished')
    session.model = None
    await session.prompt('next turn')
    prompts = [
        part.content
        for message in session.messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart)
    ]
    assert ('early direction' in prompts) is not cancel


async def test_shell_routes_steering_and_reports_expired_images(tmp_path: Path) -> None:
    agent = Agent(TestModel())
    started, release = anyio.Event(), anyio.Event()

    @agent.tool_plain
    async def wait_for_input() -> str:
        started.set()
        await release.wait()
        return 'done'

    shell = create_shell(
        agent,
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=StringIO()),
        settings=None,
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
    )
    async with editor() as (live, _, _):
        live.steer = shell.steer
        live.buffer.replace('idle')
        live.feed('enter')
        live.feed('alt-enter')
        assert live.queued_messages == ('idle',)
        assert await live.read() == 'idle'
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(shell.session.prompt, 'start')
            await started.wait()
            live.buffer.replace('new direction')
            live.feed('enter')
            assert live.queued_messages == ('new direction',)
            live.feed('alt-enter')
            assert shell.images.notice.startswith('Steering sent: new direction')
            assert live.queued_messages == ()
            release.set()
        live.buffer.replace('[image:12345678]')
        live.feed('enter')
        live.feed('alt-enter')
        assert 'expired' in shell.images.notice
        assert live.queued_messages == ()


def queue(live: LivePrompt, *texts: str) -> None:
    for text in texts:
        live.buffer.replace(text)
        live.feed('enter')


def queue_rows(live: LivePrompt) -> list[str]:
    return [Text.from_ansi(row).plain for row in live.frame()[: len(live.queued_messages)]]


async def test_up_walks_queue_newest_first_then_history() -> None:
    async with editor() as (live, _, _):
        live.buffer.history = ['old']
        queue(live, 'first', 'second')
        live.buffer.replace('draft')
        walk = [live.feed('up') or live.buffer.text for _ in range(4)]
        # History already holds the queued prompts; the walk skips those copies.
        assert walk == ['second', 'first', 'old', 'old']
        walk = [live.feed('down') or live.buffer.text for _ in range(4)]
        assert walk == ['first', 'second', 'draft', 'draft']
        assert live.queued_messages == ('first', 'second')


async def test_up_moves_within_a_multiline_draft_before_reaching_the_queue() -> None:
    async with editor() as (live, _, _):
        queue(live, 'queued')
        live.buffer.replace('one\ntwo')
        live.feed('up')
        assert (live.buffer.text, live.buffer.cursor, live.buffer.recall_offset) == ('one\ntwo', 3, None)
        live.feed('up')
        assert live.buffer.text == 'queued'
        live.feed('down')
        assert live.buffer.text == 'one\ntwo'


async def test_enter_rewrites_recalled_queued_prompt_in_place() -> None:
    async with editor() as (live, _, _):
        queue(live, 'a', 'b', 'c')
        live.feed('up')
        live.feed('up')
        assert queue_rows(live) == ['Follow-up: a', 'Follow-up (editing): b', 'Follow-up: c']
        live.feed('ctrl-u')
        live.buffer.insert('b2')
        live.feed('enter')
        assert live.queued_messages == ('a', 'b2', 'c')
        assert live.buffer.text == ''
        assert live.buffer.history[-1] == 'b2'
        assert queue_rows(live) == ['Follow-up: a', 'Follow-up: b2', 'Follow-up: c']
        live.feed('up')
        live.feed('enter')
        assert live.queued_messages == ('a', 'b2', 'c')
        assert live.buffer.history[-1] == 'b2'
        # The replaced draft stays in history; only the queued prompts' own copies are skipped.
        walk = [live.feed('up') or live.buffer.text for _ in range(5)]
        assert walk == ['c', 'b2', 'a', 'b', 'b']


async def test_walk_skips_only_the_history_copy_of_each_queued_prompt() -> None:
    async with editor() as (live, _, _):
        live.buffer.history = ['deploy']
        queue(live, 'deploy', 'clear')
        walk = [live.feed('up') or (live.buffer.text, live.buffer.recall_offset) for _ in range(4)]
        # The older `deploy` stays reachable, and raw `clear` is not shown beside its `/clear` expansion.
        assert walk == [('/clear', -1), ('deploy', -2), ('deploy', -3), ('deploy', -3)]


async def test_edited_queued_draft_survives_a_second_walk() -> None:
    async with editor() as (live, _, _):
        queue(live, 'a', 'b')
        live.feed('up')
        live.buffer.insert('x')
        live.feed('up')
        assert live.buffer.text == 'b'
        live.feed('down')
        assert live.buffer.text == 'bx'
        live.feed('enter')
        assert live.queued_messages == ('a', 'bx')


async def test_deleting_from_a_recalled_queued_prompt_survives_navigation() -> None:
    async with editor() as (live, _, _):
        queue(live, 'a', 'bc')
        live.feed('up')
        live.feed('backspace')
        live.feed('up')
        assert live.buffer.text == 'bc'
        live.feed('down')
        assert live.buffer.text == 'b'
        live.feed('enter')
        assert live.queued_messages == ('a', 'b')


async def test_history_search_from_a_recalled_queued_prompt() -> None:
    async with editor() as (live, _, _):
        live.buffer.history = ['old']
        queue(live, 'b')
        live.feed('up')
        for key in ('ctrl-r', 'o', 'enter', 'enter'):
            live.feed(key)
        # A picked history match is a new follow-up; the recalled queued prompt stays as it was.
        assert live.queued_messages == ('b', 'old')
        live.feed('up')
        live.feed('up')
        live.buffer.insert('x')
        for key in ('ctrl-r', 'o', 'ctrl-g'):
            live.feed(key)
        assert live.buffer.text == 'bx'
        live.feed('enter')
        # Cancelling the search keeps editing the queued prompt it started from.
        assert live.queued_messages == ('bx', 'old')


async def test_clearing_a_recalled_queued_prompt_removes_it() -> None:
    async with editor() as (live, _, _):
        queue(live, 'only')
        live.feed('up')
        live.feed('ctrl-u')
        live.feed('enter')
        assert live.queued_messages == ()
        live.submit('later')
        assert await live.read() == 'later'


async def test_recalling_history_leaves_the_queued_prompt_alone() -> None:
    async with editor() as (live, _, _):
        live.buffer.history = ['old']
        queue(live, 'queued')
        live.feed('up')
        live.feed('up')
        assert live.buffer.text == 'old'
        assert queue_rows(live) == ['Follow-up: queued']
        live.feed('enter')
        assert live.queued_messages == ('queued', 'old')


async def test_edit_of_a_consumed_prompt_becomes_a_new_follow_up() -> None:
    async with editor() as (live, _, _):
        queue(live, 'taken')
        live.feed('up')
        assert await live.read() == 'taken'
        live.buffer.insert(' again')
        live.feed('enter')
        assert live.queued_messages == ('taken again',)


async def test_recalled_queued_prompt_edited_into_immediate_command_leaves_queue() -> None:
    ran: list[str] = []

    def run_now(text: str) -> bool:
        ran.append(text)
        return text == '/now'

    async with editor() as (live, _, _):
        queue(live, 'keep', 'swap')
        live.run_now = run_now
        live.feed('up')
        live.buffer.replace('/now')
        live.feed('enter')
        live.feed('up')
        live.buffer.replace('changed')
        live.feed('enter')
        assert ran == ['/now', 'changed']
        assert live.queued_messages == ('changed',)
