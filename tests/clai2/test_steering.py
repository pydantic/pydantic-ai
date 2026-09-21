"""Steering uses core delivery; only explicit follow-ups start another turn."""

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
async def test_enter_steers_alt_enter_queues(sequence: str) -> None:
    accepted: list[str] = []

    def steer(text: str) -> bool:
        accepted.append(text)
        return True

    async with editor() as (live, pipe, _):
        live.steer = steer
        live.buffer.replace('change direction')
        live.feed('enter')
        assert accepted == ['change direction']
        assert live.queued_messages == ()
        assert live.buffer.text == ''
        pipe.send_text(f'follow up{sequence}')
        assert await live.read() == 'follow up'
        live.buffer.replace('/help')
        live.feed('enter')
        assert await live.read() == '/help'
        assert accepted == ['change direction']
        assert 'Enter: submit | Alt+Enter: queue' in Text.from_ansi(live.frame()[-1]).plain

        def idle(text: str) -> bool:
            return False

        live.steer = idle
        live.buffer.replace('idle prompt')
        live.feed('enter')
        assert await live.read() == 'idle prompt'


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
        assert await live.read() == 'idle'
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(shell.session.prompt, 'start')
            await started.wait()
            live.buffer.replace('new direction')
            live.feed('enter')
            assert shell.images.notice.startswith('Steering sent: new direction')
            assert live.queued_messages == ()
            release.set()
        live.buffer.replace('[image:12345678]')
        live.feed('enter')
        assert 'expired' in shell.images.notice
        assert live.queued_messages == ()
