"""`web.client` against the real web app: a headless shell served on loopback, in this process, driven as a page would."""

import io
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import anyio
import httpx
import pytest
from pydantic import TypeAdapter
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.ask_user import AskUserAnswer
from pydantic_clai2.builtin_plugins.web import WebPlugin
from pydantic_clai2.builtin_plugins.web.app import HEADER
from pydantic_clai2.builtin_plugins.web.client import SessionView, WebClient, _error  # pyright: ignore[reportPrivateUsage]
from pydantic_clai2.builtin_plugins.web.protocol import MenuEdited, TitleChanged, TurnEnded, TurnStarted
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.runtime.session_runtime import SessionRuntime, create_shell
from tests.clai2.test_web import QuestionModel

pytestmark = pytest.mark.anyio


@dataclass
class Served:
    shell: SessionRuntime[None, str]
    web: WebPlugin
    client: WebClient
    view: SessionView


@asynccontextmanager
async def serving(agent: Agent[None, str], tmp_path: Path, *extra: PluginSettings) -> AsyncGenerator[Served]:
    """A headless shell with `web` serving it on a free loopback port, and a client following it."""
    shell = create_shell(
        agent,
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=io.StringIO(), width=100),
        settings=None,
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=[
            PluginSettings(id='web', factory='pydantic_clai2.builtin_plugins.web', settings={'open_browser': False}),
            *extra,
        ],
        project=ProjectSettings(),
        headless=True,
    )
    await shell.loader.load_all()
    [web] = [entry.loaded.plugin for entry in shell.loader.entries() if entry.name == 'web' and entry.loaded]
    assert isinstance(web, WebPlugin)
    address, _, _ = (await web.start()).partition('?')
    try:
        async with anyio.create_task_group() as group, WebClient(url=address, token=web.token) as client:
            group.start_soon(shell.run)
            view = SessionView(client)
            group.start_soon(view.follow)
            with anyio.fail_after(10):
                await view.ready()
            yield Served(shell=shell, web=web, client=client, view=view)
            group.cancel_scope.cancel()
    finally:
        await shell.loader.close('exit')


async def test_a_prompt_and_its_turn(tmp_path: Path) -> None:
    """A prompt sent through the view is waited for until its turn ends; the items match the server's."""
    tools = Capability[None]()

    @tools.tool_plain
    def lookup(*, query: str) -> str:
        """Look something up."""
        return f'found {query}'

    agent = Agent(TestModel(call_tools=['lookup']), deps_type=type(None), capabilities=[tools])
    async with serving(agent, tmp_path) as served:
        view = served.view
        assert view.session is not None and view.items == [] and not view.running
        assert await view.wait(timeout=1) == 'idle'
        await view.send('look it up')
        assert await view.wait(timeout=10) == 'idle'
        assert view.last_turn is not None and view.last_turn.outcome == 'completed'
        assert [(item.kind, item.name) for item in view.items] == [('user', ''), ('tool', 'lookup'), ('assistant', '')]
        assert view.items[0].text == 'look it up' and 'found' in view.items[1].text
        state = await served.client.state()
        assert [item.text for item in state.items] == [item.text for item in view.items]
        # The turn ends for the page just before the session saves its answer; its prompt is saved already.
        transcript = await served.client.transcript(view.session.id)
        assert (transcript.id, transcript.items[0].text) == (view.session.id, 'look it up')


async def test_commands_and_their_notices(tmp_path: Path) -> None:
    """A `/command` counts as queued until it ends, though it starts no turn; its result is a notice."""
    async with serving(Agent(TestModel()), tmp_path) as served:
        view = served.view
        await view.send('/web status')
        assert await view.wait(timeout=10) == 'idle'
        [notice] = view.notices
        assert notice.command == '/web status' and notice.text.startswith('Web UI at http://127.0.0.1:')
        assert served.web.token not in repr(view.items)


async def test_a_bare_command_with_a_menu_is_refused(tmp_path: Path) -> None:
    """`/set` alone would open its menu in a page, which a client has not got: it says so, and nothing runs."""
    async with serving(Agent(TestModel()), tmp_path) as served:
        with pytest.raises(ValueError, match=r'/set opens a menu in the page; send it with arguments instead\.'):
            await served.client.send('/set')
        await served.view.send('/set display.thinking false')
        assert await served.view.wait(timeout=10) == 'idle'
        assert served.shell.context.settings.thinking is False


@pytest.mark.parametrize(
    ('answers', 'expected'),
    [
        ([AskUserAnswer(header='Colour', selected=('Blue',))], 'Blue'),
        ([AskUserAnswer(header='Colour', custom_answer='Teal')], 'Teal'),
        (None, 'declined'),
    ],
)
async def test_questions_are_answered(answers: list[AskUserAnswer] | None, expected: str, tmp_path: Path) -> None:
    """`wait` stops for a question; an answer, a typed one, or declining lets the turn go on."""
    agent = Agent(QuestionModel(call_tools=['ask_user_question']), deps_type=type(None))
    async with serving(agent, tmp_path) as served:
        view, client = served.view, served.client
        await view.send('ask me')
        assert await view.wait(timeout=10) == 'asking'
        [asked] = view.questions.values()
        assert [option.label for option in asked.questions[0].options] == ['Red', 'Blue']
        with pytest.raises(LookupError, match='CLAI answered 404'):
            await client.answer('unknown', None)
        await view.answer(asked.id, answers)
        assert await view.wait(timeout=10) == 'idle'
        assert expected in next(item for item in view.items if item.kind == 'tool').text


async def test_interrupting_a_turn(tmp_path: Path) -> None:
    started = anyio.Event()
    tools = Capability[None]()

    @tools.tool_plain
    async def wait() -> str:
        """Wait forever."""
        started.set()
        await anyio.sleep_forever()
        return 'never'  # pragma: no cover

    agent = Agent(TestModel(call_tools=['wait']), deps_type=type(None), capabilities=[tools])
    async with serving(agent, tmp_path) as served:
        view, client = served.view, served.client
        assert await client.interrupt() is False
        await view.send('wait')
        await started.wait()
        assert await view.wait(timeout=0.05) == 'timeout'
        assert view.running and [(item.kind, item.state) for item in view.items] == [
            ('user', 'done'),
            ('tool', 'running'),
        ]
        assert await client.interrupt() is True
        assert await view.wait(timeout=10) == 'idle'
        assert view.last_turn is not None and view.last_turn.outcome == 'cancelled'


async def test_dialogs_are_answered(tmp_path: Path) -> None:
    """A settings menu's dialog makes `wait` stop; answering it lets the action finish."""
    workspaces = PluginSettings(id='workspaces', factory='pydantic_clai2.builtin_plugins.workspaces')
    async with serving(Agent(TestModel()), tmp_path, workspaces) as served:
        view, client = served.view, served.client
        headers = {HEADER: served.web.token}
        results: list[httpx.Response] = []
        async with httpx.AsyncClient(base_url=client.url, trust_env=False) as page, anyio.create_task_group() as group:

            async def act() -> None:
                results.append(await page.post('/api/plugins/workspaces/fields/add/action', headers=headers))

            group.start_soon(act)
            with anyio.fail_after(10):
                while not view.dialogs:
                    await anyio.sleep(0.01)
            assert await view.wait(timeout=10) == 'asking'
            [dialog] = view.dialogs.values()
            assert dialog.title == 'New workspace kind'
            with pytest.raises(LookupError, match='CLAI answered 404'):
                await client.answer_dialog('unknown', 'x')
            await view.answer_dialog(dialog.id, 'docs')
        assert TypeAdapter(MenuEdited).validate_json(results[0].content).messages == ['Added workspace kind docs.']
        assert await view.wait(timeout=10) == 'idle' and view.dialogs == {}


async def test_the_title_and_a_closed_connection(tmp_path: Path) -> None:
    async with serving(Agent(TestModel()), tmp_path) as served:
        view = served.view
        await view.send('hello')
        assert await view.wait(timeout=10) == 'idle'
        # The first prompt titles the session, which the page hears before the turn starts.
        assert view.session is not None and view.session.title == 'hello'
        await served.web.stop()
        assert await view.wait(timeout=10) == 'closed'


async def test_refusals(tmp_path: Path) -> None:
    async with serving(Agent(TestModel()), tmp_path) as served:
        client, view = served.client, served.view
        with pytest.raises(ValueError, match='Type a prompt or a /command'):
            await view.send('  ')
        assert await view.wait(timeout=1) == 'idle', 'a refused input is not queued'
        with pytest.raises(LookupError):
            await client.transcript('00000000-0000-0000-0000-000000000000')
        stranger = WebClient(url=client.url, token='wrong')
        with pytest.raises(PermissionError, match='CLAI answered 403'):
            await stranger.state()
        await stranger.__aexit__(None, None, None)
    with pytest.raises(ValueError, match='http://HOST:PORT/'):
        WebClient(url='ws://127.0.0.1:1/', token='x')


def test_errors_map_as_the_app_raised_them() -> None:
    request = httpx.Request('GET', 'http://127.0.0.1/')
    conflict = _error(httpx.Response(409, json={'detail': 'Something is running.'}, request=request))
    assert isinstance(conflict, RuntimeError) and str(conflict) == 'CLAI answered 409: Something is running.'
    invalid = _error(httpx.Response(422, text='not json', request=request))
    assert isinstance(invalid, ValueError) and str(invalid) == 'CLAI answered 422: not json'


async def test_a_failed_send_takes_back_only_its_own_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """A send that fails removes its own queued input, not another's, and none once its turn started."""
    view = SessionView(WebClient(url='http://127.0.0.1:1/', token='x'))
    entered: list[str] = []

    async def send(text: str) -> None:
        entered.append(text)
        if text == 'started then lost':
            view.apply(TurnStarted(item=None))  # The turn started before the answer was lost.
        if text != 'fine':
            raise ConnectionError('lost')

    monkeypatch.setattr(view.client, 'send', send)
    await view.send('fine')
    with pytest.raises(ConnectionError):
        await view.send('/fails')
    assert await view.wait(timeout=0.01) == 'timeout', "the first input's turn has not started"
    view.apply(TurnStarted(item=None))
    view.apply(TurnEnded(outcome='completed'))
    assert await view.wait(timeout=1) == 'idle'
    with pytest.raises(ConnectionError):
        await view.send('started then lost')
    view.apply(TurnEnded(outcome='completed'))
    assert await view.wait(timeout=1) == 'idle' and entered == ['fine', '/fails', 'started then lost']


async def test_messages_without_a_snapshot_or_a_prompt() -> None:
    view = SessionView(WebClient(url='http://127.0.0.1:1/', token='x'))
    view.apply(TitleChanged(title='Early'))
    assert view.session is None
    view.apply(TurnStarted(item=None))  # A turn a finished delegated task woke.
    assert view.running and view.items == []
