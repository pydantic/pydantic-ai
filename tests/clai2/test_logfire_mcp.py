"""The `logfire_mcp` built-in, shown as Logfire: its settings menu, keys kept in `/keys`, credential order, and OAuth."""

import inspect
import io
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from pathlib import Path

import anyio
import pytest
from fastmcp import Client
from keyring.errors import KeyringError
from pydantic import JsonValue, SecretStr
from rich.console import Console
from termflow.tui import MenuItem
from termflow.tui.menu import Menu, MenuResult
from termflow.tui.textinput import TextInputResult

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    LoadCapabilityReturnPart,
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.logfire_mcp import LOGFIRE_EU_MCP_URL, LOGFIRE_US_MCP_URL, LogfireMCP
from pydantic_ai_harness.skills import Skills
from pydantic_clai2 import DEFAULT_PLUGINS, Session
from pydantic_clai2.builtin_plugins.logfire_destination import OTHER, REGIONS, Destination, remember, remembered
from pydantic_clai2.builtin_plugins.logfire_mcp import SETUP, LogfireMCPPlugin, LogfireMCPSource
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import api_keys
from pydantic_clai2.config.api_keys import KeyReference, SavedKey
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.logfire_oauth import DeviceAuth, LogfireSignIn, SignInError, Tokens
from pydantic_clai2.plugins import PluginHost, SessionStart, load_plugin
from pydantic_clai2.plugins.loader import PluginError, PluginLoader
from pydantic_clai2.plugins.menus import SettingsMenu
from pydantic_clai2.ui.menus import dialogs
from pydantic_clai2.ui.menus.field_menu import FieldMenu, Runners, is_save_and_close
from pydantic_clai2.ui.menus.plugin_menu import PluginMenu, open_plugins_menu
from tests.clai2.dialog_script import Answer, Asked, Back, Confirm, Pick, ScriptedDialogs, Type
from tests.clai2.menu_script import UNTIL_CLOSED, Script, pick, typed
from tests.clai2.test_workspace_defaults import per_run

BUILTIN = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'logfire_mcp')
CLOSE = MenuResult(cancelled=True)
SELF_HOSTED = 'https://logfire.example.com/mcp'
KEY_LABEL = 'Logfire API key (saved in /keys as LOGFIRE_API_KEY)'


class Shell:
    """A loader plus what a test inspects: printed output and the settings file."""

    def __init__(self, tmp_path: Path, settings: dict[str, JsonValue] | None = None, *, terminal: bool = True) -> None:
        self.path = tmp_path / 'config.db'
        self.store = SettingsStore(self.path)
        self.output = io.StringIO()
        declaration = BUILTIN if settings is None else BUILTIN.model_copy(update={'settings': settings})
        self.loader: PluginLoader[None] = PluginLoader(
            store=self.store,
            # A terminal without colours: the settings menu opens in it, and printed warnings stay plain text.
            console=Console(file=self.output, width=200, force_terminal=terminal, color_system=None),
            commands=Commands(),
            session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=self.store.load()),
            builtin=(declaration,),
        )

    def saved(self) -> dict[str, JsonValue]:
        [declaration] = self.store.plugins()
        return declaration.settings

    def capability(self) -> LogfireMCP[None]:
        """The connection chosen at session start, before a run resolves its saved key."""
        [capability] = self.loader.capabilities()
        factory = per_run(capability)
        assert inspect.ismethod(factory)
        plugin = factory.__self__
        assert isinstance(plugin, LogfireMCPPlugin) and plugin.capability is not None
        return plugin.capability

    async def run_capability(self) -> LogfireMCP[None] | None:
        """What a run gets."""
        [capability] = self.loader.capabilities()
        factory = per_run(capability)
        assert inspect.ismethod(factory)
        plugin = factory.__self__
        assert isinstance(plugin, LogfireMCPPlugin)
        return await plugin._for_run(RunContext[None](deps=None, model=TestModel(), usage=RunUsage()))  # pyright: ignore[reportPrivateUsage]

    async def act(self, key: str, answers: Sequence[Answer]) -> tuple[list[str], list[Asked]]:
        """Run a menu row's action as the web UI does, answering its dialogs; its messages and what it asked."""
        scripted = ScriptedDialogs(answers)

        async def edit(menu: SettingsMenu) -> Sequence[str]:
            return await menu.actions[key](scripted)

        messages = await self.loader.edit_settings('logfire_mcp', edit)
        assert scripted.unused == []
        return list(messages), scripted.asked


def script(
    monkeypatch: pytest.MonkeyPatch,
    lists: list[MenuResult],
    choices: list[MenuResult] | None = None,
    texts: list[TextInputResult] | None = None,
) -> Script:
    """Drive the declared menu in the terminal: its list, then each dialog an action opens."""
    scripted = Script(lists=[*lists, CLOSE], choices=choices or [], texts=texts or [])
    monkeypatch.setattr(dialogs, 'RUNNERS', scripted.runners)
    return scripted


def watch_highlights(monkeypatch: pytest.MonkeyPatch, shown: Script) -> list[object]:
    """Record which row each terminal picker starts on."""
    highlighted: list[object] = []

    def watch(menu: Menu) -> MenuResult:
        highlighted.append(menu.highlighted.value if menu.highlighted else None)
        return shown.run_choice(menu)

    monkeypatch.setattr(dialogs, 'RUNNERS', Runners(run_list=shown.run_list, run_choice=watch, run_text=shown.run_text))
    return highlighted


@pytest.fixture(autouse=True)
def opened(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record sign-in links instead of opening a browser."""
    links: list[str] = []

    def open_link(url: str) -> bool:  # pragma: no cover - a safety net; tests here stub the sign-in before a link opens
        links.append(url)
        return True

    monkeypatch.setattr('pydantic_clai2.logfire_oauth.open_browser', open_link)
    return links


def keyed(name: str) -> LogfireMCP[None]:
    return LogfireMCP[None](defer_loading=True, auth=SavedKey(name=name, setup=SETUP), read_only=True)


def test_declared_disabled_with_no_settings() -> None:
    assert (BUILTIN.factory, BUILTIN.enabled, BUILTIN.settings) == (
        'pydantic_clai2.builtin_plugins.logfire_mcp',
        False,
        {},
    )


async def test_enable_opens_the_menu_and_every_option_saves_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shown = script(
        monkeypatch,
        lists=[pick('key'), pick('url'), pick('read_only'), pick('include_instructions'), pick('oauth')],
        choices=[pick(OTHER), pick('false'), pick('false'), pick('false')],
        texts=[typed(' typed-secret '), typed('logfire.example.com')],
    )
    shell = Shell(tmp_path)
    assert await shell.loader.command(['enable', 'logfire_mcp']) == '\n'.join(
        [
            'Enabled logfire_mcp.',
            'Logfire uses the saved key LOGFIRE_API_KEY. Manage it in /keys.',
            'Logfire tools connect to logfire.example.com at https://logfire.example.com/mcp.',
            'Saved Tools.',
            'Saved Server instructions.',
            'Saved Browser sign-in.',
        ]
    )
    assert shown.opened.count('list') == 6
    assert api_keys.load_keys()['LOGFIRE_API_KEY'].get_secret_value() == 'typed-secret'
    assert shell.saved() == {
        'key': {'name': 'LOGFIRE_API_KEY'},
        'url': SELF_HOSTED,
        'oauth': False,
        'read_only': False,
        'include_instructions': False,
    }
    assert b'typed-secret' not in shell.path.read_bytes()
    capability = shell.capability()
    assert capability == LogfireMCP[None](
        defer_loading=True,
        auth=SavedKey(name='LOGFIRE_API_KEY', setup=SETUP),
        url=SELF_HOSTED,
        read_only=False,
        include_instructions=False,
    )
    # Observability setup starts from the same Logfire.
    assert remembered() == Destination(base_url='https://logfire.example.com')


async def test_reopening_repicks_a_saved_key_and_region_without_reinstalling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api_keys.save_key(name='SHARED', value='shared-secret')
    shell = Shell(tmp_path, {'key': {'name': 'LOGFIRE_API_KEY'}})
    script(monkeypatch, lists=[])
    await shell.loader.command(['enable', 'logfire_mcp'])
    assert 'Logfire has no credential: LOGFIRE_API_KEY is not in /keys. Run /plugins' in shell.output.getvalue()
    script(monkeypatch, lists=[pick('key'), pick('url')], choices=[pick('SHARED'), pick('Logfire EU')])
    assert await shell.loader.command(['configure', 'logfire_mcp']) == (
        'Logfire uses the saved key SHARED. Manage it in /keys.\n'
        'Logfire tools connect to Logfire EU at https://logfire-eu.pydantic.dev/mcp.'
    )
    assert shell.saved()['key'] == {'name': 'SHARED'}
    assert b'shared-secret' not in shell.path.read_bytes()
    capability = shell.capability()
    assert (capability.auth, capability.url) == (SavedKey(name='SHARED', setup=SETUP), LOGFIRE_EU_MCP_URL)


async def test_resetting_the_key_clears_the_choice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('LOGFIRE_API_KEY', 'env-key')
    shell = Shell(tmp_path, {'key': {'name': 'OLD'}})
    script(monkeypatch, lists=[])
    await shell.loader.enable('logfire_mcp')
    host = PluginHost[None](name='logfire_mcp', console=Console(file=io.StringIO()), settings={})
    menu = await anyio.to_thread.run_sync(FieldMenu, LogfireMCPSource(host))
    script(monkeypatch, lists=[menu.reset_marker(None, MenuItem('API key', value='key'))])
    assert await shell.loader.configure('logfire_mcp') == 'Reset API key.'
    assert shell.saved().get('key') is None
    assert shell.capability() == LogfireMCP[None](defer_loading=True, read_only=True)


async def test_a_new_key_is_typed_masked_when_no_keys_are_saved(tmp_path: Path) -> None:
    shell = Shell(tmp_path, {'oauth': False})
    await shell.loader.enable('logfire_mcp')
    messages, asked = await shell.act('key', [Type(text='masked-secret')])
    assert messages == ['Logfire uses the saved key LOGFIRE_API_KEY. Manage it in /keys.']
    assert asked == [
        Asked(
            kind='text',
            title=KEY_LABEL,
            label='Token',
            placeholder='Paste the token; it is saved in /keys as LOGFIRE_API_KEY',
            secret=True,
        )
    ]
    assert api_keys.load_keys()['LOGFIRE_API_KEY'].get_secret_value() == 'masked-secret'
    assert shell.capability() == keyed('LOGFIRE_API_KEY')


@pytest.mark.parametrize('answer', [Back(), Type(text='   ')], ids=['back', 'blank'])
async def test_cancelled_or_blank_key_changes_nothing(tmp_path: Path, answer: Answer) -> None:
    shell = Shell(tmp_path, {'oauth': False})
    await shell.loader.enable('logfire_mcp')
    assert (await shell.act('key', [answer]))[0] == []
    assert api_keys.load_keys() == {}
    assert shell.saved() == {'oauth': False}


async def test_replacing_a_shared_key_needs_confirmation(tmp_path: Path) -> None:
    api_keys.save_key(name='LOGFIRE_API_KEY', value='shared')
    shell = Shell(tmp_path)
    await shell.loader.enable('logfire_mcp')
    messages, asked = await shell.act('key', [Pick(value='enter'), Type(text='other'), Back()])
    assert messages == []
    assert [(a.kind, a.title) for a in asked] == [
        ('choose', KEY_LABEL),
        ('text', KEY_LABEL),
        ('confirm', 'LOGFIRE_API_KEY is already in /keys'),
    ]
    assert [option.value for option in asked[0].options] == ['LOGFIRE_API_KEY', 'enter']
    assert asked[2].yes == 'Replace LOGFIRE_API_KEY for every plugin and connection that uses it'
    assert api_keys.load_keys()['LOGFIRE_API_KEY'].get_secret_value() == 'shared'
    messages, _ = await shell.act('key', [Pick(value='enter'), Type(text='other'), Confirm(yes=True)])
    assert messages == ['Logfire uses the saved key LOGFIRE_API_KEY. Manage it in /keys.']
    assert api_keys.load_keys()['LOGFIRE_API_KEY'].get_secret_value() == 'other'


def test_menu_validates_resets_and_notes_where_the_key_comes_from(monkeypatch: pytest.MonkeyPatch) -> None:
    host = PluginHost[None](name='logfire_mcp', console=Console(file=io.StringIO()), settings={'read_only': False})
    source = LogfireMCPSource(host)

    def note() -> str:
        return {row.key: row for row in source.rows()}['key'].note

    rows = {row.key: row for row in source.rows()}
    assert (source.current(rows['key']), note()) == ('(none)', 'browser sign-in, if on')
    assert rows['oauth'].note == 'signed out: /logfire login signs in'
    assert (rows['url'].note, rows['read_only'].note) == ('', '')
    api_keys.save_key(name='LOGFIRE_API_KEY', value='saved')
    assert note() == 'LOGFIRE_API_KEY from /keys'
    monkeypatch.setenv('LOGFIRE_API_KEY', 'env')
    assert note() == 'LOGFIRE_API_KEY from the environment'
    source.save(source.settings.model_copy(update={'key': KeyReference(name='GONE')}))
    assert (source.current(rows['key']), note()) == ('GONE', 'missing from /keys')
    source.save(source.settings.model_copy(update={'key': KeyReference(name='LOGFIRE_API_KEY')}))
    assert note() == ''
    bad_url = 'Value error, Use an https:// URL without credentials, a query, or a fragment.'
    for url in ('http://h.test/mcp', 'https://u:pw@h.test/mcp', 'https://h.test/mcp?a=1', 'https://h.test/mcp#staging'):
        assert source.problem(rows['url'], url) == bad_url
    assert source.problem(rows['read_only'], 'maybe') == 'Input should be a valid boolean'
    assert source.problem(rows['url'], SELF_HOSTED) is None
    # A host or the URL you open Logfire at is saved as the MCP URL; another path, as earlier builds saved it, is kept.
    for url, saved in [
        ('logfire-eu.pydantic.dev', LOGFIRE_EU_MCP_URL),
        ('https://logfire.example.com', SELF_HOSTED),
        ('https://logfire.example.com/custom/mcp', 'https://logfire.example.com/custom/mcp'),
    ]:
        assert source.apply(rows['url'], url) == 'Saved Which Logfire.'
        assert source.settings.url == saved
    assert source.current(rows['read_only']) == 'false'
    assert source.reset(rows['read_only']) == 'Reset Tools.'
    assert source.current(rows['read_only']) == 'true'


@pytest.mark.parametrize(
    'settings',
    [
        {'key': 'inline-secret'},
        {'key': {'name': 'X', 'value': 'secret'}},
        {'auth': 'secret'},
        {'region': 'eu'},
        {'url': 'http://logfire-us.pydantic.dev/mcp'},
    ],
)
async def test_settings_cannot_hold_a_secret_or_invalid_options(tmp_path: Path, settings: dict[str, JsonValue]) -> None:
    shell = Shell(tmp_path, settings)
    with pytest.raises(PluginError):
        await shell.loader.enable('logfire_mcp')
    assert shell.loader.capabilities() == []


async def test_environment_key_wins_over_the_conventional_saved_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('LOGFIRE_API_KEY', 'env-key')
    api_keys.save_key(name='LOGFIRE_API_KEY', value='saved')
    shell = Shell(tmp_path)
    await shell.loader.enable('logfire_mcp')
    assert shell.capability() == LogfireMCP[None](defer_loading=True, read_only=True)


async def test_conventional_saved_key_beats_browser_sign_in_and_resolves_each_run(
    tmp_path: Path,
) -> None:
    api_keys.save_key(name='LOGFIRE_API_KEY', value='first')
    shell = Shell(tmp_path)
    await shell.loader.enable('logfire_mcp')
    capability = shell.capability()
    assert capability == keyed('LOGFIRE_API_KEY')
    assert isinstance(capability.auth, SavedKey)
    api_keys.save_key(name='LOGFIRE_API_KEY', value='second')
    assert capability.auth(None) == 'second'
    api_keys.delete_key(name='LOGFIRE_API_KEY')
    with pytest.raises(UserError, match=r'LOGFIRE_API_KEY is missing\. Run /plugins configure logfire_mcp'):
        capability.auth(None)


def stub_connections(monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
    """Serve `query_run` in-process, recording the token each run connects with."""
    connected: list[str | None] = []

    def connect(self: LogfireMCP[None], auth: str | None) -> FunctionToolset[None]:
        connected.append(auth)
        toolset = FunctionToolset[None]()

        @toolset.tool_plain
        def query_run(query: str) -> str:
            return 'rows'

        return toolset

    monkeypatch.setattr(LogfireMCP, '_connect', connect)
    return connected


def load_then_query(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Load `logfire-mcp`, whose tools stay hidden until then, query with it, and answer with the rows."""
    names = {tool.name for tool in info.function_tools}
    returns = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
    if not returns:
        assert 'logfire_query_run' not in names
        return ModelResponse(parts=[ToolCallPart('load_capability', {'id': 'logfire-mcp'})])
    if returns[0].tool_name == 'load_capability':
        assert 'logfire_query_run' in names
        return ModelResponse(parts=[ToolCallPart('logfire_query_run', {'query': 'select 1'})])
    return ModelResponse(parts=[TextPart(str(returns[0].content))])


async def stream_load_then_query(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
    """`load_then_query`, streamed as a session requests it."""
    [part] = load_then_query(messages, info).parts
    if isinstance(part, ToolCallPart):
        yield {0: DeltaToolCall(name=part.tool_name, json_args=part.args_as_json_str())}
    else:
        assert isinstance(part, TextPart)
        yield part.content


async def test_a_chosen_saved_key_gives_an_agent_run_the_logfire_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    connected = stub_connections(monkeypatch)
    api_keys.save_key(name='SHARED', value='shared-secret')
    shell = Shell(tmp_path, {'key': {'name': 'SHARED'}, 'read_only': False})
    await shell.loader.enable('logfire_mcp')
    agent = Agent(FunctionModel(load_then_query), deps_type=type(None), capabilities=shell.loader.capabilities())
    assert (await agent.run('Query Logfire.')).output == 'rows'
    assert connected == ['shared-secret']
    api_keys.delete_key(name='SHARED')
    with pytest.raises(UserError, match=r'Saved API key SHARED is missing\. Run /plugins configure logfire_mcp'):
        await agent.run('Query Logfire.')


async def test_an_environment_key_gives_an_agent_run_the_logfire_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    connected = stub_connections(monkeypatch)
    monkeypatch.setenv('LOGFIRE_API_KEY', 'env-key')
    shell = Shell(tmp_path, {'read_only': False})
    await shell.loader.enable('logfire_mcp')
    agent = Agent(FunctionModel(load_then_query), deps_type=type(None), capabilities=shell.loader.capabilities())
    assert (await agent.run('Query Logfire.')).output == 'rows'
    # `None` lets `LogfireMCP` read the environment itself.
    assert connected == [None]


async def test_a_session_with_logfire_and_skills_reads_the_skills_from_the_launch_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: `logfire_mcp`'s per-run capability left `Skills` without a workspace in `for_run`.
    stub_connections(monkeypatch)
    monkeypatch.setenv('LOGFIRE_API_KEY', 'env-key')
    monkeypatch.chdir(tmp_path)
    skill = tmp_path / '.agents' / 'skills' / 'review'
    skill.mkdir(parents=True)
    (skill / 'SKILL.md').write_text('---\ndescription: Review the change.\n---\n\nReview it.\n', encoding='utf-8')
    shell = Shell(tmp_path, {'read_only': False})
    await shell.loader.enable('logfire_mcp')
    model = FunctionModel(stream_function=stream_load_then_query)
    plugins = [Coder[None](repo_context=False, sub_agents=False), Skills[None]('.agents/skills')]
    session = Session(Agent(model, deps_type=type(None)), deps=None, plugins=[*plugins, *shell.loader.capabilities()])
    result = await session.prompt('Query Logfire.')
    assert result.output == 'rows'
    loads = [
        part
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, LoadCapabilityReturnPart)
    ]
    assert len(loads) == 1


async def test_browser_sign_in_is_the_default_and_runs_get_no_tools_until_signed_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shell = Shell(tmp_path, {'url': LOGFIRE_EU_MCP_URL})
    with anyio.fail_after(5):
        await shell.loader.enable('logfire_mcp')
    assert shell.output.getvalue() == 'Not signed in to Logfire. Run /logfire login to sign in.\n'
    client = shell.capability().client
    assert isinstance(client, Client)
    assert str(client.transport.url) == LOGFIRE_EU_MCP_URL
    assert isinstance(client.transport.auth, DeviceAuth)
    assert await shell.run_capability() is None

    def signed_in(self: LogfireSignIn) -> bool:
        return True

    monkeypatch.setattr(LogfireSignIn, 'signed_in', signed_in)
    assert await shell.run_capability() is shell.capability(), 'signing in applies to the next prompt'


async def test_with_sign_in_off_and_no_key_the_menu_still_loads_and_runs_fail_closed(tmp_path: Path) -> None:
    shell = Shell(tmp_path, {'oauth': False})
    await shell.loader.enable('logfire_mcp')
    assert f'Logfire has no credential: LOGFIRE_API_KEY is not in /keys. {SETUP}' in shell.output.getvalue()
    capability = shell.capability()
    auth = capability.auth
    assert auth == SavedKey(name='LOGFIRE_API_KEY', setup=SETUP)
    assert isinstance(auth, SavedKey)
    with pytest.raises(UserError, match='LOGFIRE_API_KEY is missing'):
        auth(None)


def logfire_command(
    settings: dict[str, JsonValue] | None = None,
) -> tuple[Callable[[list[str]], Awaitable[str]], io.StringIO]:
    output = io.StringIO()
    host = PluginHost[None](name='logfire_mcp', console=Console(file=output, width=200), settings=settings or {})
    command, old_name = load_plugin(LogfireMCPPlugin, host).commands
    assert (command.name, old_name.name, old_name.handler) == ('logfire', 'logfire_mcp', command.handler)

    async def run(args: list[str]) -> str:
        result = command.handler(args)
        return result if isinstance(result, str) else await result

    return run, output


def waiting(monkeypatch: pytest.MonkeyPatch, *screens: MenuResult) -> None:
    """Answer `/logfire login`'s waiting screens in the terminal."""
    monkeypatch.setattr(
        'pydantic_clai2.plugins.sign_in.RUNNERS', Script(lists=[], choices=list(screens), texts=[]).runners
    )


async def test_login_signs_in_now_behind_the_waiting_screen(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, bool]] = []

    async def sign_in(method: LogfireSignIn, *, show: Callable[[str], object]) -> Tokens:
        calls.append((method.resource, method.read_only))
        return Tokens(client_id='c', token_endpoint='https://t', access_token='a', expires_at=0)

    monkeypatch.setattr(LogfireSignIn, 'sign_in', sign_in)
    waiting(monkeypatch, UNTIL_CLOSED)
    command, output = logfire_command({'url': LOGFIRE_EU_MCP_URL, 'read_only': False})
    assert await command(['login']) == 'Signed in to Logfire.'
    assert calls == [(LOGFIRE_EU_MCP_URL, False)]
    assert output.getvalue() == ''
    assert await command([]) == 'Not signed in to Logfire. Run /logfire login to sign in.'


async def test_login_failures_and_cancellation_are_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    async def denied(method: LogfireSignIn, *, show: Callable[[str], object]) -> Tokens:
        raise SignInError('Logfire sign-in was denied. Run /logfire login to retry.')

    async def waits(method: LogfireSignIn, *, show: Callable[[str], object]) -> Tokens:
        await anyio.sleep_forever()
        raise AssertionError  # pragma: no cover -- cancelled first

    monkeypatch.setattr(LogfireSignIn, 'sign_in', denied)
    waiting(monkeypatch, UNTIL_CLOSED, CLOSE)
    command, _ = logfire_command()
    assert await command(['login']) == (
        'Could not sign in to Logfire: Logfire sign-in was denied. Run /logfire login to retry. '
        'Run /logfire login to try again.'
    )
    monkeypatch.setattr(LogfireSignIn, 'sign_in', waits)
    with anyio.fail_after(5):
        assert await command(['login']) == 'Logfire sign-in cancelled. Run /logfire login to try again.'


async def test_logout_forgets_only_the_sign_in(monkeypatch: pytest.MonkeyPatch) -> None:
    forgotten: list[bool] = [True, False]
    monkeypatch.setattr('pydantic_clai2.builtin_plugins.logfire_mcp.forget', lambda: forgotten.pop(0))
    api_keys.save_key(name='LOGFIRE_API_KEY', value='kept')
    command, _ = logfire_command()
    assert await command(['logout']) == 'Forgot the Logfire browser sign-in. Keys in /keys are kept.'
    assert await command(['logout']) == 'There was no Logfire browser sign-in to forget.'
    assert 'LOGFIRE_API_KEY' in api_keys.load_keys()
    with pytest.raises(ValueError, match=r'Usage: /logfire \[login \| logout \| status\]'):
        await command(['key'])


@pytest.mark.parametrize('error', [KeyringError('locked'), OSError('read-only')], ids=['keyring', 'file'])
async def test_logout_failures_are_reported(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    def refuse(*, account: str) -> None:
        raise error

    monkeypatch.setattr('pydantic_clai2.logfire_oauth.delete_credentials', refuse)
    command, _ = logfire_command()
    with pytest.raises(
        ValueError, match=r'Could not delete the saved Logfire browser sign-in \((KeyringError|OSError)\)'
    ):
        await command(['logout'])


async def test_registers_the_logout_command(tmp_path: Path) -> None:
    api_keys.save_key(name='LOGFIRE_API_KEY', value='kept')
    shell = Shell(tmp_path)
    await shell.loader.enable('logfire_mcp')
    [loaded] = [entry.loaded for entry in shell.loader.entries() if entry.loaded]
    assert [c.name for c in loaded.commands] == ['logfire', 'logfire_mcp']
    assert [c.complete([]) for c in loaded.commands] == [('login', 'logout', 'status')] * 2


async def test_add_replacing_the_builtin_opens_the_menu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    api_keys.save_key(name='LOGFIRE_API_KEY', value='saved')
    shell = Shell(tmp_path)
    script(monkeypatch, lists=[pick('read_only')], choices=[pick('true')])
    added = await shell.loader.command(
        ['add', 'logfire_mcp', 'pydantic_clai2.builtin_plugins.logfire_mcp', '{"read_only": false}']
    )
    assert added == 'Replaced built-in logfire_mcp.\nSaved Tools.'
    assert shell.capability() == keyed('LOGFIRE_API_KEY')


async def test_configure_needs_a_loaded_plugin_with_a_menu(tmp_path: Path) -> None:
    shell = Shell(tmp_path)
    with pytest.raises(ValueError, match='not loaded; enable it before configuring'):
        await shell.loader.command(['configure', 'logfire_mcp'])
    shell.store.save_plugin(
        BUILTIN.model_copy(update={'id': 'plain', 'factory': 'pydantic_clai2.builtin_plugins.repo_context'})
    )
    assert await shell.loader.command(['enable', 'plain']) == 'Enabled plain.'
    with pytest.raises(ValueError, match='no settings menu'):
        await shell.loader.configure('plain')


async def test_enabling_in_the_plugins_menu_opens_the_settings_menu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api_keys.save_key(name='LOGFIRE_API_KEY', value='saved')
    shell = Shell(tmp_path)
    script(monkeypatch, lists=[pick('read_only')], choices=[pick('false')])

    def run(menu: PluginMenu[None]) -> MenuResult:
        item, save_and_close = menu.items()
        if menu.notice is not None:  # Back from the settings menu.
            return menu.close(Redraw(), save_and_close)
        opened = menu.toggle(Redraw(), item)
        assert opened is not None
        return opened

    assert await open_plugins_menu(shell.loader, run=run) == 'Saved Tools.'
    assert shell.capability().read_only is False


async def test_plugins_menu_stays_open_when_there_is_nothing_to_configure(tmp_path: Path) -> None:
    shell = Shell(tmp_path)
    shell.store.save_plugin(
        BUILTIN.model_copy(
            update={'id': 'plain', 'factory': 'pydantic_clai2.builtin_plugins.repo_context', 'enabled': True}
        )
    )
    await shell.loader.load_all()

    def run(menu: PluginMenu[None]) -> MenuResult:
        assert menu.configure(Redraw(), MenuItem('none', value=None)) is None
        rows = [item for item in menu.items() if not is_save_and_close(item)]
        logfire_row, plain_row = sorted(rows, key=lambda item: str(item.value))
        assert menu.configure(Redraw(), logfire_row) is None
        assert menu.notice == 'Enable logfire_mcp to configure it.'
        assert menu.configure(Redraw(), plain_row) is None
        assert menu.notice == 'plain has no settings menu.'
        return menu.close(Redraw(), plain_row)

    assert await open_plugins_menu(shell.loader, run=run) == ''


class Redraw:
    def replace_items(self, items: object) -> None:
        pass


async def test_keys_are_read_at_session_start_off_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    threads: list[int] = []

    def load_keys() -> dict[str, SecretStr]:
        threads.append(threading.get_ident())
        return {}

    monkeypatch.setattr('pydantic_clai2.builtin_plugins.logfire_mcp.load_keys', load_keys)
    host = PluginHost[None](name='logfire_mcp', console=Console(file=io.StringIO()), settings={})
    plugin = load_plugin(LogfireMCPPlugin, host)
    assert isinstance(plugin.plugin, LogfireMCPPlugin) and plugin.plugin.capability is None
    assert threads == []
    shell = Shell(tmp_path)
    await shell.loader.enable('logfire_mcp')
    assert len(threads) == 1
    assert threads[0] != threading.get_ident()
    assert isinstance(shell.capability().client, Client)


async def test_the_settings_menu_reads_keys_and_sign_in_off_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shell = Shell(tmp_path)
    await shell.loader.enable('logfire_mcp')
    threads: list[int] = []

    def status(*, resource: str, read_only: bool) -> str:
        threads.append(threading.get_ident())
        return 'signed out'

    monkeypatch.setattr('pydantic_clai2.builtin_plugins.logfire_mcp.status', status)
    script(monkeypatch, lists=[])
    assert await shell.loader.command(['configure', 'logfire_mcp']) == 'Logfire: nothing changed.'
    assert threads
    assert threading.get_ident() not in threads


def test_regions_serve_the_harness_mcp_urls() -> None:
    """`logfire_destination` spells the regions out so startup does not import the MCP client; they must not drift."""
    assert [region.mcp_url for region in REGIONS.values()] == [LOGFIRE_US_MCP_URL, LOGFIRE_EU_MCP_URL]


async def test_a_new_setup_starts_from_the_logfire_last_set_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pick the region once: observability setup, or this menu, remembers it for the other."""
    remember(REGIONS['Logfire EU'])
    shell = Shell(tmp_path)
    await shell.loader.enable('logfire_mcp')
    script(monkeypatch, lists=[])
    assert await shell.loader.configure('logfire_mcp') == (
        'Logfire tools connect to Logfire EU, the Logfire you last set up. Change it under Which Logfire.'
    )
    assert shell.saved()['url'] == LOGFIRE_EU_MCP_URL
    # With no key, browser sign-in connects: the URL is the client's, as `LogfireMCP` takes no `url` with a client.
    client = shell.capability().client
    assert isinstance(client, Client)
    assert str(client.transport.url) == LOGFIRE_EU_MCP_URL
    # Once chosen here, even as the default, it stays: a later setup elsewhere does not move it.
    remember(REGIONS['Logfire US'])
    script(monkeypatch, lists=[])
    assert await shell.loader.configure('logfire_mcp') == 'Logfire: nothing changed.'
    assert shell.saved()['url'] == LOGFIRE_EU_MCP_URL


async def test_without_a_terminal_the_menu_neither_opens_nor_switches_logfire(tmp_path: Path) -> None:
    remember(REGIONS['Logfire EU'])
    shell = Shell(tmp_path, terminal=False)
    assert await shell.loader.command(['enable', 'logfire_mcp']) == (
        'Enabled logfire_mcp.\nConfigure Logfire from a terminal: /plugins configure logfire_mcp'
    )
    assert shell.saved() == {}


async def test_the_same_logfire_remembered_changes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    remember(REGIONS['Logfire US'])
    shell = Shell(tmp_path)
    await shell.loader.enable('logfire_mcp')
    script(monkeypatch, lists=[])
    assert await shell.loader.configure('logfire_mcp') == 'Logfire: nothing changed.'


async def test_any_saved_setting_keeps_the_logfire_it_has(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings given as JSON without `url` chose the default on purpose, so a remembered region does not move it."""
    remember(REGIONS['Logfire EU'])
    shell = Shell(tmp_path, {'oauth': False})
    await shell.loader.enable('logfire_mcp')
    script(monkeypatch, lists=[])
    assert await shell.loader.configure('logfire_mcp') == 'Logfire: nothing changed.'
    assert shell.capability().url == LOGFIRE_US_MCP_URL


async def test_resetting_a_seeded_logfire_starts_the_picker_from_the_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remember(REGIONS['Logfire EU'])
    shell = Shell(tmp_path)
    await shell.loader.enable('logfire_mcp')
    host = PluginHost[None](name='logfire_mcp', console=Console(file=io.StringIO()), settings={})
    # Its rows read the keyring, so the menu is built off the event loop, as `configure` builds it.
    menu = await anyio.to_thread.run_sync(FieldMenu, LogfireMCPSource(host))
    reset = menu.reset_marker(None, MenuItem('Which Logfire', value='url'))
    highlighted = watch_highlights(monkeypatch, script(monkeypatch, lists=[reset, pick('url')], choices=[CLOSE]))
    assert await shell.loader.configure('logfire_mcp') == (
        'Logfire tools connect to Logfire EU, the Logfire you last set up. Change it under Which Logfire.\n'
        'Reset Which Logfire.'
    )
    assert highlighted == ['Logfire US']
    assert shell.saved()['url'] == LOGFIRE_US_MCP_URL


@pytest.mark.parametrize(
    ('url', 'current'), [(LOGFIRE_US_MCP_URL, 'Logfire US'), ('https://logfire.example.com/custom/mcp', None)]
)
async def test_the_logfire_picker_starts_from_the_saved_one_and_esc_keeps_it(
    tmp_path: Path, url: str, current: str | None
) -> None:
    shell = Shell(tmp_path, {'url': url, 'oauth': False})
    await shell.loader.enable('logfire_mcp')
    messages, [asked] = await shell.act('url', [Back()])
    assert messages == []
    # A URL at another path, saved by an earlier build, is not one the picker can name, so none is current.
    assert (asked.title, asked.current) == ('Which Logfire?', current)
    assert [(option.value, option.label) for option in asked.options] == [
        ('Logfire US', 'Logfire US'),
        ('Logfire EU', 'Logfire EU'),
        (OTHER, 'Another Logfire...'),
    ]
    assert asked.options[1].description == (
        'UI and API  https://logfire-eu.pydantic.dev\nMCP server  https://logfire-eu.pydantic.dev/mcp'
    )
    assert 'connect to the\nchosen one' in asked.description
    assert shell.saved()['url'] == url


async def test_another_logfire_is_typed_and_esc_goes_back_to_the_list(tmp_path: Path) -> None:
    shell = Shell(tmp_path, {'oauth': False})
    await shell.loader.enable('logfire_mcp')
    messages, asked = await shell.act('url', [Pick(value=OTHER), Back(), Pick(value=OTHER), Type(text=''), Back()])
    assert messages == []
    assert [(a.kind, a.current) for a in asked] == [
        ('choose', 'Logfire US'),
        ('text', None),
        ('choose', OTHER),
        ('text', None),
        ('choose', OTHER),
    ]
    assert (asked[1].title, asked[1].label, asked[1].initial) == ('Another Logfire', 'Address', '')
    messages, _ = await shell.act('url', [Pick(value=OTHER), Type(text='https://logfire.example.com')])
    assert messages == [f'Logfire tools connect to logfire.example.com at {SELF_HOSTED}.']
    assert shell.capability().url == SELF_HOSTED
    # Typed before, so the picker starts on it, with its host to edit.
    _, asked = await shell.act('url', [Pick(value=OTHER), Back(), Back()])
    assert (asked[0].current, asked[1].initial) == (OTHER, 'logfire.example.com')


async def test_another_remembered_logfire_is_only_offered_never_connected_to_unasked(tmp_path: Path) -> None:
    """A key this plugin already uses must not reach a server nobody picked for it, so only regions are switched to."""
    remember(Destination(base_url='https://logfire.example.com'))
    api_keys.save_key(name='LOGFIRE_API_KEY', value='us-key')
    shell = Shell(tmp_path)
    await shell.loader.enable('logfire_mcp')
    _, [offered] = await shell.act('url', [Back()])
    assert offered.current == OTHER
    messages, _ = await shell.act('url', [Pick(value='Logfire EU')])
    assert messages == ['Logfire tools connect to Logfire EU at https://logfire-eu.pydantic.dev/mcp.']
    # Once Logfire EU is picked, reopening starts from it.
    _, [reopened] = await shell.act('url', [Back()])
    assert reopened.current == 'Logfire EU'
    assert shell.capability() == LogfireMCP[None](
        defer_loading=True, auth=SavedKey(name='LOGFIRE_API_KEY', setup=SETUP), url=LOGFIRE_EU_MCP_URL, read_only=True
    )


async def test_picking_the_saved_logfire_again_keeps_its_url_and_sign_in(tmp_path: Path) -> None:
    """Browser sign-ins are stored under the exact MCP URL, so one an earlier build saved is not rewritten."""
    saved = 'https://logfire-eu.pydantic.dev/mcp/'
    shell = Shell(tmp_path, {'url': saved, 'oauth': False})
    await shell.loader.enable('logfire_mcp')
    messages, _ = await shell.act('url', [Pick(value='Logfire EU')])
    assert messages == [f'Logfire tools connect to Logfire EU at {saved}.']
    assert shell.saved()['url'] == saved
    assert remembered() == REGIONS['Logfire EU']
