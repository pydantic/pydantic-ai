"""The `observability` plugin's setup: destination, Logfire's device sign-in, project, and a saved write token."""

import io
import json
import re
import webbrowser
from collections.abc import Awaitable, Callable, Generator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest
from rich.console import Console
from termflow.tui.menu import MenuResult

from pydantic_clai2.builtin_plugins import logfire as logfire_plugin, logfire_setup
from pydantic_clai2.builtin_plugins.logfire import PROJECT, LogfireAccount, LogfireSettings
from pydantic_clai2.builtin_plugins.logfire_destination import (
    OTHER,
    REGIONS,
    Destination,
    destination_problem,
    logfire_dir,
    parse_destination,
    remember,
    remembered,
)
from pydantic_clai2.builtin_plugins.logfire_setup import SIGNING_IN, Setup, SetupError
from pydantic_clai2.config.api_keys import KeyExistsError, KeyReference, load_keys, save_key
from pydantic_clai2.plugins import PluginHost, SessionEnd, load_plugin
from pydantic_clai2.plugins.menus import Option, SettingsMenu
from pydantic_clai2.ui.menus import dialogs as terminal_dialogs
from tests.clai2.dialog_script import Answer as DialogAnswer, Asked, Back, Finish, Pick, ScriptedDialogs, Type
from tests.clai2.menu_script import UNTIL_CLOSED, Script, pick
from tests.clai2.test_logfire import Recorder, observability_loader

US = REGIONS['Logfire US']
EU = REGIONS['Logfire EU']
PROJECTS = [
    {'organization_name': 'pydantic', 'project_name': 'clai2'},
    {'organization_name': 'mike', 'project_name': 'my-stuff'},
]
SIGN_IN_LINK = 'Sign in to Logfire (new users can sign up there):\nhttps://logfire-us.pydantic.dev/auth/dev-123'
SET_UP = [Pick(value='Logfire US'), Finish(), Pick(value='0')]
"""Logfire US, sign in, and the first project."""
Answer = httpx.Response | Callable[[httpx.Request], httpx.Response]


@dataclass
class FakeLogfire:
    """Logfire's device sign-in and project API, answering from scripted responses."""

    polls: list[Answer] = field(
        default_factory=lambda: [httpx.Response(200, json=None), httpx.Response(200, json={'token': 'user-token'})]
    )
    device: Answer = field(
        default_factory=lambda: httpx.Response(
            200, json={'device_code': 'dev-123', 'frontend_auth_url': 'https://logfire-us.pydantic.dev/auth/dev-123'}
        )
    )
    account: Answer = field(
        default_factory=lambda: httpx.Response(200, json={'id': 'u-1', 'name': 'Mike', 'email': 'mike@example.com'})
    )
    projects: Answer = field(default_factory=lambda: httpx.Response(200, json=PROJECTS))
    write_token: Answer = field(default_factory=lambda: httpx.Response(200, json={'token': 'pylf_v1_us_write'}))
    requests: list[httpx.Request] = field(default_factory=list[httpx.Request])

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == '/v1/device-auth/new/':
            answer = self.device
        elif path == '/v1/device-auth/wait/dev-123':
            answer = self.polls.pop(0)
        elif path == '/v1/account/me':
            answer = self.account
        elif path == '/v1/writable-projects/':
            answer = self.projects
        else:
            assert path == '/v1/organizations/pydantic/projects/clai2/write-tokens/'
            answer = self.write_token
        return answer if isinstance(answer, httpx.Response) else answer(request)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handle))


def refuse(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError('refused', request=request)


@dataclass
class Harness:
    server: FakeLogfire = field(default_factory=FakeLogfire)
    opened: list[str] = field(default_factory=list[str])
    browser: bool | None = True
    """What opening the browser does: `True` opens, `False` finds none, `None` raises."""

    def setup(self) -> Setup:
        async def no_wait(seconds: float) -> None:
            pass

        def open_browser(url: str) -> bool:
            self.opened.append(url)
            if self.browser is None:
                raise webbrowser.Error('no runnable browser')
            return self.browser

        return Setup(http=self.server.client, open_browser=open_browser, sleep=no_wait)


def make_host(*, terminal: bool = False, **settings: object) -> PluginHost[None]:
    console = Console(file=io.StringIO(), force_terminal=terminal, color_system=None)
    return PluginHost(name='observability', console=console, settings=json.loads(json.dumps(settings)))


@dataclass
class Ran:
    """What the project row's action returned and asked."""

    messages: list[str]
    asked: list[Asked]

    def shown(self) -> list[str]:
        """What the sign-in's waiting screen showed."""
        [waited] = [asked for asked in self.asked if asked.kind == 'wait']
        return waited.shown


SetUp = Callable[[PluginHost[None], Harness, Sequence[DialogAnswer]], Awaitable[Ran]]


@pytest.fixture
def recorder() -> Generator[Recorder]:
    recorded = Recorder()
    try:
        yield recorded
    finally:
        for instance in recorded.instances:
            instance.shutdown(timeout_millis=3000)


@pytest.fixture
def set_up(monkeypatch: pytest.MonkeyPatch, recorder: Recorder) -> SetUp:
    """Run the settings menu's project row as the web UI does, answering its dialogs, with `harness` in place."""

    # The recorder keeps the SDK local: a saved token would otherwise make it check the token over the network.
    monkeypatch.setattr(logfire_plugin.logfire, 'configure', recorder.configure)

    async def run(host: PluginHost[None], harness: Harness, answers: Sequence[DialogAnswer]) -> Ran:
        monkeypatch.setattr(logfire_plugin, 'SETUP', harness.setup())
        plugin = load_plugin(logfire_plugin.LogfirePlugin, host)
        scripted = ScriptedDialogs(answers)
        try:
            menu = plugin.plugin.settings_menu()
            assert menu is not None
            messages = list(await menu.actions[PROJECT](scripted))
        finally:
            await plugin.dispatch(SessionEnd(reason='exit'))
        assert scripted.unused == []
        return Ran(messages=messages, asked=scripted.asked)

    return run


async def test_sign_in_pick_a_project_and_save_its_write_token(set_up: SetUp) -> None:
    harness = Harness()
    host = make_host(service_name='mine', ui_events=True, send_to_logfire=False)
    ran = await set_up(host, harness, SET_UP)
    assert ran.messages == [
        'Logfire traces now go to pydantic/clai2. Its write token is saved in /keys as '
        'LOGFIRE_TOKEN_PYDANTIC_CLAI2; plugin settings keep only that name and the email you signed in with.'
    ]
    destination, waiting, project = ran.asked
    assert (destination.kind, destination.title, destination.current) == ('choose', 'Which Logfire?', None)
    assert [option.value for option in destination.options] == ['Logfire US', 'Logfire EU', OTHER]
    assert destination.description.startswith('Then sign in (or sign up) in your browser')
    assert (waiting.kind, waiting.title, waiting.description) == ('wait', 'Sign in to Logfire', SIGNING_IN)
    assert waiting.shown == [SIGN_IN_LINK, 'Signed in. Listing your projects...']
    assert (project.title, project.options) == (
        'Send traces to which project?',
        (Option(value='0', label='pydantic/clai2'), Option(value='1', label='mike/my-stuff')),
    )
    assert project.description == 'CLAI creates a write token for this project and saves it in /keys.'
    assert load_keys()['LOGFIRE_TOKEN_PYDANTIC_CLAI2'].get_secret_value() == 'pylf_v1_us_write'
    saved = host.settings(LogfireSettings)
    assert saved.token == KeyReference(name='LOGFIRE_TOKEN_PYDANTIC_CLAI2')
    # Saved even for a hosted region, so `LOGFIRE_BASE_URL` cannot send this token elsewhere.
    assert saved.base_url == 'https://logfire-us.pydantic.dev'
    assert remembered() == US, 'the Logfire (MCP) plugin starts from the same Logfire'
    assert (saved.service_name, saved.ui_events) == ('mine', True)  # Other settings are kept.
    assert saved.send_to_logfire == 'if-token-present'  # Setting up a project turns sending on.
    assert saved.account == LogfireAccount(
        email='mike@example.com', token=KeyReference(name='LOGFIRE_TOKEN_PYDANTIC_CLAI2')
    )
    assert harness.opened == ['https://logfire-us.pydantic.dev/auth/dev-123']
    new, *_, me, listed, minted = harness.server.requests
    assert new.url.params['machine_name']
    assert me.url.path == '/v1/account/me'
    assert me.headers['Authorization'] == listed.headers['Authorization'] == minted.headers['Authorization']
    assert minted.headers['Authorization'] == 'user-token'
    assert str(minted.url).startswith('https://logfire-us.pydantic.dev/')


async def test_the_terminal_menu_runs_setup_from_the_project_row(
    monkeypatch: pytest.MonkeyPatch, recorder: Recorder
) -> None:
    """`/plugins configure observability` in a terminal: the row, then each dialog as a termflow widget."""
    monkeypatch.setattr(logfire_plugin.logfire, 'configure', recorder.configure)
    monkeypatch.setattr(logfire_plugin, 'SETUP', Harness().setup())
    shown = Script(
        lists=[pick(PROJECT), MenuResult(cancelled=True)],
        # Which Logfire, the sign-in's waiting screen until it closes itself, and the first project.
        choices=[pick('Logfire US'), UNTIL_CLOSED, pick('0')],
        texts=[],
    )
    monkeypatch.setattr(terminal_dialogs, 'RUNNERS', shown.runners)
    host = make_host(terminal=True)
    plugin = load_plugin(logfire_plugin.LogfirePlugin, host)
    try:
        assert plugin.plugin.has_configure
        message = await plugin.plugin.configure()
    finally:
        await plugin.dispatch(SessionEnd(reason='exit'))
    assert message.startswith('Logfire traces now go to pydantic/clai2.')
    assert shown.opened == ['list', 'choice', 'choice', 'choice', 'list']
    assert host.settings(LogfireSettings).token == KeyReference(name='LOGFIRE_TOKEN_PYDANTIC_CLAI2')


async def test_setup_from_the_web_ui_loads_the_plugin_again_with_the_new_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorder: Recorder
) -> None:
    """The web UI runs the project row's action through `edit_settings`, with dialogs of its own."""
    monkeypatch.setattr(logfire_plugin.logfire, 'configure', recorder.configure)
    monkeypatch.setattr(logfire_plugin, 'SETUP', Harness().setup())
    loader, store = observability_loader(tmp_path)
    scripted = ScriptedDialogs(SET_UP)

    async def edit(menu: SettingsMenu) -> Sequence[str]:
        return await menu.actions[PROJECT](scripted)

    try:
        await loader.load_all()
        [message] = await loader.edit_settings('observability', edit)
        assert message.startswith('Logfire traces now go to pydantic/clai2.')
        assert recorder.tokens == [None, 'pylf_v1_us_write']
        [declaration] = store.plugins()
        assert declaration.settings['token'] == {'name': 'LOGFIRE_TOKEN_PYDANTIC_CLAI2'}
    finally:
        await loader.close('exit')


async def test_the_terminal_menu_closed_unchanged_says_so(monkeypatch: pytest.MonkeyPatch, recorder: Recorder) -> None:
    monkeypatch.setattr(logfire_plugin.logfire, 'configure', recorder.configure)
    shown = Script(lists=[MenuResult(cancelled=True)], choices=[], texts=[])
    monkeypatch.setattr(terminal_dialogs, 'RUNNERS', shown.runners)
    plugin = load_plugin(logfire_plugin.LogfirePlugin, make_host(terminal=True))
    try:
        assert await plugin.plugin.configure() == 'Observability (Logfire): nothing changed.'
    finally:
        await plugin.dispatch(SessionEnd(reason='exit'))


@pytest.mark.parametrize('typed', ['logfire.example.com/', 'https://logfire.example.com', 'logfire.example.com/mcp'])
async def test_another_logfire_is_typed_as_a_host_or_any_of_its_urls(typed: str, set_up: SetUp) -> None:
    harness = Harness()
    host = make_host()
    ran = await set_up(host, harness, [Pick(value=OTHER), Type(text=typed), Finish(), Pick(value='0')])
    address = ran.asked[1]
    assert (address.title, address.label, address.initial) == ('Another Logfire', 'Address', '')
    assert host.settings(LogfireSettings).base_url == 'https://logfire.example.com'
    assert {request.url.host for request in harness.server.requests} == {'logfire.example.com'}
    assert remembered() == Destination(base_url='https://logfire.example.com')


@pytest.mark.parametrize(
    'answers',
    [
        pytest.param([Back()], id='which-logfire'),
        pytest.param([Pick(value=OTHER), Back(), Back()], id='address-then-list'),
        pytest.param([Pick(value='Logfire US'), Back()], id='sign-in'),
        pytest.param([Pick(value='Logfire US'), Finish(), Back()], id='project'),
    ],
)
async def test_cancelling_changes_nothing(answers: list[DialogAnswer], set_up: SetUp) -> None:
    host = make_host()
    ran = await set_up(host, Harness(), answers)
    assert ran.messages == ['Logfire setup cancelled; settings unchanged.']
    assert host.settings(LogfireSettings) == LogfireSettings()
    assert load_keys() == {}


async def test_the_current_destination_is_preselected(set_up: SetUp) -> None:
    async def first_asked(host: PluginHost[None], answers: list[DialogAnswer]) -> Asked:
        return (await set_up(host, Harness(), answers)).asked[0]

    assert (await first_asked(make_host(), [Back()])).current is None
    eu = make_host(base_url='https://logfire-eu.pydantic.dev')
    assert (await first_asked(eu, [Back()])).current == 'Logfire EU'
    # Another Logfire starts on its row, with its address filled in.
    elsewhere = make_host(base_url='https://elsewhere.example.com')
    ran = await set_up(elsewhere, Harness(), [Pick(value=OTHER), Back(), Back()])
    assert [asked.current for asked in ran.asked if asked.kind == 'choose'] == [OTHER, OTHER]
    assert ran.asked[1].initial == 'elsewhere.example.com'
    # Without a Logfire of its own, setup starts from the one last set up here or in the Logfire (MCP) plugin.
    remember(EU)
    assert (await first_asked(make_host(), [Back()])).current == 'Logfire EU'


@pytest.mark.parametrize(
    ('browser', 'note'),
    [(False, True), (None, True), (True, False)],
)
async def test_a_missing_browser_says_to_open_the_link(browser: bool | None, note: bool, set_up: SetUp) -> None:
    ran = await set_up(make_host(), Harness(browser=browser), [Pick(value='Logfire US'), Finish(), Back()])
    missing = f'{SIGN_IN_LINK}\n\nNo browser opened; open the link yourself, on any device.'
    assert (missing in ran.shown()) == note


async def test_a_sign_in_link_with_terminal_controls_is_shown_inert(set_up: SetUp) -> None:
    hostile = 'https://logfire-us.pydantic.dev/auth/\x1b[2Jdev-123'
    server = FakeLogfire(device=httpx.Response(200, json={'device_code': 'dev-123', 'frontend_auth_url': hostile}))
    # Backing out as soon as the link shows: the screen showed it inert, and nothing was saved.
    ran = await set_up(make_host(), Harness(server=server), [Pick(value='Logfire US'), Back()])
    assert ran.shown() == [
        'Sign in to Logfire (new users can sign up there):\nhttps://logfire-us.pydantic.dev/auth/\\x1b[2Jdev-123'
    ]


@pytest.mark.parametrize(
    ('server', 'error'),
    [
        (FakeLogfire(device=refuse), 'Could not reach Logfire (ConnectError)'),
        (FakeLogfire(device=httpx.Response(503)), 'Logfire refused /v1/device-auth/new/ (HTTP 503)'),
        (FakeLogfire(device=httpx.Response(200, json={'nope': 1})), 'answered /v1/device-auth/new/ with something'),
        (
            FakeLogfire(device=httpx.Response(200, json={'device_code': 'dev-123', 'frontend_auth_url': 'http://x'})),
            'sent a sign-in link that is not https',
        ),
        (FakeLogfire(polls=[refuse] * 4), 'Lost contact with https://logfire-us.pydantic.dev'),
        (FakeLogfire(projects=httpx.Response(200, json={'bad': True})), 'answered the project list with something'),
    ],
)
async def test_sign_in_failures_say_what_went_wrong_and_save_nothing(
    server: FakeLogfire, error: str, set_up: SetUp
) -> None:
    host = make_host()
    with pytest.raises(SetupError, match=re.escape(error)):
        await set_up(host, Harness(server=server), [Pick(value='Logfire US'), Finish()])
    assert host.settings(LogfireSettings) == LogfireSettings()
    assert load_keys() == {}


@pytest.mark.parametrize(
    ('server', 'answers', 'error'),
    [
        (FakeLogfire(projects=httpx.Response(200, json=[])), SET_UP[:2], 'You cannot write to any project'),
        (FakeLogfire(write_token=httpx.Response(403)), SET_UP, 'Logfire refused /v1/organizations/pydantic'),
    ],
)
async def test_project_failures_say_what_went_wrong_and_save_nothing(
    server: FakeLogfire, answers: list[DialogAnswer], error: str, set_up: SetUp
) -> None:
    host = make_host()
    with pytest.raises(SetupError, match=re.escape(error)):
        await set_up(host, Harness(server=server), answers)
    assert host.settings(LogfireSettings) == LogfireSettings()
    assert load_keys() == {}


@pytest.mark.parametrize(
    'account', [refuse, httpx.Response(401), httpx.Response(200, json={'id': 'u-1', 'name': 'Mike'})]
)
async def test_an_unknown_account_email_still_sets_up_the_project(account: Answer, set_up: SetUp) -> None:
    host = make_host()
    ran = await set_up(host, Harness(server=FakeLogfire(account=account)), SET_UP)
    saved = host.settings(LogfireSettings)
    assert saved.token == KeyReference(name='LOGFIRE_TOKEN_PYDANTIC_CLAI2')
    assert saved.account is None
    [message] = ran.messages
    assert message.endswith('plugin settings keep only that name; Logfire did not share your email.')


async def test_polling_survives_blips_and_expires(monkeypatch: pytest.MonkeyPatch, set_up: SetUp) -> None:
    blips = FakeLogfire(
        polls=[refuse, httpx.Response(502), httpx.Response(200), httpx.Response(200, json={'token': 't'})]
    )
    await set_up(make_host(), Harness(server=blips), SET_UP)
    assert not blips.polls
    monkeypatch.setattr(logfire_setup, 'SIGN_IN_TIMEOUT', 0)
    with pytest.raises(SetupError, match='Run /plugins configure observability to retry'):
        await set_up(make_host(), Harness(), SET_UP[:2])


@pytest.mark.parametrize(
    ('typed', 'base_url'),
    [
        ('logfire.example.com', 'https://logfire.example.com'),
        (' https://logfire.example.com/ ', 'https://logfire.example.com'),
        ('https://logfire.example.com:8443', 'https://logfire.example.com:8443'),
        ('https://logfire.example.com/mcp', 'https://logfire.example.com'),
        ('logfire.example.com/mcp/', 'https://logfire.example.com'),
        ('logfire-eu.pydantic.info', 'https://logfire-eu.pydantic.info'),
        ('https://Logfire-EU.pydantic.dev/mcp', 'https://logfire-eu.pydantic.dev'),
        # From before regions: both are the US region, which serves the MCP server and sign-in.
        ('https://logfire.pydantic.dev', 'https://logfire-us.pydantic.dev'),
        ('logfire-api.pydantic.dev', 'https://logfire-us.pydantic.dev'),
    ],
)
def test_a_logfire_is_typed_as_a_host_its_url_or_its_mcp_url(typed: str, base_url: str) -> None:
    destination = parse_destination(typed)
    assert (destination.base_url, destination.mcp_url) == (base_url, f'{base_url}/mcp')
    assert destination_problem(typed) is None


@pytest.mark.parametrize(
    'typed',
    [
        'http://logfire.example.com',
        'https://logfire.example.com/app',
        'https://',
        '',
        'not a url',
        'logfire.example.com:abc',
        'x?y=1',
        'x#f',
    ],
)
def test_other_addresses_say_what_to_type(typed: str) -> None:
    problem = 'Type a host (logfire.example.com), an https URL, or an MCP URL ending in /mcp'
    with pytest.raises(ValueError, match=re.escape(problem)):
        parse_destination(typed)
    assert (destination_problem(typed) or '').startswith(problem)


def test_regions_come_from_the_harness_mcp_urls_and_others_are_named_by_host() -> None:
    assert [(region.base_url, region.mcp_url, region.label) for region in REGIONS.values()] == [
        ('https://logfire-us.pydantic.dev', 'https://logfire-us.pydantic.dev/mcp', 'Logfire US'),
        ('https://logfire-eu.pydantic.dev', 'https://logfire-eu.pydantic.dev/mcp', 'Logfire EU'),
    ]
    other = parse_destination('logfire.example.com:8443')
    assert (other.region, other.label) == (None, 'logfire.example.com:8443')


def test_an_unreadable_or_unwritable_remembered_logfire_is_no_default() -> None:
    assert remembered() is None
    directory = logfire_dir()
    directory.mkdir(parents=True)
    (directory / 'destination.json').write_text('not json')
    assert remembered() is None
    (directory / 'destination.json').unlink()
    directory.rmdir()
    directory.write_text('a file where the folder should be')
    remember(EU)  # Only a default is lost, so setup carries on.
    assert remembered() is None


def test_project_key_names_are_valid_key_names() -> None:
    project = logfire_setup.Project(organization_name='Pydantic Inc.', project_name='clai-2')
    assert project.key_name == 'LOGFIRE_TOKEN_PYDANTIC_INC_CLAI_2'
    assert project.label == 'Pydantic Inc./clai-2'
    hostile = logfire_setup.Project(organization_name='org', project_name='p\x1b]52;c;UE9JU09O\x07\n')
    assert hostile.label == 'org/p\\x1b]52;c;UE9JU09O\\x07\\x0a'


def test_self_hosted_error_text_is_made_inert() -> None:
    error = SetupError('bad \x1b[31m red')
    assert '\x1b' not in str(error)
    assert isinstance(error, ValueError)


async def test_an_unrelated_key_of_the_same_name_is_kept(set_up: SetUp, recorder: Recorder) -> None:
    save_key(name='LOGFIRE_TOKEN_PYDANTIC_CLAI2', value='someone-elses')
    host = make_host()
    await set_up(host, Harness(), SET_UP)
    assert host.settings(LogfireSettings).token == KeyReference(name='LOGFIRE_TOKEN_PYDANTIC_CLAI2_2')
    keys = load_keys()
    assert keys['LOGFIRE_TOKEN_PYDANTIC_CLAI2'].get_secret_value() == 'someone-elses'
    assert keys['LOGFIRE_TOKEN_PYDANTIC_CLAI2_2'].get_secret_value() == 'pylf_v1_us_write'
    # Setting up the same project again replaces the key the plugin owns, instead of adding another.
    renewed = FakeLogfire(write_token=httpx.Response(200, json={'token': 'pylf_v1_us_renewed'}))
    await set_up(host, Harness(server=renewed), SET_UP)
    assert recorder.tokens == [None, 'pylf_v1_us_write']  # The second activation sent with the saved key.
    keys = load_keys()
    assert keys['LOGFIRE_TOKEN_PYDANTIC_CLAI2_2'].get_secret_value() == 'pylf_v1_us_renewed'
    assert 'LOGFIRE_TOKEN_PYDANTIC_CLAI2_3' not in keys


async def test_running_out_of_key_names_says_so(set_up: SetUp, monkeypatch: pytest.MonkeyPatch) -> None:
    def taken(*, name: str, value: str, replace: bool = True) -> str:
        raise KeyExistsError(name)

    monkeypatch.setattr(logfire_setup, 'save_key', taken)
    with pytest.raises(SetupError, match='Too many /keys entries start with LOGFIRE_TOKEN_PYDANTIC_CLAI2'):
        await set_up(make_host(), Harness(), SET_UP)


def test_credentials_in_an_address_are_rejected_without_echoing_them() -> None:
    with pytest.raises(ValueError, match='Leave credentials out of the address') as error:
        parse_destination('https://mike:hunter2@logfire.example.com')
    assert 'hunter2' not in str(error.value)
