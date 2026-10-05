"""The `observability` plugin's setup menu: destination, Logfire's device sign-in, project, and a saved write token."""

import io
import json
import re
import webbrowser
from collections.abc import Awaitable, Callable, Generator
from dataclasses import dataclass, field

import httpx
import pytest
from rich.console import Console
from termflow.tui import MenuItem
from termflow.tui.menu import Menu, MenuResult
from termflow.tui.textinput import TextInput, TextInputResult

from pydantic_clai2.builtin_plugins import logfire as logfire_plugin, logfire_setup
from pydantic_clai2.builtin_plugins.logfire import LogfireAccount, LogfireSettings
from pydantic_clai2.builtin_plugins.logfire_setup import Setup, SetupError, https_origin
from pydantic_clai2.config.api_keys import KeyExistsError, KeyReference, load_keys, save_key
from pydantic_clai2.plugins import PluginHost, SessionEnd, load_plugin
from pydantic_clai2.ui.menus.field_menu import Runners
from tests.clai2.menu_script import Script, pick
from tests.clai2.test_logfire import Recorder

US = 'https://logfire-us.pydantic.dev'
PROJECTS = [
    {'organization_name': 'pydantic', 'project_name': 'clai2'},
    {'organization_name': 'mike', 'project_name': 'my-stuff'},
]
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


def scripted(choices: list[object], typed: list[str | None] | None = None) -> Runners:
    """Choose menu items by value (`None` presses Esc) and type text (`None` presses Esc), in order."""
    texts = list(typed or [])

    def run_choice(menu: Menu) -> MenuResult:
        wanted = choices.pop(0)
        if wanted is None:
            return MenuResult(cancelled=True)
        return MenuResult(item=MenuItem(str(wanted), value=wanted))

    def run_text(widget: TextInput) -> TextInputResult:
        text = texts.pop(0)
        return TextInputResult(cancelled=True) if text is None else TextInputResult(value=text)

    def run_list(menu: Menu) -> MenuResult:
        raise AssertionError('setup has no list menu')  # pragma: no cover

    return Runners(run_list=run_list, run_choice=run_choice, run_text=run_text)


@dataclass
class Harness:
    server: FakeLogfire = field(default_factory=FakeLogfire)
    lines: list[str] = field(default_factory=list[str])
    opened: list[str] = field(default_factory=list[str])
    browser: bool | None = True
    """What opening the browser does: `True` opens, `False` finds none, `None` raises."""

    def setup(self, runners: Runners) -> Setup:
        async def no_wait(seconds: float) -> None:
            pass

        def open_browser(url: str) -> bool:
            self.opened.append(url)
            if self.browser is None:
                raise webbrowser.Error('no runnable browser')
            return self.browser

        return Setup(
            announce=self.lines.append,
            runners=runners,
            http=self.server.client,
            open_browser=open_browser,
            sleep=no_wait,
        )


def make_host(**settings: object) -> PluginHost[None]:
    return PluginHost(
        name='observability', console=Console(file=io.StringIO()), settings=json.loads(json.dumps(settings))
    )


Configure = Callable[[PluginHost[None], Setup], Awaitable[str]]


@pytest.fixture
def recorder() -> Generator[Recorder]:
    recorded = Recorder()
    try:
        yield recorded
    finally:
        for instance in recorded.instances:
            instance.shutdown(timeout_millis=3000)


@pytest.fixture
def configure(monkeypatch: pytest.MonkeyPatch, recorder: Recorder) -> Configure:
    """Open the plugin's real setup menu, as `/plugins configure observability` does, with `setup` in place."""

    # The recorder keeps the SDK local: a saved token would otherwise make it check the token over the network.
    monkeypatch.setattr(logfire_plugin.logfire, 'configure', recorder.configure)

    async def run(host: PluginHost[None], setup: Setup) -> str:
        def scripted_setup(host: PluginHost[None]) -> Setup:
            return setup

        monkeypatch.setattr(logfire_plugin, 'SETUP', scripted_setup)
        # The settings menu: Enter on the project row runs setup, then Esc closes the menu.
        menu = Script(lists=[pick(logfire_plugin.PROJECT), MenuResult(cancelled=True)], choices=[], texts=[])
        monkeypatch.setattr(logfire_plugin, 'RUNNERS', menu.runners)
        plugin = load_plugin(logfire_plugin.LogfirePlugin, host)
        try:
            assert plugin.plugin.has_configure
            return await plugin.plugin.configure()
        finally:
            await plugin.dispatch(SessionEnd(reason='exit'))

    return run


async def test_sign_in_pick_a_project_and_save_its_write_token(configure: Configure) -> None:
    harness = Harness()
    host = make_host(service_name='mine', ui_events=True, send_to_logfire=False)
    message = await configure(host, harness.setup(scripted([US, logfire_setup.Project(**PROJECTS[0])])))
    assert message == (
        'Logfire traces now go to pydantic/clai2. Its write token is saved in /keys as '
        'LOGFIRE_TOKEN_PYDANTIC_CLAI2; plugin settings keep only that name and the email you signed in with.'
    )
    assert load_keys()['LOGFIRE_TOKEN_PYDANTIC_CLAI2'].get_secret_value() == 'pylf_v1_us_write'
    saved = host.settings(LogfireSettings)
    assert saved.token == KeyReference(name='LOGFIRE_TOKEN_PYDANTIC_CLAI2')
    # Saved even for a hosted region, so `LOGFIRE_BASE_URL` cannot send this token elsewhere.
    assert saved.base_url == US
    assert (saved.service_name, saved.ui_events) == ('mine', True)  # Other settings are kept.
    assert saved.send_to_logfire == 'if-token-present'  # Setting up a project turns sending on.
    assert saved.account == LogfireAccount(
        email='mike@example.com', token=KeyReference(name='LOGFIRE_TOKEN_PYDANTIC_CLAI2')
    )
    assert harness.lines == [
        'Sign in to Logfire (new users can sign up there): https://logfire-us.pydantic.dev/auth/dev-123'
    ]
    assert harness.opened == ['https://logfire-us.pydantic.dev/auth/dev-123']
    new, *_, me, listed, minted = harness.server.requests
    assert new.url.params['machine_name']
    assert me.url.path == '/v1/account/me'
    assert me.headers['Authorization'] == listed.headers['Authorization'] == minted.headers['Authorization']
    assert minted.headers['Authorization'] == 'user-token'
    assert str(minted.url).startswith(US)


async def test_self_hosted_logfire_is_remembered(configure: Configure) -> None:
    harness = Harness()
    host = make_host()
    runners = scripted([logfire_setup.SELF_HOSTED, logfire_setup.Project(**PROJECTS[0])], ['logfire.example.com/'])
    await configure(host, harness.setup(runners))
    assert host.settings(LogfireSettings).base_url == 'https://logfire.example.com'
    assert {request.url.host for request in harness.server.requests} == {'logfire.example.com'}


@pytest.mark.parametrize(
    ('choices', 'typed'),
    [([None], []), ([logfire_setup.SELF_HOSTED], [None]), ([logfire_setup.SELF_HOSTED], ['  ']), ([US, None], [])],
)
async def test_cancelling_changes_nothing(choices: list[object], typed: list[str | None], configure: Configure) -> None:
    host = make_host()
    message = await configure(host, Harness().setup(scripted(choices, typed)))
    assert message == 'Logfire setup cancelled; settings unchanged.'
    assert host.settings(LogfireSettings) == LogfireSettings()
    assert load_keys() == {}


async def test_the_current_destination_is_preselected(configure: Configure) -> None:
    seen: list[object] = []

    def run_choice(menu: Menu) -> MenuResult:
        seen.append(menu.highlighted.value if menu.highlighted else None)
        return MenuResult(cancelled=True)

    runners = Runners(run_choice=run_choice)
    host = make_host(base_url='https://logfire-eu.pydantic.dev')
    await configure(host, Harness().setup(runners))
    await configure(make_host(base_url='https://elsewhere.example.com'), Harness().setup(runners))
    assert seen == ['https://logfire-eu.pydantic.dev', US]


@pytest.mark.parametrize(
    ('browser', 'note'),
    [(False, True), (None, True), (True, False)],
)
async def test_a_missing_browser_says_to_open_the_link(browser: bool | None, note: bool, configure: Configure) -> None:
    harness = Harness(browser=browser)
    await configure(make_host(), harness.setup(scripted([US, None])))
    assert ('No browser opened; open the link above yourself, on any device.' in harness.lines) == note


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
        (FakeLogfire(projects=httpx.Response(200, json=[])), 'You cannot write to any project'),
        (FakeLogfire(projects=httpx.Response(200, json={'bad': True})), 'answered the project list with something'),
        (FakeLogfire(write_token=httpx.Response(403)), 'Logfire refused /v1/organizations/pydantic'),
    ],
)
async def test_failures_say_what_went_wrong_and_save_nothing(
    server: FakeLogfire, error: str, configure: Configure
) -> None:
    host = make_host()
    runners = scripted([US, logfire_setup.Project(**PROJECTS[0])])
    with pytest.raises(SetupError, match=re.escape(error)):
        await configure(host, Harness(server=server).setup(runners))
    assert host.settings(LogfireSettings) == LogfireSettings()
    assert load_keys() == {}


@pytest.mark.parametrize(
    'account', [refuse, httpx.Response(401), httpx.Response(200, json={'id': 'u-1', 'name': 'Mike'})]
)
async def test_an_unknown_account_email_still_sets_up_the_project(account: Answer, configure: Configure) -> None:
    host = make_host()
    harness = Harness(server=FakeLogfire(account=account))
    message = await configure(host, harness.setup(scripted([US, logfire_setup.Project(**PROJECTS[0])])))
    saved = host.settings(LogfireSettings)
    assert saved.token == KeyReference(name='LOGFIRE_TOKEN_PYDANTIC_CLAI2')
    assert saved.account is None
    assert message.endswith('plugin settings keep only that name; Logfire did not share your email.')


async def test_polling_survives_blips_and_expires(monkeypatch: pytest.MonkeyPatch, configure: Configure) -> None:
    blips = FakeLogfire(
        polls=[refuse, httpx.Response(502), httpx.Response(200), httpx.Response(200, json={'token': 't'})]
    )
    await configure(make_host(), Harness(server=blips).setup(scripted([US, logfire_setup.Project(**PROJECTS[0])])))
    assert not blips.polls
    monkeypatch.setattr(logfire_setup, 'SIGN_IN_TIMEOUT', 0)
    with pytest.raises(SetupError, match='Run /plugins configure observability to retry'):
        await configure(make_host(), Harness().setup(scripted([US])))


@pytest.mark.parametrize(
    ('typed', 'origin'),
    [
        ('logfire.example.com', 'https://logfire.example.com'),
        (' https://logfire.example.com/ ', 'https://logfire.example.com'),
        ('https://logfire.example.com:8443', 'https://logfire.example.com:8443'),
    ],
)
def test_https_origin_accepts(typed: str, origin: str) -> None:
    assert https_origin(typed) == origin


@pytest.mark.parametrize(
    'typed', ['http://logfire.example.com', 'https://logfire.example.com/app', 'https://', 'x?y=1']
)
def test_https_origin_rejects(typed: str) -> None:
    with pytest.raises(SetupError, match='https URL with no path'):
        https_origin(typed)


def test_project_key_names_are_valid_key_names() -> None:
    project = logfire_setup.Project(organization_name='Pydantic Inc.', project_name='clai-2')
    assert project.key_name == 'LOGFIRE_TOKEN_PYDANTIC_INC_CLAI_2'
    assert project.label == 'Pydantic Inc./clai-2'
    hostile = logfire_setup.Project(organization_name='org', project_name='p\x1b]52;c;UE9JU09O\x07\n')
    assert hostile.label == 'org/p\\x1b]52;c;UE9JU09O\\x07\\x0a'


async def test_self_hosted_error_text_is_made_inert(configure: Configure) -> None:
    error = SetupError('bad \x1b[31m red')
    assert '\x1b' not in str(error)
    assert isinstance(error, ValueError)


def test_the_real_setup_announces_on_the_console_with_controls_made_inert() -> None:
    host = make_host()
    setup = logfire_plugin.SETUP(host)
    setup.announce('Sign in: https://logfire.example.com/\x1b[2Jauth')
    output = host.console.file
    assert isinstance(output, io.StringIO)
    assert output.getvalue() == 'Sign in: https://logfire.example.com/\\x1b[2Jauth\n'


async def test_an_unrelated_key_of_the_same_name_is_kept(configure: Configure, recorder: Recorder) -> None:
    save_key(name='LOGFIRE_TOKEN_PYDANTIC_CLAI2', value='someone-elses')
    project = logfire_setup.Project(**PROJECTS[0])
    host = make_host()
    await configure(host, Harness().setup(scripted([US, project])))
    assert host.settings(LogfireSettings).token == KeyReference(name='LOGFIRE_TOKEN_PYDANTIC_CLAI2_2')
    keys = load_keys()
    assert keys['LOGFIRE_TOKEN_PYDANTIC_CLAI2'].get_secret_value() == 'someone-elses'
    assert keys['LOGFIRE_TOKEN_PYDANTIC_CLAI2_2'].get_secret_value() == 'pylf_v1_us_write'
    # Setting up the same project again replaces the key the plugin owns, instead of adding another.
    renewed = FakeLogfire(write_token=httpx.Response(200, json={'token': 'pylf_v1_us_renewed'}))
    await configure(host, Harness(server=renewed).setup(scripted([US, project])))
    assert recorder.tokens == [None, 'pylf_v1_us_write']  # The second activation sent with the saved key.
    keys = load_keys()
    assert keys['LOGFIRE_TOKEN_PYDANTIC_CLAI2_2'].get_secret_value() == 'pylf_v1_us_renewed'
    assert 'LOGFIRE_TOKEN_PYDANTIC_CLAI2_3' not in keys


async def test_running_out_of_key_names_says_so(configure: Configure, monkeypatch: pytest.MonkeyPatch) -> None:
    def taken(*, name: str, value: str, replace: bool = True) -> str:
        raise KeyExistsError(name)

    monkeypatch.setattr(logfire_setup, 'save_key', taken)
    with pytest.raises(SetupError, match='Too many /keys entries start with LOGFIRE_TOKEN_PYDANTIC_CLAI2'):
        await configure(make_host(), Harness().setup(scripted([US, logfire_setup.Project(**PROJECTS[0])])))


def test_https_origin_rejects_credentials_without_echoing_them() -> None:
    with pytest.raises(SetupError, match='Leave credentials out of the URL') as error:
        https_origin('https://mike:hunter2@logfire.example.com')
    assert 'hunter2' not in str(error.value)
