"""The `observability` plugin's setup menu: pick where traces go, sign in there, pick a project.

It runs Logfire's own device sign-in (the one behind `logfire auth`), not the MCP OAuth in
`pydantic_clai2.logfire_oauth`: MCP tokens are issued for the MCP server alone and cannot mint write
tokens. The user token lives only for the duration of setup. What is kept is a project write token, saved in
`/keys`, and the plugin settings naming it and the signed-in account's email, so the loader reloads the plugin
and traces go to that project, with session roots tagged with that email.
"""

import platform
import re
import time
import webbrowser
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar
from urllib.parse import urlsplit

import anyio
import httpx
from anyio import to_thread
from pydantic import BaseModel, TypeAdapter, ValidationError
from termflow.tui import MenuBuilder, MenuItem, TextInputBuilder

from pydantic_clai2.config.api_keys import KeyExistsError, KeyReference, save_key
from pydantic_clai2.ui import telemetry
from pydantic_clai2.ui.menus.field_menu import TERMINAL, Runners
from pydantic_clai2.ui.menus.menu_worker import menu_key, run_worker
from pydantic_clai2.ui.menus.slash_search import slash_search
from pydantic_clai2.ui.rendering._rendering import markdown_style
from pydantic_clai2.ui.rendering.tool_output import terminal_text

REGIONS = {'Logfire US': 'https://logfire-us.pydantic.dev', 'Logfire EU': 'https://logfire-eu.pydantic.dev'}
"""The hosted regions. Setup saves whichever URL was picked, so `LOGFIRE_BASE_URL` cannot send elsewhere."""
SELF_HOSTED = 'self-hosted'
SIGN_IN_TIMEOUT = 600.0
"""Seconds to wait for the browser approval, as `logfire auth` does."""
_POLL_FAILURES = 4

ModelT = TypeVar('ModelT', bound=BaseModel)

Announce = Callable[[str], None]
OpenBrowser = Callable[[str], bool]


class SetupError(ValueError):
    """Setup stopped; the message says why and what to do. A `ValueError`, so the loader shows it as one."""

    def __init__(self, message: str) -> None:
        # Messages can quote a self-hosted server's own text, so terminal controls in it are made inert.
        super().__init__(terminal_text(message))


class _Device(BaseModel):
    device_code: str
    frontend_auth_url: str


class _UserToken(BaseModel):
    token: str


class _Account(BaseModel):
    email: str


class Project(BaseModel):
    """A project the signed-in user can write to."""

    organization_name: str
    project_name: str

    @property
    def label(self) -> str:
        """As Logfire shows it: `organization/project`."""
        # A self-hosted server chooses these names, so terminal controls in them are made inert.
        return terminal_text(f'{self.organization_name}/{self.project_name}', keep='')

    @property
    def key_name(self) -> str:
        """The `/keys` entry for this project's write token, such as `LOGFIRE_TOKEN_PYDANTIC_CLAI2`."""
        return re.sub(r'[^A-Z0-9]+', '_', f'LOGFIRE_TOKEN_{self.organization_name}_{self.project_name}'.upper())


_PROJECTS: TypeAdapter[list[Project]] = TypeAdapter(list[Project])


@dataclass(frozen=True, kw_only=True)
class Setup:
    """Everything setup touches outside the plugin, so tests can swap each piece."""

    announce: Announce
    runners: Runners = TERMINAL
    http: Callable[[], httpx.AsyncClient] = lambda: httpx.AsyncClient(timeout=httpx.Timeout(30, read=60))
    open_browser: OpenBrowser = webbrowser.open
    sleep: Callable[[float], Awaitable[None]] = anyio.sleep


@dataclass(frozen=True)
class Chosen:
    """What setup produced: the saved key, the Logfire it belongs to, and the account that signed in."""

    token: KeyReference
    base_url: str
    project: Project
    account_email: str | None


async def run_setup(setup: Setup, *, current: str | None, owned: KeyReference | None) -> Chosen | None:
    """Pick a destination, sign in, pick a project, and save its write token; `None` when cancelled.

    `owned` is the key the plugin already uses: setting up the same project again replaces it, but any other
    key of the same name is left alone.
    """
    base_url = await run_worker(lambda: pick_destination(setup.runners, current=current))
    if base_url is None:
        return None
    with telemetry.span('logfire setup', destination=_destination(base_url)) as span:
        async with setup.http() as http:
            user_token = await sign_in(http, base_url, setup)
            account_email = await _account_email(http, base_url, user_token)
            projects = await _projects(http, base_url, user_token)
            if not projects:
                raise SetupError(f'You cannot write to any project on {base_url} yet. Create one there, then retry.')
            project = await run_worker(lambda: pick_project(setup.runners, projects))
            if project is None:
                span.set('outcome', 'cancelled')
                return None
            value = await _write_token(http, base_url, user_token, project)
        name = await to_thread.run_sync(lambda: _save(project.key_name, value, owned=owned))
        span.set('outcome', 'saved')
    return Chosen(token=KeyReference(name=name), base_url=base_url, project=project, account_email=account_email)


def pick_destination(runners: Runners, *, current: str | None) -> str | None:
    """A hosted region, or a self-hosted URL typed in; blocking, so it runs in `run_worker`."""
    items = [MenuItem(label, value=url) for label, url in REGIONS.items()]
    items.append(MenuItem('Self-hosted Logfire...', value=SELF_HOSTED))
    initial = next((index for index, item in enumerate(items) if item.value == current), 0)
    result = runners.run_choice(
        MenuBuilder('Where should Logfire traces go?')
        .style(markdown_style())
        .items(items)
        .initial_index(initial)
        .preview(
            lambda item: (
                'Next, sign in (or sign up) in the browser and pick a project. CLAI saves a write '
                'token for it in /keys; plugin settings keep only its name.'
            )
        )
        .footer_hint('Enter continue - Esc cancel')
        .key_source(menu_key)
        .build()
    )
    if result.cancelled or result.item is None or not isinstance(result.item.value, str):
        return None
    if result.item.value != SELF_HOSTED:
        return result.item.value
    typed = runners.run_text(
        TextInputBuilder('Self-hosted Logfire URL')
        .style(markdown_style())
        .prompt('https://')
        .footer_hint('Enter continue - Esc cancel')
        .key_source(menu_key)
        .build()
    )
    if typed.cancelled or not isinstance(typed.value, str) or not typed.value.strip():
        return None
    return https_origin(typed.value)


def https_origin(text: str) -> str:
    """An https origin from what was typed; the scheme may be left off."""
    text = text.strip()
    parts = urlsplit(text if '://' in text else f'https://{text}')
    # User tokens and write tokens are sent here.
    if parts.username is not None or parts.password is not None:
        # It would be saved in plaintext plugin settings; do not echo it back either.
        raise SetupError('Leave credentials out of the URL: CLAI signs in through the browser.')
    if parts.scheme != 'https' or not parts.netloc or parts.path not in ('', '/') or parts.query or parts.fragment:
        raise SetupError(f'Use an https URL with no path, like https://logfire.example.com, not {text}.')
    return f'https://{parts.netloc}'


def pick_project(runners: Runners, projects: list[Project]) -> Project | None:
    """One of the projects the user can write to; blocking, so it runs in `run_worker`."""
    builder = (
        MenuBuilder('Send traces to which project?')
        .style(markdown_style())
        .items([MenuItem(project.label, value=project) for project in projects])
        .preview(lambda item: 'CLAI creates a write token for this project and saves it in /keys.')
    )
    result = runners.run_choice(slash_search(builder, footer='enter select · esc cancel', key_source=menu_key))
    if result.cancelled or result.item is None or not isinstance(result.item.value, Project):
        return None
    return result.item.value


async def sign_in(http: httpx.AsyncClient, base_url: str, setup: Setup) -> str:
    """Logfire's device sign-in: show and open the link, then wait for approval; returns the user token."""
    response = await _call(http.post(f'{base_url}/v1/device-auth/new/', params={'machine_name': platform.node()}))
    device = _parse(_Device, response)
    if urlsplit(device.frontend_auth_url).scheme != 'https':
        raise SetupError(f'{base_url} sent a sign-in link that is not https; not opening it.')
    setup.announce(f'Sign in to Logfire (new users can sign up there): {device.frontend_auth_url}')
    if not await to_thread.run_sync(lambda: _open(setup.open_browser, device.frontend_auth_url)):
        setup.announce('No browser opened; open the link above yourself, on any device.')
    deadline = time.monotonic() + SIGN_IN_TIMEOUT
    failures = 0
    while time.monotonic() < deadline:
        # Logfire holds each poll open for a while and answers `null` until the link is approved.
        try:
            response = await http.get(f'{base_url}/v1/device-auth/wait/{device.device_code}')
            response.raise_for_status()
        except httpx.HTTPError as exc:
            failures += 1
            if failures >= _POLL_FAILURES:
                raise SetupError(f'Lost contact with {base_url} while waiting for sign-in ({type(exc).__name__}).')
            await setup.sleep(1)
            continue
        failures = 0
        if response.content.strip() not in (b'', b'null'):
            return _parse(_UserToken, response).token
    raise SetupError('The sign-in link expired before it was approved. Run /plugins configure observability to retry.')


async def _account_email(http: httpx.AsyncClient, base_url: str, user_token: str) -> str | None:
    """The signed-in account's email, from the SDK's `get_user_information` endpoint.

    `None` when the server does not say: only the session root's tag needs it, so setup carries on.
    """
    try:
        response = await _call(http.get(f'{base_url}/v1/account/me', headers={'Authorization': user_token}))
        return _parse(_Account, response).email
    except SetupError:
        return None


async def _projects(http: httpx.AsyncClient, base_url: str, user_token: str) -> list[Project]:
    response = await _call(http.get(f'{base_url}/v1/writable-projects/', headers={'Authorization': user_token}))
    try:
        return _PROJECTS.validate_json(response.content)
    except ValidationError:
        raise SetupError(f'{base_url} answered the project list with something unexpected.') from None


async def _write_token(http: httpx.AsyncClient, base_url: str, user_token: str, project: Project) -> str:
    path = f'/v1/organizations/{project.organization_name}/projects/{project.project_name}/write-tokens/'
    response = await _call(http.post(f'{base_url}{path}', headers={'Authorization': user_token}))
    return _parse(_UserToken, response).token


async def _call(request: Awaitable[httpx.Response]) -> httpx.Response:
    try:
        response = await request
    except httpx.HTTPError as exc:
        raise SetupError(f'Could not reach Logfire ({type(exc).__name__}). Check the URL and retry.') from None
    if response.is_error:
        raise SetupError(f'Logfire refused {response.request.url.path} (HTTP {response.status_code}).')
    return response


def _parse(model: type[ModelT], response: httpx.Response) -> ModelT:
    try:
        return model.model_validate_json(response.content)
    except ValidationError:
        raise SetupError(f'Logfire answered {response.request.url.path} with something unexpected.') from None


def _save(name: str, value: str, *, owned: KeyReference | None) -> str:
    """Save under `name`, or `name_2`, `name_3`, ... when another credential already has it; returns the name."""
    candidates = [name, *(f'{name}_{number}' for number in range(2, 100))]
    if owned is not None and owned.name in candidates:
        save_key(name=owned.name, value=value)
        return owned.name
    for candidate in candidates:
        try:
            save_key(name=candidate, value=value, replace=False)
        except KeyExistsError:
            continue
        return candidate
    raise SetupError(f'Too many /keys entries start with {name}; delete some and retry.')


def _destination(url: str) -> str:
    """For telemetry: which region, never a self-hosted server's name."""
    return next((label for label, region in REGIONS.items() if region == url), SELF_HOSTED)


def _open(open_browser: OpenBrowser, url: str) -> bool:
    try:
        return open_browser(url)
    except webbrowser.Error:
        return False
