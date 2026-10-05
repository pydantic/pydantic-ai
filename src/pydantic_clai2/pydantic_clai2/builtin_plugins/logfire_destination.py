"""Which Logfire to use, asked the same way by the `observability` and `logfire_mcp` plugins' setup.

Logfire serves its web UI, its API, and its MCP server (at `/mcp`) from one host, so any of these
names the same Logfire, and each plugin derives what it needs from it:

- a bare host, such as `logfire-eu.pydantic.dev`
- the URL you open Logfire at, such as `https://logfire-eu.pydantic.dev`
- the MCP URL, such as `https://logfire-eu.pydantic.dev/mcp`

The last Logfire either plugin saved is remembered, so setting up the other one starts there.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import SplitResult, urlsplit

from pydantic import BaseModel
from termflow.tui import MenuBuilder, MenuItem, TextInputBuilder

from pydantic_ai_harness.logfire_mcp import LOGFIRE_EU_MCP_URL, LOGFIRE_US_MCP_URL
from pydantic_clai2.ui.menus.field_menu import Runners
from pydantic_clai2.ui.menus.menu_worker import menu_key
from pydantic_clai2.ui.rendering._rendering import markdown_style

_MCP_PATH = '/mcp'
OTHER = 'other'
"""The picker row for a Logfire typed in: self-hosted, or a staging host such as `logfire-eu.pydantic.info`."""
_LEGACY_HOSTS = {
    'logfire.pydantic.dev': 'logfire-us.pydantic.dev',
    'logfire-api.pydantic.dev': 'logfire-us.pydantic.dev',
}
"""Hosts from before Logfire had regions; both serve the US region, which serves `/mcp` and the device sign-in."""
_REMEMBERED = 'destination.json'


@dataclass(frozen=True)
class Destination:
    """One Logfire, by the https origin that serves its UI, API, and MCP server."""

    base_url: str
    """Such as `https://logfire-eu.pydantic.dev`: where the SDK sends and setup signs in."""

    @property
    def mcp_url(self) -> str:
        """Its MCP server."""
        return f'{self.base_url}{_MCP_PATH}'

    @property
    def region(self) -> str | None:
        """`Logfire US` or `Logfire EU` for a hosted region, else `None`."""
        return next((name for name, region in REGIONS.items() if region == self), None)

    @property
    def label(self) -> str:
        """The region's name, or the host of any other Logfire."""
        return self.region or urlsplit(self.base_url).netloc


def parse_destination(text: str) -> Destination:
    """A host, Logfire URL, or MCP URL as a `Destination`; `ValueError` says what to type instead.

    The scheme may be left off; only https is accepted, since tokens are sent there.
    """
    text = text.strip()
    parts = urlsplit(text if '://' in text else f'https://{text}')
    if parts.username is not None or parts.password is not None:
        # The text would be saved in plaintext plugin settings; do not echo it back either.
        raise ValueError('Leave credentials out of the address: CLAI signs in through the browser.')
    if not _is_logfire_address(parts):
        raise ValueError(f'Type a host (logfire.example.com), an https URL, or an MCP URL ending in /mcp, not {text}.')
    netloc = parts.netloc.lower()
    return Destination(base_url=f'https://{_LEGACY_HOSTS.get(netloc, netloc)}')


def _is_logfire_address(parts: SplitResult) -> bool:
    try:
        # Raises on a port that is not a number.
        _ = parts.port
    except ValueError:
        return False
    host = parts.hostname or ''
    return (
        parts.scheme == 'https'
        and bool(host)
        and not any(char.isspace() for char in host)
        and parts.path.rstrip('/') in ('', _MCP_PATH)
        and not parts.query
        and not parts.fragment
    )


def destination_problem(text: str) -> str | None:
    """Why typed text is not a Logfire address, or `None` when it is; the text input's validator."""
    try:
        parse_destination(text)
    except ValueError as exc:
        return str(exc)
    return None


REGIONS: dict[str, Destination] = {
    'Logfire US': parse_destination(LOGFIRE_US_MCP_URL),
    'Logfire EU': parse_destination(LOGFIRE_EU_MCP_URL),
}
"""The hosted regions, from the MCP URLs harness `LogfireMCP` publishes."""


def pick_destination(runners: Runners, *, current: Destination | None, then: str) -> Destination | None:
    """A hosted region, or another Logfire typed in; `None` when cancelled. Blocking, so run it in a worker.

    `current` is highlighted; `then` says what happens after the pick, in short lines.
    """
    items = [MenuItem(name, value=region, description=region.base_url) for name, region in REGIONS.items()]
    items.append(MenuItem('Another Logfire...', value=OTHER, description='self-hosted or staging: type its address'))
    typed_before = current is not None and current.region is None
    initial = len(items) - 1 if typed_before else next((i for i, item in enumerate(items) if item.value == current), 0)
    while True:
        result = runners.run_choice(
            MenuBuilder('Which Logfire?')
            .style(markdown_style())
            .items(items)
            .initial_index(initial)
            .preview(lambda item: _preview(item, then))
            .footer_hint('Enter choose - Esc cancel, nothing changes')
            .key_source(menu_key)
            .build()
        )
        if result.cancelled or result.item is None:
            return None
        if isinstance(result.item.value, Destination):
            return result.item.value
        typed = runners.run_text(
            TextInputBuilder('Another Logfire')
            .style(markdown_style())
            .prompt('Address: ')
            .initial(urlsplit(current.base_url).netloc if current is not None and typed_before else '')
            .placeholder('logfire.example.com, https://logfire.example.com, or .../mcp')
            .validator(destination_problem)
            .footer_hint('Enter use this address - Esc back to the list')
            .key_source(menu_key)
            .build()
        )
        if not typed.cancelled and isinstance(typed.value, str):
            return parse_destination(typed.value)
        initial = len(items) - 1  # Esc goes back to the list, on the row it came from.


def _preview(item: MenuItem, then: str) -> str:
    # One short fact per line: the preview panel cuts long lines off.
    if isinstance(item.value, Destination):
        lines = [item.label, '', f'UI and API  {item.value.base_url}', f'MCP server  {item.value.mcp_url}', '', then]
    else:
        lines = [
            'A self-hosted or staging Logfire',
            '',
            'Enter: type its address, as any of',
            '  logfire.example.com',
            '  https://logfire.example.com',
            '  https://logfire.example.com/mcp',
        ]
    return '\n'.join(lines)


def logfire_dir() -> Path:
    """CLAI's private Logfire directory: the SDK reads configuration and credentials only from here."""
    config_home = Path(os.getenv('XDG_CONFIG_HOME', '')).expanduser()
    if not config_home.is_absolute():
        config_home = Path.home() / '.config'
    return config_home / 'pydantic-clai2' / 'logfire'


class _Remembered(BaseModel):
    base_url: str


def remembered() -> Destination | None:
    """The Logfire either plugin last saved, if any. Blocking file IO."""
    try:
        saved = _Remembered.model_validate_json((logfire_dir() / _REMEMBERED).read_bytes())
        return parse_destination(saved.base_url)
    except (OSError, ValueError):  # `ValidationError` is a `ValueError`: an unreadable file is no default.
        return None


def remember(destination: Destination) -> None:
    """Make `destination` the starting point of the next setup, in either plugin. Blocking file IO.

    Only a default is lost if this fails, so it never stops setup.
    """
    try:
        logfire_dir().mkdir(parents=True, exist_ok=True)
        (logfire_dir() / _REMEMBERED).write_text(_Remembered(base_url=destination.base_url).model_dump_json())
    except OSError:
        pass
