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

from pydantic_clai2.plugins.menus import Dialogs, Option

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
    'Logfire US': Destination(base_url='https://logfire-us.pydantic.dev'),
    'Logfire EU': Destination(base_url='https://logfire-eu.pydantic.dev'),
}
"""The hosted regions, whose MCP URLs are harness's `LOGFIRE_US_MCP_URL` and `LOGFIRE_EU_MCP_URL`.

Spelled out rather than imported: `pydantic_ai_harness.logfire_mcp` loads the MCP client, and `observability`
imports this module at every startup.
"""


async def choose_destination(dialogs: Dialogs, *, current: Destination | None, then: str) -> Destination | None:
    """A hosted region, or another Logfire typed in; `None` when cancelled. Asked wherever the menu was opened.

    `current` starts selected; `then` says what happens after the pick, in short lines.
    """
    options = [
        Option(value=name, label=name, description=f'UI and API  {region.base_url}\nMCP server  {region.mcp_url}')
        for name, region in REGIONS.items()
    ]
    options.append(
        Option(
            value=OTHER,
            label='Another Logfire...',
            description='\n'.join(['Self-hosted or staging: type its address,', 'as any of', *_EXAMPLES]),
        )
    )
    typed_before = urlsplit(current.base_url).netloc if current is not None and current.region is None else ''
    start = OTHER if typed_before else (current.region if current is not None else None)
    while True:
        picked = await dialogs.choose(title='Which Logfire?', options=options, current=start, description=then)
        if picked is None:
            return None
        if picked != OTHER:
            return REGIONS[picked]
        typed = await dialogs.text(
            title='Another Logfire',
            label='Address',
            placeholder='logfire.example.com, https://logfire.example.com, or .../mcp',
            initial=typed_before,
            check=destination_problem,
        )
        if typed:
            return parse_destination(typed)
        start = OTHER  # Esc, or nothing typed, goes back to the list, on the row it came from.


_EXAMPLES = ('  logfire.example.com', '  https://logfire.example.com', '  https://logfire.example.com/mcp')


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
        (logfire_dir() / _REMEMBERED).write_text(
            _Remembered(base_url=destination.base_url).model_dump_json(), encoding='utf-8'
        )
    except OSError:
        pass
