"""Signing in to a service the same way in every plugin: only when the user asks, never while loading or in a prompt.

The session waits for every plugin to load, and a prompt waits for its tools, so a browser sign-in started there that
nobody finishes holds everything up. A plugin therefore offers its sign-in from its settings menu or a command, through
`sign_in_now`, behind a waiting screen that Esc closes. Runs use only what is already stored: without a sign-in the
plugin adds no tools and `warn_if_signed_out` says how to sign in, and a sign-in the service rejects fails the run at
once with `SignInRequired`.

A `SignInMethod` is one service's sign-in. `pydantic_clai2.mcp.OAuthSignIn` is the one for an MCP server's browser
OAuth; `pydantic_clai2.pkce.PKCESignIn` and Logfire's device sign-in are others.
"""

import asyncio
import threading
from collections.abc import Callable
from typing import Protocol

from anyio import to_thread
from rich.console import Console
from termflow.tui import MenuBuilder, MenuItem

from pydantic_ai.exceptions import UserError
from pydantic_clai2.commands import Command
from pydantic_clai2.ui.menus.field_menu import TERMINAL, Runners
from pydantic_clai2.ui.menus.menu_worker import menu_key, run_worker
from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering._rendering import markdown_style

__all__ = (
    'SUBCOMMANDS',
    'SignInMethod',
    'SignInRequired',
    'Signing',
    'not_signed_in',
    'run_subcommand',
    'sign_in_command',
    'sign_in_now',
    'sign_out_now',
    'status',
    'wait_for_sign_in',
    'warn_if_signed_out',
)


class SignInRequired(UserError):
    """A run needs a sign-in that only the user can start; the message says how."""


class Signing(Protocol):
    """What the waiting screen needs: a name to show and the sign-in to wait on."""

    @property
    def service(self) -> str:
        """The service's name in messages, such as `Notion`."""
        ...

    async def sign_in(self, *, show: Callable[[str], object]) -> object:
        """Sign in now, opening the browser. `show` displays what the user may need, such as a link and a code.

        Each call replaces what the previous one showed, so a retried attempt never leaves a stale link up.
        """
        ...


class SignInMethod(Protocol):
    """One service's sign-in, a `Signing` that can also say whether it is signed in and sign out.

    `signed_in` and `sign_out` read or write the credential store, so call them off the loop.
    """

    @property
    def service(self) -> str:
        """The service's name in messages, such as `Notion`."""
        ...

    async def sign_in(self, *, show: Callable[[str], object]) -> object:
        """As `Signing.sign_in`."""
        ...

    @property
    def setup(self) -> str:
        """The command that signs in, such as `/plugins configure notion`."""
        ...

    def signed_in(self) -> bool | None:
        """Whether a sign-in is stored; `None` when the credential store cannot be read."""
        ...

    def sign_out(self) -> None:
        """Forget the stored sign-in."""
        ...


def not_signed_in(method: SignInMethod) -> str:
    """The one wording for a missing sign-in."""
    return f'Not signed in to {method.service}. Run {method.setup} to sign in.'


def status(method: SignInMethod) -> str:
    """Whether `method` is signed in, as one line. Reads the credential store; call it off the loop."""
    signed_in = method.signed_in()
    if signed_in is None:
        return f'Could not tell whether {method.service} is signed in: the keyring could not be read.'
    return f'Signed in to {method.service}.' if signed_in else not_signed_in(method)


async def warn_if_signed_out(method: SignInMethod, console: Console) -> bool:
    """For `on_session_start`: whether `method` is signed in, after printing how to sign in when it is not."""
    signed_in = await to_thread.run_sync(method.signed_in, abandon_on_cancel=True)
    if not signed_in:
        console.print(status(method), style=theme.color(theme.WARNING), markup=False, highlight=False)
    return bool(signed_in)


RUNNERS: Runners = TERMINAL
"""How the waiting screen is shown when a caller passes no runners; tests swap in scripted ones."""


async def sign_in_now(method: SignInMethod, runners: Runners | None = None) -> str:
    """Sign in behind the waiting screen, and say how it went. Esc cancels; nothing here raises a sign-in failure."""
    try:
        finished = await wait_for_sign_in(method, runners)
    except Exception as exc:  # noqa: BLE001 -- every sign-in fails its own way; each leaves the service signed out.
        return f'Could not sign in to {method.service}: {str(exc).rstrip(".")}. Run {method.setup} to try again.'
    if not finished:
        return f'{method.service} sign-in cancelled. Run {method.setup} to try again.'
    return f'Signed in to {method.service}.'


async def sign_out_now(method: SignInMethod) -> str:
    """Forget the sign-in, and say so."""
    await to_thread.run_sync(method.sign_out, abandon_on_cancel=True)
    return f'Signed out of {method.service}. Run {method.setup} to sign in again.'


SUBCOMMANDS = ('login', 'logout', 'status')
"""What `run_subcommand` handles, for a command's completion."""


async def run_subcommand(method: SignInMethod, args: list[str], runners: Runners | None = None) -> str | None:
    """`login`, `logout`, or `status` (also no arguments); `None` for anything else, so a command can add its own."""
    match args:
        case ['login']:
            return await sign_in_now(method, runners)
        case ['logout']:
            return await sign_out_now(method)
        case [] | ['status']:
            return await to_thread.run_sync(status, method, abandon_on_cancel=True)
        case _:
            return None


def sign_in_command(name: str, method: SignInMethod) -> Command:
    """`/NAME [login | logout | status]`, for a plugin whose command does nothing else."""

    async def handle(args: list[str]) -> str:
        message = await run_subcommand(method, args)
        if message is None:
            raise ValueError(f'Usage: /{name} [login | logout | status]')
        return message

    return Command(
        name=name,
        description=f'Sign in to {method.service} in the browser, sign out, or show the sign-in (/{name} login|logout).',
        handler=handle,
        complete=lambda args: SUBCOMMANDS if len(args) <= 1 else (),
    )


_REDRAW = 'sign-in:changed'
"""A key no keyboard sends and no handler takes: termflow repaints after it, showing the new lines."""


async def wait_for_sign_in(method: Signing, runners: Runners | None = None) -> bool:
    """Run `method.sign_in` behind a waiting screen; whether it finished. Esc cancels it; its errors propagate.

    The screen closes by itself when the sign-in returns, and shows the latest text the sign-in passed to `show`.
    """
    lines = [
        f'Finish signing in to {method.service} in your browser.',
        f'Esc cancels; CLAI keeps working without {method.service}.',
    ]
    shown: list[str] = []
    changed = threading.Event()

    def show(text: str) -> None:
        shown[:] = [text]
        changed.set()

    def read_key() -> str:
        # Termflow calls this on the menu's own thread every 50 ms, so a repaint here never races a paint.
        if changed.is_set():
            changed.clear()
            return _REDRAW
        return menu_key()

    screen = (
        MenuBuilder(f'Signing in to {method.service}')
        .style(markdown_style())
        .items([MenuItem('Cancel sign-in', value=None)])
        .preview(lambda item: '\n\n'.join([*lines, *shown]))
        .footer_hint('Enter or Esc cancel')
        .key_source(read_key)
        .build()
    )
    signing = asyncio.ensure_future(method.sign_in(show=show))
    shown_with = runners or RUNNERS
    waiting = asyncio.ensure_future(run_worker(lambda: shown_with.run_choice(screen)))
    try:
        await asyncio.wait({signing, waiting}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        signing.cancel()
        waiting.cancel()
        await asyncio.gather(signing, waiting, return_exceptions=True)
    if signing.cancelled():
        return False
    signing.result()
    return True
