"""Signing in to a service the same way in every plugin: only when the user asks, never while loading or in a prompt.

The session waits for every plugin to load, and a prompt waits for its tools, so a browser sign-in started there that
nobody finishes holds everything up. A plugin therefore offers its sign-in from its settings menu or a command, through
`sign_in_now`, behind a waiting screen that Esc closes. Runs use only what is already stored: without a sign-in the
plugin adds no tools and `warn_if_signed_out` says how to sign in, and a sign-in the service rejects fails the run at
once with `SignInRequiredError`.

A `SignInMethod` is one service's sign-in. `pydantic_clai2.mcp.OAuthSignIn` is the one for an MCP server's browser
OAuth; `pydantic_clai2.pkce.PKCESignIn` and Logfire's device sign-in are others.
"""

from collections.abc import Callable
from typing import Protocol

from anyio import to_thread
from rich.console import Console

from pydantic_ai.exceptions import UserError
from pydantic_clai2.commands import Command
from pydantic_clai2.plugins.menus import Dialogs
from pydantic_clai2.ui.menus.dialogs import TerminalDialogs
from pydantic_clai2.ui.menus.field_menu import TERMINAL, Runners
from pydantic_clai2.ui.rendering import theme

__all__ = (
    'SUBCOMMANDS',
    'SignInMethod',
    'SignInRequiredError',
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


class SignInRequiredError(UserError):
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
    return _status(method, method.signed_in())


def _status(method: SignInMethod, signed_in: bool | None) -> str:
    if signed_in is None:
        return f'Could not tell whether {method.service} is signed in: the keyring could not be read.'
    return f'Signed in to {method.service}.' if signed_in else not_signed_in(method)


async def warn_if_signed_out(method: SignInMethod, console: Console) -> bool:
    """For `on_session_start`: whether `method` is signed in, after printing how to sign in when it is not."""
    signed_in = await to_thread.run_sync(method.signed_in, abandon_on_cancel=True)
    if not signed_in:
        # From the answer already read off the loop: reading the keyring again here could block the loop.
        console.print(_status(method, signed_in), style=theme.color(theme.WARNING), markup=False, highlight=False)
    return bool(signed_in)


RUNNERS: Runners = TERMINAL
"""How the terminal shows the waiting screen when a caller passes no dialogs; tests swap in scripted ones."""


async def sign_in_now(method: SignInMethod, dialogs: Dialogs | None = None) -> str:
    """Sign in behind the waiting screen, and say how it went. Esc cancels; nothing here raises a sign-in failure.

    `dialogs` shows the screen, such as the one a settings menu action gets; `None` means the terminal.
    """
    try:
        finished = await wait_for_sign_in(method, dialogs)
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


async def run_subcommand(method: SignInMethod, args: list[str], dialogs: Dialogs | None = None) -> str | None:
    """`login`, `logout`, or `status` (also no arguments); `None` for anything else, so a command can add its own."""
    match args:
        case ['login']:
            return await sign_in_now(method, dialogs)
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


async def wait_for_sign_in(method: Signing, dialogs: Dialogs | None = None) -> bool:
    """Run `method.sign_in` behind `dialogs`' waiting screen; whether it finished. Esc cancels it; its errors propagate.

    The screen closes by itself when the sign-in returns, and shows the latest text the sign-in passed to `show`.
    `None` shows it in the terminal.
    """
    if dialogs is None:
        dialogs = TerminalDialogs(RUNNERS)
    return await dialogs.wait(
        title=f'Signing in to {method.service}',
        description=(
            f'Finish signing in to {method.service} in your browser.\n\n'
            f'Esc cancels; CLAI keeps working without {method.service}.'
        ),
        task=lambda show: method.sign_in(show=show),
    )


# TODO(v3): remove the `SignInRequired` alias and this `__getattr__`.
def __getattr__(name: str) -> object:
    if name == 'SignInRequired':
        import warnings

        from pydantic_ai.exceptions import PydanticAIDeprecationWarning

        warnings.warn(
            '`SignInRequired` has been renamed to `SignInRequiredError`. '
            'Update your imports; this deprecated alias will be removed in a future release.',
            PydanticAIDeprecationWarning,
            stacklevel=2,
        )
        return SignInRequiredError
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
