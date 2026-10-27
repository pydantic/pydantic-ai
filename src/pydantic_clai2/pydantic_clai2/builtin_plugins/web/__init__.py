"""Open this session in your browser: a local web UI with streaming chat, tool cards, and a sessions sidebar.

The built-in `web` plugin, off by default. `/web` starts the server and opens the browser; `clai2 --web`
starts CLAI with only the browser to drive it. The terminal and the browser drive the same session.
"""

import html
import ipaddress
import os
import secrets
import tempfile
from collections.abc import Sequence
from pathlib import Path

from anyio import to_thread
from pydantic import BaseModel, ConfigDict, Field, field_validator
from rich.console import RenderableType

from pydantic_ai import AgentStreamEvent
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai_harness.ask_user import AskUser
from pydantic_clai2.builtin_plugins.ask_user_menu import TerminalAnswerer, render_question
from pydantic_clai2.builtin_plugins.web.app import ASGIApp, create_app
from pydantic_clai2.builtin_plugins.web.server import LOOPBACK, WebServer
from pydantic_clai2.builtin_plugins.web.service import Observer, WebService
from pydantic_clai2.builtin_plugins.web.workspaces import conversation_switched
from pydantic_clai2.commands import Command
from pydantic_clai2.plugins import (
    CommandEnd,
    ConversationChanged,
    ForkEnd,
    InstallProgress,
    Plugin,
    PluginHost,
    SessionEnd,
    TaskProgress,
    TurnEnd,
)
from pydantic_clai2.ui.browser import open_browser

USAGE = 'Usage: /web [start|stop|status]'


HTTP_PORT = 80


def _unspecified(host: str) -> bool:
    """Whether `host` is an address meaning every interface, in any spelling, such as `0.0.0.0` or `::0`."""
    try:
        return ipaddress.ip_address(host).is_unspecified
    except ValueError:  # A host name, not an address.
        return False


class WebSettings(BaseModel):
    """Where the web UI listens, and whether `/web` opens the browser."""

    model_config = ConfigDict(extra='forbid', use_attribute_docstrings=True)

    host: str = '127.0.0.1'
    """The address to listen on. Keep it local: anyone who reaches it with the token can run commands as you."""

    @field_validator('host')
    @classmethod
    def _one_address(cls, value: str) -> str:
        value = value.strip().removeprefix('[').removesuffix(']')
        if not value or _unspecified(value):
            # Requests must name the address the server listens on, which a wildcard does not say.
            raise ValueError('listen on one address, such as 127.0.0.1, not on every interface')
        return value

    port: int = Field(default=0, ge=0, le=65535)
    """The port to listen on; 0 picks a free one each launch."""
    open_browser: bool = True
    """Open the browser when the server starts."""
    ask_user: bool = True
    """Answer the model's questions in the browser too, taking over from the `ask_user` plugin while loaded."""


class WebPlugin(Plugin[WebSettings]):
    """Drive this session from a local web UI: streaming chat, tool cards, sessions sidebar. /web starts it."""

    scope = 'process'
    """One server for the whole process, which shows every live session; see `WebService`."""

    def __init__(self, host: PluginHost[None], settings: WebSettings) -> None:
        super().__init__(host, settings)
        self.token = secrets.token_urlsafe(32)
        """Checked on every request; new each launch, so an old link stops working, but kept across `/update`."""
        self.port: int | None = None
        """This launch's port, replacing the saved one, as `clai2 --web --port` sets it."""
        self.listen: str | None = None
        """This launch's address, replacing the saved one: loopback for a launcher's `clai2 --web --token-file`."""
        terminal = (
            TerminalAnswerer(full_screen=host.full_screen, console=host.console) if host.console.is_terminal else None
        )
        self.service = WebService(host=host, terminal=terminal)
        self.server: WebServer | None = None
        self.url: str | None = None
        self._launch_page: Path | None = None

    def get_capabilities(self) -> Sequence[AgentCapability[None]]:
        observer = Observer(self.service)
        return (observer, AskUser(answerer=self.service.answerer)) if self.settings.ask_user else (observer,)

    def get_commands(self) -> Sequence[Command]:
        return (
            Command(
                name='web',
                description='Open this session in your browser; /web stop ends the web UI',
                handler=self.command,
                complete=lambda args: ('start', 'stop', 'status') if len(args) <= 1 else (),
                args_during_turn=True,
            ),
        )

    def render(self, event: AgentStreamEvent) -> RenderableType | None:
        # With `ask_user` taken over, its call header and answer lines still belong in the transcript.
        return render_question(event) if self.settings.ask_user else None

    async def command(self, args: list[str]) -> str:
        if args in ([], ['start']):
            started = self.url is None
            url = await self.start()
            if self.settings.open_browser:
                # A `$BROWSER` command with `%s` runs in the foreground until the browser quits.
                await to_thread.run_sync(open_browser, self._launcher(url).as_uri(), abandon_on_cancel=True)
            return f'Web UI {"running" if started else "already running"} at {url}' + (
                '' if self.settings.host in LOOPBACK else '\nOther machines can reach it: keep the link private.'
            )
        if args == ['stop']:
            if self.url is None:
                return 'The web UI is not running.'
            if not self.host.console.is_terminal:
                raise ValueError('The browser is the only way to drive this CLAI. Press Ctrl-C where it runs to quit.')
            await self.stop()
            return 'Web UI stopped.'
        if args == ['status']:
            if self.url is None:
                return 'The web UI is not running. Start it with /web.'
            return f'Web UI at {self.url} with {self.service.hub.clients} browser(s) connected.'
        raise ValueError(USAGE)

    def _launcher(self, url: str) -> Path:
        """A page only you can read that forwards to `url`, so the token never appears in a browser command line."""
        if self._launch_page is None:
            descriptor, name = tempfile.mkstemp(prefix='clai2-web-', suffix='.html')
            os.close(descriptor)
            self._launch_page = Path(name)
        self._launch_page.write_text(
            f'<!doctype html><meta http-equiv="refresh" content="0;url={html.escape(url)}">'
            f'<a href="{html.escape(url)}">Open CLAI</a>',
            encoding='utf-8',
        )
        return self._launch_page

    async def start(self) -> str:
        """Start serving, once; return the address with this launch's token."""
        if self.url is not None:
            return self.url
        host = self.settings.host if self.listen is None else self.listen
        address = f'[{host}]' if ':' in host else host

        def app(port: int) -> ASGIApp:
            names = ('127.0.0.1', 'localhost', '[::1]', address)
            hosts = frozenset(f'{name}:{port}' for name in names)
            if port == HTTP_PORT:
                # Browsers leave the scheme's default port out of `Host`.
                hosts |= frozenset(names)
            return create_app(service=self.service, token=self.token, hosts=hosts)

        self.server = WebServer.serve(app, host=host, port=self.settings.port if self.port is None else self.port)
        self.url = f'http://{address}:{self.server.port}/?token={self.token}'
        return self.url

    async def stop(self) -> None:
        server, self.server, self.url = self.server, None, None
        # An action waiting on a dialog nobody can now answer backs out, which also lets its menu close.
        self.service.dialogs.cancel_all()
        if self._launch_page is not None:
            self._launch_page.unlink(missing_ok=True)
            self._launch_page = None
        if server is not None:
            await server.stop()
        # Its chat is in the page alone: a turn no page can watch, or answer, stops.
        await self.service.orchestrator.close()

    async def on_session_end(self, event: SessionEnd) -> None:
        await self.stop()
        self.service.close()

    async def on_turn_end(self, event: TurnEnd) -> None:
        # The run ended before the session saved its outcome; the browser's turn ended with the run.
        self.service.turn_ended()

    async def on_conversation_changed(self, event: ConversationChanged) -> None:
        try:
            # Before the page and the orchestrator follow the switch, so they find the workspace naming it.
            await conversation_switched(self.service, event)
        finally:
            self.service.conversation_changed(event)

    async def on_command_end(self, event: CommandEnd) -> None:
        self.service.command_ended(event)

    async def on_task_progress(self, event: TaskProgress) -> None:
        self.service.task_progress(event)

    async def on_fork_end(self, event: ForkEnd) -> None:
        self.service.fork_ended(event)

    async def on_install_progress(self, event: InstallProgress) -> None:
        self.service.install_progress(event)
