"""`/sandbox PROVIDER` and `/computer-use PROVIDER`: run a new conversation in a sandbox, optionally with a desktop.

The sandbox gets the branch, its unpushed commits, and the uncommitted files (`seed`). Its `git` and `gh`
act as the user's `gh` login. Where the provider's proxy can add the token to requests for GitHub, the token
never enters the sandbox; elsewhere it is written into the sandbox. A conversation stays where its first turn ran: its history refers to files there.

The conversation's history records its sandbox: `/sandbox` and `/computer-use` save an exchange whose response
names it (`ModelResponse.workspace_ref`), as every later response does. Desktop sandboxes carry `-desktop` on
their ref's provider so `/resume`, `/fork`, and compaction preserve desktop mode and `ComputerUse` stays bound.
A sandbox stays until `/sandbox delete` (or `/computer-use delete`), which keeps the history: a copy of the
conversation may still use it.
"""

import logging
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

import anyio
import anyio.lowlevel
from anyio import to_thread
from rich.console import RenderableType

from pydantic_ai import AgentStreamEvent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.computer_use import ComputerResult, ComputerToolAction
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.workspaces import (
    CommandResult,
    Workspace,
    WorkspaceBackend,
    WorkspaceCommand,
    WorkspaceError,
    WorkspaceRef,
    WrapperWorkspace,
)
from pydantic_ai_harness.computer_use import SupportsDisplay
from pydantic_ai_pax.builtin_plugins.sandbox.computer_use import (
    DesktopComputerUse,
    history_frames,
    render_computer_event,
)
from pydantic_ai_pax.builtin_plugins.sandbox.providers import PROVIDERS, GitHubCredential, SandboxProvider
from pydantic_ai_pax.builtin_plugins.sandbox.providers.base import (
    SANDBOX_ENV,
    give_github_login,
    is_desktop_ref,
    ref_provider,
)
from pydantic_ai_pax.builtin_plugins.sandbox.seed import SandboxBackend, Snapshot, seed, snapshot
from pydantic_ai_pax.builtin_plugins.sandbox.viewer import DesktopSession, DesktopViewer, sandbox_key
from pydantic_ai_pax.commands import Command
from pydantic_ai_pax.config.api_keys import load_keys
from pydantic_ai_pax.gh_cli import gh_token, gh_user
from pydantic_ai_pax.models.profiles import KEY_NAME
from pydantic_ai_pax.plugins import NoSettings, PanelView, Plugin, PluginHost, SessionEnd, SidePanel
from pydantic_ai_pax.plugins.keys import choose_key
from pydantic_ai_pax.plugins.menus import Option
from pydantic_ai_pax.runtime.imported_history import imported_context
from pydantic_ai_pax.runtime.session import ConversationWorkspace, ShellLineWorkspace
from pydantic_ai_pax.ui.browser import open_browser
from pydantic_ai_pax.ui.menus.dialogs import RUNNERS, TerminalDialogs
from pydantic_ai_pax.ui.rendering.status import StatusSegment
from pydantic_ai_pax.ui.rendering.tool_output import terminal_text

__all__ = ('SandboxPlugin',)

_logger = logging.getLogger(__name__)


def history_sandbox(messages: Sequence[ModelMessage]) -> WorkspaceRef | None:
    """The sandbox the history's latest response worked in, when a `/sandbox` or `/computer-use` provider owns it."""
    response = next((message for message in reversed(messages) if isinstance(message, ModelResponse)), None)
    return _owned(response.workspace_ref if response is not None else None)


def _owned(ref: WorkspaceRef | None) -> WorkspaceRef | None:
    return ref if ref is not None and _owner(ref) is not None else None


def _owner(ref: WorkspaceRef) -> SandboxProvider | None:
    """The provider whose sandbox `ref` names; none for a `local` ref, this machine's own folder."""
    return next(
        (
            provider
            for name, provider in PROVIDERS.items()
            if ref.provider in (ref_provider(name), ref_provider(name, desktop=True))
        ),
        None,
    )


@dataclass(kw_only=True)
class _Sandbox:
    """A sandbox this session made or opened, shared by every conversation whose history names it."""

    provider: SandboxProvider
    ref: WorkspaceRef
    desktop: bool = False
    env: dict[str, str] = field(default_factory=dict[str, str], repr=False)
    """What `request_env` gave while this session holds the sandbox, added to every command; kept out of `repr`."""
    backend: SandboxBackend | None = None
    deleted: bool = False

    def open(self) -> SandboxBackend:
        """The workspace, attached on first use; `UserError` when it cannot be, so the run fails closed."""
        if self.deleted:
            raise UserError(
                f"This conversation's {self.provider.title} sandbox was deleted. Run /new to start another."
            )
        if self.backend is None:
            if (missing := self.provider.missing(desktop=self.desktop)) is not None:
                raise UserError(f'This conversation runs in a {self.provider.title} sandbox. {missing}')
            self.backend = self.provider.reopen(self.ref, desktop=self.desktop)
        return self.backend

    async def capture_screen(self) -> bytes:
        """Capture a live PNG screenshot from this sandbox's display."""
        backend = self.open()
        if not isinstance(backend, SupportsDisplay) or backend.display is None:
            raise UserError(f'This {self.provider.title} sandbox has no desktop display.')
        return await backend.display.screenshot()

    async def live_desktop_url(self) -> str | None:
        """Return the provider's native interactive desktop URL when available."""
        return await self.provider.desktop_url(self.open())

    async def close(self) -> None:
        """Close what the provider opened for the workspace; the next `open` reopens it."""
        if (backend := self.backend) is not None:
            self.backend = None
            await self.provider.close(backend)


@dataclass
class _ConversationSandbox(ConversationWorkspace[None]):
    """Supplies the run's workspace when the run's history names a sandbox."""

    plugin: 'SandboxPlugin' = field(repr=False)

    def supplies(self, messages: Sequence[ModelMessage]) -> bool:
        return history_sandbox(messages) is not None

    def get_instructions(self) -> Callable[[RunContext[None]], str]:
        return self._instructions

    def _instructions(self, ctx: RunContext[None]) -> str:
        if (ref := _owned(ctx.workspace.ref)) is None:
            return ''
        sandbox = self.plugin.sandbox(ref)
        provider = sandbox.provider
        desktop_note = (
            ' This sandbox also has a graphical desktop that you can see and control with the `computer` tool.'
            if sandbox.desktop
            else ''
        )
        return (
            f'Your shell and file tools run in a {provider.title} sandbox. {provider.working_dir} '
            "is a copy of the user's checkout, made when this conversation started; the user sees changes there "
            'only once they are pushed. If the user had a gh login when the sandbox was made, git, and gh if it is '
            'installed, are signed in to GitHub as the user'
            + (
                ', with a token saved in the sandbox, which anything run there can read; leave GH_TOKEN unset. '
                if provider.token_in_sandbox
                else ' through a proxy outside the sandbox, so the sandbox holds only a stand-in for the token, '
                'which the proxy replaces. '
            )
            + 'Otherwise the copy has no GitHub access and no origin remote. Never run `gh auth login`.'
            + desktop_note
        )

    def get_toolset(self) -> FunctionToolset[None] | None:
        # The key picker needs a terminal; a session without one, such as one the browser runs, gets no tool.
        if not self.plugin.host.console.is_terminal:
            return None
        toolset = FunctionToolset[None]()
        toolset.add_function(self.request_env, prepare=_in_sandbox)
        return toolset

    async def request_env(self, ctx: RunContext[None], name: str, reason: str) -> str:
        """Ask the user for a key a command needs, such as an API key, and set it as environment variable `name`.

        Call it when a command fails for want of a key or token. Never ask the user to paste one in the chat.
        The value never reaches you; the variable goes to every command you start afterwards.

        Args:
            ctx: Framework-provided run context.
            name: The environment variable the command reads, such as ANTHROPIC_API_KEY.
            reason: What it is for, in a few words the user reads, such as 'run the eval script'.
        """
        if re.fullmatch(KEY_NAME, name) is None:
            return f'{name!r} is not an environment variable name: use capitals, digits, and underscores.'
        if name in _SET_BY_CLAI:
            return f'Pax sets {name} in the sandbox itself, so it cannot be replaced.'
        ref = _owned(ctx.workspace.ref)
        assert ref is not None  # `_in_sandbox` offers the tool only there.
        async with self.plugin.host.full_screen():
            key = await choose_key(
                name=name,
                # The model wrote `reason`: escaped, so control characters cannot redraw the question the user answers.
                label=f'The agent asks for {name} to {terminal_text(reason, keep="")}. '
                'Anything run in the sandbox can read it.',
                dialogs=TerminalDialogs(RUNNERS),
            )
        if key is None:
            return f'The user did not give {name}.'
        saved = (await to_thread.run_sync(load_keys)).get(key.name)
        if saved is None:  # Deleted in `/keys` since the picker showed it.
            return f'The user did not give {name}.'
        self.plugin.sandbox(ref).env[name] = saved.get_secret_value()
        return (
            f'{name} is set for every command started from now on, until Pax restarts; ask again if a command '
            'then finds it missing. Restart a server started before it.'
        )

    def get_workspace(self, ctx: RunContext[None], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        conversation = self.plugin.host.conversation
        if ref is None and ctx.conversation_id == conversation.conversation_id:
            # A run of the conversation on screen without its history, such as a Monty cell or a task prompt.
            ref = history_sandbox(conversation.messages)
        return None if (ref := _owned(ref)) is None else self._workspace(ref)

    def shell_line_workspace(self, messages: Sequence[ModelMessage]) -> ShellLineWorkspace | None:
        if (ref := history_sandbox(messages)) is None:
            return None
        return ShellLineWorkspace(
            workspace=self._workspace(ref), name=f'the {self.plugin.sandbox(ref).provider.title} sandbox'
        )

    def _workspace(self, ref: WorkspaceRef) -> WorkspaceBackend:
        """The sandbox `ref` names, with the variables `request_env` gave it; agent runs and `!` lines both use it."""
        sandbox = self.plugin.sandbox(ref)
        # The provider's backend stays unwrapped on `_Sandbox`, as `provider.close` keys its clients by it.
        return _EnvWorkspace(Workspace(sandbox.open()), env=sandbox.env, ref=sandbox.ref)


class _EnvWorkspace(WrapperWorkspace):
    """Adds the variables `request_env` gave the sandbox to every command, read as each one starts.

    It reports the sandbox's ref as `/sandbox` or `/computer-use` recorded it, which responses carry, so
    `/sandbox local`'s VMs and `-desktop` sandboxes stay distinct from other workspaces.
    """

    def __init__(self, wrapped: Workspace, *, env: Mapping[str, str], ref: WorkspaceRef) -> None:
        super().__init__(wrapped)
        self._env = env
        """The sandbox's own `env`, not a copy, so a grant mid-run reaches the run's later commands."""
        self._ref = ref

    @property
    def ref(self) -> WorkspaceRef:
        return self._ref

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        cwd: str | None = None,
    ) -> CommandResult:
        # The caller's own `env` wins, so a tool that sets a variable on purpose keeps it.
        return await super().run(command, shell=shell, env={**self._env, **(env or {})}, timeout=timeout, cwd=cwd)


class SandboxPlugin(Plugin):
    """`/sandbox PROVIDER` and `/computer-use PROVIDER` move a new conversation into a sandbox (or desktop sandbox)."""

    def __init__(self, host: PluginHost[None], settings: NoSettings) -> None:
        super().__init__(host, settings)
        self._sandboxes: dict[WorkspaceRef, _Sandbox] = {}
        """Each sandbox this session made or opened."""
        self._viewer = DesktopViewer()

    def get_capabilities(self) -> Sequence[AbstractCapability[None]]:
        return (_ConversationSandbox(self), DesktopComputerUse(on_frame=self._record_frame))

    def get_commands(self) -> Sequence[Command]:
        return (
            Command(
                name='sandbox',
                description=(
                    f'Run a new conversation in a sandbox ({", ".join(PROVIDERS)}; local ones run a VM on this '
                    'machine), or show where it runs'
                ),
                handler=self.command,
                complete=self.complete,
            ),
            Command(
                name='computer-use',
                description=(
                    'Run a new conversation in a sandbox with a desktop for the model to see and control '
                    '(/computer-use [PROVIDER | view | delete])'
                ),
                handler=self.computer_use_command,
                complete=self.computer_use_complete,
                during_turn_subcommands=('view',),
            ),
        )

    def get_status_segments(self) -> Sequence[StatusSegment]:
        return (self._segment,)

    def _segment(self) -> str:
        sandbox = self.current()
        if sandbox is None:
            return ''
        return f'{sandbox.provider.name} desktop' if sandbox.desktop else f'{sandbox.provider.name} sandbox'

    def get_side_panels(self) -> Sequence[SidePanel]:
        return (SidePanel(id='desktop', title='Desktop', icon='panel', view=self._desktop_panel),)

    async def _desktop_panel(self) -> PanelView:
        sandbox = self.current()
        if sandbox is None or not sandbox.desktop or sandbox.deleted:
            return PanelView(note='Run /computer-use PROVIDER to start a sandbox with a desktop.')
        self._viewer_session(sandbox)
        return PanelView(url=self._viewer.url(sandbox.ref))

    def render(self, event: AgentStreamEvent) -> RenderableType | None:
        return render_computer_event(event)

    def sandbox(self, ref: WorkspaceRef) -> _Sandbox:
        """The sandbox `ref` names, which a `/sandbox` or `/computer-use` provider owns."""
        if (sandbox := self._sandboxes.get(ref)) is None:
            provider = _owner(ref)
            assert provider is not None
            sandbox = self._sandboxes[ref] = _Sandbox(provider=provider, ref=ref, desktop=is_desktop_ref(ref))
        return sandbox

    def current(self) -> _Sandbox | None:
        """The sandbox the conversation on screen runs in; `None` when it runs on this machine."""
        ref = history_sandbox(self.host.conversation.messages)
        return None if ref is None else self.sandbox(ref)

    def _viewer_session(self, sandbox: _Sandbox) -> DesktopSession:
        """The viewer session for `sandbox`, registered and seeded from retained history on first access."""
        session = self._viewer.get(sandbox_key(sandbox.ref))
        if session is None:
            session = self._viewer.register(
                DesktopSession(
                    ref=sandbox.ref,
                    title=sandbox.provider.title,
                    capture=sandbox.capture_screen,
                    live_url=sandbox.live_desktop_url,
                    has_live_url=sandbox.provider.has_desktop_url,
                )
            )
        if not session.seeded:
            session.mark_seeded()
            messages = self.host.conversation.messages
            if history_sandbox(messages) == sandbox.ref:
                for png, actions, result in history_frames(messages):
                    session.record(png, actions=actions, result=result)
        return session

    def _record_frame(
        self,
        ref: WorkspaceRef,
        png: bytes,
        actions: Sequence[ComputerToolAction],
        result: ComputerResult | None,
    ) -> None:
        if _owned(ref) is not None and is_desktop_ref(ref):
            self._viewer_session(self.sandbox(ref)).record(png, actions=actions, result=result)

    def complete(self, args: list[str]) -> Iterable[str]:
        return _complete_provider_command(args, extra=('delete',))

    def computer_use_complete(self, args: list[str]) -> Iterable[str]:
        return _complete_provider_command(args, extra=('view', 'delete'))

    async def command(self, args: list[str]) -> str:
        match args:
            case []:
                return self._where()
            case ['delete']:
                return await self._delete()
            case ['local']:
                return _local_choice()
            case [*words] if (name := ' '.join(words)) in PROVIDERS:
                return await self._start(PROVIDERS[name])
            case _:
                raise ValueError(f'Usage: /sandbox [{" | ".join(PROVIDERS)} | delete]')

    async def computer_use_command(self, args: list[str]) -> str:
        match args:
            case []:
                return await self._computer_use_bare()
            case ['view']:
                return await self._view()
            case ['delete']:
                return await self._delete()
            case ['local']:
                return _local_choice(desktop=True)
            case [*words] if (name := ' '.join(words)) in PROVIDERS:
                return await self._start(PROVIDERS[name], desktop=True)
            case _:
                raise ValueError(f'Usage: /computer-use [{" | ".join(PROVIDERS)} | view | delete]')

    def _where(self) -> str:
        sandbox = self.current()
        if sandbox is None:
            offer = ' or '.join(f'/sandbox {name}' for name in PROVIDERS)
            return f'This conversation runs on this machine. {offer} moves a new conversation into a sandbox.'
        if sandbox.deleted:
            return f'This conversation ran in {sandbox.provider.title} sandbox {sandbox.ref.id}, now deleted. Run /new.'
        if sandbox.desktop:
            return (
                f'This conversation runs in {sandbox.provider.title} desktop sandbox {sandbox.ref.id}, '
                f'in {sandbox.provider.working_dir}. /computer-use view opens the desktop; /sandbox delete deletes it.'
            )
        return (
            f'This conversation runs in {sandbox.provider.title} sandbox {sandbox.ref.id}, '
            f'in {sandbox.provider.working_dir}. /sandbox delete deletes it.'
        )

    async def _computer_use_bare(self) -> str:
        sandbox = self.current()
        if sandbox is not None:
            if sandbox.deleted:
                return (
                    f'This conversation ran in {sandbox.provider.title} sandbox {sandbox.ref.id}, now deleted. '
                    'Run /new.'
                )
            if sandbox.desktop:
                return (
                    f'This conversation runs in {sandbox.provider.title} desktop sandbox {sandbox.ref.id}, '
                    f'in {sandbox.provider.working_dir}. Run /computer-use view to watch it, or /computer-use delete '
                    'to delete it.'
                )
            return (
                f'This conversation runs in {sandbox.provider.title} sandbox {sandbox.ref.id} without a desktop. '
                f'Run /new, then /computer-use {sandbox.provider.name}, to start one with a desktop.'
            )
        if self.host.conversation.messages:
            raise ValueError(
                'This conversation has already run on this machine, and its history refers to files here. '
                'Run /new, then /computer-use PROVIDER, to start one with a desktop.'
            )
        if not self.host.console.is_terminal:
            offer = ' or '.join(f'/computer-use {name}' for name in PROVIDERS)
            return f'This conversation runs on this machine. {offer} starts a new conversation with a desktop.'
        picked = await self._pick_desktop_provider()
        return '' if picked is None else await self._start(PROVIDERS[picked], desktop=True)

    async def _pick_desktop_provider(self) -> str | None:
        ready: list[Option] = []
        unavailable: list[Option] = []
        for name, provider in PROVIDERS.items():
            missing = provider.missing(desktop=True)
            if missing is None:
                mode = 'interactive live desktop' if provider.has_desktop_url else 'live screen viewer'
                ready.append(Option(value=name, label=f'{provider.title} ({name})', description=f'Ready · {mode}.'))
            else:
                unavailable.append(
                    Option(value=name, label=f'{provider.title} ({name}) · setup needed', description=missing)
                )
        return await TerminalDialogs(RUNNERS).choose(
            title='Start a computer-use desktop sandbox',
            description='Pick a sandbox with a desktop the model can control.',
            options=(*ready, *unavailable),
        )

    async def _view(self) -> str:
        sandbox = self.current()
        if sandbox is None or not sandbox.desktop:
            raise ValueError(
                'This conversation has no computer-use desktop. Run /computer-use PROVIDER in a new conversation first.'
            )
        if sandbox.deleted:
            raise ValueError(
                f"This conversation's {sandbox.provider.title} desktop sandbox was deleted. Run /new to start another."
            )
        if (live_url := await self._try_live_desktop_url(sandbox)) is not None:
            opened = await self._open_in_browser(live_url)
            action = 'Opened live desktop in your browser' if opened else 'Live desktop'
            return f'{action}: {live_url}'
        session = self._viewer_session(sandbox)
        if not session.frames:
            try:
                await session.capture_live()
            except (UserError, WorkspaceError):
                # Best effort: the viewer still opens and updates on the first computer action or Refresh click.
                pass
        url = self._viewer.url(sandbox.ref)
        opened = await self._open_in_browser(url)
        action = 'Opened desktop viewer in your browser' if opened else 'Desktop viewer'
        return f'{action}: {url}'

    async def _try_live_desktop_url(self, sandbox: _Sandbox) -> str | None:
        """Fetch a login-authenticated provider live desktop URL; bearer links stay behind an explicit viewer click."""
        if not sandbox.provider.has_desktop_url or sandbox.provider.desktop_url_is_bearer:
            return None
        try:
            return await sandbox.live_desktop_url()
        except (UserError, WorkspaceError, RuntimeError):
            _logger.warning(
                'Could not open provider live desktop URL for %s; falling back to local viewer',
                sandbox.ref,
                exc_info=True,
            )
            return None

    async def _open_in_browser(self, url: str) -> bool:
        """Open `url` via a private redirect page off the event loop, keeping tokens out of process argv."""
        if not self.host.console.is_terminal:
            return False
        target = await self._viewer.launch_url(url)
        return await to_thread.run_sync(open_browser, target, abandon_on_cancel=True)

    async def _start(self, provider: SandboxProvider, *, desktop: bool = False) -> str:
        cmd = '/computer-use' if desktop else '/sandbox'
        current = self.current()
        if current is not None:
            return _already_in_sandbox(current, provider, desktop=desktop)
        conversation = self.host.conversation
        if conversation.messages:
            raise ValueError(
                'This conversation has already run on this machine, and its history refers to files here. '
                f'Run /new, then {cmd} {provider.name}, to start one in {provider.title}.'
            )
        if (missing := provider.missing(desktop=desktop)) is not None:
            raise ValueError(missing)
        github = await to_thread.run_sync(_github)
        checkout = await to_thread.run_sync(snapshot, self.host.workspace)
        ref, backend = await _create(provider, github, desktop=desktop)
        try:
            # Saved before seeding, which can take minutes, so a Pax that dies meanwhile leaves a conversation
            # naming the sandbox, which `/sandbox delete` can delete. The request is Pax's, not a prompt the user
            # typed, so `/rewind` cannot cut the history back to before it. The response only names the sandbox:
            # it is empty, so providers send nothing for it, and the run's instructions say where the tools run.
            request = imported_context(
                f'The user ran {cmd} {provider.name}.', source='clai', kind='sandbox', at=datetime.now(UTC)
            )
            await conversation.commit_messages(
                [ModelRequest(parts=[request]), ModelResponse(parts=[], workspace_ref=ref)]
            )
            if github is not None and provider.token_in_sandbox:
                # After the commit, so a sandbox holding the token is named for `/sandbox delete` even if this fails.
                await give_github_login(backend, github)
            # Without a login the sandbox cannot clone from GitHub, so the whole history goes over in the bundle.
            await seed(backend, checkout if github is not None else replace(checkout, clone_url=None))
        except BaseException:
            with anyio.CancelScope(shield=True):
                try:
                    await provider.delete(ref)
                    # Only once it is gone: a sandbox whose delete failed stays named, so `/sandbox delete` retries.
                    await conversation.commit_messages([])
                finally:
                    await provider.close(backend)
            raise
        created = _Sandbox(provider=provider, ref=ref, desktop=desktop, backend=backend)
        self._sandboxes[ref] = created
        if desktop:
            self._viewer_session(created)
        return _started(provider, ref, checkout, github=github, desktop=desktop)

    async def _delete(self) -> str:
        sandbox = self.current()
        if sandbox is None:
            return 'This conversation has no sandbox.'
        await sandbox.provider.delete(sandbox.ref)
        await sandbox.close()
        self._viewer.remove(sandbox.ref)
        sandbox.deleted = True
        return f'Deleted {sandbox.provider.title} sandbox {sandbox.ref.id}. Run /new to start another conversation.'

    async def on_session_end(self, event: SessionEnd) -> None:
        # Only closed: a sandbox is deleted only when the user says so, as its conversation can be resumed.
        await self._viewer.close()
        for sandbox in list(self._sandboxes.values()):
            with anyio.CancelScope(shield=True):
                await sandbox.close()


_SET_BY_CLAI = frozenset({'GH_TOKEN', *SANDBOX_ENV})
"""What `request_env` refuses: `GH_TOKEN` is Pax's GitHub login (a proxy's stand-in, or unset so the saved token is used)."""


def _complete_provider_command(args: list[str], *, extra: tuple[str, ...]) -> Iterable[str]:
    words = [name.split() for name in PROVIDERS]
    if len(args) <= 1:
        return (*dict.fromkeys(first for first, *_ in words), *extra)
    # The second word of a provider such as `local smol`.
    return [second for first, *rest in words for second in rest if first == args[0]] if len(args) == 2 else ()


def _already_in_sandbox(current: _Sandbox, provider: SandboxProvider, *, desktop: bool) -> str:
    if desktop and not current.desktop:
        raise ValueError(
            f'This conversation runs in {current.provider.title} sandbox {current.ref.id} without a desktop. '
            f'Run /new, then /computer-use {provider.name}.'
        )
    if current.provider is provider:
        if desktop:
            return (
                f'This conversation already runs in {provider.title} desktop sandbox {current.ref.id}. '
                'Run /computer-use view to watch it.'
            )
        return f'This conversation already runs in {provider.title} sandbox {current.ref.id}.'
    raise ValueError(f'This conversation runs in {current.provider.title} sandbox {current.ref.id}. Run /new first.')


async def _in_sandbox(ctx: RunContext[None], tool: ToolDefinition) -> ToolDefinition | None:
    """`request_env` only for a run in a `/sandbox` sandbox: on this machine, commands read the user's own environment."""
    return tool if _owned(ctx.workspace.ref) is not None else None


async def _create(
    provider: SandboxProvider, github: GitHubCredential | None, *, desktop: bool = False
) -> tuple[WorkspaceRef, SandboxBackend]:
    """`provider.create`, seen through even when `/sandbox` is cancelled meanwhile, so the sandbox is deleted, not lost.

    A provider may go on creating after its caller is cancelled, as E2B's backend does, so the call is shielded and
    the cancellation is raised only once it has finished and its sandbox is deleted.
    """
    created = None
    with anyio.CancelScope(shield=True):
        created = await provider.create(github=github, desktop=desktop)
    assert created is not None  # A shielded scope is never cancelled, so the call above finished.
    ref, backend = created
    try:
        # Raises a cancellation that arrived while creating.
        await anyio.lowlevel.checkpoint()
    except BaseException:
        with anyio.CancelScope(shield=True):
            try:
                await provider.delete(ref)
            finally:
                await provider.close(backend)
        raise
    return ref, backend


def _github() -> GitHubCredential | None:
    """`gh`'s working login on github.com; `None` without `gh`, without a login, or when GitHub refuses it.

    `UserError` when GitHub does not answer in time.
    """
    try:
        token = gh_token('github.com')
    except UserError:
        return None
    # Not caught: `gh_user` asks GitHub, and a GitHub too slow to answer is no reason to say the user has no login.
    login = gh_user('github.com') if token is not None else None
    return None if token is None or login is None else GitHubCredential(login=login, token=token)


def _local_choice(*, desktop: bool = False) -> str:
    """What `/sandbox local` or `/computer-use local` says: which engine to run the VM with, and what each needs."""
    cmd = '/computer-use' if desktop else '/sandbox'
    local = [provider for name, provider in PROVIDERS.items() if name.startswith('local ')]
    offer = ' or '.join(f'{cmd} {provider.name} ({provider.title})' for provider in local)
    suffix = ' with a desktop' if desktop else ''
    lines = [f'{cmd} local runs a new conversation in a VM on this machine{suffix}. Pick its engine: {offer}.']
    lines += [
        f'{provider.title}: {missing}'
        for provider in local
        if (missing := provider.missing(desktop=desktop)) is not None
    ]
    return '\n'.join(lines)


def _started(
    provider: SandboxProvider,
    ref: WorkspaceRef,
    checkout: Snapshot,
    *,
    github: GitHubCredential | None,
    desktop: bool = False,
) -> str:
    """What `/sandbox PROVIDER` or `/computer-use PROVIDER` says once the sandbox holds the checkout."""
    if checkout.head is None and checkout.work is None:
        holds = 'an empty folder: this folder is not in a Git repository with files'
    else:
        at = f'{checkout.branch} at {checkout.head[:7]}' if checkout.branch and checkout.head else 'this checkout'
        holds = f'{at}{", with your uncommitted changes" if checkout.changed else ""}'
    kind = 'desktop sandbox' if desktop else 'sandbox'
    lines = [f'This conversation now runs in {provider.title} {kind} {ref.id}. {provider.working_dir} holds {holds}.']
    if desktop:
        lines.append(
            'Desktop enabled (1280x800): the model can see and control the screen with the computer tool. '
            'Run /computer-use view to watch it in your browser.'
        )
    if github is None:
        cmd = '/computer-use' if desktop else '/sandbox'
        lines.append(
            'gh has no GitHub login here, so the sandbox cannot push or reach private repositories. '
            f'Run {cmd} delete and gh auth login, then /new and {cmd} {provider.name} again.'
        )
    else:
        lines.append(
            f'git and gh there act as @{github.login}, on every repository that login reaches. '
            + (
                'The token is saved in the sandbox, where anything run there can read it, until the sandbox is deleted.'
                if provider.token_in_sandbox
                else f"{provider.title}'s proxy, outside the sandbox, holds the token until the sandbox is deleted."
            )
        )
        if checkout.clone_url is None and checkout.head is not None:
            lines.append('origin is not on GitHub, so pushing it needs credentials the sandbox does not have.')
    lines.append(
        'Your files here stay as they are: ask the agent to push, then pull here. ! commands run in the sandbox too.'
    )
    return '\n'.join(lines)
