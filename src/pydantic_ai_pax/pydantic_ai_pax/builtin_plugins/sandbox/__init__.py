"""`/sandbox PROVIDER`: run a new conversation in a sandbox, local or in the cloud, holding a copy of this checkout.

The sandbox gets the branch, its unpushed commits, and the uncommitted files (`seed`). Its `git` and `gh`
act as the user's `gh` login. Where the provider's proxy can add the token to requests for GitHub, the token
never enters the sandbox; elsewhere it is written into the sandbox. A conversation stays where its first turn ran: its history refers to files there.

The conversation's history records its sandbox: `/sandbox` saves an exchange whose response names it
(`ModelResponse.workspace_ref`), as every later response does. Each run is handed that reference by Pydantic
AI, so a forked, resumed, or background conversation opens its own sandbox, whichever one is on screen.
A sandbox stays until `/sandbox delete`, which keeps the history: a copy of the conversation may still use it.

`/computer-use` lets the model see and control the sandbox's desktop. The history records whether it is on
(`computer_use_on`). `/computer-use` before `/sandbox` makes
the sandbox with a desktop. In a running sandbox it adds one where the provider can; where it cannot, it says how
to start a sandbox with one. The model's screenshots go to a loopback viewer
(`DesktopViewer`): `computer` rows link to it, `/computer-use view` opens it, and `pax --web` shows it beside
the transcript.
"""

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
from pydantic_ai.models import ModelRequestContext
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
from pydantic_ai_harness.computer_use import ActionsPerformedEvent, SupportsDisplay
from pydantic_ai_pax.builtin_plugins.sandbox.computer_use import (
    COMPUTER_USE,
    DesktopComputerUse,
    computer_use_on,
    computer_use_state,
    history_frames,
    render_computer_event,
)
from pydantic_ai_pax.builtin_plugins.sandbox.providers import PROVIDERS, GitHubCredential, SandboxProvider
from pydantic_ai_pax.builtin_plugins.sandbox.providers.base import SANDBOX_ENV, give_github_login, ref_provider
from pydantic_ai_pax.builtin_plugins.sandbox.seed import SandboxBackend, Snapshot, seed, snapshot
from pydantic_ai_pax.builtin_plugins.sandbox.viewer import DesktopSession, DesktopViewer, sandbox_key
from pydantic_ai_pax.commands import Command
from pydantic_ai_pax.config import PERMISSIONS_SETTING
from pydantic_ai_pax.config.api_keys import load_keys
from pydantic_ai_pax.gh_cli import gh_token, gh_user
from pydantic_ai_pax.models.profiles import KEY_NAME
from pydantic_ai_pax.plugins import NoSettings, PanelView, Plugin, PluginHost, SessionEnd, SidePanel
from pydantic_ai_pax.plugins.keys import choose_key
from pydantic_ai_pax.runtime.imported_history import imported_context
from pydantic_ai_pax.runtime.session import ConversationWorkspace, ShellLineWorkspace
from pydantic_ai_pax.ui.browser import open_browser
from pydantic_ai_pax.ui.menus.dialogs import RUNNERS, TerminalDialogs
from pydantic_ai_pax.ui.rendering.status import StatusSegment
from pydantic_ai_pax.ui.rendering.tool_output import terminal_text

__all__ = ('SandboxPlugin',)


def history_sandbox(messages: Sequence[ModelMessage]) -> WorkspaceRef | None:
    """The sandbox the history's latest response worked in, when a `/sandbox` provider owns it."""
    response = next((message for message in reversed(messages) if isinstance(message, ModelResponse)), None)
    return _owned(response.workspace_ref if response is not None else None)


def _owned(ref: WorkspaceRef | None) -> WorkspaceRef | None:
    return ref if ref is not None and _owner(ref) is not None else None


def _owner(ref: WorkspaceRef) -> SandboxProvider | None:
    """The provider whose sandbox `ref` names; none for a `local` ref, this machine's own folder."""
    return next((provider for name, provider in PROVIDERS.items() if ref_provider(name) == ref.provider), None)


@dataclass(kw_only=True)
class _Sandbox:
    """A sandbox this session made or opened, shared by every conversation whose history names it."""

    provider: SandboxProvider
    ref: WorkspaceRef
    env: dict[str, str] = field(default_factory=dict[str, str], repr=False)
    """What `request_env` gave while this session holds the sandbox, added to every command; kept out of `repr`."""
    desktop: bool = False
    """Whether `open` attaches the sandbox's desktop too, as `ComputerUse` needs."""
    backend: SandboxBackend | None = None
    deleted: bool = False
    _detached: list[SandboxBackend] = field(default_factory=list[SandboxBackend], repr=False)
    """Backends `add_desktop` replaced, which `close` closes."""

    def add_desktop(self) -> None:
        """Attach the desktop from the next `open` on, also when a history without one opened the sandbox first."""
        if not self.desktop:
            self.desktop = True
            if self.backend is not None:
                self._detached.append(self.backend)
                self.backend = None

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

    async def screenshot(self) -> bytes:
        """A screenshot of the sandbox's desktop, for the viewer's Refresh button."""
        backend = self.open()
        if not isinstance(backend, SupportsDisplay) or backend.display is None:
            raise UserError(f'This {self.provider.title} sandbox has no desktop.')
        return await backend.display.screenshot()

    async def close(self) -> None:
        """Close what the provider opened for the workspace; the next `open` reopens it."""
        backends = [*self._detached, *([self.backend] if self.backend is not None else [])]
        self.backend, self._detached = None, []
        for backend in backends:
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
        provider = self.plugin.sandbox(ref, ctx.messages).provider
        desktop = ''
        if computer_use_on(ctx.messages):
            desktop = (
                ' The sandbox also has a desktop, which you see and control with the `computer` tool.'
                if self.plugin.full_access()
                else ' The sandbox also has a desktop, but you get the `computer` tool for it only once the user '
                'runs /permissions full.'
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
            + desktop
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
        self.plugin.sandbox(ref, ctx.messages).env[name] = saved.get_secret_value()
        return (
            f'{name} is set for every command started from now on, until Pax restarts; ask again if a command '
            'then finds it missing. Restart a server started before it.'
        )

    def get_workspace(self, ctx: RunContext[None], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        conversation = self.plugin.host.conversation
        messages = ctx.messages
        if ref is None and ctx.conversation_id == conversation.conversation_id:
            # A run of the conversation on screen without its history, such as a Monty cell or a task prompt.
            messages = conversation.messages
            ref = history_sandbox(messages)
        return None if (ref := _owned(ref)) is None else self._workspace(ref, messages)

    async def after_model_request(
        self, ctx: RunContext[None], *, request_context: ModelRequestContext, response: ModelResponse
    ) -> ModelResponse:
        """Carry `COMPUTER_USE` to each new response, so the latest one always says whether it is on."""
        del request_context
        if _owned(ctx.workspace.ref) is None or (state := computer_use_state(ctx.messages)) is None:
            return response
        return replace(response, metadata={**(response.metadata or {}), COMPUTER_USE: state})

    def shell_line_workspace(self, messages: Sequence[ModelMessage]) -> ShellLineWorkspace | None:
        if (ref := history_sandbox(messages)) is None:
            return None
        return ShellLineWorkspace(
            workspace=self._workspace(ref, messages),
            name=f'the {self.plugin.sandbox(ref, messages).provider.title} sandbox',
        )

    def _workspace(self, ref: WorkspaceRef, messages: Sequence[ModelMessage]) -> WorkspaceBackend:
        """The sandbox `ref` names, with the variables `request_env` gave it; agent runs and `!` lines both use it."""
        sandbox = self.plugin.sandbox(ref, messages)
        # The provider's backend stays unwrapped on `_Sandbox`, as `provider.close` keys its clients by it.
        return _EnvWorkspace(Workspace(sandbox.open()), env=sandbox.env, ref=sandbox.ref)


class _EnvWorkspace(WrapperWorkspace):
    """Adds the variables `request_env` gave the sandbox to every command, read as each one starts.

    It reports the sandbox's ref as `/sandbox` recorded it, which responses carry, so `/sandbox local`'s VMs stay
    apart from the harness's own `smol` and `boxlite` ones.
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
    """`/sandbox PROVIDER` moves a new conversation into a sandbox; `/sandbox` says where it runs."""

    def __init__(self, host: PluginHost[None], settings: NoSettings) -> None:
        super().__init__(host, settings)
        self._sandboxes: dict[WorkspaceRef, _Sandbox] = {}
        """Each sandbox this session made or opened."""
        self._viewer = DesktopViewer()
        self._armed_for: str | None = None
        """The conversation whose next `/sandbox PROVIDER` gets a desktop, as `/computer-use` ran before it."""

    def get_capabilities(self) -> Sequence[AbstractCapability[None]]:
        return (
            _ConversationSandbox(self),
            DesktopComputerUse(on_frame=self._record_frame, full_access=self.full_access),
        )

    def full_access(self) -> bool:
        """Whether `/permissions` is `full`, which the `computer` tool needs."""
        return self.host.setting(PERMISSIONS_SETTING) == 'full'

    def _needs_full(self) -> str:
        return '' if self.full_access() else ' The model gets the computer tool only under /permissions full.'

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
                description="Let the model see and control the sandbox's desktop, or open the live view",
                handler=self.computer_use_command,
                complete=lambda args: ('on', 'off', 'view') if len(args) <= 1 else (),
                during_turn_subcommands=('view',),
            ),
        )

    def get_status_segments(self) -> Sequence[StatusSegment]:
        return (self._segment,)

    def _segment(self) -> str:
        sandbox = self.current()
        if sandbox is None:
            return 'computer-use' if self._armed() else ''
        return f'{sandbox.provider.name} sandbox{" (computer-use)" if self._on() else ""}'

    def get_side_panels(self) -> Sequence[SidePanel]:
        return (SidePanel(id='desktop', title='Desktop', icon='panel', view=self._desktop_panel),)

    async def _desktop_panel(self) -> PanelView:
        sandbox = self.current()
        if sandbox is None or sandbox.deleted or not self._on():
            return PanelView(note='Run /computer-use and /sandbox PROVIDER to give the model a desktop.')
        self._viewer_session(sandbox)
        return PanelView(url=self._viewer.url(sandbox.ref))

    def render(self, event: AgentStreamEvent) -> RenderableType | None:
        live_url = None
        if isinstance(event, ActionsPerformedEvent) and (sandbox := self.current()) is not None and self._on():
            # Started here, by the first `computer` row, so the link works the moment it is drawn.
            self._viewer_session(sandbox)
            live_url = self._viewer.try_url(sandbox.ref)
        return render_computer_event(event, live_url=live_url)

    def sandbox(self, ref: WorkspaceRef, messages: Sequence[ModelMessage]) -> _Sandbox:
        """The sandbox `ref` names, which a `/sandbox` provider owns; `messages`, a history naming it, says if it has a desktop."""
        if (sandbox := self._sandboxes.get(ref)) is None:
            provider = _owner(ref)
            assert provider is not None
            sandbox = self._sandboxes[ref] = _Sandbox(provider=provider, ref=ref)
        if computer_use_state(messages) is not None:
            sandbox.add_desktop()
        return sandbox

    def current(self) -> _Sandbox | None:
        """The sandbox the conversation on screen runs in; `None` when it runs on this machine."""
        messages = self.host.conversation.messages
        ref = history_sandbox(messages)
        return None if ref is None else self.sandbox(ref, messages)

    def _on(self) -> bool:
        """Whether computer use is on in the conversation on screen."""
        return computer_use_on(self.host.conversation.messages)

    def _armed(self) -> bool:
        return self._armed_for == self.host.conversation.conversation_id

    def _viewer_session(self, sandbox: _Sandbox) -> DesktopSession:
        """The viewer's session for `sandbox`, made on first use with the screenshots already in the history."""
        if (session := self._viewer.get(sandbox_key(sandbox.ref))) is None:
            session = self._viewer.register(
                DesktopSession(ref=sandbox.ref, title=sandbox.provider.title, capture=sandbox.screenshot)
            )
            messages = self.host.conversation.messages
            if history_sandbox(messages) == sandbox.ref:
                for png, actions, result in history_frames(messages):
                    session.record(png, actions=actions, result=result)
        return session

    def _record_frame(
        self, ref: WorkspaceRef, png: bytes, actions: Sequence[ComputerToolAction], result: ComputerResult | None
    ) -> None:
        if (sandbox := self._sandboxes.get(ref)) is not None:  # The run opened it.
            self._viewer_session(sandbox).record(png, actions=actions, result=result)

    def complete(self, args: list[str]) -> Iterable[str]:
        words = [name.split() for name in PROVIDERS]
        if len(args) <= 1:
            return (*dict.fromkeys(first for first, *_ in words), 'delete')
        # The second word of a provider such as `local smol`.
        return [second for first, *rest in words for second in rest if first == args[0]] if len(args) == 2 else ()

    async def command(self, args: list[str]) -> str:
        match args:
            case []:
                return self._where()
            case ['delete']:
                return await self._delete()
            case ['local']:
                return _local_choice(desktop=self._armed())
            case [*words] if (name := ' '.join(words)) in PROVIDERS:
                return await self._start(PROVIDERS[name])
            case _:
                raise ValueError(f'Usage: /sandbox [{" | ".join(PROVIDERS)} | delete]')

    async def computer_use_command(self, args: list[str]) -> str:
        match args:
            case []:
                on = self._armed() if self.current() is None else self._on()
                return await self._computer_use(on=not on)
            case ['on']:
                return await self._computer_use(on=True)
            case ['off']:
                return await self._computer_use(on=False)
            case ['view']:
                return await self._view()
            case _:
                raise ValueError('Usage: /computer-use [on | off | view]')

    def _where(self) -> str:
        sandbox = self.current()
        if sandbox is None:
            offer = ' or '.join(f'/sandbox {name}' for name in PROVIDERS)
            return f'This conversation runs on this machine. {offer} moves a new conversation into a sandbox.'
        if sandbox.deleted:
            return f'This conversation ran in {sandbox.provider.title} sandbox {sandbox.ref.id}, now deleted. Run /new.'
        computer_use = ' with computer use on (/computer-use view shows its desktop)' if self._on() else ''
        return (
            f'This conversation runs in {sandbox.provider.title} sandbox {sandbox.ref.id}{computer_use}, '
            f'in {sandbox.provider.working_dir}. /sandbox delete deletes it.'
        )

    async def _computer_use(self, *, on: bool) -> str:
        sandbox = self.current()
        if sandbox is None:
            self._armed_for = self.host.conversation.conversation_id if on else None
            if on:
                return 'Computer use is on: /sandbox PROVIDER now makes a sandbox with a desktop.' + self._needs_full()
            return 'Computer use is off.'
        if sandbox.deleted:
            raise ValueError(f"This conversation's {sandbox.provider.title} sandbox was deleted. Run /new.")
        if on == self._on():
            return f'Computer use is already {"on" if on else "off"}.'
        if not on:
            await self._record_computer_use(sandbox, on=False)
            return f'Computer use is off. {sandbox.provider.title} sandbox {sandbox.ref.id} keeps running.'
        if (missing := sandbox.provider.missing(desktop=True)) is not None:
            raise ValueError(missing)
        if sandbox.provider.desktop_at_create and computer_use_state(self.host.conversation.messages) is None:
            # Replacing the sandbox would lose any work a fork or copy of the conversation did in it.
            raise ValueError(
                f'{sandbox.provider.title} gives a desktop only to a new sandbox. '
                f'Run /sandbox delete, /new, /computer-use, then /sandbox {sandbox.provider.name}.'
            )
        sandbox.add_desktop()
        await self._record_computer_use(sandbox, on=True)
        return (
            'Computer use is on: the model can see and control the desktop. /computer-use view shows it.'
            + self._needs_full()
        )

    async def _record_computer_use(self, sandbox: _Sandbox, *, on: bool) -> None:
        """Save `/computer-use on|off` as an exchange, as `/sandbox` saves its own."""
        state = 'on' if on else 'off'
        request = imported_context(
            f'The user ran /computer-use {state}.', source='clai', kind=f'computer-use-{state}', at=datetime.now(UTC)
        )
        response = ModelResponse(parts=[], workspace_ref=sandbox.ref, metadata={COMPUTER_USE: on})
        await self.host.conversation.commit_messages(
            [*self.host.conversation.messages, ModelRequest(parts=[request]), response]
        )

    async def _view(self) -> str:
        sandbox = self.current()
        if sandbox is None or sandbox.deleted or not self._on():
            raise ValueError('Computer use is not on here. Run /computer-use and /sandbox PROVIDER first.')
        session = self._viewer_session(sandbox)
        if not session.frames:
            try:
                await session.capture_live()
            except (UserError, WorkspaceError):
                pass  # The viewer still opens, and shows the screen on the model's next action or a Refresh.
        url = self._viewer.url(sandbox.ref)
        if self.host.console.is_terminal:
            # Through a private redirect page, so the viewer's token stays out of the browser's command line.
            page = await self._viewer.launch_url(url)
            if await to_thread.run_sync(open_browser, page, abandon_on_cancel=True):
                return f'Opened the desktop viewer in your browser: {url}'
        return f'Desktop viewer: {url}'

    async def _start(self, provider: SandboxProvider) -> str:
        current = self.current()
        if current is not None:
            if current.provider is provider:
                return f'This conversation already runs in {provider.title} sandbox {current.ref.id}.'
            raise ValueError(
                f'This conversation runs in {current.provider.title} sandbox {current.ref.id}. Run /new first.'
            )
        conversation = self.host.conversation
        if conversation.messages:
            raise ValueError(
                'This conversation has already run on this machine, and its history refers to files here. '
                f'Run /new, then /sandbox {provider.name}, to start one in {provider.title}.'
            )
        desktop = self._armed()
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
                f'The user ran /sandbox {provider.name}.', source='clai', kind='sandbox', at=datetime.now(UTC)
            )
            response = ModelResponse(parts=[], workspace_ref=ref, metadata={COMPUTER_USE: True} if desktop else None)
            await conversation.commit_messages([ModelRequest(parts=[request]), response])
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
        self._armed_for = None
        self._sandboxes[ref] = _Sandbox(provider=provider, ref=ref, desktop=desktop, backend=backend)
        return _started(provider, ref, checkout, github=github, desktop=desktop, needs_full=self._needs_full())

    async def _delete(self) -> str:
        sandbox = self.current()
        if sandbox is None:
            return 'This conversation has no sandbox.'
        await self._delete_sandbox(sandbox)
        return f'Deleted {sandbox.provider.title} sandbox {sandbox.ref.id}. Run /new to start another conversation.'

    async def _delete_sandbox(self, sandbox: _Sandbox) -> None:
        await sandbox.provider.delete(sandbox.ref)
        await sandbox.close()
        self._viewer.remove(sandbox.ref)
        sandbox.deleted = True

    async def on_session_end(self, event: SessionEnd) -> None:
        # Only closed: a sandbox is deleted only when the user says so, as its conversation can be resumed.
        await self._viewer.close()
        for sandbox in list(self._sandboxes.values()):
            with anyio.CancelScope(shield=True):
                await sandbox.close()


_SET_BY_CLAI = frozenset({'GH_TOKEN', *SANDBOX_ENV})
"""What `request_env` refuses: `GH_TOKEN` is Pax's GitHub login (a proxy's stand-in, or unset so the saved token is used)."""


async def _in_sandbox(ctx: RunContext[None], tool: ToolDefinition) -> ToolDefinition | None:
    """`request_env` only for a run in a `/sandbox` sandbox: on this machine, commands read the user's own environment."""
    return tool if _owned(ctx.workspace.ref) is not None else None


async def _create(
    provider: SandboxProvider, github: GitHubCredential | None, *, desktop: bool
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


def _local_choice(*, desktop: bool) -> str:
    """What `/sandbox local` says: which engine to run the VM with, and what each still needs here."""
    local = [provider for name, provider in PROVIDERS.items() if name.startswith('local ')]
    offer = ' or '.join(f'/sandbox {provider.name} ({provider.title})' for provider in local)
    lines = [f'/sandbox local runs a new conversation in a VM on this machine. Pick its engine: {offer}.']
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
    desktop: bool,
    needs_full: str,
) -> str:
    """What `/sandbox PROVIDER` says once the sandbox holds the checkout; `needs_full` follows the desktop line."""
    if checkout.head is None and checkout.work is None:
        holds = 'an empty folder: this folder is not in a Git repository with files'
    else:
        at = f'{checkout.branch} at {checkout.head[:7]}' if checkout.branch and checkout.head else 'this checkout'
        holds = f'{at}{", with your uncommitted changes" if checkout.changed else ""}'
    lines = [f'This conversation now runs in {provider.title} sandbox {ref.id}. {provider.working_dir} holds {holds}.']
    if desktop:
        lines.append(
            'Computer use is on: the model can see and control its desktop. /computer-use view shows it.' + needs_full
        )
    if github is None:
        lines.append(
            'gh has no GitHub login here, so the sandbox cannot push or reach private repositories. '
            f'Run /sandbox delete and gh auth login, then /new and /sandbox {provider.name} again.'
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
