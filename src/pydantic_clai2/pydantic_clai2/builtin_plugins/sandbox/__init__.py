"""`/sandbox PROVIDER`: run a new conversation in a cloud sandbox holding a copy of this checkout.

The sandbox gets the branch, its unpushed commits, and the uncommitted files (`seed`). Its `git` and `gh`
act as the user's `gh` login. Where the provider's proxy can add the token to requests for GitHub, the token
never enters the sandbox; elsewhere it is written into the sandbox. A conversation stays where its first turn ran: its history refers to files there.

The conversation's history records its sandbox: `/sandbox` saves an exchange whose response names it
(`ModelResponse.workspace_ref`), as every later response does. Each run is handed that reference by Pydantic
AI, so a forked, resumed, or background conversation opens its own sandbox, whichever one is on screen.
A sandbox stays until `/sandbox delete`, which keeps the history: a copy of the conversation may still use it.
"""

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

import anyio
import anyio.lowlevel
from anyio import to_thread

from pydantic_ai import RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_clai2.builtin_plugins.sandbox.providers import PROVIDERS, GitHubCredential, SandboxProvider
from pydantic_clai2.builtin_plugins.sandbox.providers.base import SANDBOX_ENV, give_github_login
from pydantic_clai2.builtin_plugins.sandbox.seed import SandboxBackend, Snapshot, seed, snapshot
from pydantic_clai2.commands import Command
from pydantic_clai2.config.api_keys import load_keys
from pydantic_clai2.gh_cli import gh_token, gh_user
from pydantic_clai2.models.profiles import KEY_NAME
from pydantic_clai2.plugins import NoSettings, Plugin, PluginHost, SessionEnd
from pydantic_clai2.plugins.keys import choose_key
from pydantic_clai2.runtime.imported_history import imported_context
from pydantic_clai2.runtime.session import ConversationWorkspace
from pydantic_clai2.ui.menus.dialogs import RUNNERS, TerminalDialogs
from pydantic_clai2.ui.rendering.status import StatusSegment

__all__ = ('SandboxPlugin',)


def history_sandbox(messages: Sequence[ModelMessage]) -> WorkspaceRef | None:
    """The sandbox the history's latest response worked in, when a `/sandbox` provider owns it."""
    response = next((message for message in reversed(messages) if isinstance(message, ModelResponse)), None)
    return _owned(response.workspace_ref if response is not None else None)


def _owned(ref: WorkspaceRef | None) -> WorkspaceRef | None:
    return ref if ref is not None and ref.provider in PROVIDERS else None


@dataclass(kw_only=True)
class _Sandbox:
    """A sandbox this session made or opened, shared by every conversation whose history names it."""

    provider: SandboxProvider
    ref: WorkspaceRef
    backend: SandboxBackend | None = None
    deleted: bool = False

    def open(self) -> SandboxBackend:
        """The workspace, attached on first use; `UserError` when it cannot be, so the run fails closed."""
        if self.deleted:
            raise UserError(
                f"This conversation's {self.provider.title} sandbox was deleted. Run /new to start another."
            )
        if self.backend is None:
            if (missing := self.provider.missing()) is not None:
                raise UserError(f'This conversation runs in a {self.provider.title} sandbox. {missing}')
            self.backend = self.provider.reopen(self.ref)
        return self.backend

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
        provider = PROVIDERS[ref.provider]
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
            return f'CLAI sets {name} in the sandbox itself, so it cannot be replaced.'
        ref = _owned(ctx.workspace.ref)
        assert ref is not None  # `_in_sandbox` offers the tool only there.
        sandbox = self.plugin.sandbox(ref)
        async with self.plugin.host.full_screen():
            key = await choose_key(
                name=name,
                label=f'The agent asks for {name} to {reason}. Anything run in the sandbox can read it.',
                dialogs=TerminalDialogs(RUNNERS),
            )
        if key is None:
            return f'The user did not give {name}.'
        saved = (await to_thread.run_sync(load_keys)).get(key.name)
        if saved is None:  # Deleted in `/keys` since the picker showed it.
            return f'The user did not give {name}.'
        value = saved.get_secret_value()
        try:
            await sandbox.provider.set_env(sandbox.open(), name, value)
        except RuntimeError as exc:
            return f'{name} could not be set: {exc}.'
        return f'{name} is set for every command started from now on. Restart a server started before it.'

    def get_workspace(self, ctx: RunContext[None], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        conversation = self.plugin.host.conversation
        if ref is None and ctx.conversation_id == conversation.conversation_id:
            # A run of the conversation on screen without its history, such as a Monty cell or a task prompt.
            ref = history_sandbox(conversation.messages)
        if (ref := _owned(ref)) is None:
            return None
        return self.plugin.sandbox(ref).open()


class SandboxPlugin(Plugin):
    """`/sandbox PROVIDER` moves a new conversation into a cloud sandbox; `/sandbox` says where it runs."""

    def __init__(self, host: PluginHost[None], settings: NoSettings) -> None:
        super().__init__(host, settings)
        self._sandboxes: dict[WorkspaceRef, _Sandbox] = {}
        """Each sandbox this session made or opened."""

    def get_capabilities(self) -> Sequence[_ConversationSandbox]:
        return (_ConversationSandbox(self),)

    def get_commands(self) -> Sequence[Command]:
        return (
            Command(
                name='sandbox',
                description=f'Run a new conversation in a cloud sandbox ({", ".join(PROVIDERS)}), or show where it runs',
                handler=self.command,
                complete=self.complete,
            ),
        )

    def get_status_segments(self) -> Sequence[StatusSegment]:
        return (self._segment,)

    def _segment(self) -> str:
        sandbox = self.current()
        return '' if sandbox is None else f'{sandbox.provider.name} sandbox'

    def sandbox(self, ref: WorkspaceRef) -> _Sandbox:
        """The sandbox `ref` names, which a `/sandbox` provider owns."""
        if (sandbox := self._sandboxes.get(ref)) is None:
            sandbox = self._sandboxes[ref] = _Sandbox(provider=PROVIDERS[ref.provider], ref=ref)
        return sandbox

    def current(self) -> _Sandbox | None:
        """The sandbox the conversation on screen runs in; `None` when it runs on this machine."""
        ref = history_sandbox(self.host.conversation.messages)
        return None if ref is None else self.sandbox(ref)

    def complete(self, args: list[str]) -> Iterable[str]:
        return (*PROVIDERS, 'delete') if len(args) <= 1 else ()

    async def command(self, args: list[str]) -> str:
        match args:
            case []:
                return self._where()
            case ['delete']:
                return await self._delete()
            case [name] if name in PROVIDERS:
                return await self._start(PROVIDERS[name])
            case _:
                raise ValueError(f'Usage: /sandbox [{" | ".join(PROVIDERS)} | delete]')

    def _where(self) -> str:
        sandbox = self.current()
        if sandbox is None:
            offer = ' or '.join(f'/sandbox {name}' for name in PROVIDERS)
            return f'This conversation runs on this machine. {offer} moves a new conversation into a cloud sandbox.'
        if sandbox.deleted:
            return f'This conversation ran in {sandbox.provider.title} sandbox {sandbox.ref.id}, now deleted. Run /new.'
        return (
            f'This conversation runs in {sandbox.provider.title} sandbox {sandbox.ref.id}, '
            f'in {sandbox.provider.working_dir}. /sandbox delete deletes it.'
        )

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
        if (missing := provider.missing()) is not None:
            raise ValueError(missing)
        github = await to_thread.run_sync(_github)
        checkout = await to_thread.run_sync(snapshot, self.host.workspace)
        ref, backend = await _create(provider, github)
        try:
            # Saved before seeding, which can take minutes, so a CLAI that dies meanwhile leaves a conversation
            # naming the sandbox, which `/sandbox delete` can delete. The request is CLAI's, not a prompt the user
            # typed, so `/rewind` cannot cut the history back to before it. The response is what the model reads.
            request = imported_context(
                f'The user ran /sandbox {provider.name}.', source='clai', kind='sandbox', at=datetime.now(UTC)
            )
            reply = TextPart(f'Working in {provider.title} sandbox {ref.id}, at {provider.working_dir}.')
            await conversation.commit_messages(
                [ModelRequest(parts=[request]), ModelResponse(parts=[reply], workspace_ref=ref)]
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
        self._sandboxes[ref] = _Sandbox(provider=provider, ref=ref, backend=backend)
        return _started(provider, ref, checkout, github=github)

    async def _delete(self) -> str:
        sandbox = self.current()
        if sandbox is None:
            return 'This conversation has no sandbox.'
        await sandbox.provider.delete(sandbox.ref)
        await sandbox.close()
        sandbox.deleted = True
        return f'Deleted {sandbox.provider.title} sandbox {sandbox.ref.id}. Run /new to start another conversation.'

    async def on_session_end(self, event: SessionEnd) -> None:
        # Only closed: a sandbox is deleted only when the user says so, as its conversation can be resumed.
        for sandbox in list(self._sandboxes.values()):
            with anyio.CancelScope(shield=True):
                await sandbox.close()


_SET_BY_CLAI = frozenset({'GH_TOKEN', *SANDBOX_ENV})
"""What `request_env` refuses: `GH_TOKEN` is CLAI's GitHub login (a proxy's stand-in, or unset so the saved token is used)."""


async def _in_sandbox(ctx: RunContext[None], tool: ToolDefinition) -> ToolDefinition | None:
    """`request_env` only for a run in a `/sandbox` sandbox: on this machine, commands read the user's own environment."""
    return tool if _owned(ctx.workspace.ref) is not None else None


async def _create(provider: SandboxProvider, github: GitHubCredential | None) -> tuple[WorkspaceRef, SandboxBackend]:
    """`provider.create`, seen through even when `/sandbox` is cancelled meanwhile, so the sandbox is deleted, not lost.

    A provider may go on creating after its caller is cancelled, as E2B's backend does, so the call is shielded and
    the cancellation is raised only once it has finished and its sandbox is deleted.
    """
    created = None
    with anyio.CancelScope(shield=True):
        created = await provider.create(github=github)
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


def _started(
    provider: SandboxProvider, ref: WorkspaceRef, checkout: Snapshot, *, github: GitHubCredential | None
) -> str:
    """What `/sandbox PROVIDER` says once the sandbox holds the checkout."""
    if checkout.head is None and checkout.work is None:
        holds = 'an empty folder: this folder is not in a Git repository with files'
    else:
        at = f'{checkout.branch} at {checkout.head[:7]}' if checkout.branch and checkout.head else 'this checkout'
        holds = f'{at}{", with your uncommitted changes" if checkout.changed else ""}'
    lines = [f'This conversation now runs in {provider.title} sandbox {ref.id}. {provider.working_dir} holds {holds}.']
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
    lines.append('Your files here stay as they are: ask the agent to push, then pull here. ! commands still run here.')
    return '\n'.join(lines)
