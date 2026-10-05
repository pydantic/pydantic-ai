"""Opt-in Agent of Empires (AoE) status, session, and title integration.

AoE launches CLAI as a custom agent in a tmux session and reads a non-built-in agent's state from
`/tmp/aoe-hooks-<euid>/<instance>/status`. This plugin writes that file the way AoE's own hooks do,
publishes the conversation ID beside it, remembers which conversation the AoE session holds so a
restarted pane picks it up again, and pushes the conversation title with `aoe session rename`.
"""

import asyncio
import logging
import math
import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import anyio
from anyio.to_thread import run_sync
from pydantic import BaseModel, ValidationError

from pydantic_ai import AgentRunResult, RunContext
from pydantic_ai.capabilities import AbstractCapability, AgentCapability, WrapRunHandler, on_event
from pydantic_ai_harness.ask_user import AskUserAnsweredEvent, AskUserRequestedEvent
from pydantic_clai2.plugins import (
    ConversationChanged,
    NoSettings,
    Plugin,
    PluginHost,
    SessionEnd,
    SessionStart,
    TurnEnd,
    TurnStart,
)

_LOGGER = logging.getLogger(__name__)

AoeStatus = Literal['running', 'waiting', 'idle', 'error']

_INSTANCE_ID = re.compile(r'[A-Za-z0-9_-]{1,64}')
"""AoE's `validate_instance_id`: one safe path component."""
_SESSION_ID = re.compile(r'[A-Za-z0-9_.][A-Za-z0-9_.-]{0,255}')
"""AoE's `is_valid_session_id`."""
_AGENT_SESSION = re.compile(r'aoe_(?!term_).*_([A-Za-z0-9_-]{1,8})')
"""The tmux session AoE starts an agent in: `aoe_<title>_<first 8 characters of the instance ID>`."""
_CLAI2 = frozenset({'clai2', 'pydantic-clai2'})
_RUNNING_REFRESH = 60.0
"""AoE ignores a `running` status older than 15 minutes, so a long run rewrites it well before that."""
_LOOKUP_ATTEMPTS = 20
_LOOKUP_INTERVAL = 0.25
"""AoE sets the instance ID on the tmux session just after starting the pane, so allow it five seconds."""
_COMMAND_TIMEOUT = 10.0
_FILE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def hooks_base() -> Path:
    """The per-user directory AoE reads hook files from; `/tmp` regardless of `TMPDIR`, as AoE has it."""
    return Path(f'/tmp/aoe-hooks-{os.geteuid()}')


def state_directory() -> Path:
    """Where the AoE session to conversation mapping lives: `$XDG_STATE_HOME/pydantic-clai2/aoe`."""
    root = os.environ.get('XDG_STATE_HOME', '')
    state = Path(root) if os.path.isabs(root) else Path.home() / '.local' / 'state'
    return state / 'pydantic-clai2' / 'aoe'


def run(*argv: str) -> subprocess.CompletedProcess[str] | None:
    """Run a short command; `None` when it cannot start or times out."""
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=_COMMAND_TIMEOUT, check=False)
    except (OSError, subprocess.TimeoutExpired):
        _LOGGER.debug('aoe: %s failed', argv[0], exc_info=True)
        return None


def _output(*argv: str) -> str | None:
    result = run(*argv)
    return result.stdout if result is not None and result.returncode == 0 else None


@dataclass(frozen=True)
class Pane:
    """The AoE instance a tmux pane belongs to, and the pane's first process."""

    instance_id: str
    pane_pid: int


def locate(pane: str) -> Pane | None:
    """The AoE instance running in tmux pane `pane`, or `None` outside an AoE agent session.

    The ID comes from this process's environment or, as AoE sets it for custom agents, the tmux
    session's hidden environment. AoE sets that just after the pane starts, so it is retried briefly.
    """
    shown = _output('tmux', 'display-message', '-p', '-t', pane, '#{session_name}\t#{pane_pid}')
    if shown is None:
        return None
    session_name, _, pane_pid = shown.strip().rpartition('\t')
    if not pane_pid.isdigit():
        return None
    instance_id = os.environ.get('AOE_INSTANCE_ID')
    if instance_id is None:
        agent_session = _AGENT_SESSION.fullmatch(session_name)
        if agent_session is None:
            return None
        for attempt in range(_LOOKUP_ATTEMPTS):
            if attempt:
                time.sleep(_LOOKUP_INTERVAL)
            hidden = _output('tmux', 'show-environment', '-t', pane, '-h', 'AOE_INSTANCE_ID')
            if hidden is not None and hidden.startswith('AOE_INSTANCE_ID='):
                instance_id = hidden.strip().partition('=')[2]
                break
        if instance_id is None or not instance_id.startswith(agent_session.group(1)):
            return None
    return Pane(instance_id, int(pane_pid)) if _INSTANCE_ID.fullmatch(instance_id) else None


def nested(pane_pid: int) -> bool:
    """Whether another CLAI is an ancestor of this process within the pane, so that one reports instead."""
    listing = _output('ps', '-A', '-o', 'pid=,ppid=,args=')
    if listing is None:
        return False
    processes: dict[int, tuple[int, str]] = {}
    for line in listing.splitlines():
        fields = line.split(None, 2)
        if len(fields) >= 2 and fields[0].isdigit() and fields[1].isdigit():
            processes[int(fields[0])] = (int(fields[1]), fields[2] if len(fields) == 3 else '')
    pid = os.getpid()
    while pid != pane_pid and pid in processes:
        pid = processes[pid][0]
        if pid in processes and _is_clai2(processes[pid][1]):
            return True
    return False


def _is_clai2(args: str) -> bool:
    tokens = args.split()
    return any(Path(token).name in _CLAI2 for token in tokens[:2]) or tokens[1:3] == ['-m', 'pydantic_clai2']


def _open_directory(name: str | os.PathLike[str], *, parent: int | None = None, create: bool) -> int | None:
    """Open a directory as AoE's `dir_guard` does: no symlink, owned by this user, no group or other access."""
    if create:
        try:
            os.mkdir(name, 0o700, dir_fd=parent)
        except FileExistsError:
            pass
    try:
        descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent)
    except FileNotFoundError:
        return None
    info = os.fstat(descriptor)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o7077:
        os.close(descriptor)
        raise PermissionError(f'{name} must be a directory owned by this user with mode 0700')
    return descriptor


def _write_at(directory: int, name: str, content: str, *, mode: int = 0o600) -> None:
    """Replace `name` atomically: write a fresh temporary file beside it, then rename it over."""
    temporary = f'.{name}.{secrets.token_hex(8)}.tmp'
    descriptor = os.open(temporary, _FILE_FLAGS, mode, dir_fd=directory)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as file:
            file.write(content)
        os.rename(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
    except BaseException:
        os.unlink(temporary, dir_fd=directory)
        raise


@dataclass(frozen=True)
class HookFiles:
    """One AoE instance's hook directory, written with the guards AoE applies when reading it."""

    base: Path
    instance_id: str

    def write(self, name: str, content: str) -> None:
        """Write a file, creating the per-user and per-instance directories as AoE would."""
        base = _open_directory(self.base, create=True)
        assert base is not None, 'a directory just created exists'
        try:
            instance = _open_directory(self.instance_id, parent=base, create=True)
            assert instance is not None, 'a directory just created exists'
            try:
                _write_at(instance, name, content)
            finally:
                os.close(instance)
        finally:
            os.close(base)

    def remove(self, name: str) -> None:
        """Remove a file if its directory is still there; AoE deletes the directory when it stops a session."""
        base = _open_directory(self.base, create=False)
        if base is None:
            return
        try:
            instance = _open_directory(self.instance_id, parent=base, create=False)
            if instance is None:
                return
            try:
                os.unlink(name, dir_fd=instance)
            except FileNotFoundError:
                pass
            finally:
                os.close(instance)
        finally:
            os.close(base)


@dataclass(frozen=True)
class Mappings:
    """Which conversation each AoE instance holds, kept across restarts in 0600 files."""

    directory: Path

    def read(self, instance_id: str) -> str | None:
        """The conversation last recorded for `instance_id`."""
        try:
            saved = (self.directory / instance_id).read_text(encoding='utf-8').strip()
        except FileNotFoundError:
            return None
        return saved if _SESSION_ID.fullmatch(saved) else None

    def write(self, instance_id: str, conversation_id: str) -> None:
        """Record `conversation_id` for `instance_id`."""
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(self.directory, _DIRECTORY_FLAGS)
        try:
            _write_at(descriptor, instance_id, conversation_id + '\n')
        finally:
            os.close(descriptor)


class _CurrentSession(BaseModel):
    """The part of `aoe session current --json` this plugin reads."""

    id: str
    profile: str


def aoe_profile(aoe: str, instance_id: str) -> str | None:
    """The AoE profile holding this pane's session, from `aoe session current --json`."""
    shown = _output(aoe, 'session', 'current', '--json')
    if shown is None:
        return None
    try:
        current = _CurrentSession.model_validate_json(shown)
    except ValidationError:
        return None
    return current.profile if current.id == instance_id else None


def linked_worktree(directory: Path) -> bool:
    """Whether `directory` is in a linked Git worktree, whose `.git` is a file rather than a directory.

    AoE moves an AoE-managed worktree to follow a new title (`session.tie_workdir_to_name`, on by
    default) unless the agent is running, which would pull the directory out from under CLAI.
    """
    for parent in (directory, *directory.parents):
        marker = parent / '.git'
        if marker.is_dir():
            return False
        if marker.is_file():
            # A submodule's `.git` file points into `.git/modules/`, a linked worktree's into `.git/worktrees/`.
            gitdir = marker.read_text(encoding='utf-8', errors='replace').partition('gitdir:')[2].strip()
            return '/worktrees/' in gitdir.replace(os.sep, '/')
    return False


def rename(aoe: str, profile: str, instance_id: str, title: str) -> bool | None:
    """Set the AoE session title: `False` when AoE refuses, `None` when `aoe` did not run to completion."""
    # `--title=` keeps a title that starts with `-` from reading as an option.
    result = run(aoe, '-p', profile, 'session', 'rename', instance_id, f'--title={title}')
    if result is None:
        return None
    if result.returncode != 0:
        _LOGGER.debug('aoe: session rename failed: %s', result.stderr.strip())
    return result.returncode == 0


@dataclass(kw_only=True)
class _Published:
    status: AoeStatus | None = None
    status_at: float = -math.inf
    session_id: str | None = None
    title: str | None = None


@dataclass(kw_only=True)
class _Reporter:
    """Tracks what AoE should show, and publishes it from one worker task once attached to an instance."""

    host: PluginHost[None]
    running: int = 0
    waiting: dict[str, str | None] = field(default_factory=dict[str, str | None])
    """Open questions by request ID, with the run that asked each."""
    failed: bool = False
    hooks: HookFiles | None = None
    mappings: Mappings | None = None
    aoe: str | None = None
    profile: str | None = None
    titles: bool = True
    published: _Published = field(default_factory=_Published)
    changed: anyio.Event = field(default_factory=anyio.Event)
    worker: asyncio.Task[None] | None = None

    @property
    def status(self) -> AoeStatus:
        return 'waiting' if self.waiting else 'running' if self.running else 'error' if self.failed else 'idle'

    def foreground(self, ctx: RunContext[None]) -> bool:
        """A run of the shell's own conversation, rather than a background `/fork`."""
        return ctx.conversation_id == self.host.conversation.conversation_id

    def update(self) -> None:
        self.changed.set()

    async def run(self, ctx: RunContext[None], handler: WrapRunHandler) -> AgentRunResult[object]:
        foreground = self.foreground(ctx)
        self.running += foreground
        self.update()
        try:
            return await handler()
        finally:
            self.running -= foreground
            # An ended run can no longer be waiting on the questions it left open.
            self.waiting = {request: run for request, run in self.waiting.items() if run != ctx.run_id}
            self.update()

    def asked(self, ctx: RunContext[None], event: AskUserRequestedEvent) -> None:
        self.waiting[event.request.id] = ctx.run_id
        self.update()

    def answered(self, event: AskUserAnsweredEvent) -> None:
        self.waiting.pop(event.request_id, None)
        self.update()

    async def attach(self, *, hooks: HookFiles, mappings: Mappings, aoe: str | None) -> None:
        self.hooks = hooks
        self.mappings = mappings
        self.aoe = aoe
        self.titles = not await run_sync(linked_worktree, Path.cwd())
        self.worker = asyncio.create_task(self.publish(), name='clai2-aoe')

    async def publish(self) -> None:
        """Write whatever changed, one write at a time, until the plugin unloads."""
        while True:
            self.changed = anyio.Event()
            for step in (self.publish_status, self.publish_session, self.publish_title):
                try:
                    await step()
                except Exception:
                    _LOGGER.debug('aoe: publishing failed', exc_info=True)
            with anyio.move_on_after(_RUNNING_REFRESH if self.published.status == 'running' else math.inf):
                await self.changed.wait()

    @property
    def instance(self) -> HookFiles:
        assert self.hooks is not None, 'only an attached reporter publishes'
        return self.hooks

    async def publish_status(self) -> None:
        published, status = self.published, self.status
        stale = time.monotonic() - published.status_at >= _RUNNING_REFRESH
        if status != published.status or (status == 'running' and stale):
            await run_sync(self.instance.write, 'status', status)
            published.status, published.status_at = status, time.monotonic()

    async def publish_session(self) -> None:
        conversation_id = self.host.conversation.conversation_id
        if conversation_id == self.published.session_id:
            return
        assert self.mappings is not None, 'only an attached reporter publishes'
        self.published.session_id = conversation_id
        try:
            if _SESSION_ID.fullmatch(conversation_id):
                await run_sync(self.instance.write, 'session_id', conversation_id + '\n')
        finally:
            # The mapping lives outside the hooks directory, so record it even when that is unusable.
            await run_sync(self.mappings.write, self.instance.instance_id, conversation_id)

    async def publish_title(self) -> None:
        title = self.host.conversation.title
        if not self.titles or self.aoe is None or title is None or title == self.published.title:
            return
        instance_id = self.instance.instance_id
        # Whatever happens, try again only for the next title, not on every status change.
        self.published.title = title
        if self.profile is None:
            self.profile = await run_sync(aoe_profile, self.aoe, instance_id)
        if self.profile is not None and await run_sync(rename, self.aoe, self.profile, instance_id, title) is False:
            # AoE refused, as it does for a running worktree-tied session, so stop asking for this session.
            self.titles = False

    def turn(self, *, failed: bool) -> None:
        """A turn started (`failed=False`), or one failed, before or during its run: `error` lasts until the next starts."""
        self.failed = failed
        self.update()

    async def stop(self) -> None:
        with anyio.CancelScope(shield=True):
            if self.worker is not None:
                self.worker.cancel()
                try:
                    await self.worker
                except asyncio.CancelledError:
                    pass
            if self.hooks is not None:
                try:
                    await run_sync(self.hooks.remove, 'status')
                except OSError:
                    _LOGGER.debug('aoe: could not remove the status file', exc_info=True)


@dataclass
class _Reporting(AbstractCapability[None]):
    """Report foreground runs and open questions as they happen."""

    reporter: _Reporter

    async def wrap_run(self, ctx: RunContext[None], *, handler: WrapRunHandler) -> AgentRunResult[object]:
        return await self.reporter.run(ctx, handler)

    @on_event(AskUserRequestedEvent)
    async def _question(self, ctx: RunContext[None], event: AskUserRequestedEvent) -> None:
        self.reporter.asked(ctx, event)

    @on_event(AskUserAnsweredEvent)
    async def _answered(self, ctx: RunContext[None], event: AskUserAnsweredEvent) -> None:
        self.reporter.answered(event)


class AoePlugin(Plugin):
    """Report to Agent of Empires only inside its tmux sessions. Nothing happens anywhere else."""

    def __init__(self, host: PluginHost[None], settings: NoSettings) -> None:
        super().__init__(host, settings)
        self.pane = os.environ.get('TMUX_PANE')
        # Headless runs have no terminal, and AoE only starts CLAI in a tmux pane on a POSIX system.
        candidate = sys.platform != 'win32' and bool(os.environ.get('TMUX')) and bool(self.pane)
        self.reporter = _Reporter(host=host) if candidate and host.console.is_terminal else None

    def get_capabilities(self) -> Sequence[AgentCapability[None]]:
        return () if self.reporter is None else (_Reporting(self.reporter),)

    async def on_session_start(self, event: SessionStart) -> None:
        reporter, pane = self.reporter, self.pane
        if reporter is None or pane is None:
            return
        try:
            located = await run_sync(locate, pane, abandon_on_cancel=True)
            if located is None or await run_sync(nested, located.pane_pid, abandon_on_cancel=True):
                self.reporter = None
                return
            mappings = Mappings(state_directory())
            if not event.conversation_chosen:
                await self._restore(await run_sync(mappings.read, located.instance_id))
            await reporter.attach(
                hooks=HookFiles(hooks_base(), located.instance_id),
                mappings=mappings,
                aoe=await run_sync(shutil.which, 'aoe'),
            )
        except Exception:
            _LOGGER.debug('aoe: could not attach to the AoE session', exc_info=True)

    async def _restore(self, conversation_id: str | None) -> None:
        """Pick up the conversation a restarted pane held, unless this one already has history."""
        conversation = self.host.conversation
        if conversation_id is None or conversation.messages or conversation.title is not None:
            return
        try:
            self.host.console.print(await conversation.resume(conversation_id), markup=False)
        except (LookupError, ValueError, RuntimeError):
            _LOGGER.debug('aoe: could not resume %s', conversation_id, exc_info=True)

    async def on_turn_start(self, event: TurnStart) -> None:
        if self.reporter is not None:
            self.reporter.turn(failed=False)

    async def on_turn_end(self, event: TurnEnd) -> None:
        # Only a failure changes anything: a fork finishing must not clear another turn's `error`.
        if self.reporter is not None and event.outcome == 'failed':
            self.reporter.turn(failed=True)

    async def on_conversation_changed(self, event: ConversationChanged) -> None:
        if self.reporter is not None:
            self.reporter.update()

    async def on_session_end(self, event: SessionEnd) -> None:
        # Failed loads also call this, so it must tolerate a session that never started.
        if self.reporter is not None:
            await self.reporter.stop()
