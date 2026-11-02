"""A [boxd](https://boxd.sh) machine behind Pydantic AI's `WorkspaceBackend` protocol.

SDK behavior verified 2026-10-28 against boxd 0.2.20 on boxd's production cluster. Re-check these before
changing acquisition, command, file, or error handling:

- `machines.create` returns once the machine is scheduled. A name is unique in its org: creating a second
  machine under it raises `ConflictError`, and `machines.get` finds a machine by id or name.
- A command line runs under `/bin/sh -c` (dash) as the non-root `boxd` user in `/home/boxd`, with stdin at
  EOF. A command killed by a signal exits 128 plus its number.
- `stream_exec` delivers stdout and stderr as separate byte chunks and reports the exit status as soon as the
  command exits, even while a background child holds its output open.
- Cancelling the call ends the command: boxd sends its process group SIGTERM, then SIGKILL.
- A command wakes a suspended or hibernated machine. On a stopped one `stream_exec` raises `ConflictError`,
  and `machines.start` boots it again with its disk.
- `machines.fork` copies a running machine live, disk and memory, in under a second, into a machine that is
  isolated whenever the source is. A suspended machine is forked from its disk alone, without its memory or
  the writes it had not flushed.
- Deleting or stopping a machine during a command ends the call with `APIConnectionError`; `machines.get` then
  raises `NotFoundError` or reports `stopped`. A delete can return while `get` still reports the machine for a
  moment.
- An isolated machine's metadata endpoint refuses token requests (403), and boxd's env vars and integrations
  are not installed in it. A machine created from a snapshot keeps the named `networks` of the machine the
  snapshot was saved from, even with `isolated=True`, and peers with every machine on them.
- A command that prints nothing for 30 minutes is ended, its process killed, with `APIStatusError` code
  `deadline_exceeded` ("no output from the command for 30m").
- Each client exchanges `BOXD_API_KEY` for a session token on its first call, and boxd allows 30 exchanges per
  60 seconds from one source IP (`RateLimitError` past that). Missing and rejected credentials raise
  `AuthenticationError`; an exchange that could not reach boxd raises `APIConnectionError`, and one that got
  a 5xx answer `APIStatusError`.
- `machines.fs` runs as the `boxd` user and sees the files commands see. A write creates missing parents,
  writes through a symlink, and keeps an existing file's mode; a read or write of a FIFO is refused at once.
  A failure is the matching builtin `OSError` (`FileNotFoundError`, `IsADirectoryError`, ...), a symlink loop
  is `ELOOP`, and `remove` deletes a symlink itself. A machine whose boxd agent predates the file API, such as
  one restored from an older snapshot, raises `MachineAgentOutdatedError` until it reboots.
"""

from __future__ import annotations

import errno
import logging
import os
import posixpath
import shlex
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Literal, TypeVar

import anyio
import anyio.lowlevel

from pydantic_ai.exceptions import UserError
from pydantic_ai.workspaces import (
    CommandResult,
    FileEntry,
    SupportsCommands,
    SupportsFilesystem,
    SupportsRealpath,
    WorkspaceBackend,
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from pydantic_ai.workspaces.workspace import _ShellFilesystem  # pyright: ignore[reportPrivateUsage]
from pydantic_ai_harness._workspace_provider import (
    absolute_path,
    check_env,
    check_timeout,
    command_argv,
    command_deadline,
    run_to_completion,
    running_on_asyncio,
)

if TYPE_CHECKING:
    from boxd import AsyncBoxd, FsEntry, Machine

    from pydantic_ai.workspaces import WorkspaceCommand

__all__ = ('BoxdSandboxBackend',)

_T = TypeVar('_T')

try:
    import boxd
except ImportError as _import_error:
    raise ImportError('Install `pydantic-ai-harness[boxd]` to use BoxdSandbox.') from _import_error

_AUTH_MESSAGE = (
    'boxd did not accept the credentials. Set BOXD_API_KEY, run inside a boxd machine, '
    'or pass a configured `AsyncBoxd` as `client=`.'
)
_OUTPUT_LIMIT_MESSAGE = (
    'boxd command output exceeded the 10 MiB limit; redirect it to a file and read part of it instead'
)
_SILENT_COMMAND_MESSAGE = (
    'boxd ended the command after 30 minutes without output; have it print progress, '
    'or start it in the background with its output redirected to a file'
)
_NAME_PREFIX = 'pydantic-ai-'
# Bound machine creation, attachment, and deletion so a wedged control plane cannot hang a run.
_LIFECYCLE_TIMEOUT = 120.0
_REQUEST_TIMEOUT = 30.0
_TEARDOWN_TIMEOUT = 30.0
_POLL_INTERVAL = 0.25
# The local backend's cap and preview: output is held in memory until the command ends.
_MAX_OUTPUT_BYTES = 10 * 1024 * 1024
_OUTPUT_PREVIEW_BYTES = 64 * 1024

logger = logging.getLogger(__name__)


def _require_asyncio() -> None:
    """Raise `UserError` unless the caller runs on asyncio, which the SDK's gRPC channels need."""
    if not running_on_asyncio():
        raise UserError('boxd needs the asyncio event loop: the boxd SDK runs its calls on gRPC asyncio channels.')


def _check_ref(ref: WorkspaceRef) -> None:
    """Raise `ValueError` unless `ref` names a boxd machine."""
    if ref.provider != 'boxd':
        raise ValueError(f"unsupported workspace provider {ref.provider!r}; expected 'boxd'")
    if not ref.id.strip():
        raise ValueError('boxd workspace ref id cannot be empty')


def _new_client() -> AsyncBoxd:
    """An `AsyncBoxd` configured from the environment: `BOXD_TOKEN`, else `BOXD_API_KEY`, else the machine it runs in."""
    return boxd.AsyncBoxd()


async def _close_quietly(client: AsyncBoxd) -> None:
    """Close an API client, bounded and shielded; a failure is logged, not raised."""
    with anyio.move_on_after(_TEARDOWN_TIMEOUT, shield=True) as deadline:
        try:
            await client.close()
        except Exception as error:
            logger.warning('Could not close the boxd API client: %r', error)
    if deadline.cancelled_caught:
        logger.warning('Closing the boxd API client did not complete within %ss.', _TEARDOWN_TIMEOUT)


def _gone_message(machine_id: str) -> str:
    return f"The boxd machine {machine_id!r} no longer exists. Pass `workspace='new'` to start a fresh machine."


def _refusal(machine: Machine) -> str | None:
    """Why `BoxdSandbox` does not use `machine`, or `None` when it has no route to the org's other machines."""
    if not machine.networking.isolated:
        return (
            f'The boxd machine {machine.id!r} ({machine.name}) is not isolated, so BoxdSandbox does not use it: '
            "code in it could act on your boxd account and reach your organization's other machines. "
            "Pass `workspace='new'` to start a fresh machine."
        )
    if machine.networking.networks:
        return (
            f'The boxd machine {machine.id!r} ({machine.name}) is on the boxd networks '
            f'{", ".join(machine.networking.networks)}, so BoxdSandbox does not use it: every machine on them can '
            'reach it. A machine created from a snapshot keeps the networks of the machine the snapshot was saved '
            'from; save snapshots from a machine on no network.'
        )
    return None


def _credentials_rejected(error: Exception) -> bool:
    """Whether `error` says boxd found no credentials or refused them, rather than that it could not be asked."""
    return isinstance(error, boxd.AuthenticationError | boxd.PermissionDeniedError)


async def destroy_machine(client: AsyncBoxd | None, machine_id: str) -> None:
    """Delete an isolated machine by id, whatever state it is in.

    A machine that no longer exists returns quietly; rejected credentials raise
    `WorkspaceUnavailableError`, like any other operation. Without `client`, one is opened from the
    environment and closed afterwards.

    Raises:
        UserError: The event loop is not asyncio.
        ValueError: The machine is not isolated, so it is not one `BoxdSandbox` created.
        TimeoutError: boxd did not answer in time.
    """
    _require_asyncio()
    if not machine_id.strip():
        raise ValueError('boxd workspace ref id cannot be empty')
    owned: AsyncBoxd | None = None
    if client is None:
        client = owned = _new_client()
    try:
        with anyio.fail_after(_LIFECYCLE_TIMEOUT):
            try:
                machine = await client.machines.get(machine_id)
                if not machine.networking.isolated:
                    # A ref from tampered message history must not delete one of the org's other machines.
                    raise ValueError(
                        f'The boxd machine {machine_id!r} ({machine.name}) is not isolated, so BoxdSandbox does not '
                        'delete it. Delete it with the boxd CLI or SDK if you mean to.'
                    )
                await client.machines.delete(machine.id)
            except boxd.NotFoundError:
                return
            except boxd.BoxdError as error:
                if _credentials_rejected(error):
                    raise WorkspaceUnavailableError(_AUTH_MESSAGE) from error
                raise
    finally:
        if owned is not None:
            await _close_quietly(owned)


def _translate(error: Exception, context: str) -> Exception:
    """Map a boxd SDK error onto the failures the workspace protocol promises.

    Returns `error` itself for what must propagate unchanged: connection failures, rate limits and
    quotas, server errors, and anything that is not a `BoxdError`, which durable engines retry as
    transient.
    """
    if _credentials_rejected(error):
        return WorkspaceUnavailableError(_AUTH_MESSAGE)
    if isinstance(error, boxd.ConflictError) or (
        isinstance(error, boxd.APIStatusError) and error.code == 'invalid_argument'
    ):
        return WorkspaceError(f'{context}: {error}')
    return error


@asynccontextmanager
async def _mapped_errors(
    context: str, *, client: AsyncBoxd | None = None, machine_id: str | None = None
) -> AsyncGenerator[None]:
    """Raise the protocol's typed failure for a boxd SDK error, and let the backend's own failures through.

    A deleted machine can surface as a connection failure, a refusal, or a not-found answer, so with
    `machine_id`, a failure that does not already end the run is followed by one lookup of the machine.
    A file operation's failure inside the machine is already the builtin `OSError` the protocol promises,
    and passes through before a `MachinePermissionError` could be read as refused credentials.
    """
    try:
        yield
    except (WorkspaceError, boxd.MachineFilesystemError):
        raise
    except Exception as error:
        translated = _translate(error, context)
        if client is not None and machine_id is not None and not isinstance(translated, WorkspaceUnavailableError):
            if (gone := await _probe(client, machine_id)) is not None:
                raise gone from error
        if translated is error:
            raise
        raise translated from error


async def _probe(client: AsyncBoxd, machine_id: str) -> WorkspaceUnavailableError | None:
    """`WorkspaceUnavailableError` if boxd deleted or stopped the machine; an inconclusive lookup counts as running."""
    try:
        with anyio.fail_after(_REQUEST_TIMEOUT):
            machine = await client.machines.get(machine_id)
    except boxd.NotFoundError:
        return WorkspaceUnavailableError(_gone_message(machine_id))
    except Exception:
        return None
    if machine.status in ('destroyed', 'failed'):
        return WorkspaceUnavailableError(_gone_message(machine_id))
    if machine.status == 'stopped':
        return WorkspaceUnavailableError(
            f'The boxd machine {machine_id!r} was stopped; a later run that attaches to it starts it again. '
            "Pass `workspace='new'` to start a fresh machine."
        )
    return None


async def _find_named(client: AsyncBoxd, name: str, org: str | None) -> Machine | None:
    """The machine called `name`; `None` if boxd has none, or the lookup failed.

    A lookup by name resolves in the credentials' default organization, so with `org` the
    organization's machines are listed instead.
    """
    try:
        if org is None:
            return await client.machines.get(name)
        return next((m for m in await client.machines.list(org=org) if m.name == name), None)
    except boxd.BoxdError:
        return None


async def _launch(
    client: AsyncBoxd,
    name: str,
    request: Callable[[], Awaitable[Machine]],
    *,
    org: str | None,
    verb: Literal['create', 'fork'],
) -> Machine:
    """Send `request`, which makes a machine called `name` in `org`, and find the machine if the reply is lost.

    A refusal no retry can fix raises `WorkspaceUnavailableError`; connection failures, rate limits,
    and server errors propagate, as they do from every other operation.
    """
    with anyio.move_on_after(_LIFECYCLE_TIMEOUT):
        try:
            return await request()
        except (boxd.APIConnectionError, boxd.ConflictError) as error:
            # The machine may have been made before the reply was lost, by this call or an earlier one.
            if (machine := await _find_named(client, name, org)) is None:
                if isinstance(error, boxd.ConflictError):
                    raise WorkspaceUnavailableError(f'boxd refused to {verb} the machine: {error}') from error
                raise
            return machine
        except boxd.NotFoundError as error:
            raise WorkspaceUnavailableError(f'boxd could not {verb} the machine: {error}') from error
        except Exception as error:
            translated = _translate(error, f'boxd refused to {verb} the machine')
            if translated is error:
                raise
            # boxd refused the request (an invalid setting, a size the organization lacks): no retry can fix it.
            raise WorkspaceUnavailableError(str(translated)) from error
    with anyio.move_on_after(_REQUEST_TIMEOUT):
        if (machine := await _find_named(client, name, org)) is not None:
            return machine
    # A stalled control plane is transient, not a command deadline.
    raise TimeoutError(
        f'boxd did not {verb} the machine within {_LIFECYCLE_TIMEOUT:g}s; the boxd control plane may be unreachable.'
    )


def _command_line(argv: Sequence[str], *, cwd: str | None, env: Mapping[str, str], marker: str) -> str:
    """The line boxd runs under `/bin/sh -c`: `argv` in `cwd`, with `env` added and stdin at EOF.

    `exec` replaces that shell, so a stop reaches the command itself, its exit status is reported as
    is, and a program named like a shell builtin is looked up on `PATH` rather than run as the
    builtin. A missing `cwd` exits 125 with only `marker` on stderr, before the command runs.
    """
    if env:
        # `--` ends `env`'s options, so a name starting with `-` is not read as one.
        argv = ['env', '--', *(f'{name}={value}' for name, value in env.items()), *argv]
    prefix = '' if cwd is None else f'cd -- {shlex.quote(cwd)} 2>/dev/null || {{ printf %s {marker} >&2; exit 125; }}; '
    return f'{prefix}exec {shlex.join(argv)} </dev/null'


def _file_entry(entry: FsEntry) -> FileEntry:
    """The protocol's view of a boxd entry: a symlink as what it points to, with a size only for a regular file."""
    return FileEntry(name=entry.name, path=entry.path, is_dir=entry.is_dir, size=entry.size if entry.is_file else None)


class _Output:
    """A command's stdout and stderr so far, bounded by `_MAX_OUTPUT_BYTES` across both streams."""

    def __init__(self) -> None:
        self.stdout = bytearray()
        self.stderr = bytearray()

    def add(self, data: bytes, *, is_stderr: bool) -> None:
        (self.stderr if is_stderr else self.stdout).extend(data)
        if len(self.stdout) + len(self.stderr) > _MAX_OUTPUT_BYTES:
            raise WorkspaceOutputLimitError(
                _OUTPUT_LIMIT_MESSAGE,
                limit=_MAX_OUTPUT_BYTES,
                stdout=self.stdout[:_OUTPUT_PREVIEW_BYTES].decode(errors='replace'),
                stderr=self.stderr[:_OUTPUT_PREVIEW_BYTES].decode(errors='replace'),
            )

    def text(self) -> tuple[str, str]:
        return self.stdout.decode(errors='replace'), self.stderr.decode(errors='replace')


class BoxdSandboxBackend(WorkspaceBackend, SupportsCommands, SupportsFilesystem, SupportsRealpath):
    """A [boxd](https://boxd.sh) machine as a Pydantic AI [`WorkspaceBackend`][pydantic_ai.workspaces.WorkspaceBackend].

    Commands run inside an isolated boxd microVM, and file operations use boxd's file API as the same
    user.

    Building one does no I/O. The first operation creates an isolated machine named `pydantic-ai-`
    and 32 hex digits, or attaches to the one `ref` names, waking or starting it as needed;
    `get_sandbox()` returns its `boxd.Machine` record. The backend does not stop or delete the
    machine; that is the application's job.

    A command's output streams back while it runs and is returned once it exits. A shell string
    runs under `/bin/sh -c`; a timeout or cancellation stops the command and its process group.

    A deleted machine, one that is not isolated, an unknown snapshot, and rejected credentials raise
    `WorkspaceUnavailableError`.

    Args:
        client: A `boxd.AsyncBoxd` API client to create or attach with. The caller owns closing it.
            Without one, the backend opens its own from the environment on first use and closes it
            in `aclose()`.
        ref: Identity of an existing machine to attach to on first use; it must be isolated.
        org: boxd organization a newly created machine belongs to; the API key's organization, or
            your default one, when `None`.
        snapshot: boxd snapshot a newly created machine starts from, by name or id; boxd's default
            image when `None`.
        vcpu: Size of a newly created machine: 1, 2, 4, or 8 vCPUs, with 4 GiB of memory each; the
            organization's default size when `None`. It cannot be combined with `snapshot`: a
            machine created from a snapshot keeps the size it was saved at.
        auto_suspend_timeout: Idle network seconds before boxd suspends a newly created machine; `0`
            disables it, and `None` keeps the organization's default. A command wakes a suspended
            machine.
        working_dir: Absolute directory commands start in and relative paths resolve against; the
            machine's own default when `None`.
        env: Environment variables every command gets; a command's own `env` is layered on top.
            Nothing is read from the host environment.
    """

    def __init__(
        self,
        *,
        client: AsyncBoxd | None = None,
        ref: WorkspaceRef | None = None,
        org: str | None = None,
        snapshot: str | None = None,
        vcpu: Literal[1, 2, 4, 8] | None = None,
        auto_suspend_timeout: int | None = None,
        working_dir: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if ref is not None:
            _check_ref(ref)
        if snapshot is not None and vcpu is not None:
            raise ValueError('pass either `snapshot` or `vcpu`, not both: a snapshot keeps the size it was saved at')
        check_env(env)
        self._ref = ref
        self._client = client
        self._owns_client = client is None
        self._org = org
        self._snapshot = snapshot
        self._vcpu = vcpu
        self._auto_suspend_timeout = auto_suspend_timeout
        self._env = dict(env or {})
        self._working_dir = absolute_path('working_dir', working_dir)
        # Written to stderr, and nothing else, by a command whose working directory is missing.
        self._missing_cwd_marker = f'pydantic-ai-missing-working-dir-{uuid.uuid4().hex}'
        self._create_name: str | None = None
        self._ready = False
        # `working_dir()` must return a canonical absolute path, resolved once with `pwd -P`.
        self._resolved_working_dir: str | None = None
        # A machine this backend created still needs its `working_dir` made.
        self._working_dir_pending = False
        # Set once the machine's boxd agent turns out to predate the file API; files then go through the shell.
        self._agent_outdated = False
        self._lock = anyio.Lock()
        # Set while nothing will call `aclose()`: the owned client then closes when the last
        # operation in flight ends.
        self._close_after_operation = False
        self._operations = 0

    @property
    def ref(self) -> WorkspaceRef | None:
        """Identity of the machine, or `None` before one has been created.

        For a machine this backend created, the `id` is boxd's machine id.
        """
        return self._ref

    async def get_sandbox(self) -> Machine:
        """Return the machine's current `boxd.Machine` record, creating or attaching to it on first use.

        Use its `id` with your own `boxd.AsyncBoxd` for boxd features the workspace API does not
        cover, such as snapshots, checkpoints, and public URLs. Attaching by `ref` to a machine
        that no longer exists raises `WorkspaceUnavailableError`; it does not create a replacement.
        """
        async with self._operation():
            client, machine_id = await self._acquire()
            async with _mapped_errors('Could not read the boxd machine', client=client, machine_id=machine_id):
                return await client.machines.get(machine_id)

    async def fork(self) -> WorkspaceRef:
        """Copy the machine into a new isolated machine and return the copy's ref, creating or attaching on first use.

        The copy has the machine's disk and memory as they are now, running processes included, and the
        two change independently after. Pass its ref as `workspace=` to work in the copy. The copy is not
        tied to this backend, so delete it with `BoxdSandbox().destroy(ref)` when you are done.

        A machine that is not isolated, or that is on a named boxd network, raises
        `WorkspaceUnavailableError`, as attaching to it does: the copy would keep that access.
        """
        async with self._operation():
            client, machine_id = await self._acquire()
            async with _mapped_errors('Could not fork the boxd machine', client=client, machine_id=machine_id):
                source = await client.machines.get(machine_id)
            # Checked again: a machine can be put on a network after it was attached.
            if (refusal := _refusal(source)) is not None:
                raise WorkspaceUnavailableError(refusal)
            if source.status != 'running':
                # boxd forks a suspended machine from its disk alone, without its memory or unflushed writes.
                await self._wait_until_ready(client, machine_id)
            name = f'{_NAME_PREFIX}{uuid.uuid4().hex}'
            org = None if source.org is None else source.org.name

            async def make_copy() -> Machine:
                forked = await _launch(
                    client, name, lambda: client.machines.fork(machine_id, name, isolated=True), org=org, verb='fork'
                )
                # Logged here, so the copy's id is recorded even if the caller was cancelled meanwhile.
                logger.info('Forked boxd machine %s into %s (%s)', machine_id, forked.id, forked.name)
                return forked

            forked = await run_to_completion(make_copy)
        return WorkspaceRef(provider='boxd', id=forked.id)

    async def _acquire(self) -> tuple[AsyncBoxd, str]:
        """The API client and the id of a machine that is ready for commands, creating or attaching on first use."""
        _require_asyncio()
        async with self._lock:
            if self._client is None:
                self._client = _new_client()
            client = self._client
            if (ref := self._ref) is None:
                # Creation finishes even if the caller is cancelled, so a billed machine is never left unrecorded.
                ref = await run_to_completion(lambda: self._create(client))
            if not self._ready:
                machine_id = await self._wait_until_ready(client, ref.id)
                if machine_id != ref.id:
                    # A ref may name the machine; names can be reused once it is deleted, ids cannot.
                    self._ref = ref = WorkspaceRef(provider='boxd', id=machine_id)
                self._ready = True
            if self._working_dir_pending:
                # A caller cancelled during creation stops here; the next operation makes the directory.
                await anyio.lowlevel.checkpoint_if_cancelled()
                await self._make_working_dir(client, ref.id)
        await anyio.lowlevel.checkpoint_if_cancelled()
        return client, ref.id

    async def _create(self, client: AsyncBoxd) -> WorkspaceRef:
        """Create an isolated machine under a name kept across retries, so a lost reply is found rather than duplicated."""
        if self._create_name is None:
            self._create_name = f'{_NAME_PREFIX}{uuid.uuid4().hex}'
        name = self._create_name
        machine = await _launch(
            client,
            name,
            lambda: client.machines.create(
                name,
                org=self._org,
                from_snapshot=self._snapshot,
                isolated=True,
                vcpu=self._vcpu,
                auto_suspend_timeout=self._auto_suspend_timeout,
            ),
            org=self._org,
            verb='create',
        )
        return self._record_created(machine)

    def _record_created(self, machine: Machine) -> WorkspaceRef:
        self._ref = ref = WorkspaceRef(provider='boxd', id=machine.id)
        self._working_dir_pending = self._working_dir is not None
        # Logged at creation, so the id can be found whatever ends the run before it is stored.
        logger.info('Created boxd machine %s (%s)', machine.id, machine.name)
        return ref

    async def _wait_until_ready(self, client: AsyncBoxd, machine_id: str) -> str:
        """Wait until the machine accepts commands, starting it if it is stopped, and return its id.

        A suspended or hibernated machine is woken by the probe command itself. A machine that is not
        isolated, or that is on a named network, is refused: other machines could reach it, and a ref
        from tampered message history must not point the agent at one of them.
        """
        with anyio.move_on_after(_LIFECYCLE_TIMEOUT):
            async with _mapped_errors(f'Could not attach to boxd machine {machine_id!r}'):
                while True:
                    try:
                        machine = await client.machines.get(machine_id)
                    except boxd.NotFoundError as error:
                        raise WorkspaceUnavailableError(
                            f'The boxd machine {machine_id!r} was not found (it was deleted, or never existed '
                            "for these credentials). Pass `workspace='new'` to start a fresh machine."
                        ) from error
                    if (refusal := _refusal(machine)) is not None:
                        raise WorkspaceUnavailableError(refusal)
                    if machine.status in ('destroyed', 'failed'):
                        raise WorkspaceUnavailableError(_gone_message(machine_id))
                    try:
                        if machine.status == 'stopped':
                            await client.machines.start(machine.id)
                        elif machine.status in ('running', 'suspended', 'hibernated'):
                            await client.machines.exec(machine.id, 'true', timeout=_REQUEST_TIMEOUT)
                            return machine.id
                    except boxd.BoxdError as error:
                        if _credentials_rejected(error) or isinstance(error, boxd.RateLimitError):
                            raise
                        # Still booting, started by someone else meanwhile, or deleted: the next lookup tells.
                    await anyio.sleep(_POLL_INTERVAL)
        raise TimeoutError(f'boxd machine {machine_id!r} did not become ready within {_LIFECYCLE_TIMEOUT:g}s.')

    async def _make_working_dir(self, client: AsyncBoxd, machine_id: str) -> None:
        directory = self._working_dir
        assert directory is not None
        with anyio.CancelScope(shield=True):
            async with _mapped_errors(
                f'Could not create working_dir {directory!r}', client=client, machine_id=machine_id
            ):
                result = await client.machines.exec(
                    machine_id, shlex.join(['mkdir', '-p', '--', directory]), timeout=_REQUEST_TIMEOUT
                )
            if result.exit_code != 0:
                stderr = result.stderr if isinstance(result.stderr, str) else result.stderr.decode(errors='replace')
                raise WorkspaceError(
                    f'Could not create working_dir {directory!r} in boxd machine {machine_id}: {stderr.strip()}'
                )
            self._working_dir_pending = False

    async def aclose(self) -> None:
        """Close the `AsyncBoxd` API client this backend opened for itself.

        A client passed as `client=` is left alone, and the machine keeps running. A later operation
        opens a new client and attaches by `ref`. `BoxdSandbox` calls this when the run that used the
        backend ends.
        """
        if self._client is None or not self._owns_client or not running_on_asyncio():
            return
        if self._operations:
            # An operation in flight, such as a cancelled command still being stopped, needs the
            # client: the last one to finish closes it.
            self._close_after_operation = True
            return

        async def close() -> None:
            # Under the lock, so a machine still being created is recorded before its client closes.
            async with self._lock:
                client = self._detach_client()
            if client is not None:
                await _close_quietly(client)

        await run_to_completion(close)

    def _detach_client(self) -> AsyncBoxd | None:
        """Forget the owned client and what was learned through it; a later operation attaches by `ref` again."""
        client, self._client = self._client, None
        self._ready = False
        self._resolved_working_dir = None
        # A later connection may find the machine rebooted with a newer agent.
        self._agent_outdated = False
        return client

    @asynccontextmanager
    async def _operation(self) -> AsyncGenerator[None]:
        self._operations += 1
        try:
            yield
        finally:
            self._operations -= 1
            if self._operations == 0 and self._close_after_operation and self._owns_client and self._client is not None:
                # Detached before any await, so an operation starting meanwhile opens its own client.
                client = self._detach_client()
                assert client is not None
                await run_to_completion(lambda: _close_quietly(client))

    async def working_dir(self) -> str:
        """Return the canonical directory commands start in, probed once per connection."""
        async with self._operation():
            if self._resolved_working_dir is None:
                result = await self.run(['pwd', '-P'], timeout=_REQUEST_TIMEOUT)
                printed = result.stdout.removesuffix('\n')
                if result.exit_code != 0 or not posixpath.isabs(printed):
                    raise WorkspaceError(
                        'Could not determine the working directory of the boxd machine: '
                        f'`pwd -P` exited {result.exit_code} and printed {result.stdout!r}.'
                    )
                self._resolved_working_dir = printed
            return self._resolved_working_dir

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        async with self._operation():
            check_timeout(timeout)
            argv = command_argv(command, shell)
            check_env(env)
            # Acquiring the machine has its own bound; the timeout is the command's alone.
            client, machine_id = await self._acquire()
            line = _command_line(
                argv, cwd=self._working_dir, env={**self._env, **(env or {})}, marker=self._missing_cwd_marker
            )
            output = _Output()
            stream = client.machines.stream_exec(machine_id, command=line, close_stdin=True)
            async with command_deadline(timeout, stop=stream.close, output=output.text):
                async with _mapped_errors('Could not run the command', client=client, machine_id=machine_id):
                    try:
                        # Leaving the block cancels the call, which stops the command on timeout,
                        # cancellation, or an output overflow.
                        async with stream:
                            async for chunk in stream.iter_chunks():
                                output.add(chunk.data, is_stderr=chunk.is_stderr)
                    except boxd.APIStatusError as error:
                        # The call has no deadline of its own, so this is boxd's limit on a silent command.
                        if error.code != 'deadline_exceeded':
                            raise
                        stdout, stderr = output.text()
                        raise WorkspaceTimeoutError(_SILENT_COMMAND_MESSAGE, stdout=stdout, stderr=stderr) from error
            exit_code = stream.exit_code
            if exit_code is None:
                raise WorkspaceError(f'boxd ended the command in machine {machine_id} without an exit status.')
            if exit_code == 125 and not output.stdout and bytes(output.stderr) == self._missing_cwd_marker.encode():
                raise WorkspaceError(
                    f'working_dir {self._working_dir!r} does not exist in boxd machine {machine_id}. '
                    'Create it there, or pass a working_dir that exists.'
                )
            stdout, stderr = output.text()
            return CommandResult(exit_code=exit_code, stdout=stdout, stderr=stderr)

    async def read_bytes(self, path: str) -> bytes:
        return await self._files(
            f'Could not read {path!r}',
            lambda client, machine_id: client.machines.fs.read(machine_id, path),
            lambda shell: shell.read_bytes(path),
        )

    async def write_bytes(self, path: str, data: bytes) -> None:
        async def write(client: AsyncBoxd, machine_id: str) -> None:
            await client.machines.fs.write(machine_id, path, data)

        await self._files(f'Could not write {path!r}', write, lambda shell: shell.write_bytes(path, data))

    async def stat(self, path: str) -> FileEntry:
        async def stat(client: AsyncBoxd, machine_id: str) -> FileEntry | None:
            entry = await client.machines.fs.stat(machine_id, path)
            # boxd describes a dangling or looping symlink instead of refusing it; the protocol calls it missing.
            return None if entry.target_type is None else _file_entry(entry)

        entry = await self._files(f'Could not stat {path!r}', stat, lambda shell: shell.stat(path))
        if entry is None:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
        return entry

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        async def list_dir(client: AsyncBoxd, machine_id: str) -> Sequence[FileEntry]:
            entries = await client.machines.fs.list(machine_id, path)
            return sorted((_file_entry(entry) for entry in entries), key=lambda entry: entry.name)

        return await self._files(f'Could not list {path!r}', list_dir, lambda shell: shell.list_dir(path))

    async def make_dir(self, path: str) -> None:
        await self._files(
            f'Could not create directory {path!r}',
            lambda client, machine_id: client.machines.fs.mkdir(machine_id, path),
            lambda shell: shell.make_dir(path),
        )

    async def remove(self, path: str) -> None:
        async with self._operation():
            # The removal is recursive, so refuse the working directory and its ancestors, like the
            # built-in backends. The leaf stays unresolved: removing a link removes only the link.
            root = await self.working_dir()
            target = posixpath.normpath(
                posixpath.join(await self.realpath(posixpath.dirname(path)), posixpath.basename(path))
            )
            if root == target or root.startswith(target.rstrip('/') + '/'):
                raise ValueError('cannot remove the workspace root or its ancestor')
            await self._files(
                f'Could not remove {path!r}',
                lambda client, machine_id: client.machines.fs.remove(machine_id, path, recursive=True),
                lambda shell: shell.remove(path),
            )

    async def exists(self, path: str) -> bool:
        return await self._files(
            f'Could not check {path!r}',
            lambda client, machine_id: client.machines.fs.exists(machine_id, path),
            lambda shell: shell.exists(path),
        )

    async def realpath(self, path: str) -> str:
        return await self._files(
            f'Could not resolve {path!r}',
            lambda client, machine_id: client.machines.fs.realpath(machine_id, path),
            lambda shell: shell.realpath(path),
        )

    async def _files(
        self,
        context: str,
        native: Callable[[AsyncBoxd, str], Awaitable[_T]],
        shell: Callable[[_ShellFilesystem], Awaitable[_T]],
    ) -> _T:
        """Run a file operation with boxd's file API, or as shell commands on a machine whose agent predates it."""
        async with self._operation():
            client, machine_id = await self._acquire()
            if not self._agent_outdated:
                async with _mapped_errors(context, client=client, machine_id=machine_id):
                    try:
                        return await native(client, machine_id)
                    except boxd.MachineAgentOutdatedError:
                        self._agent_outdated = True
                        logger.info(
                            'The boxd agent in machine %s predates the file API, so file operations run as shell '
                            'commands until the machine reboots.',
                            machine_id,
                        )
            return await shell(_ShellFilesystem(self))
