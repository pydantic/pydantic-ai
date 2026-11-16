# Custom Workspaces

Choose a workspace per user or tenant, add your own policy to one, or connect an environment Pydantic AI
doesn't [support yet](workspace.md#supported-workspaces). Start with [Workspaces](workspace.md) if you
haven't used one.

## Supplying a workspace from a capability

Implement [`get_workspace`][pydantic_ai.capabilities.AbstractCapability.get_workspace] on your own
capability. This one gives each user their own directory, named by the user ID the run gets as `deps`:

```python {title="per_user_workspace.py"}
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.workspaces import LocalWorkspaceBackend, WorkspaceBackend, WorkspaceRef


@dataclass
class UserDirectory(AbstractCapability[str]):
    base_dir: Path

    def get_workspace(self, ctx: RunContext[str], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        if not ctx.deps.isalnum():  # keeps an ID like '../other-user' from leaving base_dir
            raise ValueError(f'Invalid user ID: {ctx.deps!r}')
        backend = LocalWorkspaceBackend(self.base_dir / ctx.deps)
        if ref is not None and ref != backend.ref:
            return None  # not this user's directory
        return backend
```

`LocalWorkspaceBackend` never creates its directory, and raises `WorkspaceUnavailableError` until it
exists, so create each user's directory when you create the user. Like `LocalWorkspace`, it separates where
users start, not what they can reach.

- With `ref=None`, return the backend for a new environment.
- With a `ref` you recognize, return a backend that attaches to it; return `None` for any other. Refs can
  come from stored or untrusted history, so validate the id before using it as a path or identifier.
- To apply a policy, return a `Workspace` around the backend, such as `ReadOnlyWorkspace(Workspace(backend))`.
- `get_workspace` can be called more than once per run, so do no I/O there: connect on the backend's first
  operation.
- A capability that supplies a workspace can't use `defer_loading=True`, because the workspace is chosen
  when the run starts.

A capability that needs a workspace can fail at run start, naming what to attach, instead of on the first
tool call. [`ctx.workspace.attached`][pydantic_ai.workspaces.Workspace.attached] is `False` when nothing
supplied one:

```python
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError


class ProjectNotes(AbstractCapability):
    async def before_run(self, ctx: RunContext) -> None:
        if not ctx.workspace.attached:
            raise UserError("`ProjectNotes` needs a workspace. Attach one, such as `LocalWorkspace('.')`.")
```

## Wrapping a workspace {#wrapping-a-workspace}

Subclass [`WrapperWorkspace`][pydantic_ai.workspaces.WrapperWorkspace], override the operations you want to
change, and call `self.wrapped` for the rest. This one refuses writes outside the working directory,
following symlinks, and refuses commands:

```python
import posixpath

from pydantic_ai.workspaces import WorkspaceReadOnlyError, WrapperWorkspace


class RootedWrites(WrapperWorkspace):
    async def write_bytes(self, path: str, data: bytes) -> None:
        root = await self.wrapped.realpath(await self.working_dir())
        target = await self.wrapped.realpath(await self.resolve(path))
        if posixpath.commonpath((root, target)) != root:
            raise WorkspaceReadOnlyError('outside the allowed root')
        await self.wrapped.write_bytes(path, data)

    async def run(self, command, **kwargs):
        raise WorkspaceReadOnlyError('commands bypass file policy')
```

- `run()` bypasses file-method policies, and so do shell and grep tools built on it, unless the wrapper
  refuses or rewrites commands too. A `run()` override takes and forwards `cwd`.
- The check and the write are separate calls, so a symlink swapped in between is not caught. Use a sandbox
  for untrusted commands, not path checks.
- `ctx.workspace.backend` reaches the provider's own API and bypasses every wrapper, `read_only` included.

## Marking who wrote the files {#workspace-authority}

Set `authority` when the agent works on files the operator didn't write: `'user'` for the user's own files,
such as uploads, and `'none'` for files nobody on your side wrote, such as a cloned third-party repository:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace

agent = Agent('{{ anthropic:medium }}', capabilities=[LocalWorkspace('./checkout', authority='none')])
```

Tools work as before. The setting is for capabilities that put the workspace's files into the prompt: the
model treats instructions as the operator speaking, so such a capability asks
[`ctx.workspace.authority_of(path)`][pydantic_ai.workspaces.Workspace.authority_of] and sends the text with
no more [authority][pydantic_ai.messages.Authority] than that, in the user's turn for `'user'` and as tool
output for `'none'`.

- No built-in capability reads authority yet. It's the setting your own capabilities check.
- Authority doesn't limit what the agent can do; use
  [tool approval](deferred-tools.md#human-in-the-loop-tool-approval) or a sandbox for that.
- It's checked when a file is read. Text already in a conversation keeps its level, so start a new
  conversation after lowering it.
- For one run, pass `workspace=AuthorityWorkspace(workspace, authority=...)`. Wrappers only lower
  authority: with several, the lowest wins.

### Authority by path {#path-authority}

`path_authority` gives directories and files their own level, for a workspace with several authors, such as
the operator's instruction files next to a cloned third-party repository:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace

agent = Agent(
    '{{ anthropic:medium }}',
    capabilities=[
        LocalWorkspace(
            './checkout',
            authority='none',
            path_authority={'~/.config/my-app': 'operator', 'docs/upload': 'user'},
        )
    ],
)
```

- The longest matching path wins.
- Relative paths resolve against the working directory. `LocalWorkspace` expands `~`; `AuthorityWorkspace`
  refuses paths starting with `~`.
- A file outside the working directory is at `'none'`, unless a `path_authority` entry contains it.
- A symlinked file gets the lower of the levels where its path is written and where it leads.
- Paths also match ignoring case, keeping the lower level, so `Vendor/` can't escape a lower `vendor/`.

## Writing a backend

A backend talks to one environment; [`Workspace`][pydantic_ai.workspaces.Workspace] wraps it to give tools the
`ctx.workspace` methods. A [`WorkspaceBackend`][pydantic_ai.workspaces.WorkspaceBackend] implements `ref`
and `working_dir`, plus [`SupportsCommands`][pydantic_ai.workspaces.SupportsCommands],
[`SupportsFilesystem`][pydantic_ai.workspaces.SupportsFilesystem], or both:

- **Commands only:** file operations run as shell commands. They work everywhere a POSIX shell does, and are
  slower than a provider's file API.
- **Filesystem only:** file tools work and `ctx.workspace.run` raises `UserError`. If the storage can hold
  symlinks, also implement `SupportsRealpath`; see [Resolving symlinks](#resolving-symlinks).
- **Both:** they must reach the same environment.

A backend for a cloud sandbox has this shape (`sandbox_sdk` stands in for the provider's SDK):

```python {title="my_sandbox.py" test="skip" lint="skip"}
from collections.abc import Mapping

import anyio
import sandbox_sdk

from pydantic_ai.workspaces import (
    CommandResult,
    SupportsCommands,
    WorkspaceBackend,
    WorkspaceCommand,
    WorkspaceRef,
    WorkspaceUnavailableError,
)


class MySandbox(WorkspaceBackend, SupportsCommands):
    def __init__(self, ref: WorkspaceRef | None = None):
        self._ref = ref
        self._sandbox: sandbox_sdk.Sandbox | None = None
        self._lock = anyio.Lock()

    @property
    def ref(self) -> WorkspaceRef | None:
        return self._ref

    async def _connect(self) -> sandbox_sdk.Sandbox:
        async with self._lock:  # concurrent first operations share one sandbox
            if self._sandbox is None:
                if self._ref is None:
                    self._sandbox = await sandbox_sdk.Sandbox.create()
                    self._ref = WorkspaceRef(provider='my-sandbox', id=self._sandbox.id)
                else:
                    try:
                        self._sandbox = await sandbox_sdk.Sandbox.connect(self._ref.id)
                    except sandbox_sdk.NotFound as error:
                        raise WorkspaceUnavailableError(f'sandbox {self._ref.id!r} no longer exists') from error
            return self._sandbox

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        sandbox = await self._connect()
        result = await sandbox.exec(command, shell=shell, env=env, timeout=timeout)
        return CommandResult(exit_code=result.exit_code, stdout=result.stdout, stderr=result.stderr)

    async def working_dir(self) -> str:
        return (await self._connect()).working_dir
```

Supply it from a capability's [`get_workspace`](#supplying-a-workspace-from-a-capability). A backend must:

- Do no I/O in its constructor. The first operation creates the environment, or attaches to the one `ref` names.
- Keep `ref` `None` until the environment exists, then set it and never change it. Set it as soon as the
  create call returns, before setup that can fail, so a failed setup still leaves a ref to clean up.
- Create one environment when first operations run concurrently. `Workspace` holds no lock, so the backend
  owns its get-or-create lock.
- Keep the ref when the caller is cancelled during creation. The example above skips this; the harness
  sandboxes show one way.
- Raise `WorkspaceUnavailableError` for a `ref` whose environment is gone, and never create a replacement.
- Never destroy the environment when a run ends, and never stop a process a command detached into its own
  session. [`Shell`](harness/shell.md) background jobs rely on that.
- Return a non-zero exit as a result: 127 for a program that isn't found, 126 for one that isn't executable.
  Raise the builtin file error for a bad path and `WorkspaceTimeoutError` for a timeout. Let the provider
  SDK's transient errors propagate, so a durable engine can retry them.

A backend whose shell and paths aren't POSIX says so with
[`SupportsPlatform`][pydantic_ai.workspaces.SupportsPlatform].

### Resolving symlinks

A path check, such as "only under `/app`", needs to know where a path really leads: with a link
`/app/data -> /secrets`, `/app/data/key` looks inside but opens `/secrets/key`.
[`Workspace.realpath`][pydantic_ai.workspaces.Workspace.realpath] answers that. File methods open
[`resolve(path)`][pydantic_ai.workspaces.Workspace.resolve], which collapses `..` as text, so check
`realpath(await ws.resolve(path))`, as [`RootedWrites`](#wrapping-a-workspace) does.

- `LocalWorkspaceBackend` uses `os.path.realpath`.
- A backend with commands but no [`SupportsRealpath`][pydantic_ai.workspaces.SupportsRealpath] follows links
  with `readlink`. If the image has no `readlink`, it raises `WorkspaceError`.
- A backend with neither only normalizes the text. That is exact for storage without symlinks, such as an
  object store; where symlinks exist, path checks such as `FileSystem`'s `root_dir` can be led outside.

### Checking a backend

Subclass [`WorkspaceBackendSuite`][pydantic_ai.workspaces.conformance.WorkspaceBackendSuite] in your pytest
suite and provide its `backend` fixture. Each test checks one rule of the contract above:

```python {test="skip" lint="skip"}
import pytest
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite


class TestMyBackend(WorkspaceBackendSuite):
    @pytest.fixture(scope='class')
    @classmethod
    def backend(cls, tmp_path_factory: pytest.TempPathFactory) -> LocalWorkspaceBackend:
        return LocalWorkspaceBackend(working_dir=tmp_path_factory.mktemp('ws'))
```

- The suite needs the anyio pytest plugin, which runs each rule on asyncio and, when installed, Trio. For an
  asyncio-only SDK, override `anyio_backend` to return `'asyncio'`, class-scoped when `backend` is.
- A class-scoped `backend` starts one environment for the whole suite instead of one per rule.
- Optional fixtures turn on more rules: `fresh_backend` for concurrent first use; `attach_backend`, a
  factory that builds a backend for a ref as your `get_workspace` does, for reattaching and durable
  execution; `destroy_environment` and `destructive_backend` for a destroyed environment. Rules whose
  fixtures are missing are skipped, so run pytest with `-rs` to see what went unchecked.
- Four boolean fixtures, `True` by default, declare a known limit. Return `False` only for a limit you also
  document for users:
    - `can_detect_exit_with_inherited_output_pipes`: the provider reports a command's exit only after every
      process holding its output has exited.
    - `filesystem_honors_shell_permissions`: the provider's file API bypasses the command user's permissions.
    - `has_real_posix_shell` and `enforces_parent_file_errors`: for in-memory test doubles only.
- For a `'windows'` platform, the suite checks commands in PowerShell.

The suite's class, fixtures and rule names are stable in 2.x. Minor releases can add or tighten rules, so
pin `pydantic-ai` if a new rule failing your CI is a problem.
