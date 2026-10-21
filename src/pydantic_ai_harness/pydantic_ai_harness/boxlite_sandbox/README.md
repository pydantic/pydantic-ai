# BoxLite Sandbox

Run your agent's commands and file edits in a [BoxLite](https://github.com/boxlite-ai/boxlite) micro-VM on your own machine, isolated from your files and processes, without a cloud account.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/boxlite_sandbox/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Install

uv:

```bash
uv add "pydantic-ai-harness[boxlite,anthropic]"
```

pip:

```bash
pip install "pydantic-ai-harness[boxlite,anthropic]"
```

The `boxlite` extra installs [BoxLite's Python SDK](https://github.com/boxlite-ai/boxlite/tree/main/sdks/python), which embeds the BoxLite runtime: there is no daemon to run. The `anthropic` extra is there because the examples use an Anthropic model; swap it for your model provider's extra, and set `ANTHROPIC_API_KEY` for the examples.

BoxLite ships wheels for macOS 14 or newer on Apple silicon and for Linux on x86_64 and aarch64 (glibc 2.28 or newer), for Python up to 3.14. On Linux, the user running the agent needs read and write access to `/dev/kvm`. See [Platforms](#platforms).

## Quick start

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.boxlite_sandbox import BoxLiteSandbox
from pydantic_ai_harness.coder import Coder

agent = Agent('anthropic:claude-opus-5-5', capabilities=[BoxLiteSandbox(image='python:3.12-slim'), Coder()])
result = agent.run_sync('Write a script that prints the first 10 primes, and run it.')
```

`Coder`'s shell and file tools now run in a VM with its own kernel and filesystem, not on your machine. The box boots on the run's first use of the workspace; with `Coder`, `RepoContext` uses it when the run starts, so use `Coder(repo_context=False)` to boot only when a tool needs it.

An `image` is any OCI image; BoxLite pulls it on this machine the first time. Without one, the box runs BoxLite's default, Alpine Linux, which has BusyBox but no Python, git, or ripgrep. Commands run as the image's user, which is `root` in both, and start in `/workspace`, which is created if it's missing.

A new box has no network access. Pass `allow_internet_access=True` to let the agent install packages or clone repositories, and `allowed_hosts` to let it reach only the hosts it needs; see [Network access](#network-access).

## Continue in the same box

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.boxlite_sandbox import BoxLiteSandbox
from pydantic_ai_harness.coder import Coder

agent = Agent('anthropic:claude-opus-5-5', capabilities=[BoxLiteSandbox(image='python:3.12-slim'), Coder()])
result = agent.run_sync('Write a script that prints the first 10 primes, and run it.')

followup = agent.run_sync('Now make it print the first 100.', message_history=result.all_messages())
```

The follow-up run finds the box in the message history and works in it, so the script is still there. Without the history, a run creates a new box. If the box has been deleted, the run raises `WorkspaceUnavailableError` instead of starting over in an empty one.

## Choose the tools

For a narrower agent, use [`Shell`](https://pydantic.dev/docs/ai/harness/shell/) and [`FileSystem`](https://pydantic.dev/docs/ai/harness/filesystem/) instead of `Coder`, or write your own tool:

```python {names="defined"}
from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.boxlite_sandbox import BoxLiteSandbox
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.shell import Shell

agent = Agent('anthropic:claude-opus-5-5', capabilities=[BoxLiteSandbox(image='python:3.12-slim'), Shell(), FileSystem()])


@agent.tool
async def run_python(ctx: RunContext, code: str) -> str:
    """Run a Python snippet in the box."""
    result = await ctx.workspace.run(['python', '-c', code], timeout=10)
    return result.stdout + result.stderr
```

See [Workspaces](https://pydantic.dev/docs/ai/core-concepts/workspace/) for more.

## Network access

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.boxlite_sandbox import BoxLiteSandbox
from pydantic_ai_harness.coder import Coder

sandbox = BoxLiteSandbox(
    image='python:3.12-slim',
    allow_internet_access=True,
    allowed_hosts=['pypi.org', 'files.pythonhosted.org'],
)
agent = Agent('anthropic:claude-opus-5-5', capabilities=[sandbox, Coder()])
result = agent.run_sync('Install requests and check that it imports.')
```

With `allow_internet_access=True` alone, the box can connect anywhere this machine can, including your local network and services and a cloud metadata endpoint, not only the internet. `allowed_hosts` limits it to the hostnames, IP addresses and CIDR ranges you list; see BoxLite's [network options](https://github.com/boxlite-ai/boxlite/tree/main/sdks/python#boxliteboxoptions) for how it matches them.

## Reattach later

To come back to the box without the message history, store the run's workspace ref and pass it back as `workspace=`:

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.boxlite_sandbox import BoxLiteSandbox
from pydantic_ai_harness.coder import Coder

agent = Agent('anthropic:claude-opus-5-5', capabilities=[BoxLiteSandbox(image='python:3.12-slim'), Coder()])

result = agent.run_sync('Write a script that prints the first 10 primes, and run it.')
ref = result.workspace.ref  # store this, e.g. in your database

later = agent.run_sync('Now make it print the first 100.', workspace=ref)
```

Pass `workspace='new'` to create a fresh box even when the message history names one. `image`, `cpus`, `memory_mb`, the network settings and `box_options` only shape a new box, so a box created with network access keeps it when you reattach; `working_dir` and `env` apply to every command.

A box stops when nothing in this process uses it any more, or when the process exits, and keeps its disk. The next use of its ref boots it again, in this process or in a later one. Only one process at a time can use a BoxLite home directory (`~/.boxlite`, or `BOXLITE_HOME`): another process that tries gets `WorkspaceUnavailableError` until the first one exits.

Already have a `boxlite.Box` from `boxlite.Boxlite.default()`? Pass `workspace=BoxLiteSandboxBackend(sandbox=box)` to the run, with `BoxLiteSandboxBackend` from `pydantic_ai_harness.boxlite_sandbox`, or `BoxLiteSandboxBackend(ref=WorkspaceRef(provider='boxlite', id=...))` with the box's name, or its id if it has none. `BoxLiteSandbox`'s settings don't apply to it; pass `working_dir=` and `env=` to the backend. The backend never deletes the box. See [Which boxes a run can use](#which-boxes-a-run-can-use) for what each way accepts.

## Other BoxLite options

`box_options` passes any other [`boxlite.BoxOptions`](https://github.com/boxlite-ai/boxlite/tree/main/sdks/python#boxliteboxoptions) argument to a new box. This one mounts the current directory into the box read-only:

```python {names="defined"}
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai_harness.boxlite_sandbox import BoxLiteSandbox
from pydantic_ai_harness.coder import Coder

sandbox = BoxLiteSandbox(
    image='python:3.12-slim',
    box_options={'volumes': [(str(Path.cwd()), '/project', True)]},
)
agent = Agent('anthropic:claude-opus-5-5', capabilities=[sandbox, Coder()])
result = agent.run_sync('Summarize what the code in /project does.')
```

Other common ones are `ports` to reach a server the agent starts, `secrets` for BoxLite to add credentials to the box's HTTPS requests to the hosts you name, `disk_size_gb`, and `user`. `BoxLiteSandbox`'s own settings own `image`, `cpus`, `memory_mib`, `network`, `working_dir`, `env` and `auto_delete` (and its deprecated `auto_remove`), so passing one of those in `box_options` raises `UserError`.

Runtime-wide settings, such as private registry logins or BoxLite's home directory, go to `boxlite.Boxlite.init_default(boxlite.Options(...))`, called before the first box is used.

## Prepare a box before the run

To seed files or install packages before the agent starts, create the backend yourself, work in it through `Workspace`, and pass it to the run:

```python {names="defined"}
import asyncio

from pydantic_ai import Agent
from pydantic_ai.workspaces import Workspace
from pydantic_ai_harness.boxlite_sandbox import BoxLiteSandbox, BoxLiteSandboxBackend
from pydantic_ai_harness.coder import Coder

agent = Agent('anthropic:claude-opus-5-5', capabilities=[BoxLiteSandbox(), Coder()])


async def main() -> None:
    backend = BoxLiteSandboxBackend(
        image='python:3.12-slim',
        allow_internet_access=True,
        allowed_hosts=['pypi.org', 'files.pythonhosted.org'],
    )
    try:
        workspace = Workspace(backend)
        await workspace.write_text('calc.py', 'def add(a, b):\n    return a - b\n')
        install = await workspace.run(['pip', 'install', 'pytest'], timeout=300)
        if install.exit_code != 0:
            raise RuntimeError(f'pip install failed: {install.stderr}')
        result = await agent.run('Fix the bug in calc.py.', workspace=backend)
        print(result.output)
    finally:
        ref = backend.ref  # set once the box exists
        if ref is not None:
            await BoxLiteSandbox().destroy(ref)


if __name__ == '__main__':
    asyncio.run(main())
```

The `finally` deletes the box even when setup or the run fails. To keep it instead, store `backend.ref` and pass it as `workspace=` to reattach later. `BoxLiteSandbox`'s settings don't apply to a backend you pass as `workspace=`; give the backend its own. `backend.get_sandbox()` returns the SDK's `boxlite.Box`, for BoxLite features the workspace API doesn't cover, such as `copy_in`, `snapshot` or `stop`.

## Clean up

A run never deletes its box, and a box keeps its disk in BoxLite's home directory until it is deleted. Delete it with the ref you stored:

```python {names="defined"}
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.boxlite_sandbox import BoxLiteSandbox


async def delete_box(ref: WorkspaceRef) -> None:
    await BoxLiteSandbox().destroy(ref)
```

Deleting a box that is already gone returns quietly. Once started, a deletion finishes even if its caller is cancelled. `BoxLiteSandbox.backend(ref)` builds an attached backend without booting anything.

Nothing deletes a box for you. Creating one logs its id (see [Telemetry](#telemetry)), so you can find one left behind without a stored ref, for example by a durable execution worker that crashed after creating it, with `await boxlite.Boxlite.default().list_info()`.

`backend.ref` is set as soon as the box starts being created, so the `finally` in [Prepare a box before the run](#prepare-a-box-before-the-run) deletes the box whatever ends the run. A run cancelled before its box has booted deletes that box itself, before the cancellation ends the run. A failed run returns no result, so there is no ref to store: delete the box in an `on_run_error` hook, as shown for [E2B](https://pydantic.dev/docs/ai/harness/e2b-sandbox/#clean-up), or create the backend yourself as in [Prepare a box before the run](#prepare-a-box-before-the-run).

## How commands run

Every command runs through the box's `/bin/sh`, in a session of its own when the image has `setsid`, as BusyBox and util-linux images do. A missing program gives exit code 127, and a program killed by a signal 128 plus its number. A `shell=True` string runs under `/bin/sh -c`. Custom images need a POSIX `/bin/sh`.

On a timeout or cancellation, the backend kills the command and the processes it started, and returns the output so far on `WorkspaceTimeoutError`. Jobs that outlive the command that started them, like a [Shell](https://pydantic.dev/docs/ai/harness/shell/) background job, keep running; the command returns at most two seconds after it exits, even if such a job still holds its output open. In an image without `setsid`, only the command itself is killed, and a program named like a shell builtin, such as `echo`, runs the builtin.

If the combined output passes 10 MiB, the backend stops the command and raises `WorkspaceOutputLimitError`, with the first 65,536 characters of each stream in `stdout` and `stderr`; redirect large output to a file instead. Output BoxLite fails to deliver whole raises `WorkspaceError` rather than come back cut short. The backend's own `mkdir -p` and `pwd -P` time out after 60 seconds, so a broken image can't hang the run.

File operations run as shell commands in the box, 64 KiB per command (see [Writing a backend](https://pydantic.dev/docs/ai/core-concepts/workspace/#writing-a-backend)), so images need the usual file utilities, which Alpine's BusyBox has.

A deleted box raises `WorkspaceUnavailableError`, and so does a command that a stop or a deletion ended; the next command boots a stopped box again. A `working_dir` that was removed raises `WorkspaceError`. Other BoxLite failures propagate as `boxlite.BoxliteError`, for durable execution engines to retry. The BoxLite SDK returns asyncio futures, so the backend needs the asyncio event loop: under Trio, operations raise `UserError`.

## Security model

The VM is the boundary: it has its own kernel and filesystem, and unless `box_options` adds `volumes` or `ports`, nothing from this machine is mounted into it, no ports are published, and none of your environment variables reach it. Inside, commands run as the image's user, `root` on most images, so anything the agent runs can change anything in the box.

- The BoxLite runtime runs in your process with your OS user's permissions, and keeps its state in its home directory.
- `image` is pulled by tag; pin a digest, such as `python:3.12-slim@sha256:...`, to get the same image every time.
- `cpus` and `memory_mb` cap a box; nothing limits how many boxes a process creates.

### Which boxes a run can use

Whoever passes a ref runs commands in the box it names. BoxLite can open any box on this machine by name, including boxes other tools made with your folders mounted. So what's accepted depends on where the box comes from:

| Where the box comes from | What's accepted |
| --- | --- |
| `BoxLiteSandbox`: a ref in the message history, `workspace=ref`, `backend(ref)` or `destroy(ref)` | Only boxes `BoxLiteSandbox` created, named `pydantic-ai-` and 32 hex digits. Any other name raises `ValueError` before anything runs. |
| `BoxLiteSandboxBackend(sandbox=box)` or `BoxLiteSandboxBackend(ref=...)`, built by your code | Any box of this process's default BoxLite runtime, including one with mounts or published ports, because your code chose it. A box from another runtime raises `UserError`: configure the default one with `boxlite.Boxlite.init_default(...)` instead. |

- A run in a box you passed in writes that box's name into the message history. A later run given only that history raises `ValueError`; it doesn't open the box or create a new one. Pass the backend again, or `workspace='new'` for a fresh box.
- Durable execution rebuilds the workspace from its ref, so it only works with boxes `BoxLiteSandbox` created.
- The name doesn't prove where a box came from: any program running as your OS user can create a box with such a name. Only pass refs your application stored itself, and don't build `BoxLiteSandboxBackend(ref=...)` from user input.

## Configuration

| Option | What it does |
| --- | --- |
| `image` | OCI image for a new box. Default: `None`, BoxLite's default, Alpine Linux. |
| `cpus` | vCPUs for a new box. Default: `None`, BoxLite's default. |
| `memory_mb` | Memory for a new box in MiB. Default: `None`, BoxLite's default. |
| `allow_internet_access` | Whether a new box has outbound network access, from this machine's network. Default: `False`. |
| `allowed_hosts` | With `allow_internet_access`, the only hostnames, IP addresses and CIDR ranges a new box can connect to. Default: `None`, any. |
| `working_dir` | Absolute directory commands start in and relative paths resolve against, created if missing. Default: `/workspace`; `None` uses the box's own (`/` on Alpine). |
| `env` | Environment variables every command gets, on top of the image's. Default: `None`. Nothing from your machine's environment reaches the box, and `env` is kept out of the capability's `repr`, but the agent's commands can read it. |
| `box_options` | Other `boxlite.BoxOptions` arguments for a new box, such as `volumes`, `ports`, `secrets` or `disk_size_gb`. Default: `None`. Kept out of the capability's `repr`. |

## Platforms

BoxLite runs VMs with Apple's Hypervisor framework on macOS and with KVM on Linux, so it runs on macOS 14 or newer on Apple silicon and on Linux on x86_64 and aarch64 with glibc 2.28 or newer. On other platforms, and on Python 3.15, the `boxlite` extra installs nothing and importing `pydantic_ai_harness.boxlite_sandbox` raises `ImportError`. On Linux, check `ls -l /dev/kvm` before the first run: a host that can't run VMs raises `WorkspaceUnavailableError` when the run first uses the box. On Ubuntu 24.04, AppArmor stops the bubblewrap that BoxLite runs its VM engine under from creating user namespaces, so boxes fail to start until you install `bubblewrap` and the AppArmor profile from BoxLite's [troubleshooting guide](https://github.com/boxlite-ai/boxlite/blob/main/docs/guides/troubleshooting.md); newer Ubuntu releases ship that profile.

## Telemetry

`BoxLiteSandbox` emits no spans of its own. Core's [instrumentation](https://pydantic.dev/docs/ai/capabilities/instrumentation/) records the box on the agent run span as `pydantic_ai.workspace.provider` (`boxlite`) and `pydantic_ai.workspace.id`, and each command or file operation a tool makes runs inside that tool call's span. Creating a box also logs `Creating BoxLite box <id>` at INFO on the `pydantic_ai_harness.boxlite_sandbox` logger before the creation starts, so the id is on record even if the run ends before it is stored.

## API reference

::: pydantic_ai_harness.boxlite_sandbox.BoxLiteSandbox

::: pydantic_ai_harness.boxlite_sandbox.BoxLiteSandboxBackend
