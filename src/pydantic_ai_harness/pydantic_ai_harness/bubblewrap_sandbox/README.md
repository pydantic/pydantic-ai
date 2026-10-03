# Bubblewrap Sandbox

Run your agent's commands in a Linux [bubblewrap](https://github.com/containers/bubblewrap) (`bwrap`) sandbox on the host they already run on, over SSH or locally.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/bubblewrap_sandbox/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Install

uv:

```bash
uv add "pydantic-ai-harness[anthropic]"
```

pip:

```bash
pip install "pydantic-ai-harness[anthropic]"
```

The sandbox's host must run Linux with `bwrap` installed (the `bubblewrap` package) and user namespaces allowed; otherwise commands raise `WorkspaceUnavailableError`. The `anthropic` extra is there because the examples use an Anthropic model; swap it for your model provider's extra.

## Quick start

`BubblewrapSandbox` wraps another workspace capability and runs its commands in a sandbox on that workspace's host. Wrap [`SSHWorkspace`](https://pydantic.dev/docs/ai/harness/ssh-workspace/) to sandbox commands on a remote host, or `LocalWorkspace` to sandbox them on this one:

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai_harness import BubblewrapSandbox, Coder, SSHWorkspace

agent = Agent(
    'anthropic:claude-opus-5-5',
    capabilities=[BubblewrapSandbox(SSHWorkspace('dev@build-box', working_dir='/srv/app')), Coder()],
)
result = agent.run_sync('Run the test suite and fix the first failure.')
```

## What the sandbox allows

Inside the sandbox, commands see the host's files read-only, a private `/tmp` and no network, and can write only to the working directory. `/run` is empty, because host daemons such as Docker listen on sockets there, and a read-only mount doesn't stop a connection. `bwrap` must be able to give the sandbox its own user namespace, and the sandbox has no capabilities.

When the working directory contains the host account's home directory, as an `SSHWorkspace` without a `working_dir` does, the files the host runs or reads when the next SSH connection logs in, before the command starts, are mounted read-only: `~/.ssh`, `~/.pam_environment`, and the startup files a login shell runs for a command (`~/.bashrc`, `~/.zshenv`, `~/.cshrc`, `~/.tcshrc` and `~/.config/fish`). Missing ones are created first, empty. If one of them is a symlink, the sandbox doesn't start, because a command could replace the link.

Without the network, a seccomp filter blocks the same system calls as the [Codex CLI](https://github.com/openai/codex)'s Linux sandbox does without network access: commands can't connect to, serve or accept on any socket, including Unix sockets the host has elsewhere on disk and the sandbox's own loopback, and can't use `ptrace` or `io_uring`. Commands can still create stream socket pairs, which language runtimes use between their own processes. Unlike Codex, Unix datagram sockets are blocked too, because a datagram can be addressed to any socket file without connecting first. So a test suite that starts a local server needs `network=True`, and so does Python's `forkserver` start method for `multiprocessing` (the default on Linux since Python 3.14), which listens on a socket: `multiprocessing.get_context('spawn')` or `'fork'` work without it. The filter covers x86_64 and aarch64 hosts; a 32-bit program in the sandbox is stopped when it makes a system call.

Pass `network=True` to share the host's network: commands can reach it, including services on the host's loopback and anything the host can reach, and can listen on the host's ports. The DNS configuration under `/run` comes back with it, and there is no seccomp filter. Add `bwrap` arguments with `bwrap_args=`: they come after the defaults, so `['--bind', path, path]` makes another directory writable and `['--tmpfs', path]` hides one, such as `~/.ssh`. The program that starts `bwrap`, and `base64` when the filter loads, is taken from a `PATH` directory outside the working directory. When the working directory contains the host account's `~/.ssh`, the sandbox bind-mounts it read-only, creating the directory if it is missing, because OpenSSH runs `~/.ssh/rc` before the next connection. That check reads an absolute `$HOME` with `cd -P` and `pwd -P`, and creates a missing directory with `/bin/mkdir`, so neither step searches `PATH`. Without an absolute home the sandbox does not start. `bwrap_args` come after that mount and can still replace it.

Commands share the host's process list rather than getting their own, so a command started in the background, such as a [Shell](https://pydantic.dev/docs/ai/harness/shell/) background job or a dev server, keeps running after the call that started it, and after the agent run, until something stops it. Later calls can check on it or stop it. The cost is that sandboxed commands can see the host's processes and their command lines, and signal the host user's own. A call's `env` values are `bwrap --setenv` arguments, so they show up in that process list while the command runs.

File methods such as `write_text`, and so the [FileSystem](https://pydantic.dev/docs/ai/harness/filesystem/) and [Coder](https://pydantic.dev/docs/ai/harness/coder/) file tools, run in the sandbox too, as shell commands, so they see what commands see: they can't write outside the working directory, even through a symlink a command swapped in after a path check. Only when the wrapped workspace is read-only, and so runs no commands, are its files read from the host directly. The run's ref is the wrapped workspace's, and its `backend` is the wrapped backend.

### What it doesn't protect

The sandbox keeps an agent from changing the host outside its working directory. It is not a boundary against an agent trying to harm the host user:

- Commands can read every file the host user can, such as SSH keys and cloud credentials, and show them to the model. Hide those directories with `bwrap_args=['--tmpfs', path]`.
- Commands can signal, and so stop, the host user's other processes.
- Anything a command writes in the working directory, such as Git hooks, a `Makefile` or an `.envrc`, runs unsandboxed if you later run it outside the sandbox.
- With the home directory writable, so are the files only an interactive login runs, such as `~/.profile` and `~/.bash_profile`, and other per-user configuration. Give the sandbox a project directory as its `working_dir` instead.
- The wrapped workspace's own settings, `bwrap_args`, and the host's `bwrap` outside the working directory are trusted.

For an agent you don't trust, give it its own user on the host, or use a cloud sandbox such as [E2B](https://pydantic.dev/docs/ai/harness/e2b-sandbox/).

## Use the workspace directly

The capability builds a `BubblewrapWorkspace`, a [`WrapperWorkspace`](https://pydantic.dev/docs/ai/core-concepts/workspace/#read-only-access) you can also use directly, around any workspace:

```python {test="skip"}
from pydantic_ai.workspaces import CommandResult, Workspace
from pydantic_ai_harness import BubblewrapWorkspace, SSHWorkspaceBackend


async def run_tests() -> CommandResult:
    workspace = BubblewrapWorkspace(Workspace(SSHWorkspaceBackend('dev@build-box')))
    # `bwrap` runs `make test` on build-box
    return await workspace.run(['make', 'test'])
```

## Telemetry

`BubblewrapSandbox` emits no spans of its own. The run's workspace is the wrapped one's, so core's [instrumentation](https://pydantic.dev/docs/ai/capabilities/instrumentation/) records its `pydantic_ai.workspace.provider` and `pydantic_ai.workspace.id` on the agent run span, and each sandboxed command runs inside its tool call's span.

## API reference

::: pydantic_ai_harness.bubblewrap_sandbox.BubblewrapSandbox

::: pydantic_ai_harness.bubblewrap_sandbox.BubblewrapWorkspace
