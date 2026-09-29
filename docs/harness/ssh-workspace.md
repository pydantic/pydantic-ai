---
title: SSH Workspace
description: "Run a Pydantic AI agent's commands and file edits on another machine over SSH, with your own ssh client and configuration."
---

# SSH Workspace

Run your agent's commands and file edits on another machine over SSH, using your own `ssh` client and its configuration.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/ssh_workspace/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Install

```bash
pip/uv-add "pydantic-ai-harness[anthropic]"
```

No extra is needed for SSH itself: the capability runs the `ssh` client already on your machine. The `anthropic` extra is there because the examples use an Anthropic model; swap it for your model provider's extra.

## Quick start

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai_harness import Coder, SSHWorkspace

agent = Agent(
    'anthropic:claude-opus-5-5',
    capabilities=[SSHWorkspace('dev@build-box', working_dir='/srv/app', env={'UV_OFFLINE': '1'}), Coder()],
)
result = agent.run_sync('Run the test suite and fix the first failure.')
```

`Coder`'s shell and file tools now run on `build-box`, as the user you log in as, with that user's full authority on the host. To confine the commands, wrap the capability in [`BubblewrapSandbox`](bubblewrap-sandbox.md).

`working_dir` defaults to the login directory, and a relative one starts there. The directory must already exist. Commands get the remote login environment plus `env=`. For a single run, pass `workspace=SSHWorkspaceBackend('dev@build-box')` to `agent.run` instead.

## Logging in

Logging in needs a key; passwords aren't supported. `ssh` runs with `BatchMode=yes`, so a host that asks for a password or a key passphrase fails right away instead of hanging the run, and `ssh_args=` can't turn prompts back on. Use a key without a passphrase, such as `ssh_args=['-i', key_path]` or `IdentityFile` in your SSH configuration, or load a key that has one into `ssh-agent` with `ssh-add`. Set the user in the destination (`dev@build-box`), with `User` in your SSH configuration, or with `ssh_args=['-l', 'dev']`. Your SSH configuration (`~/.ssh/config`) also supplies ports and jump hosts.

## How commands run

Every operation opens an SSH connection, and file operations run as shell commands on the host, so the host needs a POSIX `sh` and the usual file utilities. Turn on connection sharing in your SSH configuration (`ControlMaster auto` with a `ControlPersist` time) to make them fast.

A host that can't be reached, a missing working directory, or a connection lost mid-command raises `WorkspaceUnavailableError`. The exit code is the command's own, even when it is 255, which `ssh` also uses for its own errors.

On a timeout or cancellation, a second connection stops the command's processes on the host; a command that detached into a session of its own, like a [Shell](shell.md) background job, keeps running.

A command that leaves a background process holding its output open, such as `server &` without redirecting the server's output, doesn't return until that process exits, because `sshd` waits for the output to close: redirect it, as in `server > server.log 2>&1 &`.

## Reattach later

The ref is `WorkspaceRef(provider='ssh', id='dev@build-box:/srv/app')`, available from construction. `SSHWorkspace` only accepts a reference to its own host and directory, so a ref in message history can't point it at another machine. `SSHWorkspace(...).backend(ref)` builds the backend for such a ref without connecting, and raises `ValueError` for any other.

## Security

`SSHWorkspace` has the full authority of the remote user. Neither `working_dir` nor a [FileSystem](filesystem.md) root jails shell commands. Do not pass your local environment to the host: choose the variables the command needs with `env=`, which is kept out of the capability's `repr`.

`env=` values travel inside the command that `ssh` sends, so while a command runs they are visible in process listings (`ps`) to other users on both this machine and the host. Don't put secrets in `env=` on a shared machine: keep them in a file only the remote user can read, or in the remote user's login environment.

## Platforms

The machine running the agent must be POSIX (Linux or macOS), like [`LocalWorkspace`](../workspace.md#platforms): constructing `SSHWorkspace` on Windows raises `NotImplementedError`. The host needs a POSIX `sh` and the usual file utilities.

## Telemetry

`SSHWorkspace` emits no spans of its own. Core's [instrumentation](../capabilities/instrumentation.md) records the workspace on the agent run span as `pydantic_ai.workspace.provider` (`ssh`) and `pydantic_ai.workspace.id` (the host and directory, recorded even without `include_content`, because it identifies the environment rather than content), and each command or file operation a tool makes runs inside that tool call's span.

## API reference

::: pydantic_ai_harness.ssh_workspace.SSHWorkspace

::: pydantic_ai_harness.ssh_workspace.SSHWorkspaceBackend
