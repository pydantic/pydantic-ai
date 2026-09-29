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

Inside the sandbox, commands see the host's files read-only, a private `/tmp` and no network, and can write only to the working directory. `/run` is empty, because host daemons such as Docker listen on sockets there, and a read-only mount doesn't stop a connection.

Pass `network=True` to allow the network (the DNS configuration under `/run` comes back with it), and add `bwrap` arguments with `bwrap_args=`: they come after the defaults, so `['--bind', path, path]` makes another directory writable and `['--tmpfs', path]` hides one.

Commands share the host's process list rather than getting their own, so a command started in the background, such as a [Shell](https://pydantic.dev/docs/ai/harness/shell/) background job or a dev server, keeps running after the call that started it, and later calls can check on it or stop it. The cost is that sandboxed commands can see the host's processes and signal the host user's own.

Only commands are sandboxed. File methods such as `write_text` go to the wrapped workspace, so they see the host's `/tmp` rather than the sandbox's, and reach outside the working directory. To limit them too, use [`ReadOnlyWorkspace`](https://pydantic.dev/docs/ai/core-concepts/workspace/#read-only-access) or a [FileSystem](https://pydantic.dev/docs/ai/harness/filesystem/) root. The run's ref is the wrapped workspace's, and its `backend` is the wrapped backend.

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
