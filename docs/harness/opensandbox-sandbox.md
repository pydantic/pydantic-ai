---
title: OpenSandbox
description: "Run a Pydantic AI agent's commands and file edits in a sandbox on your own OpenSandbox server, on Docker or Kubernetes."
---

# OpenSandbox

Run your agent's commands and file edits in a sandbox on your own [OpenSandbox](https://open-sandbox.ai) server, created on first use. The server runs sandboxes as Docker containers on one machine or as pods on Kubernetes; the agent's code is the same either way.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/opensandbox_sandbox/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Install

```bash
pip/uv-add "pydantic-ai-harness[coder,opensandbox,anthropic]"
```

The `opensandbox` extra installs the [OpenSandbox SDK](https://pypi.org/project/opensandbox/). The examples also use `Coder` and an Anthropic model, hence the `coder` and `anthropic` extras and `ANTHROPIC_API_KEY`; swap them for your own tools and model provider.

## Run a server

The capability talks to an OpenSandbox server. To run one on this machine with Docker:

```bash
uvx opensandbox-server init-config ~/.sandbox.toml --example docker
uvx opensandbox-server
```

Then set `execd_image = "opensandbox/execd:release-1.2.0"` under `[runtime]` in `~/.sandbox.toml`. execd runs the commands inside each sandbox, and the capability needs version 1.1.1 or later. With an older one, the first operation raises `WorkspaceUnavailableError`.

The capability reads two environment variables:

- `OPEN_SANDBOX_DOMAIN`, the server's address. Default: `localhost:8080`. Requests use plain HTTP unless it starts with `https://`, such as `https://sandbox.example.com`.
- `OPEN_SANDBOX_API_KEY`, sent if your server requires a key.

For Kubernetes, see [Run on Kubernetes](#run-on-kubernetes).

## Quick start

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.opensandbox_sandbox import OpenSandbox

agent = Agent('{{ anthropic:large }}', capabilities=[OpenSandbox(image='python:3.13-slim'), Coder()])
result = agent.run_sync('Write a script that prints the first 10 primes, and run it.')
```

`Coder`'s shell and file tools now run in the sandbox. For a narrower agent, use [`Shell`](shell.md), [`FileSystem`](filesystem.md) or your own tools on [`ctx.workspace`](../workspace.md#using-the-workspace-in-tools).

- `Coder` creates it when the run starts, to load the repository's instructions, even if no tool is called. Pass `Coder(repo_context=False)` to wait for the first tool call.
- It keeps running after the run until you destroy it. If you set `sandbox_timeout`, the server also deletes it after that many seconds unused. See [Clean up](#clean-up).
- Commands run as the image's user, usually `root`, in `/workspace`, which is created if missing. The image needs a POSIX `sh` and the usual file utilities.
- Its network access is whatever the server configures. Pass `allow_internet_access=False` to block all outbound connections. See [Network access](#network-access).

## Continue in the same sandbox

<span id="reattach-later"></span>

Pass the message history, and the next run works in the same sandbox:

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.opensandbox_sandbox import OpenSandbox

agent = Agent('{{ anthropic:large }}', capabilities=[OpenSandbox(image='python:3.13-slim'), Coder()])
result = agent.run_sync('Write a script that prints the first 10 primes, and run it.')

followup = agent.run_sync('Now make it print the first 100.', message_history=result.all_messages())
ref = result.workspace.ref  # or store this, and pass `workspace=ref` to a later run
```

`result.workspace.ref` is a `WorkspaceRef` holding the sandbox's id. Store it to continue later or to destroy the sandbox.

If the sandbox has been destroyed, the run raises `WorkspaceUnavailableError`. Pass `workspace='new'` to start over. [Workspaces](../workspace.md#continuing-in-the-same-workspace) covers how runs find their workspace.

Whoever passes a ref runs commands in that sandbox, and the server serves every sandbox to every holder of its API key. Only pass refs your application stored itself.

## Prepare a sandbox before the run

To seed files or install packages before the agent starts, create the backend yourself, work in it through `Workspace`, and pass it to the run:

```python {names="defined"}
import asyncio

from pydantic_ai import Agent
from pydantic_ai.workspaces import Workspace
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.opensandbox_sandbox import OpenSandbox, OpenSandboxBackend

agent = Agent('{{ anthropic:large }}', capabilities=[OpenSandbox(image='python:3.13-slim'), Coder()])


async def main() -> None:
    backend = OpenSandboxBackend(image='python:3.13-slim')
    try:
        workspace = Workspace(backend)
        await workspace.write_text('calc.py', 'def add(a, b):\n    return a - b\n')
        install = await workspace.run(['pip', 'install', 'pytest'], timeout=300)
        if install.exit_code != 0:
            raise RuntimeError(f'pip install failed: {install.stderr}')
        result = await agent.run('Fix the bug in calc.py.', workspace=backend)
        print(result.output)
    finally:
        await backend.aclose()
        if backend.ref is not None:
            await OpenSandbox().destroy(backend.ref)


if __name__ == '__main__':
    asyncio.run(main())
```

The `finally` destroys the sandbox even when setup or the run fails. `aclose()` closes the backend's connection to the server. To keep the sandbox, store `backend.ref` instead.

`OpenSandbox`'s settings don't apply to a backend you pass as `workspace=`. Give the backend its own.

## Give the agent a desktop

Pass `desktop=True` and add [`ComputerUse`](computer-use.md):
`capabilities=[OpenSandbox(image='debian:bookworm-slim', desktop=True), ComputerUse()]`. The screen is 1280x800;
pass a size such as `desktop=(1440, 900)` for another.

- When the image's user is root, the first use installs [the display's programs](computer-use.md#what-the-display-needs) that are missing with `apt-get`, which needs the sandbox to reach the package repositories. Passwordless `sudo` works only where the server allows it: the Docker runtime's default `no_new_privileges = true` blocks it. Otherwise build them into the image: without the screen's programs the first use raises `WorkspaceUnavailableError` naming what is missing, and without the desktop's the screen works but stays black.
- On the Docker runtime, set `network_mode = "bridge"` under `[docker]` in the server's configuration; see [Security model](#security-model).
- Use execd 1.2 or later: on older versions every screen action is slow.
- There's no web browser. Install one in the image or before the run, as in [Prepare a sandbox before the run](#prepare-a-sandbox-before-the-run) with `OpenSandboxBackend(..., desktop=True)`, or let the agent install it with `Shell`.
- The agent can type only ASCII: an action that types or presses a character such as `é` fails, and the model is told why.

## Clean up

Destroy a sandbox, files included, with the ref you stored. A sandbox that is already gone returns quietly:

```python {names="defined"}
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.opensandbox_sandbox import OpenSandbox


async def remove_sandbox(ref: WorkspaceRef) -> None:
    await OpenSandbox().destroy(ref)
```

`OpenSandbox()` without an `image` only attaches to and destroys sandboxes by ref.

A failed run returns no result, so destroy its sandbox in an `on_run_error` hook:

```python
from typing import Any

import anyio

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.run import AgentRunResult
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.opensandbox_sandbox import OpenSandbox

hooks = Hooks()


@hooks.on.run_error
async def destroy_failed_run_sandbox(ctx: RunContext[None], *, error: BaseException) -> AgentRunResult[Any]:
    if ctx.workspace.ref is not None:
        # Shielded so a deadline around the run can't cut the destroy short, bounded so a hung call can't hang it.
        with anyio.move_on_after(30, shield=True):
            await OpenSandbox().destroy(ctx.workspace.ref)
    raise error


agent = Agent('{{ anthropic:large }}', capabilities=[OpenSandbox(image='python:3.13-slim'), Coder(), hooks])
```

To have the server clean up for you, set `sandbox_timeout`. The server deletes a sandbox unused for that many seconds, files included. Every operation restarts the countdown, but a single command must finish within it, so set it above your longest command.

### Find a lost sandbox

If your program stops before it stores the ref, for example after a crash or Ctrl-C, the sandbox is left on the server. To find it:

- List the server's sandboxes. Sandboxes Pydantic AI creates have a `pydantic-ai-create` metadata entry.
- Or search your logs for `Created OpenSandbox sandbox <id>`, logged at INFO by the `pydantic_ai_harness` logger.

Then destroy it: `await OpenSandbox().destroy(WorkspaceRef(provider='opensandbox', id='<id>'))`.

## Run on Kubernetes

On a server deployed with OpenSandbox's [Helm charts](https://github.com/opensandbox-group/OpenSandbox/blob/main/docs/deployment/index.md), sandboxes are pods. The server creates them through OpenSandbox's own `BatchSandbox` resources or through [agent-sandbox](https://agent-sandbox.sigs.k8s.io), and both work the same here.

Pods are reachable only inside the cluster. Either deploy the server's ingress gateway, or let the server proxy commands with `use_server_proxy=True`:

```python {names="defined"}
from opensandbox.config import ConnectionConfig

from pydantic_ai_harness.opensandbox_sandbox import OpenSandbox

sandbox = OpenSandbox(
    image='python:3.13-slim',
    connection=ConnectionConfig(domain='opensandbox.internal:8080', use_server_proxy=True),
    resource={'cpu': '500m', 'memory': '1Gi'},
)
```

This connects over plain HTTP inside the cluster. For HTTPS, put the server behind TLS and pass `ConnectionConfig(domain='https://opensandbox.example.com', protocol='https', use_server_proxy=True)`. The SDK uses the domain's scheme for the server's API and `protocol` for command and file requests, so a config needs both.

Each sandbox asks for one CPU and 2 GiB unless you pass `resource`. If its pod can't be scheduled within 30 seconds, the create times out, but the server may still finish it. The next operation then attaches to that sandbox instead of creating another.

## Network access

`allow_internet_access=False` asks the server to create the sandbox with a policy that blocks all outbound connections. The default, `True`, sends no policy, so the server's own configuration applies. Attaching never changes a sandbox's policy.

The server enforces the policy with its egress sidecar, a container next to the sandbox. In the default `dns` mode it filters DNS lookups only, so a command can still connect to an IP address directly. To block both, configure the server with:

```toml
[egress]
image = "opensandbox/egress:v1.1.7"
mode = "dns+nft"
```

[OpenSandbox egress configuration](https://github.com/opensandbox-group/OpenSandbox/blob/main/server/configuration.md#egress) covers runtime requirements and server-level allow and deny rules.

## Keep a token out of the sandbox

OpenSandbox's [Credential Vault](https://github.com/opensandbox-group/OpenSandbox/blob/main/docs/guides/credential-vault.md) holds a token outside the sandbox and adds it to the sandbox's outgoing requests to the hosts you name, so the token is never inside the sandbox.

Turn it on by passing `credential_proxy` and `network_policy` to `OpenSandboxBackend`. Both take the SDK's own objects and pass them on unchanged. In an image with `git` and `gh`, both then work as the token's owner, while `echo $GH_TOKEN` in the sandbox prints `placeholder`:

```python {test="skip"}
import base64

from opensandbox.models.sandboxes import (
    Credential,
    CredentialAuth,
    CredentialBinding,
    CredentialMatch,
    CredentialProxyConfig,
    InlineCredentialSource,
    NetworkPolicy,
    NetworkRule,
)

from pydantic_ai_harness.opensandbox_sandbox import OpenSandboxBackend

token = '...'  # a GitHub token, such as the output of `gh auth token`
basic = base64.b64encode(f'x-access-token:{token}'.encode()).decode()
hosts = ['github.com', 'api.github.com', 'uploads.github.com']


async def github_sandbox() -> OpenSandboxBackend:
    backend = OpenSandboxBackend(
        image='python:3.13-slim',
        env={'GH_TOKEN': 'placeholder'},  # `gh` sends a token only when one is set; the vault replaces it
        network_policy=NetworkPolicy(
            defaultAction='deny', egress=[NetworkRule(action='allow', target=host) for host in hosts]
        ),
        credential_proxy=CredentialProxyConfig(enabled=True),
    )
    sandbox = await backend.get_sandbox()
    await sandbox.credential_vault.create(
        credentials=[
            Credential(name='github-basic', source=InlineCredentialSource(value=basic)),
            Credential(name='github-token', source=InlineCredentialSource(value=token)),
        ],
        bindings=[
            CredentialBinding(  # git over HTTPS
                name='git',
                match=CredentialMatch(hosts=['github.com']),
                auth=CredentialAuth(type='basic', credential='github-basic'),
            ),
            CredentialBinding(  # gh, including release uploads
                name='gh',
                match=CredentialMatch(hosts=['api.github.com', 'uploads.github.com']),
                auth=CredentialAuth(type='bearer', credential='github-token'),
            ),
        ],
    )
    return backend
```

Pass the backend as `workspace=`, as in [Prepare a sandbox before the run](#prepare-a-sandbox-before-the-run). `credential_proxy` is on `OpenSandboxBackend` only, because the vault has to be filled after the sandbox exists and before the agent's first command.

Before you rely on it:

- The server must filter traffic in `dns+nft` mode (see [Network access](#network-access)), or creating the sandbox fails.
- The policy must allow every host in the vault, and any other host the agent needs, such as a package index.
- Only HTTPS requests get the token, replacing any `Authorization` header they carried. Use `https://` remotes, not `git@github.com:owner/repo.git`.
- The agent can't see the token, but it can use it for anything the token allows.
- The vault doesn't hide the token in responses. List only hosts that never send request headers back, or a command could read the token there.
- The vault is kept in memory next to the sandbox. Pausing and resuming a Kubernetes sandbox, or restarting the server's network proxy, empties it, so fill it again after you reattach.
- To swap the token, call `credential_vault.patch(credentials={'replace': [...]})` on `await backend.get_sandbox()`.
- `allow_internet_access=False` already sets a policy that blocks everything, so pass it or `network_policy`, not both. For both, `OpenSandbox` raises `UserError` and `OpenSandboxBackend` raises `ValueError`.

## Mount volumes into the sandbox

A deleted sandbox takes its files with it. To keep a workspace's files after you delete its sandbox, pass `volumes` to `OpenSandboxBackend`: the SDK's own `Volume` objects, for a host directory, a named volume (a Kubernetes PVC or a Docker named volume) or an OSSFS bucket, passed to `Sandbox.create` unchanged. Mount the volume at the working directory and a later sandbox created with the same volume starts with the files the last one left:

```python {test="skip"}
from opensandbox.models.sandboxes import PVC, Volume

backend = OpenSandboxBackend(
    image='python:3.13-slim',
    working_dir='/workspace',
    volumes=[Volume(name='workspace', pvc=PVC(claimName='project-42'), mountPath='/workspace')],
)
```

Your application picks the volume for each workspace and deletes it when it's done. `volumes` only shapes a new sandbox: attaching to a ref leaves its mounts as they are.

## Endpoint authentication

`secure_access=True` asks the server to turn on [Secure Access](https://github.com/opensandbox-group/OpenSandbox/blob/main/docs/guides/secure-access.md) for a new sandbox. The gateway and the server proxy then reject requests to the sandbox's endpoints that lack its token. The SDK fetches the token with the server API key and sends it with every command and file request.

- It needs Kubernetes with the server's ingress gateway: `[ingress]` with `mode = "gateway"`. Docker rejects it.
- Requests that reach the pod without passing through the gateway or proxy, such as from elsewhere in the cluster, are not checked.

## Security model

The sandbox is as isolated as the server's runtime makes it. A Docker container shares the host's kernel, and a Kubernetes pod shares its node's, unless the server is configured with a secure runtime such as gVisor or Kata Containers. Commands run as the image's user, usually `root`.

- With the default `allow_internet_access=True`, a Docker-backed sandbox can reach its host and LAN, including the OpenSandbox server if it listens there. `False` on a server in `dns+nft` mode blocks that.
- Requests to the server's API carry the server API key. Command and file requests carry commands, file contents and `env` values. Over plain `http`, all of them are unencrypted. A `ConnectionConfig` you pass is used as given, so set both an `https://` domain and `protocol='https'` on it.
- The server API key, set as `api_key` under `[server]`, controls every sandbox on that server and can fetch their Secure Access tokens. Keep it out of the sandbox's `env`.
- Without the server proxy, command and file requests go straight to the sandbox's endpoint, without the server API key. The endpoint checks only what the server configures for it, such as Secure Access. With `use_server_proxy=True`, they go through the server, which checks the API key.
- With `desktop` on the Docker runtime, the default `network_mode = "host"` puts every sandbox on the host's network. The desktop doesn't start if the host or another sandbox already runs a second X server (a program that draws a screen) on display `:1`. One that starts there later can take the agent's clicks and screenshots instead of its own screen. Set `network_mode = "bridge"` under `[docker]` in the server's configuration to give each sandbox its own network.
- `image` is pulled by tag; pin a digest, such as `python:3.13-slim@sha256:...`, to get the same image every time.

## Configuration

[Agent specs](../agent.md#agent-specs) can set every option except `connection` and `network_policy`. A spec reaches the server through `OPEN_SANDBOX_DOMAIN` and `OPEN_SANDBOX_API_KEY`.

| Option | What it does |
| --- | --- |
| `image` | Image for a new sandbox, such as `'python:3.13-slim'`. Default: `None`, which only attaches to and destroys sandboxes by ref. |
| `connection` | The SDK's `ConnectionConfig` for your server. Default: `None`, which reads `OPEN_SANDBOX_DOMAIN` and `OPEN_SANDBOX_API_KEY`. A config you pass is used as given. |
| `resource` | Resource limits for a new sandbox, such as `{'cpu': '500m', 'memory': '1Gi'}`. Default: `None`, one CPU and 2 GiB. |
| `allow_internet_access` | Whether a new sandbox may connect out. Default: `True`, which sends no policy, so the server's configuration applies. `False` asks the server to block all outbound connections. |
| `network_policy` | The SDK's `NetworkPolicy` for a new sandbox, passed to `Sandbox.create` unchanged. Default: `None`. Cannot be combined with `allow_internet_access=False`. A spec cannot set it. |
| `secure_access` | Whether a new sandbox's endpoints require a token at the server's Kubernetes ingress gateway and proxy. Default: `False`. Docker rejects `True`. |
| `sandbox_timeout` | Seconds a sandbox may go unused before the server deletes it, files included. Every operation restarts the countdown, and a single command must finish within it. Default: `None`, kept until destroyed. |
| `working_dir` | Absolute directory commands start in and relative paths resolve against, created on first use if missing. Default: `/workspace`. `None` uses the image's own. |
| `env` | Environment variables every command gets. Default: `None`. Nothing from your machine's environment reaches the sandbox. The agent's commands can read these, so leave out secrets it shouldn't see. |
| `desktop` | Give the sandbox a graphical display for [`ComputerUse`](computer-use.md): `True` for 1280x800, or its `(width, height)`. The first use installs the screen's programs with `apt-get` when the image's user is root; passwordless `sudo` works only where the server allows it. The size applies when the display first starts. Default: `False`. |
| `authority` | Who wrote the sandbox's files: `'none'` for files nobody on your side wrote, such as a cloned third-party repository, or `'user'` for the user's own. Capabilities that put a file into the prompt give it no more [authority](../workspace-custom.md#workspace-authority) than this. Default: `'operator'`, files you wrote. |
| `path_authority` | An `authority` level for specific directories and files, overriding `authority` under them. The longest matching path wins. Paths are sandbox paths, relative to the working directory or absolute, and `~` isn't expanded. Default: `None`. |

`image`, `resource`, `allow_internet_access`, `network_policy` and `secure_access` only shape a new sandbox. `working_dir`, `env`, `sandbox_timeout`, `desktop`, `authority` and `path_authority` also apply when you reattach.

## Limits

<span id="how-commands-run"></span>

- Each command takes about a fifth of a second. File operations run as commands, so a large file takes several seconds per MiB. Keep large data out of the workspace, or fetch it from inside the sandbox.
- Creating the sandbox, including pulling the image, isn't bounded by a command's `timeout`.
- On a timeout or cancellation, the command and the processes it started are stopped.
- The OpenSandbox SDK needs asyncio. Under Trio, the capability raises `UserError`.

## Durable execution

Add a durability capability next to `OpenSandbox(image=...)`. [Workspaces](../workspace.md#durable-execution) covers what changes. Then:

- Configure the same capability and server connection on every worker.
- Set `sandbox_timeout` above the longest gap between workspace operations, including workflow waits and retries. Nothing restarts the countdown while the workflow is idle.
- A retried attempt can create a second sandbox if the first attempt's ref wasn't recorded yet. To avoid that, create the sandbox and store its ref before starting the workflow, then pass that ref as `workspace=`.

## Telemetry

`OpenSandbox` traces creating or reconnecting to the sandbox as `workspace create` or `workspace attach`, under the call that needed it, and `destroy(ref)` called inside a run as `workspace destroy`; see [workspace spans](../logfire.md#workspace-spans). Core's [instrumentation](../capabilities/instrumentation.md) also records the sandbox on the agent run span as `pydantic_ai.workspace.provider` (`opensandbox`) and `pydantic_ai.workspace.id`. Each command or file operation a tool makes runs inside that tool call's span. Creating a sandbox logs its ID at `INFO` (`Created OpenSandbox sandbox <id>`) on the `pydantic_ai_harness.opensandbox_sandbox._backend` logger.

The OpenSandbox SDK reports how long each create took to your server's `/metrics/events`. Set `OPENSANDBOX_DISABLE_METRICS=1`, or pass `ConnectionConfig(disable_metrics=True)`, to turn that off.

## API reference

::: pydantic_ai_harness.opensandbox_sandbox.OpenSandbox

::: pydantic_ai_harness.opensandbox_sandbox.OpenSandboxBackend
