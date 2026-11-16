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

Set `execd_image = "opensandbox/execd:release-1.2.0"` under `[runtime]` in `~/.sandbox.toml`: the capability needs execd 1.1.1 or later, and with an older one the first operation raises `WorkspaceUnavailableError`.

The capability finds the server through `OPEN_SANDBOX_DOMAIN` (default `localhost:8080`), and sends `OPEN_SANDBOX_API_KEY` if your server requires a key. Requests use plain HTTP unless the domain starts with `https://`, such as `OPEN_SANDBOX_DOMAIN=https://sandbox.example.com`. For Kubernetes, see [Run on Kubernetes](#run-on-kubernetes).

## Quick start

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.opensandbox_sandbox import OpenSandbox

agent = Agent('{{ anthropic:large }}', capabilities=[OpenSandbox(image='python:3.13-slim'), Coder()])
result = agent.run_sync('Write a script that prints the first 10 primes, and run it.')
```

`Coder`'s shell and file tools now run in the sandbox. For a narrower agent, use [`Shell`](shell.md), [`FileSystem`](filesystem.md) or your own tools on [`ctx.workspace`](../workspace.md#using-the-workspace-in-tools).

- **It keeps running after the run** until you destroy it, or until `sandbox_timeout` passes with no use. See [Clean up](#clean-up).
- **Commands run as the image's user, usually `root`, in `/workspace`**, which is created if missing. The image needs a POSIX `sh` and the usual file utilities.
- **Its network access is whatever the server configures.** Pass `allow_internet_access=False` for deny-all egress; see [Network access](#network-access).

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

If the sandbox has been destroyed, the run raises `WorkspaceUnavailableError`; pass `workspace='new'` to start over. [Workspaces](../workspace.md#continuing-in-the-same-workspace) covers how runs find their workspace. `image`, `resource`, `allow_internet_access` and `secure_access` only shape a new sandbox; `working_dir`, `env` and `sandbox_timeout` also apply when you reattach.

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

The `finally` destroys the sandbox even when setup or the run fails, and `aclose()` closes the backend's connection to the server. To keep the sandbox, store `backend.ref` instead. `OpenSandbox`'s settings don't apply to a backend you pass as `workspace=`; give the backend its own.

## Clean up

Destroy a sandbox, files included, with the ref you stored. A sandbox that is already gone returns quietly:

```python {names="defined"}
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.opensandbox_sandbox import OpenSandbox


async def remove_sandbox(ref: WorkspaceRef) -> None:
    await OpenSandbox().destroy(ref)
```

`OpenSandbox()` without an `image` only attaches to and destroys sandboxes by ref. A failed run returns no result, so destroy its sandbox in an `on_run_error` hook, as shown for [E2B](e2b-sandbox.md#clean-up).

To have the server clean up for you, set `sandbox_timeout`: a sandbox unused for that many seconds is deleted, files included. Every operation renews it, but a single command must finish within it, so set it above your longest command.

Every sandbox the capability creates carries a `pydantic-ai-create` metadata entry, and its id is logged as `Created OpenSandbox sandbox <id>` at INFO on the `pydantic_ai_harness.opensandbox_sandbox._backend` logger, so one left without a stored ref can be found by listing the server's sandboxes.

## Run on Kubernetes

On a server deployed with OpenSandbox's [Helm charts](https://github.com/opensandbox-group/OpenSandbox/blob/main/docs/deployment/index.md), sandboxes are pods, created through OpenSandbox's own `BatchSandbox` resources or through [agent-sandbox](https://agent-sandbox.sigs.k8s.io); both work the same here. Pods are reachable only inside the cluster, so either deploy the server's ingress gateway, or let the server proxy commands:

```python {names="defined"}
from opensandbox.config import ConnectionConfig

from pydantic_ai_harness.opensandbox_sandbox import OpenSandbox

sandbox = OpenSandbox(
    image='python:3.13-slim',
    connection=ConnectionConfig(domain='opensandbox.internal:8080', use_server_proxy=True),
    resource={'cpu': '500m', 'memory': '1Gi'},
)
```

This connects over plain HTTP inside the cluster. For HTTPS, put the server behind TLS and pass `ConnectionConfig(domain='https://opensandbox.example.com', protocol='https', use_server_proxy=True)`: the SDK uses the domain's scheme for the server's API and `protocol` for command and file requests, so a config needs both.

Each sandbox asks for one CPU and 2 GiB unless you pass `resource`. A pod that can't be scheduled makes the create time out after 30 seconds while the server may still finish it; the backend finds that sandbox by a tag and attaches to it on the next operation.

## Network access

`allow_internet_access=False` asks the server to create the sandbox with a deny-all egress policy. The default, `True`, sends no policy, so the server's own configuration applies. Attaching never changes a sandbox's policy.

The server enforces the policy with its egress sidecar. In the default `dns` mode it filters DNS lookups only, so a command can still connect to an IP address directly. To block both, configure the server with:

```toml
[egress]
image = "opensandbox/egress:v1.1.7"
mode = "dns+nft"
```

See [OpenSandbox egress configuration](https://github.com/opensandbox-group/OpenSandbox/blob/main/server/configuration.md#egress) for runtime requirements and server-level allow and deny rules.

## Endpoint authentication

`secure_access=True` asks the server to turn on [Secure Access](https://github.com/opensandbox-group/OpenSandbox/blob/main/docs/guides/secure-access.md) for a new sandbox. It needs Kubernetes with the server's ingress gateway (`[ingress]` with `mode = "gateway"`); Docker rejects it.

With it on, the gateway and the server proxy reject requests to the sandbox's endpoints that lack its token. The SDK fetches the token with the server API key and sends it with every command and file request. Requests that reach the pod without passing through the gateway or proxy, such as from elsewhere in the cluster, are not checked.

## Security model

The sandbox's isolation is the server's runtime: a Docker container shares the host's kernel, and a Kubernetes pod shares its node's, unless the server is configured with a secure runtime such as gVisor or Kata Containers. Commands run as the image's user, usually `root`.

- With the default `allow_internet_access=True`, a Docker-backed sandbox can reach its host and LAN, including the OpenSandbox server if it listens there. `False` on a server in `dns+nft` mode blocks that.
- Requests to the server's API carry the server API key; command and file requests carry commands, file contents and `env` values. Over plain `http` all of them are unencrypted. A `ConnectionConfig` you pass is used as given, so set both an `https://` domain and `protocol='https'` on it.
- The server API key, set as `api_key` under `[server]`, controls every sandbox on that server and can fetch their Secure Access tokens. Keep it out of the sandbox's `env`.
- Without the server proxy, command and file requests go straight to the sandbox's endpoint, without the server API key, and the endpoint checks only what the server configures for it, such as Secure Access. With `use_server_proxy=True`, they go through the server, which checks the API key.
- `image` is pulled by tag; pin a digest, such as `python:3.13-slim@sha256:...`, to get the same image every time.

## Configuration

[Agent specs](../agent.md#agent-specs) can set every option except `connection`; a spec reaches the server through `OPEN_SANDBOX_DOMAIN` and `OPEN_SANDBOX_API_KEY`.

| Option | What it does |
| --- | --- |
| `image` | Image for a new sandbox, such as `'python:3.13-slim'`. Default: `None`, which only attaches to and destroys sandboxes by ref. |
| `connection` | The SDK's `ConnectionConfig` for your server. Default: `None`, which reads `OPEN_SANDBOX_DOMAIN` and `OPEN_SANDBOX_API_KEY`. A config you pass is used as given. |
| `resource` | Resource limits for a new sandbox, such as `{'cpu': '500m', 'memory': '1Gi'}`. Default: `None`, one CPU and 2 GiB. |
| `allow_internet_access` | Default: `True`, which sends no policy, so the server's configuration applies. `False` asks the server for deny-all egress on a new sandbox. |
| `secure_access` | Default: `False`. `True` asks the server to require a token for a new sandbox's endpoints at its Kubernetes ingress gateway and proxy. Docker rejects `True`. |
| `sandbox_timeout` | Seconds a sandbox may go unused before the server deletes it, files included; every operation renews it, and a single command must finish within it. Default: `None`, kept until destroyed. |
| `working_dir` | Absolute directory commands start in and relative paths resolve against, created on first use if missing. Default: `/workspace`; `None` uses the image's own. |
| `env` | Environment variables every command gets. Default: `None`. Nothing from your machine's environment reaches the sandbox. The agent's commands can read these, so leave out secrets it shouldn't see. |

## Limits

<span id="how-commands-run"></span>

- Each command takes about a fifth of a second, and file operations run as commands, so a large file takes several seconds per MiB. Keep large data out of the workspace or fetch it from inside the sandbox.
- Creating the sandbox, including pulling the image, isn't bounded by a command's `timeout`.
- On a timeout or cancellation, the command and the processes it started are stopped.
- The OpenSandbox SDK needs asyncio; under Trio, the capability raises `UserError`.

## Durable execution

Add a durability capability next to `OpenSandbox(image=...)`; [Workspaces](../workspace.md#durable-execution) covers what changes. Configure the same capability and server connection on every worker. Set `sandbox_timeout` above the longest gap between workspace operations, including workflow waits and retries, since nothing renews it while the workflow is idle. To avoid a second sandbox when an attempt is retried before its ref is recorded, create the sandbox and store its ref before starting the workflow, then pass that ref as `workspace=`.

## Telemetry

`OpenSandbox` emits no spans of its own. Core's [instrumentation](../capabilities/instrumentation.md) records `pydantic_ai.workspace.provider` (`opensandbox`) and `pydantic_ai.workspace.id` on the agent run span, and each command or file operation runs inside its tool call's span.

The OpenSandbox SDK reports how long each create took to your server's `/metrics/events`. Set `OPENSANDBOX_DISABLE_METRICS=1`, or pass `ConnectionConfig(disable_metrics=True)`, to turn that off.

## API reference

::: pydantic_ai_harness.opensandbox_sandbox.OpenSandbox

::: pydantic_ai_harness.opensandbox_sandbox.OpenSandboxBackend
