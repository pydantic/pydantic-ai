# OpenSandbox

Run your agent's commands and file edits in a sandbox on your own [OpenSandbox](https://open-sandbox.ai) server, created on first use. The server runs sandboxes as Docker containers on one machine or as pods on Kubernetes; the agent's code is the same either way.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/opensandbox_sandbox/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Install

uv:

```bash
uv add "pydantic-ai-harness[coder,opensandbox,anthropic]"
```

pip:

```bash
pip install "pydantic-ai-harness[coder,opensandbox,anthropic]"
```

The `opensandbox` extra installs the [OpenSandbox SDK](https://pypi.org/project/opensandbox/), and `coder` installs `Coder`'s dependencies. The `anthropic` extra is there because the examples use an Anthropic model; swap it for your model provider's extra, and set `ANTHROPIC_API_KEY` for the examples.

## Run a server

The capability talks to an OpenSandbox server. To run one on this machine with Docker:

```bash
uvx opensandbox-server init-config ~/.sandbox.toml --example docker
uvx opensandbox-server
```

The capability needs execd 1.1.1 or later, the server's in-sandbox command service: set `execd_image = "opensandbox/execd:release-1.2.0"` under `[runtime]` in `~/.sandbox.toml`. Older versions drop the line breaks from command output, so the first operation raises `WorkspaceUnavailableError` saying so.

The capability finds the server through `OPEN_SANDBOX_DOMAIN` (default `localhost:8080`), and sends `OPEN_SANDBOX_API_KEY` if your server requires a key. For Kubernetes, see [Run on Kubernetes](#run-on-kubernetes).

## Quick start

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.opensandbox_sandbox import OpenSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[OpenSandbox(image='python:3.13-slim'), Coder()])
result = agent.run_sync('Write a script that prints the first 10 primes, and run it.')
```

`Coder`'s shell and file tools now run in the sandbox, not on your machine. The sandbox is created on the run's first use of the workspace. With `Coder`, `RepoContext` uses it when the run starts, so use `Coder(repo_context=False)` to create it only when a tool needs it. The sandbox keeps running after the run ends; see [Clean up](#clean-up).

The image needs a POSIX `sh` and the usual file utilities. Commands run as the image's user, usually `root`, and start in `/workspace`, which is created if it's missing.

## Continue in the same sandbox

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.opensandbox_sandbox import OpenSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[OpenSandbox(image='python:3.13-slim'), Coder()])
result = agent.run_sync('Write a script that prints the first 10 primes, and run it.')

followup = agent.run_sync('Now make it print the first 100.', message_history=result.all_messages())
```

The follow-up run finds the sandbox in the message history and works in it, so the script is still there. Without the history, a run creates a new sandbox. If the sandbox has been destroyed, the run raises `WorkspaceUnavailableError` instead of starting over in an empty one.

## Reattach later

To come back to the sandbox without the message history, store the run's workspace ref and pass it back as `workspace=`:

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.opensandbox_sandbox import OpenSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[OpenSandbox(image='python:3.13-slim'), Coder()])

result = agent.run_sync('Write a script that prints the first 10 primes, and run it.')
ref = result.workspace.ref  # store this, e.g. in your database

later = agent.run_sync('Now make it print the first 100.', workspace=ref)
```

The ref is `WorkspaceRef(provider='opensandbox', id=<sandbox id>)`. Pass `workspace='new'` to create a fresh sandbox even when the message history names one. `image`, `resource`, `allow_internet_access` and `secure_access` only shape a new sandbox; `working_dir`, `env` and `sandbox_timeout` apply to an attached one too.

A ref is trusted input: whoever passes one runs commands in that sandbox, and the server serves every sandbox to every holder of its API key. Only pass refs your application stored itself.

## Prepare a sandbox before the run

To seed files or install packages before the agent starts, create the backend yourself, work in it through `Workspace`, and pass it to the run:

```python {names="defined"}
import asyncio

from pydantic_ai import Agent
from pydantic_ai.workspaces import Workspace
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.opensandbox_sandbox import OpenSandbox, OpenSandboxBackend

agent = Agent('anthropic:claude-opus-5-5', capabilities=[OpenSandbox(image='python:3.13-slim'), Coder()])


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

The `finally` destroys the sandbox even when setup or the run fails, and `aclose()` closes the backend's connection to the server. To keep the sandbox instead, store `backend.ref` and pass it as `workspace=` to reattach later. `OpenSandbox`'s settings don't apply to a backend you pass as `workspace=`; give the backend its own.

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

Each sandbox asks for one CPU and 2 GiB unless you pass `resource`. A pod that can't be scheduled leaves the create waiting; the SDK gives up after its 30-second request timeout while the server may still finish it, so the backend finds the sandbox by a tag it set and keeps its ref, and the next operation attaches to it.

## Endpoint authentication

On Kubernetes with the server's ingress gateway enabled (`[ingress]` with `mode = "gateway"`), pass `OpenSandbox(image='python:3.13-slim', secure_access=True)` to require a per-sandbox token for endpoint access. The SDK obtains and sends the token automatically, including when attaching to a secured sandbox. The setting only applies when creating a sandbox; attaching does not enable or disable its existing protection.

The default is `False`; Docker rejects `secure_access=True`. A management API key alone does not protect direct sandbox endpoints. Restrict unsecured endpoints to trusted callers. See [OpenSandbox Secure Access](https://github.com/opensandbox-group/OpenSandbox/blob/main/docs/guides/secure-access.md) for gateway configuration and how endpoint authentication differs from execd's own access token.

## Network access

`allow_internet_access=True` (the default) leaves network access to the server's configuration. Pass `OpenSandbox(image='python:3.13-slim', allow_internet_access=False)` to request deny-all egress for a new sandbox. It does not change the policy when attaching to an existing sandbox.

Deny-all requires the server's egress sidecar. For packet-level enforcement, including direct IP connections, configure the server with:

```toml
[egress]
image = "opensandbox/egress:v1.1.7"
mode = "dns+nft"
```

The default `dns` mode only filters DNS lookups; it does not block direct IP connections. Server-level allow/deny overrides still apply. See [OpenSandbox egress configuration](https://github.com/opensandbox-group/OpenSandbox/blob/main/server/configuration.md#egress) for runtime requirements and configuration.

## Clean up

A run never destroys its sandbox. Destroy it, files included, with the ref you stored:

```python {names="defined"}
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.opensandbox_sandbox import OpenSandbox


async def remove_sandbox(ref: WorkspaceRef) -> None:
    await OpenSandbox().destroy(ref)
```

Destroying a sandbox that is already gone returns quietly. `OpenSandbox()` without an `image` attaches to and destroys sandboxes by ref, and leaves a run without a ref to the agent's other workspace capabilities.

To have the server clean up for you, set `sandbox_timeout`: a sandbox unused for that many seconds is deleted, files included. Every operation renews it as it starts, so a long run keeps its sandbox, but a single command must finish within it: set it above your longest command. Without it, a sandbox lives until it is destroyed.

A failed run returns no result, so there is no ref to store. Destroy the sandbox in an `on_run_error` hook, as shown for [E2B](https://pydantic.dev/docs/ai/harness/e2b-sandbox/#clean-up), or create the backend yourself as in [Prepare a sandbox before the run](#prepare-a-sandbox-before-the-run). Every sandbox the capability creates carries a `pydantic-ai-create` metadata entry, so one left without a stored ref can be found by listing the server's sandboxes.

## How commands run

On a timeout or cancellation, the command and the processes it started are stopped in the sandbox. Creating the sandbox, including pulling the image, is not bounded by a command's `timeout`.

Each command is a request to the sandbox that takes about a fifth of a second, and file operations run as commands, so reading or writing a large file takes several seconds per MiB. Keep large data out of the workspace or fetch it from inside the sandbox.

A sandbox that is gone, or destroyed mid-command, raises `WorkspaceUnavailableError`. A command whose working directory has been deleted raises `WorkspaceError`. Other errors from the server, such as an image it can't pull, are raised as the SDK reports them.

The OpenSandbox SDK runs on asyncio only, so under Trio the capability raises `UserError`.

## Durable execution

Compose `OpenSandbox(image=...)` with [`TemporalDurability`](https://pydantic.dev/docs/ai/capabilities/durable_execution/temporal/), [`DBOSDurability`](https://pydantic.dev/docs/ai/capabilities/durable_execution/dbos/) or [`PrefectDurability`](https://pydantic.dev/docs/ai/capabilities/durable_execution/prefect/). Durable execution records the initial workspace operation's ref so later operations and replays can reach the same sandbox. Configure the same workspace capability and server connection on every worker.

Temporal's activity-local backends close their SDK-owned HTTP transports after each workspace operation. In-process engines can keep the run-owned transport open until the run ends. A transport passed through `ConnectionConfig` remains yours to close.

Sandbox creation is not exactly-once. If a create response is lost, the backend attempts to recover the ref using sandbox metadata; another operation on that same backend can attach to it. A fresh activity attempt or worker restart before the ref is recorded can create another sandbox. To avoid creation during workflow retries, create the sandbox and persist its ref before starting the workflow, then pass that ref as `workspace=`, not a live backend.

Set `sandbox_timeout` above the longest gap between workspace operations, including workflow waits and retries. There is no background renewal while the workflow is idle; without a timeout, clean up sandboxes left without a stored ref as described in [Clean up](#clean-up).

## Security model

The sandbox's isolation is the server's runtime: a Docker container shares the host's kernel, and a Kubernetes pod shares its node's, unless the server is configured with a secure runtime such as gVisor or Kata Containers. Inside, commands run as the image's user, usually `root`.

- `allow_internet_access=False` requests deny-all egress for new sandboxes. Enforcement depends on the server's egress configuration; see [Network access](#network-access).
- With network access enabled, a Docker-backed sandbox can reach services on its host and LAN, including the OpenSandbox API if it is exposed there. To restrict that access, use `allow_internet_access=False` with `dns+nft`, or a server-side network policy that blocks those destinations.
- Set `api_key` under `[server]` and authenticate clients with `OPEN_SANDBOX_API_KEY`. Keep the key out of the sandbox's `env`: it grants access to every sandbox on that server.
- The management API key does not protect direct sandbox endpoints. Use `secure_access=True` with a Kubernetes ingress gateway before exposing them to untrusted clients; Docker requires separate endpoint restrictions. See [Endpoint authentication](#endpoint-authentication).
- `env` values are sent to the server with every command.
- `image` is pulled by tag; pin a digest, such as `python:3.13-slim@sha256:...`, to get the same image every time.

## Configuration

[Agent specs](https://pydantic.dev/docs/ai/core-concepts/agent/#agent-specs) can set `image`, `resource`, `allow_internet_access`, `secure_access`, `sandbox_timeout`, `working_dir` and `env`. Configure the server connection through `OPEN_SANDBOX_DOMAIN` and `OPEN_SANDBOX_API_KEY`; a custom `ConnectionConfig` is a Python-only option.

| Option | What it does |
| --- | --- |
| `image` | Image for a new sandbox, such as `'python:3.13-slim'`. Default: `None`, which only attaches to and destroys sandboxes by ref. |
| `connection` | The SDK's `ConnectionConfig` for your server. Default: `None`, which reads `OPEN_SANDBOX_DOMAIN` and `OPEN_SANDBOX_API_KEY`. |
| `resource` | Resource limits for a new sandbox, such as `{'cpu': '500m', 'memory': '1Gi'}`. Default: `None`, one CPU and 2 GiB. |
| `allow_internet_access` | Default: `True`, which keeps the server's network configuration. `False` requests deny-all egress for new sandboxes; see [Network access](#network-access) for server requirements. |
| `secure_access` | Default: `False`. `True` requires a per-sandbox endpoint token for new sandboxes on Kubernetes with an ingress gateway. Docker rejects `True`; attachment keeps existing protection. |
| `sandbox_timeout` | Seconds a sandbox may go unused before the server deletes it, files included; every operation renews it, and a single command must finish within it. Default: `None`, kept until destroyed. |
| `working_dir` | Absolute directory commands start in and relative paths resolve against, created on first use if missing. Default: `/workspace`; `None` uses the image's own. |
| `env` | Environment variables every command gets. Default: `None`. Nothing from your machine's environment reaches the sandbox, and `env` is kept out of the capability's `repr`, but the agent's commands can read it. |

## Telemetry

`OpenSandbox` emits no spans of its own. Core's [instrumentation](https://pydantic.dev/docs/ai/capabilities/instrumentation/) records the sandbox on the agent run span as `pydantic_ai.workspace.provider` (`opensandbox`) and `pydantic_ai.workspace.id`. Each command or file operation a tool makes runs inside that tool call's span. Creating a sandbox logs its ID at `INFO` (`Created OpenSandbox sandbox <id>`) on the `pydantic_ai_harness.opensandbox_sandbox._backend` logger.

The OpenSandbox SDK reports how long each create took to your server's `/metrics/events`. Set `OPENSANDBOX_DISABLE_METRICS=1`, or pass `ConnectionConfig(disable_metrics=True)`, to turn that off.

## API reference

::: pydantic_ai_harness.opensandbox_sandbox.OpenSandbox

::: pydantic_ai_harness.opensandbox_sandbox.OpenSandboxBackend
