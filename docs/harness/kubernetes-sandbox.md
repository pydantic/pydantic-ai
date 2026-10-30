---
title: Kubernetes Sandbox
description: "Run a Pydantic AI agent's commands and file edits in a Kubernetes sandbox pod from a kubernetes-sigs/agent-sandbox warm pool."
---

# Kubernetes Sandbox

Run your agent's commands and file edits in a sandbox pod on your own Kubernetes cluster, claimed from a [kubernetes-sigs/agent-sandbox](https://agent-sandbox.sigs.k8s.io) warm pool.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/kubernetes_sandbox/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Install

```bash
pip/uv-add "pydantic-ai-harness[coder,kubernetes,anthropic]"
```

The `coder` extra is there because the examples use `Coder`; the sandbox itself doesn't need it. The `anthropic` extra is there because the examples use an Anthropic model; swap it for your model provider's extra. The client uses your kubeconfig, or the pod's service account inside the cluster. The default port-forward connection also needs `kubectl` on your `PATH`.

## Set up the cluster

Install agent-sandbox (v1.0.7 or later) with its extensions:

```bash
kubectl apply --server-side -f https://github.com/kubernetes-sigs/agent-sandbox/releases/download/v1.0.7/sandbox-with-extensions.yaml
```

Each sandbox pod must run [sandboxd](https://github.com/kubernetes-sigs/agent-sandbox/tree/main/packages/sandboxd), agent-sandbox's in-pod server. Commands run in the container that runs sandboxd, so that container's image decides which tools the agent has. This template copies sandboxd into `python:3.12` and keeps three sandboxes warm:

```yaml
apiVersion: extensions.agents.x-k8s.io/v1beta1
kind: SandboxTemplate
metadata:
  name: python
spec:
  podTemplate:
    spec:
      automountServiceAccountToken: false
      securityContext: {runAsNonRoot: true, fsGroup: 1000}
      initContainers:
        - name: install-sandboxd
          image: us-central1-docker.pkg.dev/k8s-staging-images/agent-sandbox/sandboxd:v20261015-v1.0.7
          command: [/bin/sh, -c, cp /usr/local/bin/sandboxd /sandboxd-bin/sandboxd]
          securityContext: {runAsUser: 1000}
          volumeMounts: [{name: sandboxd-bin, mountPath: /sandboxd-bin}]
      containers:
        - name: app
          image: python:3.12
          command: [/sandboxd-bin/sandboxd, --root-dir=/workspace]
          env: [{name: HOME, value: /workspace}]
          securityContext: {runAsUser: 1000, allowPrivilegeEscalation: false}
          readinessProbe: {httpGet: {path: /v1/health, port: 8080}}
          volumeMounts:
            - {name: sandboxd-bin, mountPath: /sandboxd-bin, readOnly: true}
            - {name: workspace, mountPath: /workspace}
      volumes:
        - {name: sandboxd-bin, emptyDir: {}}
        - {name: workspace, emptyDir: {}}
---
apiVersion: extensions.agents.x-k8s.io/v1beta1
kind: SandboxWarmPool
metadata:
  name: python-pool
spec:
  replicas: 3
  sandboxTemplateRef: {name: python}
```

agent-sandbox doesn't publish sandboxd to `registry.k8s.io` yet, so this template pulls the v1.0.7 build from its staging registry, which may not stay available. For production, build sandboxd from the release tag's `packages/sandboxd/Dockerfile` and push it to your own registry.

For stronger isolation, set `runtimeClassName` (gVisor or Kata) on the pod template and [connect from inside the cluster](#connect-to-the-sandbox).

## Quick start

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.kubernetes_sandbox import KubernetesSandbox

agent = Agent('{{ anthropic:large }}', capabilities=[KubernetesSandbox(warm_pool='python-pool'), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
```

`Coder`'s shell and file tools now run in a sandbox pod, not on your machine. The run claims an already-running pod from the warm pool the first time it needs the workspace; with `Coder`, `RepoContext` does that when the run starts, even without a tool call. The sandbox keeps running after the run ends; see [Clean up](#clean-up).

## Continue in the same sandbox

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.kubernetes_sandbox import KubernetesSandbox

agent = Agent('{{ anthropic:large }}', capabilities=[KubernetesSandbox(warm_pool='python-pool'), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')

followup = agent.run_sync(
    'Which capability would you add next, and where would it live?',
    message_history=result.all_messages(),
)
```

The follow-up run finds the sandbox in the message history and works in it, so the clone is still there. If the claim has been deleted, the run raises `WorkspaceUnavailableError` instead of starting over in an empty sandbox.

## Choose the tools

For a narrower agent, use [`Shell`](shell.md) and [`FileSystem`](filesystem.md) instead of `Coder`, or write your own tool that runs in the sandbox:

```python {names="defined"}
from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.kubernetes_sandbox import KubernetesSandbox
from pydantic_ai_harness.shell import Shell

agent = Agent('{{ anthropic:large }}', capabilities=[KubernetesSandbox(warm_pool='python-pool'), Shell(), FileSystem()])


@agent.tool
async def run_python(ctx: RunContext, code: str) -> str:
    """Run a Python snippet in the sandbox."""
    result = await ctx.workspace.run(['python', '-c', code], timeout=10)
    return result.stdout + result.stderr
```

Commands start in sandboxd's root directory, `/workspace` in the template above, and run as the container's user.

### What a timeout stops

A command timeout starts once the sandbox is claimed. A timeout or a cancelled run kills the command's process group, and the sandbox keeps running for other commands. `WorkspaceTimeoutError` carries the output so far. A job that detaches with `setsid` may keep running.

A background job that keeps the command's output open delays the result by about 5 seconds. A command whose combined output passes 10 MiB is stopped and raises `WorkspaceOutputLimitError`, with the first 64 KiB of each stream in `stdout` and `stderr`. Redirect large or background output to a file instead; `Shell.start_command` manages its own output log.

## Reattach later

To come back to the sandbox without the message history, pass its ref back as `workspace=`:

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.kubernetes_sandbox import KubernetesSandbox

agent = Agent('{{ anthropic:large }}', capabilities=[KubernetesSandbox(warm_pool='python-pool'), Coder()])

result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
ref = result.workspace.ref  # store this, e.g. in your database

later = agent.run_sync('Which capability would you add next, and where would it live?', workspace=ref)
```

The ref is `<namespace>/<claim name>`. It holds no credentials, so the process that reattaches needs its own access to the cluster. Pass `workspace='new'` to claim a fresh sandbox even when the message history names one.

`warm_pool`, `namespace` and `sandbox_timeout` only shape a new claim; `connection`, `working_dir` and `env` apply to every command, including after you reattach.

`KubernetesSandbox(...).backend(ref)` returns a backend for a stored ref that connects on first use. A run leaves a backend you built open: close it with `await backend.aclose()`, which closes the connection (and any `kubectl port-forward`) but not the sandbox.

## Prepare a sandbox before the run

To seed files before the agent starts, build the backend yourself, write through a `Workspace`, and pass the backend to the run:

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai.workspaces import Workspace
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.kubernetes_sandbox import KubernetesSandbox, KubernetesSandboxBackend

agent = Agent('{{ anthropic:large }}', capabilities=[KubernetesSandbox(warm_pool='python-pool'), Coder()])


async def run_in_prepared_sandbox() -> str:
    backend = KubernetesSandboxBackend(warm_pool='python-pool')
    try:
        await Workspace(backend).write_text('TASK.md', 'Add a `--version` flag to cli.py.\n')
        result = await agent.run('Do the task in TASK.md.', workspace=backend)
        return result.output
    finally:
        await backend.aclose()
        ref = backend.ref  # set once the sandbox is claimed
        if ref is not None:
            await KubernetesSandbox(warm_pool='python-pool').destroy(ref)
```

The run uses the backend as it is, so `KubernetesSandbox`'s settings don't apply to it, and leaves it open: close it with `aclose()`. To keep the sandbox instead, store `backend.ref` and pass it as `workspace=` to reattach later.

## Connect to the sandbox

By default the client reaches each sandbox through `kubectl port-forward`, which suits running the agent on your machine. An agent running in the cluster connects directly:

```python {names="defined"}
from pydantic_ai_harness.kubernetes_sandbox import KubernetesSandbox

sandbox = KubernetesSandbox(warm_pool='python-pool', namespace='agents', connection='in-cluster-pod-ip')
```

`'in-cluster-service'` connects through the sandbox's Service instead, which needs `service: true` on the template. Either way, agent-sandbox's default NetworkPolicy blocks traffic from other pods, so add a `networkPolicy` to the template that lets your agent's pods reach TCP ports 8080 and 9090. Port-forwarding doesn't work on gVisor or Kata, so use an in-cluster connection there.

For sandboxd on other ports than 8080 and 9090, or a port-forward that needs longer to start, pass the SDK's connection config:

```python {names="defined"}
from k8s_agent_sandbox.models import SandboxdPodTunnelConnectionConfig
from pydantic_ai_harness.kubernetes_sandbox import KubernetesSandbox

sandbox = KubernetesSandbox(
    warm_pool='python-pool',
    connection=SandboxdPodTunnelConnectionConfig(grpc_port=9191, rest_port=8181, port_forward_ready_timeout=60),
)
```

`SandboxdInClusterConnectionConfig` takes the same ports, with `mode='in-cluster-pod-ip'` or `mode='in-cluster-service'`.

## Clean up

Ending a run never deletes the sandbox. A sandbox claimed by `KubernetesSandbox` is deleted after `sandbox_timeout` (24 hours by default) whether or not anyone still uses it; delete it sooner with the ref you stored:

```python {names="defined"}
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.kubernetes_sandbox import KubernetesSandbox


async def delete_sandbox(ref: WorkspaceRef) -> None:
    await KubernetesSandbox(warm_pool='python-pool').destroy(ref)
```

`destroy(ref)` deletes the claim, and with it the pod; a claim that is already gone returns quietly.

## Configuration

| Option | What it does |
| --- | --- |
| `warm_pool` | Name of the `SandboxWarmPool` to claim new sandboxes from. Required. |
| `namespace` | Namespace of the warm pool and the claims. Default: `'default'`. |
| `connection` | `'port-forward'` (default), `'in-cluster-pod-ip'` or `'in-cluster-service'`; or the SDK's `SandboxdPodTunnelConnectionConfig` or `SandboxdInClusterConnectionConfig`, for other ports and timeouts. |
| `sandbox_timeout` | Seconds after which a newly claimed sandbox is deleted. Default: `86_400` (24 hours); `None` keeps it until `destroy(ref)`. |
| `working_dir` | Absolute directory commands start in and relative paths resolve against. Default: `None`, sandboxd's root directory. It must already exist. |
| `env` | Environment variables every command gets, on top of the container's own. Default: `None`. |

## Durable execution

Outside a run, every command opens its own connection (with the default connection, a `kubectl port-forward`) and closes it when it ends, so the worker has nothing to close. File operations are shell commands, and one may run a few, so each activity takes a little longer than it does inside a plain run. Run a Temporal dev server on `localhost:7233` first.

```python
import asyncio
import uuid

from pydantic_ai import Agent
from pydantic_ai.durable_exec.temporal import PydanticAIPlugin, PydanticAIWorkflow, TemporalDurability
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.kubernetes_sandbox import KubernetesSandbox
from temporalio import workflow
from temporalio.client import Client
from temporalio.worker import Worker

agent = Agent(
    '{{ anthropic:large }}',
    name='kubernetes_coder',
    capabilities=[KubernetesSandbox(warm_pool='python-pool'), Coder(), TemporalDurability()],
)


@workflow.defn
class SandboxWorkflow(PydanticAIWorkflow):
    __pydantic_ai_agents__ = [agent]

    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await agent.run(prompt)).output


async def main() -> None:
    client = await Client.connect('localhost:7233', plugins=[PydanticAIPlugin()])
    async with Worker(client, task_queue='sandbox', workflows=[SandboxWorkflow]):
        print(
            await client.execute_workflow(
                SandboxWorkflow.run,
                'Use the shell tool to run pwd.',
                id=f'sandbox-{uuid.uuid4()}', task_queue='sandbox',
            )
        )


if __name__ == '__main__':
    asyncio.run(main())
```

`Coder`, `Shell`, and `FileSystem` work under DBOS, Temporal and Prefect. See the [Coder](coder.md#durable-execution), [Shell](shell.md#durable-execution), and [FileSystem](filesystem.md#durable-execution) guides for engine-specific limits.

Removing a capability while workflows using it are still running changes their replay history. Drain those workflows or use [Temporal worker versioning](https://docs.temporal.io/production-deployment/worker-deployments/worker-versioning) before deploying the change.

## Telemetry

`KubernetesSandbox` emits no spans of its own. Core's [instrumentation](../capabilities/instrumentation.md) records the sandbox on the agent run span as `pydantic_ai.workspace.provider` and `pydantic_ai.workspace.id`, and each command or file operation a tool makes runs inside that tool call's span. Claiming a sandbox logs its ref at `INFO` (`Claimed Kubernetes sandbox <namespace>/<claim name>`) on the `pydantic_ai_harness.kubernetes_sandbox._backend` logger, so a sandbox that outlives its run can still be found.

## Limitations

- Works on the asyncio event loop only; under Trio, workspace operations and `destroy` raise `UserError`.
- Commands start through `/bin/sh` and file operations run as shell commands, so the image needs a POSIX shell with coreutils (`base64`, `dd`, `find`).
- If a sandbox's pod is lost while its claim remains, for example when its node fails, agent-sandbox starts a new pod. Commands fail until it is ready, with an error durable execution retries. The new pod starts empty except for what the template keeps on a persistent volume; the template above uses `emptyDir`, so files are lost.
- sandboxd's optional bearer token is not supported yet; rely on the NetworkPolicy to restrict who can reach the pods.

See [Workspaces](https://pydantic.dev/docs/ai/core-concepts/workspace/) for more.

## API reference

::: pydantic_ai_harness.kubernetes_sandbox.KubernetesSandbox

::: pydantic_ai_harness.kubernetes_sandbox.KubernetesSandboxBackend
