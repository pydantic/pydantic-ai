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
pip/uv-add "pydantic-ai-harness[kubernetes,anthropic]"
```

The `anthropic` extra is there because the examples use an Anthropic model; swap it for your model provider's extra. The client uses your kubeconfig (or the pod's service account inside the cluster), and `kubectl` on your `PATH` when it connects through a port-forward.

## Set up the cluster

Install agent-sandbox (v1.0.7 or later) with its extensions:

```bash
kubectl apply --server-side -f https://github.com/kubernetes-sigs/agent-sandbox/releases/download/v1.0.7/sandbox-with-extensions.yaml
```

The sandbox pods must run [sandboxd](https://github.com/kubernetes-sigs/agent-sandbox/tree/main/packages/sandboxd), agent-sandbox's runtime daemon: commands run in the container that runs it, so that container's image decides which tools the agent has. This template copies sandboxd into `python:3.12-slim` and keeps three sandboxes warm:

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
          image: python:3.12-slim
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

agent-sandbox doesn't publish sandboxd to `registry.k8s.io` yet, so this template uses the build of the v1.0.7 release from its staging registry. Staging images aren't guaranteed to stay available: for production, build sandboxd from the release tag's `packages/sandboxd/Dockerfile` and push it to your own registry. For stronger isolation, set `runtimeClassName` (gVisor or Kata) on the pod template, and connect from inside the cluster (see [Connect from inside the cluster](#connect-from-inside-the-cluster)).

## Quick start

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.kubernetes_sandbox import KubernetesSandbox

agent = Agent('{{ anthropic:large }}', capabilities=[KubernetesSandbox(warm_pool='python-pool'), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
```

`Coder`'s shell and file tools now run in a sandbox pod, not on your machine. The first workspace operation claims a ready sandbox from the warm pool, so a run starts in a pod that is already running. The sandbox keeps running after the run ends; see [Clean up](#clean-up).

Commands start in sandboxd's root directory, `/workspace` in the template above, and run as the container's user.

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
ref = result.workspace.ref  # or store this, and pass it back later as `workspace=ref`
```

The follow-up run finds the sandbox in the message history and works in it, so the clone is still there. The ref names the claim as `<namespace>/<claim name>` and holds no credentials. If the claim has been deleted, the run raises `WorkspaceUnavailableError` instead of starting over in an empty sandbox. Pass `workspace='new'` to claim a fresh sandbox even when the message history names one.

## Connect from inside the cluster

By default the client reaches each sandbox through `kubectl port-forward`, which suits running the agent on your machine. An agent running in the cluster connects directly:

```python {names="defined"}
from pydantic_ai_harness.kubernetes_sandbox import KubernetesSandbox

sandbox = KubernetesSandbox(warm_pool='python-pool', namespace='agents', connection='in-cluster-pod-ip')
```

`'in-cluster-service'` connects through the sandbox's Service instead, which needs `service: true` on the template. Either way, agent-sandbox's default NetworkPolicy admits traffic only from its router, so the template's `networkPolicy` must admit your agent's pods on TCP ports 8080 and 9090. Port-forwarding does not work on gVisor or Kata, so use an in-cluster connection there.

## Other ports and timeouts

The connection modes use sandboxd's default ports, 8080 and 9090, and the SDK's default timeouts. If your template runs sandboxd on other ports, or a port-forward needs longer to start, pass the SDK's own connection config instead:

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

`destroy(ref)` deletes the claim, and with it the pod; a claim that is already gone returns quietly. `KubernetesSandbox.backend(ref)` builds a backend for a stored ref without connecting.

## Configuration

| Option | What it does |
| --- | --- |
| `warm_pool` | Name of the `SandboxWarmPool` to claim new sandboxes from. Required. |
| `namespace` | Namespace of the warm pool and the claims. Default: `'default'`. |
| `connection` | `'port-forward'` (default), `'in-cluster-pod-ip'` or `'in-cluster-service'`; or the SDK's `SandboxdPodTunnelConnectionConfig` or `SandboxdInClusterConnectionConfig`, for other ports and timeouts. |
| `sandbox_timeout` | Seconds after which a newly claimed sandbox is deleted. Default: `86_400` (24 hours); `None` keeps it until `destroy(ref)`. |
| `working_dir` | Absolute directory commands start in and relative paths resolve against. Default: `None`, sandboxd's root directory. It must already exist. |
| `env` | Environment variables every command gets, on top of the container's own. Default: `None`. |

## Limitations

- Commands run on the asyncio event loop only; under Trio, `run` raises `UserError`.
- Commands start through `/bin/sh` and file operations run as shell commands, so the image needs a POSIX shell with coreutils (`base64`, `dd`, `find`). sandboxd's own file API only serves its root directory, so it cannot stand in for the whole filesystem the agent's commands see.
- A command's timeout or cancellation kills its process group; a job that detaches with `setsid` may keep running. A background job that keeps the command's output open delays the result by about 5 seconds; redirect its output to a file.
- A command whose combined output passes 10 MiB is stopped the same way and raises `WorkspaceOutputLimitError`, with the first 64 KiB of each stream in `stdout` and `stderr`. Redirect large output to a file instead.
- If a sandbox's pod is deleted while its claim remains, as when its node fails, the controller recreates it. Until the new pod is ready, commands fail with an error that durable runs retry, and the new pod keeps only what the template stores on a persistent volume.
- sandboxd's optional bearer token is not supported yet; rely on the NetworkPolicy to restrict who can reach the pods.

See [Workspaces](https://pydantic.dev/docs/ai/core-concepts/workspace/) for more.

## API reference

::: pydantic_ai_harness.kubernetes_sandbox.KubernetesSandbox

::: pydantic_ai_harness.kubernetes_sandbox.KubernetesSandboxBackend
