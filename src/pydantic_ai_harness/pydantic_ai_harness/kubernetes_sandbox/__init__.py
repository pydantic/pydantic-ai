"""Kubernetes agent sandbox capability: runs agents in sandboxes from a kubernetes-sigs/agent-sandbox warm pool.

`KubernetesSandbox` is the supported entry point; build an agent with it and add tools
that use the workspace, such as `Coder`.
`KubernetesSandboxBackend` is the agent-sandbox implementation of Pydantic AI's workspace backend
protocol, public for applications that want to claim or attach to a sandbox themselves and pass it
to a run as `workspace=`.
"""

from pydantic_ai_harness.kubernetes_sandbox._backend import KubernetesSandboxBackend
from pydantic_ai_harness.kubernetes_sandbox._capability import KubernetesSandbox

__all__ = [
    'KubernetesSandbox',
    'KubernetesSandboxBackend',
]
