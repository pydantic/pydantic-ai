"""`/sandbox kubernetes`: the GitHub token is written into the sandbox pod, where anything run there can read it.

The sandbox is a pod claimed from a kubernetes-sigs/agent-sandbox warm pool on the cluster `kubectl`'s current
context names. No proxy there can sign requests, so `give_github_login` writes the token into the pod's home.
sandboxd, the pod's command server, has no file API the backend uses, so the token reaches the pod base64-encoded
inside a command sent to sandboxd, through `kubectl port-forward`, as `Workspace` writes files there; anyone who
can exec into the pod or read its volume can read it too. It goes when the claim is deleted, pod and volume with it.

The claim has no time limit, so the sandbox lives until `/sandbox delete`.
"""

import shutil
from dataclasses import dataclass
from importlib.util import find_spec
from typing import TYPE_CHECKING

import anyio

from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_pax.builtin_plugins.sandbox.providers.base import (
    SANDBOX_ENV,
    GitHubCredential,
    saved_key,
)
from pydantic_ai_pax.builtin_plugins.sandbox.seed import SandboxBackend

if TYPE_CHECKING:
    from pydantic_ai_harness.kubernetes_sandbox import KubernetesSandboxBackend

WARM_POOL_NAME = 'CLAI_KUBERNETES_WARM_POOL'
NAMESPACE_NAME = 'CLAI_KUBERNETES_NAMESPACE'


@dataclass(frozen=True)
class KubernetesProvider:
    """Pods from an [agent-sandbox](https://agent-sandbox.sigs.k8s.io) warm pool, named by `CLAI_KUBERNETES_WARM_POOL`."""

    name: str = 'kubernetes'
    title: str = 'Kubernetes'
    working_dir: str = '/workspace/checkout'
    """Below sandboxd's default root, which the docs' templates also make the home, so the token is not in the checkout."""
    token_in_sandbox: bool = True

    def missing(self) -> str | None:
        if find_spec('k8s_agent_sandbox') is None or find_spec('grpc') is None:
            return "Kubernetes support is not installed. Install CLAI with it: uv tool install 'pydantic-ai-pax[kubernetes]'."
        if shutil.which('kubectl') is None:
            return 'Install kubectl: CLAI reaches the sandbox pods through kubectl port-forward.'
        if saved_key(WARM_POOL_NAME) is None:
            return (
                f'Add the agent-sandbox warm pool to claim sandboxes from in /keys as {WARM_POOL_NAME}, '
                f"and its namespace as {NAMESPACE_NAME} if it is not 'default'. kubectl's current context names the cluster."
            )
        return None

    async def create(self, *, github: GitHubCredential | None) -> tuple[WorkspaceRef, SandboxBackend]:
        from pydantic_ai_harness.kubernetes_sandbox import KubernetesSandboxBackend

        backend = KubernetesSandboxBackend(
            warm_pool=saved_key(WARM_POOL_NAME),
            namespace=saved_key(NAMESPACE_NAME) or 'default',
            sandbox_timeout=None,
            working_dir=self.working_dir,
            env=SANDBOX_ENV,
        )
        try:
            # A claim that fails is rolled back by the backend, so only its `kubectl port-forward` is left to stop.
            await backend.get_sandbox()
        except BaseException:
            with anyio.CancelScope(shield=True):
                await backend.aclose()
            raise
        assert backend.ref is not None
        return backend.ref, backend

    def reopen(self, ref: WorkspaceRef) -> SandboxBackend:
        from pydantic_ai_harness.kubernetes_sandbox import KubernetesSandboxBackend

        return KubernetesSandboxBackend(ref=ref, working_dir=self.working_dir, env=SANDBOX_ENV)

    async def close(self, backend: SandboxBackend) -> None:
        # Stops the `kubectl port-forward` the backend started.
        await _kubernetes(backend).aclose()

    async def delete(self, ref: WorkspaceRef) -> None:
        from pydantic_ai_harness.kubernetes_sandbox import KubernetesSandbox

        # The ref names the claim's namespace; the warm pool is only for new claims.
        namespace = ref.id.partition('/')[0]
        await KubernetesSandbox[None](warm_pool='', namespace=namespace).destroy(ref)


def _kubernetes(backend: SandboxBackend) -> 'KubernetesSandboxBackend':
    from pydantic_ai_harness.kubernetes_sandbox import KubernetesSandboxBackend

    assert isinstance(backend, KubernetesSandboxBackend)
    return backend
