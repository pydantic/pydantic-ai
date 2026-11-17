"""The cloud sandboxes `/sandbox` can move a conversation into.

`/sandbox` signs `git` and `gh` in to GitHub as the user in one of two ways. Where the provider's network
proxy can add the token to requests for GitHub (E2B, boxd, Daytona, OpenSandbox), the token stays outside the
sandbox, which holds a stand-in. Elsewhere (Modal, Render, Sprites, Kubernetes) the token is written into the
sandbox, where anything run there can read it (`SandboxProvider.token_in_sandbox`). Adding one is a `SandboxProvider` in
`PROVIDERS`.
"""

from collections.abc import Mapping

from pydantic_clai2.builtin_plugins.sandbox.providers.base import GITHUB_HOSTS, GitHubCredential, SandboxProvider
from pydantic_clai2.builtin_plugins.sandbox.providers.boxd import BoxdProvider
from pydantic_clai2.builtin_plugins.sandbox.providers.daytona import DaytonaProvider
from pydantic_clai2.builtin_plugins.sandbox.providers.e2b import E2BProvider
from pydantic_clai2.builtin_plugins.sandbox.providers.kubernetes import KubernetesProvider
from pydantic_clai2.builtin_plugins.sandbox.providers.modal import ModalProvider
from pydantic_clai2.builtin_plugins.sandbox.providers.opensandbox import OpenSandboxProvider
from pydantic_clai2.builtin_plugins.sandbox.providers.render import RenderProvider
from pydantic_clai2.builtin_plugins.sandbox.providers.sprites import SpritesProvider

__all__ = ('GITHUB_HOSTS', 'PROVIDERS', 'GitHubCredential', 'SandboxProvider')

PROVIDERS: Mapping[str, SandboxProvider] = {
    provider.name: provider
    for provider in (
        E2BProvider(),
        BoxdProvider(),
        DaytonaProvider(),
        OpenSandboxProvider(),
        ModalProvider(),
        RenderProvider(),
        SpritesProvider(),
        KubernetesProvider(),
    )
}
"""The providers `/sandbox` offers, by name."""
