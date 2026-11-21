"""The cloud sandboxes `/sandbox` can move a conversation into.

`/sandbox` signs `git` and `gh` in to GitHub as the user in one of two ways. Where the provider's network
proxy can add the token to requests for GitHub (E2B, boxd, Daytona, OpenSandbox), the token stays outside the
sandbox, which holds a stand-in. Elsewhere (Modal, Render, Sprites, Kubernetes) the token is written into the
sandbox, where anything run there can read it (`SandboxProvider.token_in_sandbox`). Adding one is a `SandboxProvider`
class in `PROVIDERS`.
"""

from collections.abc import Callable, Mapping

from pydantic_ai_pax.builtin_plugins.sandbox.providers.base import GITHUB_HOSTS, GitHubCredential, SandboxProvider
from pydantic_ai_pax.builtin_plugins.sandbox.providers.boxd import BoxdProvider
from pydantic_ai_pax.builtin_plugins.sandbox.providers.daytona import DaytonaProvider
from pydantic_ai_pax.builtin_plugins.sandbox.providers.e2b import E2BProvider
from pydantic_ai_pax.builtin_plugins.sandbox.providers.kubernetes import KubernetesProvider
from pydantic_ai_pax.builtin_plugins.sandbox.providers.modal import ModalProvider
from pydantic_ai_pax.builtin_plugins.sandbox.providers.opensandbox import OpenSandboxProvider
from pydantic_ai_pax.builtin_plugins.sandbox.providers.render import RenderProvider
from pydantic_ai_pax.builtin_plugins.sandbox.providers.sprites import SpritesProvider

__all__ = ('GITHUB_HOSTS', 'PROVIDERS', 'GitHubCredential', 'SandboxProvider')

PROVIDERS: Mapping[str, Callable[[], SandboxProvider]] = {
    make().name: make
    for make in (
        E2BProvider,
        BoxdProvider,
        DaytonaProvider,
        OpenSandboxProvider,
        ModalProvider,
        RenderProvider,
        SpritesProvider,
        KubernetesProvider,
    )
}
"""What makes each provider `/sandbox` offers, by name.

Each session makes its own, so what one holds open for a sandbox, such as an API client, is that session's alone,
and closing it when the session ends leaves another session's commands in the same sandbox running.
"""
