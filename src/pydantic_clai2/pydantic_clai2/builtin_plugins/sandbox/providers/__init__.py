"""The cloud sandboxes `/sandbox` can move a conversation into.

`/sandbox` signs `git` and `gh` in to GitHub as the user without the token ever entering the sandbox:
the provider's network proxy, outside it, adds the token to requests for GitHub. Only providers whose
proxy can do that are listed. Adding one is a `SandboxProvider` in `PROVIDERS`.
"""

from collections.abc import Mapping

from pydantic_clai2.builtin_plugins.sandbox.providers.base import GITHUB_HOSTS, GitHubCredential, SandboxProvider
from pydantic_clai2.builtin_plugins.sandbox.providers.boxd import BoxdProvider
from pydantic_clai2.builtin_plugins.sandbox.providers.e2b import E2BProvider

__all__ = ('GITHUB_HOSTS', 'PROVIDERS', 'GitHubCredential', 'SandboxProvider')

PROVIDERS: Mapping[str, SandboxProvider] = {provider.name: provider for provider in (E2BProvider(), BoxdProvider())}
"""The providers `/sandbox` offers, by name."""
