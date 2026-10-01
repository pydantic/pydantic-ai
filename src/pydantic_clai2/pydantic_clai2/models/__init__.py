"""CLAI's model integrations. Submodules load on demand; this package stays cheap to import."""

from collections.abc import Iterable

CLAI_PROVIDERS = frozenset({'github-copilot', 'openai-codex', 'openrouter', 'vllm'})
"""Prefixes CLAI's own model resolver handles before Pydantic AI sees the name."""

LOGINS = ('codex', 'copilot')
"""`/login NAME` sign-ins CLAI ships; bare `/login` is `codex`."""

LOGIN_ALIASES = {'openai-codex': 'codex', 'github-copilot': 'copilot'}
"""Provider names `/login` still accepts for its sign-ins."""


def login_names(plugin_logins: Iterable[str] = ()) -> tuple[str, ...]:
    """What `/login` lists and completes: CLAI's sign-ins, then the ones plugins add."""
    return (*LOGINS, *sorted(set(plugin_logins)))
