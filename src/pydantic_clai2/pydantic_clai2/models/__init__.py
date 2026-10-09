"""CLAI's model integrations. Submodules load on demand; this package stays cheap to import."""

from collections.abc import Iterable

CLAI_PROVIDERS = frozenset({'chain', 'github-copilot', 'openai-codex', 'openrouter', 'vllm'})
"""Prefixes CLAI's own model resolver handles before Pydantic AI sees the name."""

LOGINS = ('openai-codex', 'github-copilot')
"""`/login NAME` sign-ins CLAI ships, named after the model prefix each one unlocks."""

LOGIN_ALIASES = {'codex': 'openai-codex', 'copilot': 'github-copilot'}
"""Earlier sign-in names `/login` still accepts."""


def login_names(plugin_logins: Iterable[str] = ()) -> tuple[str, ...]:
    """What `/login` lists and completes: CLAI's sign-ins, then the ones plugins add."""
    return (*LOGINS, *sorted(set(plugin_logins)))
