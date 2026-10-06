"""Fallback chains: `chain:NAME` runs its models in order, moving on when one fails.

Pydantic AI's `FallbackModel` does the work; a chain only names the models. They are created, edited,
renamed, and deleted in the `/model` picker (`ui/menus/chain_menu.py`). Pair a chain with auth
profiles to pool accounts, such as two Codex logins: `openai-codex:gpt-6-astra` then
`openai-codex@work:gpt-6-astra`. A run falls back on a model API error, such as a rate or usage
limit; a profile that is not signed in fails the run instead.
"""

from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.models.profiles import parse_model

PREFIX = 'chain'


def chain_name(model: str) -> str | None:
    """`NAME` for `chain:NAME`, else `None`."""
    prefix, separator, name = model.partition(':')
    return name if separator and prefix == PREFIX else None


def settings_model(store: SettingsStore, model: str) -> str:
    """The model whose `/model settings` controls and defaults a chain takes: its first model."""
    name = chain_name(model)
    members = store.chains().get(name) if name is not None else None
    return members[0] if members else model


def check_member(model: str) -> str:
    """A provider-qualified, installed model that is not itself a chain; raises `ValueError` otherwise."""
    from pydantic_clai2.models.model_catalog import check_installed

    if chain_name(model) is not None:
        raise ValueError(f'{model}: a chain cannot contain another chain.')
    if not parse_model(model).provider:
        raise ValueError(f'{model}: name the provider, as PROVIDER[@PROFILE]:MODEL.')
    check_installed(model)
    return model
