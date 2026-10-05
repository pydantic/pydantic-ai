"""Fallback chains: `chain:NAME` runs its models in order, moving on when one fails.

Pydantic AI's `FallbackModel` does the work; a chain only names the models. Pair it with auth
profiles to pool accounts, such as two Codex logins: `/chain codex openai-codex:gpt-6-astra
openai-codex@work:gpt-6-astra`, then `/model chain:codex`. A run falls back on a model API error,
such as a rate or usage limit; a profile that is not signed in fails the run instead.
"""

from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.models.profiles import check_name, parse_model

PREFIX = 'chain'
USAGE = (
    'Usage: /chain (list), /chain NAME (show), /chain NAME MODEL MODEL... (save), /chain remove NAME. '
    'Select a chain with /model chain:NAME.'
)


def chain_name(model: str) -> str | None:
    """`NAME` for `chain:NAME`, else `None`."""
    prefix, separator, name = model.partition(':')
    return name if separator and prefix == PREFIX else None


def settings_model(store: SettingsStore, model: str) -> str:
    """The model whose `/model settings` controls and defaults a chain takes: its first model."""
    name = chain_name(model)
    members = store.chains().get(name) if name is not None else None
    return members[0] if members else model


def chain_command(store: SettingsStore, args: list[str]) -> str:
    """List, show, save, or remove chains; selecting one is `/model chain:NAME`."""
    chains = store.chains()
    if not args:
        if not chains:
            return 'No chains. ' + USAGE
        return '\n'.join(f'chain:{name}  {" -> ".join(models)}' for name, models in chains.items())
    if args[0] == 'remove':
        if len(args) != 2 or args[1] not in chains:
            raise ValueError(USAGE if len(args) != 2 else f'No chain named {args[1]}.')
        if not store.remove_model(name=f'chain:{args[1]}'):
            raise ValueError(f'chain:{args[1]} is the saved default model. Select another model first.')
        return f'Removed chain:{args[1]}.'
    name = check_name(args[0], kind='Chain name')
    if len(args) == 1:
        if name not in chains:
            raise ValueError(f'No chain named {name}. ' + USAGE)
        return f'chain:{name}  {" -> ".join(chains[name])}'
    if len(args) == 2:
        raise ValueError('A chain needs at least two models. ' + USAGE)
    models = [_member(model) for model in args[1:]]
    store.save_chain(name=name, models=models)
    return f'Saved chain:{name} ({" -> ".join(models)}). Select it with /model chain:{name}.'


def _member(model: str) -> str:
    """A provider-qualified, installed model that is not itself a chain."""
    from pydantic_clai2.models.model_catalog import check_installed

    if chain_name(model) is not None:
        raise ValueError(f'{model}: a chain cannot contain another chain.')
    if not parse_model(model).provider:
        raise ValueError(f'{model}: name the provider, as PROVIDER[@PROFILE]:MODEL.')
    check_installed(model)
    return model


def chain_completions(store: SettingsStore, args: list[str]) -> list[str]:
    """Chain names and `remove` first, then saved models for the members."""
    if len(args) <= 1:
        return ['remove', *store.chains()]
    if args[0] == 'remove':
        return list(store.chains()) if len(args) == 2 else []
    return [model for model in store.models() if chain_name(model) is None]
