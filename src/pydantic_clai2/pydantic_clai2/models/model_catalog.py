"""Models the `/model add` menu can offer, with what is known about each.

genai-prices is the first source. Add another (models.dev, a provider API) as one more
function returning `CatalogModel`s and merge it in `catalog()`.
"""

import sys
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.metadata import metadata

from genai_prices.data_snapshot import get_snapshot

from pydantic_ai.models import known_model_names
from pydantic_ai.providers import infer_provider_class
from pydantic_clai2.models import github_copilot
from pydantic_clai2.models.profiles import provider_of

CODEX_MODELS = tuple(
    f'openai-codex:{model}'
    for model in (
        'gpt-6.1-sol',
        'gpt-6-astra',
        'gpt-6-sol',
        'gpt-6-luna',
        'gpt-5.6-luna',
        'gpt-5.6-terra',
        'gpt-5.6-sol',
    )
)
"""Subscription models offered in the catalog and setting completions."""

EXTRA_PROVIDERS = frozenset({'github-copilot', 'openai-codex'})
"""Provider prefixes core can run but does not list in `known_model_names()`."""


@dataclass(frozen=True, kw_only=True)
class CatalogModel:
    """One model as the menu shows it."""

    name: str
    provider: str
    label: str
    context_window: int | None = None
    prices: str | None = None


def runnable_providers() -> frozenset[str]:
    """Provider prefixes that `infer_model` accepts."""
    return frozenset(name.partition(':')[0] for name in known_model_names()) | EXTRA_PROVIDERS


def check_installed(name: str) -> None:
    """Raise `ValueError` with install commands if the SDK for `name`'s provider is missing here.

    Core imports provider SDKs lazily, so without this a missing extra only surfaces on the next turn.
    """
    provider = provider_of(name)
    try:
        infer_provider_class(provider)
    except ValueError:
        return  # Not a core provider, like CLAI's own `vllm:`; the turn resolves or reports it.
    except ImportError as exc:
        where = f'the Python CLAI2 runs on ({sys.executable})'
        if provider not in (metadata('pydantic-clai2').get_all('Provides-Extra') or ()):
            raise ValueError(f'Cannot use {name} with {where}: {exc}') from exc
        raise ValueError(
            f'Cannot use {name}: the `{provider}` extra is not installed in {where}. '
            f'Install it there with `pip install "pydantic-ai-slim[{provider}]"`, '
            f'or from a pydantic-ai checkout run `uv run --package pydantic-clai2 --extra {provider} clai2`.'
        ) from exc


def genai_prices_models() -> list[CatalogModel]:
    """Current (not deprecated) models from genai-prices, for providers core can run."""
    providers = runnable_providers()
    found: list[CatalogModel] = []
    for provider in get_snapshot().providers:
        if provider.id not in providers:
            continue
        for model in provider.models:
            if model.deprecated or ':' in model.id:
                continue
            found.append(
                CatalogModel(
                    name=f'{provider.id}:{model.id}',
                    provider=provider.id,
                    label=model.name or model.id,
                    context_window=model.context_window,
                    prices=str(model.prices),
                )
            )
    return found


async def github_copilot_models() -> list[CatalogModel]:
    """Discover the signed-in account's visible, Chat Completions-compatible models."""
    return [
        CatalogModel(name=f'github-copilot:{name}', provider='github-copilot', label=name)
        for name in await github_copilot.discover()
    ]


def catalog(*, include: Iterable[str] = (), discovered: Iterable[CatalogModel] = ()) -> list[CatalogModel]:
    """Merge sources; authenticated discovery replaces static entries for the providers it contains."""
    models = {model.name: model for model in genai_prices_models()}
    for name in (
        *known_model_names(),
        *CODEX_MODELS,
        *include,
    ):
        if name and name not in models:
            models[name] = CatalogModel(name=name, provider=provider_of(name), label=name.partition(':')[2])
    discovered = tuple(discovered)
    providers = {model.provider for model in discovered}
    models = {name: model for name, model in models.items() if model.provider not in providers}
    models.update((model.name, model) for model in discovered)
    return [models[name] for name in sorted(models)]
