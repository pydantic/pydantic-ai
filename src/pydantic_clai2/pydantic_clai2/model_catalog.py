"""Models the `/add_model` menu can offer, with what is known about each.

genai-prices is the first source. Add another (models.dev, a provider API) as one more
function returning `CatalogModel`s and merge it in `catalog()`.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from genai_prices.data_snapshot import get_snapshot
from pydantic_ai.models import known_model_names

from . import github_copilot

CODEX_MODELS = tuple(
    f'openai-codex:{model}'
    for model in ('gpt-6-astra', 'gpt-6-sol', 'gpt-6-luna', 'gpt-5.6-luna', 'gpt-5.6-terra', 'gpt-5.6-sol')
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
            provider, _, label = name.partition(':')
            models[name] = CatalogModel(name=name, provider=provider, label=label)
    discovered = tuple(discovered)
    providers = {model.provider for model in discovered}
    models = {name: model for name, model in models.items() if model.provider not in providers}
    models.update((model.name, model) for model in discovered)
    return [models[name] for name in sorted(models)]
