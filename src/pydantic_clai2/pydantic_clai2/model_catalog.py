"""Models the `/model` menu can offer, with what is known about each.

genai-prices is the first source. Add another (models.dev, a provider API) as one more
function returning `CatalogModel`s and merge it in `catalog()`.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from genai_prices.data_snapshot import get_snapshot
from pydantic_ai.models import known_model_names

EXTRA_PROVIDERS = frozenset({'openai-codex'})
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


def catalog(*, include: Iterable[str] = ()) -> list[CatalogModel]:
    """Every source merged and sorted by name; `include` adds names not in any source."""
    models = {model.name: model for model in genai_prices_models()}
    for name in (
        *known_model_names(),
        *(f'openai-codex:{model}' for model in ('gpt-6-astra', 'gpt-5.6-luna', 'gpt-5.6-terra', 'gpt-5.6-sol')),
        *include,
    ):
        if name and name not in models:
            provider, _, label = name.partition(':')
            models[name] = CatalogModel(name=name, provider=provider, label=label)
    return [models[name] for name in sorted(models)]
