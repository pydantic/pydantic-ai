#!/usr/bin/env python3
"""Pre-commit hook to update model pricing in popular-models.md from genai-prices"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from urllib.request import urlopen

from typing_extensions import NotRequired, TypedDict

# Skip in CI
if os.environ.get('CI'):
    sys.exit(0)

PRICES_URL = 'https://raw.githubusercontent.com/pydantic/genai-prices/refs/heads/main/prices/data.json'
DOCS_DIR = Path(__file__).parent.parent
CONFIG_FILE = DOCS_DIR / 'models/popular-models-config.json'
CACHE_FILE = DOCS_DIR / '.model-prices-cache.json'
TARGET_FILE = DOCS_DIR / 'models/popular-models.md'


# Config types
class ModelMapping(TypedDict):
    provider: str
    model_id: str
    default_price: NotRequired[str]


class Config(TypedDict):
    models: dict[str, ModelMapping]


# Tiered pricing types
class PriceTier(TypedDict):
    start: int
    price: float


class TieredPrice(TypedDict):
    base: float
    tiers: list[PriceTier]


Price = float | int | TieredPrice


# genai-prices API types
class ModelPrices(TypedDict, total=False):
    input_mtok: Price
    output_mtok: Price


class GenaIPricesModel(TypedDict):
    id: str
    prices: NotRequired[ModelPrices | list[ModelPrices]]


class GenaIPricesProvider(TypedDict):
    id: str
    models: list[GenaIPricesModel]


# Cached prices type
class CachedPrice(TypedDict):
    input_mtok: Price
    output_mtok: Price


def load_config() -> Config:
    """Load model mapping config"""
    return json.loads(CONFIG_FILE.read_text())


def fetch_prices() -> dict[str, CachedPrice]:
    """Fetch prices from genai-prices and cache locally"""
    with urlopen(PRICES_URL) as resp:
        data: list[GenaIPricesProvider] = json.loads(resp.read())

    prices: dict[str, CachedPrice] = {}
    for provider in data:
        provider_id = provider['id']
        for model in provider.get('models', []):
            model_id = model['id']
            raw_prices = model.get('prices')
            if raw_prices is None:
                continue

            price_data: ModelPrices
            if isinstance(raw_prices, list):
                price_data = raw_prices[0] if raw_prices else {}
            else:
                price_data = raw_prices

            input_price = price_data.get('input_mtok')
            output_price = price_data.get('output_mtok')

            if input_price is not None and output_price is not None:
                prices[f'{provider_id}:{model_id}'] = {
                    'input_mtok': input_price,
                    'output_mtok': output_price,
                }

    CACHE_FILE.write_text(json.dumps(prices, indent=2))
    return prices


def format_price(input_p: Price, output_p: Price) -> str:
    """Format price, handling tiered pricing"""

    def fmt(p: Price) -> str:
        if isinstance(p, dict):
            base = float(p['base'])
            tiers = p.get('tiers')
            tier_price = float(tiers[0]['price']) if tiers else base
            if base != tier_price:
                return f'${base:.2f}-{tier_price:.2f}'
            return f'${base:.2f}'
        return f'${float(p):.2f}'

    return f'{fmt(input_p)} / {fmt(output_p)} per 1M tokens (in/out)'


def get_price_for_model(
    doc_model_id: str, config: Config, prices: dict[str, CachedPrice]
) -> tuple[str | None, str | None]:
    """Get formatted price for a model, or None if not found"""
    model_config = config['models'].get(doc_model_id)
    if not model_config:
        return None, f'Model {doc_model_id!r} not in config'

    # Check for default price override
    if 'default_price' in model_config:
        return model_config['default_price'], None

    # Lookup in fetched prices
    provider = model_config['provider']
    model_id = model_config['model_id']
    key = f'{provider}:{model_id}'

    price_data = prices.get(key)
    if not price_data:
        return None, f'Price not found for {key!r} (doc model: {doc_model_id!r})'

    return format_price(price_data['input_mtok'], price_data['output_mtok']), None


def update_pricing(content: str, config: Config, prices: dict[str, CachedPrice]) -> tuple[str, list[str]]:
    """Update pricing lines in markdown content"""
    lines = content.split('\n')
    result: list[str] = []
    current_model_id: str | None = None
    errors: list[str] = []

    for line in lines:
        # Detect model ID line: ID: `provider:model`
        if line.startswith('ID: `'):
            match = re.search(r'ID: `([^`]+)`', line)
            if match:
                current_model_id = match.group(1)

        # Update pricing line
        if line.startswith('**Pricing:**') and current_model_id:
            price, error = get_price_for_model(current_model_id, config, prices)
            if price:
                line = f'**Pricing:** {price}'
            elif error:
                errors.append(error)
            current_model_id = None  # Reset after processing

        result.append(line)

    return '\n'.join(result), errors


def main() -> int:
    config = load_config()
    prices = fetch_prices()
    content = TARGET_FILE.read_text()
    updated, errors = update_pricing(content, config, prices)

    if errors:
        print('ERROR: Missing prices (add default_price to config or update genai-prices):')
        for error in errors:
            print(f'  - {error}')
        return 1

    if content != updated:
        TARGET_FILE.write_text(updated)
        print(f'Updated pricing in {TARGET_FILE}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
