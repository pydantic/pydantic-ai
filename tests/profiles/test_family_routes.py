"""Every route that names a model by its vendor keeps the family facts of that model (R23).

The shared family map in `profiles/_family.py` gives each vendor's models their family profile. A route resolves a
model id through the map, its own vendor table, or its own overlay, and nothing else checked that the family facts
survive the trip: #12383 had to route Kimi on Vercel, whose own table lacked the `moonshotai` vendor, and #13187 had to
restore GLM-5.3's inlined `$defs` that Bedrock's overlay dropped.

The family facts are the family-wide contracts https://github.com/pydantic/pydantic-ai/issues/12104 names: whether the
model thinks, whether it always does, and whether its JSON schemas get their `$defs` inlined. A route may add facts,
but may not lose one. Each route is spelled the way it names models; a route that names models without a vendor is
exempt, with its reason.

`ROUTE_GAPS` lists the families each route loses today, with the issue that owns closing the gap. It is checked both
ways: a new loss fails until it is listed, and a listed family that a route now keeps fails until it is removed.

Offline: `Provider.model_profile()` only.
"""

from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

import pytest

from pydantic_ai import providers
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.profiles._family import _VENDOR_FAMILY_PROFILES  # pyright: ignore[reportPrivateUsage]
from pydantic_ai.providers import Provider

from ..capability_cases import caused_by_missing_dependency
from ..manifests import Exempt, assert_classified

FAMILY_PROBES: dict[str, tuple[str, ...]] = {
    'amazon': ('nova-2-lite-v1',),
    'anthropic': ('claude-sonnet-4-6',),
    'cohere': ('command-a-reasoning-08-2025',),
    'deepseek': ('deepseek-r1', 'deepseek-v4-flash'),
    'google': ('gemini-2.5-pro',),
    'inclusionai': ('ling-3',),
    'meta': ('llama-4-maverick',),
    'minimax': ('minimax-m2.7',),
    'mistral': ('magistral-medium-latest',),
    'moonshotai': ('kimi-k3',),
    'nvidia': ('nemotron-3-super',),
    'openai': ('gpt-5.4', 'gpt-oss-120b'),
    'qwen': ('qwen3-235b-a22b',),
    'stepfun': ('step-5-preview',),
    'tencent': ('hy4',),
    'thinkingmachines': ('inkling',),
    'xai': ('grok-4.3',),
    'xiaomi': ('mimo-v2.6',),
    'zai': ('glm-5.3',),
}
"""Model names, as the family profile gets them after the vendor, for each family in the shared family map.

A family is named by one of its vendors in the map; the matrix tries all of them on each route.
"""

_REF_SCHEMA = {
    'type': 'object',
    'properties': {'a': {'$ref': '#/$defs/A'}},
    '$defs': {'A': {'type': 'object', 'properties': {'x': {'type': 'string'}}}},
}


def _facts(profile: ModelProfile | None) -> frozenset[str]:
    """The family facts `profile` states."""
    if profile is None:
        return frozenset()
    facts: set[str] = set()
    if profile.get('supports_thinking'):
        facts.add('thinks')
    if profile.get('thinking_always_enabled'):
        facts.add('always thinks')
    # Checked by its output rather than by the transformer class, since a route may inline `$defs` with its own one.
    if (transformer := profile.get('json_schema_transformer')) is not None:
        if '$ref' not in json.dumps(transformer(json.loads(json.dumps(_REF_SCHEMA))).walk()):
            facts.add('inlines $defs')
    return frozenset(facts)


@dataclass(frozen=True)
class Spelled:
    """A route that names a family's model by its vendor."""

    spell: Callable[[str, str], str]
    """The route's id for a vendor and a model name."""
    reason: str


_VENDOR_SLASH_MODEL = Spelled(lambda vendor, model: f'{vendor}/{model}', 'Names a model `vendor/model`.')
_OWN_FAMILY = Exempt('Serves its own models, under names with no vendor.')
_UNPREFIXED = Exempt('Names models from several vendors without the vendor, matching each family by name.')
_NOT_CHAT = Exempt('Serves no chat model.')

ROUTES: dict[str, Spelled | Exempt] = {
    'AlibabaProvider': _OWN_FAMILY,
    'AnthropicProvider': _OWN_FAMILY,
    'AzureProvider': _UNPREFIXED,
    'BedrockMantleProvider': Exempt('Serves only OpenAI models, through its own profile.'),
    'BedrockProvider': Spelled(lambda vendor, model: f'{vendor}.{model}', 'Names a model `vendor.model`.'),
    'CerebrasProvider': _UNPREFIXED,
    'CohereProvider': _OWN_FAMILY,
    'CrusoeProvider': _VENDOR_SLASH_MODEL,
    'DeepSeekProvider': _OWN_FAMILY,
    'ElevenLabsProvider': _NOT_CHAT,
    'FireworksProvider': _UNPREFIXED,
    'GitHubCopilotProvider': _UNPREFIXED,
    'GitHubProvider': _VENDOR_SLASH_MODEL,
    'GoogleCloudProvider': _OWN_FAMILY,
    'GoogleProvider': _OWN_FAMILY,
    'GroqProvider': _VENDOR_SLASH_MODEL,
    'HerokuProvider': _UNPREFIXED,
    'HuggingFaceProvider': _VENDOR_SLASH_MODEL,
    'LiteLLMProvider': _VENDOR_SLASH_MODEL,
    'MetaProvider': _OWN_FAMILY,
    'MistralProvider': _OWN_FAMILY,
    'MoonshotAIProvider': _OWN_FAMILY,
    'NebiusProvider': _VENDOR_SLASH_MODEL,
    'OVHcloudProvider': _UNPREFIXED,
    'OllamaProvider': _UNPREFIXED,
    'OpenAIChatGPTProvider': _OWN_FAMILY,
    'OpenAICodexProvider': _OWN_FAMILY,
    'OpenAICompatibleProvider': _VENDOR_SLASH_MODEL,
    'OpenAIDecisionsProvider': _NOT_CHAT,
    'OpenAIProvider': _OWN_FAMILY,
    'OpenRouterProvider': _VENDOR_SLASH_MODEL,
    'OtariProvider': Spelled(
        lambda vendor, model: f'together:{vendor}/{model}', 'Names a model `host:vendor/model`; Together is one host.'
    ),
    'SambaNovaProvider': _UNPREFIXED,
    'SentenceTransformersProvider': _NOT_CHAT,
    'SnowflakeProvider': _UNPREFIXED,
    'StavProvider': _VENDOR_SLASH_MODEL,
    'SystemOneProvider': _NOT_CHAT,
    'TogetherProvider': _VENDOR_SLASH_MODEL,
    'TypeSafeProvider': _NOT_CHAT,
    'VLLMProvider': _VENDOR_SLASH_MODEL,
    'VercelProvider': _VENDOR_SLASH_MODEL,
    'VoyageAIProvider': _NOT_CHAT,
    'XaiProvider': _OWN_FAMILY,
    'ZaiProvider': _OWN_FAMILY,
}
"""How each `Provider` names a model, or why the matrix leaves it out."""

_SHARED_RESOLVER = 'https://github.com/pydantic/pydantic-ai/issues/12101'
_VERCEL_MAP = 'https://github.com/pydantic/pydantic-ai/issues/12074'
_CROSS_ROUTE = 'https://github.com/pydantic/pydantic-ai/issues/12104'

ROUTE_GAPS: dict[str, dict[str, str]] = {
    # Bedrock keeps its own vendor table and overlays, off the shared map.
    'BedrockProvider': dict.fromkeys(
        ('inclusionai', 'nvidia', 'openai', 'qwen', 'stepfun', 'tencent', 'thinkingmachines', 'xiaomi'), _CROSS_ROUTE
    ),
    'CrusoeProvider': dict.fromkeys(
        (
            'amazon',
            'anthropic',
            'cohere',
            'inclusionai',
            'minimax',
            'mistral',
            'nvidia',
            'stepfun',
            'tencent',
            'thinkingmachines',
            'xai',
            'xiaomi',
        ),
        _SHARED_RESOLVER,
    ),
    'GitHubProvider': dict.fromkeys(
        (
            'amazon',
            'anthropic',
            'google',
            'inclusionai',
            'minimax',
            'mistral',
            'moonshotai',
            'nvidia',
            'openai',
            'qwen',
            'stepfun',
            'tencent',
            'thinkingmachines',
            'xiaomi',
            'zai',
        ),
        _SHARED_RESOLVER,
    ),
    'GroqProvider': dict.fromkeys(
        (
            'amazon',
            'anthropic',
            'cohere',
            'deepseek',
            'google',
            'inclusionai',
            'minimax',
            'mistral',
            'moonshotai',
            'nvidia',
            'openai',
            'stepfun',
            'tencent',
            'thinkingmachines',
            'xai',
            'xiaomi',
            'zai',
        ),
        _SHARED_RESOLVER,
    ),
    'HuggingFaceProvider': dict.fromkeys(
        (
            'amazon',
            'anthropic',
            'cohere',
            'inclusionai',
            'minimax',
            'nvidia',
            'openai',
            'stepfun',
            'tencent',
            'thinkingmachines',
            'xai',
            'xiaomi',
        ),
        _SHARED_RESOLVER,
    ),
    'NebiusProvider': dict.fromkeys(
        (
            'amazon',
            'anthropic',
            'cohere',
            'inclusionai',
            'minimax',
            'nvidia',
            'stepfun',
            'tencent',
            'thinkingmachines',
            'xai',
            'xiaomi',
        ),
        _SHARED_RESOLVER,
    ),
    # OpenRouter's MiniMax profile keeps the always-on thinking but not the inlined `$defs`.
    'OpenRouterProvider': {'minimax': _CROSS_ROUTE},
    'TogetherProvider': {
        'openai': (
            "Together's only OpenAI models are gpt-oss, so `providers/together.py:_gpt_oss_only_profile` gives other"
            ' `openai/*` ids no OpenAI profile.'
        )
    },
    'VLLMProvider': dict.fromkeys(
        (
            'amazon',
            'anthropic',
            'google',
            'inclusionai',
            'minimax',
            'moonshotai',
            'nvidia',
            'openai',
            'stepfun',
            'tencent',
            'thinkingmachines',
            'xai',
            'xiaomi',
        ),
        _SHARED_RESOLVER,
    ),
    'VercelProvider': dict.fromkeys(('amazon', 'google', 'inclusionai', 'meta', 'minimax', 'qwen'), _VERCEL_MAP),
}
"""The families each spelled route loses a fact of today, each with the issue that owns the gap or the reason the
route differs on purpose."""


@cache
def _walk_providers() -> tuple[dict[str, type[Provider[object]]], tuple[str, ...]]:
    """Every concrete `Provider` class in `pydantic_ai.providers`, and the modules skipped for a missing extra."""
    found: dict[str, type[Provider[object]]] = {}
    skipped: list[str] = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for module_info in pkgutil.iter_modules(providers.__path__, f'{providers.__name__}.'):
            try:
                module = importlib.import_module(module_info.name)
            except Exception as exc:
                if not caused_by_missing_dependency(exc):
                    raise
                skipped.append(module_info.name)
                continue
            for obj in vars(module).values():
                if (
                    inspect.isclass(obj)
                    and obj.__module__ == module.__name__
                    and not inspect.isabstract(obj)
                    and issubclass(obj, Provider)
                ):
                    found[obj.__name__] = obj
    return found, tuple(skipped)


def _lost_families(route: type[Provider[object]], spelled: Spelled) -> set[str]:
    """The families with a probe whose facts no vendor of the family keeps on `route`."""
    lost: set[str] = set()
    for family, model_names in FAMILY_PROBES.items():
        family_profile = _VENDOR_FAMILY_PROFILES[family]
        vendors = [vendor for vendor, profile in _VENDOR_FAMILY_PROFILES.items() if profile is family_profile]
        for model_name in model_names:
            required = _facts(family_profile(model_name))
            if not any(required <= _facts(route.model_profile(spelled.spell(v, model_name))) for v in vendors):
                lost.add(family)
    return lost


def test_every_family_has_a_probe():
    """Each family in the shared map has a probe, and each probe states a fact a route could lose."""
    named = [_VENDOR_FAMILY_PROFILES[family] for family in FAMILY_PROBES]
    assert len(set(named)) == len(named), 'two `FAMILY_PROBES` keys name the same family'
    assert set(named) == set(_VENDOR_FAMILY_PROFILES.values())
    vacuous = [
        f'{family}/{model_name}'
        for family, model_names in FAMILY_PROBES.items()
        for model_name in model_names
        if not _facts(_VENDOR_FAMILY_PROFILES[family](model_name))
    ]
    assert not vacuous, f'these probes state no family fact, so they check nothing: {vacuous}'


def test_every_provider_is_classified():
    """A new `Provider` says how it names models before the matrix can check it, or why it is left out."""
    found, skipped = _walk_providers()
    assert_classified(found, ROUTES, complete_walk=not skipped, table_name='ROUTES')
    assert set(ROUTE_GAPS) <= {name for name, route in ROUTES.items() if isinstance(route, Spelled)}


@pytest.mark.parametrize('route_name', [name for name, route in ROUTES.items() if isinstance(route, Spelled)])
def test_route_keeps_family_facts(route_name: str):
    """The families `route_name` loses a fact of are exactly the ones `ROUTE_GAPS` lists for it."""
    spelled = ROUTES[route_name]
    assert isinstance(spelled, Spelled)
    found, _ = _walk_providers()
    if route_name not in found:
        pytest.skip(f'{route_name} needs an extra that is not installed')
    listed = set(ROUTE_GAPS.get(route_name, {}))
    lost = _lost_families(found[route_name], spelled)
    assert not (lost - listed), f'{route_name} loses family facts of {sorted(lost - listed)}: fix it, or list it'
    assert not (listed - lost), f'{route_name} keeps the family facts of {sorted(listed - lost)} now: unlist it'
