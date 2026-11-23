"""Each provider SDK extra keeps the upper bound https://github.com/pydantic/pydantic-ai/issues/11919 decided.

A provider SDK extra installs its SDK below the next major that hasn't been tested
(https://github.com/pydantic/pydantic-ai/pull/13316). Without the bound, `pip install pydantic-ai-slim[<extra>]`
resolves a new major the day it ships, before the adapter has run against it.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from ._source_roots import REPO_ROOT
from .manifests import Exempt, assert_classified

_SLIM = tomllib.loads((REPO_ROOT / 'pydantic_ai_slim' / 'pyproject.toml').read_text(encoding='utf-8'))
EXTRAS: dict[str, list[str]] = _SLIM['tool']['hatch']['metadata']['hooks']['uv-dynamic-versioning'][
    'optional-dependencies'
]


@dataclass(frozen=True)
class ProviderSdk:
    """An extra that installs a provider's SDK, which must carry an upper bound."""

    sdks: tuple[str, ...]

    @property
    def reason(self) -> str:
        return f'installs the provider SDK {", ".join(self.sdks)}'


_TOOL = Exempt('a tool integration, outside the provider SDK decision')
_BUNDLE = Exempt('bundles other extras, each classified on its own')
_ENGINE = Exempt('a durable execution engine, outside the provider SDK decision')
_SERVING = Exempt('serves or adapts agents over HTTP, outside the provider SDK decision')

POLICY: dict[str, ProviderSdk | Exempt] = {
    'logfire': Exempt('observability, outside the provider SDK decision'),
    'openai': ProviderSdk(('openai',)),
    'cohere': ProviderSdk(('cohere',)),
    'google': ProviderSdk(('google-genai',)),
    'anthropic': ProviderSdk(('anthropic',)),
    'xai': ProviderSdk(('xai-sdk',)),
    'groq': ProviderSdk(('groq',)),
    'openrouter': ProviderSdk(('openai',)),
    'mistral': ProviderSdk(('mistralai',)),
    'bedrock': ProviderSdk(('boto3',)),
    'bedrock-mantle': ProviderSdk(('openai', 'botocore')),
    'zai': ProviderSdk(('openai',)),
    'snowflake': ProviderSdk(('openai',)),
    'crusoe': ProviderSdk(('openai',)),
    'cerebras': ProviderSdk(('openai',)),
    'typesafe': ProviderSdk(('typesafe-sdk',)),
    'huggingface': ProviderSdk(('huggingface-hub',)),
    'sentence-transformers': Exempt('runs embedding models locally rather than calling a provider API'),
    'voyageai': ProviderSdk(('voyageai',)),
    'duckduckgo': _TOOL,
    'tavily': _TOOL,
    'exa': _TOOL,
    'web-fetch': _TOOL,
    'cli': Exempt('the terminal client, outside the provider SDK decision'),
    'realtime': Exempt('the realtime transport, outside the provider SDK decision'),
    'openai-realtime': _BUNDLE,
    'google-realtime': _BUNDLE,
    'xai-realtime': _BUNDLE,
    'elevenlabs-agents-realtime': _BUNDLE,
    'elevenlabs-realtime': _BUNDLE,
    'mcp': Exempt('the MCP client, outside the provider SDK decision'),
    'mcp-tasks': Exempt('the MCP client, outside the provider SDK decision'),
    'evals': Exempt('a workspace package, pinned to the same release'),
    'ui': _SERVING,
    'ag-ui': _SERVING,
    'web': _SERVING,
    'retries': Exempt('HTTP retry helpers, outside the provider SDK decision'),
    'temporal': _ENGINE,
    'dbos': _ENGINE,
    'prefect': _ENGINE,
    'spec': Exempt('agent spec loading, outside the provider SDK decision'),
}
"""Every `pydantic-ai-slim` extra, and whether the provider SDK cap applies to it."""


def test_every_extra_is_classified():
    assert_classified(EXTRAS, POLICY, complete_walk=True, table_name='POLICY')


def test_every_provider_sdk_extra_caps_its_sdk():
    uncapped: list[str] = []
    for extra, entry in POLICY.items():
        if not isinstance(entry, ProviderSdk) or extra not in EXTRAS:
            continue
        requirements = {
            canonicalize_name(requirement.name): requirement for requirement in map(Requirement, EXTRAS[extra])
        }
        for sdk in entry.sdks:
            requirement = requirements.get(canonicalize_name(sdk))
            if requirement is None or not any(
                spec.operator in ('<', '<=', '==', '~=') for spec in requirement.specifier
            ):
                uncapped.append(f'{extra}: {requirement or sdk}')
    assert uncapped == [], 'cap each provider SDK below its next untested major:\n' + '\n'.join(uncapped)
