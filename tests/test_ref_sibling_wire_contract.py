"""Wire-contract tests: preserved `$ref` sibling keywords reach provider APIs and are accepted.

Issue #6591 / PR #6592: when inlining `$defs`, keywords sitting beside a `$ref` (a field-level
`description` or `default`) must survive. They were previously dropped, so nothing proved providers
still accept the schema once the keywords are preserved. The risky keyword is `default` — OpenAI
strict structured outputs disallow it, and Gemini's schema validation is historically picky.

Each case runs a real (recorded) request and asserts BOTH:
1. the preserved siblings reach the request wire body (so the changed inlining branch ran and nothing
   downstream silently dropped them again), and
2. the provider accepted the schema and returned a valid structured output (no 4xx, output parses).

The two triggers of the changed branch are both covered:
- `StructuredDict` with `$defs` (`output.py`) is provider-agnostic, so OpenAI / Anthropic / Google
  exercise the branch even though their own transformers don't prefer inlined defs.
- The Qwen profile's `InlineDefsJsonSchemaTransformer` inlines a nested model's pydantic-emitted
  `$ref` siblings directly as the provider transformer (the real issue #6591 scenario end-to-end).

This is a VCR test rather than a unit test because the whole point is what a provider does with the
preserved keywords on the wire; a unit test would only re-pin the internal walk shape already covered
in `test_json_schema.py::test_inline_defs_preserves_ref_sibling_keywords`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pytest
from pydantic import BaseModel, Field, JsonValue
from vcr.cassette import Cassette

from pydantic_ai import Agent, StructuredDict

from .cassette_utils import single_request_body
from .conftest import try_import

with try_import() as openai_imports:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider

with try_import() as anthropic_imports:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as google_imports:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

from pydantic_ai.models import Model

pytestmark = [pytest.mark.anyio, pytest.mark.vcr]

# A distinctive marker so the recursive wire search can't match an unrelated `description`.
SENTINEL_DESCRIPTION = 'pyai-ref-sibling-sentinel: primary mailing address'

# `$ref` sibling `description` AND `default` on a nested field, plus a `$defs` block: the exact shape
# `InlineDefsJsonSchemaTransformer` inlines. `StructuredDict` runs the changed branch at construction.
STRUCTURED_DICT_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'name': {'type': 'string', 'description': 'The person full name'},
        'address': {
            '$ref': '#/$defs/Address',
            'description': SENTINEL_DESCRIPTION,
            'default': None,
        },
    },
    'required': ['name'],
    '$defs': {
        'Address': {
            'type': 'object',
            'properties': {'street': {'type': 'string'}, 'city': {'type': 'string'}},
            'required': ['street', 'city'],
        },
    },
}


class Address(BaseModel):
    street: str
    city: str


class Person(BaseModel):
    """Nested-model output for the Qwen profile path: pydantic emits `address` as a `$ref` carrying a
    sibling `description`, which the provider `InlineDefsJsonSchemaTransformer` must inline in place."""

    name: str
    address: Address = Field(description=SENTINEL_DESCRIPTION)


_PROVIDER_SKIP_MARKS: dict[str, pytest.MarkDecorator] = {
    'openai': pytest.mark.skipif(not openai_imports(), reason='openai not installed'),
    'anthropic': pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed'),
    'google': pytest.mark.skipif(not google_imports(), reason='google-genai not installed'),
    'openrouter': pytest.mark.skipif(not openai_imports(), reason='openai (openrouter) not installed'),
}


@dataclass(frozen=True)
class WireCase:
    id: str
    provider: str
    model_name: str
    output: Literal['structured_dict', 'nested_model']
    """`structured_dict` exercises the `output.py` inlining trigger; `nested_model` exercises the
    Qwen profile transformer trigger on real pydantic-emitted `$ref` siblings."""

    @property
    def marks(self) -> tuple[pytest.MarkDecorator, ...]:
        return (_PROVIDER_SKIP_MARKS[self.provider],)


CASES = [
    WireCase(id='openai-structured-dict', provider='openai', model_name='gpt-4o-mini', output='structured_dict'),
    WireCase(
        id='anthropic-structured-dict',
        provider='anthropic',
        model_name='claude-haiku-4-5',
        output='structured_dict',
    ),
    WireCase(id='google-structured-dict', provider='google', model_name='gemini-2.5-flash', output='structured_dict'),
    WireCase(
        id='qwen-nested-model',
        provider='openrouter',
        model_name='qwen/qwen3-30b-a3b-instruct-2507',
        output='nested_model',
    ),
]


def _build_model(
    case: WireCase, *, openai_api_key: str, anthropic_api_key: str, gemini_api_key: str, openrouter_api_key: str
) -> Model:
    if case.provider == 'openai':
        return OpenAIChatModel(case.model_name, provider=OpenAIProvider(api_key=openai_api_key))
    if case.provider == 'anthropic':
        return AnthropicModel(case.model_name, provider=AnthropicProvider(api_key=anthropic_api_key))
    if case.provider == 'google':
        return GoogleModel(case.model_name, provider=GoogleProvider(api_key=gemini_api_key))
    if case.provider == 'openrouter':
        return OpenAIChatModel(case.model_name, provider=OpenRouterProvider(api_key=openrouter_api_key))
    raise ValueError(f'unknown provider {case.provider!r}')  # pragma: no cover


def _find_sibling_node(body: JsonValue) -> dict[str, JsonValue] | None:
    """Recursively locate the inlined schema node carrying the sentinel field-level description.

    Provider-agnostic: each provider nests the tool/output schema differently, so we search the whole
    request body rather than hard-coding a path.
    """
    if isinstance(body, dict):
        if body.get('description') == SENTINEL_DESCRIPTION:
            return body
        for value in body.values():
            if (found := _find_sibling_node(value)) is not None:
                return found
    elif isinstance(body, list):
        for item in body:
            if (found := _find_sibling_node(item)) is not None:
                return found
    return None


@pytest.mark.parametrize('case', [pytest.param(c, id=c.id, marks=c.marks) for c in CASES])
async def test_ref_sibling_wire_contract(
    case: WireCase,
    allow_model_requests: None,
    openai_api_key: str,
    anthropic_api_key: str,
    gemini_api_key: str,
    openrouter_api_key: str,
    vcr: Cassette,
):
    """A `$ref` with preserved sibling keywords reaches the provider on the wire and is accepted."""
    model = _build_model(
        case,
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
        gemini_api_key=gemini_api_key,
        openrouter_api_key=openrouter_api_key,
    )
    output_type = StructuredDict(STRUCTURED_DICT_SCHEMA, name='person') if case.output == 'structured_dict' else Person
    agent = Agent(model, output_type=output_type)

    result = await agent.run('Record this person: Ada Lovelace, who lives at 12 Baker Street in London.')

    # 1. The preserved siblings reached the request wire body: the referenced `Address` def was inlined
    #    in place (no dangling `$ref`) and the field-level `description` survived rather than being
    #    silently dropped as before the fix.
    node = _find_sibling_node(single_request_body(vcr))
    assert node is not None, 'preserved sibling `description` did not reach the request wire body'
    assert '$ref' not in node, f'`$ref` was not inlined: {node}'
    assert node.get('type') == 'object'
    assert 'properties' in node
    if case.output == 'structured_dict':
        # The risky keyword: `default` was dropped before this fix. It now reaches the wire (non-strict on
        # OpenAI, passed through by Anthropic and Gemini) and every provider accepted it.
        assert 'default' in node and node['default'] is None, f'sibling `default` did not reach the wire: {node}'

    # 2. The provider accepted the schema (a 4xx would have raised) and returned a valid structured output.
    output = result.output
    if case.output == 'structured_dict':
        assert isinstance(output, dict)
        assert output['name']
    else:
        assert isinstance(output, Person)
        assert output.name
