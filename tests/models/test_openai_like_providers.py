from __future__ import annotations

from collections.abc import Callable

import pytest

from pydantic_ai._json_schema import JsonSchema
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer


def _openai_transformer_factory(schema: JsonSchema) -> OpenAIJsonSchemaTransformer:
    return OpenAIJsonSchemaTransformer(schema, strict=True)


TransformerFactory = Callable[[JsonSchema], OpenAIJsonSchemaTransformer]


@pytest.mark.parametrize('transformer_factory', [_openai_transformer_factory])
def test_openai_compatible_transformers_flatten_allof(
    transformer_factory: TransformerFactory,
) -> None:
    schema: JsonSchema = {
        'type': 'object',
        'allOf': [
            {
                'type': 'object',
                'properties': {'foo': {'type': 'string'}},
                'required': ['foo'],
            },
            {
                'type': 'object',
                'properties': {'bar': {'type': 'integer'}},
                'required': ['bar'],
            },
        ],
    }

    transformer = transformer_factory(schema)
    transformed = transformer.walk()

    # allOf should have been flattened by the transformer
    assert 'allOf' not in transformed
    assert transformed['type'] == 'object'
    assert set(transformed.get('required', [])) == {'foo', 'bar'}
    assert transformed['properties']['foo']['type'] == 'string'
    assert transformed['properties']['bar']['type'] == 'integer'
