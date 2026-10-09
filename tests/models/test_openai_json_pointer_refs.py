"""OpenAI strict mode only resolves a `$ref` to the root (`#`) or to a top-level definition.

The MCP TypeScript SDK converts zod v3 tool schemas with `zod-to-json-schema`, whose default
`$refStrategy: 'root'` points a reused subschema at its first occurrence, e.g. `#/properties/from`.
Strict mode answers that with a 400, so a schema carrying one must not infer `strict=True`.
"""

from __future__ import annotations as _annotations

from typing import Any

import pytest
from pydantic import BaseModel

from pydantic_ai import Agent, Tool, ToolReturnPart

from .._inline_snapshot import snapshot
from ..conftest import RequestCapture, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
]

ADDRESS_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {'street': {'type': 'string'}, 'city': {'type': 'string'}},
    'required': ['street', 'city'],
    'additionalProperties': False,
}

# The input schema the MCP TypeScript SDK sends for a zod v3 tool `{ from: Address, to: Address }`.
MCP_POINTER_REF_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {'from': ADDRESS_SCHEMA, 'to': {'$ref': '#/properties/from'}},
    'required': ['from', 'to'],
    'additionalProperties': False,
    '$schema': 'http://json-schema.org/draft-07/schema#',
}

# The input schema the MCP TypeScript SDK sends for a zod v4 tool `{ tree: Tree }` with a recursive `Tree`: zod v4 puts
# the recursive subschema under draft-07 `definitions`, and leaves `additionalProperties` unset.
MCP_ZOD4_RECURSIVE_SCHEMA: dict[str, Any] = {
    '$schema': 'http://json-schema.org/draft-07/schema#',
    'type': 'object',
    'properties': {'tree': {'$ref': '#/definitions/__schema0'}},
    'required': ['tree'],
    'definitions': {
        '__schema0': {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
                'children': {'type': 'array', 'items': {'$ref': '#/definitions/__schema0'}},
            },
            'required': ['name', 'children'],
        }
    },
}


@pytest.mark.parametrize(
    'schema,definitions,strict_compatible',
    [
        pytest.param(
            MCP_ZOD4_RECURSIVE_SCHEMA,
            snapshot(
                {
                    '__schema0': {
                        'type': 'object',
                        'properties': {
                            'name': {'type': 'string'},
                            'children': {'type': 'array', 'items': {'$ref': '#/definitions/__schema0'}},
                        },
                        'required': ['name', 'children'],
                        'additionalProperties': False,
                    }
                }
            ),
            True,
            id='zod-v4-recursion',
        ),
        pytest.param(
            {
                'type': 'object',
                'properties': {'x': {'$ref': '#/definitions/X'}},
                'required': ['x'],
                'definitions': {
                    'X': {
                        'type': 'object',
                        'properties': {'a': {'type': 'string'}, 'b': {'type': 'string'}},
                        'required': ['a'],
                    }
                },
            },
            snapshot(
                {
                    'X': {
                        'type': 'object',
                        'properties': {'a': {'type': 'string'}, 'b': {'type': 'string'}},
                        'required': ['a'],
                        'additionalProperties': False,
                    }
                }
            ),
            False,
            id='optional-property',
        ),
    ],
)
def test_definitions_are_walked_like_defs(schema: dict[str, Any], definitions: dict[str, Any], strict_compatible: bool):
    """`definitions` entries get the same strict-mode handling as `$defs` entries, and count toward inferring it.

    Unit test: the VCR test below records the zod v4 shape going out strict; this pins the transformed entries and a
    `definitions` entry that keeps the schema from being strict-compatible.
    """
    transformer = OpenAIJsonSchemaTransformer(schema)

    assert transformer.walk()['definitions'] == definitions
    assert transformer.is_strict_compatible is strict_compatible


@pytest.mark.parametrize(
    'ref,strict_compatible',
    [
        pytest.param('#/properties/from', False, id='property'),
        pytest.param('#/properties/list/items', False, id='array-items'),
        pytest.param('#/$defs/Address/properties/city', False, id='inside-defs'),
        pytest.param('#/definitions/Address/properties/city', False, id='inside-definitions'),
        pytest.param('#', True, id='root'),
        pytest.param('#/$defs/Address', True, id='defs'),
        pytest.param('#/definitions/Address', True, id='definitions'),
    ],
)
def test_ref_strict_compatibility(ref: str, strict_compatible: bool):
    """Only a `$ref` to the root or to a top-level definition leaves the schema strict-compatible.

    Each case was checked against the live API with `strict` on and off: the incompatible ones are
    rejected with a 400 in strict mode only. Unit test: the VCR tests below record the MCP shape,
    and one recording per `$ref` form would pin the same inference.
    """
    schema: dict[str, Any] = {
        'type': 'object',
        'properties': {'from': ADDRESS_SCHEMA, 'list': {'type': 'array', 'items': ADDRESS_SCHEMA}, 'to': {'$ref': ref}},
        'required': ['from', 'list', 'to'],
        'additionalProperties': False,
        '$defs': {'Address': ADDRESS_SCHEMA},
        'definitions': {'Address': ADDRESS_SCHEMA},
    }
    transformer = OpenAIJsonSchemaTransformer(schema)

    assert transformer.walk()['properties']['to'] == {'$ref': ref}
    assert transformer.is_strict_compatible is strict_compatible


@pytest.mark.parametrize('strict', [True, False])
def test_explicit_strict_ignores_json_pointer_ref(strict: bool):
    """With `strict` set explicitly, a JSON-pointer `$ref` leaves `is_strict_compatible` as it was.

    Unit test: only an inferred `strict` reads the flag, so no request can show it.
    """
    schema: dict[str, Any] = {
        'type': 'object',
        'properties': {'from': ADDRESS_SCHEMA, 'to': {'$ref': '#/properties/from'}},
        'required': ['from', 'to'],
        'additionalProperties': False,
    }
    transformer = OpenAIJsonSchemaTransformer(schema, strict=strict)

    assert transformer.walk()['properties']['to'] == {'$ref': '#/properties/from'}
    assert transformer.is_strict_compatible is True


def test_recursive_model_stays_strict_compatible():
    """A recursive model's self-reference is rewritten to `#`, which strict mode resolves.

    Unit test: pins the inference for the shape the transformer itself produces; no request is needed
    to show it stays strict.
    """

    class Node(BaseModel):
        name: str
        children: list[Node]

    transformer = OpenAIJsonSchemaTransformer(Node.model_json_schema())

    assert transformer.walk()['properties']['children'] == {'type': 'array', 'items': {'$ref': '#'}}
    assert transformer.is_strict_compatible is True


SENT_PARAMETERS: dict[str, Any] = {
    'type': 'object',
    'properties': {'from': ADDRESS_SCHEMA, 'to': {'$ref': '#/properties/from'}},
    'required': ['from', 'to'],
    'additionalProperties': False,
}


@pytest.mark.parametrize(
    'api,sent_tools',
    [
        pytest.param(
            'chat',
            [
                {
                    'type': 'function',
                    'function': {
                        'name': 'ship_order',
                        'description': 'Ship an order from one address to another',
                        'parameters': SENT_PARAMETERS,
                    },
                }
            ],
            id='chat',
        ),
        pytest.param(
            'responses',
            [
                {
                    'name': 'ship_order',
                    'parameters': SENT_PARAMETERS,
                    'type': 'function',
                    'description': 'Ship an order from one address to another',
                    'strict': False,
                }
            ],
            id='responses',
        ),
    ],
)
async def test_mcp_json_pointer_ref_tool(
    allow_model_requests: None,
    openai_api_key: str,
    request_capture: RequestCapture,
    api: str,
    sent_tools: list[dict[str, Any]],
):
    """A tool with a JSON-pointer `$ref` is sent as is but without `strict`, and the model calls it.

    Chat Completions omits `strict` when it's off; the Responses API sends `strict: False`.
    """

    def ship_order(**kwargs: Any) -> str:
        return f'shipped from {kwargs["from"]["city"]} to {kwargs["to"]["city"]}'

    tool = Tool.from_schema(
        ship_order,
        name='ship_order',
        description='Ship an order from one address to another',
        json_schema=MCP_POINTER_REF_SCHEMA,
    )
    provider = OpenAIProvider(api_key=openai_api_key, http_client=request_capture.client)
    model = (
        OpenAIChatModel('gpt-4.1-mini', provider=provider)
        if api == 'chat'
        else OpenAIResponsesModel('gpt-4.1-mini', provider=provider)
    )

    result = await Agent(model, tools=[tool]).run(
        'Ship an order from 1 Main St, Springfield to 9 Elm Rd, Shelbyville, then say done.'
    )

    tool_returns = [
        part.content for message in result.all_messages() for part in message.parts if isinstance(part, ToolReturnPart)
    ]
    assert tool_returns == snapshot(['shipped from Springfield to Shelbyville'])
    assert request_capture.body()['tools'] == sent_tools


@pytest.mark.parametrize(
    'api,tool_returns,sent_tools',
    [
        pytest.param(
            'chat',
            snapshot(['saved root with 1 children']),
            snapshot(
                [
                    {
                        'type': 'function',
                        'function': {
                            'name': 'save_tree',
                            'description': 'Save a tree',
                            'parameters': {
                                'type': 'object',
                                'properties': {'tree': {'$ref': '#/definitions/__schema0'}},
                                'required': ['tree'],
                                'definitions': {
                                    '__schema0': {
                                        'type': 'object',
                                        'properties': {
                                            'name': {'type': 'string'},
                                            'children': {'type': 'array', 'items': {'$ref': '#/definitions/__schema0'}},
                                        },
                                        'required': ['name', 'children'],
                                        'additionalProperties': False,
                                    }
                                },
                                'additionalProperties': False,
                            },
                            'strict': True,
                        },
                    }
                ]
            ),
            id='chat',
        ),
        pytest.param(
            'responses',
            snapshot(['saved root with 1 children']),
            snapshot(
                [
                    {
                        'name': 'save_tree',
                        'parameters': {
                            'type': 'object',
                            'properties': {'tree': {'$ref': '#/definitions/__schema0'}},
                            'required': ['tree'],
                            'definitions': {
                                '__schema0': {
                                    'type': 'object',
                                    'properties': {
                                        'name': {'type': 'string'},
                                        'children': {'type': 'array', 'items': {'$ref': '#/definitions/__schema0'}},
                                    },
                                    'required': ['name', 'children'],
                                    'additionalProperties': False,
                                }
                            },
                            'additionalProperties': False,
                        },
                        'type': 'function',
                        'description': 'Save a tree',
                        'strict': True,
                    }
                ]
            ),
            id='responses',
        ),
    ],
)
async def test_mcp_zod4_definitions_tool(
    allow_model_requests: None,
    openai_api_key: str,
    request_capture: RequestCapture,
    api: str,
    tool_returns: list[str],
    sent_tools: list[dict[str, Any]],
):
    """A zod v4 tool whose recursive subschema sits under `definitions` is sent strict, and the model calls it."""

    def save_tree(**kwargs: Any) -> str:
        return f'saved {kwargs["tree"]["name"]} with {len(kwargs["tree"]["children"])} children'

    tool = Tool.from_schema(
        save_tree, name='save_tree', description='Save a tree', json_schema=MCP_ZOD4_RECURSIVE_SCHEMA
    )
    provider = OpenAIProvider(api_key=openai_api_key, http_client=request_capture.client)
    model = (
        OpenAIChatModel('gpt-4.1-mini', provider=provider)
        if api == 'chat'
        else OpenAIResponsesModel('gpt-4.1-mini', provider=provider)
    )

    result = await Agent(model, tools=[tool]).run(
        'Save a tree named root with one child named leaf that has no children, then say done.'
    )

    assert [
        part.content for message in result.all_messages() for part in message.parts if isinstance(part, ToolReturnPart)
    ] == tool_returns
    assert request_capture.body()['tools'] == sent_tools
