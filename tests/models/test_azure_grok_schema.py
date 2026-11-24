from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, Literal

import httpx2
import pytest
from inline_snapshot import snapshot
from pydantic import Field, JsonValue
from typing_extensions import TypeAliasType

from pydantic_ai import Agent, NativeOutput

from ..conftest import RequestCapture, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
    from pydantic_ai.providers.azure import AzureProvider
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


@dataclass
class Replacement:
    old_text: str
    new_text: str


@dataclass
class Edits:
    replacements: list[Replacement]


@dataclass
class Node:
    value: str
    children: list[Node] | None = None


Key = TypeAliasType('Key', str)
Positive = TypeAliasType('Positive', Annotated[int, Field(gt=0)])


def edit_file(path: str, replacements: list[Replacement] | None = None) -> str:
    """Edit a file by exact string replacement."""
    return 'ok'  # pragma: no cover


def accept_keys(values: dict[Key, int]) -> str:
    return 'ok'  # pragma: no cover


def add_contains(schema: dict[str, JsonValue]) -> None:
    items = schema['items']
    assert isinstance(items, dict)
    options = items['anyOf']
    assert isinstance(options, list)
    schema['contains'] = deepcopy(options[0])


def accept_contains(values: Annotated[list[Positive | str], Field(json_schema_extra=add_contains)]) -> str:
    return 'ok'  # pragma: no cover


def accept_node(node: Node) -> str:
    return 'ok'  # pragma: no cover


_INFERRED_TOOL_STRICT_FIELDS: dict[str, tuple[bool, bool | None]] = {
    'chat': (False, None),
    'responses': (True, False),
}
_NATIVE_STRICT_FLAGS: dict[bool | None, bool] = {None: True, True: True, False: False}
_EDITS_OUTPUT = Edits([Replacement('old', 'new')])
_EDITS_CONTENT = '{"replacements": [{"old_text": "old", "new_text": "new"}]}'


@dataclass(frozen=True)
class Case:
    id: str
    tool_schemas: dict[bool | None, dict[str, JsonValue]]
    provider_name: Literal['azure', 'azure-custom', 'openai', 'openai-azure'] = 'azure'
    model_name: str = 'grok-4-1-fast-reasoning'
    function: Callable[..., str] = edit_file
    native_schema: dict[bool | None, dict[str, JsonValue]] | None = None
    expected_output: str | Edits = 'hello'
    response_content: str = 'hello'
    strict_modes: tuple[bool | None, ...] = (None, True, False)
    check_inferred_tool_strict: bool = False


CASES: list[Case] = [
    Case(
        'azure-grok',
        snapshot(
            {
                None: {
                    'additionalProperties': False,
                    'properties': {
                        'path': {'type': 'string'},
                        'replacements': {
                            'default': None,
                            'anyOf': [
                                {
                                    'items': {
                                        'properties': {'old_text': {'type': 'string'}, 'new_text': {'type': 'string'}},
                                        'required': ['old_text', 'new_text'],
                                        'type': 'object',
                                        'additionalProperties': False,
                                    },
                                    'type': 'array',
                                },
                                {'type': 'null'},
                            ],
                        },
                    },
                    'required': ['path'],
                    'type': 'object',
                },
                True: {
                    'additionalProperties': False,
                    'properties': {
                        'path': {'type': 'string'},
                        'replacements': {
                            'anyOf': [
                                {
                                    'items': {
                                        'properties': {'old_text': {'type': 'string'}, 'new_text': {'type': 'string'}},
                                        'required': ['old_text', 'new_text'],
                                        'type': 'object',
                                        'additionalProperties': False,
                                    },
                                    'type': 'array',
                                },
                                {'type': 'null'},
                            ]
                        },
                    },
                    'required': ['path', 'replacements'],
                    'type': 'object',
                },
                False: {
                    'additionalProperties': False,
                    'properties': {
                        'path': {'type': 'string'},
                        'replacements': {
                            'default': None,
                            'anyOf': [
                                {
                                    'items': {
                                        'properties': {'old_text': {'type': 'string'}, 'new_text': {'type': 'string'}},
                                        'required': ['old_text', 'new_text'],
                                        'type': 'object',
                                    },
                                    'type': 'array',
                                },
                                {'type': 'null'},
                            ],
                        },
                    },
                    'required': ['path'],
                    'type': 'object',
                },
            }
        ),
        model_name='Grok-4-1-fast-reasoning',
        check_inferred_tool_strict=True,
    ),
    Case(
        'azure-openai',
        snapshot(
            {
                None: {
                    'additionalProperties': False,
                    'properties': {
                        'path': {'type': 'string'},
                        'replacements': {
                            'default': None,
                            'anyOf': [{'items': {'$ref': '#/$defs/Replacement'}, 'type': 'array'}, {'type': 'null'}],
                        },
                    },
                    'required': ['path'],
                    'type': 'object',
                    '$defs': {
                        'Replacement': {
                            'properties': {'old_text': {'type': 'string'}, 'new_text': {'type': 'string'}},
                            'required': ['old_text', 'new_text'],
                            'type': 'object',
                            'additionalProperties': False,
                        }
                    },
                },
            }
        ),
        model_name='gpt-4o',
        strict_modes=(None,),
    ),
    Case(
        'generic-openai',
        snapshot(
            {
                None: {
                    'additionalProperties': False,
                    'properties': {
                        'path': {'type': 'string'},
                        'replacements': {
                            'default': None,
                            'anyOf': [{'items': {'$ref': '#/$defs/Replacement'}, 'type': 'array'}, {'type': 'null'}],
                        },
                    },
                    'required': ['path'],
                    'type': 'object',
                    '$defs': {
                        'Replacement': {
                            'properties': {'old_text': {'type': 'string'}, 'new_text': {'type': 'string'}},
                            'required': ['old_text', 'new_text'],
                            'type': 'object',
                            'additionalProperties': False,
                        }
                    },
                },
            }
        ),
        provider_name='openai',
        strict_modes=(None,),
    ),
    Case(
        'azure-grok-native',
        snapshot(
            {
                None: {
                    'additionalProperties': False,
                    'properties': {
                        'path': {'type': 'string'},
                        'replacements': {
                            'default': None,
                            'anyOf': [
                                {
                                    'items': {
                                        'properties': {'old_text': {'type': 'string'}, 'new_text': {'type': 'string'}},
                                        'required': ['old_text', 'new_text'],
                                        'type': 'object',
                                        'additionalProperties': False,
                                    },
                                    'type': 'array',
                                },
                                {'type': 'null'},
                            ],
                        },
                    },
                    'required': ['path'],
                    'type': 'object',
                },
                True: {
                    'additionalProperties': False,
                    'properties': {
                        'path': {'type': 'string'},
                        'replacements': {
                            'anyOf': [
                                {
                                    'items': {
                                        'properties': {'old_text': {'type': 'string'}, 'new_text': {'type': 'string'}},
                                        'required': ['old_text', 'new_text'],
                                        'type': 'object',
                                        'additionalProperties': False,
                                    },
                                    'type': 'array',
                                },
                                {'type': 'null'},
                            ]
                        },
                    },
                    'required': ['path', 'replacements'],
                    'type': 'object',
                },
                False: {
                    'additionalProperties': False,
                    'properties': {
                        'path': {'type': 'string'},
                        'replacements': {
                            'default': None,
                            'anyOf': [
                                {
                                    'items': {
                                        'properties': {'old_text': {'type': 'string'}, 'new_text': {'type': 'string'}},
                                        'required': ['old_text', 'new_text'],
                                        'type': 'object',
                                    },
                                    'type': 'array',
                                },
                                {'type': 'null'},
                            ],
                        },
                    },
                    'required': ['path'],
                    'type': 'object',
                },
            }
        ),
        native_schema=snapshot(
            {
                None: {
                    'properties': {'replacements': {'items': {'$ref': '#/$defs/Replacement'}, 'type': 'array'}},
                    'required': ['replacements'],
                    'type': 'object',
                    'additionalProperties': False,
                    '$defs': {
                        'Replacement': {
                            'properties': {'old_text': {'type': 'string'}, 'new_text': {'type': 'string'}},
                            'required': ['old_text', 'new_text'],
                            'type': 'object',
                            'additionalProperties': False,
                        }
                    },
                },
                True: {
                    'properties': {'replacements': {'items': {'$ref': '#/$defs/Replacement'}, 'type': 'array'}},
                    'required': ['replacements'],
                    'type': 'object',
                    'additionalProperties': False,
                    '$defs': {
                        'Replacement': {
                            'properties': {'old_text': {'type': 'string'}, 'new_text': {'type': 'string'}},
                            'required': ['old_text', 'new_text'],
                            'type': 'object',
                            'additionalProperties': False,
                        }
                    },
                },
                False: {
                    'properties': {'replacements': {'items': {'$ref': '#/$defs/Replacement'}, 'type': 'array'}},
                    'required': ['replacements'],
                    'type': 'object',
                    '$defs': {
                        'Replacement': {
                            'properties': {'old_text': {'type': 'string'}, 'new_text': {'type': 'string'}},
                            'required': ['old_text', 'new_text'],
                            'type': 'object',
                        }
                    },
                },
            }
        ),
        expected_output=_EDITS_OUTPUT,
        response_content=_EDITS_CONTENT,
    ),
    Case(
        'azure-custom-profile-native',
        snapshot(
            {
                None: {
                    'additionalProperties': False,
                    'properties': {
                        'path': {'type': 'string'},
                        'replacements': {
                            'default': None,
                            'anyOf': [{'items': {'$ref': '#/$defs/Replacement'}, 'type': 'array'}, {'type': 'null'}],
                        },
                    },
                    'required': ['path'],
                    'type': 'object',
                    '$defs': {
                        'Replacement': {
                            'properties': {'old_text': {'type': 'string'}, 'new_text': {'type': 'string'}},
                            'required': ['old_text', 'new_text'],
                            'type': 'object',
                            'additionalProperties': False,
                        }
                    },
                },
                True: {
                    'additionalProperties': False,
                    'properties': {
                        'path': {'type': 'string'},
                        'replacements': {
                            'anyOf': [{'items': {'$ref': '#/$defs/Replacement'}, 'type': 'array'}, {'type': 'null'}]
                        },
                    },
                    'required': ['path', 'replacements'],
                    'type': 'object',
                    '$defs': {
                        'Replacement': {
                            'properties': {'old_text': {'type': 'string'}, 'new_text': {'type': 'string'}},
                            'required': ['old_text', 'new_text'],
                            'type': 'object',
                            'additionalProperties': False,
                        }
                    },
                },
                False: {
                    'additionalProperties': False,
                    'properties': {
                        'path': {'type': 'string'},
                        'replacements': {
                            'default': None,
                            'anyOf': [{'items': {'$ref': '#/$defs/Replacement'}, 'type': 'array'}, {'type': 'null'}],
                        },
                    },
                    'required': ['path'],
                    'type': 'object',
                    '$defs': {
                        'Replacement': {
                            'properties': {'old_text': {'type': 'string'}, 'new_text': {'type': 'string'}},
                            'required': ['old_text', 'new_text'],
                            'type': 'object',
                        }
                    },
                },
            }
        ),
        provider_name='azure-custom',
        native_schema=snapshot(
            {
                None: {
                    'properties': {'replacements': {'items': {'$ref': '#/$defs/Replacement'}, 'type': 'array'}},
                    'required': ['replacements'],
                    'type': 'object',
                    'additionalProperties': False,
                    '$defs': {
                        'Replacement': {
                            'properties': {'old_text': {'type': 'string'}, 'new_text': {'type': 'string'}},
                            'required': ['old_text', 'new_text'],
                            'type': 'object',
                            'additionalProperties': False,
                        }
                    },
                },
                True: {
                    'properties': {'replacements': {'items': {'$ref': '#/$defs/Replacement'}, 'type': 'array'}},
                    'required': ['replacements'],
                    'type': 'object',
                    'additionalProperties': False,
                    '$defs': {
                        'Replacement': {
                            'properties': {'old_text': {'type': 'string'}, 'new_text': {'type': 'string'}},
                            'required': ['old_text', 'new_text'],
                            'type': 'object',
                            'additionalProperties': False,
                        }
                    },
                },
                False: {
                    'properties': {'replacements': {'items': {'$ref': '#/$defs/Replacement'}, 'type': 'array'}},
                    'required': ['replacements'],
                    'type': 'object',
                    '$defs': {
                        'Replacement': {
                            'properties': {'old_text': {'type': 'string'}, 'new_text': {'type': 'string'}},
                            'required': ['old_text', 'new_text'],
                            'type': 'object',
                        }
                    },
                },
            }
        ),
        expected_output=_EDITS_OUTPUT,
        response_content=_EDITS_CONTENT,
    ),
    Case(
        'azure-grok-property-names',
        snapshot(
            {
                None: {
                    'additionalProperties': False,
                    'properties': {
                        'values': {
                            'additionalProperties': {'type': 'integer'},
                            'propertyNames': {'type': 'string'},
                            'type': 'object',
                            'properties': {},
                        }
                    },
                    'required': ['values'],
                    'type': 'object',
                },
                True: {
                    'additionalProperties': False,
                    'properties': {
                        'values': {
                            'additionalProperties': False,
                            'type': 'object',
                            'description': "propertyNames={'type': 'string'}",
                            'properties': {},
                            'required': [],
                        }
                    },
                    'required': ['values'],
                    'type': 'object',
                },
                False: {
                    'additionalProperties': False,
                    'properties': {
                        'values': {
                            'additionalProperties': {'type': 'integer'},
                            'propertyNames': {'type': 'string'},
                            'type': 'object',
                            'properties': {},
                        }
                    },
                    'required': ['values'],
                    'type': 'object',
                },
            }
        ),
        function=accept_keys,
    ),
    Case(
        'azure-grok-contains',
        snapshot(
            {
                None: {
                    'additionalProperties': False,
                    'properties': {
                        'values': {
                            'contains': {'exclusiveMinimum': 0, 'type': 'integer'},
                            'items': {'anyOf': [{'exclusiveMinimum': 0, 'type': 'integer'}, {'type': 'string'}]},
                            'type': 'array',
                        }
                    },
                    'required': ['values'],
                    'type': 'object',
                },
                True: {
                    'additionalProperties': False,
                    'properties': {
                        'values': {
                            'items': {'anyOf': [{'exclusiveMinimum': 0, 'type': 'integer'}, {'type': 'string'}]},
                            'type': 'array',
                            'description': "contains={'exclusiveMinimum': 0, 'type': 'integer'}",
                        }
                    },
                    'required': ['values'],
                    'type': 'object',
                },
                False: {
                    'additionalProperties': False,
                    'properties': {
                        'values': {
                            'contains': {'exclusiveMinimum': 0, 'type': 'integer'},
                            'items': {'anyOf': [{'exclusiveMinimum': 0, 'type': 'integer'}, {'type': 'string'}]},
                            'type': 'array',
                        }
                    },
                    'required': ['values'],
                    'type': 'object',
                },
            }
        ),
        function=accept_contains,
    ),
    Case(
        'azure-grok-recursive',
        snapshot(
            {
                None: {
                    '$defs': {
                        'Node': {
                            'properties': {
                                'value': {'type': 'string'},
                                'children': {
                                    'default': None,
                                    'anyOf': [{'items': {'$ref': '#'}, 'type': 'array'}, {'type': 'null'}],
                                },
                            },
                            'required': ['value'],
                            'type': 'object',
                            'additionalProperties': False,
                        }
                    },
                    'properties': {
                        'value': {'type': 'string'},
                        'children': {
                            'default': None,
                            'anyOf': [{'items': {'$ref': '#'}, 'type': 'array'}, {'type': 'null'}],
                        },
                    },
                    'required': ['value'],
                    'type': 'object',
                    'additionalProperties': False,
                },
                True: {
                    '$defs': {
                        'Node': {
                            'properties': {
                                'value': {'type': 'string'},
                                'children': {'anyOf': [{'items': {'$ref': '#'}, 'type': 'array'}, {'type': 'null'}]},
                            },
                            'required': ['value', 'children'],
                            'type': 'object',
                            'additionalProperties': False,
                        }
                    },
                    'properties': {
                        'value': {'type': 'string'},
                        'children': {'anyOf': [{'items': {'$ref': '#'}, 'type': 'array'}, {'type': 'null'}]},
                    },
                    'required': ['value', 'children'],
                    'type': 'object',
                    'additionalProperties': False,
                },
                False: {
                    '$defs': {
                        'Node': {
                            'properties': {
                                'value': {'type': 'string'},
                                'children': {
                                    'default': None,
                                    'anyOf': [{'items': {'$ref': '#'}, 'type': 'array'}, {'type': 'null'}],
                                },
                            },
                            'required': ['value'],
                            'type': 'object',
                        }
                    },
                    'properties': {
                        'value': {'type': 'string'},
                        'children': {
                            'default': None,
                            'anyOf': [{'items': {'$ref': '#'}, 'type': 'array'}, {'type': 'null'}],
                        },
                    },
                    'required': ['value'],
                    'type': 'object',
                },
            }
        ),
        function=accept_node,
        check_inferred_tool_strict=True,
    ),
    Case(
        'generic-openai-azure-profile',
        snapshot(
            {
                None: {
                    'additionalProperties': False,
                    'properties': {
                        'path': {'type': 'string'},
                        'replacements': {
                            'default': None,
                            'anyOf': [
                                {
                                    'items': {
                                        'properties': {'old_text': {'type': 'string'}, 'new_text': {'type': 'string'}},
                                        'required': ['old_text', 'new_text'],
                                        'type': 'object',
                                        'additionalProperties': False,
                                    },
                                    'type': 'array',
                                },
                                {'type': 'null'},
                            ],
                        },
                    },
                    'required': ['path'],
                    'type': 'object',
                },
            }
        ),
        provider_name='openai-azure',
        strict_modes=(None,),
    ),
]


@pytest.mark.parametrize('api', ['chat', 'responses'])
@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in CASES])
async def test_azure_grok_tool_schema(
    allow_model_requests: None,
    request_capture: RequestCapture,
    api: Literal['chat', 'responses'],
    case: Case,
):
    """Capture the SDK's actual request; Azure acceptance is reporter-verified, not simulated here."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        if api == 'chat':
            return httpx2.Response(
                200,
                json={
                    'id': 'chatcmpl-test',
                    'object': 'chat.completion',
                    'created': 0,
                    'model': case.model_name,
                    'choices': [
                        {
                            'index': 0,
                            'message': {'role': 'assistant', 'content': case.response_content},
                            'finish_reason': 'stop',
                        }
                    ],
                },
            )
        return httpx2.Response(
            200,
            json={
                'id': 'resp-test',
                'object': 'response',
                'created_at': 0,
                'model': case.model_name,
                'status': 'completed',
                'output': [
                    {
                        'id': 'msg-test',
                        'type': 'message',
                        'role': 'assistant',
                        'status': 'completed',
                        'content': [{'type': 'output_text', 'text': case.response_content, 'annotations': []}],
                    }
                ],
            },
        )

    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler),
        event_hooks={'request': list(request_capture.client.event_hooks['request'])},
    ) as client:
        endpoint = 'https://example.services.ai.azure.com/openai/v1'
        provider = (
            AzureProvider(azure_endpoint=endpoint, api_key='test', http_client=client)
            if case.provider_name in ('azure', 'azure-custom')
            else OpenAIProvider(base_url=endpoint, api_key='test', http_client=client)
        )
        profile = (
            OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer)
            if case.provider_name == 'azure-custom'
            else AzureProvider.model_profile(case.model_name)
            if case.provider_name == 'openai-azure'
            else None
        )
        model = (OpenAIChatModel if api == 'chat' else OpenAIResponsesModel)(
            case.model_name, provider=provider, profile=profile
        )
        # All three strict modes exercise different conversion and wire-flag behavior.
        # Both APIs share schema expectations but serialize strict flags differently.
        tool_schemas: dict[bool | None, JsonValue] = {}
        native_schemas: dict[bool | None, JsonValue] = {}
        for strict in case.strict_modes:
            agent = (
                Agent(model, output_type=NativeOutput(Edits, strict=strict))
                if case.native_schema is not None
                else Agent(model)
            )

            agent.tool_plain(strict=strict)(case.function)

            assert (await agent.run('Say hi.')).output == case.expected_output

            body = request_capture.bodies('/chat/completions' if api == 'chat' else '/responses')[-1]
            tools = body['tools']
            assert isinstance(tools, list)
            tool = tools[0]
            assert isinstance(tool, dict)
            if api == 'chat':
                tool = tool['function']
                assert isinstance(tool, dict)
            tool_schemas[strict] = tool['parameters']
            if strict is None and case.check_inferred_tool_strict:
                assert ('strict' in tool, tool.get('strict')) == _INFERRED_TOOL_STRICT_FIELDS[api]

            if case.native_schema is not None:
                output_format = body['response_format'] if api == 'chat' else body['text']
                assert isinstance(output_format, dict)
                output_format = output_format['json_schema'] if api == 'chat' else output_format['format']
                assert isinstance(output_format, dict)
                native_schemas[strict] = output_format['schema']
                assert output_format['strict'] is _NATIVE_STRICT_FLAGS[strict]

    assert tool_schemas == case.tool_schemas
    if case.native_schema is not None:
        assert native_schemas == case.native_schema
